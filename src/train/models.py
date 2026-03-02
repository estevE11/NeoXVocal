from typing import List, Optional

import torch
import torch.nn as nn
from transformers import DebertaV2Model


class CrossAttentionFusion(nn.Module):
    """Cross-attention between audio and text modalities.
    
    Modes:
        'audio_to_text': audio tokens query into text (audio seeks linguistic context)
        'text_to_audio': text tokens query into audio (text enriched with acoustics)
        'gated_bidirectional': both directions with a learned gate
    """
    def __init__(self, hidden_size, num_heads=8, num_layers=1, dropout=0.1,
                 mode='audio_to_text', gate_init=0.5):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers

        # Audio -> Text direction (audio queries, text keys/values)
        if mode in ('audio_to_text', 'gated_bidirectional'):
            self.a2t_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.a2t_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        # Text -> Audio direction (text queries, audio keys/values)
        if mode in ('text_to_audio', 'gated_bidirectional'):
            self.t2a_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.t2a_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        # Learned gate for bidirectional mode
        if mode == 'gated_bidirectional':
            # gate_raw is passed through sigmoid -> gate in [0,1]
            # output = gate * a2t_audio + (1-gate) * t2a_audio  for audio tokens
            # output = gate * a2t_text  + (1-gate) * t2a_text   for text tokens
            init_val = torch.log(torch.tensor(gate_init / (1.0 - gate_init)))  # inverse sigmoid
            self.gate_raw = nn.Parameter(torch.full((1,), init_val.item()))

    def forward(self, audio_tokens, text_tokens, text_key_padding_mask=None):
        """
        Args:
            audio_tokens: (B, Na, D) where Na=2 (audio_feat + wav2vec_emb)
            text_tokens:  (B, Nt, D) where Nt=512 (DeBERTa output)
            text_key_padding_mask: (B, Nt) True=ignore (padding). Used when text is K/V.
        Returns:
            audio_out: (B, Na, D)
            text_out:  (B, Nt, D)
        """
        if self.mode == 'audio_to_text':
            # Audio queries attend to text keys/values
            a = audio_tokens
            for attn, norm in zip(self.a2t_layers, self.a2t_norms):
                a_res, _ = attn(query=a, key=text_tokens, value=text_tokens,
                                key_padding_mask=text_key_padding_mask)
                a = norm(a + a_res)
            return a, text_tokens  # text unchanged

        elif self.mode == 'text_to_audio':
            # Text queries attend to audio keys/values (no mask needed, audio has no padding)
            t = text_tokens
            for attn, norm in zip(self.t2a_layers, self.t2a_norms):
                t_res, _ = attn(query=t, key=audio_tokens, value=audio_tokens)
                t = norm(t + t_res)
            return audio_tokens, t  # audio unchanged

        else:  # gated_bidirectional
            gate = torch.sigmoid(self.gate_raw)  # scalar in [0,1]

            a_cross = audio_tokens
            for attn, norm in zip(self.a2t_layers, self.a2t_norms):
                a_res, _ = attn(query=a_cross, key=text_tokens, value=text_tokens,
                                key_padding_mask=text_key_padding_mask)
                a_cross = norm(a_cross + a_res)

            t_cross = text_tokens
            for attn, norm in zip(self.t2a_layers, self.t2a_norms):
                t_res, _ = attn(query=t_cross, key=audio_tokens, value=audio_tokens)
                t_cross = norm(t_cross + t_res)

            audio_out = gate * a_cross + (1 - gate) * audio_tokens
            text_out = gate * text_tokens + (1 - gate) * t_cross
            return audio_out, text_out


def _build_classifier(
    input_size: int,
    hidden_layers: int,
    hidden_width: int,
    dropout: float,
    num_classes: int,
    activation: str = 'relu',
) -> nn.Sequential:
    """Build a configurable classifier head.
    
    Args:
        input_size: Input dimension from the transformer/pooling layer
        hidden_layers: Number of hidden layers (0 = direct projection to output)
        hidden_width: Width of each hidden layer
        dropout: Dropout probability between layers
        num_classes: Number of output classes (1 for binary with BCEWithLogitsLoss)
        activation: Activation function ('relu', 'gelu', 'tanh', 'leaky_relu')
    """
    activation_fn = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
    }.get(activation.lower(), nn.ReLU)
    
    layers: List[nn.Module] = []
    current_size = input_size
    
    for i in range(hidden_layers):
        layers.append(nn.Linear(current_size, hidden_width))
        layers.append(activation_fn())
        layers.append(nn.Dropout(dropout))
        current_size = hidden_width
    
    # Final projection to num_classes
    layers.append(nn.Linear(current_size, num_classes))
    
    return nn.Sequential(*layers)


class NeuroXVocal(nn.Module):
    def __init__(
        self,
        num_audio_features: int,
        num_embedding_features: int,
        text_embedding_model: str,
        # Transformer encoder configuration
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: Optional[int] = None,
        transformer_dropout: float = 0.35,
        transformer_activation: str = 'gelu',
        # Feature projection configuration
        feature_projection_dropout: float = 0.3,
        # Classifier head configuration
        classifier_hidden_layers: int = 1,
        classifier_hidden_width: Optional[int] = None,
        classifier_dropout: float = 0.45,
        classifier_activation: str = 'relu',
        num_classes: int = 1,
        # Text model configuration
        freeze_text_model: bool = False,
        freeze_text_model_layers: Optional[int] = None,
        # Pooling strategy
        pooling_strategy: str = 'first',
        # Cross-attention configuration (None = disabled, backward compatible)
        cross_attention_mode: Optional[str] = None,
        cross_attention_num_heads: int = 8,
        cross_attention_num_layers: int = 1,
        cross_attention_dropout: float = 0.1,
        cross_attention_placement: str = 'before',
        cross_attention_gate_init: float = 0.5,
    ):
        """
        NeuroXVocal: Multimodal model combining text, audio features, and audio embeddings.
        
        Args:
            num_audio_features: Number of audio features (e.g., MFCC features)
            num_embedding_features: Dimension of audio embeddings (e.g., wav2vec)
            text_embedding_model: HuggingFace model name for text encoding
            
            transformer_num_heads: Number of attention heads in transformer encoder
            transformer_num_layers: Number of transformer encoder layers
            transformer_dim_feedforward: Feedforward dimension (default: 4 * hidden_size)
            transformer_dropout: Dropout in transformer encoder
            transformer_activation: Activation function in transformer ('relu', 'gelu')
            
            feature_projection_dropout: Dropout in audio/embedding projection layers
            
            classifier_hidden_layers: Number of hidden layers in classifier (0 = linear)
            classifier_hidden_width: Width of classifier hidden layers (default: hidden_size // 2)
            classifier_dropout: Dropout in classifier
            classifier_activation: Activation in classifier ('relu', 'gelu', 'tanh', 'leaky_relu')
            num_classes: Number of output classes (1 for binary classification)
            
            freeze_text_model: Whether to freeze the entire text model
            freeze_text_model_layers: Number of text model layers to freeze from bottom (None = don't freeze by layer)
            
            pooling_strategy: How to pool transformer output ('first', 'mean', 'max', 'cls_token')
            
            cross_attention_mode: None (disabled) | 'audio_to_text' | 'text_to_audio' | 'gated_bidirectional'
            cross_attention_num_heads: Heads for cross-attention
            cross_attention_num_layers: Stacked cross-attention layers
            cross_attention_dropout: Dropout in cross-attention
            cross_attention_placement: 'before' (cross-attn then self-attn fusion)
                                     | 'replace' (cross-attn only, no self-attn fusion)
                                     | 'hybrid' (same as before, kept for clarity)
            cross_attention_gate_init: Initial gate value for gated_bidirectional (0.5=balanced)
        """
        super(NeuroXVocal, self).__init__()
        
        # Store configuration
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        self.cross_attention_mode = cross_attention_mode
        self.cross_attention_placement = cross_attention_placement

        # Text model
        self.text_model = DebertaV2Model.from_pretrained(text_embedding_model)
        self.hidden_size = self.text_model.config.hidden_size
        
        # Handle text model freezing
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        elif freeze_text_model_layers is not None and freeze_text_model_layers > 0:
            # Freeze embeddings
            for param in self.text_model.embeddings.parameters():
                param.requires_grad = False
            # Freeze specified number of encoder layers
            for i, layer in enumerate(self.text_model.encoder.layer):
                if i < freeze_text_model_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Audio features projection
        self.audio_fc = nn.Sequential(
            nn.Linear(num_audio_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(feature_projection_dropout),
            nn.ReLU()
        )

        # Audio embeddings projection
        self.embedding_fc = nn.Sequential(
            nn.Linear(num_embedding_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(feature_projection_dropout),
            nn.ReLU()
        )

        # Cross-attention fusion (optional)
        self.cross_attention = None
        if cross_attention_mode is not None:
            self.cross_attention = CrossAttentionFusion(
                hidden_size=self.hidden_size,
                num_heads=cross_attention_num_heads,
                num_layers=cross_attention_num_layers,
                dropout=cross_attention_dropout,
                mode=cross_attention_mode,
                gate_init=cross_attention_gate_init,
            )

        # Transformer encoder (self-attention fusion over concatenated sequence)
        # Skipped when placement='replace' and cross-attention is enabled
        dim_feedforward = transformer_dim_feedforward or (4 * self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=transformer_num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        # Classifier head
        classifier_width = classifier_hidden_width or (self.hidden_size // 2)
        self.classifier = _build_classifier(
            input_size=self.hidden_size,
            hidden_layers=classifier_hidden_layers,
            hidden_width=classifier_width,
            dropout=classifier_dropout,
            num_classes=num_classes,
            activation=classifier_activation,
        )

    def _pool_output(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """Pool transformer output based on configured strategy.
        
        Args:
            transformer_output: Shape (seq_len, batch, hidden_size)
            
        Returns:
            Pooled output of shape (batch, hidden_size)
        """
        if self.pooling_strategy == 'first':
            # Use first token (audio features position)
            return transformer_output[0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling over sequence
            return transformer_output.mean(dim=0)
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence
            return transformer_output.max(dim=0)[0]
        elif self.pooling_strategy == 'cls_token':
            # Same as first for this architecture
            return transformer_output[0]
        else:
            # Default to first token
            return transformer_output[0]

    def forward(self, text_input, audio_input, embedding_input):
        input_ids = text_input['input_ids'].squeeze(1)
        attention_mask = text_input['attention_mask'].squeeze(1)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state  # (B, Nt, D)
        audio_embeddings = self.audio_fc(audio_input)             # (B, D)
        audio_embeddings = audio_embeddings.unsqueeze(1)          # (B, 1, D)
        embedding_embeddings = self.embedding_fc(embedding_input) # (B, D)
        embedding_embeddings = embedding_embeddings.unsqueeze(1)  # (B, 1, D)

        # --- Cross-attention path ---
        if self.cross_attention is not None:
            audio_tokens = torch.cat((audio_embeddings, embedding_embeddings), dim=1)  # (B, 2, D)
            # Invert attention_mask: DeBERTa uses 1=real, MHA key_padding_mask uses True=ignore
            text_pad_mask = (attention_mask == 0)  # (B, Nt)

            audio_out, text_out = self.cross_attention(
                audio_tokens, text_embeddings, text_key_padding_mask=text_pad_mask)

            # Use whichever side was updated (or both for gated)
            fused_audio = audio_out if audio_out is not None else audio_tokens
            fused_text = text_out if text_out is not None else text_embeddings

            if self.cross_attention_placement == 'replace':
                # Pool directly from cross-attended audio tokens, skip self-attention fusion
                pooled_output = fused_audio.mean(dim=1)  # (B, D) mean of 2 audio tokens
            else:
                # 'before' or 'hybrid': feed cross-attended representations into self-attention
                combined = torch.cat((fused_audio, fused_text), dim=1)  # (B, Na+Nt, D)
                combined = combined.permute(1, 0, 2)                    # (Na+Nt, B, D)
                transformer_output = self.transformer_encoder(combined)
                pooled_output = self._pool_output(transformer_output)
        else:
            # --- Original path (no cross-attention, fully backward compatible) ---
            combined_embeddings = torch.cat(
                (audio_embeddings, embedding_embeddings, text_embeddings), dim=1)
            combined_embeddings = combined_embeddings.permute(1, 0, 2)
            transformer_output = self.transformer_encoder(combined_embeddings)
            pooled_output = self._pool_output(transformer_output)

        logits = self.classifier(pooled_output)

        # Squeeze for binary classification
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        return logits

    def reset_parameters(self):
        def reset_layer_parameters(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.audio_fc.apply(reset_layer_parameters)
        self.embedding_fc.apply(reset_layer_parameters)
        self.transformer_encoder.apply(reset_layer_parameters)
        self.classifier.apply(reset_layer_parameters)
        if self.cross_attention is not None:
            self.cross_attention.apply(reset_layer_parameters)
