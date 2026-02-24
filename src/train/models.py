from typing import List, Optional

import torch
import torch.nn as nn
from transformers import DebertaV2Model


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
        """
        super(NeuroXVocal, self).__init__()
        
        # Store configuration
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes

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

        # Transformer encoder
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
        text_embeddings = text_outputs.last_hidden_state
        audio_embeddings = self.audio_fc(audio_input)  
        audio_embeddings = audio_embeddings.unsqueeze(1) 
        embedding_embeddings = self.embedding_fc(embedding_input)  
        embedding_embeddings = embedding_embeddings.unsqueeze(1)
        combined_embeddings = torch.cat(
            (audio_embeddings, embedding_embeddings, text_embeddings),
            dim=1
        )  
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
