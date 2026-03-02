# Cross-Attention Study for NeuroXVocal

## 1. Executive Summary

**Goal**: Integrate explicit cross-attention into NeuroXVocal for multimodal fusion between audio features and text representations, replacing or augmenting the current implicit cross-modal interaction via self-attention over concatenated tokens.

**Three experimental configurations**:
1. **Audio-to-Text** (`audio_to_text`): Audio tokens query into DeBERTa text output
2. **Text-to-Audio** (`text_to_audio`): Text tokens query into projected audio tokens
3. **Gated Bidirectional** (`gated_bidirectional`): Both directions with a learned scalar gate

**Three placement options** for each:
- `before`: Cross-attention runs first, then existing self-attention TransformerEncoder
- `replace`: Cross-attention replaces the TransformerEncoder entirely
- `hybrid`: Same as `before` (alias for clarity)

**Current baseline**: Self-attention only over concatenated `[audio_feat, wav2vec_emb, text_0..text_511]` sequence (514 tokens). No explicit cross-modal attention mechanism.

---

## 2. Current Architecture Analysis

### 2.1 Three Modalities and Tensor Shapes

| Modality | Source | Raw Shape | Projected Shape |
|----------|--------|-----------|-----------------|
| Audio features | 47 handcrafted features (pitch, pauses, MFCCs, etc.) | `(B, 47)` | `(B, 1, 768)` |
| Audio embeddings | Wav2Vec2 mean-pooled `last_hidden_state` | `(B, 768)` | `(B, 1, 768)` |
| Text | DeBERTa-v3-base `last_hidden_state` | `(B, 512, 768)` | `(B, 512, 768)` |

### 2.2 Current Forward Pass

```
text_input ──► DeBERTa ──► (B, 512, 768) ──────────────────────────────────┐
audio_input ─► Linear(47,768)+LN+Drop+ReLU ──► (B,768) ► unsqueeze ► (B,1,768) ──┤
emb_input ──► Linear(768,768)+LN+Drop+ReLU ─► (B,768) ► unsqueeze ► (B,1,768) ──┤
                                                                            │
                                           cat(dim=1) ◄─────────────────────┘
                                               │
                                        (B, 514, 768)
                                               │
                                        permute(1,0,2)
                                               │
                                        (514, B, 768)
                                               │
                                    TransformerEncoder (2 layers, 8 heads)
                                               │
                                        (514, B, 768)
                                               │
                                    pool 'first' ──► output[0] = audio_feat token
                                               │
                                          (B, 768)
                                               │
                                    Linear(768,384)►ReLU►Drop►Linear(384,1)
                                               │
                                          (B,) logits
```

### 2.3 How Self-Attention Currently Handles Cross-Modal Interaction

The `nn.TransformerEncoder` (2 layers, 8 heads, `d_model=768`) applies **full bidirectional self-attention** over the 514-token concatenated sequence. Every token attends to every other token, so:

- Audio tokens attend to all 512 text tokens (and each other)
- Text tokens attend to both audio tokens (and each other)

This is **implicit** cross-modal interaction: cross-modal information flows only because the tokens share the same sequence. There is no mechanism specifically designed to align or fuse across modalities.

### 2.4 Identified Limitations

1. **No explicit cross-modal attention mechanism**: Cross-modal interaction competes with same-modal self-attention for capacity in the same heads.
2. **Asymmetric representation**: 2 audio tokens vs 512 text tokens. Audio gets drowned out in self-attention (softmax denominator = 514).
3. **No positional encodings in the fusion transformer**: The model cannot distinguish token positions within the concatenated sequence.
4. **No modality-type embeddings**: The model has no explicit signal indicating which tokens are audio vs text.
5. **First-token pooling bias**: Pooling from `output[0]` (the audio features token position) biases the representation toward whatever information concentrates at that position.
6. **Text attention mask not propagated**: DeBERTa's `attention_mask` is used internally but **not passed to the fusion TransformerEncoder**. Padding tokens (from `max_length=512` tokenization) participate freely in fusion self-attention.

---

## 3. Cross-Attention: Theory

### 3.1 Q/K/V Decomposition Across Modalities

In standard self-attention, Q, K, V all come from the same sequence. In **cross-attention**, they come from different modalities:

```
Q = W_q @ modality_A     # The "querying" modality
K = W_k @ modality_B     # The "attended-to" modality
V = W_v @ modality_B     # Values come from same source as keys

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

The output has the **same sequence length as the query modality** but is enriched with information from the key/value modality.

### 3.2 How Cross-Attention Differs from Self-Attention for Multimodal Fusion

| Property | Self-Attention (current) | Cross-Attention |
|----------|------------------------|-----------------|
| Q, K, V source | Same concatenated sequence | Q from one modality, K/V from another |
| Cross-modal interaction | Implicit (shared sequence) | Explicit (dedicated mechanism) |
| Attention budget | Shared across all tokens | Dedicated to cross-modal alignment |
| Output shape | Same as input | Same as query modality |
| Asymmetry handling | Poor (2 vs 512 tokens compete) | Natural (each direction is separate) |

### 3.3 Relevant Architectures

- **Perceiver** (Jaegle et al., 2021): Uses cross-attention from a small set of latent queries into a large input. Analogous to our audio-to-text direction (2 audio tokens querying 512 text tokens).
- **Flamingo** (Alayrac et al., 2022): Interleaves cross-attention layers between a frozen LM and visual features. Similar spirit to our approach of cross-attending from text to audio.
- **Bottleneck Transformer** (Nagrani et al., 2021): Uses a small set of bottleneck tokens that cross-attend to multiple modalities. Our audio tokens act as natural bottleneck tokens.

---

## 4. Proposed Experiments

### Experiment 1: Audio-to-Text (`audio_to_text`)

**Q=audio, K=V=text**: The 2 audio tokens query into the 512 DeBERTa text tokens.

```
audio_tokens (B,2,768)  ────► Q
text_tokens  (B,512,768) ──► K, V

CrossAttn output: (B, 2, 768)  ← audio enriched with linguistic context
```

**Intuition**: Audio representations seek the most relevant linguistic context. The 2 audio tokens act like Perceiver-style latent queries, compressing 512 text tokens into 2 cross-attended representations.

**Tensor shape walkthrough** (`placement='before'`):
```
audio_proj:  (B, 1, 768)     ┐
emb_proj:    (B, 1, 768)     ├► cat ► audio_tokens: (B, 2, 768)
                              ┘
text_emb:    (B, 512, 768)   ──► text_tokens

CrossAttention:
  Q = audio_tokens (B, 2, 768)
  K = text_tokens  (B, 512, 768)
  V = text_tokens  (B, 512, 768)
  Output: fused_audio (B, 2, 768), text unchanged (B, 512, 768)

Then: cat(fused_audio, text) ► (B, 514, 768) ► permute ► TransformerEncoder ► pool ► classifier
```

**Pros**:
- Very lightweight: only 2 query tokens, so cross-attention is O(2 * 512) per head
- Natural bottleneck: compresses text information into audio-sized representation
- Audio tokens are no longer drowned by 512 text tokens in self-attention
- Minimal parameter overhead

**Cons**:
- Unidirectional: text tokens never explicitly attend to audio
- With only 2 query positions, limited capacity to capture diverse text aspects
- Text representations pass through unchanged (they still benefit from self-attention later in `before` placement)

---

### Experiment 2: Text-to-Audio (`text_to_audio`)

**Q=text, K=V=audio**: The 512 text tokens query into the 2 audio tokens.

```
text_tokens  (B,512,768) ──► Q
audio_tokens (B,2,768)   ──► K, V

CrossAttn output: (B, 512, 768)  ← text enriched with acoustic info
```

**Intuition**: Each text token learns to attend to the 2 audio representations, enriching linguistic features with acoustic information (prosody, pauses, speech quality).

**Tensor shape walkthrough** (`placement='before'`):
```
CrossAttention:
  Q = text_tokens  (B, 512, 768)
  K = audio_tokens (B, 2, 768)
  V = audio_tokens (B, 2, 768)
  Output: audio unchanged (B, 2, 768), fused_text (B, 512, 768)

Then: cat(audio, fused_text) ► (B, 514, 768) ► permute ► TransformerEncoder ► pool ► classifier
```

**Pros**:
- Every text token gets acoustic context
- Attention over only 2 keys is very cheap: O(512 * 2) per head
- Text tokens can selectively weight audio features vs wav2vec embedding

**Cons**:
- Audio tokens never explicitly attend to text
- 512 text queries attending to only 2 audio keys may not be expressive enough
- Risk: each text token may learn a near-identical weighting of the 2 audio tokens

---

### Experiment 3: Gated Bidirectional (`gated_bidirectional`)

**Both directions simultaneously**, blended with a learned gate.

```
Forward (audio→text):
  Q = audio_tokens, K = V = text_tokens → fused_audio (B, 2, 768)

Reverse (text→audio):
  Q = text_tokens, K = V = audio_tokens → fused_text (B, 512, 768)

Gate: sigmoid(gate_raw) ∈ [0,1]  (scalar, initialized at 0.5)

audio_out = gate * fused_audio + (1-gate) * original_audio
text_out  = gate * original_text + (1-gate) * fused_text
```

**Intuition**: Let the model discover which direction of cross-attention is more informative. The gate starts balanced (0.5) and the model learns to shift weight during training. If audio→text is more useful, gate → 1. If text→audio is more useful, gate → 0.

**Gate initialization**: `gate_init=0.5` means `gate_raw = log(0.5/0.5) = 0.0`. After sigmoid: `sigmoid(0.0) = 0.5`. This provides a balanced starting point.

**Pros**:
- Captures both directions of cross-modal interaction
- Learned gating avoids manual selection of direction
- The gate value itself is interpretable (logs which direction the model prefers)
- Subsumes experiments 1 and 2 as special cases (gate→1.0 or gate→0.0)

**Cons**:
- ~2x parameters compared to single-direction
- Gate may get stuck if learning rate is too low
- Slightly more complex to interpret attention patterns
- The two directions are computed independently (no iterative refinement between them)

---

## 5. Integration Points in the Codebase

### 5.1 Implementation Summary

The implementation adds **one new class** (`CrossAttentionFusion`, ~80 lines) and modifies the existing `NeuroXVocal` class with:
- 6 new constructor parameters (all with defaults preserving backward compatibility)
- ~30 lines of new logic in `forward()`
- 2 lines in `reset_parameters()`

**Files changed**:
| File | What changed |
|------|-------------|
| `src/train/models.py` | Added `CrossAttentionFusion` class (lines 8-89). Added cross-attention params to `NeuroXVocal.__init__` (lines 157-163). Modified `forward()` (lines 320-348). Updated `reset_parameters()` (lines 366-367). |
| `src/train/config.py` | Added 6 new defaults (lines 72-78): `CROSS_ATTENTION_MODE`, `_NUM_HEADS`, `_NUM_LAYERS`, `_DROPOUT`, `_PLACEMENT`, `_GATE_INIT` |
| `src/train/main.py` | Added `--cross_attention_*` CLI args (lines 104-119). Wired into `model_config` dict (lines 222-228). Automatically logged to W&B via `**model_config`. |

### 5.2 Placement Options

#### Option A: `placement='before'`

Cross-attention runs BEFORE the existing TransformerEncoder. The cross-attended representations are then fed into self-attention fusion.

```
DeBERTa → text_tokens ──────────────┐
audio_fc → audio_proj ──┐           │
emb_fc → emb_proj ─────┤           │
                        ├► audio_tokens
                        │           │
                 CrossAttentionFusion(audio_tokens, text_tokens)
                        │           │
                  fused_audio    fused_text
                        │           │
                        └──── cat ──┘
                              │
                      TransformerEncoder (existing)
                              │
                         pool + classify
```

Lines affected in `forward()`: 336-341 (new code, original TransformerEncoder path preserved)

#### Option B: `placement='replace'`

Cross-attention **replaces** the TransformerEncoder. No self-attention fusion at all.

```
CrossAttentionFusion(audio_tokens, text_tokens)
        │
  fused_audio (B, 2, 768)
        │
  mean pool over 2 audio tokens → (B, 768)
        │
  classifier → logits
```

Lines affected in `forward()`: 333-335. The TransformerEncoder is still instantiated (for state_dict compatibility) but not called.

#### Option C: `placement='hybrid'`

Same behavior as `before`. Included as an explicit choice for experiment logging clarity.

### 5.3 New Config Parameters

```python
# config.py (lines 72-78)
CROSS_ATTENTION_MODE = None         # None | 'audio_to_text' | 'text_to_audio' | 'gated_bidirectional'
CROSS_ATTENTION_NUM_HEADS = 8
CROSS_ATTENTION_NUM_LAYERS = 1
CROSS_ATTENTION_DROPOUT = 0.1
CROSS_ATTENTION_PLACEMENT = 'before'  # 'before' | 'replace' | 'hybrid'
CROSS_ATTENTION_GATE_INIT = 0.5
```

### 5.4 CLI Arguments

```bash
--cross_attention_mode {audio_to_text,text_to_audio,gated_bidirectional}
--cross_attention_num_heads 8
--cross_attention_num_layers 1
--cross_attention_dropout 0.1
--cross_attention_placement {before,replace,hybrid}
--cross_attention_gate_init 0.5
```

### 5.5 Impact on `reset_parameters()`

The `cross_attention` module is now included in reset (line 366-367):
```python
if self.cross_attention is not None:
    self.cross_attention.apply(reset_layer_parameters)
```

This means cross-attention weights are properly reset between folds in k-fold CV. The `initial_state` clone in `train.py` also captures cross-attention parameters automatically (since they're part of `model.state_dict()`).

### 5.6 Impact on Training

- **New parameters**: Cross-attention adds `nn.MultiheadAttention` layers (each: 4 * 768^2 = 2.36M params for Q/K/V/out projections) plus `LayerNorm` (1536 params per layer).
- **For 1-layer audio_to_text**: ~2.36M + 1536 = ~2.36M new params
- **For 1-layer gated_bidirectional**: ~4.72M + 3072 + 1 (gate) = ~4.72M new params
- **Learning rate**: The cross-attention layers use the same optimizer and LR as the rest of the model. Consider using the same or slightly lower LR.
- **W&B logging**: All cross-attention params are automatically logged via `**model_config` in `wandb_config`.

### 5.7 Key Design Decision: Attention Mask Propagation

The implementation now **properly propagates the DeBERTa attention mask** to cross-attention:

```python
text_pad_mask = (attention_mask == 0)  # Invert: DeBERTa uses 1=real, MHA uses True=ignore
```

This means padding tokens are excluded from cross-attention keys/values when text is the K/V source (`audio_to_text` and `gated_bidirectional`). This is a fix compared to the original architecture where padding tokens participated in fusion self-attention.

---

## 6. Additional Architectural Improvements (Not Yet Implemented)

These are suggestions for future iterations, not part of the current implementation:

### 6.1 Positional Encodings for the Fusion Sequence

The current TransformerEncoder has no positional encodings. Adding sinusoidal or learned positional embeddings to the 514-token sequence would let the model distinguish position:
```python
self.fusion_pos_embed = nn.Parameter(torch.randn(514, 1, 768) * 0.02)
combined = combined + self.fusion_pos_embed  # before TransformerEncoder
```

### 6.2 Modality-Type Embeddings

Learned embeddings indicating audio vs text modality:
```python
self.modality_embed = nn.Embedding(2, 768)  # 0=audio, 1=text
# Add modality_embed[0] to audio tokens, modality_embed[1] to text tokens
```

### 6.3 Layer Normalization Placement

The current cross-attention uses **post-norm** (residual + LayerNorm after attention). Pre-norm (LayerNorm before attention) is often more stable for training and may be worth exploring.

---

## 7. Experimental Plan Summary

### 7.1 Configuration Table

| Exp | Mode | Placement | Q | K/V | Gate | New Params (~) |
|-----|------|-----------|---|-----|------|---------------|
| 1a | `audio_to_text` | `before` | audio (2 tokens) | text (512 tokens) | N/A | 2.4M |
| 1b | `audio_to_text` | `replace` | audio (2 tokens) | text (512 tokens) | N/A | 2.4M |
| 2a | `text_to_audio` | `before` | text (512 tokens) | audio (2 tokens) | N/A | 2.4M |
| 2b | `text_to_audio` | `replace` | text (512 tokens) | audio (2 tokens) | N/A | 2.4M |
| 3a | `gated_bidirectional` | `before` | both | both | learned | 4.7M |
| 3b | `gated_bidirectional` | `replace` | both | both | learned | 4.7M |
| baseline | None (original) | N/A | N/A | N/A | N/A | 0 |

### 7.2 Suggested Evaluation Metrics

- **Primary**: Validation F1 (binary), Validation Accuracy (5-fold CV average)
- **Secondary**: Test F1, Test Accuracy (on held-out test-dist set)
- **Diagnostic**: Train-val gap (overfitting indicator), convergence speed (epochs to best val loss), gate value evolution (for gated mode)
- **W&B tracking**: All metrics + learning rate + gate value (log `sigmoid(model.cross_attention.gate_raw).item()` per epoch for gated mode)

### 7.3 Expected Behavior / Hypotheses

| Experiment | Hypothesis |
|-----------|-----------|
| 1a (a2t, before) | Audio tokens get much richer representations. Self-attention fusion then refines. Likely best single-direction setup. |
| 1b (a2t, replace) | Lightweight but may underperform 1a since there's no self-attention refinement. Good for speed comparison. |
| 2a (t2a, before) | Text tokens enriched with acoustics. May help if text-based features are currently underweighted. |
| 2b (t2a, replace) | Risky: pooling from audio tokens that were never cross-attended means audio representation is unmodified. Likely worst performer. |
| 3a (gated, before) | Should match or beat 1a/2a. Watch the gate value: tells you which direction matters more. |
| 3b (gated, replace) | Moderate. Gate + replace removes self-attention, but gating provides flexibility. |

---

## 8. Recommendations

### 8.1 Which Placement to Start With

**`before`** (Option A). It keeps the existing self-attention fusion as a safety net while adding explicit cross-modal attention. This is the lowest-risk, highest-information-gain starting point.

### 8.2 Suggested Order of Experiments

1. **Baseline** (no cross-attention) - establish reference numbers with the exact same training setup
2. **Exp 1a**: `--cross_attention_mode audio_to_text --cross_attention_placement before` - most promising single direction
3. **Exp 2a**: `--cross_attention_mode text_to_audio --cross_attention_placement before` - compare direction
4. **Exp 3a**: `--cross_attention_mode gated_bidirectional --cross_attention_placement before` - let model choose
5. If results are promising, try `replace` variants (1b, 2b, 3b) to test if self-attention fusion is even necessary

### 8.3 Hyperparameter Starting Points

```bash
# Experiment 1a (recommended first run)
python main.py \
  --cross_attention_mode audio_to_text \
  --cross_attention_placement before \
  --cross_attention_num_heads 8 \
  --cross_attention_num_layers 1 \
  --cross_attention_dropout 0.1 \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 16 \
  --run_tag "xattn_a2t_before"

# Experiment 3a (gated bidirectional)
python main.py \
  --cross_attention_mode gated_bidirectional \
  --cross_attention_placement before \
  --cross_attention_num_heads 8 \
  --cross_attention_num_layers 1 \
  --cross_attention_dropout 0.1 \
  --cross_attention_gate_init 0.5 \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 16 \
  --run_tag "xattn_gated_before"
```

**Tips**:
- Keep `cross_attention_num_layers=1` initially. 2+ layers add significant parameters and risk overfitting on this small dataset (~166 samples).
- `cross_attention_dropout=0.1` is intentionally lower than `transformer_dropout=0.35` because cross-attention has fewer parameters to regularize.
- Monitor gate values for experiment 3: if gate→0 or gate→1 early, one direction dominates and you can simplify.
- Consider `--freeze_text_model_layers 6` to reduce total trainable parameters when adding cross-attention.
- The `replace` placement saves compute (no 514-token self-attention) which helps if memory is tight.

---

## 9. Appendix

### 9.1 Complete Tensor Shape Reference

#### Common to all experiments

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `(B, 512)` | DeBERTa tokenized input (after squeeze) |
| `attention_mask` | `(B, 512)` | 1=real token, 0=padding |
| `text_embeddings` | `(B, 512, 768)` | DeBERTa `last_hidden_state` |
| `audio_input` | `(B, 47)` | Handcrafted audio features |
| `audio_proj` | `(B, 1, 768)` | After `audio_fc` + unsqueeze |
| `emb_proj` | `(B, 1, 768)` | After `embedding_fc` + unsqueeze |
| `audio_tokens` | `(B, 2, 768)` | `cat(audio_proj, emb_proj, dim=1)` |
| `text_pad_mask` | `(B, 512)` | `True` where padding (inverted attention_mask) |

#### Experiment 1: audio_to_text

| Step | Tensor | Shape |
|------|--------|-------|
| Cross-attn Q | `audio_tokens` | `(B, 2, 768)` |
| Cross-attn K, V | `text_embeddings` | `(B, 512, 768)` |
| Cross-attn output | `fused_audio` | `(B, 2, 768)` |
| Text (unchanged) | `text_embeddings` | `(B, 512, 768)` |
| **before**: cat | `combined` | `(B, 514, 768)` |
| **before**: TransformerEncoder | `output` | `(514, B, 768)` |
| **before**: pool | `pooled` | `(B, 768)` |
| **replace**: mean pool fused_audio | `pooled` | `(B, 768)` |
| Classifier output | `logits` | `(B,)` |

#### Experiment 2: text_to_audio

| Step | Tensor | Shape |
|------|--------|-------|
| Cross-attn Q | `text_embeddings` | `(B, 512, 768)` |
| Cross-attn K, V | `audio_tokens` | `(B, 2, 768)` |
| Cross-attn output | `fused_text` | `(B, 512, 768)` |
| Audio (unchanged) | `audio_tokens` | `(B, 2, 768)` |
| **before**: cat | `combined` | `(B, 514, 768)` |
| **before**: TransformerEncoder | `output` | `(514, B, 768)` |
| **before**: pool | `pooled` | `(B, 768)` |
| **replace**: mean pool audio_tokens | `pooled` | `(B, 768)` |
| Classifier output | `logits` | `(B,)` |

#### Experiment 3: gated_bidirectional

| Step | Tensor | Shape |
|------|--------|-------|
| A2T cross-attn Q | `audio_tokens` | `(B, 2, 768)` |
| A2T cross-attn K, V | `text_embeddings` | `(B, 512, 768)` |
| A2T output | `a_cross` | `(B, 2, 768)` |
| T2A cross-attn Q | `text_embeddings` | `(B, 512, 768)` |
| T2A cross-attn K, V | `audio_tokens` | `(B, 2, 768)` |
| T2A output | `t_cross` | `(B, 512, 768)` |
| `gate` | `sigmoid(gate_raw)` | `(1,)` scalar |
| `audio_out` | `gate * a_cross + (1-gate) * audio_tokens` | `(B, 2, 768)` |
| `text_out` | `gate * text_emb + (1-gate) * t_cross` | `(B, 512, 768)` |
| **before**: cat | `combined` | `(B, 514, 768)` |
| **before**: TransformerEncoder | `output` | `(514, B, 768)` |
| **before**: pool | `pooled` | `(B, 768)` |
| **replace**: mean pool audio_out | `pooled` | `(B, 768)` |
| Classifier output | `logits` | `(B,)` |

### 9.2 Parameter Count Estimates

| Component | Parameters |
|-----------|-----------|
| DeBERTa-v3-base | ~86M (frozen or fine-tuned) |
| `audio_fc` (Linear 47→768 + LN) | 37,632 |
| `embedding_fc` (Linear 768→768 + LN) | 591,360 |
| TransformerEncoder (2 layers) | ~9.5M |
| Classifier (768→384→1) | 295,681 |
| **CrossAttn 1 layer (single dir)** | **~2.36M** |
| **CrossAttn 1 layer (gated bidir)** | **~4.72M + 1** |

### 9.3 Architecture Diagrams per Experiment x Placement

#### Exp 1a: audio_to_text + before

```
DeBERTa ─────────────────────────────────── text (B,512,768) ─────── K,V
audio_fc ─► (B,1,768) ─┐                                              │
emb_fc ──► (B,1,768) ──┤► audio_tokens (B,2,768) ─────────────────── Q
                        │                                              │
                        │                    CrossAttn(a→t) ◄──────────┘
                        │                         │
                        │                   fused_audio (B,2,768)
                        │                         │
                        └──── text unchanged ─────┤
                                                  │ cat
                                           (B, 514, 768)
                                                  │ permute
                                        TransformerEncoder
                                                  │ pool[0]
                                            (B, 768)
                                                  │
                                            Classifier
```

#### Exp 2a: text_to_audio + before

```
DeBERTa ──── text (B,512,768) ──── Q
audio_fc + emb_fc ─► audio_tokens (B,2,768) ──── K,V
                                                   │
                              CrossAttn(t→a) ◄─────┘
                                    │
                              fused_text (B,512,768)
                                    │
audio_tokens unchanged ─────────────┤ cat
                              (B, 514, 768)
                                    │ permute
                           TransformerEncoder
                                    │ pool[0]
                              (B, 768)
                                    │
                              Classifier
```

#### Exp 3a: gated_bidirectional + before

```
DeBERTa ──── text ──────────┬──── K,V (a2t) ──── Q (t2a) ────┐
audio_fc + emb_fc ─► audio ─┼──── Q (a2t) ───── K,V (t2a) ───┤
                             │                                 │
                     CrossAttn(a→t)                  CrossAttn(t→a)
                             │                                 │
                       a_cross (B,2,768)               t_cross (B,512,768)
                             │                                 │
                             └───── gate blending ─────────────┘
                                        │
                               audio_out    text_out
                                        │ cat
                                 (B, 514, 768)
                                        │ permute
                              TransformerEncoder
                                        │ pool[0]
                                   (B, 768)
                                        │
                                   Classifier
```

#### Any experiment + replace

```
CrossAttentionFusion(audio_tokens, text_tokens)
           │
     fused_audio (B, 2, 768)
           │
     mean(dim=1) → (B, 768)
           │
     Classifier → (B,) logits

[TransformerEncoder exists but is not called]
```
