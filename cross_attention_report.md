# Cross-Attention Study: Implementation Report

## 1. Executive Summary

**Goal**: Integrate explicit cross-attention into NeuroXVocal for multimodal fusion between audio and text, replacing or augmenting the current implicit cross-modal interaction via self-attention over a concatenated token sequence.

**What was implemented**:
- One new class: `CrossAttentionFusion` (~80 lines in `models.py:8-89`)
- Six new parameters added to `NeuroXVocal.__init__` (all default to `None`/original behavior)
- ~30 lines of routing logic in `forward()`
- Six new CLI arguments in `main.py`
- Six new config defaults in `config.py`

**Three experimental modes** controlled by `--cross_attention_mode`:
1. `audio_to_text` -- audio tokens (Q) attend to text tokens (K,V)
2. `text_to_audio` -- text tokens (Q) attend to audio tokens (K,V)
3. `gated_bidirectional` -- both directions with a learned scalar gate

**Three placement options** controlled by `--cross_attention_placement`:
- `before` -- cross-attention then existing self-attention TransformerEncoder
- `replace` -- cross-attention only, TransformerEncoder is skipped
- `hybrid` -- same as `before` (alias for experiment naming clarity)

**Backward compatibility**: When `--cross_attention_mode` is omitted (default: `None`), the model behaves identically to the original. No existing behavior is changed.

---

## 2. Current Architecture Analysis

### 2.1 Three Modalities

| Modality | Source | Raw Shape | After Projection |
|----------|--------|-----------|-----------------|
| Audio features | 47 handcrafted (pitch, pauses, MFCCs...) | `(B, 47)` | `(B, 1, 768)` |
| Audio embeddings | Wav2Vec2 mean-pooled | `(B, 768)` | `(B, 1, 768)` |
| Text | DeBERTa-v3-base `last_hidden_state` | `(B, 512, 768)` | `(B, 512, 768)` |

### 2.2 Original Forward Pass (no cross-attention)

```
text_input ──> DeBERTa ──> (B, 512, 768) ─────────────────────────┐
audio_input ─> Linear(47,768)+LN+Drop+ReLU > unsqueeze > (B,1,768) ──┤
emb_input ──> Linear(768,768)+LN+Drop+ReLU > unsqueeze > (B,1,768) ──┤
                                                                   │
                                          cat(dim=1) <─────────────┘
                                              |
                                       (B, 514, 768)
                                              |
                                       permute(1,0,2)
                                              |
                                       (514, B, 768)
                                              |
                                   TransformerEncoder (2 layers, 8 heads)
                                              |
                                       (514, B, 768)
                                              |
                                   pool 'first' -> output[0]
                                              |
                                         (B, 768)
                                              |
                                   Linear(768,384)>ReLU>Drop>Linear(384,1)
                                              |
                                         (B,) logits
```

### 2.3 Identified Limitations of the Original

1. **No explicit cross-modal attention**: Cross-modal interaction competes with same-modal self-attention in shared heads.
2. **Asymmetric representation**: 2 audio tokens vs 512 text tokens. Audio is overwhelmed in self-attention (softmax denominator = 514).
3. **No positional encodings** in the fusion transformer.
4. **No modality-type embeddings**: No signal telling the model which tokens are audio vs text.
5. **First-token pooling bias**: `pool='first'` always takes the audio features token position.
6. **Text padding mask not propagated**: DeBERTa's `attention_mask` is NOT passed to the fusion TransformerEncoder. Padding tokens participate in fusion attention.

---

## 3. Cross-Attention: Theory

In standard self-attention, Q, K, V all come from the same sequence. In cross-attention:

```
Q = W_q @ modality_A     # The "querying" modality
K = W_k @ modality_B     # The "attended-to" modality
V = W_v @ modality_B     # Values from same source as keys

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Output has the **same sequence length as Q** but is enriched with information from the K/V modality.

| Property | Self-Attention (original) | Cross-Attention (new) |
|----------|--------------------------|----------------------|
| Q, K, V source | Same concatenated sequence | Q from one modality, K/V from another |
| Cross-modal interaction | Implicit (shared sequence) | Explicit (dedicated mechanism) |
| Attention budget | Shared across all 514 tokens | Dedicated to cross-modal alignment |
| Output shape | Same as input | Same as query modality |
| Asymmetry handling | Poor (2 vs 512 compete) | Natural (each direction is separate) |

**Relevant architectures**: Perceiver (latent queries into large input -- analogous to our audio-to-text), Flamingo (cross-attention between frozen LM and visual features), Bottleneck Transformer (bottleneck tokens cross-attend to multiple modalities).

---

## 4. Proposed Experiments

### Experiment 1: Audio-to-Text (`audio_to_text`)

**Q = audio (2 tokens), K = V = text (512 tokens)**

The 2 audio tokens query into 512 DeBERTa text tokens. Acts like a Perceiver: 2 latent queries compress 512 text tokens into 2 cross-attended representations.

**Intuition**: Audio representations seek the most relevant linguistic context.

**Tensor flow** (placement=`before`):
```
audio_tokens (B,2,768) ────> Q ─┐
text_tokens  (B,512,768) ──> K,V ┘──> CrossAttn ──> fused_audio (B,2,768)
                                                     text unchanged (B,512,768)
  cat ──> (B,514,768) ──> permute ──> TransformerEncoder ──> pool ──> classifier
```

**Pros**: Very lightweight (O(2*512) per head), natural bottleneck, audio tokens no longer drowned.
**Cons**: Unidirectional; text never explicitly attends to audio; only 2 query positions limits capacity.

### Experiment 2: Text-to-Audio (`text_to_audio`)

**Q = text (512 tokens), K = V = audio (2 tokens)**

Each of the 512 text tokens attends to the 2 audio tokens, enriching linguistic features with acoustic information.

**Intuition**: Text representations learn to selectively incorporate prosody, pause patterns, and speech quality.

**Tensor flow** (placement=`before`):
```
text_tokens  (B,512,768) ──> Q ─┐
audio_tokens (B,2,768)   ──> K,V ┘──> CrossAttn ──> fused_text (B,512,768)
                                                     audio unchanged (B,2,768)
  cat ──> (B,514,768) ──> permute ──> TransformerEncoder ──> pool ──> classifier
```

**Pros**: Every text token gets acoustic context; very cheap (O(512*2) per head).
**Cons**: Audio tokens never attend to text; 512 queries attending to only 2 keys may learn near-identical weightings.

### Experiment 3: Gated Bidirectional (`gated_bidirectional`)

**Both directions simultaneously**, blended with a learned gate.

```
Forward (audio->text):  Q=audio, K=V=text  ──> a_cross (B,2,768)
Reverse (text->audio):  Q=text, K=V=audio  ──> t_cross (B,512,768)

gate = sigmoid(gate_raw)  # scalar, initialized at 0.5

audio_out = gate * a_cross + (1-gate) * original_audio
text_out  = gate * original_text + (1-gate) * t_cross
```

**Gate initialization**: `gate_init=0.5` maps to `gate_raw = log(0.5/0.5) = 0.0`, so `sigmoid(0.0) = 0.5`. Balanced start.

**Intuition**: Model discovers which direction is more informative. If gate -> 1.0, audio-to-text dominates. If gate -> 0.0, text-to-audio dominates.

**Pros**: Subsumes experiments 1 and 2 as special cases; gate is interpretable; captures both directions.
**Cons**: ~2x parameters; gate may get stuck; two directions computed independently (no iterative cross-refinement).

---

## 5. Integration Points in the Codebase

### 5.1 Files Changed

| File | Lines Changed | What |
|------|--------------|------|
| `src/train/models.py` | +141 lines | Added `CrossAttentionFusion` class (lines 8-89). Added 6 cross-attention params to `NeuroXVocal.__init__` (lines 157-163). Modified `forward()` (lines 320-348). Updated `reset_parameters()` (lines 366-367). |
| `src/train/config.py` | +7 lines | Added 6 new defaults (lines 72-78) |
| `src/train/main.py` | +24 lines | Added 6 CLI args (lines 104-119). Wired into `model_config` (lines 222-228). Auto-logged to W&B via `**model_config`. |

### 5.2 New `CrossAttentionFusion` Class (`models.py:8-89`)

One class, three modes. Uses `nn.MultiheadAttention` directly with `batch_first=True`.

**Architecture per mode**:

| Mode | Modules Created | Params (1 layer, D=768, H=8) |
|------|----------------|------------------------------|
| `audio_to_text` | `a2t_layers` (MHA) + `a2t_norms` (LN) | ~2.36M |
| `text_to_audio` | `t2a_layers` (MHA) + `t2a_norms` (LN) | ~2.36M |
| `gated_bidirectional` | Both above + `gate_raw` (1 param) | ~4.72M |

Each cross-attention layer applies **post-norm residual**: `out = LayerNorm(x + CrossAttn(x))`.

### 5.3 Placement Logic in `forward()` (`models.py:320-348`)

```python
if self.cross_attention is not None:
    audio_tokens = cat(audio_proj, emb_proj)          # (B, 2, D)
    text_pad_mask = (attention_mask == 0)              # True = padding

    audio_out, text_out = self.cross_attention(audio_tokens, text_embeddings, text_pad_mask)
    fused_audio = audio_out if audio_out is not None else audio_tokens
    fused_text  = text_out  if text_out  is not None else text_embeddings

    if placement == 'replace':
        pooled = fused_audio.mean(dim=1)               # Skip TransformerEncoder
    else:  # 'before' or 'hybrid'
        combined = cat(fused_audio, fused_text)         # Feed into TransformerEncoder
        transformer_output = self.transformer_encoder(combined.permute(1,0,2))
        pooled = self._pool_output(transformer_output)
else:
    # Original path (unchanged)
```

**Key fix**: The implementation **propagates DeBERTa's attention mask** to cross-attention via `text_key_padding_mask`. Padding tokens are now properly excluded when text is the K/V source. This is an improvement over the original architecture where padding participated in fusion attention.

### 5.4 New Config Parameters (`config.py:72-78`)

```python
CROSS_ATTENTION_MODE = None           # None | 'audio_to_text' | 'text_to_audio' | 'gated_bidirectional'
CROSS_ATTENTION_NUM_HEADS = 8
CROSS_ATTENTION_NUM_LAYERS = 1
CROSS_ATTENTION_DROPOUT = 0.1
CROSS_ATTENTION_PLACEMENT = 'before'  # 'before' | 'replace' | 'hybrid'
CROSS_ATTENTION_GATE_INIT = 0.5
```

### 5.5 New CLI Arguments (`main.py:104-119`)

```
--cross_attention_mode {audio_to_text,text_to_audio,gated_bidirectional}
--cross_attention_num_heads 8
--cross_attention_num_layers 1
--cross_attention_dropout 0.1
--cross_attention_placement {before,replace,hybrid}
--cross_attention_gate_init 0.5
```

All automatically logged to W&B via `**model_config` in the wandb config dict (no extra wiring needed).

### 5.6 Impact on `reset_parameters()` (`models.py:366-367`)

```python
if self.cross_attention is not None:
    self.cross_attention.apply(reset_layer_parameters)
```

Cross-attention weights are properly reset between folds. The `initial_state` clone in `train.py` (line 64) also captures cross-attention parameters automatically since they are part of `model.state_dict()`.

### 5.7 Impact on Training

- **New parameters**: ~2.36M per single-direction layer, ~4.72M for gated bidirectional
- **Optimizer**: Cross-attention parameters are included in `model.parameters()` automatically. Same Adam optimizer and LR apply.
- **State dict**: Cross-attention is part of the model's `state_dict()`, so checkpoint save/load, fold reset, and W&B model saving all work without changes.
- **`train.py`**: No changes needed. The training loop calls `model(text_data, audio_data, embedding_data)` which routes through cross-attention transparently.

---

## 6. Additional Architectural Improvements (Not Implemented -- Future Work)

1. **Positional encodings** for the fusion sequence: `nn.Parameter(torch.randn(514, 1, 768) * 0.02)` added before TransformerEncoder.
2. **Modality-type embeddings**: `nn.Embedding(2, 768)` -- add `embed[0]` to audio tokens, `embed[1]` to text tokens.
3. **Pre-norm vs post-norm**: Current implementation uses post-norm. Pre-norm (LayerNorm before attention) is often more stable.
4. **Attention mask in the fusion TransformerEncoder**: Even in `before`/`hybrid` placement, the TransformerEncoder still doesn't receive a padding mask. Could pass one for the text portion of the concatenated sequence.

---

## 7. Experimental Plan Summary

### 7.1 Configuration Table

| Exp | `--cross_attention_mode` | `--cross_attention_placement` | Q | K/V | Gate | New Params |
|-----|--------------------------|-------------------------------|---|-----|------|-----------|
| 1a | `audio_to_text` | `before` | audio (2) | text (512) | -- | ~2.4M |
| 1b | `audio_to_text` | `replace` | audio (2) | text (512) | -- | ~2.4M |
| 2a | `text_to_audio` | `before` | text (512) | audio (2) | -- | ~2.4M |
| 2b | `text_to_audio` | `replace` | text (512) | audio (2) | -- | ~2.4M |
| 3a | `gated_bidirectional` | `before` | both | both | learned | ~4.7M |
| 3b | `gated_bidirectional` | `replace` | both | both | learned | ~4.7M |
| baseline | (omit flag) | -- | -- | -- | -- | 0 |

### 7.2 Evaluation Metrics

- **Primary**: Validation F1 (binary), Validation Accuracy -- 5-fold CV average
- **Secondary**: Test F1, Test Accuracy on held-out test-dist
- **Diagnostic**: Train-val gap (overfitting), convergence speed (epochs to best val loss), gate value evolution (for gated mode -- log `sigmoid(model.cross_attention.gate_raw).item()` per epoch)

### 7.3 Expected Behavior

| Experiment | Hypothesis |
|-----------|-----------|
| 1a (a2t + before) | Most promising. Audio tokens get rich linguistic context, then self-attention refines. |
| 1b (a2t + replace) | Lightweight but may underperform 1a (no self-attention refinement). |
| 2a (t2a + before) | Text enriched with acoustics. Useful if text features are currently underweighted. |
| 2b (t2a + replace) | Risky: pooling from unmodified audio tokens. Likely worst performer. |
| 3a (gated + before) | Should match or beat 1a/2a. Gate value reveals which direction matters. |
| 3b (gated + replace) | Moderate. Gate provides flexibility but no self-attention refinement. |

---

## 8. Recommendations

### 8.1 Start with `before` placement

It keeps the existing self-attention as a safety net. Lowest risk, highest information gain.

### 8.2 Suggested experiment order

1. **Baseline** -- re-run without `--cross_attention_mode` to establish reference numbers
2. **Exp 1a** -- `audio_to_text` + `before` (most promising single direction)
3. **Exp 2a** -- `text_to_audio` + `before` (compare direction)
4. **Exp 3a** -- `gated_bidirectional` + `before` (let model choose)
5. If promising, try `replace` variants (1b, 2b, 3b) to test if self-attention is even necessary

### 8.3 Hyperparameter starting points

**Experiment 1a** (recommended first run):
```bash
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
```

**Experiment 3a** (gated bidirectional):
```bash
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

### 8.4 Tips

- **Keep `num_layers=1`** initially. With ~166 training samples, more layers risk severe overfitting.
- **`cross_attention_dropout=0.1`** is intentionally lower than `transformer_dropout=0.35` because cross-attention has fewer parameters to regularize.
- **Monitor the gate** in experiment 3: if `sigmoid(gate_raw)` converges to 0 or 1 early, one direction dominates and you can simplify to that single direction.
- **Consider `--freeze_text_model_layers 6`** to reduce total trainable parameters when adding cross-attention. DeBERTa's bottom layers capture generic language; only top layers need task-specific tuning.
- **`replace` placement** saves significant compute (no 514-token self-attention). Useful if GPU memory is tight or for faster iteration.
- **The `replace` + `text_to_audio` combination** (Exp 2b) pools from the *unmodified* audio tokens via `fused_audio.mean(dim=1)`. Since audio tokens were unchanged in `text_to_audio` mode, this is essentially a projection-only baseline. Useful as a sanity check but unlikely to be competitive.

---

## 9. Appendix

### 9.1 Complete Tensor Shape Reference

**Common to all experiments:**

| Tensor | Shape | Source |
|--------|-------|--------|
| `input_ids` | `(B, 512)` | DeBERTa tokenizer, max_length=512 |
| `attention_mask` | `(B, 512)` | 1=real, 0=padding |
| `text_embeddings` | `(B, 512, 768)` | DeBERTa `last_hidden_state` |
| `audio_input` | `(B, 47)` | Handcrafted audio features |
| `audio_proj` | `(B, 1, 768)` | `audio_fc(audio_input).unsqueeze(1)` |
| `emb_proj` | `(B, 1, 768)` | `embedding_fc(embedding_input).unsqueeze(1)` |
| `audio_tokens` | `(B, 2, 768)` | `cat(audio_proj, emb_proj, dim=1)` |
| `text_pad_mask` | `(B, 512)` | `(attention_mask == 0)`, True=padding |

**Per-experiment shapes:**

| Step | audio_to_text | text_to_audio | gated_bidirectional |
|------|--------------|---------------|---------------------|
| CrossAttn Q | `(B, 2, 768)` | `(B, 512, 768)` | both |
| CrossAttn K,V | `(B, 512, 768)` | `(B, 2, 768)` | both |
| audio_out | `(B, 2, 768)` fused | `(B, 2, 768)` unchanged | `(B, 2, 768)` gated |
| text_out | `(B, 512, 768)` unchanged | `(B, 512, 768)` fused | `(B, 512, 768)` gated |
| cat (before) | `(B, 514, 768)` | `(B, 514, 768)` | `(B, 514, 768)` |
| TransformerEncoder | `(514, B, 768)` | `(514, B, 768)` | `(514, B, 768)` |
| pool (before) | `(B, 768)` | `(B, 768)` | `(B, 768)` |
| pool (replace) | `(B, 768)` mean of 2 | `(B, 768)` mean of 2 | `(B, 768)` mean of 2 |
| logits | `(B,)` | `(B,)` | `(B,)` |

### 9.2 Parameter Count Estimates

| Component | Parameters |
|-----------|-----------|
| DeBERTa-v3-base | ~86M |
| `audio_fc` (47->768 + LN) | ~37.6K |
| `embedding_fc` (768->768 + LN) | ~591K |
| TransformerEncoder (2 layers) | ~9.5M |
| Classifier (768->384->1) | ~296K |
| **CrossAttn 1 layer (single dir)** | **~2.36M** |
| **CrossAttn 1 layer (gated bidir)** | **~4.72M** |

### 9.3 Architecture Diagrams

**Exp 1a: audio_to_text + before**
```
DeBERTa ────> text (B,512,768) ──────> K,V ──┐
audio_fc ──> (B,1,768) ──┐                    |
emb_fc ───> (B,1,768) ──┤> audio (B,2,768) > Q
                         |                    |
                         |        CrossAttn(a->t)
                         |              |
                         |        fused_audio (B,2,768)
                         |              |
                         +-- text ------+ cat -> (B,514,768)
                                        |
                                 TransformerEncoder
                                        |
                                   pool -> (B,768) -> Classifier
```

**Exp 2a: text_to_audio + before**
```
DeBERTa ────> text (B,512,768) ──────> Q ────┐
audio_fc + emb_fc ──> audio (B,2,768) > K,V ─┘
                                              |
                                    CrossAttn(t->a)
                                              |
                                     fused_text (B,512,768)
                                              |
                     audio unchanged ─────────+ cat -> (B,514,768)
                                              |
                                       TransformerEncoder
                                              |
                                         pool -> Classifier
```

**Exp 3a: gated_bidirectional + before**
```
DeBERTa ──> text ────> K,V (a2t) / Q (t2a) ──┐
audio_fc+emb ──> audio > Q (a2t) / K,V (t2a) ─┤
                                                |
                              CrossAttn(a->t)  CrossAttn(t->a)
                                 |                  |
                           a_cross (B,2,768)  t_cross (B,512,768)
                                 |                  |
                                 +-- gate blend ----+
                                        |
                              audio_out    text_out
                                        | cat -> (B,514,768)
                                        |
                                 TransformerEncoder
                                        |
                                   pool -> Classifier
```

**Any experiment + replace**
```
CrossAttentionFusion(audio_tokens, text_tokens)
           |
     fused_audio (B, 2, 768)
           |
     mean(dim=1) -> (B, 768)
           |
     Classifier -> (B,) logits

     [TransformerEncoder exists but is NOT called]
```
