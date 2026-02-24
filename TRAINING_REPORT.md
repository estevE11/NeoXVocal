## NeuroXVocal training & preprocessing report (this fork)

This document summarizes **how data is preprocessed** and **how training is run** in the current codebase, with emphasis on: windowing/chunking, train/val splitting, and possible leakage.

Assumption (as you stated): **each `.wav` corresponds to a different subject**.

---

### 1) What is a “sample” in this project?

A single training sample corresponds to **one recording** (`{patient_id}.wav`), represented by:

- **Text**: `{patient_id}.txt` transcript (Whisper output, then lightly cleaned).
- **Audio handcrafted features**: one row in `audio_features_*.csv` (summary stats over the whole wav).
- **Audio embeddings**: one row in `audio_embeddings_*.csv` (Wav2Vec2 mean-pooled embedding over the whole wav).
- **Label**: `Control=0`, `ProbableAD=1` (assigned by which folder/CSV is used).

The training dataloader returns:
`(text_tokens, audio_feature_vector, audio_embedding_vector, label)`.

---

### 2) Preprocessing pipeline (ADReSSo21 diagnosis)

The wrapper script `scripts/prepare_adresso_diagnosis.sh` builds a processed dataset tree under:

- `/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/`

It processes:

- **Train AD**: `.../ADReSSo21/diagnosis/train/audio/ad/*.wav`
- **Train CN**: `.../ADReSSo21/diagnosis/train/audio/cn/*.wav`
- **Test-dist**: `.../ADReSSo21/diagnosis/test-dist/audio/*.wav`

Outputs written (train):

```
.../processed_data/diagnosis/train/ad/
  *.txt
  audio_features_ad.csv
  audio_embeddings_ad.csv

.../processed_data/diagnosis/train/cn/
  *.txt
  audio_features_cn.csv
  audio_embeddings_cn.csv
```

Outputs written (test-dist):

```
.../processed_data/diagnosis/test-dist/
  *.txt
  audio_features_test.csv
  audio_embeddings_test.csv
```

The wrapper is restartable: it skips steps whose output file already exists and is non-empty.

---

### 2.1 Text extraction (Whisper)

`src/data_extraction/transcribe_audio.py` runs Whisper on **the full wav** and writes one transcript file per recording:

- output: `{patient_id}.txt`
- existing non-empty output is skipped

Then `src/data_processing/preprocess_texts.py` applies a minimal cleanup:

- lowercase
- whitespace normalization
- regex removing most special characters
- existing non-empty output is skipped

There is **no segmentation into 15-second chunks** for text.

---

### 2.2 Audio handcrafted features (librosa + parselmouth)

`src/data_extraction/extract_audio_features.py` extracts many summary statistics from the wav (pause features, pitch stats, MFCC means/stds, jitter/shimmer, formants, etc.) and writes **one CSV row per wav**.

Important distinction from your “15s chunks with overlap” setup:

- This script uses **short-time windowing internally** (frame_length / hop_length) to compute some features, but the final representation is **aggregated for the entire file**.
- Default parameters:
  - sample rate: 22050 Hz
  - frame_length: 2048 samples (~92.9 ms)
  - hop_length: 512 samples (~23.2 ms)

So the “windowing” here is **millisecond-scale analysis frames**, not seconds-scale segmentation into multiple samples.

Known data quality issue:
- Some feature columns (notably jitter/shimmer/formants) can be NaN for certain recordings.
- The training dataloader explicitly coerces non-numeric/NaN/Inf to 0.0 at load time so training does not produce NaN loss.

---

### 2.3 Audio embeddings (Wav2Vec2)

`src/data_extraction/extract_audio_embeddings.py`:

- loads the wav (resamples to 16 kHz if needed)
- runs a pretrained Wav2Vec2 model (default: `facebook/wav2vec2-base-960h`)
- mean-pools over time to a single 768-d embedding (one per wav)
- writes one CSV row per wav

Again: **no 15s chunking with overlap**; it embeds the whole recording and mean-pools.

---

### 3) Model inputs and forward pass (high level)

The model `src/train/models.py::NeuroXVocal` combines the three modalities:

- **Text**: DeBERTaV2 encoder produces a sequence of token embeddings.
- **Audio features**: projected to hidden size (Linear → LayerNorm → Dropout → ReLU).
- **Audio embeddings**: projected to hidden size (same idea).
- Concatenate `[audio_token, embedding_token, text_tokens]` then a small TransformerEncoder and a classifier head.

Output is a single logit; training uses `BCEWithLogitsLoss`.

---

### 4) Training loop overview

Training is run by `src/train/main.py` and `src/train/train.py`.

#### 4.1 Splitting strategy (“80/20 split”)

Training uses **StratifiedKFold** with `n_splits=NUM_FOLDS` (default 5), so each fold is approximately an **80/20 train/val** split:

- Train: ~4/5 of samples
- Val: ~1/5 of samples

Stratification keeps class proportions similar in each fold.

#### 4.2 Optimization

Per fold:

- optimizer: Adam
- scheduler: ReduceLROnPlateau (monitors val loss)
- gradient clipping: max_norm=1.0
- optional checkpointing of best val loss
- early stopping based on `EARLY_STOPPING_PATIENCE`

#### 4.3 What is logged

- A plain text `training.log` under the selected results directory.
- W&B logs:
  - **hyperparameters/config** from `wandb.init(config=...)` in `src/train/main.py`
  - per epoch: train/val loss, acc, F1, and LR

---

### 5) Leakage analysis (given “each wav is a different subject”)

If **each wav truly corresponds to one unique subject** and there are no duplicate/near-duplicate recordings, then:

- **Within a fold**, leakage between train and validation is unlikely because indices are disjoint.

However, there are still two realistic leakage / evaluation pitfalls:

#### 5.1 Cross-fold contamination (important for reporting CV performance)

The current fold reset uses `NeuroXVocal.reset_parameters()` which resets:
- audio_fc
- embedding_fc
- transformer_encoder
- classifier

It does **not** reset the pretrained DeBERTa text backbone (`self.text_model`).

Implication:
- Fold 2 starts with a text backbone already updated on Fold 1 train data.
- That can inflate multi-fold results if you average across folds.

If you want “true CV” estimates, the safest approach is to **instantiate a new model per fold** (or explicitly re-load the DeBERTa pretrained weights at fold start, or freeze the text model).

#### 5.2 Preprocessing leakage via shared transforms/statistics

In this repo, most preprocessing is per-file and does not compute dataset-wide statistics.
If you later introduce normalization/scalers fit on the whole dataset, you must fit them on **train-only** and apply to val/test to avoid leakage.

---

### 6) How this differs from “15s chunks with 25% overlap”

Your previous setup (chunking) creates **multiple samples per subject**. That introduces a big leakage risk unless you split by subject first:

- If chunks from the same original wav appear in both train and val, validation accuracy becomes artificially high.

In this codebase:

- There is **no chunking** into multiple training samples.
- The only “windowing” is short-time feature extraction inside librosa/parselmouth (milliseconds), and Wav2Vec2’s internal frame processing followed by mean pooling.

If you want to add 15s-chunking in the future, the split must be **group-aware by patient/subject id**, not random by chunk.

---

### 7) Practical sanity checks to detect problems

- **Check non-finite features**: if handcrafted features contain NaNs/Inf and they’re not handled, BCE loss can become NaN quickly.
- **Check class balance per fold**: StratifiedKFold addresses this.
- **Compare against simple baselines**:
  - TF-IDF over transcripts can already be very strong on this dataset; very high accuracy is not impossible.
- **Always report performance on an external set**:
  - the ADReSSo test-dist (now that you have released labels) is the best “unseen” check.

