#!/usr/bin/env bash
set -euo pipefail

ADRESSO_ROOT="/home/usuaris/veussd/roger.esteve.sanchez/adresso"
ADRESSO21="${ADRESSO_ROOT}/ADReSSo21/diagnosis"
OUT_BASE="${ADRESSO_ROOT}/processed_data/diagnosis"

WHISPER_MODEL="${WHISPER_MODEL:-base}"
W2V2_MODEL="${W2V2_MODEL:-facebook/wav2vec2-base-960h}"

train_ad="${ADRESSO21}/train/audio/ad"
train_cn="${ADRESSO21}/train/audio/cn"
test_audio="${ADRESSO21}/test-dist/audio"

out_train_ad="${OUT_BASE}/train/ad"
out_train_cn="${OUT_BASE}/train/cn"
out_test="${OUT_BASE}/test-dist"

mkdir -p "$out_train_ad" "$out_train_cn" "$out_test"

scaler_audio_features="${SCALER_AUDIO_FEATURES:-src/inference/scaler_params_audio_features.pkl}"
scaler_audio_embeddings="${SCALER_AUDIO_EMB:-src/inference/scaler_params_audio_emb.pkl}"

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
fi

run_if_missing() {
  local out="$1"
  shift
  if [[ -s "$out" ]]; then
    echo "Skipping (exists): $out"
    return 0
  fi
  "$@"
}

run_if_missing_then_mark() {
  local marker="$1"
  shift
  if [[ -s "$marker" ]]; then
    echo "Skipping (exists): $marker"
    return 0
  fi
  "$@"
  date > "$marker"
}

echo "== Train AD =="
"$PYTHON" src/data_extraction/transcribe_audio.py "$train_ad" --output_dir "$out_train_ad" --model "$WHISPER_MODEL"
"$PYTHON" src/data_processing/preprocess_texts.py "$out_train_ad" "$out_train_ad"
run_if_missing "$out_train_ad/audio_features_ad.csv" "$PYTHON" src/data_extraction/extract_audio_features.py "$train_ad" --output_csv "$out_train_ad/audio_features_ad.csv"
run_if_missing "$out_train_ad/audio_embeddings_ad.csv" "$PYTHON" src/data_extraction/extract_audio_embeddings.py "$train_ad" --output_csv "$out_train_ad/audio_embeddings_ad.csv" --model_name "$W2V2_MODEL"

echo "== Train CN =="
"$PYTHON" src/data_extraction/transcribe_audio.py "$train_cn" --output_dir "$out_train_cn" --model "$WHISPER_MODEL"
"$PYTHON" src/data_processing/preprocess_texts.py "$out_train_cn" "$out_train_cn"
run_if_missing "$out_train_cn/audio_features_cn.csv" "$PYTHON" src/data_extraction/extract_audio_features.py "$train_cn" --output_csv "$out_train_cn/audio_features_cn.csv"
run_if_missing "$out_train_cn/audio_embeddings_cn.csv" "$PYTHON" src/data_extraction/extract_audio_embeddings.py "$train_cn" --output_csv "$out_train_cn/audio_embeddings_cn.csv" --model_name "$W2V2_MODEL"

echo "== Test-dist (unlabeled) =="
"$PYTHON" src/data_extraction/transcribe_audio.py "$test_audio" --output_dir "$out_test" --model "$WHISPER_MODEL"
"$PYTHON" src/data_processing/preprocess_texts.py "$out_test" "$out_test"
run_if_missing "$out_test/audio_features_test.csv" "$PYTHON" src/data_extraction/extract_audio_features.py "$test_audio" --output_csv "$out_test/audio_features_test.csv"
run_if_missing "$out_test/audio_embeddings_test.csv" "$PYTHON" src/data_extraction/extract_audio_embeddings.py "$test_audio" --output_csv "$out_test/audio_embeddings_test.csv" --model_name "$W2V2_MODEL"

echo "== Preprocess (standardize) CSVs =="
if [[ ! -s "$scaler_audio_features" || ! -s "$scaler_audio_embeddings" ]]; then
  echo "Skipping audio CSV preprocessing: missing scaler .pkl files."
  echo "Expected:"
  echo "  audio features scaler: $scaler_audio_features"
  echo "  audio embeddings scaler: $scaler_audio_embeddings"
  echo "Provide them at those paths or set SCALER_AUDIO_FEATURES / SCALER_AUDIO_EMB."
else
  run_if_missing_then_mark "$out_train_ad/.audio_features_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_features.py \
      --input_path "$out_train_ad/audio_features_ad.csv" \
      --output_path "$out_train_ad" \
      --scaler_path "$scaler_audio_features"

  run_if_missing_then_mark "$out_train_cn/.audio_features_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_features.py \
      --input_path "$out_train_cn/audio_features_cn.csv" \
      --output_path "$out_train_cn" \
      --scaler_path "$scaler_audio_features"

  run_if_missing_then_mark "$out_test/.audio_features_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_features.py \
      --input_path "$out_test/audio_features_test.csv" \
      --output_path "$out_test" \
      --scaler_path "$scaler_audio_features"

  run_if_missing_then_mark "$out_train_ad/.audio_embeddings_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_emb.py \
      "$out_train_ad/audio_embeddings_ad.csv" \
      "$scaler_audio_embeddings" \
      "$out_train_ad/audio_embeddings_ad.csv"

  run_if_missing_then_mark "$out_train_cn/.audio_embeddings_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_emb.py \
      "$out_train_cn/audio_embeddings_cn.csv" \
      "$scaler_audio_embeddings" \
      "$out_train_cn/audio_embeddings_cn.csv"

  run_if_missing_then_mark "$out_test/.audio_embeddings_preprocessed" \
    "$PYTHON" src/data_processing/preprocess_audio_emb.py \
      "$out_test/audio_embeddings_test.csv" \
      "$scaler_audio_embeddings" \
      "$out_test/audio_embeddings_test.csv"
fi

echo "Done. Outputs in: $OUT_BASE"