## ADReSSo21 (diagnosis) preprocessing

This repo expects **processed** data in the structure below for training:

```
processed_data/diagnosis/train/
  ad/
    *.txt
    audio_features_ad.csv
    audio_embeddings_ad.csv
  cn/
    *.txt
    audio_features_cn.csv
    audio_embeddings_cn.csv
```

### One-command preparation (your system paths)

Run:

```bash
bash scripts/prepare_adresso_diagnosis.sh
```

Outputs are written to:

- `/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/`

### Optional: choose pretrained models

```bash
WHISPER_MODEL=base W2V2_MODEL=facebook/wav2vec2-base-960h \
  bash scripts/prepare_adresso_diagnosis.sh
```

