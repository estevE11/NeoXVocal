## Changes since fork (local additions)

### W&B + Slurm training

- Added W&B logging to training (metrics per epoch/fold, LR, F1; Slurm job id in run name/group and checkpoint name).
  - `src/train/main.py`
  - `src/train/train.py`
- Added Slurm submission script and docs:
  - `slurm/train_neuroxvocal.sbatch`
  - `slurm/README.md`

### ADReSSo21 preprocessing (diagnosis)

- Added restartable preprocessing wrapper writing next to ADReSSo21 under `.../adresso/processed_data/`:
  - `scripts/prepare_adresso_diagnosis.sh`
  - `scripts/README.md`
- Restored extraction scripts with skip-if-output-exists behavior:
  - `src/data_extraction/transcribe_audio.py`
  - `src/data_extraction/extract_audio_features.py`
  - `src/data_extraction/extract_audio_embeddings.py`
  - `src/data_processing/preprocess_texts.py`

### Environment

- Created `.venv/` and a combined requirements file:
  - `requirements_cls.txt`
  - `requirements_calcula.txt`

