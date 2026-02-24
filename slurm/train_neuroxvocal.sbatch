#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/results/logs/sbatch/neuroxvocal/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal"
cd "$REPO_DIR"

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Cache dirs (prefer node-local scratch if provided by site)
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-$REPO_DIR}}"
export HF_HOME="${SCRATCH_BASE}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export WANDB_DIR="${SCRATCH_BASE}/wandb/${SLURM_JOB_ID}"
mkdir -p "$HF_HOME" "$WANDB_DIR"

date

# Defaults (override by passing args to sbatch or exporting env vars)
TRAIN_DIR_DEFAULT="/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train"
TRAIN_DIR="${1:-$TRAIN_DIR_DEFAULT}"
RESULTS_DIR="${2:-${REPO_DIR}/results/${SLURM_JOB_ID}}"

WANDB_PROJECT="${WANDB_PROJECT:-NeoXVocal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Optional provenance strings for W&B config
ASR_MODEL="${ASR_MODEL:-}"
AUDIO_EMBEDDING_MODEL="${AUDIO_EMBEDDING_MODEL:-facebook/wav2vec2-base-960h}"

mkdir -p "$RESULTS_DIR" "/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/results/logs/sbatch/neuroxvocal"

srun python -u src/train/main.py \
  --train_dir "$TRAIN_DIR" \
  --results_dir "$RESULTS_DIR" \
  --batch_size 8 \
  --epochs 25 \
  --lr 1e-4 \
  --num_folds 5 \
  --early_stopping_patience 10 \
  --weight_decay 1e-4 \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  --wandb_mode "$WANDB_MODE" \
  --audio_embedding_model "$AUDIO_EMBEDDING_MODEL" \
  --asr_model "$ASR_MODEL" \
  --run_tag "slurm"

