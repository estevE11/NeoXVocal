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

# NCCL Stability Fix (Prevents timeouts on nodes like veuc12)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Redirect cache to project directory to avoid /tmp space issues
export HF_HOME="/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/.cache/huggingface"
export WANDB_DIR="/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/.cache/wandb"
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

# ===================================
# Training Configuration
# ===================================
# Training Hyperparameters: batch_size=16, lr=1e-3, weight_decay=1e-5, epochs=200 (from original paper)
# Model Architecture: Matches paper (2 transformer layers, 8 heads, 1 classifier layer with width 384)
# ===================================

srun python -u src/train/main.py \
  --train_dir "$TRAIN_DIR" \
  --results_dir "$RESULTS_DIR" \
  --batch_size 8 \
  --epochs 200 \
  --lr 5e-5 \
  --num_folds 5 \
  --early_stopping_patience 15 \
  --weight_decay 1e-3 \
  --text_embedding_model 'microsoft/deberta-v3-base' \
  --transformer_num_layers 2 \
  --transformer_num_heads 8 \
  --transformer_dropout 0.4 \
  --classifier_hidden_layers 1 \
  --classifier_hidden_width 384 \
  --classifier_dropout 0.5 \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  --wandb_mode "$WANDB_MODE" \
  --audio_embedding_model "$AUDIO_EMBEDDING_MODEL" \
  --asr_model "$ASR_MODEL" \
  --cross_attention_mode gated_bidirectional \
  --cross_attention_placement before \
  --cross_attention_num_heads 4 \
  --cross_attention_num_layers 2 \
  --cross_attention_dropout 0.3 \
  --cross_attention_gate_init 0.5 \
  --run_tag "xattn_gated_regularized"

