#!/bin/bash
#SBATCH --job-name=prepare-adresso-data
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/logs/sbatch/prepare_data_%j.txt
#SBATCH --error=/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/logs/sbatch/prepare_data_%j.err
#SBATCH --partition=veu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --nodelist=veuc12

echo "Starting data preparation job on $(hostname)"
date

# Redirect cache to project directory to avoid /tmp space issues
export HF_HOME="/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/.cache/huggingface"
export WANDB_DIR="/home/usuaris/veussd/roger.esteve.sanchez/NeoXVocal/.cache/wandb"
mkdir -p $HF_HOME $WANDB_DIR

# =================================================================================
# OPTIMIZED SETTINGS FOR 24GB GPU (RTX 3090)
#
# 1. Use a more accurate Whisper model. The 'medium' model fits easily in 24GB VRAM.
# 2. Increase the Wav2Vec2 chunk size to capture more context.
#    A 240-second (4-minute) chunk is a safe and significant improvement over 20s.
# =================================================================================
export WHISPER_MODEL='medium'
export W2V2_CHUNK_SECONDS='0'


# --- Execute the main preparation script ---
# The script will automatically pick up the environment variables set above.
srun ./scripts/prepare_adresso_diagnosis.sh

echo "Data preparation job finished"
date
