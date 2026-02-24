import argparse
import os
import csv

import torch
import torch.nn as nn
import wandb

from config import *
from data_loader import create_full_dataset
from models import NeuroXVocal
from train import train_model


def _infer_num_audio_features(audio_csv_path: str) -> int:
    with open(audio_csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    feature_cols = [c for c in header if c not in ('patient_id', 'class')]
    return len(feature_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=None, help='Processed train dir with ad/ and cn/ subdirs')
    parser.add_argument('--results_dir', type=str, default=None, help='Where to write logs/checkpoints')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--wandb_project', type=str, default=os.environ.get('WANDB_PROJECT', 'NeoXVocal'))
    parser.add_argument('--wandb_entity', type=str, default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--asr_model', type=str, default='')
    parser.add_argument('--audio_embedding_model', type=str, default='')
    args = parser.parse_args()

    train_dir = args.train_dir or BASE_DIR
    results_dir = args.results_dir or 'results'
    os.makedirs(results_dir, exist_ok=True)

    batch_size = args.batch_size if args.batch_size is not None else (None if isinstance(BATCH_SIZE, str) else BATCH_SIZE)
    epochs = args.epochs if args.epochs is not None else (None if isinstance(EPOCHS, str) else EPOCHS)
    lr = args.lr if args.lr is not None else (None if isinstance(LEARNING_RATE, str) else LEARNING_RATE)
    num_folds = args.num_folds if args.num_folds is not None else (None if isinstance(NUM_FOLDS, str) else NUM_FOLDS)
    early_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else (None if isinstance(EARLY_STOPPING_PATIENCE, str) else EARLY_STOPPING_PATIENCE)
    )

    missing = [k for k, v in {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'num_folds': num_folds,
        'early_stopping_patience': early_patience,
    }.items() if v is None]
    if missing:
        missing_str = ', '.join(missing)
        raise ValueError(f'Missing required values (set in config.py or pass CLI): {missing_str}')

    log_path = os.path.join(results_dir, 'training.log')
    save_model_path = os.path.join(results_dir, 'model')
    open(log_path, 'w').close()

    ad_dir = os.path.join(train_dir, 'ad')
    cn_dir = os.path.join(train_dir, 'cn')
    ad_audio_csv = os.path.join(ad_dir, 'audio_features_ad.csv')
    cn_audio_csv = os.path.join(cn_dir, 'audio_features_cn.csv')
    num_audio_features = _infer_num_audio_features(ad_audio_csv)
    full_dataset = create_full_dataset(
        ad_dir,
        cn_dir,
        ad_audio_csv,
        cn_audio_csv,
        os.path.join(ad_dir, 'audio_embeddings_ad.csv'),
        os.path.join(cn_dir, 'audio_embeddings_cn.csv'),
    )

    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')

    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    run_id = f'job{slurm_job_id}' + (f'_arr{slurm_array_id}' if slurm_array_id else '')
    run_name = f'neuroxvocal-{run_id}' + (f'-{args.run_tag}' if args.run_tag else '')

    model = NeuroXVocal(
        num_audio_features=num_audio_features,
        num_embedding_features=NUM_EMBEDDING_FEATURES,
        text_embedding_model=TEXT_EMBEDDING_MODEL,
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=run_id,
        mode=args.wandb_mode,
        config={
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': args.weight_decay,
            'num_folds': num_folds,
            'early_stopping_patience': early_patience,
            'text_embedding_model': TEXT_EMBEDDING_MODEL,
            'num_audio_features': num_audio_features,
            'num_embedding_features': NUM_EMBEDDING_FEATURES,
            'asr_model': args.asr_model,
            'audio_embedding_model': args.audio_embedding_model,
            'slurm_job_id': slurm_job_id,
            'slurm_array_task_id': slurm_array_id,
            'device': str(device),
            'num_gpus_visible': torch.cuda.device_count(),
            'results_dir': results_dir,
            'train_dir': train_dir,
        },
    )

    train_model(
        model,
        full_dataset,
        epochs,
        lr,
        log_path,
        save_model_path,
        device,
        num_folds,
        SAVE_BEST_MODEL,
        batch_size=batch_size,
        early_stopping_patience=early_patience,
        weight_decay=args.weight_decay,
        wandb_run=run,
        run_id=run_id,
    )
    run.save(log_path)
    wandb.finish()


if __name__ == '__main__':
    main()

