import argparse
import os
import csv

import torch
import torch.nn as nn

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
    parser = argparse.ArgumentParser(
        description='Train NeuroXVocal multimodal model for AD classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # === Data and output paths ===
    data_group = parser.add_argument_group('Data paths')
    data_group.add_argument('--train_dir', type=str, default=None,
                            help='Processed train dir with ad/ and cn/ subdirs')
    data_group.add_argument('--results_dir', type=str, default=None,
                            help='Where to write logs/checkpoints')
    
    # === Training hyperparameters ===
    train_group = parser.add_argument_group('Training hyperparameters')
    train_group.add_argument('--batch_size', type=int, default=None,
                             help='Number of samples per training batch')
    train_group.add_argument('--epochs', type=int, default=None,
                             help='Total number of training epochs')
    train_group.add_argument('--lr', '--learning_rate', type=float, default=None, dest='lr',
                             help='Learning rate for the optimizer')
    train_group.add_argument('--num_folds', type=int, default=None,
                             help='Number of folds for cross-validation')
    train_group.add_argument('--early_stopping_patience', type=int, default=None,
                             help='Epochs with no improvement to trigger early stopping')
    train_group.add_argument('--weight_decay', type=float, default=1e-4,
                             help='Weight decay (L2 regularization)')
    train_group.add_argument('--gradient_clip_norm', type=float, default=1.0,
                             help='Max norm for gradient clipping')
    train_group.add_argument('--scheduler_factor', type=float, default=0.5,
                             help='Factor for ReduceLROnPlateau scheduler')
    train_group.add_argument('--scheduler_patience', type=int, default=5,
                             help='Patience for ReduceLROnPlateau scheduler')
    
    # === Text model configuration ===
    text_group = parser.add_argument_group('Text model configuration')
    text_group.add_argument('--text_embedding_model', type=str, default=None,
                            help='HuggingFace model name for text encoding (e.g., microsoft/deberta-v3-base)')
    text_group.add_argument('--freeze_text_model', action='store_true',
                            help='Freeze the entire text model (no gradient updates)')
    text_group.add_argument('--freeze_text_model_layers', type=int, default=None,
                            help='Number of text model layers to freeze from bottom')
    
    # === Transformer encoder configuration ===
    transformer_group = parser.add_argument_group('Transformer encoder configuration')
    transformer_group.add_argument('--transformer_num_heads', type=int, default=8,
                                   help='Number of attention heads in transformer encoder')
    transformer_group.add_argument('--transformer_num_layers', type=int, default=2,
                                   help='Number of transformer encoder layers')
    transformer_group.add_argument('--transformer_dim_feedforward', type=int, default=None,
                                   help='Feedforward dimension in transformer (default: 4 * hidden_size)')
    transformer_group.add_argument('--transformer_dropout', type=float, default=0.35,
                                   help='Dropout in transformer encoder')
    transformer_group.add_argument('--transformer_activation', type=str, default='gelu',
                                   choices=['relu', 'gelu'],
                                   help='Activation function in transformer')
    
    # === Feature projection configuration ===
    proj_group = parser.add_argument_group('Feature projection configuration')
    proj_group.add_argument('--feature_projection_dropout', type=float, default=0.3,
                            help='Dropout in audio/embedding projection layers')
    
    # === Classifier head configuration ===
    classifier_group = parser.add_argument_group('Classifier head configuration')
    classifier_group.add_argument('--classifier_hidden_layers', type=int, default=1,
                                  help='Number of hidden layers in classifier (0 = linear projection)')
    classifier_group.add_argument('--classifier_hidden_width', type=int, default=None,
                                  help='Width of classifier hidden layers (default: hidden_size // 2)')
    classifier_group.add_argument('--classifier_dropout', type=float, default=0.45,
                                  help='Dropout in classifier')
    classifier_group.add_argument('--classifier_activation', type=str, default='relu',
                                  choices=['relu', 'gelu', 'tanh', 'leaky_relu'],
                                  help='Activation function in classifier')
    classifier_group.add_argument('--num_classes', type=int, default=1,
                                  help='Number of output classes (1 for binary with BCEWithLogitsLoss)')
    
    # === Pooling configuration ===
    pool_group = parser.add_argument_group('Pooling configuration')
    pool_group.add_argument('--pooling_strategy', type=str, default='first',
                            choices=['first', 'mean', 'max', 'cls_token'],
                            help='How to pool transformer output')
    
    # === Logging and tracking ===
    log_group = parser.add_argument_group('Logging and tracking')
    log_group.add_argument('--wandb_project', type=str, default=os.environ.get('WANDB_PROJECT', 'NeoXVocal'),
                           help='Weights & Biases project name')
    log_group.add_argument('--wandb_entity', type=str, default=os.environ.get('WANDB_ENTITY'),
                           help='Weights & Biases entity/team name')
    log_group.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'),
                           choices=['online', 'offline', 'disabled'],
                           help='Weights & Biases logging mode')
    log_group.add_argument('--run_tag', type=str, default='',
                           help='Custom tag for the run')
    log_group.add_argument('--no_save_best_model', action='store_false', dest='save_best_model', default=True,
                           help='Disable saving the best performing model based on validation loss')
    
    # === Metadata (for tracking preprocessing) ===
    meta_group = parser.add_argument_group('Metadata (for tracking)')
    meta_group.add_argument('--asr_model', type=str, default='',
                            help='ASR model used for transcription (metadata only)')
    meta_group.add_argument('--audio_embedding_model', type=str, default='',
                            help='Audio embedding model used (metadata only)')
    
    args = parser.parse_args()

    train_dir = args.train_dir or BASE_DIR
    results_dir = args.results_dir or 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Resolve training hyperparameters (CLI args override config.py)
    batch_size = args.batch_size if args.batch_size is not None else (None if isinstance(BATCH_SIZE, str) else BATCH_SIZE)
    epochs = args.epochs if args.epochs is not None else (None if isinstance(EPOCHS, str) else EPOCHS)
    lr = args.lr if args.lr is not None else (None if isinstance(LEARNING_RATE, str) else LEARNING_RATE)
    num_folds = args.num_folds if args.num_folds is not None else (None if isinstance(NUM_FOLDS, str) else NUM_FOLDS)
    early_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else (None if isinstance(EARLY_STOPPING_PATIENCE, str) else EARLY_STOPPING_PATIENCE)
    )
    text_embedding_model = args.text_embedding_model or TEXT_EMBEDDING_MODEL

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

    # Build model configuration dictionary
    model_config = {
        'num_audio_features': num_audio_features,
        'num_embedding_features': NUM_EMBEDDING_FEATURES,
        'text_embedding_model': text_embedding_model,
        # Transformer encoder configuration
        'transformer_num_heads': args.transformer_num_heads,
        'transformer_num_layers': args.transformer_num_layers,
        'transformer_dim_feedforward': args.transformer_dim_feedforward,
        'transformer_dropout': args.transformer_dropout,
        'transformer_activation': args.transformer_activation,
        # Feature projection configuration
        'feature_projection_dropout': args.feature_projection_dropout,
        # Classifier head configuration
        'classifier_hidden_layers': args.classifier_hidden_layers,
        'classifier_hidden_width': args.classifier_hidden_width,
        'classifier_dropout': args.classifier_dropout,
        'classifier_activation': args.classifier_activation,
        'num_classes': args.num_classes,
        # Text model configuration
        'freeze_text_model': args.freeze_text_model,
        'freeze_text_model_layers': args.freeze_text_model_layers,
        # Pooling strategy
        'pooling_strategy': args.pooling_strategy,
    }

    model = NeuroXVocal(**model_config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Prepare wandb configuration to be passed to train_model
    # Each fold will create its own wandb run with this config
    wandb_config = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'base_run_name': run_name,
        'group': run_id,
        'mode': args.wandb_mode,
        'run_tag': args.run_tag,
        'slurm_job_id': slurm_job_id,
        'config': {
            # Training hyperparameters
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': args.weight_decay,
            'gradient_clip_norm': args.gradient_clip_norm,
            'scheduler_factor': args.scheduler_factor,
            'scheduler_patience': args.scheduler_patience,
            'num_folds': num_folds,
            'early_stopping_patience': early_patience,
            # Model architecture (all parameters)
            **model_config,
            # Metadata
            'asr_model': args.asr_model,
            'audio_embedding_model': args.audio_embedding_model,
            'slurm_job_id': slurm_job_id,
            'slurm_array_task_id': slurm_array_id,
            'device': str(device),
            'num_gpus_visible': torch.cuda.device_count(),
            'results_dir': results_dir,
            'train_dir': train_dir,
        },
    }

    train_model(
        model,
        full_dataset,
        epochs,
        lr,
        log_path,
        save_model_path,
        device,
        num_folds,
        args.save_best_model,
        batch_size=batch_size,
        early_stopping_patience=early_patience,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        wandb_config=wandb_config,
        run_id=run_id,
    )


if __name__ == '__main__':
    main()

