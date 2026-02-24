import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def _clone_state_dict(model: torch.nn.Module) -> dict:
    state = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _print_split_stats(prefix: str, y: np.ndarray) -> None:
    y = np.asarray(y).astype(int)
    total = int(y.size)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_pct = (100.0 * n_pos / total) if total else 0.0
    neg_pct = (100.0 * n_neg / total) if total else 0.0
    print(f'{prefix}: n={total} | Control(0)={n_neg} ({neg_pct:.1f}%) | ProbableAD(1)={n_pos} ({pos_pct:.1f}%)')


def train_model(
    model,
    full_dataset,
    epochs,
    learning_rate,
    log_path,
    save_model_path,
    device,
    num_folds=5,
    save_best_model=False,
    batch_size=32,
    early_stopping_patience=10,
    weight_decay=1e-4,
    wandb_config=None,
    run_id='',
):
    criterion = BCEWithLogitsLoss()
    if hasattr(full_dataset, 'datasets') and len(getattr(full_dataset, 'datasets', [])) == 2:
        ad_len = len(full_dataset.datasets[0])
        cn_len = len(full_dataset.datasets[1])
        y = np.array([1] * ad_len + [0] * cn_len)
    else:
        y = np.zeros(len(full_dataset), dtype=int)
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    _print_split_stats('Dataset', y)

    # IMPORTANT: reset *all* parameters between folds (including DeBERTa).
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    initial_state = _clone_state_dict(base_model)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(full_dataset)), y)):
        print(f'Fold {fold+1}/{num_folds}')
        _print_split_stats('  Train_split', y[train_indices])
        _print_split_stats('  Val_split', y[val_indices])

        # Initialize a separate wandb run for each fold
        wandb_run = None
        if wandb_config is not None:
            slurm_job_id = wandb_config.get('slurm_job_id', os.environ.get('SLURM_JOB_ID', 'local'))
            fold_run_name = f"{wandb_config.get('base_run_name', 'neuroxvocal')}_fold{fold+1}"
            fold_tags = [f"slurm_job_{slurm_job_id}"]
            if wandb_config.get('run_tag'):
                fold_tags.append(wandb_config['run_tag'])

            wandb_run = wandb.init(
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity'),
                name=fold_run_name,
                group=wandb_config.get('group', run_id),
                mode=wandb_config.get('mode', 'online'),
                tags=fold_tags,
                config={
                    **wandb_config.get('config', {}),
                    'fold': fold + 1,
                    'num_folds': num_folds,
                },
                reinit=True,
            )

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(initial_state)
        else:
            model.load_state_dict(initial_state)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            for (text_data, audio_data, embedding_data, label) in tqdm(
                train_loader, desc=f'Fold {fold+1}, Epoch {epoch+1}/{epochs} - Training'
            ):
                optimizer.zero_grad()
                text_data = {key: value.to(device) for key, value in text_data.items()}
                audio_data = audio_data.to(device)
                embedding_data = embedding_data.to(device)
                label = label.to(device)

                outputs = model(text_data, audio_data, embedding_data)
                print("train--------------------------------")
                print(outputs)
                print(label)
                print("--------------------------------")
                loss = criterion(outputs, label.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                all_train_outputs.extend(outputs.detach().cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

            train_loss = running_loss / len(train_loader)
            train_predictions = (np.array(all_train_outputs) > 0).astype(int)
            train_accuracy = accuracy_score(np.array(all_train_labels), train_predictions)
            train_f1 = f1_score(np.array(all_train_labels), train_predictions, average='binary', zero_division=0)
            train_report = classification_report(
                np.array(all_train_labels),
                train_predictions,
                target_names=['Control', 'ProbableAD'],
                digits=4,
                zero_division=0,
            )

            model.eval()
            val_running_loss = 0.0
            all_val_outputs = []
            all_val_labels = []
            with torch.no_grad():
                for (text_data, audio_data, embedding_data, label) in tqdm(
                    val_loader, desc=f'Fold {fold+1}, Epoch {epoch+1}/{epochs} - Validation'
                ):
                    text_data = {key: value.to(device) for key, value in text_data.items()}
                    audio_data = audio_data.to(device)
                    embedding_data = embedding_data.to(device)
                    label = label.to(device)

                    outputs = model(text_data, audio_data, embedding_data)
                    print("val--------------------------------")
                    print(outputs)
                    print(label)
                    print("--------------------------------")
                    loss = criterion(outputs, label.float())

                    val_running_loss += loss.item()
                    all_val_outputs.extend(outputs.detach().cpu().numpy())
                    all_val_labels.extend(label.cpu().numpy())

            val_loss = val_running_loss / len(val_loader)
            val_predictions = (np.array(all_val_outputs) > 0).astype(int)
            val_accuracy = accuracy_score(np.array(all_val_labels), val_predictions)
            val_f1 = f1_score(np.array(all_val_labels), val_predictions, average='binary', zero_division=0)
            val_report = classification_report(
                np.array(all_val_labels),
                val_predictions,
                target_names=['Control', 'ProbableAD'],
                digits=4,
                zero_division=0,
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if wandb_run is not None:
                wandb_run.log(
                    {
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'lr': current_lr,
                        'train/loss': train_loss,
                        'train/acc': train_accuracy,
                        'train/f1': train_f1,
                        'val/loss': val_loss,
                        'val/acc': val_accuracy,
                        'val/f1': val_f1,
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if save_best_model:
                    suffix = f'_{run_id}' if run_id else ''
                    best_model_path = f'{save_model_path}{suffix}_fold{fold+1}_best.pth'
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), best_model_path)
                    else:
                        torch.save(model.state_dict(), best_model_path)
                    if wandb_run is not None:
                        wandb_run.save(best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                    break

            with open(log_path, 'a') as f:
                f.write(
                    f"Fold {fold+1}, Epoch {epoch+1}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                )
                f.write("Train Classification Report:\n")
                f.write(f"{train_report}\n")
                f.write("Validation Classification Report:\n")
                f.write(f"{val_report}\n")

            print(
                f"Fold {fold+1}, Epoch {epoch+1}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            print("Train Classification Report:")
            print(train_report)
            print("Validation Classification Report:")
            print(val_report)

        # Finish the wandb run for this fold
        if wandb_run is not None:
            wandb_run.save(log_path)
            wandb.finish()
