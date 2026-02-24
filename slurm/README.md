## Slurm training

This repo provides `slurm/train_neuroxvocal.sbatch` to train `NeuroXVocal` on the ADReSSo21 diagnosis task using the processed data created by `scripts/prepare_adresso_diagnosis.sh`.

### Submit

```bash
sbatch slurm/train_neuroxvocal.sbatch
```

Defaults:
- Train dir: `/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train`
- Results dir: `results/$SLURM_JOB_ID`

To override:

```bash
sbatch slurm/train_neuroxvocal.sbatch /path/to/train_dir /path/to/results_dir
```

### W&B

You can set:

```bash
export WANDB_PROJECT="NeoXVocal"
export WANDB_ENTITY="veu"
export WANDB_MODE="online"   # or offline
```

