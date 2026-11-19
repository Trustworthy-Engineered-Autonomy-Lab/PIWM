# Spatial LunarLander AE

The current model uses `SpatialLanderAE` from `train_lander_spatial_ae.py` and
the `mask_only` configuration in `train_spatial_minimal_losses.py`. Both the
encoder and decoder train from random initialization. The encoder produces
three state estimates and a 16 x 6 x 8 appearance grid from a 96 x 128 RGB frame.

The original architecture and checkpoint format also support the conditional
diffusion pipeline in [README_diffusion.md](README_diffusion.md).
The older VAE score network uses a different latent representation.

## Current training loss

| Term | Weight |
| --- | --- |
| Full-image RGB BCE | 1.0 |
| Chroma | 4.0 |
| Latent state | 1.0 |
| Balanced purple-mask BCE on eligible crops | 0.25 |
| Dice, boundary, moments, local RGB L1 | 0.0 |

Frames have equal weights. Chroma compares the R-G, R-B and G-B contrasts using
mean error, the worst one percent of pixels, and a penalty above 0.02. Its exact
formula is in `chroma()` in the minimal-loss trainer. State supervision uses
the first three latent values for x, y and theta. Checkpoint selection maximizes
validation theta satisfaction on eligible frames, with IoU as a tie-break.

The trainer also retains the other ablation configurations for reproducibility.
The original `train_lander_spatial_ae.slurm` and purple-leak fine-tuning files
describe earlier experiments. Use `train_spatial_mask.slurm` for the current AE.

## Train

Run commands from the repository root with Python 3.10 or newer and PyTorch,
NumPy, SciPy and Matplotlib installed.

```bash
python spatial_ae/train_spatial_minimal_losses.py self-test
python spatial_ae/train_spatial_minimal_losses.py train \
  --arm mask_only --data /path/to/lunar_train_intrinsic.npz \
  --split /path/to/split_indices.npz --state-cols 0 1 2 \
  --appearance-channels 16 --epochs 40 --lr 1e-4 \
  --batch-size 32 --workers 4 --seed 42 --out checkpoints/spatial_mask_ae
```

The reconstruction archive contains `frame` with shape (N,96,128,3) and `state`
with x/y/theta in columns 0/1/2. The split file contains `train` and `validation`
indices that partition that archive. Training writes `best.pt`, `last.pt`,
`settings.json`, `summary.json`, metrics, and `pictures_best` reconstruction panels.

On a Slurm cluster, submit from the repository root with your allocation's
account and partition options.

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  spatial_ae/train_spatial_mask.slurm
```

The launcher defaults to the repository directory and its `.venv`. Configure
paths through `PIWM_PROJECT_DIR`, `PIWM_CHECKPOINT_ROOT`, `PIWM_VENV`,
`PIWM_AE_DATA` and `PIWM_AE_SPLIT`. Set `PIWM_REPO_DIR` when submitting outside
the repository root. Load any cluster-specific Python modules before submission.

Training writes its checkpoint to the path printed at the end of the job.
Set `PIWM_AE_CHECKPOINT` to that file when preparing latents and training diffusion.
The generic default is `checkpoints/spatial_ae/best.pt`. Model weights, split
indices, and datasets are supplied separately from these scripts.
