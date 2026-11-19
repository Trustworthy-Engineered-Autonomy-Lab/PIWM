#!/bin/bash
#SBATCH --job-name=sp_mask_ae
#SBATCH --output=sp_mask_ae_%j.log
#SBATCH --error=sp_mask_ae_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Submit from the repository root, or set PIWM_REPO_DIR to that directory.
set -euo pipefail
repo_dir="${PIWM_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
project_dir="${PIWM_PROJECT_DIR:-$repo_dir}"
checkpoint_root="${PIWM_CHECKPOINT_ROOT:-$project_dir/checkpoints}"
source "${PIWM_VENV:-$project_dir/.venv}/bin/activate"
cd "$repo_dir"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

data="${PIWM_AE_DATA:-$project_dir/data/lunar_train_intrinsic.npz}"
split="${PIWM_AE_SPLIT:-$checkpoint_root/spatial_ae/split_indices.npz}"
out="$checkpoint_root/spatial_mask_ae_${SLURM_JOB_ID}"
python -u spatial_ae/train_spatial_minimal_losses.py self-test
python -u spatial_ae/train_spatial_minimal_losses.py train \
    --arm mask_only --out "$out" --data "$data" --split "$split" \
    --state-cols 0 1 2 --appearance-channels 16 \
    --epochs 40 --lr 1e-4 --batch-size 32 --workers 4 --seed 42

echo "AE checkpoint: $out/best.pt"
echo "Reconstruction panels: $out/pictures_best"
