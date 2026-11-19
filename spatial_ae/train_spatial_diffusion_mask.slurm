#!/bin/bash
#SBATCH --job-name=spatial_diff
#SBATCH --output=spatial_diff_%j.log
#SBATCH --error=spatial_diff_%j.err
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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg

run_dir="$checkpoint_root/spatial_diffusion_mask_${SLURM_JOB_ID}"
ae_checkpoint="${PIWM_AE_CHECKPOINT:-$checkpoint_root/spatial_ae/best.pt}"
trajectory_root="${PIWM_TRAJECTORY_ROOT:-$project_dir/dynamic/data/lunar/lunar_discrete/train}"
dynamics_checkpoint="${PIWM_DYNAMICS_CHECKPOINT:-$project_dir/dynamic/checkpoints/dynamics_discrete/best.pt}"

python -u spatial_ae/spatial_ae_diffusion.py self-test
python -u spatial_ae/spatial_ae_diffusion.py prepare \
    --ae-checkpoint "$ae_checkpoint" --train-root "$trajectory_root" \
    --test-root "$trajectory_root/shard_0029" --exclude-shard shard_0029 \
    --horizon 2 --max-train-files 400 --max-test-files 50 \
    --max-train-pairs 20000 --max-eval-pairs 400 --out "$run_dir/data"
python -u spatial_ae/spatial_ae_diffusion.py train \
    --ae-checkpoint "$ae_checkpoint" --cache "$run_dir/data" \
    --out "$run_dir/model" --epochs 100 --batch-size 64 --lr 2e-4 \
    --diffusion-steps 1000 --shape-weight 0.1 --rgb-weight 0.05 --save-every 10
python -u spatial_ae/spatial_ae_diffusion.py sample \
    --ae-checkpoint "$ae_checkpoint" --cache "$run_dir/data" \
    --diffusion-checkpoint "$run_dir/model/best.pt" \
    --dynamics-checkpoint "$dynamics_checkpoint" --project "$project_dir" \
    --pose-mode both --sample-steps 50 --max-pairs 400 --save-n 40 \
    --out "$run_dir/test"

echo "Denoiser: $run_dir/model/best.pt"
echo "Results: $run_dir/test/summary.json"
echo "Images: $run_dir/test/comparisons.zip"
