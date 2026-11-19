#!/bin/bash
#SBATCH --job-name=sp_guided
#SBATCH --output=spatial_guided_mask_%j.log
#SBATCH --error=spatial_guided_mask_%j.err
#SBATCH --time=12:00:00
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

ae_checkpoint="${PIWM_AE_CHECKPOINT:-$checkpoint_root/spatial_ae/best.pt}"
cache="${PIWM_DIFFUSION_CACHE:-$checkpoint_root/spatial_diffusion/data}"
denoiser="${PIWM_DIFFUSION_CHECKPOINT:-$checkpoint_root/spatial_diffusion/model/best.pt}"
dynamics_checkpoint="${PIWM_DYNAMICS_CHECKPOINT:-$project_dir/dynamic/checkpoints/dynamics_discrete/best.pt}"
out="$checkpoint_root/spatial_guided_mask_${SLURM_JOB_ID}"

python -u spatial_ae/spatial_guided_diffusion.py --self-test
python -u spatial_ae/spatial_guided_diffusion.py \
    --ae-checkpoint "$ae_checkpoint" --cache "$cache" \
    --diffusion-checkpoint "$denoiser" \
    --dynamics-checkpoint "$dynamics_checkpoint" --project "$project_dir" \
    --out "$out" --pose-mode dynamics --split test \
    --max-pairs "${PIWM_GUIDED_PAIRS:-400}" --initialization diffusion \
    --save-n 40 --sample-steps 50 --iterations 10 \
    --start-t 20 --refine-steps 21 --guidance-step 1.0 \
    --max-update-rms 0.02 --theta-weight 0.02 --shape-weight 0.10 --seed 42

echo "Results: $out/summary.json"
echo "Images: $out/comparisons.zip"
