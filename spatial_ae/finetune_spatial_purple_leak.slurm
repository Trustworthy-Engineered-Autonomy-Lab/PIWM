#!/bin/bash
#SBATCH --job-name=sp_purple_leak
#SBATCH --output=sp_purple_leak_%j.log
#SBATCH --error=sp_purple_leak_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --account=iruchkin

set -euo pipefail
module purge
module load python/3.10
source /blue/iruchkin/liammcglothlin/piwm_diffusion/.venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

for required in train_lander_spatial_ae.py train_lunar.py constraint_checker.py \
    evaluate_lander_crop_holdout.py retest_spatial_shape_theta.py \
    checkpoints/spatial_ae_43750874/best_shape.pt \
    checkpoints/vae_shape_first_42508780/split_indices.npz; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

python -u finetune_spatial_purple_leak.py --self-test

run_dir="./checkpoints/spatial_purple_leak_${SLURM_JOB_ID}"
python -u finetune_spatial_purple_leak.py \
    --checkpoint ./checkpoints/spatial_ae_43750874/best_shape.pt \
    --split ./checkpoints/vae_shape_first_42508780/split_indices.npz \
    --out "$run_dir" \
    --epochs 8 \
    --batch-size 32 \
    --workers 4 \
    --lr 1e-5 \
    --radius 6 \
    --lambda-leak 0.1 \
    --far-leak-boost 5.0 \
    --max-iou-drop 0.01 \
    --max-near-theta-drop 0.02 \
    --seed 42

# Only the selected validation checkpoint is tested on held-out trajectories.
python -u retest_spatial_shape_theta.py \
    --checkpoint "$run_dir/best.pt" \
    --test-root ./PIWM/dynamic/data/lunar/lunar_discrete/train/shard_0029 \
    --path-contains shard_0029 \
    --max-files 200 \
    --theta-col 4 \
    --threshold 0.30 \
    --near-threshold 0.65 \
    --min-pixels 20 \
    --batch-size 64 \
    --workers 4 \
    --save-n 40 \
    --out "$run_dir/holdout"

echo "Selected checkpoint: $run_dir/best.pt"
echo "Holdout summary: $run_dir/holdout/summary.json"
echo "Images: $run_dir/holdout/spatial_shape_theta_comparisons.zip"
