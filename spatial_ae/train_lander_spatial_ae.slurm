#!/bin/bash
#SBATCH --job-name=spatial_ae
#SBATCH --output=/blue/iruchkin/liammcglothlin/piwm_diffusion/spatial_ae_%j.log
#SBATCH --error=/blue/iruchkin/liammcglothlin/piwm_diffusion/spatial_ae_%j.err
#SBATCH --chdir=/blue/iruchkin/liammcglothlin/piwm_diffusion
#SBATCH --time=12:00:00
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
cd /blue/iruchkin/liammcglothlin/piwm_diffusion

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

for required in train_lander_spatial_ae.py train_lunar.py constraint_checker.py \
    checkpoints/intrinsic_vae_200ep/best.pt \
    checkpoints/vae_shape_first_42508780/last.pt \
    checkpoints/vae_shape_first_42508780/split_indices.npz \
    checkpoints/deterministic_ae_43703401/ae/best_qualified.pt; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

# Stop before training if shape, gradient, warm-start or checkpoint tests fail.
python -u train_lander_spatial_ae.py --self-test

# One spatial AE, 40 epochs. Baseline models are evaluated, NOT retrained.
python -u train_lander_spatial_ae.py \
    --data /blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz \
    --checkpoint ./checkpoints/vae_shape_first_42508780/last.pt \
    --original-checkpoint ./checkpoints/intrinsic_vae_200ep/best.pt \
    --flat-checkpoint ./checkpoints/deterministic_ae_43703401/ae/best_qualified.pt \
    --split ./checkpoints/vae_shape_first_42508780/split_indices.npz \
    --out "./checkpoints/spatial_ae_${SLURM_JOB_ID}" \
    --appearance-channels 16 \
    --epochs 40 \
    --lr 1e-4 \
    --batch-size 32 \
    --workers 4 \
    --state-cols 0 1 2 \
    --state-weight 1.0 \
    --near-weight 5.0 \
    --global-weight 1.0 \
    --mask-bce-weight 1.0 \
    --dice-weight 1.0 \
    --boundary-weight 1.0 \
    --moment-weight 0.2 \
    --rgb-weight 0.1 \
    --save-every 5 \
    --seed 42

echo "Finished: checkpoints/spatial_ae_${SLURM_JOB_ID}/"
echo "Download: checkpoints/spatial_ae_${SLURM_JOB_ID}/spatial_ae_comparisons.zip"
