#!/usr/bin/env python3
"""
Train DVBF predictor 4/15: VAE Fold 2, Noise Type: states
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_dvbf_predictor import train_dvbf_predictor

if __name__ == "__main__":
    # Configuration
    config = {'data_dir': 'donkeydata', 'results_dir': 'dvbf_training/results', 'learning_rate': 0.001, 'weight_decay': 1e-05, 'batch_size': 32, 'epochs': 50, 'num_workers': 0, 'action_dim': 2, 'hidden_dim': 256, 'log_interval': 10, 'save_interval': 10}

    # Train DVBF predictor
    print(f"Training DVBF Predictor 4/15: VAE Fold 2, Noise Type: states")

    results = train_dvbf_predictor(
        vae_fold_num=2,
        predictor_config=config,
        noise_type='states'
    )

    print("Training completed!")
    print("Results:", results)
