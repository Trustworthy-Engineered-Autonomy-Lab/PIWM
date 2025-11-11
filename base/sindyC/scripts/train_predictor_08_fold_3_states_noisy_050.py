#!/usr/bin/env python3
"""
Train predictor 8/15: VAE Fold 3, Noise Type: states_noisy_050
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_predictor import train_single_predictor

if __name__ == "__main__":
    # Configuration
    config = {'data_dir': 'donkeydata', 'results_dir': 'predictor_training/results', 'learning_rate': 0.001, 'weight_decay': 1e-05, 'batch_size': 32, 'epochs': 100, 'num_workers': 0, 'action_dim': 2, 'hidden_dim': 128, 'num_layers': 2, 'log_interval': 10, 'save_interval': 20}

    # Train predictor
    print(f"Training Predictor 8/15: VAE Fold 3, Noise Type: states_noisy_050")

    results = train_single_predictor(
        vae_fold_num=3,
        predictor_config=config,
        noise_type='states_noisy_050'
    )

    print("Training completed!")
    print("Results:", results)
