#!/usr/bin/env python3
"""
Unified DVBF training CLI.

This replaces:
  train_dvbf_01_fold_1_states.py
  ...
  train_dvbf_15_fold_5_states_noisy_100.py

You can:
  - Use --exp_id 1..15 to reproduce the old scripts exactly (fold + noise combo).
  - Or specify --vae_fold_num / --noise_type directly.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import train_dvbf_predictor
sys.path.append(str(Path(__file__).parent.parent))

from train_dvbf_predictor import train_dvbf_predictor


# Mapping from "01..15" experiment index → (vae_fold_num, noise_type)
EXPERIMENT_CONFIGS = {
    1: {"vae_fold_num": 1, "noise_type": "states"},
    2: {"vae_fold_num": 1, "noise_type": "states_noisy_050"},
    3: {"vae_fold_num": 1, "noise_type": "states_noisy_100"},
    4: {"vae_fold_num": 2, "noise_type": "states"},
    5: {"vae_fold_num": 2, "noise_type": "states_noisy_050"},
    6: {"vae_fold_num": 2, "noise_type": "states_noisy_100"},
    7: {"vae_fold_num": 3, "noise_type": "states"},
    8: {"vae_fold_num": 3, "noise_type": "states_noisy_050"},
    9: {"vae_fold_num": 3, "noise_type": "states_noisy_100"},
    10: {"vae_fold_num": 4, "noise_type": "states"},
    11: {"vae_fold_num": 4, "noise_type": "states_noisy_050"},
    12: {"vae_fold_num": 4, "noise_type": "states_noisy_100"},
    13: {"vae_fold_num": 5, "noise_type": "states"},
    14: {"vae_fold_num": 5, "noise_type": "states_noisy_050"},
    15: {"vae_fold_num": 5, "noise_type": "states_noisy_100"},
    # If train_dvbf_predictor expects slightly different noise_type strings,
    # just edit the values above accordingly.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DVBF predictor (unified CLI for all 15 experiments)."
    )

    # High-level experiment selector
    parser.add_argument(
        "--exp_id",
        type=int,
        choices=sorted(EXPERIMENT_CONFIGS.keys()),
        help=(
            "Which of the 15 preset experiments to run (1..15). "
            "If provided, this sets vae_fold_num and noise_type "
            "to match the old train_dvbf_XX scripts."
        ),
    )

    # Direct overrides / manual configuration
    parser.add_argument(
        "--vae_fold_num",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="VAE fold number (overrides value from --exp_id if given).",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        help=(
            "Noise type string passed to train_dvbf_predictor "
            "(e.g. 'states', 'states_noisy_050', 'states_noisy_100'). "
            "Overrides value from --exp_id if given."
        ),
    )

    # Predictor config (same defaults as your original small scripts)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="donkeydata",
        help="Base directory containing training data (default: donkeydata).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="dvbf_training/results",
        help="Directory to store results/checkpoints (default: dvbf_training/results).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=2,
        help="Action dimension (default: 2).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for DVBF predictor (default: 256).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between logging during training (default: 10).",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Epochs between checkpoint saves (default: 10).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine vae_fold_num and noise_type from exp_id presets, then apply overrides
    vae_fold_num = None
    noise_type = None

    if args.exp_id is not None:
        preset = EXPERIMENT_CONFIGS[args.exp_id]
        vae_fold_num = preset["vae_fold_num"]
        noise_type = preset["noise_type"]

    # Explicit CLI flags override the preset
    if args.vae_fold_num is not None:
        vae_fold_num = args.vae_fold_num
    if args.noise_type is not None:
        noise_type = args.noise_type

    # Minimal sanity check
    if vae_fold_num is None:
        raise SystemExit(
            "You must specify either --exp_id or --vae_fold_num "
            "(and optionally --noise_type)."
        )
    if noise_type is None:
        raise SystemExit(
            "noise_type is not set. Use --exp_id or explicitly pass --noise_type."
        )

    # Build predictor_config dict like your original script
    predictor_config = {
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
    }

    # Try to reconstruct a nice "k/15" label if exp_id used
    exp_label = ""
    if args.exp_id is not None:
        exp_label = f"{args.exp_id}/15: "

    print(
        f"Training DVBF Predictor {exp_label}"
        f"VAE Fold {vae_fold_num}, Noise Type: {noise_type}"
    )

    results = train_dvbf_predictor(
        vae_fold_num=vae_fold_num,
        predictor_config=predictor_config,
        noise_type=noise_type,
    )

    print("Training completed!")
    print("Results:", results)


if __name__ == "__main__":
    main()
