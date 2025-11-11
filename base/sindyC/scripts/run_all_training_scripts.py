#!/usr/bin/env python3
"""
Run all 15 predictor training scripts
"""

import subprocess
import sys
from pathlib import Path

def main():
    scripts_dir = Path(__file__).parent

    # Find all training scripts
    scripts = sorted(scripts_dir.glob('train_predictor_*.py'))

    print(f"Found {len(scripts)} training scripts")

    for i, script in enumerate(scripts, 1):
        print(f"\n{'='*60}")
        print(f"Running script {i}/{len(scripts)}: {script.name}")
        print(f"{'='*60}")

        try:
            result = subprocess.run([sys.executable, str(script)],
                                  cwd=scripts_dir,
                                  check=True)
            print(f"Completed {script.name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed {script.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print("All predictor training scripts completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
