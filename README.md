# PIWM

## Overview

**Physically Interpretable World Models (PIWM)** is a novel deep learning architecture that learns physically meaningful representations from high-dimensional sensor data (e.g., camera images) using only weak distributional supervision. Unlike traditional world models that function as "black boxes," PIWM aligns its learned latent representations with real-world physical quantities, enabling better interpretability, trustworthiness, and controllability for autonomous systems.


### Key Features

- **Physical Interpretability**: Learned latent representations correspond to meaningful physical states (e.g., position, velocity, orientation)
- **Weak Supervision**: Trains with distributional supervision instead of requiring precise ground-truth physical annotations
- **Flexible Architecture**: Supports both intrinsic and extrinsic encoding approaches with continuous or discrete latent spaces
- **Physics-Informed Dynamics**: Integrates partially known physical equations as structural priors for prediction
- **Superior Performance**: Achieves state-of-the-art prediction accuracy while maintaining physical grounding

## Architecture

PIWM consists of two core components:

1. **Physically Interpretable Autoencoder**: Maps high-dimensional observations to low-dimensional physically meaningful latent states
   - **Extrinsic Approach** (recommended): Two-stage process with vision autoencoder + physical encoder
   - **Intrinsic Approach**: Single end-to-end encoder for direct physical state extraction

2. **Learnable Dynamics Model**: Predicts temporal evolution using known physics equations with learnable parameters
   - Supports partially known dynamics (e.g., bicycle model, cart-pole dynamics)
   - Learns unknown physical parameters (e.g., mass, friction, wheelbase)



## Project Structure

```
PIWM-main/
├── vq/                    # VQ-VAE implementation (discrete latent space)
│   ├── vq.py              # Vector-Quantized VAE model
│   └── extractor.py       # State extraction from VQ-VAE latents
├── ex-conti/              # Extrinsic approach with continuous latent space
│   ├── train_vae.py       # VAE training
│   ├── vae_inference.py   # VAE inference utilities
│   ├── extractor.py       # Physical state extractor
│   └── static-test.py     # Static encoding evaluation
├── in-conti/              # Intrinsic approach with continuous latent space
│   └── train.py           # Intrinsic VAE training
├── dynamic/               # Dynamics models
│   └── BicycleDynamics.py # Bicycle dynamics for Donkey Car
├── PIWM_ICLR2026-8.pdf    # Research paper
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory for training

### Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/PIWM.git
cd PIWM
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support with CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Extrinsic Approach 

The extrinsic approach achieves the best performance by decoupling visual perception from physical state inference.

#### Step 1: Train Vision VAE
```bash
cd ex-conti
python train_vae.py --data_path /path/to/your/data.npz \
                    --output_dir ./vae_output \
                    --latent_dim 128 \
                    --batch_size 32 \
                    --epochs 50
```

#### Step 2: Extract Physical States
```bash
python extractor.py --data_path /path/to/your/data.npz \
                   --vae_checkpoint ./vae_output/best_model.pth \
                   --output_dir ./extractor_output \
                   --d_model 128 \
                   --nhead 4 \
                   --num_layers 3
```

### VQ-VAE Approach (Discrete Latent Space)

For discrete latent representations with quantization:

```bash
cd vq
python vq.py --data_path /path/to/your/data.npz \
             --output_dir ./vqvae_output \
             --latent_dim 256 \
             --num_embeddings 512 \
             --commitment_cost 0.25 \
             --batch_size 32 \
             --epochs 100
```

Then extract states:
```bash
python extractor.py --data_path /path/to/your/data.npz \
                   --vae_checkpoint ./vqvae_output/best_model.pth \
                   --output_dir ./vq_extractor_output
```

### Intrinsic Approach

For end-to-end learning with single encoder:

```bash
cd in-conti
python train.py --data_path /path/to/your/data.npz \
                --output_dir ./intrinsic_output \
                --latent_dim 128 \
                --state_weight 1000.0 \
                --batch_size 32 \
                --epochs 50
```

## Data Format

The code expects data in NPZ format with the following structure:

```python
{
    'frame': np.ndarray,  # Shape: (N, H, W, C), Images (uint8, 0-255)
    'state': np.ndarray,  # Shape: (N, state_dim), Physical states (float)
    'action': np.ndarray  # Shape: (N, action_dim), Actions (optional)
}
```

Where:
- `N`: Number of samples
- `H, W, C`: Image height, width, channels (e.g., 224x224x3)
- `state_dim`: Dimension of physical state (e.g., 2 for [x, y], 4 for [x, y, theta, v])

## Experiments

The paper evaluates PIWM on three environments:

1. **CartPole**: Classic control task with 4D state space
2. **Lunar Lander**: Spacecraft landing with 8D state space
3. **Donkey Car**: Autonomous racing with bicycle dynamics

### Key Results

- **Prediction Accuracy**: PIWM achieves lower RMSE than LSTM, Transformer, DVBF, and SindyC baselines
- **Physical Parameter Recovery**: Successfully learns ground-truth physical parameters (mass, length, wheelbase)
- **Encoding Quality**: Extrinsic discrete approach achieves best static encoding RMSE
- **Robustness**: Maintains performance under varying levels of supervision noise (δ = 0%, 5%, 10%)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
anonymous2025can,
title={Can Weak Quantization Make World Models Physically Interpretable?},
author={Anonymous},
booktitle={Submitted to The Fourteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=MSL8gSuCj2},
note={under review}
}
```


## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0

See `requirements.txt` for complete list.

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` parameter
- Reduce `latent_dim` or `d_model`
- Use gradient accumulation

### Poor Reconstruction Quality
- Increase number of training epochs
- Adjust `kl_weight` or `commitment_cost`
- Increase model capacity (`hidden_dims`, `num_layers`)

### Poor Physical State Extraction
- Increase `state_weight` parameter
- Ensure state distributions are properly normalized
- Try extrinsic approach if using intrinsic

## License

This project is released for research purposes. Please see the paper for more details.

