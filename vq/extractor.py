import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import math


class VAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 14 * 14)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 14, 14)
        return self.decoder(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ImprovedStateTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, dropout=0.1, output_dim=2):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.d_model = d_model

    def forward(self, latent):
        x = latent.unsqueeze(1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out


class LatentStateDataset(Dataset):
    def __init__(self, npz_path, vae_model, device='cuda', indices=None):
        data = np.load(npz_path)
        all_images = data['frame']
        all_states = data['state'][:, :2]

        if indices is not None:
            self.images = all_images[indices]
            self.states = all_states[indices]
        else:
            self.images = all_images
            self.states = all_states

        print(f"Loaded {len(self.images)} samples")
        print(f"Images shape: {self.images.shape}")
        print(f"States shape: {self.states.shape}")
        print(f"State statistics:")
        print(f"  Dim 0 - mean: {self.states[:, 0].mean():.4f}, std: {self.states[:, 0].std():.4f}")
        print(f"  Dim 1 - mean: {self.states[:, 1].mean():.4f}, std: {self.states[:, 1].std():.4f}")

        print("\nEncoding images to continuous latent codes...")
        self.latent_codes = self._encode_all_images(vae_model, device)
        print(f"Latent codes shape: {self.latent_codes.shape}")
        print(f"Latent codes - mean: {self.latent_codes.mean():.4f}, std: {self.latent_codes.std():.4f}")

    def _encode_all_images(self, vae_model, device, batch_size=32):
        vae_model.eval()
        latent_codes = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.images), batch_size), desc="Encoding"):
                batch = self.images[i:i+batch_size]
                batch = batch.astype(np.float32) / 255.0
                batch_tensor = torch.from_numpy(batch).to(device)

                mu, logvar = vae_model.encode(batch_tensor)
                latent_codes.append(mu.cpu().numpy())

        return np.concatenate(latent_codes, axis=0)

    def __len__(self):
        return len(self.latent_codes)

    def __getitem__(self, idx):
        latent = torch.from_numpy(self.latent_codes[idx]).float()
        state = torch.from_numpy(self.states[idx]).float()
        return latent, state


def compute_rmse(predictions, targets):
    mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    mse_overall = np.mean((predictions - targets) ** 2)
    rmse_overall = np.sqrt(mse_overall)
    return rmse_per_dim, rmse_overall


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training')
    for latent, states in pbar:
        latent = latent.to(device)
        states = states.to(device)

        optimizer.zero_grad()

        predictions = model(latent)
        loss = criterion(predictions, states)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for latent, states in tqdm(val_loader, desc='Validation'):
            latent = latent.to(device)
            states = states.to(device)

            predictions = model(latent)

            loss = criterion(predictions, states)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(states.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    rmse_per_dim, total_rmse = compute_rmse(all_predictions, all_targets)

    return total_loss / len(val_loader), rmse_per_dim, total_rmse, all_predictions, all_targets


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}")
    data = np.load(args.data_path)
    total_samples = len(data['frame'])
    print(f"Total samples: {total_samples}")

    train_size = int(args.train_ratio * total_samples)
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, total_samples)

    print(f"Train samples: {len(train_indices)} (0 to {train_size-1})")
    print(f"Test samples: {len(test_indices)} ({train_size} to {total_samples-1})\n")

    print(f"Loading VAE model from {args.vae_checkpoint}")
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae_latent_dim = vae_checkpoint['latent_dim']
    vae_model = VAE(latent_dim=vae_latent_dim, image_channels=3).to(device)
    vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
    vae_model.eval()

    for param in vae_model.parameters():
        param.requires_grad = False

    print(f"VAE loaded with latent_dim={vae_latent_dim}\n")

    print("Creating train dataset...")
    train_dataset = LatentStateDataset(args.data_path, vae_model, device, indices=train_indices)

    print("\nCreating test dataset...")
    test_dataset = LatentStateDataset(args.data_path, vae_model, device, indices=test_indices)

    train_size_mlp = int(0.8 * len(train_dataset))
    val_size_mlp = len(train_dataset) - train_size_mlp
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size_mlp, val_size_mlp],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = ImprovedStateTransformer(
        input_dim=vae_latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dim=2
    ).to(device)

    print("Improved Transformer Model:")
    print(f"  Input dim: {vae_latent_dim}")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dim_feedforward: {args.dim_feedforward}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_rmse = float('inf')

    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, rmse_per_dim, total_rmse, predictions, targets = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"RMSE - Dim 0: {rmse_per_dim[0]:.6f}, Dim 1: {rmse_per_dim[1]:.6f}, Total: {total_rmse:.6f}")
        print(f"Learning rate: {current_lr:.6e}\n")

        if total_rmse < best_rmse:
            best_rmse = total_rmse
            best_rmse_per_dim = rmse_per_dim
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse_per_dim': rmse_per_dim,
                'total_rmse': total_rmse,
                'args': vars(args)
            }, output_dir / 'best_transformer_improved.pth')
            print(f"Saved best model with RMSE: {total_rmse:.6f}\n")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse_per_dim': rmse_per_dim,
                'total_rmse': total_rmse,
            }, output_dir / f'transformer_improved_epoch_{epoch+1}.pth')

    print("=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    best_checkpoint = torch.load(output_dir / 'best_transformer_improved.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_rmse_per_dim, test_total_rmse, test_predictions, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results (Best Model from Epoch {best_checkpoint['epoch']}):")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  RMSE Position X: {test_rmse_per_dim[0]:.6f}")
    print(f"  RMSE Position Y: {test_rmse_per_dim[1]:.6f}")
    print(f"  Total RMSE: {test_total_rmse:.6f}")

    print(f"\nValidation Results:")
    print(f"  Best Val RMSE: {best_rmse:.6f}")
    print(f"  Best Val RMSE Dimension 0: {best_rmse_per_dim[0]:.6f}")
    print(f"  Best Val RMSE Dimension 1: {best_rmse_per_dim[1]:.6f}")

    corr_0 = np.corrcoef(test_targets[:, 0], test_predictions[:, 0])[0, 1]
    corr_1 = np.corrcoef(test_targets[:, 1], test_predictions[:, 1])[0, 1]

    ss_res_0 = np.sum((test_targets[:, 0] - test_predictions[:, 0]) ** 2)
    ss_tot_0 = np.sum((test_targets[:, 0] - test_targets[:, 0].mean()) ** 2)
    r2_0 = 1 - (ss_res_0 / ss_tot_0)

    ss_res_1 = np.sum((test_targets[:, 1] - test_predictions[:, 1]) ** 2)
    ss_tot_1 = np.sum((test_targets[:, 1] - test_targets[:, 1].mean()) ** 2)
    r2_1 = 1 - (ss_res_1 / ss_tot_1)

    print(f"\nCorrelation:")
    print(f"  Position X: {corr_0:.6f}")
    print(f"  Position Y: {corr_1:.6f}")
    print(f"\nR^2 Score:")
    print(f"  Position X: {r2_0:.6f}")
    print(f"  Position Y: {r2_1:.6f}")

    test_results_path = output_dir / 'test_results_improved.txt'
    with open(test_results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Improved Transformer Test Results\n")
        f.write("Using VAE Continuous Latent\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Improved Transformer\n")
        f.write(f"Input: VAE latent ({vae_latent_dim}D continuous)\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Test RMSE:\n")
        f.write(f"  Position X: {test_rmse_per_dim[0]:.6f}\n")
        f.write(f"  Position Y: {test_rmse_per_dim[1]:.6f}\n")
        f.write(f"  Total: {test_total_rmse:.6f}\n\n")
        f.write(f"Correlation:\n")
        f.write(f"  Position X: {corr_0:.6f}\n")
        f.write(f"  Position Y: {corr_1:.6f}\n\n")
        f.write(f"R^2 Score:\n")
        f.write(f"  Position X: {r2_0:.6f}\n")
        f.write(f"  Position Y: {r2_1:.6f}\n")

    print(f"\nSaved test results to {test_results_path}")
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Improved Transformer with VAE latent')

    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                       help='Path to VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='./transformer_improved_output',
                       help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    parser.add_argument('--d_model', type=int, default=128,
                       help='Transformer dimension')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint interval')

    args = parser.parse_args()
    main(args)
