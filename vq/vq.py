import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256], latent_dim=256):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dims=[256, 128, 64], out_channels=3):
        super().__init__()

        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU()
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                                 kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        return self.decoder(z)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256],
                 latent_dim=256, num_embeddings=512, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, encodings = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity


class DonkeycarDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with np.load(data_path) as data:
            self.frames = data['frame'].copy()
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame)

        if self.transform:
            frame = self.transform(frame)

        return frame


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)

        optimizer.zero_grad()

        x_recon, vq_loss, perplexity = model(batch)
        recon_loss = F.mse_loss(x_recon, batch)
        loss = recon_loss + vq_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'vq': f'{vq_loss.item():.4f}',
            'perp': f'{perplexity.item():.2f}'
        })

    n = len(train_loader)
    return total_loss/n, total_recon_loss/n, total_vq_loss/n, total_perplexity/n


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)

            x_recon, vq_loss, perplexity = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            loss = recon_loss + vq_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

    n = len(val_loader)
    return total_loss/n, total_recon_loss/n, total_vq_loss/n, total_perplexity/n


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading data from {args.data_path}")
    full_dataset = DonkeycarDataset(args.data_path)
    print(f"Total samples: {len(full_dataset)}")

    train_size = int(args.train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = VQVAE(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)

    print(f"\nModel architecture:")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Num embeddings: {args.num_embeddings}")
    print(f"Commitment cost: {args.commitment_cost}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')

    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_recon, train_vq, train_perp = train_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, val_recon, val_vq, val_perp = validate(model, test_loader, device)

        scheduler.step(val_loss)

        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
              f"VQ: {train_vq:.4f}, Perplexity: {train_perp:.2f}")
        print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, "
              f"VQ: {val_vq:.4f}, Perplexity: {val_perp:.2f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"Saved best model with val_loss: {val_loss:.4f}\n")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, output_dir / 'final_model.pth')

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE on Donkeycar data')

    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for models')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256],
                       help='Hidden dimensions')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--num_embeddings', type=int, default=512,
                       help='Number of embeddings in codebook')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                       help='Commitment cost for VQ loss')

    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint interval')

    args = parser.parse_args()
    main(args)
