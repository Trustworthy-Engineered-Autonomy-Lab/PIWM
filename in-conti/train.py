import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class IntrinsicVAE(nn.Module):
    def __init__(self, latent_dim=128, state_dim=2, image_channels=3):
        super(IntrinsicVAE, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ImageStateDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        data = np.load(npz_path)
        all_images = data['frame']
        all_states = data['state'][:, :2]

        if indices is not None:
            self.images = all_images[indices]
            self.states = all_states[indices]
            print(f"Loaded {len(self.images)} samples (subset) from {npz_path}")
        else:
            self.images = all_images
            self.states = all_states
            print(f"Loaded {len(self.images)} samples (all) from {npz_path}")

        print(f"Image shape: {self.images.shape}, dtype: {self.images.dtype}")
        print(f"State shape: {self.states.shape}")
        print(f"State statistics:")
        print(f"  Dim 0 - mean: {self.states[:, 0].mean():.4f}, std: {self.states[:, 0].std():.4f}, "
              f"min: {self.states[:, 0].min():.4f}, max: {self.states[:, 0].max():.4f}")
        print(f"  Dim 1 - mean: {self.states[:, 1].mean():.4f}, std: {self.states[:, 1].std():.4f}, "
              f"min: {self.states[:, 1].min():.4f}, max: {self.states[:, 1].max():.4f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        state = self.states[idx].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(state)


def intrinsic_vae_loss(recon_x, x, mu, logvar, state, state_weight=1.0, kl_weight=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    state_pred = mu[:, :2]
    state_loss = nn.functional.mse_loss(state_pred, state, reduction='sum')
    total_loss = BCE + kl_weight * KLD + state_weight * state_loss
    return total_loss, BCE, KLD, state_loss


def train_epoch(model, dataloader, optimizer, device, state_weight=1.0, kl_weight=1.0):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0
    total_state_loss = 0

    for batch_img, batch_state in tqdm(dataloader, desc="Training"):
        batch_img = batch_img.to(device)
        batch_state = batch_state.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(batch_img)
        loss, bce, kld, state_loss = intrinsic_vae_loss(
            recon_batch, batch_img, mu, logvar, batch_state, state_weight, kl_weight
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
        total_state_loss += state_loss.item()

    n_samples = len(dataloader.dataset)
    return (total_loss / n_samples, total_bce / n_samples,
            total_kld / n_samples, total_state_loss / n_samples)


def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    model.eval()
    with torch.no_grad():
        batch_img, batch_state = next(iter(dataloader))
        batch_img = batch_img[:num_samples].to(device)
        recon, _, _ = model(batch_img)

        batch_np = batch_img.cpu().numpy().transpose(0, 2, 3, 1)
        recon_np = recon.cpu().numpy().transpose(0, 2, 3, 1)

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        for i in range(num_samples):
            axes[0, i].imshow(batch_np[i])
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(recon_np[i])
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction samples to {save_path}")


def main():
    latent_dim = 128
    state_dim = 2
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    kl_weight = 1.0
    state_weight = 1000.0

    RANDOM_SEED = 42
    train_ratio = 0.8

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    data_path = './traj2.npz'
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading data from {data_path}")
    data = np.load(data_path)
    total_samples = len(data['frame'])
    print(f"Total samples: {total_samples}")

    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    train_size = int(total_samples * train_ratio)
    train_indices = np.sort(all_indices[:train_size])
    test_indices = np.sort(all_indices[train_size:])

    print(f"Train samples: {len(train_indices)} ({train_ratio*100:.0f}%)")
    print(f"Test samples: {len(test_indices)} ({(1-train_ratio)*100:.0f}%)")

    np.savez(os.path.join(save_dir, 'data_split.npz'),
             train_indices=train_indices,
             test_indices=test_indices,
             random_seed=RANDOM_SEED)
    print(f"Saved data split indices to {os.path.join(save_dir, 'data_split.npz')}")

    print("\nCreating dataset...")
    dataset = ImageStateDataset(data_path, indices=train_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = IntrinsicVAE(latent_dim=latent_dim, state_dim=state_dim, image_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent dimension: {latent_dim}")
    print(f"State dimension: {state_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL weight: {kl_weight}")
    print(f"State weight: {state_weight}\n")

    best_loss = float('inf')
    losses = []

    for epoch in range(num_epochs):
        avg_loss, avg_bce, avg_kld, avg_state_loss = train_epoch(
            model, dataloader, optimizer, device, state_weight, kl_weight
        )
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, "
              f"KLD: {avg_kld:.4f}, State Loss: {avg_state_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            checkpoint_path = os.path.join(save_dir, f'intrinsic_vae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': latent_dim,
                'state_dim': state_dim,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            sample_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch+1}.png')
            save_reconstruction_samples(model, dataloader, device, sample_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'intrinsic_vae_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': latent_dim,
                'state_dim': state_dim,
            }, best_model_path)

    final_model_path = os.path.join(save_dir, 'intrinsic_vae_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'latent_dim': latent_dim,
        'state_dim': state_dim,
    }, final_model_path)
    print(f"\nTraining completed! Final model saved to {final_model_path}")




if __name__ == '__main__':
    main()
