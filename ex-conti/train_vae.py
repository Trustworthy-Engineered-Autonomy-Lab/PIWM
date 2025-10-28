import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
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


class ImageDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['frame']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        return torch.from_numpy(image)


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kl_weight * KLD


def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar, kl_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)


def main():
    latent_dim = 128
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    kl_weight = 1.0
    data_path = 'E:/25fall/piwm-iclr/realDonkey/traj1.npz'
    save_path = 'E:/25fall/piwm-iclr/realDonkey/ex_upload/vae_best.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = VAE(latent_dim=latent_dim, image_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, kl_weight)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': latent_dim,
            }, save_path)


if __name__ == '__main__':
    main()
