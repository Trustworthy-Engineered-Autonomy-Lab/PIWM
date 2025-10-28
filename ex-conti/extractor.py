import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train_vae import VAE


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LatentStateDataset(Dataset):
    def __init__(self, npz_path, vae_model, device='cuda'):
        data = np.load(npz_path)
        self.images = data['frame']
        self.states = data['state'][:, :2]
        self.latent_codes = self._encode_all_images(vae_model, device)

    def _encode_all_images(self, vae_model, device, batch_size=32):
        vae_model.eval()
        latent_codes = []
        with torch.no_grad():
            for i in range(0, len(self.images), batch_size):
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for latent, state in dataloader:
        latent = latent.to(device)
        state = state.to(device)
        optimizer.zero_grad()
        pred = model(latent)
        loss = criterion(pred, state)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(latent)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for latent, state in dataloader:
            latent = latent.to(device)
            state = state.to(device)
            pred = model(latent)
            loss = criterion(pred, state)
            total_loss += loss.item() * len(latent)
    return total_loss / len(dataloader.dataset)


def main():
    latent_dim = 128
    state_dim = 2
    hidden_dims = [256, 128, 64]
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    train_split = 0.8

    vae_checkpoint = 'E:/25fall/piwm-iclr/realDonkey/ex_upload/vae_best.pt'
    data_path = 'E:/25fall/piwm-iclr/realDonkey/traj1.npz'
    save_path = 'E:/25fall/piwm-iclr/realDonkey/ex_upload/mlp_best.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_checkpoint_data = torch.load(vae_checkpoint, map_location=device)
    vae_model = VAE(latent_dim=latent_dim, image_channels=3).to(device)
    vae_model.load_state_dict(vae_checkpoint_data['model_state_dict'])
    vae_model.eval()

    for param in vae_model.parameters():
        param.requires_grad = False

    full_dataset = LatentStateDataset(data_path, vae_model, device)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mlp_model = MLP(input_dim=latent_dim, output_dim=state_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(mlp_model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(mlp_model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mlp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)


if __name__ == '__main__':
    main()
