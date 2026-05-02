import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=64, input_channels=3):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 120 x 160
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 x 60 x 80
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 30 x 40
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 15 x 20
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 10
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 4 x 5
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the actual flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 120, 160)
            dummy_output = self.encoder(dummy_input)
            self.flattened_size = dummy_output.shape[1]
            print(f"Calculated flattened size: {self.flattened_size}")

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        # Calculate the feature map dimensions for reshaping
        # Working backwards from flattened_size = 512 * h * w
        self.decoder_h = self.flattened_size // 512 // 5  # Assuming width is 5
        self.decoder_w = 5
        if self.decoder_h * self.decoder_w * 512 != self.flattened_size:
            # If not exact, calculate properly
            total_spatial = self.flattened_size // 512
            # Find factors that are close to the expected ratio
            for h in range(1, 20):
                if total_spatial % h == 0:
                    w = total_spatial // h
                    if abs(h/w - 4/5) < 0.1:  # Close to 4:5 ratio
                        self.decoder_h, self.decoder_w = h, w
                        break
            else:
                # Fallback: use square-ish dimensions
                self.decoder_h = int((total_spatial) ** 0.5)
                self.decoder_w = total_spatial // self.decoder_h

        print(f"Decoder reshape: 512 x {self.decoder_h} x {self.decoder_w}")

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, self.decoder_h, self.decoder_w)
        h = self.decoder(h)

        # Adaptive cropping/padding to get exact input size (120, 160)
        current_h, current_w = h.shape[-2:]

        # Crop or pad height to 120
        if current_h > 120:
            start_h = (current_h - 120) // 2
            h = h[:, :, start_h:start_h+120, :]
        elif current_h < 120:
            pad_h = 120 - current_h
            h = F.pad(h, (0, 0, pad_h//2, pad_h - pad_h//2))

        # Crop or pad width to 160
        current_w = h.shape[-1]
        if current_w > 160:
            start_w = (current_w - 160) // 2
            h = h[:, :, :, start_w:start_w+160]
        elif current_w < 160:
            pad_w = 160 - current_w
            h = F.pad(h, (pad_w//2, pad_w - pad_w//2, 0, 0))

        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss