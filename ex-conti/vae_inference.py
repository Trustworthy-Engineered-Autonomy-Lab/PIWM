import torch
import numpy as np
from train_vae import VAE


def load_vae_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    latent_dim = checkpoint['latent_dim']
    model = VAE(latent_dim=latent_dim, image_channels=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, latent_dim


def encode_images(model, images, device='cuda'):
    model.eval()
    with torch.no_grad():
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        images_tensor = torch.from_numpy(images).to(device)
        mu, logvar = model.encode(images_tensor)
        latent_codes = mu.cpu().numpy()
    return latent_codes


def decode_latents(model, latent_codes, device='cuda'):
    model.eval()
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latent_codes).float().to(device)
        recon = model.decode(latent_tensor)
        images = (recon.cpu().numpy() * 255).astype(np.uint8)
    return images


def reconstruct_images(model, images, device='cuda'):
    latent_codes = encode_images(model, images, device)
    reconstructed = decode_latents(model, latent_codes, device)
    return reconstructed


def sample_from_latent_space(model, num_samples=8, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated = model.decode(z)
        images = (generated.cpu().numpy() * 255).astype(np.uint8)
    return images


def interpolate_between_images(model, img1, img2, num_steps=10, device='cuda'):
    latent1 = encode_images(model, img1[np.newaxis, :], device)[0]
    latent2 = encode_images(model, img2[np.newaxis, :], device)[0]
    alphas = np.linspace(0, 1, num_steps)
    latent_interpolated = np.array([(1 - alpha) * latent1 + alpha * latent2 for alpha in alphas])
    interpolated = decode_latents(model, latent_interpolated, device)
    return interpolated
