import torch
import numpy as np
from train_vae import VAE
from train_latent_to_state import MLP


def load_models(vae_checkpoint_path, mlp_checkpoint_path, device='cuda'):
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_latent_dim = vae_checkpoint['latent_dim']
    vae_model = VAE(latent_dim=vae_latent_dim, image_channels=3).to(device)
    vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location=device)
    state_dict = mlp_checkpoint['model_state_dict']
    input_dim = state_dict['network.0.weight'].shape[1]
    last_layer_key = [k for k in state_dict.keys() if k.startswith('network.') and k.endswith('.weight')][-1]
    output_dim = state_dict[last_layer_key].shape[0]
    hidden_dims = []
    for key in sorted(state_dict.keys()):
        if key.startswith('network.') and key.endswith('.weight') and key != last_layer_key:
            hidden_dims.append(state_dict[key].shape[0])
    mlp_model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims).to(device)
    mlp_model.load_state_dict(state_dict)
    mlp_model.eval()
    return vae_model, mlp_model


def compute_rmse(predictions, targets):
    mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    mse_overall = np.mean((predictions - targets) ** 2)
    rmse_overall = np.sqrt(mse_overall)
    return {'rmse_per_dim': rmse_per_dim, 'rmse_overall': rmse_overall}


def predict_states(vae_model, mlp_model, images, device='cuda', batch_size=32):
    vae_model.eval()
    mlp_model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = batch.astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch).to(device)
            mu, logvar = vae_model.encode(batch_tensor)
            pred_state = mlp_model(mu)
            predictions.append(pred_state.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main():
    vae_checkpoint = ''
    mlp_checkpoint = ''
    test_data_path = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_model, mlp_model = load_models(vae_checkpoint, mlp_checkpoint, device)
    data = np.load(test_data_path)
    images = data['frame']
    states = data['state'][:, :2]
    predictions = predict_states(vae_model, mlp_model, images, device, batch_size=32)
    rmse_dict = compute_rmse(predictions, states)
    return rmse_dict


if __name__ == '__main__':
    results = main()
