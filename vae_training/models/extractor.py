import torch
import torch.nn as nn
import torch.nn.functional as F

class StateExtractor(nn.Module):
    """
    MLP that transforms VAE latent embeddings to state predictions
    """
    def __init__(self, latent_dim=64, output_dim=3, hidden_dims=[128, 64]):
        super(StateExtractor, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build MLP layers
        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latent):
        """
        Args:
            latent: (batch_size, latent_dim) - VAE latent embeddings

        Returns:
            states: (batch_size, output_dim) - Predicted states
        """
        return self.mlp(latent)

def extractor_loss(pred_states, true_states):
    """
    MSE loss for state prediction
    """
    return F.mse_loss(pred_states, true_states)

def calculate_rmse(pred_states, true_states):
    """
    Calculate RMSE for each dimension

    Args:
        pred_states: (N, 3) predicted states
        true_states: (N, 3) true states

    Returns:
        rmse_per_dim: (3,) RMSE for each dimension
        rmse_total: scalar overall RMSE
    """
    mse_per_dim = torch.mean((pred_states - true_states) ** 2, dim=0)
    rmse_per_dim = torch.sqrt(mse_per_dim)
    rmse_total = torch.sqrt(torch.mean((pred_states - true_states) ** 2))

    return rmse_per_dim, rmse_total