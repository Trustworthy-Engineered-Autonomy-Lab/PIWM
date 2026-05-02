import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    """
    LSTM that predicts next latent state given current latent state and (optional) action.
    Input: latent [+ action] -> Output: next latent
    """
    def __init__(self, latent_dim=64, action_dim=2, hidden_dim=128, num_layers=2):
        super(LSTMPredictor, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = latent_dim + (action_dim if action_dim > 0 else 0)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, latent_seq, action_seq=None, hidden=None):
        """
        Args:
            latent_seq: (B, T, latent_dim)
            action_seq: (B, T, action_dim) or None if action_dim==0
            hidden: optional (h0, c0)

        Returns:
            next_latent_seq: (B, T, latent_dim)
            hidden: (hT, cT)
        """
        if self.action_dim > 0:
            if action_seq is None:
                raise ValueError("action_seq is required when action_dim > 0")
            input_seq = torch.cat([latent_seq, action_seq], dim=-1)
        else:
            input_seq = latent_seq

        lstm_out, hidden = self.lstm(input_seq, hidden)
        next_latent_seq = self.output_proj(lstm_out)
        return next_latent_seq, hidden

    def predict_step(self, latent, action=None, hidden=None):
        """
        Single-step autoregressive prediction.

        Args:
            latent: (B, latent_dim)
            action: (B, action_dim) or None if action_dim==0
            hidden: (h, c)

        Returns:
            next_latent: (B, latent_dim)
            hidden: updated (h, c)
        """
        latent_seq = latent.unsqueeze(1)

        if self.action_dim > 0:
            if action is None:
                raise ValueError("action is required when action_dim > 0")
            action_seq = action.unsqueeze(1)
        else:
            action_seq = None

        next_latent_seq, hidden = self.forward(latent_seq, action_seq, hidden)
        next_latent = next_latent_seq.squeeze(1)
        return next_latent, hidden

    def forward_last(self, latent_seq, action_seq=None, hidden=None):
        if self.action_dim > 0:
            if action_seq is None:
                raise ValueError("action_seq is required when action_dim > 0")
            input_seq = torch.cat([latent_seq, action_seq], dim=-1)
        else:
            input_seq = latent_seq

        lstm_out, hidden = self.lstm(input_seq, hidden)
        last_h = lstm_out[:, -1, :]
        next_latent_last = self.output_proj(last_h)
        return next_latent_last, hidden

    def forward_multi(self, latent_seq, action_seq=None, future_actions=None, pred_len=5, hidden=None):
        """
        Multi-step autoregressive prediction.

        Args:
            latent_seq: (B, T_ctx, latent_dim)
            action_seq: (B, T_ctx, action_dim) or None if action_dim==0
            future_actions: (B, pred_len, action_dim) or None if action_dim==0
            pred_len: number of future latent steps to predict
            hidden: optional (h0, c0)

        Returns:
            pred_seq: (B, pred_len, latent_dim)
            hidden: final (hT, cT)
        """
        # Warm up on context
        if self.action_dim > 0:
            if action_seq is None:
                raise ValueError("action_seq is required when action_dim > 0")
            if future_actions is None:
                raise ValueError("future_actions is required when action_dim > 0")
            input_seq = torch.cat([latent_seq, action_seq], dim=-1)
        else:
            input_seq = latent_seq

        _, hidden = self.lstm(input_seq, hidden)

        current_latent = latent_seq[:, -1, :]
        preds = []

        for k in range(pred_len):
            if self.action_dim > 0:
                action_k = future_actions[:, k, :]
                step_input = torch.cat([current_latent, action_k], dim=-1).unsqueeze(1)
            else:
                step_input = current_latent.unsqueeze(1)

            lstm_out_step, hidden = self.lstm(step_input, hidden)
            next_latent = self.output_proj(lstm_out_step[:, -1, :])

            preds.append(next_latent)
            current_latent = next_latent

        pred_seq = torch.stack(preds, dim=1)
        return pred_seq, hidden


def predictor_loss(pred_latent, true_latent):
    return F.mse_loss(pred_latent, true_latent)

def calculate_latent_rmse(pred_latent, true_latent):
    mse_per_dim = torch.mean((pred_latent - true_latent) ** 2, dim=0)
    rmse_per_dim = torch.sqrt(mse_per_dim)
    rmse_total = torch.sqrt(torch.mean((pred_latent - true_latent) ** 2))
    return rmse_per_dim, rmse_total