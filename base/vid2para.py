#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime
from tqdm import tqdm
import logging
import sys
import argparse


# ----- Logging -----
def setup_logging(log_file):
    """Set up logging to file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ----- Dataset -----
class DonkeyDataset(Dataset):
    def __init__(self, data_dirs, noise_type="states", seq_len=10):
        self.noise_type = noise_type
        self.seq_len = seq_len
        self.sequences = []

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        for data_dir in data_dirs:
            files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

            for file in files:
                data = np.load(os.path.join(data_dir, file))
                images = data["imgs"]
                actions = data["acts"]

                # Choose state noise variant
                if noise_type == "states":
                    states = data["states"]
                elif noise_type == "noisy_states_2":
                    states = data["noisy_states_2"]
                elif noise_type == "noisy_states_5":
                    states = data["noisy_states_5"]
                elif noise_type == "noisy_states_10":
                    states = data["noisy_states_10"]
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")

                # Build sliding sequences of length seq_len
                num_sequences = len(images) - seq_len + 1
                for i in range(num_sequences):
                    img_seq = images[i : i + seq_len]
                    state_seq = states[i : i + seq_len]
                    action_seq = actions[i : i + seq_len]

                    self.sequences.append({
                        "images": img_seq,
                        "states": state_seq,
                        "actions": action_seq,
                    })

        print(
            f"Loaded data: {len(self.sequences)} sequences, "
            f"noise_type: {noise_type}, seq_len: {seq_len}"
        )
        if len(self.sequences) > 0:
            print(f"Image shape: {self.sequences[0]['images'].shape}")
            print(f"State shape: {self.sequences[0]['states'].shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Images: (T, H, W, C) -> (T, C, H, W)
        images = seq["images"].astype(np.float32) / 255.0
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        states = torch.from_numpy(seq["states"].astype(np.float32))
        actions = torch.from_numpy(seq["actions"].astype(np.float32))

        # Next-state sequence (if needed later)
        next_states = torch.cat([states[1:], states[-1:]], dim=0)

        return images, states, next_states, actions


# ----- VRNN Modules -----
class Encoder(nn.Module):
    """VRNN encoder: images + states -> latent distribution."""

    def __init__(self, image_channels=3, state_dim=4, hidden_dim=256, z_dim=128):
        super().__init__()
        # For Donkey images (120x160)
        self.conv = nn.Sequential(
            nn.Conv2d(
                image_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # 120x160 -> 60x80
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 60x80 -> 30x40
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 30x40 -> 15x20
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 15x20 -> 7x10
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_output_dim = 256 * 7 * 10

        self.fc_img = nn.Linear(conv_output_dim, hidden_dim)
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, image, state):
        img_features = self.fc_img(self.conv(image))
        state_features = self.fc_state(state)
        merged_features = self.fc_merge(
            torch.cat([img_features, state_features], dim=1)
        )
        mu = self.fc_mu(merged_features)
        logvar = self.fc_logvar(merged_features)
        return mu, logvar


class Decoder(nn.Module):
    """VRNN decoder: latent + RNN hidden -> reconstructed image."""

    def __init__(self, z_dim=128, hidden_dim=256, image_channels=3):
        super().__init__()
        conv_input_dim = 256 * 7 * 10

        self.fc = nn.Linear(z_dim + hidden_dim, conv_input_dim)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)
            ),  # 7x10 -> 15x20
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 15x20 -> 30x40
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 30x40 -> 60x80
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, image_channels, kernel_size=4, stride=2, padding=1
            ),  # 60x80 -> 120x160
            nn.Sigmoid(),
        )

    def forward(self, z, rnn_h):
        z_combined = torch.cat([z, rnn_h], dim=1)
        x = self.fc(z_combined).view(-1, 256, 7, 10)
        return self.conv_trans(x)


class Prior(nn.Module):
    """Prior network: RNN hidden -> prior over latent."""

    def __init__(self, hidden_dim=256, z_dim=128):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, rnn_h):
        mu = self.fc_mu(rnn_h)
        logvar = self.fc_logvar(rnn_h)
        return mu, logvar


class VRNNCar(nn.Module):
    """Full VRNN model for car dynamics."""

    def __init__(
        self, image_channels=3, state_dim=4, hidden_dim=256, z_dim=128, l_dim=1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.l_dim = l_dim

        self.encoder = Encoder(image_channels, state_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, image_channels)
        self.rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.prior = Prior(hidden_dim, z_dim)

        # Wheelbase / physical parameter prediction
        self.fc_l_mu = nn.Linear(z_dim, l_dim)
        self.fc_l_logvar = nn.Linear(z_dim, l_dim)

        # State predictor (supervised term)
        self.state_predictor = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, s_seq, h_0=None):
        """Process a full sequence."""
        B, T, C, H, W = x_seq.size()

        x_rec_seq, mu_l_seq, logvar_l_seq = [], [], []
        mu_enc_seq, logvar_enc_seq = [], []
        mu_prior_seq, logvar_prior_seq = [], []
        state_pred_seq = []

        h_t = (
            h_0
            if h_0 is not None
            else torch.zeros(B, self.hidden_dim, device=x_seq.device)
        )

        for t in range(T):
            x_t = x_seq[:, t]
            s_t = s_seq[:, t]

            # Prior
            mu_prior_t, logvar_prior_t = self.prior(h_t)

            # Encoder/posterior
            mu_enc_t, logvar_enc_t = self.encoder(x_t, s_t)
            z_t = self.reparameterize(mu_enc_t, logvar_enc_t)

            # Physical parameter prediction
            mu_l_t = self.fc_l_mu(z_t)
            logvar_l_t = self.fc_l_logvar(z_t)

            # State prediction
            state_pred_t = self.state_predictor(z_t)

            # Decode to image
            x_rec_t = self.decoder(z_t, h_t)

            # RNN update
            z_input = z_t.unsqueeze(1)
            _, h_t = self.rnn(z_input, h_t.unsqueeze(0))
            h_t = h_t.squeeze(0)

            # Collect
            x_rec_seq.append(x_rec_t)
            mu_l_seq.append(mu_l_t)
            logvar_l_seq.append(logvar_l_t)
            mu_enc_seq.append(mu_enc_t)
            logvar_enc_seq.append(logvar_enc_t)
            mu_prior_seq.append(mu_prior_t)
            logvar_prior_seq.append(logvar_prior_t)
            state_pred_seq.append(state_pred_t)

        return (
            torch.stack(x_rec_seq, dim=1),
            torch.stack(mu_l_seq, dim=1),
            torch.stack(logvar_l_seq, dim=1),
            torch.stack(mu_enc_seq, dim=1),
            torch.stack(logvar_enc_seq, dim=1),
            torch.stack(mu_prior_seq, dim=1),
            torch.stack(logvar_prior_seq, dim=1),
            torch.stack(state_pred_seq, dim=1),
        )


# ----- Loss -----
def vrnn_loss(
    x_rec,
    x_orig,
    mu_enc,
    logvar_enc,
    mu_prior,
    logvar_prior,
    mu_l,
    logvar_l,
    state_pred,
    state_true,
    alpha=1.0,
    beta=0.01,
    gamma=1.0,
):
    """Total VRNN loss (recon + KL + supervised + physical regularization)."""
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(x_rec, x_orig, reduction="sum")

    # 2. KL divergence (posterior vs prior)
    kl_loss = -0.5 * torch.sum(
        1
        + logvar_enc
        - logvar_prior
        - (mu_enc - mu_prior).pow(2) / torch.exp(logvar_prior)
        - torch.exp(logvar_enc) / torch.exp(logvar_prior)
    )

    # 3. State prediction loss
    supervised_loss = F.mse_loss(state_pred, state_true, reduction="sum")

    # 4. Physical regularization (target wheelbase L ~ 2.5)
    l_target = torch.full_like(mu_l, 2.5)
    physical_loss = F.mse_loss(mu_l, l_target, reduction="sum")

    total_loss = (
        alpha * recon_loss
        + beta * kl_loss
        + gamma * supervised_loss
        + 0.1 * physical_loss
    )

    return total_loss, recon_loss, kl_loss, supervised_loss, physical_loss


# ----- Early stopping -----
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ----- Evaluation -----
def evaluate_model(model, test_loader, device, logger):
    model.eval()

    all_state_preds = []
    all_state_trues = []
    total_recon_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, states, next_states, actions in tqdm(
            test_loader, desc="Evaluating"
        ):
            images, states = images.to(device), states.to(device)

            (
                x_rec,
                mu_l,
                logvar_l,
                mu_enc,
                logvar_enc,
                mu_prior,
                logvar_prior,
                state_pred,
            ) = model(images, states)

            recon_mse = F.mse_loss(x_rec, images, reduction="sum").item()
            total_recon_mse += recon_mse

            all_state_preds.append(state_pred.cpu().numpy())
            all_state_trues.append(states.cpu().numpy())

            num_samples += images.size(0)

    avg_recon_mse = float(total_recon_mse / num_samples)

    all_state_preds = np.concatenate(all_state_preds, axis=0)
    all_state_trues = np.concatenate(all_state_trues, axis=0)

    # Average over time dimension
    state_preds_mean = np.mean(all_state_preds, axis=1)
    state_trues_mean = np.mean(all_state_trues, axis=1)

    state_rmse = float(np.sqrt(mean_squared_error(state_trues_mean, state_preds_mean)))

    dim_rmse = []
    for i in range(state_trues_mean.shape[1]):
        rmse = float(
            np.sqrt(mean_squared_error(state_trues_mean[:, i], state_preds_mean[:, i]))
        )
        dim_rmse.append(rmse)

    return avg_recon_mse, state_rmse, dim_rmse


# ----- Visualization -----
def visualize_reconstruction(model, test_loader, device, save_path, logger):
    model.eval()

    data_iter = iter(test_loader)
    images, states, next_states, actions = next(data_iter)
    images = images.to(device)
    states = states.to(device)

    with torch.no_grad():
        (x_rec, _, _, _, _, _, _, _) = model(images, states)

    n_samples = min(4, images.size(0))

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))

    for i in range(n_samples):
        original = images[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(original)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original (t=0)", fontsize=10)

        reconstructed = x_rec[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(reconstructed)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed (t=0)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Reconstruction visualization saved to: {save_path}")


# ----- Training -----
def train_vrnn(
    fold_idx,
    noise_type,
    epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    seq_len=10,
    output_base_dir=".",
):
    output_dir = os.path.join(
        output_base_dir, f"vrnn_output_fold{fold_idx}_{noise_type}"
    )
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "training.log")
    logger = setup_logging(log_file)

    params = {
        "fold_idx": fold_idx,
        "noise_type": noise_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seq_len": seq_len,
        "hidden_dim": 256,
        "z_dim": 128,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("=" * 60)
    logger.info(f"Starting VRNN training - Fold {fold_idx}, noise_type: {noise_type}")
    logger.info(f"Training parameters: {json.dumps(params, indent=2)}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data folds: 1..5; current fold is val, others are train
    train_dirs = []
    val_dir = f"../donkeynew_fold{fold_idx}"

    for i in range(1, 6):
        if i != fold_idx:
            train_dirs.append(f"../donkeynew_fold{i}")

    logger.info(f"Validation dir: {val_dir}")
    logger.info(f"Training dirs: {train_dirs}")

    # Load data
    try:
        train_dataset = DonkeyDataset(train_dirs, noise_type, seq_len)
        val_dataset = DonkeyDataset(val_dir, noise_type, seq_len)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VRNNCar(
        image_channels=3, state_dim=4, hidden_dim=256, z_dim=128, l_dim=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)

    train_losses = []
    val_losses = []
    train_details = []
    val_details = []

    logger.info("Training loop started...")
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_supervised_loss = 0.0
        train_physical_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_idx, (images, states, next_states, actions) in enumerate(train_bar):
            images, states = images.to(device), states.to(device)

            optimizer.zero_grad()

            (
                x_rec,
                mu_l,
                logvar_l,
                mu_enc,
                logvar_enc,
                mu_prior,
                logvar_prior,
                state_pred,
            ) = model(images, states)

            (
                loss,
                recon_loss,
                kl_loss,
                supervised_loss,
                physical_loss,
            ) = vrnn_loss(
                x_rec,
                images,
                mu_enc,
                logvar_enc,
                mu_prior,
                logvar_prior,
                mu_l,
                logvar_l,
                state_pred,
                states,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_supervised_loss += supervised_loss.item()
            train_physical_loss += physical_loss.item()

            train_bar.set_postfix({
                "loss": loss.item() / (images.size(0) * images.size(1)),
                "recon": recon_loss.item() / (images.size(0) * images.size(1)),
            })

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_supervised_loss = 0.0
        val_physical_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for images, states, next_states, actions in val_bar:
                images, states = images.to(device), states.to(device)

                (
                    x_rec,
                    mu_l,
                    logvar_l,
                    mu_enc,
                    logvar_enc,
                    mu_prior,
                    logvar_prior,
                    state_pred,
                ) = model(images, states)

                (
                    loss,
                    recon_loss,
                    kl_loss,
                    supervised_loss,
                    physical_loss,
                ) = vrnn_loss(
                    x_rec,
                    images,
                    mu_enc,
                    logvar_enc,
                    mu_prior,
                    logvar_prior,
                    mu_l,
                    logvar_l,
                    state_pred,
                    states,
                )

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_supervised_loss += supervised_loss.item()
                val_physical_loss += physical_loss.item()

                val_bar.set_postfix({
                    "loss": loss.item() / (images.size(0) * images.size(1)),
                    "recon": recon_loss.item() / (images.size(0) * images.size(1)),
                })

        # Aggregate losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        train_details.append({
            "epoch": epoch + 1,
            "total_loss": avg_train_loss,
            "recon_loss": train_recon_loss / len(train_dataset),
            "kl_loss": train_kl_loss / len(train_dataset),
            "supervised_loss": train_supervised_loss / len(train_dataset),
            "physical_loss": train_physical_loss / len(train_dataset),
        })

        val_details.append({
            "epoch": epoch + 1,
            "total_loss": avg_val_loss,
            "recon_loss": val_recon_loss / len(val_dataset),
            "kl_loss": val_kl_loss / len(val_dataset),
            "supervised_loss": val_supervised_loss / len(val_dataset),
            "physical_loss": val_physical_loss / len(val_dataset),
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}"
            )
            logger.info(
                "  Train - Recon: {:.4f}, KL: {:.4f}, Supervised: {:.4f}, Physical: {:.4f}".format(
                    train_recon_loss / len(train_dataset),
                    train_kl_loss / len(train_dataset),
                    train_supervised_loss / len(train_dataset),
                    train_physical_loss / len(train_dataset),
                )
            )

        if early_stopping(avg_val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (epoch {best_epoch})")

    logger.info("=" * 60)
    logger.info("Training finished. Starting evaluation...")
    logger.info("=" * 60)

    avg_recon_mse, state_rmse, dim_rmse = evaluate_model(
        model, val_loader, device, logger
    )

    logger.info("Validation evaluation:")
    logger.info(f"  Reconstruction MSE: {avg_recon_mse:.6f}")
    logger.info(f"  State RMSE (overall): {state_rmse:.6f}")
    logger.info("  State RMSE per dimension:")
    for i, rmse in enumerate(dim_rmse):
        logger.info(f"    dim {i}: {rmse:.6f}")

    vis_path = os.path.join(output_dir, "reconstruction.png")
    visualize_reconstruction(model, val_loader, device, vis_path, logger)

    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")

    # Convert numpy types to native Python types
    def convert_to_python_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        else:
            return obj

    results = {
        "parameters": params,
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "train_details": convert_to_python_types(train_details),
        "val_details": convert_to_python_types(val_details),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_epoch": int(len(train_losses)),
        "evaluation": {
            "recon_mse": avg_recon_mse,
            "state_rmse": state_rmse,
            "dim_rmse": dim_rmse,
        },
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Parameters saved to: {params_path}")

    logger.info("=" * 60)
    logger.info(f"VRNN training complete for fold {fold_idx}, noise_type {noise_type}")
    logger.info("=" * 60)

    return model, results


# ----- CLI -----
def parse_args():
    parser = argparse.ArgumentParser(description="Train VRNN model on Donkey dataset")

    parser.add_argument(
        "--fold_idx",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Fold index (1–5).",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="states",
        choices=["states", "noisy_states_5", "noisy_states_10"],
        help="State noise type.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Sequence length for training.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=".",
        help="Base directory for outputs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print(
        f"Starting VRNN training - Fold {args.fold_idx}, noise_type: {args.noise_type}"
    )
    print("=" * 60)

    model, results = train_vrnn(
        fold_idx=args.fold_idx,
        noise_type=args.noise_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
        output_base_dir=args.output_base_dir,
    )

    out_dir = os.path.join(
        args.output_base_dir, f"vrnn_output_fold{args.fold_idx}_{args.noise_type}"
    )
    print("=" * 60)
    print(
        f"VRNN training complete for fold {args.fold_idx}, noise_type {args.noise_type}"
    )
    print(f"Outputs saved in: {out_dir}/")
    print("=" * 60)

