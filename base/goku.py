import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
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

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class DonkeyDataset(Dataset):
    def __init__(self, data_dirs, noise_type='states', seq_len=10):
        self.noise_type = noise_type
        self.seq_len = seq_len
        self.sequences = []
        
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        for data_dir in data_dirs:
            files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
            
            for file in files:
                data = np.load(os.path.join(data_dir, file))
                images = data['imgs']
                actions = data['acts']
                
                if noise_type == 'states':
                    states = data['states']
                elif noise_type == 'noisy_states_2':
                    states = data['noisy_states_2']
                elif noise_type == 'noisy_states_5':
                    states = data['noisy_states_5']
                elif noise_type == 'noisy_states_10':
                    states = data['noisy_states_10']
                
                num_sequences = len(images) - seq_len + 1
                for i in range(num_sequences):
                    img_seq = images[i:i+seq_len]
                    state_seq = states[i:i+seq_len]
                    action_seq = actions[i:i+seq_len]
                    
                    self.sequences.append({
                        'images': img_seq,
                        'states': state_seq,
                        'actions': action_seq
                    })
        
        print(f"Loaded data: {len(self.sequences)} sequences, noise type: {noise_type}, sequence length: {seq_len}")
        if len(self.sequences) > 0:
            print(f"Image shape: {self.sequences[0]['images'].shape}")
            print(f"State shape: {self.sequences[0]['states'].shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        images = seq['images'].astype(np.float32) / 255.0
        if images.shape[-1] == 3:
            images = np.mean(images, axis=-1, keepdims=True)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        states = torch.from_numpy(seq['states'].astype(np.float32))
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        return images, states, actions

class DonkeyTestDataset(Dataset):
    def __init__(self, data_dir, noise_type='states', test_seq_len=30):
        self.noise_type = noise_type
        self.test_seq_len = test_seq_len
        self.test_sequences = []
        self.file_names = []
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        for file in files:
            data = np.load(os.path.join(data_dir, file))
            images = data['imgs']
            actions = data['acts']
            
            if noise_type == 'states':
                states = data['states']
            elif noise_type == 'noisy_states_2':
                states = data['noisy_states_2']
            elif noise_type == 'noisy_states_5':
                states = data['noisy_states_5']
            elif noise_type == 'noisy_states_10':
                states = data['noisy_states_10']
            
            num_test_sequences = len(images) - test_seq_len + 1
            for i in range(num_test_sequences):
                img_seq = images[i:i+test_seq_len]
                state_seq = states[i:i+test_seq_len]
                action_seq = actions[i:i+test_seq_len]
                
                self.test_sequences.append({
                    'images': img_seq,
                    'states': state_seq,
                    'actions': action_seq
                })
                self.file_names.append(f"{file}_seq{i}")
        
        print(f"Loaded test data: {len(self.test_sequences)} sequences of {test_seq_len} steps")
    
    def __len__(self):
        return len(self.test_sequences)
    
    def __getitem__(self, idx):
        seq = self.test_sequences[idx]
        
        images = seq['images'].astype(np.float32) / 255.0
        if images.shape[-1] == 3:
            images = np.mean(images, axis=-1, keepdims=True)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        states = torch.from_numpy(seq['states'].astype(np.float32))
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        return images, states, actions, self.file_names[idx]

class BicycleODE(nn.Module):
    def __init__(self):
        super(BicycleODE, self).__init__()

    def forward(self, t, z, theta, get_u_func):
        x, y, psi, v = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        L = theta[:, 0]
        u = get_u_func(t)
        delta, a = u[:, 0], u[:, 1]
        
        dxdt = v * torch.cos(psi)
        dydt = v * torch.sin(psi)
        dpsidt = v / L * torch.tan(delta)
        dvdt = a
        
        dzdt = torch.stack([dxdt, dydt, dpsidt, dvdt], dim=1)
        return dzdt

class EncoderNet(nn.Module):
    def __init__(self, latent_dim, theta_dim, image_size):
        super(EncoderNet, self).__init__()
        
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        
        self.fc_mu_z0 = nn.Linear(128, latent_dim)
        self.fc_logvar_z0 = nn.Linear(128, latent_dim)
        self.fc_mu_theta = nn.Linear(128, theta_dim)
        self.fc_logvar_theta = nn.Linear(128, theta_dim)

    def forward(self, images):
        batch_size, seq_len, c, h, w = images.shape
        images_flat = images.view(batch_size * seq_len, c, h, w)
        
        features = self.conv_extractor(images_flat)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, (h_n, _) = self.lstm(features)
        h_last = h_n[-1]
        
        mu_z0 = self.fc_mu_z0(h_last)
        logvar_z0 = self.fc_logvar_z0(h_last)
        mu_theta = self.fc_mu_theta(h_last)
        logvar_theta = self.fc_logvar_theta(h_last)
        
        return mu_z0, logvar_z0, mu_theta, logvar_theta

class DecoderNet(nn.Module):
    def __init__(self, latent_dim, image_size):
        super(DecoderNet, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 7 * 10)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size, seq_len, latent_dim = z.shape
        z_flat = z.view(batch_size * seq_len, latent_dim)
        
        h = self.fc(z_flat)
        h = h.view(batch_size * seq_len, 256, 7, 10)
        
        x_hat = self.deconv(h)
        
        x_hat = F.interpolate(x_hat, size=(120, 160), mode='bilinear', align_corners=False)
        x_hat = x_hat.view(batch_size, seq_len, 1, 120, 160)
        
        return x_hat

class GOKUNet(nn.Module):
    def __init__(self, latent_dim=4, theta_dim=1, image_size=(120, 160)):
        super(GOKUNet, self).__init__()
        
        self.latent_dim = latent_dim
        self.theta_dim = theta_dim
        self.image_size = image_size
        
        self.encoder = EncoderNet(latent_dim, theta_dim, image_size)
        self.decoder = DecoderNet(latent_dim, image_size)
        self.ode_func = BicycleODE()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, images, t_points, actions):
        batch_size, seq_len, c, h, w = images.shape
        
        mu_z0, logvar_z0, mu_theta, logvar_theta = self.encoder(images)
        
        z0 = self.reparameterize(mu_z0, logvar_z0)
        theta = self.reparameterize(mu_theta, logvar_theta)
        
        def get_u_func(t):
            t_idx = (t * (seq_len - 1)).long().clamp(0, seq_len - 1)
            return actions[:, t_idx]
        
        z_hat = odeint(
            lambda t, z: self.ode_func(t, z, theta, get_u_func),
            z0,
            t_points,
            method='rk4',
            options={'step_size': 0.01}
        )
        
        z_hat = z_hat.permute(1, 0, 2)
        
        x_hat = self.decoder(z_hat)
        
        return x_hat, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat

def gokunet_loss(x_hat, x, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat, states, 
                  recon_weight=1.0, kl_weight=0.1, state_weight=1.0, physics_weight=0.1):
    batch_size, seq_len, c, h, w = x.shape
    
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    kl_z0 = -0.5 * torch.sum(1 + logvar_z0 - mu_z0.pow(2) - logvar_z0.exp())
    kl_theta = -0.5 * torch.sum(1 + logvar_theta - mu_theta.pow(2) - logvar_theta.exp())
    kl_loss = kl_z0 + kl_theta
    
    state_loss = F.mse_loss(z_hat, states, reduction='sum')
    
    physics_loss = torch.tensor(0.0, device=x.device)
    if seq_len > 1:
        z_diff = z_hat[:, 1:] - z_hat[:, :-1]
        physics_loss = torch.sum(z_diff.pow(2))
    
    total_loss = (recon_weight * recon_loss + 
                  kl_weight * kl_loss + 
                  state_weight * state_loss + 
                  physics_weight * physics_loss)
    
    return total_loss, recon_loss, kl_loss, state_loss, physics_loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def evaluate_model(model, loader, device, logger):
    model.eval()
    total_recon_loss = 0
    total_state_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, states, actions in loader:
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            
            batch_size, seq_len = images.shape[:2]
            t_points = torch.linspace(0, 1, seq_len, device=device)
            
            try:
                x_hat, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat = model(images, t_points, actions)
                
                recon_loss = F.mse_loss(x_hat, images, reduction='sum').item()
                state_loss = F.mse_loss(z_hat, states, reduction='sum').item()
                
                total_recon_loss += recon_loss
                total_state_loss += state_loss
                total_samples += batch_size * seq_len
            except Exception as e:
                logger.warning(f"Evaluation batch failed: {e}")
                continue
    
    avg_recon_loss = total_recon_loss / total_samples if total_samples > 0 else float('inf')
    avg_state_loss = total_state_loss / total_samples if total_samples > 0 else float('inf')
    
    return avg_recon_loss, avg_state_loss

def evaluate_30step_prediction(model, test_loader, device, logger, output_dir):
    model.eval()
    
    logger.info("="*60)
    logger.info("Starting 30-step prediction evaluation...")
    
    step_rmse_sums = np.zeros(30)
    step_counts = np.zeros(30)
    total_rmse = []
    
    with torch.no_grad():
        for batch_idx, (images, states, actions, file_names) in enumerate(test_loader):
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            
            batch_size = images.shape[0]
            
            for sample_idx in range(batch_size):
                sample_images = images[sample_idx:sample_idx+1, :10]
                sample_states = states[sample_idx]
                sample_actions = actions[sample_idx:sample_idx+1]
                
                mu_z0, _, _, _ = model.encoder(sample_images)
                
                z0 = mu_z0
                
                predicted_states = []
                z_current = z0.clone()
                
                for t in range(30):
                    def get_u_func(t_val):
                        return sample_actions[:, t]
                    
                    t_span = torch.tensor([0.0, 1.0], device=device)
                    z_next = odeint(
                        lambda t, z: model.ode_func(t, z, torch.tensor([[2.5]], device=device), get_u_func),
                        z_current,
                        t_span,
                        method='rk4'
                    )
                    
                    z_current = z_next[-1]
                    predicted_states.append(z_current.cpu().numpy())
                
                predicted_states = np.array(predicted_states).squeeze()
                true_states = sample_states.cpu().numpy()
                
                for t in range(30):
                    rmse = np.sqrt(mean_squared_error(true_states[t], predicted_states[t]))
                    step_rmse_sums[t] += rmse
                    step_counts[t] += 1
                
                total_rmse.append(np.sqrt(mean_squared_error(true_states.flatten(), predicted_states.flatten())))
    
    step_rmse_list = []
    for t in range(30):
        avg_rmse = step_rmse_sums[t] / step_counts[t] if step_counts[t] > 0 else 0
        step_rmse_list.append(float(avg_rmse))
        if (t + 1) % 5 == 0:
            logger.info(f"  Step {t+1}: Average RMSE = {avg_rmse:.6f}")
    
    total_30step_rmse = float(np.mean(total_rmse))
    logger.info(f"Total 30-step RMSE: {total_30step_rmse:.6f}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(range(1, 31), step_rmse_list, 'b-', marker='o', markersize=4)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Average RMSE')
    ax.set_title('30-Step Prediction Error')
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'prediction_error.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction error plot saved to: {plot_path}")
    logger.info("="*60)
    
    return step_rmse_list, total_30step_rmse

def visualize_reconstruction(model, loader, device, save_path, logger):
    model.eval()
    
    with torch.no_grad():
        for images, states, actions in loader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)
            
            batch_size, seq_len = images.shape[:2]
            t_points = torch.linspace(0, 1, seq_len, device=device)
            
            try:
                x_hat, _, _, _, _, z_hat = model(images, t_points, actions)
                
                sample_idx = 0
                num_frames = min(5, seq_len)
                
                fig, axes = plt.subplots(2, num_frames, figsize=(15, 6))
                
                for i in range(num_frames):
                    orig_img = images[sample_idx, i, 0].cpu().numpy()
                    recon_img = x_hat[sample_idx, i, 0].cpu().numpy()
                    
                    axes[0, i].imshow(orig_img, cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('Original')
                    
                    axes[1, i].imshow(recon_img, cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('Reconstructed')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Visualization saved to: {save_path}")
                break
                
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
                break

def train_gokunet(args):
    fold_idx = args.fold_idx
    noise_type = args.noise_type
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    seq_len = args.seq_len
    latent_dim = args.latent_dim
    theta_dim = args.theta_dim
    test_seq_len = args.test_seq_len
    patience = args.patience
    base_data_dir = args.base_data_dir
    output_base_dir = args.output_base_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = os.path.join(output_base_dir, f'gokunet_output_fold{fold_idx}_{noise_type}')
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logging(log_file)
    
    logger.info("="*60)
    logger.info(f"Training GOKU-net - Fold {fold_idx}, Noise type: {noise_type}")
    logger.info("="*60)
    
    params = {
        'model': 'GOKU-net',
        'fold': fold_idx,
        'noise_type': noise_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_len': seq_len,
        'latent_dim': latent_dim,
        'theta_dim': theta_dim,
        'test_seq_len': test_seq_len,
        'patience': patience,
        'device': str(device),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info("Training Parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    train_dirs = []
    val_dirs = []
    test_dirs = []
    
    for i in range(1, 6):
        fold_dir = os.path.join(base_data_dir, f'fold_{i}')
        if i == fold_idx:
            test_dirs.append(fold_dir)
        elif i == (fold_idx % 5) + 1:
            val_dirs.append(fold_dir)
        else:
            train_dirs.append(fold_dir)
    
    logger.info(f"Training folds: {[os.path.basename(d) for d in train_dirs]}")
    logger.info(f"Validation fold: {[os.path.basename(d) for d in val_dirs]}")
    logger.info(f"Test fold: {[os.path.basename(d) for d in test_dirs]}")
    
    train_dataset = DonkeyDataset(train_dirs, noise_type=noise_type, seq_len=seq_len)
    val_dataset = DonkeyDataset(val_dirs, noise_type=noise_type, seq_len=seq_len)
    test_dataset = DonkeyTestDataset(test_dirs[0], noise_type=noise_type, test_seq_len=test_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    model = GOKUNet(latent_dim=latent_dim, theta_dim=theta_dim, image_size=(120, 160)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    train_details = []
    val_details = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_state_loss = 0
        train_physics_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (images, states, actions) in enumerate(train_bar):
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            
            seq_len_batch = images.size(1)
            t_points = torch.linspace(0, 1, seq_len_batch, device=device)
            
            try:
                x_hat, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat = model(images, t_points, actions)
                
                loss, recon_loss, kl_loss, state_loss, physics_loss = gokunet_loss(
                    x_hat, images, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat, states
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_state_loss += state_loss.item()
                train_physics_loss += physics_loss.item()
                
            except Exception as e:
                logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
            
            train_bar.set_postfix({
                'loss': loss.item() / (images.size(0) * images.size(1)),
                'recon': recon_loss.item() / (images.size(0) * images.size(1))
            })
        
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_state_loss = 0
        val_physics_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, states, actions in val_bar:
                images, states, actions = images.to(device), states.to(device), actions.to(device)
                
                seq_len_batch = images.size(1)
                t_points = torch.linspace(0, 1, seq_len_batch, device=device)
                
                try:
                    x_hat, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat = model(images, t_points, actions)
                    
                    loss, recon_loss, kl_loss, state_loss, physics_loss = gokunet_loss(
                        x_hat, images, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat, states
                    )
                    
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_state_loss += state_loss.item()
                    val_physics_loss += physics_loss.item()
                    
                except Exception as e:
                    continue
                
                val_bar.set_postfix({
                    'loss': loss.item() / (images.size(0) * images.size(1)),
                    'recon': recon_loss.item() / (images.size(0) * images.size(1))
                })
        
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            avg_train_loss = train_loss / len(train_dataset)
            avg_val_loss = val_loss / len(val_dataset)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            train_details.append({
                'epoch': epoch + 1,
                'total_loss': avg_train_loss,
                'recon_loss': train_recon_loss / len(train_dataset),
                'kl_loss': train_kl_loss / len(train_dataset),
                'state_loss': train_state_loss / len(train_dataset),
                'physics_loss': train_physics_loss / len(train_dataset)
            })
            
            val_details.append({
                'epoch': epoch + 1,
                'total_loss': avg_val_loss,
                'recon_loss': val_recon_loss / len(val_dataset),
                'kl_loss': val_kl_loss / len(val_dataset),
                'state_loss': val_state_loss / len(val_dataset),
                'physics_loss': val_physics_loss / len(val_dataset)
            })
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
            
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
            
            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered! Stopping training at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (epoch {best_epoch})")
    
    logger.info("="*60)
    logger.info("Training complete! Starting evaluation...")
    logger.info("="*60)
    
    avg_recon_loss, avg_state_loss = evaluate_model(model, val_loader, device, logger)
    
    logger.info("Evaluation results:")
    logger.info(f"  - Reconstruction loss: {avg_recon_loss:.6f}")
    logger.info(f"  - State loss: {avg_state_loss:.6f}")
    
    step_rmse_list, total_30step_rmse = evaluate_30step_prediction(model, test_loader, device, logger, output_dir)
    
    vis_path = os.path.join(output_dir, 'reconstruction.png')
    visualize_reconstruction(model, val_loader, device, vis_path, logger)
    
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved as: {model_path}")
    
    def convert_to_python_types(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    results = {
        'parameters': convert_to_python_types(params),
        'train_losses': convert_to_python_types([float(x) for x in train_losses]),
        'val_losses': convert_to_python_types([float(x) for x in val_losses]),
        'train_details': convert_to_python_types(train_details),
        'val_details': convert_to_python_types(val_details),
        'best_epoch': int(best_epoch) if 'best_epoch' in locals() else 0,
        'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else 0,
        'final_epoch': int(len(train_losses)),
        'evaluation': convert_to_python_types({
            'recon_loss': avg_recon_loss,
            'state_loss': avg_state_loss,
            'prediction_30step': {
                'step_rmse_list': step_rmse_list,
                'total_rmse': total_30step_rmse
            }
        })
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved as: {results_path}")
    
    params_path = os.path.join(output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Parameters saved as: {params_path}")
    
    logger.info("="*60)
    logger.info(f"GOKU-net Fold {fold_idx}, noise type {noise_type} training complete!")
    logger.info("="*60)
    
    return model, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GOKU-net model')
    
    parser.add_argument('--fold_idx', type=int, default=1, help='Fold index for cross-validation (1-5)')
    parser.add_argument('--noise_type', type=str, default='states', 
                        choices=['states', 'noisy_states_2', 'noisy_states_5', 'noisy_states_10'],
                        help='Type of noise in states')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for training')
    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of latent space')
    parser.add_argument('--theta_dim', type=int, default=1, help='Dimension of theta parameters')
    parser.add_argument('--test_seq_len', type=int, default=30, help='Sequence length for testing')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--base_data_dir', type=str, default='./data/driving_datasets', 
                        help='Base directory for data')
    parser.add_argument('--output_base_dir', type=str, default='./outputs', 
                        help='Base directory for outputs')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Starting training GOKU-net - Fold {args.fold_idx}, Noise type: {args.noise_type}")
    print("="*60)
    
    model, results = train_gokunet(args)
    
    print("="*60)
    print(f"GOKU-net Fold {args.fold_idx}, noise type {args.noise_type} training complete!")
    print(f"Results saved in: {args.output_base_dir}/gokunet_output_fold{args.fold_idx}_{args.noise_type}/")
    print("="*60)
