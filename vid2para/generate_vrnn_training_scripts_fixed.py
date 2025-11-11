import os

# VRNN训练脚本基础模板 - 修复版
base_template = """import torch
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

# 设置日志
def setup_logging(log_file):
    \"\"\"设置日志记录\"\"\"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# 数据集类
class DonkeyDataset(Dataset):
    def __init__(self, data_dirs, noise_type='states', seq_len=10):
        self.noise_type = noise_type
        self.seq_len = seq_len
        self.sequences = []
        
        # 支持多个目录
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        for data_dir in data_dirs:
            files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
            
            for file in files:
                data = np.load(os.path.join(data_dir, file))
                images = data['imgs']
                actions = data['acts']
                
                # 根据噪声类型选择状态
                if noise_type == 'states':
                    states = data['states']
                elif noise_type == 'noisy_states_2':
                    states = data['noisy_states_2']
                elif noise_type == 'noisy_states_5':
                    states = data['noisy_states_5']
                elif noise_type == 'noisy_states_10':
                    states = data['noisy_states_10']
                
                # 创建序列，每个序列长度为seq_len
                num_sequences = len(images) - seq_len + 1
                for i in range(num_sequences):
                    img_seq = images[i:i+seq_len]
                    state_seq = states[i:i+seq_len]
                    action_seq = actions[i:i+seq_len]
                    
                    self.sequences.append({{
                        'images': img_seq,
                        'states': state_seq,
                        'actions': action_seq
                    }})
        
        print(f"加载数据: {{len(self.sequences)}} 个序列, 噪声类型: {{noise_type}}, 序列长度: {{seq_len}}")
        if len(self.sequences) > 0:
            print(f"图像形状: {{self.sequences[0]['images'].shape}}")
            print(f"状态形状: {{self.sequences[0]['states'].shape}}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 处理图像序列: (T, H, W, C) -> (T, C, H, W)
        images = seq['images'].astype(np.float32) / 255.0
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        
        # 处理状态序列
        states = torch.from_numpy(seq['states'].astype(np.float32))
        
        # 处理动作序列
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        # 创建下一时刻状态（用于物理损失计算）
        next_states = torch.cat([states[1:], states[-1:]], dim=0)
        
        return images, states, next_states, actions

# 测试数据集类 - 专门用于30步预测测试
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
            
            # 根据噪声类型选择状态
            if noise_type == 'states':
                states = data['states']
            elif noise_type == 'noisy_states_2':
                states = data['noisy_states_2']
            elif noise_type == 'noisy_states_5':
                states = data['noisy_states_5']
            elif noise_type == 'noisy_states_10':
                states = data['noisy_states_10']
            
            # 创建30步长的测试序列
            num_test_sequences = len(images) - test_seq_len + 1
            for i in range(num_test_sequences):
                img_seq = images[i:i+test_seq_len]
                state_seq = states[i:i+test_seq_len]
                action_seq = actions[i:i+test_seq_len]
                
                self.test_sequences.append({{
                    'images': img_seq,
                    'states': state_seq,
                    'actions': action_seq
                }})
                self.file_names.append(f"{{file}}_seq{{i}}")
        
        print(f"加载测试数据: {{len(self.test_sequences)}} 个{{test_seq_len}}步序列")
    
    def __len__(self):
        return len(self.test_sequences)
    
    def __getitem__(self, idx):
        seq = self.test_sequences[idx]
        
        # 处理图像序列
        images = seq['images'].astype(np.float32) / 255.0
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        # 处理状态序列
        states = torch.from_numpy(seq['states'].astype(np.float32))
        
        # 处理动作序列
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        return images, states, actions, self.file_names[idx]

# --- VRNN模块定义 ---
class Encoder(nn.Module):
    \"\"\"
    VRNN编码器：将图像和状态信息编码成潜在表示。
    \"\"\"
    def __init__(self, image_channels=3, state_dim=4, hidden_dim=256, z_dim=128):
        super().__init__()
        # 适应donkey数据集的图像尺寸 (120x160)
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # 120x160 -> 60x80
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 60x80 -> 30x40
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 30x40 -> 15x20
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 15x20 -> 7x10
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算卷积输出维度
        conv_output_dim = 256 * 7 * 10  # 17920
        
        self.fc_img = nn.Linear(conv_output_dim, hidden_dim)
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, image, state):
        img_features = self.fc_img(self.conv(image))
        state_features = self.fc_state(state)
        merged_features = self.fc_merge(torch.cat([img_features, state_features], dim=1))
        mu = self.fc_mu(merged_features)
        logvar = self.fc_logvar(merged_features)
        return mu, logvar

class Decoder(nn.Module):
    \"\"\"
    VRNN解码器：将潜在表示和RNN隐藏状态解码回图像。
    \"\"\"
    def __init__(self, z_dim=128, hidden_dim=256, image_channels=3):
        super().__init__()
        conv_input_dim = 256 * 7 * 10  # 17920
        
        self.fc = nn.Linear(z_dim + hidden_dim, conv_input_dim)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1,0)),  # 7x10 -> 15x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 15x20 -> 30x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 30x40 -> 60x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # 60x80 -> 120x160
            nn.Sigmoid()
        )

    def forward(self, z, rnn_h):
        z_combined = torch.cat([z, rnn_h], dim=1)
        x = self.fc(z_combined).view(-1, 256, 7, 10)
        return self.conv_trans(x)

class Prior(nn.Module):
    \"\"\"
    先验网络：基于RNN的隐藏状态预测潜在表示的先验分布。
    \"\"\"
    def __init__(self, hidden_dim=256, z_dim=128):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, rnn_h):
        mu = self.fc_mu(rnn_h)
        logvar = self.fc_logvar(rnn_h)
        return mu, logvar

class VRNNCar(nn.Module):
    \"\"\"
    完整的VRNN模型，用于车辆动力学参数推断。
    \"\"\"
    def __init__(self, image_channels=3, state_dim=4, hidden_dim=256, z_dim=128, l_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.l_dim = l_dim

        self.encoder = Encoder(image_channels, state_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, image_channels)
        self.rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.prior = Prior(hidden_dim, z_dim)

        # 物理参数预测层
        self.fc_l_mu = nn.Linear(z_dim, l_dim)
        self.fc_l_logvar = nn.Linear(z_dim, l_dim)
        
        # 状态预测层（用于监督学习）
        self.state_predictor = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )

    def reparameterize(self, mu, logvar):
        \"\"\"重参数化技巧，用于从潜在分布中采样。\"\"\"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, s_seq, h_0=None):
        \"\"\"
        前向传播：处理一个完整的序列。
        \"\"\"
        B, T, C, H, W = x_seq.size()

        # 初始化存储列表
        x_rec_seq, mu_l_seq, logvar_l_seq = [], [], []
        mu_enc_seq, logvar_enc_seq = [], []
        mu_prior_seq, logvar_prior_seq = [], []
        state_pred_seq = []

        # 初始化RNN隐藏状态
        h_t = h_0 if h_0 is not None else torch.zeros(B, self.hidden_dim, device=x_seq.device)
        
        z_seq = []

        for t in range(T):
            # 获取当前时间步的数据
            x_t = x_seq[:, t]
            s_t = s_seq[:, t]

            # 1. 先验预测
            mu_prior_t, logvar_prior_t = self.prior(h_t)

            # 2. 编码器推断
            mu_enc_t, logvar_enc_t = self.encoder(x_t, s_t)
            z_t = self.reparameterize(mu_enc_t, logvar_enc_t)

            # 3. 物理参数预测
            mu_l_t = self.fc_l_mu(z_t)
            logvar_l_t = self.fc_l_logvar(z_t)
            
            # 4. 状态预测
            state_pred_t = self.state_predictor(z_t)

            # 5. 解码器生成
            x_rec_t = self.decoder(z_t, h_t)

            # 6. RNN更新
            z_input = z_t.unsqueeze(1)  # (B, 1, z_dim)
            _, h_t = self.rnn(z_input, h_t.unsqueeze(0))
            h_t = h_t.squeeze(0)  # (B, hidden_dim)

            # 存储结果
            x_rec_seq.append(x_rec_t)
            mu_l_seq.append(mu_l_t)
            logvar_l_seq.append(logvar_l_t)
            mu_enc_seq.append(mu_enc_t)
            logvar_enc_seq.append(logvar_enc_t)
            mu_prior_seq.append(mu_prior_t)
            logvar_prior_seq.append(logvar_prior_t)
            state_pred_seq.append(state_pred_t)
            z_seq.append(z_t)

        return (torch.stack(x_rec_seq, dim=1), torch.stack(mu_l_seq, dim=1), torch.stack(logvar_l_seq, dim=1),
                torch.stack(mu_enc_seq, dim=1), torch.stack(logvar_enc_seq, dim=1),
                torch.stack(mu_prior_seq, dim=1), torch.stack(logvar_prior_seq, dim=1),
                torch.stack(state_pred_seq, dim=1))
    
    def predict_sequence(self, x_init, s_init, predict_steps=30):
        \"\"\"
        给定初始图像和状态，预测未来序列
        \"\"\"
        self.eval()
        B = x_init.size(0)
        
        with torch.no_grad():
            # 初始化
            h_t = torch.zeros(B, self.hidden_dim, device=x_init.device)
            
            # 使用初始输入更新隐藏状态
            mu_enc_init, logvar_enc_init = self.encoder(x_init, s_init)
            z_init = self.reparameterize(mu_enc_init, logvar_enc_init)
            z_input = z_init.unsqueeze(1)
            _, h_t = self.rnn(z_input, h_t.unsqueeze(0))
            h_t = h_t.squeeze(0)
            
            # 预测序列
            predicted_states = []
            predicted_images = []
            
            current_state = s_init
            
            for step in range(predict_steps):
                # 先验预测
                mu_prior, logvar_prior = self.prior(h_t)
                z_pred = self.reparameterize(mu_prior, logvar_prior)
                
                # 状态预测
                state_pred = self.state_predictor(z_pred)
                predicted_states.append(state_pred)
                
                # 图像重建
                img_pred = self.decoder(z_pred, h_t)
                predicted_images.append(img_pred)
                
                # 更新RNN隐藏状态
                z_input = z_pred.unsqueeze(1)
                _, h_t = self.rnn(z_input, h_t.unsqueeze(0))
                h_t = h_t.squeeze(0)
                
                # 更新当前状态（用于下一步预测）
                current_state = state_pred
        
        return torch.stack(predicted_states, dim=1), torch.stack(predicted_images, dim=1)

# 损失函数
def vrnn_loss(x_rec, x_orig, mu_enc, logvar_enc, mu_prior, logvar_prior, 
              mu_l, logvar_l, state_pred, state_true, alpha=1.0, beta=0.01, gamma=1.0):
    \"\"\"
    VRNN总损失函数
    \"\"\"
    # 1. 重建损失
    recon_loss = F.mse_loss(x_rec, x_orig, reduction='sum')
    
    # 2. KL损失（编码器分布与先验分布的KL散度）
    kl_loss = -0.5 * torch.sum(
        1 + logvar_enc - logvar_prior - 
        (mu_enc - mu_prior).pow(2) / torch.exp(logvar_prior) - 
        torch.exp(logvar_enc) / torch.exp(logvar_prior)
    )
    
    # 3. 状态监督损失
    supervised_loss = F.mse_loss(state_pred, state_true, reduction='sum')
    
    # 4. 物理参数正则化（假设真实轴距L=2.5，这是典型的小车轴距）
    l_target = torch.full_like(mu_l, 2.5)
    physical_loss = F.mse_loss(mu_l, l_target, reduction='sum')
    
    total_loss = alpha * recon_loss + beta * kl_loss + gamma * supervised_loss + 0.1 * physical_loss
    
    return total_loss, recon_loss, kl_loss, supervised_loss, physical_loss

# 早停机制
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

# 静态编码和重建评估函数
def evaluate_static_encoding_reconstruction(model, test_loader, device, logger):
    \"\"\"评估静态编码RMSE和静态重建MSE\"\"\"
    model.eval()
    
    all_encoding_errors = []
    all_reconstruction_errors = []
    
    with torch.no_grad():
        for images, states, next_states, actions in tqdm(test_loader, desc="静态评估"):
            images, states = images.to(device), states.to(device)
            B, T = images.size(0), images.size(1)
            
            # 对每个时间步进行静态编码和重建
            for t in range(T):
                x_t = images[:, t]
                s_t = states[:, t]
                
                # 静态编码（不使用RNN状态）
                mu_enc, logvar_enc = model.encoder(x_t, s_t)
                z_static = model.reparameterize(mu_enc, logvar_enc)
                
                # 静态状态预测
                state_pred_static = model.state_predictor(z_static)
                
                # 静态重建（使用零隐藏状态）
                h_zero = torch.zeros(x_t.size(0), model.hidden_dim, device=device)
                x_rec_static = model.decoder(z_static, h_zero)
                
                # 计算编码RMSE（状态预测误差）
                encoding_mse = F.mse_loss(state_pred_static, s_t, reduction='none')  # (B, state_dim)
                encoding_rmse = torch.sqrt(encoding_mse.mean(dim=0))  # 每个维度的RMSE
                all_encoding_errors.append(encoding_rmse.cpu().numpy())
                
                # 计算重建MSE
                recon_mse = F.mse_loss(x_rec_static, x_t, reduction='none')  # (B, C, H, W)
                recon_mse_per_sample = recon_mse.view(recon_mse.size(0), -1).mean(dim=1)  # (B,)
                all_reconstruction_errors.append(recon_mse_per_sample.cpu().numpy())
    
    # 计算总体静态编码RMSE
    all_encoding_errors = np.array(all_encoding_errors)  # (num_samples, state_dim)
    static_encoding_rmse = np.mean(all_encoding_errors, axis=0)  # 每个状态维度的平均RMSE
    static_encoding_rmse_total = np.mean(static_encoding_rmse)  # 总体RMSE
    
    # 计算总体静态重建MSE
    all_reconstruction_errors = np.concatenate(all_reconstruction_errors)  # (total_samples,)
    static_reconstruction_mse = np.mean(all_reconstruction_errors)
    
    logger.info("静态评估结果:")
    logger.info(f"  - 静态编码总RMSE: {{static_encoding_rmse_total:.6f}}")
    logger.info(f"  - 静态编码各维度RMSE:")
    for i, rmse in enumerate(static_encoding_rmse):
        logger.info(f"    维度{{i}}: {{rmse:.6f}}")
    logger.info(f"  - 静态重建MSE: {{static_reconstruction_mse:.6f}}")
    
    return static_encoding_rmse_total, static_encoding_rmse.tolist(), static_reconstruction_mse

# 30步预测评估函数
def evaluate_30step_prediction(model, test_loader, device, logger, output_dir):
    \"\"\"评估30步预测性能并进行可视化\"\"\"
    model.eval()
    
    all_step_rmse = []  # 每一步的RMSE列表
    first_sequence_data = None  # 保存第一个序列的数据用于可视化
    
    with torch.no_grad():
        for batch_idx, (images, states, actions, file_names) in enumerate(tqdm(test_loader, desc="30步预测评估")):
            images, states = images.to(device), states.to(device)
            B = images.size(0)
            
            for sample_idx in range(B):
                # 获取单个样本
                sample_images = images[sample_idx:sample_idx+1]  # (1, 30, C, H, W)
                sample_states = states[sample_idx:sample_idx+1]  # (1, 30, state_dim)
                
                # 使用第一帧进行30步预测
                init_image = sample_images[:, 0]  # (1, C, H, W)
                init_state = sample_states[:, 0]  # (1, state_dim)
                ground_truth_states = sample_states[:, 1:]  # (1, 29, state_dim) - 除了第一帧的真实状态
                ground_truth_images = sample_images[:, 1:]  # (1, 29, C, H, W)
                
                # 进行29步预测（因为第一步是初始状态）
                pred_states, pred_images = model.predict_sequence(init_image, init_state, predict_steps=29)
                
                # 计算每一步的RMSE
                step_rmse = []
                for step in range(29):
                    pred_state_step = pred_states[:, step]  # (1, state_dim)
                    true_state_step = ground_truth_states[:, step]  # (1, state_dim)
                    
                    mse = F.mse_loss(pred_state_step, true_state_step, reduction='mean')
                    rmse = torch.sqrt(mse).item()
                    step_rmse.append(rmse)
                
                all_step_rmse.append(step_rmse)
                
                # 保存第一个序列的数据用于可视化
                if batch_idx == 0 and sample_idx == 0:
                    first_sequence_data = {{
                        'init_image': init_image.cpu(),
                        'init_state': init_state.cpu(),
                        'pred_states': pred_states.cpu(),
                        'pred_images': pred_images.cpu(),
                        'true_states': ground_truth_states.cpu(),
                        'true_images': ground_truth_images.cpu(),
                        'file_name': file_names[sample_idx]
                    }}
                
                # 只处理少量样本以节省时间
                if batch_idx >= 10:  # 限制处理的batch数量
                    break
            
            if batch_idx >= 10:
                break
    
    # 计算每一步的平均RMSE
    all_step_rmse = np.array(all_step_rmse)  # (num_samples, 29)
    mean_step_rmse = np.mean(all_step_rmse, axis=0)  # (29,)
    
    logger.info("30步预测评估结果:")
    logger.info("各步骤RMSE:")
    for step, rmse in enumerate(mean_step_rmse):
        logger.info(f"  步骤{{step+1:2d}}: {{rmse:.6f}}")
    
    total_30step_rmse = np.mean(mean_step_rmse)
    logger.info(f"30步预测总平均RMSE: {{total_30step_rmse:.6f}}")
    
    # 可视化第一个序列
    if first_sequence_data is not None:
        visualize_30step_prediction(first_sequence_data, output_dir, logger)
    
    return mean_step_rmse.tolist(), total_30step_rmse

def visualize_30step_prediction(sequence_data, output_dir, logger):
    \"\"\"可视化30步预测结果\"\"\"
    # 状态预测可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('30-Step Prediction vs Ground Truth - States', fontsize=16)
    
    state_names = ['X Position', 'Y Position', 'Orientation', 'Speed']
    
    init_state = sequence_data['init_state'].squeeze().numpy()  # (state_dim,)
    pred_states = sequence_data['pred_states'].squeeze().numpy()  # (29, state_dim)
    true_states = sequence_data['true_states'].squeeze().numpy()  # (29, state_dim)
    
    steps = np.arange(1, 30)  # 步骤1-29
    
    for i in range(4):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 绘制初始状态
        ax.plot(0, init_state[i], 'go', markersize=8, label='Initial')
        
        # 绘制预测状态
        ax.plot(steps, pred_states[:, i], 'r-', linewidth=2, label='Predicted')
        
        # 绘制真实状态
        ax.plot(steps, true_states[:, i], 'b--', linewidth=2, label='Ground Truth')
        
        ax.set_title(f'{{state_names[i]}}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    state_vis_path = os.path.join(output_dir, '30step_state_prediction.png')
    plt.savefig(state_vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图像预测可视化（选择几个关键时间步）
    key_steps = [0, 4, 9, 14, 19, 24, 28]  # 第1, 5, 10, 15, 20, 25, 29步
    fig, axes = plt.subplots(3, len(key_steps), figsize=(len(key_steps)*3, 9))
    
    # 初始图像
    init_img = sequence_data['init_image'].squeeze().permute(1, 2, 0).numpy()
    pred_imgs = sequence_data['pred_images'].squeeze().permute(0, 2, 3, 1).numpy()  # (29, H, W, C)
    true_imgs = sequence_data['true_images'].squeeze().permute(0, 2, 3, 1).numpy()  # (29, H, W, C)
    
    for i, step in enumerate(key_steps):
        if step == 0:
            # 显示初始图像
            axes[0, i].imshow(init_img)
            axes[0, i].set_title(f'Step {{step+1}} (Init)')
            axes[1, i].imshow(init_img)  # 预测图像行也显示初始图像
            axes[2, i].imshow(init_img)  # 真实图像行也显示初始图像
        else:
            # 显示预测和真实图像
            axes[1, i].imshow(pred_imgs[step-1])
            axes[1, i].set_title(f'Step {{step+1}} (Pred)')
            axes[2, i].imshow(true_imgs[step-1])
            axes[2, i].set_title(f'Step {{step+1}} (True)')
            
            # 第一行显示差异图
            diff_img = np.abs(pred_imgs[step-1] - true_imgs[step-1])
            axes[0, i].imshow(diff_img)
            axes[0, i].set_title(f'Step {{step+1}} (Diff)')
        
        for row in range(3):
            axes[row, i].axis('off')
    
    # 设置行标签
    axes[0, 0].set_ylabel('Difference', fontsize=12)
    axes[1, 0].set_ylabel('Predicted', fontsize=12)
    axes[2, 0].set_ylabel('Ground Truth', fontsize=12)
    
    plt.tight_layout()
    image_vis_path = os.path.join(output_dir, '30step_image_prediction.png')
    plt.savefig(image_vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"30步预测可视化已保存:")
    logger.info(f"  - 状态预测: {{state_vis_path}}")
    logger.info(f"  - 图像预测: {{image_vis_path}}")

# 评估函数
def evaluate_model(model, test_loader, device, logger):
    model.eval()
    
    all_state_preds = []
    all_state_trues = []
    total_recon_mse = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, states, next_states, actions in tqdm(test_loader, desc="评估中"):
            images, states = images.to(device), states.to(device)
            
            (x_rec, mu_l, logvar_l, mu_enc, logvar_enc, 
             mu_prior, logvar_prior, state_pred) = model(images, states)
            
            # 计算重构MSE
            recon_mse = F.mse_loss(x_rec, images, reduction='sum').item()
            total_recon_mse += recon_mse
            
            # 收集状态预测
            all_state_preds.append(state_pred.cpu().numpy())
            all_state_trues.append(states.cpu().numpy())
            
            num_samples += images.size(0)
    
    # 计算平均重构MSE
    avg_recon_mse = float(total_recon_mse / num_samples)
    
    # 计算状态预测RMSE
    all_state_preds = np.concatenate(all_state_preds, axis=0)
    all_state_trues = np.concatenate(all_state_trues, axis=0)
    
    # 平均时间步的状态预测误差
    state_preds_mean = np.mean(all_state_preds, axis=1)
    state_trues_mean = np.mean(all_state_trues, axis=1)
    
    state_rmse = float(np.sqrt(mean_squared_error(state_trues_mean, state_preds_mean)))
    
    # 计算每个维度的RMSE
    dim_rmse = []
    for i in range(state_trues_mean.shape[1]):
        rmse = float(np.sqrt(mean_squared_error(state_trues_mean[:, i], state_preds_mean[:, i])))
        dim_rmse.append(rmse)
    
    return avg_recon_mse, state_rmse, dim_rmse

# 可视化函数
def visualize_reconstruction(model, test_loader, device, save_path, logger):
    model.eval()
    
    # 获取一个batch
    data_iter = iter(test_loader)
    images, states, next_states, actions = next(data_iter)
    images = images.to(device)
    states = states.to(device)
    
    with torch.no_grad():
        (x_rec, _, _, _, _, _, _, _) = model(images, states)
    
    # 选择前4个样本展示，每个样本显示第一个时间步
    n_samples = min(4, images.size(0))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
    
    for i in range(n_samples):
        # 原图（第一个时间步）
        original = images[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(original)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original (t=0)', fontsize=10)
        
        # 重构图（第一个时间步）
        reconstructed = x_rec[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(reconstructed)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed (t=0)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"重构可视化已保存到: {{save_path}}")

# 训练函数
def train_vrnn(fold_idx, noise_type, epochs=100, batch_size=16, learning_rate=1e-3, seq_len=10):
    # 创建输出目录
    output_dir = f'vrnn_output_fold{{fold_idx}}_{{noise_type}}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logging(log_file)
    
    # 记录训练参数
    params = {{
        'fold_idx': fold_idx,
        'noise_type': noise_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_len': seq_len,
        'hidden_dim': 256,
        'z_dim': 128,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }}
    
    logger.info("="*60)
    logger.info(f"开始VRNN训练 - Fold {{fold_idx}}, 噪声类型: {{noise_type}}")
    logger.info(f"训练参数: {{json.dumps(params, indent=2)}}")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {{device}}")
    
    # 修复路径：使用../../donkeynew_fold而不是../donkeynew_fold
    train_dirs = []
    val_dir = f'../../donkeynew_fold{{fold_idx}}'
    
    for i in range(1, 6):
        if i != fold_idx:
            train_dirs.append(f'../../donkeynew_fold{{i}}')
    
    logger.info(f"验证集目录: {{val_dir}}")
    logger.info(f"训练集目录: {{train_dirs}}")
    
    # 加载数据
    try:
        train_dataset = DonkeyDataset(train_dirs, noise_type, seq_len)
        val_dataset = DonkeyDataset(val_dir, noise_type, seq_len)
        
        # 创建测试数据集（用于30步预测）
        test_dataset = DonkeyTestDataset(val_dir, noise_type, test_seq_len=30)
    except Exception as e:
        logger.error(f"数据加载失败: {{e}}")
        raise
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # 小batch size用于测试
    
    # 初始化模型
    model = VRNNCar(image_channels=3, state_dim=4, hidden_dim=256, z_dim=128, l_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_details = []
    val_details = []
    
    logger.info("开始训练...")
    best_val_loss = float('inf')
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_supervised_loss = 0
        train_physical_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{epochs}} [Train]')
        for batch_idx, (images, states, next_states, actions) in enumerate(train_bar):
            images, states = images.to(device), states.to(device)
            
            optimizer.zero_grad()
            
            (x_rec, mu_l, logvar_l, mu_enc, logvar_enc, 
             mu_prior, logvar_prior, state_pred) = model(images, states)
            
            loss, recon_loss, kl_loss, supervised_loss, physical_loss = vrnn_loss(
                x_rec, images, mu_enc, logvar_enc, mu_prior, logvar_prior,
                mu_l, logvar_l, state_pred, states
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_supervised_loss += supervised_loss.item()
            train_physical_loss += physical_loss.item()
            
            # 更新进度条
            train_bar.set_postfix({{
                'loss': loss.item() / (images.size(0) * images.size(1)),
                'recon': recon_loss.item() / (images.size(0) * images.size(1))
            }})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_supervised_loss = 0
        val_physical_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {{epoch+1}}/{{epochs}} [Val]')
            for images, states, next_states, actions in val_bar:
                images, states = images.to(device), states.to(device)
                
                (x_rec, mu_l, logvar_l, mu_enc, logvar_enc, 
                 mu_prior, logvar_prior, state_pred) = model(images, states)
                
                loss, recon_loss, kl_loss, supervised_loss, physical_loss = vrnn_loss(
                    x_rec, images, mu_enc, logvar_enc, mu_prior, logvar_prior,
                    mu_l, logvar_l, state_pred, states
                )
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_supervised_loss += supervised_loss.item()
                val_physical_loss += physical_loss.item()
                
                val_bar.set_postfix({{
                    'loss': loss.item() / (images.size(0) * images.size(1)),
                    'recon': recon_loss.item() / (images.size(0) * images.size(1))
                }})
        
        # 记录损失
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 记录详细损失
        train_details.append({{
            'epoch': epoch + 1,
            'total_loss': avg_train_loss,
            'recon_loss': train_recon_loss / len(train_dataset),
            'kl_loss': train_kl_loss / len(train_dataset),
            'supervised_loss': train_supervised_loss / len(train_dataset),
            'physical_loss': train_physical_loss / len(train_dataset)
        }})
        
        val_details.append({{
            'epoch': epoch + 1,
            'total_loss': avg_val_loss,
            'recon_loss': val_recon_loss / len(val_dataset),
            'kl_loss': val_kl_loss / len(val_dataset),
            'supervised_loss': val_supervised_loss / len(val_dataset),
            'physical_loss': val_physical_loss / len(val_dataset)
        }})
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        
        # 记录日志
        if (epoch + 1) % 5 == 0:
            logger.info(f'Epoch {{epoch+1:3d}}: Train Loss = {{avg_train_loss:.4f}}, Val Loss = {{avg_val_loss:.4f}}')
            logger.info(f'  Train - Recon: {{train_recon_loss/len(train_dataset):.4f}}, '
                       f'KL: {{train_kl_loss/len(train_dataset):.4f}}, '
                       f'Supervised: {{train_supervised_loss/len(train_dataset):.4f}}, '
                       f'Physical: {{train_physical_loss/len(train_dataset):.4f}}')
        
        # 早停检查
        if early_stopping(avg_val_loss):
            logger.info(f"早停触发！在epoch {{epoch+1}}停止训练")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"加载最佳模型 (epoch {{best_epoch}})")
    
    logger.info("="*60)
    logger.info("训练完成！开始评估...")
    logger.info("="*60)
    
    # 标准评估
    avg_recon_mse, state_rmse, dim_rmse = evaluate_model(model, val_loader, device, logger)
    
    logger.info("标准评估结果:")
    logger.info(f"  - 重构MSE: {{avg_recon_mse:.6f}}")
    logger.info(f"  - 状态预测总RMSE: {{state_rmse:.6f}}")
    logger.info(f"  - 各维度RMSE:")
    for i, rmse in enumerate(dim_rmse):
        logger.info(f"    维度{{i}}: {{rmse:.6f}}")
    
    # 静态编码和重建评估
    static_encoding_rmse_total, static_encoding_rmse_dims, static_reconstruction_mse = evaluate_static_encoding_reconstruction(
        model, val_loader, device, logger
    )
    
    # 30步预测评估
    step_rmse_list, total_30step_rmse = evaluate_30step_prediction(model, test_loader, device, logger, output_dir)
    
    # 可视化重构结果
    vis_path = os.path.join(output_dir, 'reconstruction.png')
    visualize_reconstruction(model, val_loader, device, vis_path, logger)
    
    # 保存模型
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存为: {{model_path}}")
    
    # 保存训练结果
    def convert_to_python_types(obj):
        '''Recursively convert numpy types to Python native types'''
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {{key: convert_to_python_types(value) for key, value in obj.items()}}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    results = {{
        'parameters': params,
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_details': convert_to_python_types(train_details),
        'val_details': convert_to_python_types(val_details),
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'final_epoch': int(len(train_losses)),
        'evaluation': {{
            'standard': {{
                'recon_mse': avg_recon_mse,
                'state_rmse': state_rmse,
                'dim_rmse': dim_rmse
            }},
            'static': {{
                'encoding_rmse_total': static_encoding_rmse_total,
                'encoding_rmse_dims': static_encoding_rmse_dims,
                'reconstruction_mse': static_reconstruction_mse
            }},
            'prediction_30step': {{
                'step_rmse_list': step_rmse_list,
                'total_rmse': total_30step_rmse
            }}
        }}
    }}
    
    # 保存结果JSON
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存为: {{results_path}}")
    
    # 保存参数JSON
    params_path = os.path.join(output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"参数已保存为: {{params_path}}")
    
    logger.info("="*60)
    logger.info(f"VRNN Fold {{fold_idx}}, 噪声类型 {{noise_type}} 训练完成!")
    logger.info("="*60)
    
    return model, results

if __name__ == "__main__":
    # 训练参数
    FOLD_IDX = {fold_idx}
    NOISE_TYPE = '{noise_type}'
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    SEQ_LEN = 10
    
    print("="*60)
    print(f"开始训练 VRNN - Fold {{FOLD_IDX}}, 噪声类型: {{NOISE_TYPE}}")
    print("="*60)
    
    # 开始训练
    model, results = train_vrnn(FOLD_IDX, NOISE_TYPE, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEQ_LEN)
    
    print("="*60)
    print(f"VRNN Fold {{FOLD_IDX}}, 噪声类型 {{NOISE_TYPE}} 训练完成!")
    print(f"结果保存在: vrnn_output_fold{{FOLD_IDX}}_{{NOISE_TYPE}}/")
    print("="*60)
"""

def generate_vrnn_training_scripts_fixed():
    # 5个fold和3种噪声类型
    folds = [1, 2, 3, 4, 5]
    noise_types = ['states', 'noisy_states_5', 'noisy_states_10']
    
    print("开始生成15个修复版VRNN训练脚本...")
    
    # 确保目录存在
    os.makedirs('vrnn_training_scripts_fixed', exist_ok=True)
    
    script_count = 0
    for fold in folds:
        for noise_type in noise_types:
            script_count += 1
            filename = f'vrnn_training_scripts_fixed/train_vrnn_fold{fold}_{noise_type}_fixed.py'
            
            # 生成脚本内容
            script_content = base_template.format(
                fold_idx=fold,
                noise_type=noise_type
            )
            
            # 写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            print(f"生成脚本 {script_count}/15: {filename}")
    
    print(f"\n总共生成了 {script_count} 个修复版VRNN训练脚本!")
    print("\n修复版VRNN脚本列表:")
    
    for fold in folds:
        for noise_type in noise_types:
            filename = f'train_vrnn_fold{fold}_{noise_type}_fixed.py'
            print(f"  - {filename}")
    
    print("\n修复和新增功能:")
    print("  - 修复路径问题：使用../../donkeynew_fold")
    print("  - 静态编码RMSE评估")
    print("  - 静态重建MSE评估")
    print("  - 30步预测测试与每步RMSE记录")
    print("  - 第一个序列30步预测可视化")
    print("  - 状态和图像预测对比可视化")
    print("  - 完整的结果保存和日志记录")

if __name__ == "__main__":
    generate_vrnn_training_scripts_fixed()