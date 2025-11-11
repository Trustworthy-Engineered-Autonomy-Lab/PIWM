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

# 设置日志
def setup_logging(log_file):
    """设置日志记录"""
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
                    
                    self.sequences.append({
                        'images': img_seq,
                        'states': state_seq,
                        'actions': action_seq
                    })
        
        print(f"加载数据: {len(self.sequences)} 个序列, 噪声类型: {noise_type}, 序列长度: {seq_len}")
        if len(self.sequences) > 0:
            print(f"图像形状: {self.sequences[0]['images'].shape}")
            print(f"状态形状: {self.sequences[0]['states'].shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 处理图像序列: 转换为灰度图并标准化 (T, H, W, C) -> (T, 1, H, W)
        images = seq['images'].astype(np.float32) / 255.0
        # 转换为灰度图: RGB -> Gray
        if images.shape[-1] == 3:
            images = np.mean(images, axis=-1, keepdims=True)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        
        # 处理状态序列
        states = torch.from_numpy(seq['states'].astype(np.float32))
        
        # 处理动作序列
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        return images, states, actions

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
                
                self.test_sequences.append({
                    'images': img_seq,
                    'states': state_seq,
                    'actions': action_seq
                })
                self.file_names.append(f"{file}_seq{i}")
        
        print(f"加载测试数据: {len(self.test_sequences)} 个{test_seq_len}步序列")
    
    def __len__(self):
        return len(self.test_sequences)
    
    def __getitem__(self, idx):
        seq = self.test_sequences[idx]
        
        # 处理图像序列: 转换为灰度图
        images = seq['images'].astype(np.float32) / 255.0
        if images.shape[-1] == 3:
            images = np.mean(images, axis=-1, keepdims=True)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        # 处理状态序列
        states = torch.from_numpy(seq['states'].astype(np.float32))
        
        # 处理动作序列
        actions = torch.from_numpy(seq['actions'].astype(np.float32))
        
        return images, states, actions, self.file_names[idx]

# ----------------------------------------------------
# GOKU-net 模型定义
# ----------------------------------------------------

# 1. ODE 函数 (物理模型)
class BicycleODE(nn.Module):
    """
    Bicycle Model 的连续时间 ODE 函数。

    状态向量 z: [x, y, heading_angle, speed]
    参数 theta: [wheelbase]
    控制输入 u: [delta_angle, acceleration]
    """

    def __init__(self):
        super(BicycleODE, self).__init__()

    def forward(self, t, z, theta, get_u_func):
        """
        根据当前状态和参数，计算状态的变化率 (dz/dt)。
        """
        # 从状态z中解包物理变量
        x, y, psi, v = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        # 从参数theta中解包轴距
        L = theta[:, 0]

        # 获取当前时间点的控制输入u
        u = get_u_func(t)
        delta, a = u[:, 0], u[:, 1]

        # 应用Bicycle Model的微分方程
        dxdt = v * torch.cos(psi)
        dydt = v * torch.sin(psi)
        dpsidt = v / L * torch.tan(delta)
        dvdt = a

        # 将导数打包成一个张量
        dzdt = torch.stack([dxdt, dydt, dpsidt, dvdt], dim=1)

        return dzdt

# 2. 推理网络 (Encoder)
class EncoderNet(nn.Module):
    """
    将图像序列编码为初始状态和参数的均值和方差。
    """

    def __init__(self, latent_dim, theta_dim, image_size):
        super(EncoderNet, self).__init__()

        # 适应donkey数据集的图像尺寸 (120x160) - 灰度图
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 120x160 -> 60x80
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

        # RNN处理序列
        self.rnn = nn.LSTM(input_size=conv_output_dim, hidden_size=512, num_layers=2, batch_first=True, dropout=0.1)

        # 输出层，用于推断z0和theta的均值和方差
        self.fc_z0_mu = nn.Linear(512, latent_dim)
        self.fc_z0_logvar = nn.Linear(512, latent_dim)
        self.fc_theta_mu = nn.Linear(512, theta_dim)
        self.fc_theta_logvar = nn.Linear(512, theta_dim)

    def forward(self, x_seq):
        batch_size, seq_len, C, H, W = x_seq.shape
        x_reshaped = x_seq.view(batch_size * seq_len, C, H, W)

        features = self.conv_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, -1)

        rnn_output, _ = self.rnn(features)
        final_state = rnn_output[:, -1, :]

        mu_z0 = self.fc_z0_mu(final_state)
        logvar_z0 = self.fc_z0_logvar(final_state)
        mu_theta = self.fc_theta_mu(final_state)
        logvar_theta = self.fc_theta_logvar(final_state)

        return mu_z0, logvar_z0, mu_theta, logvar_theta

# 3. 发射网络 (Decoder)
class DecoderNet(nn.Module):
    """
    将潜在状态序列解码为图像序列。
    """

    def __init__(self, latent_dim, image_size):
        super(DecoderNet, self).__init__()
        self.image_size = image_size

        # 改进的解码器
        self.fc = nn.Linear(latent_dim, 256 * 7 * 10)
        
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1,0)),  # 7x10 -> 15x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 15x20 -> 30x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 30x40 -> 60x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 60x80 -> 120x160
            nn.Sigmoid()
        )

    def forward(self, z_seq):
        batch_size, seq_len, latent_dim = z_seq.shape
        z_reshaped = z_seq.view(batch_size * seq_len, latent_dim)

        # 全连接层
        x = self.fc(z_reshaped)
        x = x.view(batch_size * seq_len, 256, 7, 10)
        
        # 转置卷积
        x_hat = self.conv_decoder(x)
        x_hat = x_hat.view(batch_size, seq_len, 1, self.image_size[0], self.image_size[1])

        return x_hat

# 4. GOKU-net 核心模型
class GOKUnet(nn.Module):
    def __init__(self, encoder_net, decoder_net, ode_func, initial_z_dim, theta_dim, image_size):
        super(GOKUnet, self).__init__()

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ode_func = ode_func
        self.initial_z_dim = initial_z_dim
        self.theta_dim = theta_dim

        # Grounding networks
        self.z0_grounding = nn.Sequential(
            nn.Linear(initial_z_dim, initial_z_dim),
        )
        
        self.theta_grounding = nn.Sequential(
            nn.Linear(theta_dim, theta_dim),
            nn.Softplus()  # 确保轴距为正
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, t_points, actions_seq):
        batch_size = x_seq.size(0)
        
        # 创建控制输入函数
        def get_u_func(t):
            # 简化：使用最近邻插值获取控制输入
            if torch.is_tensor(t):
                if t.dim() == 0:  # 标量张量
                    t_val = t.item()
                elif t.numel() == 1:  # 单元素张量
                    t_val = t.item()
                else:
                    t_val = t.flatten()[0].item()
            else:
                t_val = float(t)
            
            # 检查并修复actions_seq维度
            current_actions = actions_seq
            if current_actions.dim() == 2:
                # 如果是2维，扩展为3维
                if current_actions.shape[1] == 2:
                    seq_len_target = len(t_points)
                    current_actions = current_actions.unsqueeze(1).repeat(1, seq_len_target, 1)
                else:
                    raise ValueError(f"无法处理的actions维度: {current_actions.shape}")
            
            # 计算时间索引
            seq_len_actual = current_actions.size(1)
            normalized_t = (t_val - t_points[0].item()) / (t_points[-1].item() - t_points[0].item())
            time_index = int(torch.clamp(
                torch.round(torch.tensor(normalized_t * (seq_len_actual - 1))),
                0, seq_len_actual - 1
            ).item())
            
            return current_actions[:, time_index, :]  # shape: (batch_size, 2)

        # 1. 推理阶段
        mu_z0, logvar_z0, mu_theta, logvar_theta = self.encoder_net(x_seq)

        # 重参数化
        z0_hat = self.z0_grounding(self.reparameterize(mu_z0, logvar_z0))
        theta_hat = self.theta_grounding(self.reparameterize(mu_theta, logvar_theta))

        # 2. ODE求解阶段
        def ode_func_with_params(t, z):
            return self.ode_func(t, z, theta_hat, get_u_func)

        try:
            z_hat_seq = odeint(
                ode_func_with_params,
                z0_hat,
                t_points,
                method='rk4',
                options={'step_size': 0.01}
            )
            z_hat_seq = z_hat_seq.permute(1, 0, 2)
        except Exception as e:
            # 如果ODE求解失败，使用简单的线性预测
            print(f"ODE求解失败，使用线性预测: {e}")
            seq_len = len(t_points)
            z_hat_seq = z0_hat.unsqueeze(1).repeat(1, seq_len, 1)

        # 3. 重构阶段
        x_hat_seq = self.decoder_net(z_hat_seq)

        return x_hat_seq, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat_seq

    def predict_sequence(self, x_init, predict_steps=30):
        """
        给定初始图像序列，预测未来序列
        """
        self.eval()
        batch_size = x_init.size(0)
        
        with torch.no_grad():
            # 编码初始状态和参数
            mu_z0, _, mu_theta, _ = self.encoder_net(x_init)
            z0_pred = self.z0_grounding(mu_z0)
            theta_pred = self.theta_grounding(mu_theta)
            
            # 预测时间点
            pred_t_points = torch.linspace(0, 1, predict_steps, device=x_init.device)
            
            # 简化的控制输入（零输入或随机输入）
            dummy_actions = torch.zeros(batch_size, predict_steps, 2, device=x_init.device)
            
            def get_u_func_pred(t):
                if torch.is_tensor(t):
                    if t.dim() == 0:  # 标量张量
                        t_val = t.item()
                    elif t.numel() == 1:  # 单元素张量
                        t_val = t.item()
                    else:
                        t_val = t.flatten()[0].item()
                else:
                    t_val = float(t)
                
                # 检查并修复dummy_actions维度
                current_actions = dummy_actions
                if current_actions.dim() == 2:
                    # 如果是2维，扩展为3维
                    if current_actions.shape[1] == 2:
                        seq_len_target = len(pred_t_points)
                        current_actions = current_actions.unsqueeze(1).repeat(1, seq_len_target, 1)
                    else:
                        raise ValueError(f"无法处理的dummy_actions维度: {current_actions.shape}")
                
                # 计算时间索引
                seq_len_actual = current_actions.size(1)
                normalized_t = t_val / pred_t_points[-1].item() if pred_t_points[-1].item() != 0 else 0
                time_index = int(torch.clamp(
                    torch.round(torch.tensor(normalized_t * (seq_len_actual - 1))),
                    0, seq_len_actual - 1
                ).item())
                
                return current_actions[:, time_index, :]  # shape: (batch_size, 2)
            
            # ODE求解
            def ode_func_with_params_pred(t, z):
                return self.ode_func(t, z, theta_pred, get_u_func_pred)
            
            try:
                z_pred_seq = odeint(
                    ode_func_with_params_pred,
                    z0_pred,
                    pred_t_points,
                    method='rk4',
                    options={'step_size': 0.01}
                )
                z_pred_seq = z_pred_seq.permute(1, 0, 2)
            except:
                # 回退到简单预测
                z_pred_seq = z0_pred.unsqueeze(1).repeat(1, predict_steps, 1)
            
            # 解码为图像
            x_pred_seq = self.decoder_net(z_pred_seq)
            
        return z_pred_seq, x_pred_seq

# 损失函数
def gokunet_loss(x_hat, x_true, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat_seq, states_true, 
                alpha=1.0, beta=0.01, gamma=1.0, delta=0.1):
    """
    GOKU-net总损失函数
    """
    # 1. 重建损失
    recon_loss = F.mse_loss(x_hat, x_true, reduction='sum')
    
    # 2. KL损失
    kl_z0 = -0.5 * torch.sum(1 + logvar_z0 - mu_z0.pow(2) - logvar_z0.exp())
    kl_theta = -0.5 * torch.sum(1 + logvar_theta - mu_theta.pow(2) - logvar_theta.exp())
    kl_loss = kl_z0 + kl_theta
    
    # 3. 状态监督损失
    state_loss = F.mse_loss(z_hat_seq, states_true, reduction='sum')
    
    # 4. 物理参数正则化
    theta_target = torch.full_like(mu_theta, 2.5)  # 假设轴距为2.5
    physics_loss = F.mse_loss(mu_theta, theta_target, reduction='sum')
    
    total_loss = alpha * recon_loss + beta * kl_loss + gamma * state_loss + delta * physics_loss
    
    return total_loss, recon_loss, kl_loss, state_loss, physics_loss

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

# 评估函数
def evaluate_model(model, test_loader, device, logger):
    model.eval()
    
    total_recon_loss = 0
    total_state_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, states, actions in tqdm(test_loader, desc="评估中"):
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            
            # 创建时间点
            seq_len = images.size(1)
            t_points = torch.linspace(0, 1, seq_len, device=device)
            
            x_hat, mu_z0, logvar_z0, mu_theta, logvar_theta, z_hat = model(images, t_points, actions)
            
            # 计算损失
            recon_loss = F.mse_loss(x_hat, images, reduction='sum').item()
            state_loss = F.mse_loss(z_hat, states, reduction='sum').item()
            
            total_recon_loss += recon_loss
            total_state_loss += state_loss
            num_samples += images.size(0)
    
    avg_recon_loss = total_recon_loss / num_samples
    avg_state_loss = total_state_loss / num_samples
    
    return avg_recon_loss, avg_state_loss

# 30步预测评估函数
def evaluate_30step_prediction(model, test_loader, device, logger, output_dir):
    """评估30步预测性能"""
    model.eval()
    
    all_step_rmse = []
    first_sequence_data = None
    
    with torch.no_grad():
        for batch_idx, (images, states, actions, file_names) in enumerate(tqdm(test_loader, desc="30步预测评估")):
            images, states = images.to(device), states.to(device)
            B = images.size(0)
            
            for sample_idx in range(B):
                sample_images = images[sample_idx:sample_idx+1]
                sample_states = states[sample_idx:sample_idx+1]
                
                # 使用前5帧进行预测
                init_images = sample_images[:, :5]  # (1, 5, C, H, W)
                ground_truth_states = sample_states[:, 5:]  # (1, 25, state_dim)
                
                # 预测未来25步
                pred_states, pred_images = model.predict_sequence(init_images, predict_steps=25)
                
                # 计算每步RMSE
                step_rmse = []
                for step in range(25):
                    pred_state_step = pred_states[:, step]
                    true_state_step = ground_truth_states[:, step]
                    
                    mse = F.mse_loss(pred_state_step, true_state_step, reduction='mean')
                    rmse = torch.sqrt(mse).item()
                    step_rmse.append(rmse)
                
                all_step_rmse.append(step_rmse)
                
                # 保存第一个序列的数据
                if batch_idx == 0 and sample_idx == 0:
                    first_sequence_data = {
                        'init_images': init_images.cpu(),
                        'pred_states': pred_states.cpu(),
                        'pred_images': pred_images.cpu(),
                        'true_states': ground_truth_states.cpu(),
                        'file_name': file_names[sample_idx]
                    }
                
                if batch_idx >= 10:
                    break
            
            if batch_idx >= 10:
                break
    
    # 计算平均RMSE
    all_step_rmse = np.array(all_step_rmse)
    mean_step_rmse = np.mean(all_step_rmse, axis=0)
    
    logger.info("30步预测评估结果:")
    for step, rmse in enumerate(mean_step_rmse[:10]):  # 只显示前10步
        logger.info(f"  步骤{step+1:2d}: {rmse:.6f}")
    
    total_30step_rmse = np.mean(mean_step_rmse)
    logger.info(f"30步预测总平均RMSE: {total_30step_rmse:.6f}")
    
    return mean_step_rmse.tolist(), total_30step_rmse

# 可视化函数
def visualize_reconstruction(model, test_loader, device, save_path, logger):
    model.eval()
    
    data_iter = iter(test_loader)
    images, states, actions = next(data_iter)
    images, states, actions = images.to(device), states.to(device), actions.to(device)
    
    with torch.no_grad():
        seq_len = images.size(1)
        t_points = torch.linspace(0, 1, seq_len, device=device)
        x_hat, _, _, _, _, _ = model(images, t_points, actions)
    
    # 选择前4个样本展示
    n_samples = min(4, images.size(0))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
    
    for i in range(n_samples):
        # 原图（第一个时间步）
        original = images[i, 0, 0].cpu().numpy()  # 灰度图
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original (t=0)', fontsize=10)
        
        # 重构图（第一个时间步）
        reconstructed = x_hat[i, 0, 0].cpu().numpy()
        axes[1, i].imshow(reconstructed, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed (t=0)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"重构可视化已保存到: {save_path}")

# 训练函数
def train_gokunet(fold_idx, noise_type, epochs=100, batch_size=8, learning_rate=1e-4, seq_len=10):
    # 创建输出目录
    output_dir = f'gokunet_output_fold{fold_idx}_{noise_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logging(log_file)
    
    # 记录训练参数
    params = {
        'fold_idx': fold_idx,
        'noise_type': noise_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seq_len': seq_len,
        'model': 'GOKU-net',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info("="*60)
    logger.info(f"开始GOKU-net训练 - Fold {fold_idx}, 噪声类型: {noise_type}")
    logger.info(f"训练参数: {json.dumps(params, indent=2)}")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据路径
    train_dirs = []
    val_dir = f'../donkeynew_fold{fold_idx}'
    
    for i in range(1, 6):
        if i != fold_idx:
            train_dirs.append(f'../donkeynew_fold{i}')
    
    logger.info(f"验证集目录: {val_dir}")
    logger.info(f"训练集目录: {train_dirs}")
    
    # 加载数据
    try:
        train_dataset = DonkeyDataset(train_dirs, noise_type, seq_len)
        val_dataset = DonkeyDataset(val_dir, noise_type, seq_len)
        test_dataset = DonkeyTestDataset(val_dir, noise_type, test_seq_len=30)
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 模型参数
    image_size = (120, 160)
    initial_z_dim = 4  # [x, y, heading, speed]
    theta_dim = 1  # [wheelbase]
    
    # 初始化模型
    encoder_net = EncoderNet(initial_z_dim, theta_dim, image_size)
    decoder_net = DecoderNet(initial_z_dim, image_size)
    bicycle_ode = BicycleODE()
    model = GOKUnet(encoder_net, decoder_net, bicycle_ode, initial_z_dim, theta_dim, image_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
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
        train_state_loss = 0
        train_physics_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (images, states, actions) in enumerate(train_bar):
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            
            # 创建时间点
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
                logger.warning(f"训练批次 {batch_idx} 失败: {e}")
                continue
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': loss.item() / (images.size(0) * images.size(1)),
                'recon': recon_loss.item() / (images.size(0) * images.size(1))
            })
        
        # 验证阶段
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
        
        # 记录损失
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            avg_train_loss = train_loss / len(train_dataset)
            avg_val_loss = val_loss / len(val_dataset)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 记录详细损失
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
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
            
            # 记录日志
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
            
            # 早停检查
            if early_stopping(avg_val_loss):
                logger.info(f"早停触发！在epoch {epoch+1}停止训练")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"加载最佳模型 (epoch {best_epoch})")
    
    logger.info("="*60)
    logger.info("训练完成！开始评估...")
    logger.info("="*60)
    
    # 评估
    avg_recon_loss, avg_state_loss = evaluate_model(model, val_loader, device, logger)
    
    logger.info("评估结果:")
    logger.info(f"  - 重构损失: {avg_recon_loss:.6f}")
    logger.info(f"  - 状态损失: {avg_state_loss:.6f}")
    
    # 30步预测评估
    step_rmse_list, total_30step_rmse = evaluate_30step_prediction(model, test_loader, device, logger, output_dir)
    
    # 可视化重构结果
    vis_path = os.path.join(output_dir, 'reconstruction.png')
    visualize_reconstruction(model, val_loader, device, vis_path, logger)
    
    # 保存模型
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存为: {model_path}")
    
    # 保存训练结果
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
    
    # 保存结果JSON
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存为: {results_path}")
    
    # 保存参数JSON
    params_path = os.path.join(output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"参数已保存为: {params_path}")
    
    logger.info("="*60)
    logger.info(f"GOKU-net Fold {fold_idx}, 噪声类型 {noise_type} 训练完成!")
    logger.info("="*60)
    
    return model, results

if __name__ == "__main__":
    # 训练参数
    FOLD_IDX = 4
    NOISE_TYPE = 'states'
    EPOCHS = 80
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    SEQ_LEN = 10
    
    print("="*60)
    print(f"开始训练 GOKU-net - Fold {FOLD_IDX}, 噪声类型: {NOISE_TYPE}")
    print("="*60)
    
    # 开始训练
    model, results = train_gokunet(FOLD_IDX, NOISE_TYPE, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEQ_LEN)
    
    print("="*60)
    print(f"GOKU-net Fold {FOLD_IDX}, 噪声类型 {NOISE_TYPE} 训练完成!")
    print(f"结果保存在: gokunet_output_fold{FOLD_IDX}_{NOISE_TYPE}/")
    print("="*60)
