import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# --- 1. 模块定义 ---
# 将模型中的各个组件封装为独立的 PyTorch 模块，提高代码可读性。

class Encoder(nn.Module):
    """
    VRNN编码器：将图像和状态信息编码成潜在表示。
    输入: 图像 (B, C, H, W) 和状态 (B, state_dim)
    输出: 潜在空间的均值 (mu) 和对数方差 (logvar)
    """

    def __init__(self, image_channels, state_dim, hidden_dim, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_img = nn.Linear(64 * 16 * 16, hidden_dim)  # 假设输入图像为 64x64
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
    """
    VRNN解码器：将潜在表示和RNN隐藏状态解码回图像。
    输入: 潜在表示 (z) 和隐藏状态 (rnn_h)
    输出: 重建图像
    """

    def __init__(self, z_dim, hidden_dim, image_channels):
        super().__init__()
        self.fc = nn.Linear(z_dim + hidden_dim, 64 * 16 * 16)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, rnn_h):
        z_combined = torch.cat([z, rnn_h], dim=1)
        x = self.fc(z_combined).view(-1, 64, 16, 16)
        return self.conv_trans(x)


class Prior(nn.Module):
    """
    先验网络：基于RNN的隐藏状态预测潜在表示的先验分布。
    输入: RNN隐藏状态 (rnn_h)
    输出: 先验分布的均值 (mu) 和对数方差 (logvar)
    """

    def __init__(self, hidden_dim, z_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, rnn_h):
        mu = self.fc_mu(rnn_h)
        logvar = self.fc_logvar(rnn_h)
        return mu, logvar


class VRNNCar(nn.Module):
    """
    完整的VRNN模型，用于车辆动力学参数推断。
    它包含编码器、解码器、RNN单元和物理参数预测层。
    """

    def __init__(self, image_channels, state_dim, hidden_dim, z_dim, l_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.l_dim = l_dim

        self.encoder = Encoder(image_channels, state_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, image_channels)
        self.rnn = nn.GRU(z_dim, hidden_dim)  # 使用GRU作为循环单元
        self.prior = Prior(hidden_dim, z_dim)

        # 物理参数 L (轴距) 的映射层
        self.fc_l_mu = nn.Linear(z_dim, l_dim)
        self.fc_l_logvar = nn.Linear(z_dim, l_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧，用于从潜在分布中采样。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, s_seq, h_0=None):
        """
        前向传播：处理一个完整的序列。
        输入:
            x_seq (T, B, C, H, W): 图像序列
            s_seq (T, B, S_dim): 状态序列
            h_0 (1, B, H_dim): 初始RNN隐藏状态
        输出:
            x_rec_seq (T, B, C, H, W): 重建图像序列
            mu_l_seq, logvar_l_seq (T, B, 1): 轴距L的预测分布
            mu_enc_seq, logvar_enc_seq (T, B, z_dim): 编码器潜在空间的分布
            mu_prior_seq, logvar_prior_seq (T, B, z_dim): 先验分布
        """
        T, B, _, _, _ = x_seq.size()

        # 初始化存储列表
        x_rec_seq, mu_l_seq, logvar_l_seq = [], [], []
        mu_enc_seq, logvar_enc_seq = [], []
        mu_prior_seq, logvar_prior_seq = [], []

        # 初始化RNN隐藏状态
        h_t = h_0 if h_0 is not None else torch.zeros(1, B, self.hidden_dim, device=x_seq.device)

        for t in range(T):
            # 获取当前时间步的数据
            x_t = x_seq[t]
            s_t = s_seq[t]

            # 1. 先验预测：基于上一步的隐藏状态预测当前潜在空间的先验分布
            mu_prior_t, logvar_prior_t = self.prior(h_t.squeeze(0))

            # 2. 编码器推断：基于当前图像和状态推断潜在分布
            mu_enc_t, logvar_enc_t = self.encoder(x_t, s_t)
            z_t = self.reparameterize(mu_enc_t, logvar_enc_t)

            # 3. 物理参数预测：从潜在表示中推断轴距 L
            mu_l_t = self.fc_l_mu(z_t)
            logvar_l_t = self.fc_l_logvar(z_t)

            # 4. 解码器生成：从潜在表示和隐藏状态重建图像
            x_rec_t = self.decoder(z_t, h_t.squeeze(0))

            # 5. RNN更新：使用推断出的 z 更新隐藏状态
            _, h_t = self.rnn(z_t.unsqueeze(0), h_t)

            # 存储结果
            x_rec_seq.append(x_rec_t)
            mu_l_seq.append(mu_l_t)
            logvar_l_seq.append(logvar_l_t)
            mu_enc_seq.append(mu_enc_t)
            logvar_enc_seq.append(logvar_enc_t)
            mu_prior_seq.append(mu_prior_t)
            logvar_prior_seq.append(logvar_prior_t)

        return (torch.stack(x_rec_seq), torch.stack(mu_l_seq), torch.stack(logvar_l_seq),
                torch.stack(mu_enc_seq), torch.stack(logvar_enc_seq),
                torch.stack(mu_prior_seq), torch.stack(logvar_prior_seq))


# --- 2. 损失函数和数据生成 ---

def car_dynamics_loss(mu_l, logvar_l, s_seq, s_seq_next, delta_seq, dt):
    """
    计算基于自行车模型的物理损失 (L_物理)。
    使用高斯负对数似然，假设我们有真实的轴距L标签。
    """
    l_pred_dist = torch.distributions.Normal(mu_l, torch.exp(0.5 * logvar_l))
    # 注意：这里需要真实的L标签作为监督。由于我们没有，代码中用一个虚拟值。
    # 在实际应用中，你需要从数据集中获取真实的L值。
    # 为了演示，我们假设真实 L = 1.0。
    l_gt = torch.ones_like(mu_l) * 1.0

    # 计算物理损失（高斯负对数似然）
    loss = -l_pred_dist.log_prob(l_gt).mean()

    return loss


def custom_loss(x_rec, x_orig, mu_enc, logvar_enc, mu_prior, logvar_prior, mu_l, logvar_l, s_seq, s_seq_next, delta_seq,
                dt, alpha, beta, gamma):
    """
    总损失函数，结合重建、KL和物理损失。
    """
    # 1. 重建损失
    # 比较重建图像和原始图像
    loss_reconstruction = F.mse_loss(x_rec, x_orig)

    # 2. KL 损失
    # 比较编码器推断的分布与先验分布
    loss_kl = -0.5 * torch.sum(
        1 + logvar_enc - logvar_prior - (mu_enc - mu_prior) ** 2 / torch.exp(logvar_prior) - torch.exp(
            logvar_enc) / torch.exp(logvar_prior))

    # 3. 物理损失
    loss_physical = car_dynamics_loss(mu_l, logvar_l, s_seq, s_seq_next, delta_seq, dt)

    return (alpha * loss_reconstruction,
            beta * loss_kl,
            gamma * loss_physical,
            alpha * loss_reconstruction + beta * loss_kl + gamma * loss_physical)


class DummyCarDataset(Dataset):
    """
    虚拟数据集类，用于演示和测试。
    生成随机的图像、状态和转向角序列。
    """

    def __init__(self, num_sequences=100, seq_len=10):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.images = torch.randn(num_sequences, seq_len, 3, 64, 64)
        self.states = torch.randn(num_sequences, seq_len, 4)  # (x, y, theta, speed)
        self.deltas = torch.randn(num_sequences, seq_len, 1)  # 转向角
        self.l_label = torch.ones(num_sequences, 1)  # 假设真实轴距 L=1.0

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # 返回一个完整的序列
        return (self.images[idx],
                self.states[idx],
                torch.cat([self.states[idx, 1:], self.states[idx, -1:]], dim=0),  # 下一时刻状态
                self.deltas[idx],
                self.l_label[idx])


# --- 3. 训练循环 ---

def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    ALPHA = 1.0  # 重建损失权重
    BETA = 0.01  # KL 损失权重 (通常较小，防止潜在空间坍缩)
    GAMMA = 10.0  # 物理损失权重

    # 模型参数
    IMAGE_CHANNELS = 3
    STATE_DIM = 4
    HIDDEN_DIM = 256
    Z_DIM = 128
    L_DIM = 1
    SEQ_LEN = 10
    BATCH_SIZE = 32
    EPOCHS = 10

    # 数据集和加载器
    dataset = DummyCarDataset(seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型、优化器和时间步长
    model = VRNNCar(IMAGE_CHANNELS, STATE_DIM, HIDDEN_DIM, Z_DIM, L_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dt = 1.0  # 假设时间步长为1秒

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss, total_kl_loss, total_phys_loss, total_loss = 0, 0, 0, 0

        for batch_images, batch_states, batch_next_states, batch_deltas, batch_l_labels in dataloader:
            # 移动数据到指定设备
            batch_images = batch_images.permute(1, 0, 2, 3, 4).to(device)  # 调整为 (T, B, C, H, W)
            batch_states = batch_states.permute(1, 0, 2).to(device)  # (T, B, S_dim)
            batch_next_states = batch_next_states.permute(1, 0, 2).to(device)
            batch_deltas = batch_deltas.permute(1, 0, 2).to(device)

            optimizer.zero_grad()

            # 前向传播
            (x_rec, mu_l, logvar_l,
             mu_enc, logvar_enc,
             mu_prior, logvar_prior) = model(batch_images, batch_states)

            # 计算总损失
            recon_loss, kl_loss, phys_loss, loss = custom_loss(
                x_rec, batch_images, mu_enc, logvar_enc, mu_prior, logvar_prior,
                mu_l, logvar_l, batch_states, batch_next_states, batch_deltas, dt,
                ALPHA, BETA, GAMMA
            )

            # 反向传播
            loss.backward()
            optimizer.step()

            # 累加损失
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_phys_loss += phys_loss.item()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"  Total Loss: {total_loss / len(dataloader):.4f}")
        print(f"  Recon Loss: {total_recon_loss / len(dataloader):.4f}")
        print(f"  KL Loss:    {total_kl_loss / len(dataloader):.4f}")
        print(f"  Phys Loss:  {total_phys_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    main()