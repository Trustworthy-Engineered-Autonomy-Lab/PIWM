import os, sys

# Path of this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = go up until you reach world_model_2
# If analyze_joint_knn.py is in external/piwm-donkeycar/, then root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, PROJECT_ROOT)
print("[analyze] PROJECT_ROOT =", PROJECT_ROOT)

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ---- import your models ----
from vae_training.models.vae_fixed import VAE
from vae_training.models.predictor import LSTMPredictor
from vae_training.models.extractor import StateExtractor

# ---------------------------
# CONFIG (edit these)
# ---------------------------
NPZ_PATH   = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"
CKPT_PATH  = r"C:\Users\liamm\world_model_2\joint_training\results\2026-03-02_15-59-08\joint_best.pt"

IMG_KEY   = "frame"
ACT_KEY   = "action"
STATE_KEY = "state"

CTX_LEN     = 14
STRIDE_STEP = 5
VAL_RATIO   = 0.2
BATCH_SIZE  = 64
ACTION_PAD_TO = 3

RESIZE_TO = (120, 160)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset (same as training, minimal)
# ---------------------------
class NPZJointRelativeDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, ctx_len, stride_step, resize_to, action_pad_to=None):
        self.ctx_len = int(ctx_len)
        self.stride_step = int(stride_step)
        self.resize_to = resize_to
        self.action_pad_to = action_pad_to

        d = np.load(npz_path, allow_pickle=True)
        raw = d["data"]
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            raw = raw.item()

        self.frames  = raw[IMG_KEY]
        self.actions = raw[ACT_KEY].astype(np.float32)
        self.state   = raw[STATE_KEY].astype(np.float32)

        self.T = int(self.frames.shape[0])
        needed = self.ctx_len + 1
        self.t0s = list(range(0, self.T - needed + 1, self.stride_step))

    def __len__(self):
        return len(self.t0s)

    def _to_chw_float01(self, img):
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        # HWC -> CHW if needed
        if img.ndim == 3 and img.shape[-1] in (1,3,4):
            img = np.transpose(img, (2,0,1))
        x = torch.from_numpy(img)  # (C,H,W)

        if self.resize_to is not None:
            H, W = self.resize_to
            x = F.interpolate(x.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False).squeeze(0)
        return x

    @staticmethod
    def _global_to_rel_xy(xt, yt, x0, y0, yaw0_deg):
        dx = float(xt - x0)
        dy = float(yt - y0)
        yaw0 = math.radians(float(yaw0_deg))
        c = math.cos(yaw0)
        s = math.sin(yaw0)
        xr =  c * dx + s * dy
        yr = -s * dx + c * dy
        return xr, yr

    def __getitem__(self, i):
        t0 = self.t0s[i]
        t1 = t0 + self.ctx_len

        x_ctx_np = self.frames[t0:t1]
        x_tgt_np = self.frames[t1]
        x_ctx = torch.stack([self._to_chw_float01(x_ctx_np[j]) for j in range(self.ctx_len)], dim=0)
        x_tgt = self._to_chw_float01(x_tgt_np)

        a_ctx = self.actions[t0:t1]
        if self.action_pad_to is not None and a_ctx.shape[1] < self.action_pad_to:
            pad = self.action_pad_to - a_ctx.shape[1]
            a_ctx = np.pad(a_ctx, ((0,0),(0,pad)), mode="constant", constant_values=0.0)
        a_ctx = torch.from_numpy(a_ctx)

        x0, y0, yaw0 = self.state[t0,0], self.state[t0,1], self.state[t0,2]
        xt, yt       = self.state[t1,0], self.state[t1,1]
        xr, yr = self._global_to_rel_xy(xt, yt, x0, y0, yaw0)
        xy = torch.tensor([xr, yr], dtype=torch.float32)

        # Extra metadata for similarity features
        v0 = float(self.state[t0,3]) if self.state.shape[1] > 3 else 0.0
        return x_ctx, a_ctx, x_tgt, xy, t0, v0, float(yaw0)

# ---------------------------
# Helpers
# ---------------------------
@torch.no_grad()
def run_split(loader, vae, predictor, extractor):
    all_feats = []
    all_err = []
    all_t0 = []

    for x_ctx, a_ctx, x_tgt, xy_true, t0, v0, yaw0 in loader:
        x_ctx = x_ctx.to(DEVICE)      # (B,T,C,H,W)
        a_ctx = a_ctx.to(DEVICE)      # (B,T,A)
        x_tgt = x_tgt.to(DEVICE)      # (B,C,H,W)
        xy_true = xy_true.to(DEVICE)  # (B,2)

        B, T, C, H, W = x_ctx.shape
        x_ctx_flat = x_ctx.view(B*T, C, H, W)

        _, mu_ctx, _ = vae(x_ctx_flat)
        z_ctx = mu_ctx.view(B, T, -1)

        _, mu_tgt, _ = vae(x_tgt)
        z_pred, _ = predictor.forward_last(z_ctx, a_ctx)
        xy_pred = extractor(z_pred)

        # error metric (same as your RMSE per-sample)
        per = torch.sqrt(torch.sum((xy_pred - xy_true)**2, dim=1))  # (B,)
        all_err.append(per.detach().cpu().numpy())
        all_t0.append(t0.numpy())

        # similarity feature vector per window:
        # - mean & std of actions
        # - speed v0
        # - yaw0 (wrapped)
        # - z_ctx mean (compressed)
        a_mean = a_ctx.mean(dim=1).detach().cpu().numpy()
        a_std  = a_ctx.std(dim=1).detach().cpu().numpy()
        v0 = v0.numpy().reshape(-1,1)
        # yaw as sin/cos to avoid wrap issues
        yaw = yaw0.numpy()
        yaw_s = np.sin(np.deg2rad(yaw)).reshape(-1,1)
        yaw_c = np.cos(np.deg2rad(yaw)).reshape(-1,1)

        # compress z_ctx mean to a few dims (first 8) to keep kNN cheap
        z_mean = z_ctx.mean(dim=1)[:, :8].detach().cpu().numpy()

        feats = np.concatenate([a_mean, a_std, v0, yaw_s, yaw_c, z_mean], axis=1)
        all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)
    all_err   = np.concatenate(all_err, axis=0)
    all_t0    = np.concatenate(all_t0, axis=0)
    return all_feats, all_err, all_t0

def knn_indices(X_train, x, k=10):
    # brute force L2 (fine for moderate N)
    d = np.sum((X_train - x[None,:])**2, axis=1)
    idx = np.argsort(d)[:k]
    return idx, d[idx]

# ---------------------------
# Main
# ---------------------------
def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    vae_cfg = ckpt.get("vae_cfg", {})
    latent_dim = int(vae_cfg.get("latent_dim", 64))
    input_channels = int(vae_cfg.get("input_channels", 3))

    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(DEVICE)
    predictor = LSTMPredictor(latent_dim=latent_dim, action_dim=ACTION_PAD_TO, hidden_dim=128, num_layers=2).to(DEVICE)
    extractor = StateExtractor(latent_dim=latent_dim, output_dim=2, hidden_dims=[128,64]).to(DEVICE)

    vae.load_state_dict(ckpt["vae"])
    predictor.load_state_dict(ckpt["predictor"])
    extractor.load_state_dict(ckpt["extractor"])

    vae.eval(); predictor.eval(); extractor.eval()

    ds = NPZJointRelativeDataset(NPZ_PATH, CTX_LEN, STRIDE_STEP, RESIZE_TO, action_pad_to=ACTION_PAD_TO)
    val_len = int(VAL_RATIO * len(ds))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    Xtr, Etr, T0tr = run_split(train_ld, vae, predictor, extractor)
    Xva, Eva, T0va = run_split(val_ld,   vae, predictor, extractor)

    print(f"Train windows: {len(Etr)} | Val windows: {len(Eva)}")
    print(f"Train error: mean={Etr.mean():.4f}  p90={np.quantile(Etr,0.9):.4f}  max={Etr.max():.4f}")
    print(f"Val   error: mean={Eva.mean():.4f}  p90={np.quantile(Eva,0.9):.4f}  max={Eva.max():.4f}")

    # take worst 20 val points and see if they have similar bad points in train
    worst_idx = np.argsort(Eva)[-20:][::-1]

    print("\n=== Worst val points: kNN train neighbor check ===")
    for j in worst_idx:
        x = Xva[j]
        idx, dist = knn_indices(Xtr, x, k=10)
        neigh_err = Etr[idx]
        print(
            f"val_t0={int(T0va[j])}  val_err={Eva[j]:.4f} | "
            f"train_neigh_err mean={neigh_err.mean():.4f} min={neigh_err.min():.4f} max={neigh_err.max():.4f}"
        )

    # quick diagnostic conclusion
    # if worst val points have neighbors with similarly high error -> not generalization
    # if worst val points have neighbors with low error -> generalization problem
    ratio_bad = 0
    for j in worst_idx:
        idx,_ = knn_indices(Xtr, Xva[j], k=10)
        if Etr[idx].mean() > np.quantile(Etr, 0.9):
            ratio_bad += 1
    print(f"\nWorst-val points whose train-neighbors are also 'bad' (mean > train p90): {ratio_bad}/20")

if __name__ == "__main__":
    main()