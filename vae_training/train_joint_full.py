# train_joint_full.py
# Jointly trains: VAE + LSTMPredictor + StateExtractor
#
# New goal:
#   - use a 20-frame window
#   - predictor input: first 15 frames
#   - predictor target: next 5 frames
#   - extractor predicts RELATIVE [x_rel, y_rel] for each of the 5 target steps
#   - relative is defined in the window-origin frame (origin is at t0):
#       [dx, dy] = [x_t - x_t0, y_t - y_t0] rotated by -yaw_t0
#
# This keeps the same general setup as before, but changes:
#   - single-step target -> 5-step target
#   - one target action -> future action sequence
#   - one relative xy target -> 5 relative xy targets

import csv
import datetime
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ---- project imports ----
from models.vae_fixed import VAE
try:
    from models.vae_fixed import vae_loss as _vae_loss
except Exception:
    _vae_loss = None

from models.predictor import LSTMPredictor, predictor_loss
from models.extractor import StateExtractor


# -------------------------------------------------------
# Config
# -------------------------------------------------------
NPZ_PATH  = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_30_step5_wrapped.npz"
VAE_CKPT  = r"C:\Users\liamm\world_model_2\vae_training\results\npz\checkpoints\best.pth"
RUN_ROOT  = r"C:\Users\liamm\world_model_2\joint_training\results"

# Your npz['data'] keys
IMG_KEY   = "frame"    # (T,3,224,224) uint8
ACT_KEY   = "action"   # (T,2) float32
STATE_KEY = "state"    # (T,4) float32 => [x, y, yaw_deg, v] (assumed)

# Windowing
CTX_LEN       = 15
PRED_LEN      = 15
TOTAL_LEN     = 30
STRIDE_STEP   = 5
VAL_RATIO     = 0.2

# Image preproc
RESIZE_TO      = (120, 160)
INPUT_CHANNELS = 3

# Training
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 1e-4
WD         = 1e-5

# Loss weights
W_VAE = 0.001
W_Z   = 0.1
W_XY  = 1.0

DEFAULT_BETA = 1.0

# IMPORTANT: keep action_dim aligned with old training/viz if you want
# - If ACTION_PAD_TO = 3 and your action is (T,2), we add a 3rd zero channel.
# - If you want to truly use 2-d actions, set ACTION_PAD_TO = None.
ACTION_PAD_TO = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_DIR = Path(RUN_ROOT) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[train_joint] Saving logs/checkpoints to: {RUN_DIR}")


# -------------------------------------------------------
# Loss helper
# -------------------------------------------------------
def vae_loss_fallback(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl, recon_loss, kl


def vae_loss(recon, x, mu, logvar, beta=1.0):
    if _vae_loss is not None:
        return _vae_loss(recon, x, mu, logvar, beta=beta)
    return vae_loss_fallback(recon, x, mu, logvar, beta=beta)


@torch.no_grad()
def xy_rmse(xy_pred: torch.Tensor, xy_true: torch.Tensor):
    """
    xy_pred, xy_true: (..., 2)
    returns rmse over all leading dims
    """
    x_err = xy_pred[..., 0] - xy_true[..., 0]
    y_err = xy_pred[..., 1] - xy_true[..., 1]
    x_rmse = torch.sqrt(torch.mean(x_err ** 2))
    y_rmse = torch.sqrt(torch.mean(y_err ** 2))
    total_rmse = torch.sqrt(torch.mean(x_err ** 2 + y_err ** 2))
    return x_rmse, y_rmse, total_rmse


def write_metrics_csv(path: Path, history_rows):
    if not history_rows:
        return
    keys = list(history_rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history_rows:
            clean = {k: (float(v) if torch.is_tensor(v) else v) for k, v in row.items()}
            w.writerow(clean)


# -------------------------------------------------------
# Dataset: produces raw images + actions + 5-step RELATIVE xy target
# -------------------------------------------------------
class NPZJointRelativeDataset(Dataset):
    """
    Returns:
      x_ctx      : (CTX_LEN, C, H, W) float32 [0,1]
      a_ctx      : (CTX_LEN, A) float32
      x_tgt      : (PRED_LEN, C, H, W) float32 [0,1]
      a_tgt      : (PRED_LEN, A) float32
      xy_rel_tgt : (PRED_LEN, 2) float32

    Assumptions:
      state[t] = [x, y, yaw_deg, ...]
      yaw_deg is degrees in global frame.
    """

    def __init__(
        self,
        npz_path: str,
        ctx_len: int,
        pred_len: int,
        stride_step: int,
        resize_to=(120, 160),
        input_channels=3,
        action_pad_to=None,
    ):
        self.ctx_len = int(ctx_len)
        self.pred_len = int(pred_len)
        self.stride_step = int(stride_step)
        self.resize_to = resize_to
        self.input_channels = int(input_channels)
        self.action_pad_to = action_pad_to

        d = np.load(npz_path, allow_pickle=True)
        if list(d.files) != ["data"]:
            raise ValueError(f"Expected only ['data'] in NPZ, got {d.files}")

        raw = d["data"]
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            raw = raw.item()
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict inside npz['data'], got {type(raw)}")

        for k in (IMG_KEY, ACT_KEY, STATE_KEY):
            if k not in raw:
                raise ValueError(f"Missing key '{k}' in npz['data']. Have keys: {list(raw.keys())}")

        self.frames = raw[IMG_KEY]
        self.actions = raw[ACT_KEY]
        self.state = raw[STATE_KEY]

        if self.frames.ndim != 4:
            raise ValueError(f"frames expected (T,C,H,W) or (T,H,W,C), got {self.frames.shape}")
        if self.actions.ndim != 2:
            raise ValueError(f"actions expected (T,A), got {self.actions.shape}")
        if self.state.ndim != 2 or self.state.shape[1] < 3:
            raise ValueError(f"state expected (T,>=3), got {self.state.shape}")

        self.T = int(self.frames.shape[0])
        if self.actions.shape[0] != self.T or self.state.shape[0] != self.T:
            raise ValueError("Time length mismatch between frame/action/state arrays.")

        needed = self.ctx_len + self.pred_len
        self.t0s = list(range(0, self.T - needed + 1, self.stride_step))
        if len(self.t0s) == 0:
            raise ValueError(f"No windows available. T={self.T}, need >= {needed}.")

    def __len__(self):
        return len(self.t0s)

    def _to_chw_float01(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0

        if img.ndim != 3:
            raise ValueError(f"Expected 3D image, got {img.shape}")

        # convert HWC->CHW if needed
        if img.shape[-1] in (1, 3, 4):
            img = np.transpose(img, (2, 0, 1))

        # optional grayscale collapse
        if self.input_channels == 1 and img.shape[0] > 1:
            img = img.mean(axis=0, keepdims=True)

        x = torch.from_numpy(img)  # (C,H,W)
        if self.resize_to is not None:
            H, W = self.resize_to
            x = F.interpolate(
                x.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        return x

    @staticmethod
    def _global_to_rel_xy(x_t, y_t, x0, y0, yaw0_deg):
        dx = float(x_t - x0)
        dy = float(y_t - y0)

        yaw0 = math.radians(float(yaw0_deg))
        c = math.cos(yaw0)
        s = math.sin(yaw0)

        xr =  c * dx + s * dy
        yr = -s * dx + c * dy
        return xr, yr

    def __getitem__(self, i):
        t0 = self.t0s[i]
        t1 = t0 + self.ctx_len
        t2 = t1 + self.pred_len

        # images
        x_ctx_np = self.frames[t0:t1]   # (CTX_LEN,...)
        x_tgt_np = self.frames[t1:t2]   # (PRED_LEN,...)

        x_ctx = torch.stack(
            [self._to_chw_float01(x_ctx_np[j]) for j in range(self.ctx_len)],
            dim=0
        )
        x_tgt = torch.stack(
            [self._to_chw_float01(x_tgt_np[j]) for j in range(self.pred_len)],
            dim=0
        )

        # actions
        a_ctx = self.actions[t0:t1].astype(np.float32)
        a_tgt = self.actions[t1:t2].astype(np.float32)

        if self.action_pad_to is not None:
            if a_ctx.shape[1] < self.action_pad_to:
                pad = self.action_pad_to - a_ctx.shape[1]
                a_ctx = np.pad(a_ctx, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
            if a_tgt.shape[1] < self.action_pad_to:
                pad = self.action_pad_to - a_tgt.shape[1]
                a_tgt = np.pad(a_tgt, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

        a_ctx = torch.from_numpy(a_ctx)
        a_tgt = torch.from_numpy(a_tgt)

        # relative xy targets for each future step, all measured from origin at t0
        x0, y0, yaw0 = self.state[t0, 0], self.state[t0, 1], self.state[t0, 2]
        xy_list = []
        for tt in range(t1, t2):
            xt, yt = self.state[tt, 0], self.state[tt, 1]
            xr, yr = self._global_to_rel_xy(xt, yt, x0, y0, yaw0)
            xy_list.append([xr, yr])

        xy_rel_tgt = torch.tensor(xy_list, dtype=torch.float32)  # (PRED_LEN,2)

        return x_ctx, a_ctx, x_tgt, a_tgt, xy_rel_tgt


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # ---- load VAE ckpt (init weights, but NOT frozen) ----
    ckpt = torch.load(VAE_CKPT, map_location=DEVICE)
    vae_cfg = ckpt.get("config", {})
    latent_dim = int(vae_cfg.get("latent_dim", 64))
    input_channels = int(vae_cfg.get("input_channels", INPUT_CHANNELS))
    beta = float(vae_cfg.get("beta", DEFAULT_BETA))

    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(DEVICE)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.train()

    # ---- dataset ----
    ds_full = NPZJointRelativeDataset(
        NPZ_PATH,
        ctx_len=CTX_LEN,
        pred_len=PRED_LEN,
        stride_step=STRIDE_STEP,
        resize_to=RESIZE_TO,
        input_channels=input_channels,
        action_pad_to=ACTION_PAD_TO,
    )

    val_len = int(VAL_RATIO * len(ds_full))
    train_len = len(ds_full) - val_len
    train_ds, val_ds = random_split(ds_full, [train_len, val_len])

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # infer action_dim
    x_ctx0, a_ctx0, x_tgt0, a_tgt0, xy0 = ds_full[0]
    action_dim = int(a_ctx0.shape[-1])

    print(
        f"[dataset] sample shapes: "
        f"x_ctx={tuple(x_ctx0.shape)}, "
        f"a_ctx={tuple(a_ctx0.shape)}, "
        f"x_tgt={tuple(x_tgt0.shape)}, "
        f"a_tgt={tuple(a_tgt0.shape)}, "
        f"xy_rel={tuple(xy0.shape)}"
    )
    print(f"[dataset] action_dim={action_dim}, latent_dim={latent_dim}")

    # ---- models ----
    predictor = LSTMPredictor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_layers=2
    ).to(DEVICE)

    extractor = StateExtractor(
        latent_dim=latent_dim,
        output_dim=2,
        hidden_dims=[128, 64]
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        list(vae.parameters()) + list(predictor.parameters()) + list(extractor.parameters()),
        lr=LR,
        weight_decay=WD
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8
    )

    # ---- training ----
    best_val = float("inf")
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        vae.train()
        predictor.train()
        extractor.train()

        tr_total = tr_vae = tr_z = tr_xy = 0.0
        tr_xy_preds, tr_xy_trues = [], []

        for x_ctx, a_ctx, x_tgt, a_tgt, xy_rel_tgt in train_ld:
            x_ctx = x_ctx.to(DEVICE)            # (B,15,C,H,W)
            a_ctx = a_ctx.to(DEVICE)            # (B,15,A)
            x_tgt = x_tgt.to(DEVICE)            # (B,5,C,H,W)
            a_tgt = a_tgt.to(DEVICE)            # (B,5,A)
            xy_rel_tgt = xy_rel_tgt.to(DEVICE)  # (B,5,2)

            B, Tctx, C, H, W = x_ctx.shape
            _, Tpred, _, _, _ = x_tgt.shape

            # ---- VAE forward (ctx) ----
            x_ctx_flat = x_ctx.view(B * Tctx, C, H, W)
            recon_ctx, mu_ctx, logvar_ctx = vae(x_ctx_flat)
            z_ctx = mu_ctx.view(B, Tctx, -1)  # (B,15,D)

            # ---- VAE forward (tgt) ----
            x_tgt_flat = x_tgt.view(B * Tpred, C, H, W)
            recon_tgt, mu_tgt, logvar_tgt = vae(x_tgt_flat)
            z_tgt = mu_tgt.view(B, Tpred, -1)  # (B,5,D)

            # ---- predictor multi-step ----
            z_pred, _ = predictor.forward_multi(
                z_ctx,
                action_seq=a_ctx,
                future_actions=a_tgt,
                pred_len=PRED_LEN
            )  # (B,5,D)

            # ---- extractor (relative xy) ----
            z_pred_flat = z_pred.reshape(B * Tpred, -1)
            xy_pred_flat = extractor(z_pred_flat)          # (B*5,2)
            xy_pred = xy_pred_flat.view(B, Tpred, 2)       # (B,5,2)

            # ---- losses ----
            l_z = predictor_loss(
                z_pred.reshape(B * Tpred, -1),
                z_tgt.reshape(B * Tpred, -1)
            )
            l_xy = predictor_loss(
                xy_pred.reshape(B * Tpred, 2),
                xy_rel_tgt.reshape(B * Tpred, 2)
            )

            l_vae_ctx, _, _ = vae_loss(recon_ctx, x_ctx_flat, mu_ctx, logvar_ctx, beta=beta)
            l_vae_tgt, _, _ = vae_loss(recon_tgt, x_tgt_flat, mu_tgt, logvar_tgt, beta=beta)
            l_vae = l_vae_ctx + l_vae_tgt

            loss = W_VAE * l_vae + W_Z * l_z + W_XY * l_xy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(predictor.parameters()) + list(extractor.parameters()),
                5.0
            )
            opt.step()

            tr_total += float(loss.item())
            tr_vae   += float(l_vae.item())
            tr_z     += float(l_z.item())
            tr_xy    += float(l_xy.item())

            tr_xy_preds.append(xy_pred.detach())
            tr_xy_trues.append(xy_rel_tgt.detach())

        tr_xy_preds = torch.cat(tr_xy_preds, dim=0)   # (N,5,2)
        tr_xy_trues = torch.cat(tr_xy_trues, dim=0)   # (N,5,2)
        tr_xrmse, tr_yrmse, tr_totalrmse = xy_rmse(tr_xy_preds, tr_xy_trues)

        # ---- validation ----
        vae.eval()
        predictor.eval()
        extractor.eval()

        va_total = va_vae = va_z = va_xy = 0.0
        va_xy_preds, va_xy_trues = [], []

        with torch.no_grad():
            for x_ctx, a_ctx, x_tgt, a_tgt, xy_rel_tgt in val_ld:
                x_ctx = x_ctx.to(DEVICE)
                a_ctx = a_ctx.to(DEVICE)
                x_tgt = x_tgt.to(DEVICE)
                a_tgt = a_tgt.to(DEVICE)
                xy_rel_tgt = xy_rel_tgt.to(DEVICE)

                B, Tctx, C, H, W = x_ctx.shape
                _, Tpred, _, _, _ = x_tgt.shape

                x_ctx_flat = x_ctx.view(B * Tctx, C, H, W)
                recon_ctx, mu_ctx, logvar_ctx = vae(x_ctx_flat)
                z_ctx = mu_ctx.view(B, Tctx, -1)

                x_tgt_flat = x_tgt.view(B * Tpred, C, H, W)
                recon_tgt, mu_tgt, logvar_tgt = vae(x_tgt_flat)
                z_tgt = mu_tgt.view(B, Tpred, -1)

                z_pred, _ = predictor.forward_multi(
                    z_ctx,
                    action_seq=a_ctx,
                    future_actions=a_tgt,
                    pred_len=PRED_LEN
                )

                z_pred_flat = z_pred.reshape(B * Tpred, -1)
                xy_pred_flat = extractor(z_pred_flat)
                xy_pred = xy_pred_flat.view(B, Tpred, 2)

                l_z = predictor_loss(
                    z_pred.reshape(B * Tpred, -1),
                    z_tgt.reshape(B * Tpred, -1)
                )
                l_xy = predictor_loss(
                    xy_pred.reshape(B * Tpred, 2),
                    xy_rel_tgt.reshape(B * Tpred, 2)
                )

                l_vae_ctx, _, _ = vae_loss(recon_ctx, x_ctx_flat, mu_ctx, logvar_ctx, beta=beta)
                l_vae_tgt, _, _ = vae_loss(recon_tgt, x_tgt_flat, mu_tgt, logvar_tgt, beta=beta)
                l_vae = l_vae_ctx + l_vae_tgt

                total = W_VAE * l_vae + W_Z * l_z + W_XY * l_xy

                va_total += float(total.item())
                va_vae   += float(l_vae.item())
                va_z     += float(l_z.item())
                va_xy    += float(l_xy.item())

                va_xy_preds.append(xy_pred)
                va_xy_trues.append(xy_rel_tgt)

        ntr = max(1, len(train_ld))
        nva = max(1, len(val_ld))

        tr_total /= ntr
        tr_vae   /= ntr
        tr_z     /= ntr
        tr_xy    /= ntr

        va_total /= nva
        va_vae   /= nva
        va_z     /= nva
        va_xy    /= nva

        va_xy_preds = torch.cat(va_xy_preds, dim=0)
        va_xy_trues = torch.cat(va_xy_trues, dim=0)
        va_xrmse, va_yrmse, va_totalrmse = xy_rmse(va_xy_preds, va_xy_trues)

        sched.step(va_total)

        print(
            f"epoch {epoch:03d} | "
            f"train total={tr_total:.6f} (vae={tr_vae:.6f}, z={tr_z:.6f}, xy={tr_xy:.6f}) | "
            f"val total={va_total:.6f} (vae={va_vae:.6f}, z={va_z:.6f}, xy={va_xy:.6f})\n"
            f"       RMSE(rel) train: x={float(tr_xrmse):.6f} y={float(tr_yrmse):.6f} total={float(tr_totalrmse):.6f} | "
            f"val: x={float(va_xrmse):.6f} y={float(va_yrmse):.6f} total={float(va_totalrmse):.6f}"
        )

        if va_total < best_val:
            best_val = va_total
            torch.save({
                "vae": vae.state_dict(),
                "predictor": predictor.state_dict(),
                "extractor": extractor.state_dict(),
                "vae_cfg": {
                    "latent_dim": latent_dim,
                    "input_channels": input_channels,
                    "beta": beta
                },
                "train_cfg": {
                    "NPZ_PATH": NPZ_PATH,
                    "CTX_LEN": CTX_LEN,
                    "PRED_LEN": PRED_LEN,
                    "TOTAL_LEN": TOTAL_LEN,
                    "STRIDE_STEP": STRIDE_STEP,
                    "RESIZE_TO": RESIZE_TO,
                    "ACTION_PAD_TO": ACTION_PAD_TO,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "LR": LR,
                    "WD": WD,
                    "W_VAE": W_VAE,
                    "W_Z": W_Z,
                    "W_XY": W_XY,
                }
            }, RUN_DIR / "joint_best.pt")

        row = {
            "epoch": epoch,
            "train_total": tr_total,
            "train_vae": tr_vae,
            "train_z": tr_z,
            "train_xy": tr_xy,
            "val_total": va_total,
            "val_vae": va_vae,
            "val_z": va_z,
            "val_xy": va_xy,
            "train_x_rmse": tr_xrmse,
            "train_y_rmse": tr_yrmse,
            "train_total_xy_rmse": tr_totalrmse,
            "val_x_rmse": va_xrmse,
            "val_y_rmse": va_yrmse,
            "val_total_xy_rmse": va_totalrmse,
        }
        history_rows.append(row)
        write_metrics_csv(RUN_DIR / "metrics.csv", history_rows)

    print(f"[train_joint] Best val loss: {best_val:.6f}")
    print(f"[train_joint] Checkpoint saved to: {RUN_DIR / 'joint_best.pt'}")


if __name__ == "__main__":
    main()