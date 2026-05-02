# audit_joint_errors.py
#
# Purpose: isolate why evaluation RMSE differs from training RMSE:
#   (a) metric/pipeline mismatch
#   (b) not training well (train error already high)
#   (c) poor generalization (train low, val high)
#
# What it does:
# 1) Recomputes per-window relative XY error EXACTLY like train_joint_full.py:
#    - windowing by CTX_LEN + STRIDE_STEP on raw NPZ (npz["data"] dict)
#    - xy_rel computed from global state using rotate-by(-yaw0) (global_to_rel_xy)
#    - VAE encoding uses mu from vae(x) (vae returns recon, mu, logvar)
#    - predictor.forward_last + extractor
#    - action padding to ACTION_PAD_TO
# 2) Saves per-window metrics to CSV: per_window_errors.csv
# 3) Prints overall RMSE + stats, and also a deterministic train/val split RMSE.
# 4) Prints top-K worst windows and their yaw/speed metadata.

import os
import sys
import math
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------
# Path setup (external/piwm-donkeycar -> world_model_2 root)
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from vae_training.models.vae_fixed import VAE
from vae_training.models.predictor import LSTMPredictor
from vae_training.models.extractor import StateExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# EDIT THESE PATHS
# -------------------------------------------------------
NPZ_PATH   = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"
JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-02-16_17-38-06\joint_best.pt"

# Must match how you trained THIS checkpoint
CTX_LEN     = 14
STRIDE_STEP = 5
RESIZE_TO   = (120, 160)
ACTION_PAD_TO = 3  # training used 3 (pads action from 2->3)

# NPZ dict keys (train_joint_full.py)
IMG_KEY   = "frame"
ACT_KEY   = "action"
STATE_KEY = "state"

# Split params (deterministic)
VAL_RATIO = 0.2
SPLIT_SEED = 123  # fixed for reproducibility

# Output
OUT_CSV = os.path.join(BASE_DIR, "per_window_errors.csv")
TOPK = 25

# -------------------------------------------------------
# Helpers (must match train_joint_full.py)
# -------------------------------------------------------
def to_chw_float01(img: np.ndarray, resize_to=(120, 160), input_channels=3) -> torch.Tensor:
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")

    # HWC -> CHW if needed
    if img.shape[-1] in (1, 3, 4):
        img = np.transpose(img, (2, 0, 1))

    if input_channels == 1 and img.shape[0] > 1:
        img = img.mean(axis=0, keepdims=True)

    x = torch.from_numpy(img)  # (C,H,W)
    if resize_to is not None:
        H, W = resize_to
        x = F.interpolate(x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    return x

def global_to_rel_xy(x_t, y_t, x0, y0, yaw0_deg):
    # exactly like NPZJointRelativeDataset._global_to_rel_xy
    dx = float(x_t - x0)
    dy = float(y_t - y0)
    yaw0 = math.radians(float(yaw0_deg))
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    xr =  c * dx + s * dy
    yr = -s * dx + c * dy
    return xr, yr

def rmse_stats(dx: np.ndarray, dy: np.ndarray):
    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    rmse_xy = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    dist = np.sqrt(dx * dx + dy * dy)
    stats = {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(dist.max()),
    }
    return rmse_x, rmse_y, rmse_xy, stats

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    joint = torch.load(JOINT_CKPT, map_location=DEVICE)
    if not all(k in joint for k in ("vae", "predictor", "extractor", "vae_cfg")):
        raise KeyError(f"JOINT_CKPT missing keys. Have: {list(joint.keys())}")

    vae_cfg = joint["vae_cfg"]
    latent_dim = int(vae_cfg["latent_dim"])
    input_channels = int(vae_cfg.get("input_channels", 3))

    # Load VAE from JOINT checkpoint (critical)
    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(DEVICE)
    vae.load_state_dict(joint["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Build predictor/extractor
    action_dim = int(ACTION_PAD_TO) if ACTION_PAD_TO is not None else 2
    predictor = LSTMPredictor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=128, num_layers=2).to(DEVICE)
    extractor = StateExtractor(latent_dim=latent_dim, output_dim=2, hidden_dims=[128, 64]).to(DEVICE)
    predictor.load_state_dict(joint["predictor"])
    extractor.load_state_dict(joint["extractor"])
    predictor.eval()
    extractor.eval()

    # Load NPZ
    d = np.load(NPZ_PATH, allow_pickle=True)
    raw = d["data"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    frames  = raw[IMG_KEY]
    actions = raw[ACT_KEY]
    state   = raw[STATE_KEY]

    T_total = int(frames.shape[0])
    needed = CTX_LEN + 1
    t0s = list(range(0, T_total - needed + 1, STRIDE_STEP))
    N = len(t0s)

    print(f"[audit] windows = {N}  (T={T_total}, CTX_LEN={CTX_LEN}, STRIDE_STEP={STRIDE_STEP})")
    print(f"[audit] latent_dim={latent_dim}, input_channels={input_channels}, action_dim={action_dim}")

    rows = []
    errs = []

    with torch.no_grad():
        for w, t0 in enumerate(t0s):
            t1 = t0 + CTX_LEN

            # images (ctx + tgt)
            x_ctx_np = frames[t0:t1]
            x_tgt_np = frames[t1]

            x_ctx = torch.stack(
                [to_chw_float01(x_ctx_np[j], RESIZE_TO, input_channels) for j in range(CTX_LEN)],
                dim=0
            )  # (T,C,H,W)
            x_tgt = to_chw_float01(x_tgt_np, RESIZE_TO, input_channels)

            # actions (pad to 3 if needed)
            a_ctx = actions[t0:t1].astype(np.float32)  # (CTX_LEN,2) likely
            if ACTION_PAD_TO is not None and a_ctx.shape[1] < ACTION_PAD_TO:
                pad = ACTION_PAD_TO - a_ctx.shape[1]
                a_ctx = np.pad(a_ctx, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

            # compute true relative xy from global state
            x0, y0, yaw0 = state[t0, 0], state[t0, 1], state[t0, 2]
            xt, yt, yawt = state[t1, 0], state[t1, 1], state[t1, 2]
            v0 = state[t0, 3] if state.shape[1] > 3 else np.nan
            vt = state[t1, 3] if state.shape[1] > 3 else np.nan

            xr_true, yr_true = global_to_rel_xy(xt, yt, x0, y0, yaw0)

            # encode ctx with mu
            x_ctx_b = x_ctx.unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
            B, Tctx, C, H, W = x_ctx_b.shape
            x_ctx_flat = x_ctx_b.view(B * Tctx, C, H, W)
            _, mu_ctx, _ = vae(x_ctx_flat)
            z_ctx = mu_ctx.view(B, Tctx, -1)  # (1,T,D)

            # predictor/extractor
            a_ctx_b = torch.from_numpy(a_ctx).unsqueeze(0).to(DEVICE)  # (1,T,A)
            z_pred, _ = predictor.forward_last(z_ctx, a_ctx_b)
            xy_pred = extractor(z_pred)[0].cpu().numpy().astype(np.float32)

            err_x = float(xy_pred[0] - xr_true)
            err_y = float(xy_pred[1] - yr_true)
            err_d = float(math.sqrt(err_x * err_x + err_y * err_y))

            # yaw change (wrap-safe)
            dyaw = float(yawt - yaw0)
            dyaw = (dyaw + 180.0) % 360.0 - 180.0  # map to [-180,180]

            rows.append({
                "window_idx": w,
                "t0": int(t0),
                "t1": int(t1),
                "x0": safe_float(x0), "y0": safe_float(y0), "yaw0": safe_float(yaw0),
                "xt": safe_float(xt), "yt": safe_float(yt), "yawt": safe_float(yawt),
                "v0": safe_float(v0), "vt": safe_float(vt),
                "dyaw_deg": safe_float(dyaw),
                "xr_true": safe_float(xr_true), "yr_true": safe_float(yr_true),
                "xr_pred": safe_float(xy_pred[0]), "yr_pred": safe_float(xy_pred[1]),
                "err_x": err_x, "err_y": err_y, "err_dist": err_d,
            })
            errs.append([err_x, err_y])

    errs = np.asarray(errs, dtype=np.float32)
    dx, dy = errs[:, 0], errs[:, 1]
    rmse_x, rmse_y, rmse_xy, stats = rmse_stats(dx, dy)

    print("\n===== OVERALL RELATIVE RMSE (same target as training) =====")
    print(f"RMSE X     : {rmse_x:.6f} m")
    print(f"RMSE Y     : {rmse_y:.6f} m")
    print(f"RMSE XY    : {rmse_xy:.6f} m")
    print(f"dist err   : mean={stats['mean']:.6f}  median={stats['median']:.6f}  p95={stats['p95']:.6f}  max={stats['max']:.6f}")
    print("===========================================================\n")

    # Deterministic split (approx of random_split if you didn't save indices)
    rng = np.random.default_rng(SPLIT_SEED)
    indices = np.arange(N)
    rng.shuffle(indices)
    val_n = int(round(VAL_RATIO * N))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    dx_tr, dy_tr = dx[train_idx], dy[train_idx]
    dx_va, dy_va = dx[val_idx], dy[val_idx]

    trx, try_, trxy, trstats = rmse_stats(dx_tr, dy_tr)
    vax, vay, vaxy, vastats = rmse_stats(dx_va, dy_va)

    print("===== DETERMINISTIC SPLIT RMSE (seeded) =====")
    print(f"train N={len(train_idx)} | RMSE xy={trxy:.6f} (x={trx:.6f}, y={try_:.6f}) | p95={trstats['p95']:.6f}")
    print(f"val   N={len(val_idx)} | RMSE xy={vaxy:.6f} (x={vax:.6f}, y={vay:.6f}) | p95={vastats['p95']:.6f}")
    print("=============================================\n")

    # Write CSV
    Path(os.path.dirname(OUT_CSV)).mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[audit] wrote CSV: {OUT_CSV}")

    # Show top-K worst windows (by distance error)
    rows_sorted = sorted(rows, key=lambda r: r["err_dist"], reverse=True)
    print(f"\n===== TOP {TOPK} WORST WINDOWS (by err_dist) =====")
    for r in rows_sorted[:TOPK]:
        print(
            f"w={r['window_idx']:4d} t0={r['t0']:5d} t1={r['t1']:5d} "
            f"err={r['err_dist']:.4f}m  dyaw={r['dyaw_deg']:+7.2f}deg  "
            f"v0={r['v0']:.3f} vt={r['vt']:.3f}"
        )
    print("===============================================\n")

    # Quick correlation hints (not perfect, but helpful)
    dyaw_all = np.array([r["dyaw_deg"] for r in rows], dtype=np.float32)
    v_all = np.array([r["vt"] for r in rows], dtype=np.float32)
    dist_all = np.sqrt(dx * dx + dy * dy)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return np.nan
        aa = a[m] - a[m].mean()
        bb = b[m] - b[m].mean()
        return float((aa * bb).mean() / (aa.std() * bb.std() + 1e-8))

    print("===== SIMPLE CORRELATIONS (heuristic) =====")
    print(f"corr(|dyaw|, err_dist) = {corr(np.abs(dyaw_all), dist_all):.4f}")
    print(f"corr(vt, err_dist)     = {corr(v_all, dist_all):.4f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()