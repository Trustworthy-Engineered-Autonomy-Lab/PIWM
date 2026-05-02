# visualize_joint_fullaware.py
#
# Visualize joint-trained (VAE + predictor + extractor) predictions
# in GLOBAL coordinates, computed the SAME way as train_joint_full.py:
#   - windows from raw NPZ using CTX_LEN and STRIDE_STEP
#   - xy_rel target computed from global state (rotate by -yaw0)
#   - VAE encode uses mu (via vae(x) returning recon, mu, logvar)
#   - predictor.forward_last
#   - extractor outputs xy_rel_pred
#   - rel_to_global to plot
#
# Also prints:
#   - RELATIVE RMSE (same metric as training RMSE(rel))
#   - GLOBAL RMSE after converting predicted rel to global (sanity check)
#   - yaw sanity check (we force pred yaw arrows to equal true yaw arrows)
#
# Plus:
#   - dotted lines between each true/pred pair (plot_pair_lines)

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

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
JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-02-23_17-19-08\joint_best.pt"
#JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-02-16_17-38-06\joint_best.pt"

# Must match training
CTX_LEN       = 14
STRIDE_STEP   = 5
RESIZE_TO     = (120, 160)
ACTION_PAD_TO = 3  # same as training (pad 2->3 with zeros)

# NPZ dict keys (train_joint_full.py)
IMG_KEY   = "frame"
ACT_KEY   = "action"
STATE_KEY = "state"

# Lap detection
LAP_THRESHOLD = 0.3
LAP_MIN_GAP   = 200

# Quiver scale
ARROW_SCALE = 100

# Pair-line density (1 = draw every line; higher = less clutter)
PAIR_LINE_EVERY = 1


# -------------------------------------------------------
# Plot helper: dotted pair lines
# -------------------------------------------------------
def plot_pair_lines(true_states: np.ndarray, pred_states: np.ndarray, every: int = 1):
    """
    Draw dotted lines connecting each true point to its predicted point.
    true_states, pred_states: (N,3) arrays [x,y,yaw]
    every: draw every k-th line to reduce clutter
    """
    true_xy = true_states[:, :2]
    pred_xy = pred_states[:, :2]
    N = min(len(true_xy), len(pred_xy))

    step = max(1, int(every))
    for i in range(0, N, step):
        plt.plot(
            [true_xy[i, 0], pred_xy[i, 0]],
            [true_xy[i, 1], pred_xy[i, 1]],
            linestyle=":",
            linewidth=0.8,
            alpha=0.6,
        )


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

    # optional grayscale collapse
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


def rel_to_global(rel_xy, origin_xyz):
    xr, yr = float(rel_xy[0]), float(rel_xy[1])
    x0, y0, yaw0_deg = float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])

    yaw0 = math.radians(yaw0_deg)
    c, s = math.cos(yaw0), math.sin(yaw0)

    xg = x0 + c * xr - s * yr
    yg = y0 + s * xr + c * yr
    return xg, yg


def find_first_lap(xs: np.ndarray, ys: np.ndarray, threshold=0.3, min_gap=200) -> int:
    x0, y0 = xs[0], ys[0]
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    cand = np.where(d2 < threshold ** 2)[0]
    cand = cand[cand > min_gap]
    return int(cand[0]) if len(cand) > 0 else len(xs)


def plot_states(states: np.ndarray, color: str, label: str, scale=100):
    X = states[:, 0]
    Y = states[:, 1]
    headings = states[:, 2]
    U = np.cos(np.deg2rad(headings))
    V = np.sin(np.deg2rad(headings))
    plt.quiver(X, Y, U, V, color=color, scale=scale, label=label)


def rmse_xy(errors_xy: np.ndarray):
    dx = errors_xy[:, 0]
    dy = errors_xy[:, 1]
    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    rmse_total = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    dist = np.sqrt(dx * dx + dy * dy)
    stats = {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(dist.max()),
    }
    return rmse_x, rmse_y, rmse_total, stats


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

    # Load VAE FROM JOINT checkpoint (critical for joint training)
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

    # Load NPZ raw dict (same as training script)
    d = np.load(NPZ_PATH, allow_pickle=True)
    if "data" not in d.files:
        raise KeyError(f"Expected top-level key 'data' in NPZ. Found keys: {list(d.files)}")

    raw = d["data"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if not isinstance(raw, dict):
        raise TypeError(f"npz['data'] expected dict, got {type(raw)}")

    for k in (IMG_KEY, ACT_KEY, STATE_KEY):
        if k not in raw:
            raise KeyError(f"Missing key '{k}' in npz['data']. Have keys: {list(raw.keys())}")

    frames  = raw[IMG_KEY]     # (T,3,224,224) uint8 (or sometimes THWC)
    actions = raw[ACT_KEY]     # (T,2) float32
    state   = raw[STATE_KEY]   # (T,4) float32 [x,y,yaw_deg,v]

    xs = state[:, 0].astype(np.float32)
    ys = state[:, 1].astype(np.float32)

    lap_end = find_first_lap(xs, ys, threshold=LAP_THRESHOLD, min_gap=LAP_MIN_GAP)

    needed = CTX_LEN + 1
    T_total = int(frames.shape[0])
    if T_total < needed:
        raise RuntimeError(f"Not enough frames: T={T_total}, need at least {needed} for CTX_LEN={CTX_LEN}")

    t0s = list(range(0, T_total - needed + 1, STRIDE_STEP))

    # containers
    true_global = []
    pred_global = []
    rel_errs = []
    glob_errs = []
    yaw_true_list = []
    yaw_used_list = []

    with torch.no_grad():
        for t0 in t0s:
            t1 = t0 + CTX_LEN  # target index

            if t1 >= lap_end:
                break

            # ---- Build ctx/tgt images ----
            x_ctx_np = frames[t0:t1]  # (CTX_LEN, C, H, W) or (CTX_LEN,H,W,C)
            x_tgt_np = frames[t1]

            x_ctx = torch.stack(
                [to_chw_float01(x_ctx_np[j], RESIZE_TO, input_channels) for j in range(CTX_LEN)],
                dim=0
            )  # (T,C,H,W)
            x_tgt = to_chw_float01(x_tgt_np, RESIZE_TO, input_channels)  # (C,H,W)

            # ---- Actions (pad 2->3 if needed) ----
            a_ctx = actions[t0:t1].astype(np.float32)  # (CTX_LEN,2)
            if ACTION_PAD_TO is not None and a_ctx.shape[1] < ACTION_PAD_TO:
                pad = ACTION_PAD_TO - a_ctx.shape[1]
                a_ctx = np.pad(a_ctx, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

            # ---- True relative target from GLOBAL state ----
            x0, y0, yaw0 = state[t0, 0], state[t0, 1], state[t0, 2]
            xt, yt, yawt = state[t1, 0], state[t1, 1], state[t1, 2]

            xr_true, yr_true = global_to_rel_xy(xt, yt, x0, y0, yaw0)
            xy_rel_true = np.array([xr_true, yr_true], dtype=np.float32)

            # ---- VAE encode using mu like training ----
            x_ctx_b = x_ctx.unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
            B, Tctx, C, H, W = x_ctx_b.shape
            x_ctx_flat = x_ctx_b.view(B * Tctx, C, H, W)

            # vae(x) returns (recon, mu, logvar) in your codebase
            _, mu_ctx, _ = vae(x_ctx_flat)
            z_ctx = mu_ctx.view(B, Tctx, -1)  # (1,T,D)

            # ---- predictor/extractor ----
            a_ctx_b = torch.from_numpy(a_ctx).unsqueeze(0).to(DEVICE)  # (1,T,A)
            z_pred, _ = predictor.forward_last(z_ctx, a_ctx_b)         # (1,D)
            xy_rel_pred = extractor(z_pred)[0].cpu().numpy().astype(np.float32)  # (2,)

            # ---- relative error ----
            rel_errs.append(xy_rel_pred - xy_rel_true)

            # ---- convert to GLOBAL for plot + global error ----
            origin_xyz = np.array([x0, y0, yaw0], dtype=np.float32)
            xg_pred, yg_pred = rel_to_global(xy_rel_pred, origin_xyz)

            true_global.append([float(xt), float(yt), float(yawt)])
            # IMPORTANT: for arrows, use true target yaw so arrows “match”
            pred_global.append([float(xg_pred), float(yg_pred), float(yawt)])

            yaw_true_list.append(float(yawt))
            yaw_used_list.append(float(yawt))

            glob_errs.append(np.array([float(xg_pred - xt), float(yg_pred - yt)], dtype=np.float32))

    if len(true_global) == 0:
        raise RuntimeError("Collected 0 points. Check CTX_LEN/STRIDE_STEP or lap detection parameters.")

    true_global = np.asarray(true_global, dtype=np.float32)
    pred_global = np.asarray(pred_global, dtype=np.float32)
    rel_errs = np.asarray(rel_errs, dtype=np.float32)
    glob_errs = np.asarray(glob_errs, dtype=np.float32)

    # --- RMSE ---
    rmx, rmy, rmxy, rstats = rmse_xy(rel_errs)
    gmx, gmy, gmxy, gstats = rmse_xy(glob_errs)

    yaw_true = np.array(yaw_true_list, dtype=np.float32)
    yaw_used = np.array(yaw_used_list, dtype=np.float32)
    yaw_diff = float(np.max(np.abs(yaw_true - yaw_used)))

    print(f"[viz] Loaded: {JOINT_CKPT}")
    print(f"[viz] latent_dim={latent_dim}, input_channels={input_channels}, CTX_LEN={CTX_LEN}, STRIDE_STEP={STRIDE_STEP}")
    print(f"[viz] points plotted (first lap) = {len(true_global)}")
    print(f"[viz] max(|yaw_true - yaw_used|) = {yaw_diff:.6f} deg (should be 0)\n")

    print("===== RELATIVE RMSE (matches train_joint_full.py target) =====")
    print(f"RMSE X     : {rmx:.6f} m")
    print(f"RMSE Y     : {rmy:.6f} m")
    print(f"RMSE XY    : {rmxy:.6f} m")
    print(f"dist err   : mean={rstats['mean']:.6f}  median={rstats['median']:.6f}  p95={rstats['p95']:.6f}  max={rstats['max']:.6f}")
    print("==============================================================\n")

    print("===== GLOBAL RMSE (after rel_to_global conversion) =====")
    print(f"RMSE X     : {gmx:.6f} m")
    print(f"RMSE Y     : {gmy:.6f} m")
    print(f"RMSE XY    : {gmxy:.6f} m")
    print(f"dist err   : mean={gstats['mean']:.6f}  median={gstats['median']:.6f}  p95={gstats['p95']:.6f}  max={gstats['max']:.6f}")
    print("========================================================\n")

    # --- Plot ---
    plt.figure(figsize=(8, 7))
    plt.plot(xs[:lap_end], ys[:lap_end], "k-", linewidth=1.0, alpha=0.4, label="Global lap (path)")

    plot_states(true_global, "b", "True (every stride)", scale=ARROW_SCALE)
    plot_states(pred_global, "r", "Pred (every stride, true yaw)", scale=ARROW_SCALE)

    # dotted lines from true -> pred
    plot_pair_lines(true_global, pred_global, every=PAIR_LINE_EVERY)

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (global)")
    plt.ylabel("y (global)")
    plt.title("Joint-trained model predictions over first full lap (global coords)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
