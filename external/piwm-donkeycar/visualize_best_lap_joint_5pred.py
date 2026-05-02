# visualize_best_lap_joint_fullaware_5step.py
#
# Finds which lap has the BEST RMSE (relative/global) and plots that lap
# for a 15->5 predictor:
#   - windows from raw NPZ using CTX_LEN=15 and STRIDE_STEP=5
#   - each window predicts the next PRED_LEN=5 future steps
#   - xy_rel target computed from global state (rotate by -yaw0)
#   - VAE encode uses mu (vae(x)-> recon, mu, logvar)
#   - predictor.forward_multi(...)
#   - extractor outputs xy_rel_pred for each future step
#   - rel_to_global for plotting
#
# Outputs:
#   - per-lap RMSE summary
#   - best lap index + metrics
#   - plot for best lap

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
NPZ_PATH   = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_50_step5_wrapped.npz"
JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-03-30_16-09-00\joint_best.pt"

# 25 :JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-03-30_13-24-58\joint_best.pt"


#15 pred: JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-03-29_18-51-39\joint_best.pt"


# 5 pred: JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-03-23_13-19-51\joint_best.pt"

# Must match training
CTX_LEN       = 15
PRED_LEN      = 35
STRIDE_STEP   = 5
RESIZE_TO     = (120, 160)
ACTION_PAD_TO = 3  # same as training (pad 2->3 with zeros)

# NPZ dict keys (train_joint_full.py)
IMG_KEY   = "frame"
ACT_KEY   = "action"
STATE_KEY = "state"

# Lap detection
LAP_THRESHOLD = 0.30   # meters, distance to start
LAP_MIN_GAP   = 200    # frames; helps prevent false "laps" right away

# Plot controls
ARROW_SCALE = 100
PAIR_LINE_EVERY = 1    # 1 = every pair line, increase to reduce clutter

# Choose what defines "best"
# Options: "rel_rmse_xy" or "glob_rmse_xy"
BEST_BY = "rel_rmse_xy"


# -------------------------------------------------------
# Plot helper: dotted pair lines
# -------------------------------------------------------
def plot_pair_lines(true_states: np.ndarray, pred_states: np.ndarray, every: int = 1):
    true_xy = true_states[:, :2]
    pred_xy = pred_states[:, :2]
    n = min(len(true_xy), len(pred_xy))
    step = max(1, int(every))
    for i in range(0, n, step):
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
        h, w = resize_to
        x = F.interpolate(
            x.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
    return x


def global_to_rel_xy(x_t, y_t, x0, y0, yaw0_deg):
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


def plot_states(states: np.ndarray, color: str, label: str, scale=100):
    x = states[:, 0]
    y = states[:, 1]
    headings = states[:, 2]
    u = np.cos(np.deg2rad(headings))
    v = np.sin(np.deg2rad(headings))
    plt.quiver(x, y, u, v, color=color, scale=scale, label=label)


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
# Lap detection (multiple laps)
# -------------------------------------------------------
def find_lap_boundaries(xs: np.ndarray, ys: np.ndarray, threshold=0.3, min_gap=200):
    """
    Returns boundaries as a list of indices:
      boundaries = [0, end_of_lap0, end_of_lap1, ..., len(xs)]
    A lap "end" is when we're within threshold of the start point after min_gap.
    We also enforce at least min_gap between consecutive lap ends.
    """
    x0, y0 = float(xs[0]), float(ys[0])
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    close = np.where(d2 < threshold ** 2)[0]

    # ignore very early closeness
    close = close[close > min_gap]

    lap_ends = []
    last_end = 0
    for idx in close:
        if idx - last_end >= min_gap:
            lap_ends.append(int(idx))
            last_end = int(idx)

    boundaries = [0] + lap_ends + [int(len(xs))]
    cleaned = [boundaries[0]]
    for b in boundaries[1:]:
        if b > cleaned[-1]:
            cleaned.append(b)
    return cleaned


# -------------------------------------------------------
# Core evaluation on a lap segment [start, end)
# -------------------------------------------------------
def eval_lap(
    frames, actions, state,
    start_idx: int, end_idx: int,
    vae, predictor, extractor,
    latent_dim: int, input_channels: int
):
    """
    Evaluates windows whose t0 starts in [start_idx, end_idx) and whose
    prediction horizon fits inside the segment.

    For each window:
      context = frames[t0 : t0+CTX_LEN]
      targets = frames[t1 : t1+PRED_LEN]

    Returns dict with metrics + plotting arrays.
    """
    needed = CTX_LEN + PRED_LEN
    if (end_idx - start_idx) < needed:
        return None

    t0s = list(range(start_idx, end_idx - needed + 1, STRIDE_STEP))

    true_global = []
    pred_global = []
    rel_errs = []
    glob_errs = []

    with torch.no_grad():
        for t0 in t0s:
            t1 = t0 + CTX_LEN
            t2 = t1 + PRED_LEN
            if t2 > end_idx:
                break

            # ---- Images ----
            x_ctx_np = frames[t0:t1]
            x_ctx = torch.stack(
                [to_chw_float01(x_ctx_np[j], RESIZE_TO, input_channels) for j in range(CTX_LEN)],
                dim=0
            )

            # ---- Context actions ----
            a_ctx = actions[t0:t1].astype(np.float32)
            if ACTION_PAD_TO is not None and a_ctx.shape[1] < ACTION_PAD_TO:
                pad = ACTION_PAD_TO - a_ctx.shape[1]
                a_ctx = np.pad(a_ctx, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

            # ---- Future actions ----
            a_tgt = actions[t1:t2].astype(np.float32)
            if ACTION_PAD_TO is not None and a_tgt.shape[1] < ACTION_PAD_TO:
                pad = ACTION_PAD_TO - a_tgt.shape[1]
                a_tgt = np.pad(a_tgt, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

            # ---- True relative targets from GLOBAL state ----
            x0, y0, yaw0 = state[t0, 0], state[t0, 1], state[t0, 2]
            origin_xyz = np.array([x0, y0, yaw0], dtype=np.float32)

            xy_rel_true_all = []
            future_true_global = []
            for k in range(PRED_LEN):
                tt = t1 + k
                xt, yt, yawt = state[tt, 0], state[tt, 1], state[tt, 2]
                xr_true, yr_true = global_to_rel_xy(xt, yt, x0, y0, yaw0)
                xy_rel_true_all.append([xr_true, yr_true])
                future_true_global.append([float(xt), float(yt), float(yawt)])

            xy_rel_true_all = np.asarray(xy_rel_true_all, dtype=np.float32)   # (5,2)
            future_true_global = np.asarray(future_true_global, dtype=np.float32)

            # ---- VAE encode using mu ----
            x_ctx_b = x_ctx.unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
            bsz, tctx, c, h, w = x_ctx_b.shape
            x_ctx_flat = x_ctx_b.view(bsz * tctx, c, h, w)

            _, mu_ctx, _ = vae(x_ctx_flat)
            z_ctx = mu_ctx.view(bsz, tctx, -1)  # (1,15,D)

            # ---- predictor ----
            a_ctx_b = torch.from_numpy(a_ctx).unsqueeze(0).to(DEVICE)  # (1,15,A)
            a_tgt_b = torch.from_numpy(a_tgt).unsqueeze(0).to(DEVICE)  # (1,5,A)

            z_pred, _ = predictor.forward_multi(
                z_ctx,
                action_seq=a_ctx_b,
                future_actions=a_tgt_b,
                pred_len=PRED_LEN
            )  # (1,5,D)

            # ---- extractor ----
            z_pred_flat = z_pred.reshape(PRED_LEN, -1)                  # (5,D)
            xy_rel_pred_all = extractor(z_pred_flat).cpu().numpy()      # (5,2)

            # ---- errors + global coords ----
            for k in range(PRED_LEN):
                xy_rel_pred_k = xy_rel_pred_all[k]
                xy_rel_true_k = xy_rel_true_all[k]

                xt, yt, yawt = future_true_global[k]
                xg_pred, yg_pred = rel_to_global(xy_rel_pred_k, origin_xyz)

                rel_errs.append(xy_rel_pred_k - xy_rel_true_k)
                glob_errs.append(np.array([float(xg_pred - xt), float(yg_pred - yt)], dtype=np.float32))

                true_global.append([float(xt), float(yt), float(yawt)])
                pred_global.append([float(xg_pred), float(yg_pred), float(yawt)])  # use true yaw for arrows

    if len(true_global) == 0:
        return None

    true_global = np.asarray(true_global, dtype=np.float32)
    pred_global = np.asarray(pred_global, dtype=np.float32)
    rel_errs = np.asarray(rel_errs, dtype=np.float32)
    glob_errs = np.asarray(glob_errs, dtype=np.float32)

    rmx, rmy, rmxy, rstats = rmse_xy(rel_errs)
    gmx, gmy, gmxy, gstats = rmse_xy(glob_errs)

    return {
        "start": int(start_idx),
        "end": int(end_idx),
        "n_points": int(len(true_global)),
        "rel_rmse_x": rmx,
        "rel_rmse_y": rmy,
        "rel_rmse_xy": rmxy,
        "rel_stats": rstats,
        "glob_rmse_x": gmx,
        "glob_rmse_y": gmy,
        "glob_rmse_xy": gmxy,
        "glob_stats": gstats,
        "true_global": true_global,
        "pred_global": pred_global,
    }


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

    # Load models from joint checkpoint
    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(DEVICE)
    vae.load_state_dict(joint["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    action_dim = int(ACTION_PAD_TO) if ACTION_PAD_TO is not None else 2
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

    predictor.load_state_dict(joint["predictor"])
    extractor.load_state_dict(joint["extractor"])
    predictor.eval()
    extractor.eval()

    # Load NPZ
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

    frames = raw[IMG_KEY]
    actions = raw[ACT_KEY]
    state = raw[STATE_KEY]

    xs = state[:, 0].astype(np.float32)
    ys = state[:, 1].astype(np.float32)

    # Detect laps
    boundaries = find_lap_boundaries(xs, ys, threshold=LAP_THRESHOLD, min_gap=LAP_MIN_GAP)
    laps = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    print(f"[bestlap] Loaded: {JOINT_CKPT}")
    print(
        f"[bestlap] latent_dim={latent_dim}, input_channels={input_channels}, "
        f"CTX_LEN={CTX_LEN}, PRED_LEN={PRED_LEN}, STRIDE_STEP={STRIDE_STEP}"
    )
    print(f"[bestlap] total frames: {len(xs)}")
    print(f"[bestlap] detected lap segments: {len(laps)}")
    for i, (a, b) in enumerate(laps):
        print(f"  lap {i}: [{a}, {b})  len={b-a}")

    # Evaluate each lap
    results = []
    for i, (start, end) in enumerate(laps):
        r = eval_lap(frames, actions, state, start, end, vae, predictor, extractor, latent_dim, input_channels)
        if r is None:
            print(f"[bestlap] lap {i}: skipped (too short / no points)")
            continue
        r["lap_index"] = int(i)
        results.append(r)

    if len(results) == 0:
        raise RuntimeError("No laps produced valid evaluation points. Try lowering LAP_MIN_GAP or checking data.")

    # Print per-lap summary
    print("\n===== PER-LAP RMSE SUMMARY =====")
    for r in results:
        i = r["lap_index"]
        print(
            f"lap {i:02d}  points={r['n_points']:4d}  "
            f"rel_rmse_xy={r['rel_rmse_xy']:.6f}  glob_rmse_xy={r['glob_rmse_xy']:.6f}  "
            f"segment=[{r['start']},{r['end']})"
        )
    print("================================\n")

    # Choose best lap
    if BEST_BY not in ("rel_rmse_xy", "glob_rmse_xy"):
        raise ValueError("BEST_BY must be 'rel_rmse_xy' or 'glob_rmse_xy'")

    best = min(results, key=lambda rr: rr[BEST_BY])
    best_i = best["lap_index"]

    print(f"[bestlap] BEST_BY={BEST_BY} -> lap {best_i} is best")
    print("===== BEST LAP METRICS =====")
    print(f"lap index : {best_i}")
    print(f"segment   : [{best['start']},{best['end']})  len={best['end']-best['start']}")
    print(f"points    : {best['n_points']}")

    print("\n--- RELATIVE RMSE ---")
    print(f"RMSE X     : {best['rel_rmse_x']:.6f} m")
    print(f"RMSE Y     : {best['rel_rmse_y']:.6f} m")
    print(f"RMSE XY    : {best['rel_rmse_xy']:.6f} m")
    rs = best["rel_stats"]
    print(f"dist err   : mean={rs['mean']:.6f}  median={rs['median']:.6f}  p95={rs['p95']:.6f}  max={rs['max']:.6f}")

    print("\n--- GLOBAL RMSE ---")
    print(f"RMSE X     : {best['glob_rmse_x']:.6f} m")
    print(f"RMSE Y     : {best['glob_rmse_y']:.6f} m")
    print(f"RMSE XY    : {best['glob_rmse_xy']:.6f} m")
    gs = best["glob_stats"]
    print(f"dist err   : mean={gs['mean']:.6f}  median={gs['median']:.6f}  p95={gs['p95']:.6f}  max={gs['max']:.6f}")
    print("=============================\n")

    # Plot best lap
    start, end = best["start"], best["end"]
    true_global = best["true_global"]
    pred_global = best["pred_global"]

    plt.figure(figsize=(8, 7))

    # show path for this lap segment
    plt.plot(xs[start:end], ys[start:end], "k-", linewidth=1.0, alpha=0.4, label=f"Lap {best_i} path")

    plot_states(true_global, "b", "True predicted points", scale=ARROW_SCALE)
    plot_states(pred_global, "r", "Predicted points", scale=ARROW_SCALE)

    plot_pair_lines(true_global, pred_global, every=PAIR_LINE_EVERY)

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (global)")
    plt.ylabel("y (global)")
    plt.title(f"Best lap (lap {best_i}) predictions | {BEST_BY}={best[BEST_BY]:.6f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()