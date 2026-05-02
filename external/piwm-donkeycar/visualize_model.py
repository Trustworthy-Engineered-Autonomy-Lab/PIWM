import sys
import os
import math

import numpy as np
import torch
from matplotlib import pyplot as plt

# -------------------------------------------------------
# Path setup so we can import your world_model code
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

WM_ROOT  = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
VAE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vae_training"))

sys.path.append(WM_ROOT)
sys.path.append(VAE_ROOT)

# -------------------------------------------------------
# Imports from your project
# -------------------------------------------------------
from vae_training.models.vae_fixed import VAE
from vae_training.models.predictor import LSTMPredictor
from vae_training.models.extractor import StateExtractor
from vae_training.utils.predictor_dataloader import NPZRelativePredictorDataset
from piwm_utils import numpy_mse, numpy_rmse  # we'll use these on meter errors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def rel_to_global(rel_pose, global_origin):
    """
    rel_pose: [x_rel, y_rel]
    global_origin: [x0, y0, yaw0_deg] at the start of the window
    returns: (xg, yg)  -- yaw is handled outside (we reuse true yaw)
    """
    xr, yr = rel_pose
    x0, y0, yaw0_deg = global_origin

    yaw0_rad = math.radians(yaw0_deg)
    cg, sg = math.cos(yaw0_rad), math.sin(yaw0_rad)

    xg = x0 + cg * xr - sg * yr
    yg = y0 + sg * xr + cg * yr

    return xg, yg


def plot_states(states: np.ndarray, color: str, label: str):
    """
    states: (N, 3) with columns [x, y, yaw_deg]
    """
    X = states[:, 0]
    Y = states[:, 1]
    headings = states[:, 2]

    U = np.cos(np.deg2rad(headings))
    V = np.sin(np.deg2rad(headings))

    plt.quiver(X, Y, U, V, color=color, scale=100, label=label)


def find_first_lap(xs: np.ndarray, ys: np.ndarray,
                   threshold: float = 0.3, min_gap: int = 200) -> int:
    """
    Returns index where the first lap ends (first return near start after min_gap frames).
    """
    x0, y0 = xs[0], ys[0]
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    cand = np.where(d2 < threshold ** 2)[0]
    cand = cand[cand > min_gap]
    return int(cand[0]) if len(cand) > 0 else len(xs)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # ----- Paths (edit if needed) -----

    #5 window length, 5 step stride, RMSE 0.060984 m
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_wrapped.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2025-11-18_21-02-28\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"
    
    #15 window length, 15 stride step, RMSE 0.215619 m
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_15_wrapped_like_A.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2026-01-18_23-02-28\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"

    #15 window length, 5 step stride, RMSE 0.236654 m
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2026-01-31_21-00-46\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"

    #Retrained 5 window length, 5 stride step, RMSE 0.061099 m (This was a sanity check)
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_wrapped.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2026-02-01_14-53-48\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"

    #15 window length, 5 stide step, RMSE 0.236403 m (Made tweak to model, didnt seem to make an impact)
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2026-02-01_20-12-32\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"

    #5 window length, 5 step stride,  RMSE 0.063310 m, (Same new model, similar result to first training)
    #NPZ_PATH   = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_wrapped.npz"
    #JOINT_CKPT = r"C:\Users\liamm\world_model\joint_training\results\2026-02-01_22-14-00\joint_best.pt"
    #VAE_CKPT   = r"C:\Users\liamm\world_model\vae_training\results\npz\checkpoints\best.pth"

    
    NPZ_PATH   = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"
    JOINT_CKPT = r"C:\Users\liamm\world_model_2\joint_training\results\2026-02-16_17-38-06\joint_best.pt"
    VAE_CKPT   = r"C:\Users\liamm\world_model_2\vae_training\results\npz\checkpoints\best.pth"



    # ----- Load frozen VAE -----
    vae_ckpt = torch.load(VAE_CKPT, map_location=DEVICE)
    vae_cfg = vae_ckpt["config"]

    vae = VAE(
        latent_dim=vae_cfg["latent_dim"],
        input_channels=vae_cfg["input_channels"],
    ).to(DEVICE)

    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ----- Load joint (predictor + extractor) -----
    joint = torch.load(JOINT_CKPT, map_location=DEVICE)
    D = vae_cfg["latent_dim"]

    predictor = LSTMPredictor(
        latent_dim=D,
        action_dim=3,
        hidden_dim=128,
        num_layers=2,
    ).to(DEVICE)

    extractor = StateExtractor(
        latent_dim=D,
        output_dim=2,     # predicts ONLY [x_rel, y_rel]
        hidden_dims=[128, 64],
    ).to(DEVICE)

    predictor.load_state_dict(joint["predictor"])
    extractor.load_state_dict(joint["extractor"])
    predictor.eval()
    extractor.eval()

    # ----- Dataset -----
    ds = NPZRelativePredictorDataset(
        NPZ_PATH,
        vae_model=vae,
        device=DEVICE,
        resize_to=(120, 160),
    )

    meta = ds.meta
    global_state = meta["state"].astype(np.float32)   # (T,4) -> [x, y, yaw_deg, v]
    seq_len = ds.seq_len                             # should be 5

    T = global_state.shape[0]
    N = len(ds)
    stride = T // N                                  # expected 5
    print(f"T={T}, N={N}, seq_len={seq_len}, stride={stride}")

    xs = global_state[:, 0]
    ys = global_state[:, 1]

    # ----- find first lap -----
    lap_end = find_first_lap(xs, ys)
    print(f"First lap ends at global index {lap_end} / {T}")

    true_global = []
    pred_global = []

    # use every window whose target frame is inside the first lap
    for idx in range(N):
        origin_idx = idx * stride
        target_idx = origin_idx + (seq_len - 1)
        if target_idx >= lap_end:
            break  # stop once we leave the first lap

        # one training example (relative window)
        z_ctx, a_ctx, z_tgt, pose_rel_tgt = ds[idx]

        # ----- true global pose at the 5th step -----
        gt_pose = global_state[target_idx]          # [xg, yg, yawg_deg, v]
        xg, yg, yawg_deg, _ = gt_pose
        true_global.append([xg, yg, yawg_deg])

        # ----- model prediction (relative) -----
        with torch.no_grad():
            z_ctx_b = z_ctx.unsqueeze(0).to(DEVICE)
            a_ctx_b = a_ctx.unsqueeze(0).to(DEVICE)

            z_pred_seq, _ = predictor(z_ctx_b, a_ctx_b)  # (1,4,D)
            z_pred = z_pred_seq[:, -1, :]                # (1,D)
            rel_pred = extractor(z_pred)[0].cpu().numpy()  # [x_rel_hat, y_rel_hat]

        origin = global_state[origin_idx][:3]       # [x0, y0, yaw0_deg]
        xgp, ygp = rel_to_global(rel_pred, origin)

        # use TRUE TARGET yaw for visualization
        pred_global.append([xgp, ygp, yawg_deg])

    true_global = np.array(true_global)  # (M,3)
    pred_global = np.array(pred_global)  # (M,3)

    # ================================
    # ==== ERROR STATISTICS ==========
    # ================================
    true_xy = true_global[:, :2]   # (M,2)
    pred_xy = pred_global[:, :2]

    dx = pred_xy[:, 0] - true_xy[:, 0]
    dy = pred_xy[:, 1] - true_xy[:, 1]

    # per-point Euclidean distance error
    dist_err = np.sqrt(dx**2 + dy**2)

    mse_x  = numpy_mse(dx)
    rmse_x = numpy_rmse(dx)
    mse_y  = numpy_mse(dy)
    rmse_y = numpy_rmse(dy)
    mse_d  = numpy_mse(dist_err)
    rmse_d = numpy_rmse(dist_err)

    mean_d   = float(dist_err.mean())
    median_d = float(np.median(dist_err))
    max_d    = float(dist_err.max())
    p95_d    = float(np.percentile(dist_err, 95))

    print("\n===== First-lap error stats (learned model, meters) =====")
    print(f"X:  MSE {mse_x:.6f} m^2, RMSE {rmse_x:.6f} m")
    print(f"Y:  MSE {mse_y:.6f} m^2, RMSE {rmse_y:.6f} m")
    print(f"XY distance: MSE {mse_d:.6f} m^2, RMSE {rmse_d:.6f} m")
    print(f"Distance error: mean   {mean_d:.6f} m")
    print(f"                  median {median_d:.6f} m")
    print(f"                  95th   {p95_d:.6f} m")
    print(f"                  max    {max_d:.6f} m")
    print("=========================================================\n")

    # ----- Plot -----
    plt.figure(figsize=(8, 7))

    # full global lap path as a line for context
    plt.plot(xs[:lap_end], ys[:lap_end], "k-", linewidth=1.0, alpha=0.4,
             label="Global lap (path)")

    plot_states(true_global, "b", "True (lap, every stride)")
    plot_states(pred_global, "r", "Pred (lap, every stride, true yaw)")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (global)")
    plt.ylabel("y (global)")
    plt.title("Relative-trained model: predictions over first full lap (global coords)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
