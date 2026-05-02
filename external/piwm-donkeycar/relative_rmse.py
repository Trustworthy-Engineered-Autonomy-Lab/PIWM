import os
import sys
import math
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

# Must match training
CTX_LEN = 14
STRIDE_STEP = 5
RESIZE_TO = (120, 160)
ACTION_PAD_TO = 3  # same as training

IMG_KEY   = "frame"
ACT_KEY   = "action"
STATE_KEY = "state"

# -------------------------------------------------------
# Helpers (must match train_joint_full.py)
# -------------------------------------------------------
def to_chw_float01(img: np.ndarray, resize_to=(120,160), input_channels=3) -> torch.Tensor:
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    # HWC->CHW if needed
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
    dx = float(x_t - x0)
    dy = float(y_t - y0)
    yaw0 = math.radians(float(yaw0_deg))
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    xr =  c * dx + s * dy
    yr = -s * dx + c * dy
    return xr, yr

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    joint = torch.load(JOINT_CKPT, map_location=DEVICE)
    if "vae" not in joint or "predictor" not in joint or "extractor" not in joint or "vae_cfg" not in joint:
        raise KeyError(f"JOINT_CKPT missing keys. Have: {list(joint.keys())}")

    vae_cfg = joint["vae_cfg"]
    latent_dim = int(vae_cfg["latent_dim"])
    input_channels = int(vae_cfg.get("input_channels", 3))

    # Load VAE from JOINT checkpoint (critical!)
    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(DEVICE)
    vae.load_state_dict(joint["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Infer action_dim used in training from saved train_cfg if present
    train_cfg = joint.get("train_cfg", {})
    action_pad_to = int(train_cfg.get("ACTION_PAD_TO", ACTION_PAD_TO))

    # Build predictor/extractor (must match checkpoint action_dim)
    action_dim = int(action_pad_to) if action_pad_to is not None else 2  # default fallback
    predictor = LSTMPredictor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=128, num_layers=2).to(DEVICE)
    extractor = StateExtractor(latent_dim=latent_dim, output_dim=2, hidden_dims=[128, 64]).to(DEVICE)

    predictor.load_state_dict(joint["predictor"])
    extractor.load_state_dict(joint["extractor"])
    predictor.eval()
    extractor.eval()

    # Load NPZ raw dict
    d = np.load(NPZ_PATH, allow_pickle=True)
    raw = d["data"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    frames = raw[IMG_KEY]    # (T,3,224,224) uint8
    actions = raw[ACT_KEY]   # (T,2) float32
    state = raw[STATE_KEY]   # (T,4) float32 [x,y,yaw_deg,v]

    T_total = int(frames.shape[0])
    needed = CTX_LEN + 1
    t0s = list(range(0, T_total - needed + 1, STRIDE_STEP))

    errs = []

    with torch.no_grad():
        for t0 in t0s:
            t1 = t0 + CTX_LEN

            # Build x_ctx, x_tgt exactly like training
            x_ctx_np = frames[t0:t1]
            x_tgt_np = frames[t1]

            x_ctx = torch.stack([to_chw_float01(x_ctx_np[j], RESIZE_TO, input_channels) for j in range(CTX_LEN)], dim=0)
            x_tgt = to_chw_float01(x_tgt_np, RESIZE_TO, input_channels)

            # Actions (pad to 3 if training did)
            a_ctx = actions[t0:t1].astype(np.float32)  # (CTX_LEN,2)
            if action_pad_to is not None and a_ctx.shape[1] < action_pad_to:
                pad = action_pad_to - a_ctx.shape[1]
                a_ctx = np.pad(a_ctx, ((0,0),(0,pad)), mode="constant", constant_values=0.0)

            # Relative target from state
            x0, y0, yaw0 = state[t0,0], state[t0,1], state[t0,2]
            xt, yt = state[t1,0], state[t1,1]
            xr, yr = global_to_rel_xy(xt, yt, x0, y0, yaw0)
            xy_true = np.array([xr, yr], dtype=np.float32)

            # VAE encode ctx + tgt using mu like training
            # ctx
            x_ctx_b = x_ctx.unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
            B, Tctx, C, H, W = x_ctx_b.shape
            x_ctx_flat = x_ctx_b.view(B*Tctx, C, H, W)
            _, mu_ctx, _ = vae(x_ctx_flat)
            z_ctx = mu_ctx.view(B, Tctx, -1)  # (1,T,D)

            # tgt
            x_tgt_b = x_tgt.unsqueeze(0).to(DEVICE)  # (1,C,H,W)
            _, mu_tgt, _ = vae(x_tgt_b)
            z_tgt = mu_tgt  # (1,D) (not used for rmse, but matches training)

            # predictor + extractor
            a_ctx_b = torch.from_numpy(a_ctx).unsqueeze(0).to(DEVICE)  # (1,T,A)
            z_pred, _ = predictor.forward_last(z_ctx, a_ctx_b)         # (1,D)
            xy_pred = extractor(z_pred)[0].cpu().numpy()               # (2,)

            errs.append(xy_pred - xy_true)

    errs = np.asarray(errs, dtype=np.float32)
    dx, dy = errs[:,0], errs[:,1]
    dist = np.sqrt(dx*dx + dy*dy)

    rmse_x = float(np.sqrt(np.mean(dx*dx)))
    rmse_y = float(np.sqrt(np.mean(dy*dy)))
    rmse_xy = float(np.sqrt(np.mean(dx*dx + dy*dy)))

    print("\n===== RELATIVE RMSE (MATCHING train_joint_full.py) =====")
    print(f"samples    : {len(errs)}")
    print(f"RMSE X     : {rmse_x:.6f} m")
    print(f"RMSE Y     : {rmse_y:.6f} m")
    print(f"RMSE XY    : {rmse_xy:.6f} m")
    print("\nDistance error stats:")
    print(f"Mean       : {float(dist.mean()):.6f} m")
    print(f"Median     : {float(np.median(dist)):.6f} m")
    print(f"95th pct   : {float(np.percentile(dist, 95)):.6f} m")
    print(f"Max        : {float(dist.max()):.6f} m")
    print("========================================================\n")

if __name__ == "__main__":
    main()
