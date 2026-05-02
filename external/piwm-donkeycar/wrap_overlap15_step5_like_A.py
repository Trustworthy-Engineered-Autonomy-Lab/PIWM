import numpy as np

# ==========================================================
# EDIT THESE PATHS
# ==========================================================
# Base file that contains the TRUE global arrays (frame/state/timestamps/action/etc)
BASE_WRAPPED_A = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_wrapped.npz"

# Your 2026 windowed file: data is ndarray (N, 15, 5) dtype=object
WINDOWED_2026  = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_50_step5.npz"

# Output: wrapped like A, but rel_state is (N,15,4) with overlap every 5
OUT_PATH       = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_50_step5_wrapped.npz"
# ==========================================================

SEQ_LEN_EXPECTED = 50
STRIDE_STEP_EXPECTED = 5  # overlap every 5 frames


def unwrap_zero_dim_object(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    return x


def main():
    # ----------------------------
    # Load base wrapped (true global)
    # ----------------------------
    a = np.load(BASE_WRAPPED_A, allow_pickle=True)
    if "data" not in a.files:
        raise RuntimeError(f"BASE_WRAPPED_A missing key 'data'. keys={a.files}")

    base = unwrap_zero_dim_object(a["data"])
    if not isinstance(base, dict):
        raise RuntimeError(f"BASE_WRAPPED_A['data'] did not unwrap to dict. type={type(base)}")

    for k in ["frame", "timestamps", "state"]:
        if k not in base:
            raise RuntimeError(f"BASE_WRAPPED_A missing '{k}'. keys={list(base.keys())}")

    T = int(base["frame"].shape[0])
    print(f"[base] Loaded global timeline: T={T}")
    print(f"[base] frame={base['frame'].shape} state={base['state'].shape} timestamps={base['timestamps'].shape}")

    # ----------------------------
    # Load 2026 windowed file (N,15,5)
    # ----------------------------
    b = np.load(WINDOWED_2026, allow_pickle=True)
    if "data" not in b.files:
        raise RuntimeError(f"WINDOWED_2026 missing key 'data'. keys={b.files}")

    win = unwrap_zero_dim_object(b["data"])
    if not isinstance(win, np.ndarray) or win.ndim != 3:
        raise RuntimeError(f"WINDOWED_2026['data'] must be ndarray (N,seq_len,fields). got type={type(win)} shape={getattr(win,'shape',None)}")

    N, seq_len, fields = win.shape
    print(f"[2026] windowed data: N={N}, seq_len={seq_len}, fields={fields}")

    if seq_len != SEQ_LEN_EXPECTED:
        raise RuntimeError(f"Expected seq_len={SEQ_LEN_EXPECTED}, but got {seq_len}")

    if fields < 3:
        raise RuntimeError("Expected at least 3 fields per step: [frame, timestamp, rel_pose, ...]")

    # ----------------------------
    # Determine stride_step
    # ----------------------------
    # If your windows are generated from a timeline of length T with stride_step,
    # then N ≈ floor((T - seq_len)/stride_step) + 1
    # => stride_step ≈ (T - seq_len) / (N - 1)
    if N <= 1:
        inferred_stride = seq_len
    else:
        inferred_stride = int(round((T - seq_len) / (N - 1)))
    inferred_stride = max(1, inferred_stride)

    print(f"[infer] inferred stride_step ≈ {inferred_stride}")

    # You said you want overlap every 5 frames.
    # If inference disagrees, we still *force* stride_step=5 for indexing global speed,
    # but we warn loudly.
    stride_step = STRIDE_STEP_EXPECTED
    if inferred_stride != STRIDE_STEP_EXPECTED:
        print(f"[WARN] inferred stride_step={inferred_stride} but expected {STRIDE_STEP_EXPECTED}.")
        print("[WARN] Proceeding with stride_step=5 anyway (as you requested).")
        print("[WARN] If global speed looks misaligned, regenerate the windowed dataset with stride_step=5.")

    # ----------------------------
    # Build rel_state (N,15,4)
    # rel_state[i,t,:3] = rel_pose from win[i,t][2]
    # rel_state[i,t, 3] = speed from base['state'][origin_idx+t, 3]
    # ----------------------------
    rel_state = np.zeros((N, seq_len, 4), dtype=np.float32)

    # sanity counters
    bad_rel_pose = 0
    bad_speed = 0

    for i in range(N):
        origin_idx = i * stride_step
        for t in range(seq_len):
            step = win[i, t]

            rp = np.asarray(step[2], dtype=np.float32).reshape(-1)
            if rp.shape[0] < 3 or not np.isfinite(rp[:3]).all():
                bad_rel_pose += 1
                rel_state[i, t, :3] = 0.0
            else:
                rel_state[i, t, :3] = rp[:3]

            gidx = origin_idx + t
            if 0 <= gidx < base["state"].shape[0] and base["state"].shape[1] >= 4:
                v = base["state"][gidx, 3]
                if np.isfinite(v):
                    rel_state[i, t, 3] = np.float32(v)
                else:
                    bad_speed += 1
                    rel_state[i, t, 3] = 0.0
            else:
                bad_speed += 1
                rel_state[i, t, 3] = 0.0

    print(f"[build] rel_state built: shape={rel_state.shape}")
    if bad_rel_pose:
        print(f"[warn] bad rel_pose entries: {bad_rel_pose} (set to 0)")
    if bad_speed:
        print(f"[warn] bad/missing speed entries: {bad_speed} (set to 0)")

    # ----------------------------
    # Patch base dict (keep true global arrays!)
    # ----------------------------
    base["rel_state"] = rel_state
    base["rel_window"] = np.array(seq_len, dtype=np.int64)
    base["rel_stride_step"] = np.array(stride_step, dtype=np.int64)  # helpful metadata

    # ----------------------------
    # Save wrapped like A (small)
    # ----------------------------
    wrapped = np.array(base, dtype=object)  # 0-d object array holding dict
    np.savez(OUT_PATH, data=wrapped)

    print("\n✅ Saved:", OUT_PATH)
    print("   frame:", base["frame"].shape, base["frame"].dtype)
    print("   state:", base["state"].shape, base["state"].dtype)
    print("   rel_state:", base["rel_state"].shape, base["rel_state"].dtype)
    print("   rel_window:", int(base["rel_window"]))
    print("   rel_stride_step:", int(base["rel_stride_step"]))


if __name__ == "__main__":
    main()
