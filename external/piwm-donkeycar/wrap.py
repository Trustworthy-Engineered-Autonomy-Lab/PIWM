import numpy as np

IN_PATH  = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_1-31-2026.npz"   # your B file
OUT_PATH = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_1-31-2026_wrapped.npz"

def unwrap_zero_dim_object(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    return x

def main():
    d = np.load(IN_PATH, allow_pickle=True)
    if "data" not in d.files:
        raise RuntimeError(f"Expected top-level key 'data'. Found: {d.files}")

    base = unwrap_zero_dim_object(d["data"])

    # We expect B-format: ndarray (N, seq_len, fields)
    if not isinstance(base, np.ndarray) or base.ndim != 3:
        raise RuntimeError(f"'data' must be an ndarray (N, seq_len, fields). Got type={type(base)} shape={getattr(base,'shape',None)}")

    N, seq_len, fields = base.shape
    if fields < 5:
        raise RuntimeError(f"Expected at least 5 fields per timestep. Got fields={fields}")

    # Field layout (based on what you printed):
    # [0]=frame (3,224,224) uint8
    # [1]=timestamp float32
    # [2]=rel_pose (3,) float32  [x_rel, y_rel, yaw_rel]
    # [3]=action   (2,) float32  [steer, throttle] (very likely)
    # [4]=state    (4,) float32  [x, y, yaw, speed] (very likely)
    frames = []
    times  = []
    acts   = []
    states = []
    rel_pose = np.zeros((N, seq_len, 3), dtype=np.float32)
    rel_state = np.zeros((N, seq_len, 4), dtype=np.float32)

    for i in range(N):
        for t in range(seq_len):
            step = base[i, t]
            frame = step[0]
            ts    = step[1]
            rp    = step[2]
            act   = step[3]
            st    = step[4]

            frames.append(frame)
            times.append(np.float32(ts))
            acts.append(np.asarray(act, dtype=np.float32))
            states.append(np.asarray(st, dtype=np.float32))

            rp = np.asarray(rp, dtype=np.float32).reshape(-1)
            if rp.shape[0] < 3:
                raise RuntimeError(f"rel_pose at (i={i},t={t}) has shape {rp.shape}, expected >=3")

            rel_pose[i, t, :3] = rp[:3]
            rel_state[i, t, :3] = rp[:3]
            # 4th dim: use speed from global state if available, else 0
            if np.asarray(st).shape[0] >= 4:
                rel_state[i, t, 3] = np.float32(st[3])
            else:
                rel_state[i, t, 3] = 0.0

    frame_arr = np.stack(frames, axis=0)                        # (T, 3, H, W)
    timestamps = np.asarray(times, dtype=np.float32)            # (T,)
    action_arr = np.stack(acts, axis=0).astype(np.float32)      # (T,2)
    state_arr  = np.stack(states, axis=0).astype(np.float32)    # (T,4)

    # fps estimate from timestamps if possible
    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 1e-6)]
        fps = np.float32(1.0 / np.median(dt)) if dt.size else np.float32(0.0)
    else:
        fps = np.float32(0.0)

    base_dict = {
        "frame": frame_arr,
        "fps": np.array(fps, dtype=np.float32),
        "timestamps": timestamps,
        "channels_first": np.array(True, dtype=bool),
        "shape": np.array(frame_arr.shape[1:], dtype=np.int32),     # e.g. (3,224,224)
        "state": state_arr,
        "interpolated": np.zeros((frame_arr.shape[0],), dtype=np.int32),
        "cte": np.zeros((frame_arr.shape[0],), dtype=np.float32),
        "map_world": np.zeros((0, 2), dtype=np.float32),
        "action": action_arr,
        "rel_state": rel_state,                                     # (N,seq_len,4)
        "rel_window": np.array(seq_len, dtype=np.int64),
    }

    # IMPORTANT: wrap dict inside 0-d object array under key 'data'
    wrapped = np.array(base_dict, dtype=object)
    np.savez(OUT_PATH, data=wrapped)

    print("Saved:", OUT_PATH)
    print("Top-level keys:", ["data"])
    print("Unwrapped keys:", list(base_dict.keys()))
    print("frame:", base_dict["frame"].shape, base_dict["frame"].dtype)
    print("rel_state:", base_dict["rel_state"].shape, base_dict["rel_state"].dtype)
    print("action:", base_dict["action"].shape, base_dict["action"].dtype)
    print("state:", base_dict["state"].shape, base_dict["state"].dtype)
    print("rel_window:", int(base_dict["rel_window"]))

if __name__ == "__main__":
    main()
