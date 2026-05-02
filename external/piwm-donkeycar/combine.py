import numpy as np

A = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_wrapped.npz"
B = r"C:\Users\liamm\world_model\debug_npz\traj2_relative_15_wrapped_like_A.npz"

def unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    return x

def summarize(name, path):
    z = np.load(path, allow_pickle=True)
    base = unwrap(z["data"])
    print(f"\n===== {name} =====")
    print("type(base):", type(base))

    if isinstance(base, dict):
        print("keys:", list(base.keys()))
        for k, v in base.items():
            vv = unwrap(v)
            shp = getattr(vv, "shape", None)
            dt  = getattr(vv, "dtype", None)
            print(f"  {k:12s} type={type(vv).__name__} shape={shp} dtype={dt}")
    elif isinstance(base, np.ndarray):
        print("ndarray shape:", base.shape, "dtype:", base.dtype)
    else:
        print("base repr:", repr(base)[:200])

summarize("A (wrapped)", A)
summarize("B (15_DICT)", B)
