# check_wrapped_npz_schema_hardcoded.py
import sys
import numpy as np

# ==========================================================
# EDIT THESE TWO PATHS
# ==========================================================
FILE_A = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_20_step5_wrapped.npz"
FILE_B = r"C:\Users\liamm\world_model_2\debug_npz\traj2_relative_15_step5_wrapped_like_A.npz"

# Optional toggles
COMPARE_SHAPES_DTYPES = True   # also compare shapes/dtypes for common keys
STRICT_SHAPES_DTYPES  = False  # if True, mismatches become a failure exit code
# ==========================================================


def unwrap_zero_dim_object(x):
    """Unwrap 0-d or size-1 object arrays (common when saving dicts in npz)."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x


def load_wrapped_dict(npz_path: str):
    npz = np.load(npz_path, allow_pickle=True)
    if "data" not in npz.files:
        raise KeyError(f"Missing top-level key 'data'. Top-level keys: {list(npz.files)}")

    d = unwrap_zero_dim_object(npz["data"])
    if not isinstance(d, dict):
        raise TypeError(f"'data' did not unwrap to dict. Got type={type(d)}")
    return d


def describe_value(v):
    v = unwrap_zero_dim_object(v)
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape} dtype={v.dtype}"
    return f"{type(v).__name__}"


def main():
    ok = True

    # ---------- load ----------
    try:
        a = load_wrapped_dict(FILE_A)
        print(f"[OK] A has top-level 'data' dict with {len(a)} subkeys:\n  {sorted(a.keys())}")
    except Exception as e:
        print(f"[FAIL] A ({FILE_A}): {e}")
        sys.exit(1)

    try:
        b = load_wrapped_dict(FILE_B)
        print(f"\n[OK] B has top-level 'data' dict with {len(b)} subkeys:\n  {sorted(b.keys())}")
    except Exception as e:
        print(f"[FAIL] B ({FILE_B}): {e}")
        sys.exit(1)

    # ---------- compare subkeys ----------
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    if only_a or only_b:
        ok = False
        print("\n[FAIL] Subkey mismatch under 'data':")
        if only_a:
            print(f"  Keys only in A ({len(only_a)}): {only_a}")
        if only_b:
            print(f"  Keys only in B ({len(only_b)}): {only_b}")
    else:
        print("\n[OK] Both files have identical subkeys under 'data'.")

    # ---------- optional: compare shapes/dtypes ----------
    if COMPARE_SHAPES_DTYPES:
        print("\n[INFO] Comparing shapes/dtypes for common keys (best-effort):")
        shape_mismatches = []
        type_mismatches = []

        for k in common:
            va = unwrap_zero_dim_object(a[k])
            vb = unwrap_zero_dim_object(b[k])

            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                if va.shape != vb.shape or va.dtype != vb.dtype:
                    shape_mismatches.append(
                        (k, f"A: shape={va.shape}, dtype={va.dtype}", f"B: shape={vb.shape}, dtype={vb.dtype}")
                    )
            else:
                if type(va) != type(vb):
                    type_mismatches.append((k, f"A: {describe_value(va)}", f"B: {describe_value(vb)}"))

        if not shape_mismatches and not type_mismatches:
            print("  [OK] All common keys match in shape/dtype (arrays) and type (non-arrays).")
        else:
            if shape_mismatches:
                for (k, da, db) in shape_mismatches:
                    print(f"  [WARN] {k}: {da} | {db}")
            if type_mismatches:
                for (k, da, db) in type_mismatches:
                    print(f"  [WARN] {k}: {da} | {db}")

            if STRICT_SHAPES_DTYPES:
                ok = False
                print("\n[FAIL] STRICT_SHAPES_DTYPES=True → treating mismatches as failure.")
            else:
                print("\n[WARN] Mismatches found, but STRICT_SHAPES_DTYPES=False → not a hard failure.")

    # ---------- final ----------
    if ok:
        print("\n✅ Schema check PASSED.")
        sys.exit(0)
    else:
        print("\n❌ Schema check FAILED.")
        sys.exit(2)


if __name__ == "__main__":
    main()
