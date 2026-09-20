import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.track_tools.minimonaco import DEFAULT_REGION_JSON_PATH, load_region_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


REQUIRED_COLUMNS = {
    "timestamp_ms",
    "steering_act",
    "throttle_act",
    "pos_x",
    "pos_z",
    "speed",
    "yaw",
    "anomaly_param",
    "anomaly_intensity",
}


def parse_float(value):
    if value is None or value == "":
        return np.nan
    return float(value)


def clean_set(value):
    if not value or value == "{}":
        return "nominal"
    return value.strip("{}").replace(",", "+") or "nominal"


def safe_filename_part(value):
    text = str(value).strip() or "unknown"
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "unknown"


def read_log(path, control_mode, skip_frames=0, max_step_distance=None):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) < 3:
        return None

    if control_mode == "delayed":
        steer_col = "steering_delayed"
        throttle_col = "throttle_delayed"
    elif control_mode == "cmd":
        steer_col = "steering_cmd"
        throttle_col = "throttle_cmd"
    else:
        steer_col = "steering_act"
        throttle_col = "throttle_act"

    data = {
        "timestamp_ms": np.array([parse_float(r["timestamp_ms"]) for r in rows], dtype=float),
        "x": np.array([parse_float(r["pos_x"]) for r in rows], dtype=float),
        "z": np.array([parse_float(r["pos_z"]) for r in rows], dtype=float),
        "yaw": np.deg2rad(np.array([parse_float(r["yaw"]) for r in rows], dtype=float)),
        "speed": np.array([parse_float(r["speed"]) for r in rows], dtype=float),
        "steer": np.array([parse_float(r.get(steer_col, r["steering_act"])) for r in rows], dtype=float),
        "throttle": np.array([parse_float(r.get(throttle_col, r["throttle_act"])) for r in rows], dtype=float),
    }

    mask = np.ones(len(rows), dtype=bool)
    for values in data.values():
        mask &= np.isfinite(values)
    for key in data:
        data[key] = data[key][mask]

    if skip_frames > 0:
        for key in data:
            data[key] = data[key][skip_frames:]

    if len(data["x"]) < 3:
        return None

    if max_step_distance is not None:
        step_dist = np.hypot(np.diff(data["x"]), np.diff(data["z"]))
        jumps = np.flatnonzero(step_dist > max_step_distance)
        if len(jumps) > 0:
            # Keep the longest continuous segment split by large position jumps.
            cuts = np.r_[0, jumps + 1, len(data["x"])]
            starts = cuts[:-1]
            ends = cuts[1:]
            lengths = ends - starts
            best = int(np.argmax(lengths))
            start, end = int(starts[best]), int(ends[best])
            for key in data:
                data[key] = data[key][start:end]

    if len(data["x"]) < 3:
        return None

    data["yaw"] = np.unwrap(data["yaw"])
    dt = np.diff(data["timestamp_ms"]) / 1000.0
    good_dt = np.isfinite(dt) & (dt > 0.001) & (dt < 0.5)
    if not np.any(good_dt):
        return None
    fill_dt = float(np.median(dt[good_dt]))
    dt = np.where(good_dt, dt, fill_dt)
    data["dt"] = dt
    data["path"] = str(path)
    data["case"] = clean_set(rows[0].get("anomaly_param", "{}"))
    data["intensity"] = clean_set(rows[0].get("anomaly_intensity", "{}"))
    return data


def discover_logs(trials_dir):
    return sorted(Path(trials_dir).glob("**/runs/*/log_*.csv"))


def apply_steering_delay(logs, delay_frames):
    if delay_frames <= 0:
        return
    for log in logs:
        steer = log["steer"]
        if len(steer) == 0:
            continue
        delayed = np.empty_like(steer)
        delayed[:delay_frames] = steer[0]
        delayed[delay_frames:] = steer[:-delay_frames]
        log["steer"] = delayed


def heading_candidates(yaw):
    return {
        "cos_sin": (np.cos(yaw), np.sin(yaw)),
        "cos_neg_sin": (np.cos(yaw), -np.sin(yaw)),
        "neg_cos_sin": (-np.cos(yaw), np.sin(yaw)),
        "neg_cos_neg_sin": (-np.cos(yaw), -np.sin(yaw)),
        "sin_cos": (np.sin(yaw), np.cos(yaw)),
        "sin_neg_cos": (np.sin(yaw), -np.cos(yaw)),
        "neg_sin_cos": (-np.sin(yaw), np.cos(yaw)),
        "neg_sin_neg_cos": (-np.sin(yaw), -np.cos(yaw)),
    }


def step(state, steer, throttle, dt, params):
    x, z, yaw, speed, steer_state = state
    hx, hz = heading_candidates(np.array([yaw]))[params["heading_convention"]]
    x_next = x + params["x_speed_scale"] * speed * float(hx[0]) * dt + params["x_bias"] * dt
    z_next = z + params["z_speed_scale"] * speed * float(hz[0]) * dt + params["z_bias"] * dt
    effective_steer_cmd = steer / (1.0 + params["steering_staturation"] * abs(steer))
    steer_state = steer_state + params["steer_response"] * (effective_steer_cmd - steer_state)
    yaw_next = yaw + params["steer_over_wheelbase"] * speed * steer_state * dt + params["yaw_bias"] * dt
    speed_next = speed + (
        params["throttle_accel_gain"] * throttle
        + params["speed_drag_gain"] * speed
        + params["accel_bias"]
    ) * dt
    return np.array([x_next, z_next, yaw_next, max(0.0, speed_next), steer_state], dtype=float)

def default_bicycle_params():
    return {
        "heading_convention": "sin_cos",
        "x_speed_scale": 0.9,
        "x_bias": 0.0,
        "z_speed_scale": 0.9,
        "z_bias": 0.0,
        "steer_over_wheelbase": 1.395,
        "yaw_bias": 0.0,
        "throttle_accel_gain": 14.7,
        "speed_drag_gain": -0.18,
        "accel_bias": -1.2,
        "steering_staturation": 0.4,
        "steer_response": 0.3,
    }

# # Params are the modifications to better fit the data
# # n is the number of steps forward we are looking
# # start is the state information at the start of this rollout
# def evaluate_next_n_steps(n, params, start):
#     base_state = np.array([start["x"], start["z"], start["yaw"], start["speed"], start["steer"]], dtype=float)
#     rollout = np.zeros((n, 5), dtype=float)
#     rollout[0] = base_state
#     for i in range(n):
#         input = #Get controller input
#         rollout[i+1] = step(rollout[i], input["steer"][i], input["throttle"][i], input["dt"][i], params)
#         #function that sends rollout position back to the controller
#     return #Depends what we want to track and where

def finite_metric(values, reducer):
    if len(values) == 0:
        return float("nan")
    return float(reducer(values))


def finite_mean(values):
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def evaluate_log(log, params, region):
    n = len(log["x"])
    one_step = np.zeros((n - 1, 5), dtype=float)
    rollout = np.zeros((n, 5), dtype=float)
    rollout[0] = [log["x"][0], log["z"][0], log["yaw"][0], log["speed"][0], log["steer"][0]]
    crash_step = None
    crash_margin = None

    initial_track_result = region.check_point(rollout[0][0], rollout[0][1])
    if not initial_track_result["inside"]:
        crash_step = 0
        crash_margin = float(initial_track_result["signed_margin"])

    for i in range(n - 1):
        if crash_step is not None:
            break
        measured_state = np.array([log["x"][i], log["z"][i], log["yaw"][i], log["speed"][i], log["steer"][i]], dtype=float)
        one_step[i] = step(measured_state, log["steer"][i], log["throttle"][i], log["dt"][i], params)
        rollout[i + 1] = step(rollout[i], log["steer"][i], log["throttle"][i], log["dt"][i], params)
        track_result = region.check_point(rollout[i + 1][0], rollout[i + 1][1])
        if not track_result["inside"]:
            crash_step = i + 1
            crash_margin = float(track_result["signed_margin"])
            break

    valid_n = crash_step + 1 if crash_step is not None else n
    valid_rollout = rollout[:valid_n]
    valid_one_step_n = max(0, valid_n - 1)

    truth_next = np.column_stack([
        log["x"][1:valid_n],
        log["z"][1:valid_n],
        log["yaw"][1:valid_n],
        log["speed"][1:valid_n],
    ])
    one_pos_err = np.linalg.norm(one_step[:valid_one_step_n, :2] - truth_next[:, :2], axis=1)
    roll_pos_err = np.linalg.norm(
        valid_rollout[:, :2] - np.column_stack([log["x"][:valid_n], log["z"][:valid_n]]),
        axis=1,
    )
    return {
        "one_step_pos_rmse": finite_metric(one_pos_err, lambda v: np.sqrt(np.mean(v**2))),
        "one_step_pos_mae": finite_metric(one_pos_err, np.mean),
        "one_step_pos_p50": finite_metric(one_pos_err, lambda v: np.percentile(v, 50)),
        "one_step_pos_p95": finite_metric(one_pos_err, lambda v: np.percentile(v, 95)),
        "one_step_pos_max": finite_metric(one_pos_err, np.max),
        "rollout_pos_rmse": finite_metric(roll_pos_err, lambda v: np.sqrt(np.mean(v**2))),
        "rollout_pos_mae": finite_metric(roll_pos_err, np.mean),
        "rollout_pos_p50": finite_metric(roll_pos_err, lambda v: np.percentile(v, 50)),
        "rollout_pos_p95": finite_metric(roll_pos_err, lambda v: np.percentile(v, 95)),
        "rollout_final_pos_error": finite_metric(roll_pos_err, lambda v: v[-1]),
        "frames": int(valid_n),
        "original_frames": int(n),
        "crashed": crash_step is not None,
        "crash_step": "" if crash_step is None else int(crash_step),
        "crash_margin": "" if crash_margin is None else crash_margin,
        "rollout": valid_rollout,
    }


def group_summary(rows):
    groups = {}
    for row in rows:
        key = (row["case"], row["intensity"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (case, intensity), values in sorted(groups.items()):
        summary.append({
            "case": case,
            "intensity": intensity,
            "runs": len(values),
            "frames": int(sum(v["frames"] for v in values)),
            "original_frames": int(sum(v["original_frames"] for v in values)),
            "crashes": int(sum(1 for v in values if v["crashed"])),
            "one_step_pos_rmse": finite_mean([v["one_step_pos_rmse"] for v in values]),
            "one_step_pos_mae": finite_mean([v["one_step_pos_mae"] for v in values]),
            "one_step_pos_p50": finite_mean([v["one_step_pos_p50"] for v in values]),
            "one_step_pos_p95": finite_mean([v["one_step_pos_p95"] for v in values]),
            "one_step_pos_max": finite_mean([v["one_step_pos_max"] for v in values]),
            "rollout_pos_rmse": finite_mean([v["rollout_pos_rmse"] for v in values]),
            "rollout_pos_mae": finite_mean([v["rollout_pos_mae"] for v in values]),
            "rollout_pos_p50": finite_mean([v["rollout_pos_p50"] for v in values]),
            "rollout_pos_p95": finite_mean([v["rollout_pos_p95"] for v in values]),
            "rollout_final_pos_error": finite_mean([v["rollout_final_pos_error"] for v in values]),
        })
    return summary


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def plot_rollouts(output_dir, evaluated, limit):
    if plt is None:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (log, metrics) in enumerate(evaluated[:limit]):
        pred = metrics["rollout"]
        frames = metrics["frames"]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(log["x"][:frames], log["z"][:frames], label="actual", linewidth=2)
        ax.plot(pred[:, 0], pred[:, 1], label="bicycle rollout", linewidth=2)
        if metrics["crashed"]:
            ax.scatter(pred[-1, 0], pred[-1, 1], color="red", s=28, label="predicted crash")
        title = f"{log['case']} {log['intensity']} run {idx}"
        if metrics["crashed"]:
            title += f" — predicted crash at frame {metrics['crash_step']}"
        ax.set_title(title)
        ax.set_xlabel("pos_x")
        ax.set_ylabel("pos_z")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        case = safe_filename_part(log["case"])
        intensity = safe_filename_part(log["intensity"])
        fig.savefig(output_dir / f"rollout_{idx:03d}_{case}_{intensity}.png", dpi=140)
        plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate an effective bicycle model on DonkeyCar trial logs.")
    parser.add_argument("--trials-dir", default=r"C:\Users\Jack\physical-donkeycar-anomalies2\phda\physical-donkeycar-anomalies\trials")
    parser.add_argument("--output-dir", default="outputs/bicycle_eval")
    parser.add_argument("--control-mode", choices=["delayed", "actual", "cmd"], default="delayed")
    parser.add_argument("--plot-limit", type=int, default=12)
    parser.add_argument("--skip-frames", type=int, default=0, help="Drop this many initial rows from every log.")
    parser.add_argument(
        "--steering-delay-frames",
        type=int,
        default=0,
        help="Lag steering by this many frames before fitting/evaluation; useful for actuator or logging delay.",
    )
    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=None,
        help="If set, split logs at position jumps larger than this and keep the longest continuous segment.",
    )
    parser.add_argument(
        "--region-json",
        type=Path,
        default=DEFAULT_REGION_JSON_PATH,
        help="Track region JSON used to detect predicted off-track crashes.",
    )
    args = parser.parse_args()

    logs = [
        read_log(path, args.control_mode, args.skip_frames, args.max_step_distance)
        for path in discover_logs(args.trials_dir)
    ]
    logs = [log for log in logs if log is not None]
    if not logs:
        raise SystemExit(f"No usable log_*.csv files found under {args.trials_dir}")
    if args.steering_delay_frames < 0:
        raise SystemExit("--steering-delay-frames must be non-negative")
    apply_steering_delay(logs, args.steering_delay_frames)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = default_bicycle_params()
    region = load_region_json(args.region_json)
    evaluated = []
    run_rows = []
    for log in logs:
        metrics = evaluate_log(log, params, region)
        evaluated.append((log, metrics))
        row = {
            "path": log["path"],
            "case": log["case"],
            "intensity": log["intensity"],
            **{k: v for k, v in metrics.items() if k != "rollout"},
        }
        run_rows.append(row)

    summary = group_summary(run_rows)
    with (output_dir / "fit_params.json").open("w") as f:
        json.dump({
            "control_mode": args.control_mode,
            "skip_frames": args.skip_frames,
            "steering_delay_frames": args.steering_delay_frames,
            "max_step_distance": args.max_step_distance,
            "region_json": str(args.region_json),
            "params": params,
        }, f, indent=2)
    write_csv(output_dir / "run_metrics.csv", run_rows, [
        "path",
        "case",
        "intensity",
        "frames",
        "original_frames",
        "crashed",
        "crash_step",
        "crash_margin",
        "one_step_pos_rmse",
        "one_step_pos_mae",
        "one_step_pos_p50",
        "one_step_pos_p95",
        "one_step_pos_max",
        "rollout_pos_rmse",
        "rollout_pos_mae",
        "rollout_pos_p50",
        "rollout_pos_p95",
        "rollout_final_pos_error",
    ])
    write_csv(output_dir / "case_summary.csv", summary, [
        "case",
        "intensity",
        "runs",
        "frames",
        "original_frames",
        "crashes",
        "one_step_pos_rmse",
        "one_step_pos_mae",
        "one_step_pos_p50",
        "one_step_pos_p95",
        "one_step_pos_max",
        "rollout_pos_rmse",
        "rollout_pos_mae",
        "rollout_pos_p50",
        "rollout_pos_p95",
        "rollout_final_pos_error",
    ])
    wrote_plots = plot_rollouts(output_dir / "plots", evaluated, args.plot_limit)

    print(f"Loaded {len(logs)} logs.")
    print(f"Wrote {output_dir / 'case_summary.csv'}")
    print(f"Wrote {output_dir / 'run_metrics.csv'}")
    print(f"Wrote {output_dir / 'fit_params.json'}")
    if not wrote_plots:
        print("Skipped plots because matplotlib is not installed for this Python.")
    for row in summary:
        print(
            f"{row['case']} {row['intensity']}: "
            f"one-step RMSE={row['one_step_pos_rmse']:.4f}, "
            f"rollout RMSE={row['rollout_pos_rmse']:.4f}, "
            f"final error={row['rollout_final_pos_error']:.4f}"
        )


if __name__ == "__main__":
    main()
