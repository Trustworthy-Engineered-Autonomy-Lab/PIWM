"""Decoder-only fine-tuning to suppress distant purple pixels.

Starts from an existing SpatialLanderAE checkpoint. Keeps the encoder, pose
head, and appearance head frozen. Uses the original crop-shape and image losses
plus a differentiable penalty for predicted purple outside a 6-pixel dilation
of the real lander mask. The hard checker is unchanged and used only to select
among validation checkpoints. Run the held-out evaluation separately.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader, Subset

from constraint_checker import extract_theta
from train_lander_spatial_ae import (
    StateFrames,
    hard_mask,
    load_spatial,
    loss_terms,
    seed_worker,
    soft_mask_log,
    to_device,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def leakage_loss(pred, target_mask, temperature=0.02, radius=6, min_pixels=20):
    """Per-image extraneous purple mass / real foreground pixels.

    Target is an *unmodified* hard mask from the real image. Only target-mask
    dilation defines where a predicted lander is allowed. The prediction uses
    the smooth purple test already present in train_lander_spatial_ae.py.
    """
    if target_mask.ndim != 3 or pred.shape[-2:] != target_mask.shape[-2:]:
        raise ValueError("Expected predicted BCHW and target BHW at matching resolution")
    width = 2 * radius + 1
    allowed = F.max_pool2d(target_mask[:, None], kernel_size=width,
                           stride=1, padding=radius)[:, 0].detach()
    purple = soft_mask_log(pred, temperature).exp()
    outside = purple * (1.0 - allowed)
    return outside.sum((-2, -1)) / target_mask.sum((-2, -1)).clamp_min(min_pixels)


def loss_args_from_checkpoint(checkpoint):
    """Preserve the selected model's original reconstruction objective."""
    settings = checkpoint.get("settings")
    required = (
        "crop_size", "near_threshold", "min_pixels", "state_cols",
        "temperature", "global_weight", "rgb_weight", "mask_bce_weight",
        "dice_weight", "boundary_weight", "moment_weight", "state_weight",
        "near_weight",
    )
    if not isinstance(settings, dict) or any(key not in settings for key in required):
        raise ValueError("Checkpoint lacks required original training settings")
    if list(settings["state_cols"]) != [0, 1, 2]:
        raise ValueError("Expected intrinsic archive labels in columns x=0,y=1,theta=2")
    return SimpleNamespace(**settings)


def theta_error(image, truth):
    estimate = extract_theta(image)
    if estimate is None or not np.isfinite(estimate) or not np.isfinite(truth):
        return float("nan")
    return float(abs(np.arctan2(np.sin(estimate - truth),
                                np.cos(estimate - truth))))


def summary(rows):
    groups = {
        "all": rows,
        "eligible": [r for r in rows if r["eligible"]],
        "eligible_near": [r for r in rows if r["eligible"] and r["near"]],
        "eligible_far": [r for r in rows if r["eligible"] and not r["near"]],
    }
    result = {}
    for name, group in groups.items():
        finite = [r["theta_error"] for r in group if np.isfinite(r["theta_error"])]
        passes = int(sum(bool(np.isfinite(r["theta_error"]) and
                              r["theta_error"] <= 0.30) for r in group))
        source_passes = int(sum(bool(np.isfinite(r["source_theta_error"]) and
                                     r["source_theta_error"] <= 0.30) for r in group))
        result[name] = {
            "count": len(group),
            "theta_finite": len(finite),
            # Missing theta remains in the denominator.
            "theta_pass": passes,
            "theta_rate": passes / len(group) if group else None,
            "mean_theta_error_finite": float(np.mean(finite)) if finite else None,
            "mean_iou": float(np.mean([r["iou"] for r in group])) if group else None,
            "mean_leakage": float(np.mean([r["leakage"] for r in group])) if group else None,
            "mean_extra_purple_pixels": (
                float(np.mean([r["extra_purple_pixels"] for r in group])) if group else None
            ),
            "source_theta_rate": source_passes / len(group) if group else None,
        }
    return result


@torch.no_grad()
def evaluate(model, dataset, loader, device, args, name, path):
    model.eval()
    rows = []
    for cpu in loader:
        batch = to_device(cpu, device)
        pred, _ = model(batch["image"])
        leak = leakage_loss(pred, batch["mask"], args.temperature,
                            args.radius, args.min_pixels).cpu().numpy()
        predicted = pred.permute(0, 2, 3, 1).cpu().numpy()
        for j, index in enumerate(cpu["index"].tolist()):
            target = dataset.rgb(index)
            pmask = hard_mask(predicted[j])
            tmask = hard_mask(target)
            allowed = np.zeros_like(tmask)
            # Diagnostic is only for counting hard pixels. The training loss
            # itself uses the tensor max-pool dilation above.
            if tmask.any():
                allowed = binary_dilation(
                    tmask, structure=np.ones((2 * args.radius + 1,) * 2),
                )
            union = (pmask | tmask).sum()
            rows.append({
                "index": index,
                "near": bool(dataset.meta[index]["near"]),
                "eligible": bool(dataset.meta[index]["valid"]),
                "theta_target": float(dataset.theta[index]),
                "theta_error": theta_error(predicted[j], dataset.theta[index]),
                "source_theta_error": theta_error(target, dataset.theta[index]),
                "iou": float((pmask & tmask).sum() / union) if union else 1.0,
                "leakage": float(leak[j]),
                "extra_purple_pixels": int((pmask & ~allowed).sum()),
            })
    if not rows:
        raise ValueError("Validation loader returned no frames")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = summary(rows)
    print(f"{name}: ", end="", flush=True)
    for group_name in ("eligible", "eligible_near", "eligible_far"):
        group = metrics[group_name]
        print(f"{group_name} theta={group['theta_pass']}/{group['count']} "
              f"({100 * group['theta_rate']:.1f}%) "
              f"IoU={group['mean_iou']:.4f} "
              f"extra-purple={group['mean_extra_purple_pixels']:.2f}; ",
              end="", flush=True)
    print(flush=True)
    return metrics


def valid_candidate(current, baseline, max_iou_drop, max_near_theta_drop):
    reasons = []
    for group in ("eligible", "eligible_near", "eligible_far"):
        if current[group]["mean_iou"] < baseline[group]["mean_iou"] - max_iou_drop:
            reasons.append(f"{group} silhouette IoU dropped")
    if (current["eligible_near"]["theta_rate"] <
            baseline["eligible_near"]["theta_rate"] - max_near_theta_drop):
        reasons.append("near-screen theta dropped")
    return reasons


def checkpoint(path, model, optimizer, epoch, args, metrics, source):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": "SpatialLanderAE",
        "model_type": "deterministic_spatial_ae",
        "appearance_channels": model.appearance_channels,
        "spatial_shape": [6, 8],
        "latent_dim": 3 + model.appearance_channels * 6 * 8,
        "state_dim": 3,
        "epoch": epoch,
        "source_epoch": source["epoch"],
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        # Keep original objective metadata for later fine-tuning/checking.
        "settings": dict(source["settings"], **vars(args)),
        "validation": metrics,
        "encoder_unchanged": True,
        "diffusion_compatible": False,
        "pose_disentanglement_guaranteed": False,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def self_test():
    from train_lander_spatial_ae import SpatialLanderAE
    torch.manual_seed(2)
    target = torch.zeros(1, 96, 128)
    target[:, 30:38, 61:67] = 1
    target[:, 38:43, 61:63] = 1  # a thin leg inside the allowed region
    real = torch.zeros(1, 3, 96, 128)
    real[:, 0, 30:43, 61:67] = .5
    real[:, 1, 30:43, 61:67] = .25
    real[:, 2, 30:43, 61:67] = .85
    far = real.clone().requires_grad_(True)
    with torch.no_grad():
        far[:, 0, 85, 100] = .5
        far[:, 1, 85, 100] = .25
        far[:, 2, 85, 100] = .85
    clean = leakage_loss(real, target)[0]
    polluted = leakage_loss(far, target)[0]
    assert polluted > clean, (clean, polluted)
    grad, = torch.autograd.grad(polluted, far)
    assert torch.isfinite(grad).all() and grad[:, :, 85, 100].abs().sum() > 0
    assert grad[:, :, 39, 62].abs().sum() == 0, "Leg within allowed region was penalized"
    model = SpatialLanderAE()
    for module in (model.encoder, model.state_head, model.appearance_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    output, _ = model(real)
    leakage_loss(output, target).mean().backward()
    assert model.decoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad is None
    print("SELF-TEST PASSED: distant-purple penalty, protected leg, gradients, frozen encoder.",
          flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz")
    p.add_argument("--split", default="./checkpoints/vae_shape_first_42508780/split_indices.npz")
    p.add_argument("--checkpoint", default="./checkpoints/spatial_ae_43750874/best_shape.pt")
    p.add_argument("--out")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-leak", type=float, default=.1)
    p.add_argument("--far-leak-boost", type=float, default=5.)
    p.add_argument("--radius", type=int, default=6)
    p.add_argument("--max-iou-drop", type=float, default=.01)
    p.add_argument("--max-near-theta-drop", type=float, default=.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.out:
        p.error("--out is required")
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0
            or not np.isfinite(args.lambda_leak) or args.lambda_leak < 0
            or args.radius < 0 or args.max_iou_drop < 0
            or not np.isfinite(args.far_leak_boost) or args.far_leak_boost < 1
            or not 0 <= args.max_near_theta_drop <= 1):
        p.error("Invalid hyperparameters")
    for required in (args.data, args.split, args.checkpoint):
        if not Path(required).is_file():
            p.error(f"Missing required file: {required}")
    out = Path(args.out).resolve()
    if out.exists():
        p.error("Output directory already exists; use a new run name")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if source.get("architecture") != "SpatialLanderAE":
        raise ValueError("Not a SpatialLanderAE checkpoint")
    old_args = loss_args_from_checkpoint(source)
    args.temperature = old_args.temperature
    args.min_pixels = old_args.min_pixels
    dataset_args = SimpleNamespace(data=args.data, crop_size=old_args.crop_size,
                                   near_threshold=old_args.near_threshold,
                                   min_pixels=old_args.min_pixels,
                                   state_cols=old_args.state_cols)
    dataset = StateFrames(dataset_args)
    with np.load(args.split, allow_pickle=False) as split:
        train, val = split["train"], split["validation"]
    joined = np.concatenate((train, val))
    if (train.ndim != 1 or val.ndim != 1 or not len(train) or not len(val)
            or not np.issubdtype(joined.dtype, np.integer)
            or not np.array_equal(np.sort(joined), np.arange(len(dataset)))):
        raise ValueError("Split does not partition the archive")
    if not dataset.near[val].any() or dataset.near[val].all():
        raise ValueError("Validation needs near and far examples")
    if not (dataset.valid[val] & dataset.near[val]).any() or not (
            dataset.valid[val] & ~dataset.near[val]).any():
        raise ValueError("Validation needs eligible near and far examples")

    model, source_epoch = load_spatial(args.checkpoint, device)
    for module in (model.encoder, model.state_head, model.appearance_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / "training_script.py")
    np.savez(out / "split_indices.npz", train=train, validation=val)
    settings = dict(vars(args), source_epoch=source_epoch, device=str(device),
                    original_objective={key: vars(old_args)[key] for key in (
                        "temperature", "global_weight", "rgb_weight", "mask_bce_weight",
                        "dice_weight", "boundary_weight", "moment_weight", "state_weight",
                        "near_weight")},
                    train_frames=len(train), validation_frames=len(val),
                    split_sha256=hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note="Validation is a frame split of training archive; held-out trajectory "
                         "evaluation runs afterward in SLURM. Checker unchanged.")
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)
    train_loader = DataLoader(
        Subset(dataset, train), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(Subset(dataset, val), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    normalization = float(np.where(dataset.near[train], old_args.near_weight, 1.).mean())
    baseline = evaluate(model, dataset, val_loader, device, args, "Epoch 0 baseline",
                        out / "metrics_epoch_00.csv")
    checkpoint(out / "best.pt", model, optimizer, 0, args, baseline, source)
    score = (baseline["eligible"]["theta_rate"],
             baseline["eligible_far"]["theta_rate"],
             -baseline["eligible"]["mean_leakage"])
    best_epoch = 0
    history = [{"epoch": 0, "validation": baseline, "selected": True}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, baseline_loss, leak_loss, sample_count = 0., 0., 0., 0
        for step, cpu in enumerate(train_loader, 1):
            batch = to_device(cpu, device)
            prediction, _ = model(batch["image"])
            original = loss_terms(prediction, batch, old_args)["total"]
            leak = leakage_loss(prediction, batch["mask"], old_args.temperature,
                                args.radius, old_args.min_pixels)
            weights = torch.where(batch["near"], old_args.near_weight, 1.0) / normalization
            # The original shape objective keeps its near-ground weighting.
            # Boost the *leak* term for far frames, where false pixels were worst.
            leak_boost = torch.where(batch["near"], 1.0, args.far_leak_boost)
            combined = (original + args.lambda_leak * leak_boost * leak) * weights
            if not torch.isfinite(combined).all():
                raise RuntimeError(f"Nonfinite loss epoch={epoch} batch={step}")
            optimizer.zero_grad(set_to_none=True)
            combined.mean().backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.,
                error_if_nonfinite=True,
            )
            optimizer.step()
            n = len(weights)
            total += float(combined.detach().mean()) * n
            baseline_loss += float((original.detach() * weights).mean()) * n
            leak_loss += float((leak.detach() * leak_boost * weights).mean()) * n
            sample_count += n
            if step % 200 == 0:
                print(f"Epoch {epoch}/{args.epochs} batch {step}/{len(train_loader)} "
                      f"loss={combined.mean().item():.5f}", flush=True)
        metrics = evaluate(model, dataset, val_loader, device, args,
                           f"Epoch {epoch}", out / f"metrics_epoch_{epoch:02d}.csv")
        reasons = valid_candidate(metrics, baseline, args.max_iou_drop,
                                  args.max_near_theta_drop)
        current_score = (metrics["eligible"]["theta_rate"],
                         metrics["eligible_far"]["theta_rate"],
                         -metrics["eligible"]["mean_leakage"])
        selected = not reasons and current_score > score
        if selected:
            score, best_epoch = current_score, epoch
            checkpoint(out / "best.pt", model, optimizer, epoch, args, metrics, source)
        entry = {"epoch": epoch,
                 "train": {"total": total / sample_count,
                           "original": baseline_loss / sample_count,
                           "leakage_weighted_before_lambda": leak_loss / sample_count},
                 "validation": metrics, "guard_reasons": reasons,
                 "selected": selected, "best_epoch": best_epoch}
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint(out / "last.pt", model, optimizer, epoch, args, metrics, source)
        print(f"  best_epoch={best_epoch}, selected={selected}, "
              f"guards={reasons or 'passed'}", flush=True)

    summary_file = out / "summary.json"
    summary_file.write_text(json.dumps({"baseline": baseline,
                                        "selected_epoch": best_epoch,
                                        "selected": (baseline if best_epoch == 0
                                                     else history[best_epoch]["validation"]),
                                        "last": history[-1]["validation"]}, indent=2) + "\n")
    print(f"Finished. Best epoch {best_epoch}; checkpoint: {out / 'best.pt'}", flush=True)
    if best_epoch == 0:
        print("No fine-tuned epoch beat initialization while passing the shape and near-theta guards.",
              flush=True)


if __name__ == "__main__":
    main()
