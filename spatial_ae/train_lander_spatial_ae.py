"""Spatial deterministic autoencoder: reconstruction-only LunarLander experiment.

Self-contained training/helpers; requires existing train_lunar.py (reference VAE)
and constraint_checker.py (unchanged evaluation). Not diffusion-compatible.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from torch.utils.data import Dataset, DataLoader, Subset

def hard_mask(image):
    """HWC [0,1]; same quantization and rules as the supplied hard checker."""
    rgb = (np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.int16)
    r, g, b = rgb.transpose(2, 0, 1)
    return (b > 100) & (b > g + 20) & (r > 50) & (r < 220)


def soft_mask_log(image, temperature=0.02):
    r, g, b = image.unbind(1)
    # Work in log space to keep useful gradients when purple is initially faint.
    return (F.logsigmoid((b - 100 / 255) / temperature)
            + F.logsigmoid((b - g - 20 / 255) / temperature)
            + F.logsigmoid((r - 50 / 255) / temperature)
            + F.logsigmoid((220 / 255 - r) / temperature))


def mask_metadata(mask, crop_size=32, threshold=0.65, min_pixels=20):
    h, w = mask.shape
    ys, xs = np.where(mask)
    n = len(xs)
    cy = float(ys.mean() / h) if n else float('nan')
    # This is a screen-height proxy, NOT a measurement of terrain clearance.
    near = n >= min_pixels and cy >= threshold
    clipped = bool(n and (xs.min() == 0 or ys.min() == 0
                         or xs.max() == w - 1 or ys.max() == h - 1))
    x0 = int(np.clip(round(float(xs.mean()) - crop_size / 2), 0, w - crop_size)) if n else 0
    y0 = int(np.clip(round(float(ys.mean()) - crop_size / 2), 0, h - crop_size)) if n else 0
    reason = 'valid'
    if n < min_pixels:
        reason = 'too_few_purple_pixels'
    elif clipped:
        reason = 'source_clipped'
    elif not (xs.min() >= x0 + 2 and xs.max() < x0 + crop_size - 2
              and ys.min() >= y0 + 2 and ys.max() < y0 + crop_size - 2):
        reason = 'crop_does_not_cover_mask_with_margin'
    return dict(near=near, valid=reason == 'valid', clipped=clipped,
                cy=cy, count=n, x0=x0, y0=y0, reason=reason)


class Frames(Dataset):
    def __init__(self, path, crop_size=32, threshold=0.65, min_pixels=20, theta_col=2):
        with np.load(path, allow_pickle=False) as z:
            self.images = z['frame']
            if 'state' in z and theta_col >= 0:
                states = z['state']
                if states.ndim != 2 or states.shape[0] != len(self.images) or theta_col >= states.shape[1]:
                    raise ValueError('Invalid state shape or --theta-col.')
                self.theta = states[:, theta_col].astype(np.float64)
            else:
                self.theta = np.full(len(self.images), np.nan)
        if self.images.ndim != 4 or self.images.shape[1:] != (96, 128, 3):
            raise ValueError(f'Expected (N,96,128,3), got {self.images.shape}')
        self.byte = self.images.dtype == np.uint8
        if not self.byte and not np.issubdtype(self.images.dtype, np.floating):
            raise ValueError('Images must be uint8 [0,255] or float [0,1].')
        if len(self.images) < 20:
            raise ValueError('Need at least 20 frames.')
        self.crop_size = crop_size
        self.meta = []
        # Inspect individually: avoid allocating another full float RGB archive.
        for i in range(len(self.images)):
            im = self.rgb(i)
            if not np.isfinite(im).all() or im.min() < 0 or im.max() > 1:
                raise ValueError(f'Invalid image range at frame {i}.')
            self.meta.append(mask_metadata(hard_mask(im), crop_size, threshold, min_pixels))
        self.near = np.array([m['near'] for m in self.meta])
        self.valid = np.array([m['valid'] for m in self.meta])

    def rgb(self, i):
        im = self.images[i].astype(np.float32)
        return im / 255 if self.byte else im

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.rgb(i)
        meta = self.meta[i]
        x0, y0, s = meta['x0'], meta['y0'], self.crop_size
        mask = hard_mask(im)
        crop = mask[y0:y0+s, x0:x0+s]
        boundary = crop & ~binary_erosion(crop)
        dist = distance_transform_edt(~boundary).astype(np.float32) if boundary.any() else np.zeros_like(crop, dtype=np.float32)
        return dict(image=torch.from_numpy(im).permute(2, 0, 1),
                    mask=torch.from_numpy(mask.astype(np.float32)),
                    distance=torch.from_numpy(dist), x0=x0, y0=y0,
                    near=meta['near'], valid=meta['valid'], index=i)


def crops(x, batch, size):
    """Integer target-anchored crops; identical coordinates, no resize or recenter."""
    yy = batch['y0'][:, None, None] + torch.arange(size, device=x.device)[None, :, None]
    xx = batch['x0'][:, None, None] + torch.arange(size, device=x.device)[None, None, :]
    ii = torch.arange(x.shape[0], device=x.device)[:, None, None]
    if x.ndim == 3:
        return x[ii, yy, xx]
    return x.permute(0, 2, 3, 1)[ii, yy, xx].permute(0, 3, 1, 2)


def balanced(error, mask, weight=None):
    """Foreground/background balanced mean; tiny legs aren't diluted by sky."""
    if weight is None:
        weight = torch.ones_like(mask)
    fg, bg = mask * weight, (1 - mask) * weight
    dims = (-2, -1)
    return 0.5 * ((error * fg).sum(dims) / fg.sum(dims).clamp_min(1)
                  + (error * bg).sum(dims) / bg.sum(dims).clamp_min(1))


def moments(p):
    """Centroid and normalized covariance; no eigenvector sign ambiguity."""
    h, w = p.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=p.device, dtype=p.dtype),
                            torch.arange(w, device=p.device, dtype=p.dtype), indexing='ij')
    mass = p.sum((-2, -1)).clamp_min(1)
    cx = (p * xx).sum((-2, -1)) / mass
    cy = (p * yy).sum((-2, -1)) / mass
    dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
    cxx = (p * dx.square()).sum((-2, -1)) / mass
    cyy = (p * dy.square()).sum((-2, -1)) / mass
    cxy = (p * dx * dy).sum((-2, -1)) / mass
    trace = (cxx + cyy).clamp_min(1)
    cov = torch.stack((cxx / trace, cyy / trace, cxy / trace), 1)
    anisotropy = torch.sqrt((cxx - cyy).square() + 4 * cxy.square() + 1e-8) / trace
    return torch.stack((cx / w, cy / h), 1), cov, anisotropy


def loss_terms(pred, batch, args):
    target = batch['image']
    valid = batch['valid'].float()
    log_full = soft_mask_log(pred, args.temperature)
    logp = crops(log_full, batch, args.crop_size).clamp(max=-1e-7)
    p = logp.exp()
    t = crops(batch['mask'], batch, args.crop_size)
    dims = (-2, -1)
    # Balanced BCE provides gradients to missing foreground even for tiny p.
    log_not_p = torch.log(-torch.expm1(logp))
    pixel_bce = -t * logp - (1 - t) * log_not_p
    silhouette_bce = balanced(pixel_bce, t)
    dice = 1 - (2 * (p * t).sum(dims) + 1e-6) / (p.square().sum(dims) + t.sum(dims) + 1e-6)
    # Penalize misplaced occupancy using distance from the REAL boundary.
    # This is target-distance-weighted mask error, not a differentiable EDT.
    distance = batch['distance'].clamp(max=8)
    boundary = balanced((p - t).abs(), t, 1 + distance / 4)
    pc, pv, _ = moments(p)
    tc, tv, ta = moments(t)
    moment = (pc - tc).square().mean(1) + (pv - tv).square().mean(1) * (ta > 0.15)
    # Fixed binary target requires correct mask membership. Color still affects
    # mask membership; no RGB-based silhouette is completely color-independent.
    shape = (args.mask_bce_weight * silhouette_bce + args.dice_weight * dice
             + args.boundary_weight * boundary + args.moment_weight * moment) * valid
    rgb = (crops(pred, batch, args.crop_size) - crops(target, batch, args.crop_size)).abs().mean((1, 2, 3)) * valid
    global_bce = F.binary_cross_entropy(pred, target, reduction='none').mean((1, 2, 3))
    # Full-image BCE penalizes fragments outside the crop as well.
    total = args.global_weight * global_bce + shape + args.rgb_weight * rgb
    return dict(total=total, shape=shape, global_bce=global_bce, crop_rgb=rgb,
                mask_bce=silhouette_bce * valid, dice=dice * valid,
                boundary=boundary * valid, moment=moment * valid)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def angle_error(estimate, target):
    if estimate is None or not np.isfinite(estimate) or not np.isfinite(target):
        return float('nan')
    return float(abs(np.arctan2(np.sin(estimate - target), np.cos(estimate - target))))


def image_metrics(pred, target, theta, extract_theta):
    pm, tm = hard_mask(pred), hard_mask(target)
    union = (pm | tm).sum()
    _, n = label(pm, structure=np.ones((3, 3)))
    estimated = extract_theta(pred)
    error = angle_error(estimated, theta)
    return dict(hard_iou=float((pm & tm).sum() / union) if union else float('nan'),
                components=int(n), theta_error=error,
                axis_error=min(error, np.pi - error) if np.isfinite(error) else float('nan'),
                theta_available=bool(estimated is not None and np.isfinite(estimated)))


def summarize(rows):
    summary = {}
    for name, select in [('all', lambda r: True), ('near_ground', lambda r: r['near']),
                         ('not_near_ground', lambda r: not r['near']),
                         ('invalid_shape', lambda r: not r['valid'])]:
        group = [r for r in rows if select(r)]
        valid = [r for r in group if r['valid']]
        def mean(key, subset):
            vals = [r[key] for r in subset if np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else None
        labeled = [r for r in group if r['theta_labeled']]
        finite = [r for r in labeled if np.isfinite(r['theta_error'])]
        summary[name] = dict(count=len(group), valid_shape_count=len(valid),
                             total=mean('total', group), global_bce=mean('global_bce', group),
                             shape=mean('shape', valid), hard_iou=mean('hard_iou', valid),
                             components=mean('components', valid),
                             theta_labeled=len(labeled), theta_finite=len(finite),
                             theta_pass=sum(r['theta_error'] <= 0.30 for r in finite),
                             theta_pass_rate=sum(r['theta_error'] <= 0.30 for r in finite) / len(labeled) if labeled else None,
                             theta_mean=mean('theta_error', finite),
                             axis_mean=mean('axis_error', finite),
                             theta_median=float(np.median([r['theta_error'] for r in finite])) if finite else None)
    return summary


class StateFrames(Frames):
    def __init__(self, args):
        super().__init__(args.data, args.crop_size, args.near_threshold,
                         args.min_pixels, args.state_cols[2])
        with np.load(args.data, allow_pickle=False) as data:
            raw = data['state']
            if raw.ndim != 2 or max(args.state_cols) >= raw.shape[1]:
                raise ValueError('state columns must explicitly identify x,y,theta.')
            self.states = raw[:, args.state_cols].astype(np.float32)
        if not np.isfinite(self.states).all():
            raise ValueError('Non-finite physical state labels.')

    def __getitem__(self, i):
        batch = super().__getitem__(i)
        batch['state'] = torch.from_numpy(self.states[i])
        return batch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def latent_mean(model, image):
    # AE never calls reparameterize or fc_logvar, including during training.
    if isinstance(model, SpatialLanderAE):
        return model.encode_latent(image)
    return model.fc_mu(model.encoder(image).flatten(1))


def state_loss(z, state):
    # Same unwrapped coordinate MSE as original intrinsic supervision.
    # Evaluation also reports circular theta error for physical interpretation.
    return (z[:, :3] - state).square().sum(1)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, dataset, loader, device, args, checker, out_csv, real=False):
    model.eval()
    rows = []
    for cpu in loader:
        batch = to_device(cpu, device)
        mu = latent_mean(model, batch['image'])
        pred = batch['image'] if real else model.decode(mu)
        values = {k: v.cpu().numpy() for k, v in loss_terms(pred, batch, args).items()}
        physical = (mu[:, :3] - batch['state']).cpu().numpy()
        images = pred.permute(0, 2, 3, 1).cpu().numpy()
        for j, idx in enumerate(cpu['index'].tolist()):
            row = dict(index=idx, near=dataset.meta[idx]['near'], valid=dataset.meta[idx]['valid'],
                       reason=dataset.meta[idx]['reason'], theta_labeled=True)
            row.update({k: float(v[j]) for k, v in values.items()})
            row.update(image_metrics(images[j], dataset.rgb(idx), dataset.theta[idx], checker))
            row.update(latent_x_mae=float(abs(physical[j, 0])),
                       latent_y_mae=float(abs(physical[j, 1])),
                       latent_theta_mae=float(abs(np.arctan2(np.sin(physical[j, 2]), np.cos(physical[j, 2])))),
                       latent_state_mse=float(np.mean(physical[j] ** 2)))
            rows.append(row)
    if not all(np.isfinite(r['total']) and np.isfinite(r['latent_state_mse']) for r in rows):
        raise RuntimeError('Non-finite validation outputs.')
    write_rows(out_csv, rows)
    result = summarize(rows)
    for group, select in [('all', lambda r: True), ('near_ground', lambda r: r['near']),
                          ('not_near_ground', lambda r: not r['near']),
                          ('invalid_shape', lambda r: not r['valid'])]:
        selected = [r for r in rows if select(r)]
        for key in ('latent_x_mae', 'latent_y_mae', 'latent_theta_mae', 'latent_state_mse'):
            result[group][key] = float(np.mean([r[key] for r in selected])) if selected and not real else None
    return result


def balanced_iou(metrics):
    vals = [metrics[g]['hard_iou'] for g in ('near_ground', 'not_near_ground')
            if metrics[g]['hard_iou'] is not None]
    if not vals:
        raise ValueError('Validation contains no usable silhouette targets.')
    return float(np.mean(vals))


def rejection_reasons(metrics, baseline, args):
    reasons = []
    for group in ('all', 'near_ground', 'not_near_ground'):
        current, old = metrics[group], baseline[group]
        if not old['count']:
            continue
        if current['theta_pass_rate'] < old['theta_pass_rate'] - args.max_theta_drop:
            reasons.append(group + ': theta success regressed')
        if current['global_bce'] > old['global_bce'] * (1 + args.max_bce_regression):
            reasons.append(group + ': full-image BCE regressed')
        for key, tol in [('latent_x_mae', args.max_xy_regression),
                         ('latent_y_mae', args.max_xy_regression),
                         ('latent_theta_mae', args.max_latent_theta_regression)]:
            if current[key] > old[key] + tol:
                reasons.append(group + ': ' + key + ' regressed')
    return reasons


@torch.no_grad()
def preview_arrays(model, dataset, indices, device):
    model.eval()
    return [model.decode(latent_mean(model, dataset[i]['image'][None].to(device)))[0]
            .permute(1, 2, 0).cpu().numpy() for i in indices]


def save_panels(dataset, indices, columns, folder, args, checker):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    for j, idx in enumerate(indices):
        x, y, s = dataset.meta[idx]['x0'], dataset.meta[idx]['y0'], args.crop_size
        entries = [('Real image', dataset.rgb(idx))] + [(name, frames[j]) for name, frames in columns]
        fig, axes = plt.subplots(3, len(entries), figsize=(3 * len(entries), 8), squeeze=False)
        for c, (name, rgb) in enumerate(entries):
            met = image_metrics(rgb, dataset.rgb(idx), dataset.theta[idx], checker)
            theta = f"{met['theta_error']:.3f}" if np.isfinite(met['theta_error']) else 'missing'
            axes[0, c].imshow(rgb, interpolation='nearest')
            axes[0, c].set_title(f'{name}\nIoU={met["hard_iou"]:.3f}; theta err={theta} rad', fontsize=8)
            axes[1, c].imshow(rgb[y:y+s, x:x+s], interpolation='nearest')
            axes[2, c].imshow(hard_mask(rgb)[y:y+s, x:x+s], cmap='gray', vmin=0, vmax=1,
                              interpolation='nearest')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle(f'Frame {idx}; near-ground proxy={dataset.near[idx]}; {dataset.meta[idx]["reason"]}\n'
                     'Full image / fixed native-resolution crop / hard silhouette')
        fig.tight_layout()
        fig.savefig(folder / f'frame_{idx:07d}.png', dpi=120)
        plt.close(fig)


def print_metrics(label_text, metrics):
    parts = []
    for group in ('all', 'near_ground', 'not_near_ground'):
        m = metrics[group]
        if m['count']:
            iou = 'N/A' if m['hard_iou'] is None else f'{m["hard_iou"]:.4f}'
            parts.append(f'{group}: theta={m["theta_pass"]}/{m["theta_labeled"]} '
                         f'({100*m["theta_pass_rate"]:.1f}%), IoU={iou}')
    print(label_text + ' | ' + ' | '.join(parts), flush=True)


class SpatialLanderAE(nn.Module):
    """Spatial appearance plus supervised pose; NOT guaranteed disentangled.

    No image skip connections: decoder sees only the stored latent.
    The packed vector format is a serialization convenience, not an FC bottleneck.
    """
    def __init__(self, appearance_channels=16):
        super().__init__()
        self.appearance_channels = appearance_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        self.appearance_head = nn.Conv2d(256, appearance_channels, 1)
        self.state_head = nn.Linear(256 * 6 * 8, 3)
        self.decode_input = nn.Sequential(
            nn.Conv2d(appearance_channels + 3, 256, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, image):
        features = self.encoder(image)
        return self.state_head(features.flatten(1)), self.appearance_head(features)

    def encode_latent(self, image):
        state, appearance = self.encode(image)
        return torch.cat((state, appearance.flatten(1)), dim=1)

    def decode(self, latent):
        expected = 3 + self.appearance_channels * 6 * 8
        if latent.ndim != 2 or latent.shape[1] != expected:
            raise ValueError(f'Expected packed spatial latent (B,{expected}), got {latent.shape}')
        state = latent[:, :3, None, None].expand(-1, -1, 6, 8)
        appearance = latent[:, 3:].reshape(-1, self.appearance_channels, 6, 8)
        return self.decoder(self.decode_input(torch.cat((appearance, state), dim=1)))

    def forward(self, image):
        latent = self.encode_latent(image)
        return self.decode(latent), latent


def load_reference(path, device):
    # Only load checkpoints you trust: these local checkpoints contain Python metadata.
    from train_lunar import IntrinsicVAE
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('latent_dim', 128) != 128 or payload.get('state_dim', 3) != 3:
        raise ValueError('Reference must be the existing 128D intrinsic model.')
    model = IntrinsicVAE(latent_dim=128, state_dim=3)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.to(device)


def warm_start(model, reference):
    """Reuse only shape-compatible convolutional blocks, never old FC layers."""
    # Sequential slices retain the original numeric names. Copy individual
    # modules so old decoder layers 2/4/6/8 map to new layers 0/2/4/6.
    for index in (0, 2, 4, 6):
        model.encoder[index].load_state_dict(reference.encoder[index].state_dict(), strict=True)
        model.decoder[index].load_state_dict(reference.decoder[index + 2].state_dict(), strict=True)


def objective_spatial(model, batch, args):
    prediction, latent = model(batch['image'])
    reconstruction = loss_terms(prediction, batch, args)['total']
    physical = state_loss(latent, batch['state'])
    return dict(total=reconstruction + args.state_weight * physical,
                reconstruction=reconstruction, physical=physical)


def save_spatial(path, model, optimizer, epoch, metrics, args):
    payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                   architecture='SpatialLanderAE', model_type='deterministic_spatial_ae',
                   appearance_channels=args.appearance_channels, spatial_shape=[6, 8],
                   latent_dim=3 + args.appearance_channels * 6 * 8, state_dim=3,
                   epoch=epoch, validation=metrics, settings=vars(args),
                   encoder_unchanged=False, diffusion_compatible=False,
                   pose_disentanglement_guaranteed=False)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_spatial(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('architecture') != 'SpatialLanderAE':
        raise ValueError('Not a spatial autoencoder checkpoint.')
    model = SpatialLanderAE(payload['appearance_channels']).to(device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model, payload['epoch']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz')
    p.add_argument('--checkpoint', default='./checkpoints/vae_shape_first_42508780/last.pt')
    p.add_argument('--original-checkpoint', default='./checkpoints/intrinsic_vae_200ep/best.pt')
    p.add_argument('--flat-checkpoint', help='Optional selected flat AE from the prior experiment.')
    p.add_argument('--split', default='./checkpoints/vae_shape_first_42508780/split_indices.npz')
    p.add_argument('--out')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--appearance-channels', type=int, default=16)
    p.add_argument('--state-cols', type=int, nargs=3, default=[0, 1, 2])
    p.add_argument('--state-weight', type=float, default=1.)
    p.add_argument('--crop-size', type=int, default=32)
    p.add_argument('--min-pixels', type=int, default=20)
    p.add_argument('--near-threshold', type=float, default=.65)
    p.add_argument('--near-weight', type=float, default=5.)
    p.add_argument('--temperature', type=float, default=.02)
    p.add_argument('--global-weight', type=float, default=1.)
    p.add_argument('--rgb-weight', type=float, default=.1)
    p.add_argument('--mask-bce-weight', type=float, default=1.)
    p.add_argument('--dice-weight', type=float, default=1.)
    p.add_argument('--boundary-weight', type=float, default=1.)
    p.add_argument('--moment-weight', type=float, default=.2)
    p.add_argument('--max-theta-drop', type=float, default=.02)
    p.add_argument('--max-bce-regression', type=float, default=.10)
    p.add_argument('--max-xy-regression', type=float, default=.02)
    p.add_argument('--max-latent-theta-regression', type=float, default=.05)
    p.add_argument('--min-improvement', type=float, default=.001)
    p.add_argument('--preview-per-group', type=int, default=6)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--self-test', action='store_true')
    return p


def self_test(args):
    from train_lunar import IntrinsicVAE
    from constraint_checker import extract_theta
    seed_all(42)
    torch.set_num_threads(2)
    image = np.zeros((96, 128, 3), dtype=np.float32)
    image[62:69, 59:68] = [.5, .25, .85]
    image[69:74, 58:60] = [.5, .25, .85]
    image[69:74, 67:69] = [.5, .25, .85]
    mask = hard_mask(image)
    meta = mask_metadata(mask, args.crop_size)
    assert meta['valid'] and meta['near']
    x, y, s = meta['x0'], meta['y0'], args.crop_size
    crop = mask[y:y+s, x:x+s]
    boundary = crop & ~binary_erosion(crop)
    batch = dict(image=torch.tensor(image.transpose(2, 0, 1))[None],
                 mask=torch.tensor(mask, dtype=torch.float32)[None],
                 distance=torch.tensor(distance_transform_edt(~boundary), dtype=torch.float32)[None],
                 x0=torch.tensor([x]), y0=torch.tensor([y]), valid=torch.tensor([True]),
                 near=torch.tensor([True]), state=torch.tensor([[.1, .2, -.3]]))
    model = SpatialLanderAE(args.appearance_channels)
    reference = IntrinsicVAE(latent_dim=128, state_dim=3)
    warm_start(model, reference)
    for index in (0, 2, 4, 6):
        for name in ('weight', 'bias'):
            assert torch.equal(getattr(model.encoder[index], name), getattr(reference.encoder[index], name))
            assert torch.equal(getattr(model.decoder[index], name), getattr(reference.decoder[index + 2], name))
    state, appearance = model.encode(batch['image'])
    assert appearance.shape == (1, args.appearance_channels, 6, 8)
    assert state.shape == (1, 3)
    a, latent = model(batch['image'])
    assert a.shape == batch['image'].shape
    assert torch.equal(a, model(batch['image'])[0]), 'Inference is not deterministic.'
    assert latent.shape == (1, 3 + args.appearance_channels * 48)
    params = dict(model.named_parameters())
    names = ['encoder.0.weight', 'appearance_head.weight', 'state_head.weight',
             'decode_input.0.weight', 'decoder.0.weight']
    before = {k: params[k].detach().clone() for k in names}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    objective_spatial(model, batch, args)['total'].mean().backward()
    for name in names:
        grad = params[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    optimizer.step()
    assert all(not torch.equal(before[k], params[k]) for k in names)
    # Check gradient paths from image reconstruction to BOTH latent branches.
    latent = model.encode_latent(batch['image']).detach().requires_grad_(True)
    reconstruction = loss_terms(model.decode(latent), batch, args)['total'].mean()
    gradient, = torch.autograd.grad(reconstruction, latent)
    assert torch.isfinite(gradient).all()
    assert gradient[:, :3].abs().sum() > 0 and gradient[:, 3:].abs().sum() > 0
    exact = loss_terms(batch['image'], batch, args)['shape'].item()
    missing = batch['image'].clone()
    missing[:, :, 69:74] = 0
    filled = batch['image'].clone()
    filled[:, :, 69:74, 60:67] = torch.tensor([.5, .25, .85])[None, :, None, None]
    assert loss_terms(missing, batch, args)['shape'].item() > exact
    assert loss_terms(filled, batch, args)['shape'].item() > exact
    assert extract_theta(image) is not None
    row = dict(valid=True, near=True, theta_labeled=True, total=0., global_bce=0.,
               shape=0., hard_iou=0., components=0, theta_error=float('nan'), axis_error=float('nan'))
    assert summarize([row])['all']['theta_pass_rate'] == 0.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'test.pt'
        save_spatial(path, model, optimizer, 1, {}, args)
        restored, epoch = load_spatial(path, torch.device('cpu'))
        assert epoch == 1 and torch.equal(model(batch['image'])[0], restored(batch['image'])[0])
    print('SELF-TEST PASSED: latent/output shapes; warm start; deterministic inference; '
          'encoder/heads/decoder updates; latent gradients; missing legs and filled gap; '
          'missing theta counts as failure; checkpoint round trip.', flush=True)


def main():
    p = parser()
    args = p.parse_args()
    positive = ('lr', 'state_weight', 'temperature', 'near_weight')
    nonnegative = ('global_weight', 'rgb_weight', 'mask_bce_weight', 'dice_weight',
                   'boundary_weight', 'moment_weight', 'max_theta_drop', 'max_bce_regression',
                   'max_xy_regression', 'max_latent_theta_regression', 'min_improvement')
    if (any(not np.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive)
        or any(not np.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in nonnegative)
        or args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.save_every < 1
        or args.appearance_channels < 1 or not 8 <= args.crop_size <= 96
        or args.min_pixels < 1 or args.preview_per_group < 1 or not 0 < args.near_threshold < 1
        or args.near_weight < 1 or min(args.state_cols) < 0 or len(set(args.state_cols)) != 3):
        p.error('Invalid training argument.')
    if args.self_test:
        self_test(args)
        return
    if not args.out:
        p.error('--out is required.')
    required = [args.data, args.checkpoint, args.original_checkpoint, args.split]
    if args.flat_checkpoint:
        required.append(args.flat_checkpoint)
    for path in required:
        if not Path(path).is_file():
            p.error(f'Missing required file: {path}. No silent checkpoint/split substitutions.')
    root = Path(args.out).resolve()
    if root.exists():
        p.error('Output exists; use a new directory to preserve earlier results.')
    from constraint_checker import extract_theta
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StateFrames(args)
    with np.load(args.split, allow_pickle=False) as split:
        train, val = split['train'], split['validation']
    joined = np.concatenate((train, val))
    if (train.ndim != 1 or val.ndim != 1 or not len(train) or not len(val)
        or not np.issubdtype(joined.dtype, np.integer)
        or not np.array_equal(np.sort(joined), np.arange(len(dataset)))):
        raise ValueError('Split must partition the archive exactly without duplicates.')
    if not dataset.valid[train].any() or not dataset.valid[val].any():
        raise ValueError('Train and validation both need usable silhouette targets.')
    root.mkdir(parents=True)
    shutil.copy2(__file__, root / 'training_script.py')
    np.savez(root / 'split_indices.npz', train=train, validation=val)
    settings = vars(args) | dict(device=str(device), train_frames=len(train), validation_frames=len(val),
        split_sha256=hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=str(torch.__version__), latent_shape=[args.appearance_channels, 6, 8],
        note='Partial convolutional warm start; new heads; full-model training; no KL or sampling. '
             'Same frame split of original training archive, NOT an independent test.')
    write_json(root / 'settings.json', settings)
    write_rows(root / 'mask_audit.csv', [dict(index=i, **m) for i, m in enumerate(dataset.meta)])
    print(json.dumps(settings, indent=2), flush=True)
    print('Physical labels must be x,y,theta in columns 0,1,2 (intrinsic archive, not raw Gym state).', flush=True)
    print('Near-ground means cy/H >= threshold; this is NOT terrain clearance.', flush=True)
    val_loader = DataLoader(Subset(dataset, val), batch_size=args.batch_size, num_workers=args.workers)
    previews = []
    for near in (True, False):
        previews.extend(val[dataset.valid[val] & (dataset.near[val] == near)][:args.preview_per_group].tolist())
    previews.extend(val[~dataset.valid[val]][:2].tolist())
    np.save(root / 'preview_indices.npy', np.array(previews))
    references = [('Original VAE', args.original_checkpoint), ('Shape-first VAE', args.checkpoint)]
    if args.flat_checkpoint:
        references.append(('Previous flat AE', args.flat_checkpoint))
    columns, baselines = [], {}
    for name, path in references:
        reference = load_reference(path, device)
        metrics = evaluate(reference, dataset, val_loader, device, args, extract_theta,
                           root / f'{name.replace(" ", "_")}_metrics.csv')
        baselines[name] = metrics
        columns.append((name, preview_arrays(reference, dataset, previews, device)))
        print_metrics(name, metrics)
        if name == 'Shape-first VAE':
            real = evaluate(reference, dataset, val_loader, device, args, extract_theta,
                            root / 'real_metrics.csv', real=True)
            baselines['Real image checker'] = real
            print_metrics('Real image checker', real)
        del reference
    write_json(root / 'baselines.json', baselines)
    baseline = baselines['Shape-first VAE']
    seed_all(args.seed)
    model = SpatialLanderAE(args.appearance_channels).to(device)
    reference = load_reference(args.checkpoint, device)
    warm_start(model, reference)
    del reference
    print(f'Training spatial AE: {sum(p.numel() for p in model.parameters()):,} parameters. '
          'No direct image skip connections; old score network is incompatible.', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(Subset(dataset, train), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    normalization = float(np.where(dataset.near[train], args.near_weight, 1.).mean())
    initial = evaluate(model, dataset, val_loader, device, args, extract_theta, root / 'metrics_epoch_00.csv')
    print_metrics('Spatial initialization (new heads, NOT trained yet)', initial)
    save_panels(dataset, previews, columns + [('Spatial initialization', preview_arrays(model, dataset, previews, device))],
                root / 'comparisons_epoch_00', args, extract_theta)
    save_spatial(root / 'initial.pt', model, optimizer, 0, initial, args)
    # Retain initialization as a fallback, but never claim it is qualified.
    save_spatial(root / 'best_shape.pt', model, optimizer, 0, initial, args)
    save_spatial(root / 'best_theta.pt', model, optimizer, 0, initial, args)
    best_shape = balanced_iou(initial)
    best_theta_key = (initial['all']['theta_pass_rate'], best_shape)
    qualified_score = balanced_iou(baseline)
    selected_epochs = dict(best_shape=0, best_theta=0, best_qualified=None)
    selected_metrics = dict(best_shape=initial, best_theta=initial, best_qualified=None)
    with (root / 'history.jsonl').open('w') as history:
        history.write(json.dumps(dict(epoch=0, validation=initial), allow_nan=False) + '\n')
        history.flush()
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = dict(total=0., reconstruction=0., physical=0.)
            count = 0
            for step, cpu in enumerate(train_loader, 1):
                batch = to_device(cpu, device)
                terms = objective_spatial(model, batch, args)
                weights = torch.where(batch['near'], args.near_weight, 1.) / normalization
                loss = (terms['total'] * weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {step}.')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                n = len(weights)
                for key, value in terms.items():
                    sums[key] += float((value.detach() * weights).mean()) * n
                count += n
                if step % 200 == 0:
                    print(f'Epoch {epoch}/{args.epochs}, batch {step}/{len(train_loader)}, loss={loss.item():.4f}', flush=True)
            metrics = evaluate(model, dataset, val_loader, device, args, extract_theta,
                               root / f'metrics_epoch_{epoch:02d}.csv')
            score = balanced_iou(metrics)
            reasons = rejection_reasons(metrics, baseline, args)
            updates = []
            if score > best_shape:
                best_shape = score
                updates.append('best_shape')
            theta_key = (metrics['all']['theta_pass_rate'], score)
            if theta_key > best_theta_key:
                best_theta_key = theta_key
                updates.append('best_theta')
            if not reasons and score > qualified_score + args.min_improvement:
                qualified_score = score
                updates.append('best_qualified')
            for key in updates:
                save_spatial(root / f'{key}.pt', model, optimizer, epoch, metrics, args)
                selected_epochs[key], selected_metrics[key] = epoch, metrics
            save_spatial(root / 'last.pt', model, optimizer, epoch, metrics, args)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_spatial(root / f'epoch_{epoch:02d}.pt', model, optimizer, epoch, metrics, args)
            save_panels(dataset, previews,
                        columns + [(f'Spatial AE epoch {epoch}', preview_arrays(model, dataset, previews, device))],
                        root / f'comparisons_epoch_{epoch:02d}', args, extract_theta)
            row = dict(epoch=epoch, train={k: v/count for k, v in sums.items()}, validation=metrics,
                       balanced_iou=score, selected_epochs=dict(selected_epochs), rejection_reasons=reasons)
            history.write(json.dumps(row, allow_nan=False) + '\n')
            history.flush()
            print_metrics(f'Spatial AE epoch {epoch}', metrics)
            print(f'  selected epochs={selected_epochs}; safety guards={reasons or "passed"}', flush=True)
    del model
    for key, epoch in selected_epochs.items():
        if epoch is None:
            continue
        selected, loaded_epoch = load_spatial(root / f'{key}.pt', device)
        save_panels(dataset, previews,
                    columns + [(f'Spatial {key}, epoch {loaded_epoch}', preview_arrays(selected, dataset, previews, device))],
                    root / f'comparisons_{key}', args, extract_theta)
        del selected
    summary = dict(baselines=baselines, selected_epochs=selected_epochs, selected_metrics=selected_metrics,
                   last_metrics=metrics, diffusion_compatible=False)
    write_json(root / 'summary.json', summary)
    # Small, directly downloadable bundle; checkpoints remain in the job directory.
    bundle = root / 'selected_comparisons'
    bundle.mkdir()
    for key, epoch in selected_epochs.items():
        if epoch is not None:
            shutil.copytree(root / f'comparisons_{key}', bundle / key)
    for name in ('summary.json', 'settings.json', 'history.jsonl'):
        shutil.copy2(root / name, bundle / name)
    shutil.make_archive(str(root / 'spatial_ae_comparisons'), 'zip', root, 'selected_comparisons')
    print(f'Finished: {root}', flush=True)
    print(f'Image bundle: {root}/spatial_ae_comparisons.zip', flush=True)
    if selected_epochs['best_qualified'] is None:
        print('No checkpoint both improved balanced IoU over the shape-first VAE and passed all guards. '
              'Inspect best_shape/best_theta; neither is automatically an approved replacement.', flush=True)
    print('Do NOT load these checkpoints into the old diffusion model. This is reconstruction-only.', flush=True)


if __name__ == '__main__':
    main()

