"""Frozen spatial-AE diffusion with matched OFF/ON guidance experiments.

Requires spatial_ae_diffusion.py, spatial_diffusion_data.py,
train_lander_spatial_ae.py and constraint_checker.py from the existing project.
No training. No future RGB/grid is supplied to the sampler or shape target.
With --initialization source, iteration 0 is source appearance + future pose,
before any denoising. With --initialization diffusion (default), iteration 0
is the completed initial DDIM sample. Both use ten subsequent refinement rounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import spatial_ae_diffusion as base
from spatial_diffusion_data import file_hash, load_episodes, physical_pose
from train_lander_spatial_ae import hard_mask, soft_mask_log

EXPECTED_X = np.array([0.01558235, -0.99001140])
EXPECTED_Y = np.array([-0.02103094, 1.40286245])


def coordinates(mask):
    h, w = mask.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=mask.device, dtype=mask.dtype),
                          torch.arange(w, device=mask.device, dtype=mask.dtype), indexing='ij')
    return x, y


def mask_stats(mask):
    x, y = coordinates(mask)
    mass = mask.sum((-2, -1)).clamp_min(1e-8)
    cx = (mask*x).sum((-2, -1))/mass
    cy = (mask*y).sum((-2, -1))/mass
    dx, dy = x-cx[:, None, None], y-cy[:, None, None]
    xx = (mask*dx.square()).sum((-2, -1))/mass
    yy = (mask*dy.square()).sum((-2, -1))/mass
    xy = (mask*dx*dy).sum((-2, -1))/mass
    return mass, cx, cy, xx, yy, xy


def shape_terms(mask, target):
    dims = (-2, -1)
    area = target.sum(dims).clamp_min(1.)
    dice = 1-(2*(mask*target).sum(dims)+1)/(mask.sum(dims)+area+1)
    missing = (target*(1-mask)).sum(dims)/area
    excess = ((1-target)*mask).sum(dims)/area
    edge = mask.new_zeros(len(mask))
    for dim in (-2, -1):
        pd, td = torch.diff(mask, dim=dim), torch.diff(target, dim=dim)
        edge = edge + (pd-td).abs().sum(dims)/(td.abs().sum(dims)+1)
    return dice + missing + .35*excess + .25*edge, dice


def guidance_terms(image, pose, target, coef_x, coef_y, a):
    mask = soft_mask_log(image, a.temperature).exp()
    mass, cx, cy, xx, yy, xy = mask_stats(mask)
    x = coef_x[0]*cx+coef_x[1]
    y = coef_y[0]*cy+coef_y[1]
    xy_loss = ((x-pose[:, 0])/.06).square()+((y-pose[:, 1])/.10).square()
    # Narrow axis, represented using doubled angle. No head-sign heuristic.
    # Screen y points down; desired covariance vector is (cos(2theta),-sin(2theta)).
    q = torch.stack((xx-yy, 2*xy), -1)
    q = q / (q.square().sum(-1, keepdim=True)+1e-6).sqrt()
    desired = torch.stack((torch.cos(2*pose[:, 2]), -torch.sin(2*pose[:, 2])), -1)
    axis = 1-(q*desired).sum(-1)
    shape, dice = (mask.new_zeros(len(mask)), mask.new_zeros(len(mask)))
    if target is not None:
        shape, dice = shape_terms(mask, target)
    total = xy_loss + a.theta_weight*axis + a.shape_weight*shape
    return total.mean(), dict(xy=xy_loss.mean(), axis=axis.mean(), shape=shape.mean(),
                               dice=dice.mean(), mass=mass.mean())


@torch.no_grad()
def make_shape_target(source_rgb, source_theta, requested_pose, coef_x, coef_y, dev):
    """Rigid image-plane silhouette target from observed source only.

    Translation aligns the source mask centroid with requested calibrated x/y.
    Rotation uses the observed source state theta, never a generated head guess.
    The state-to-image rotation is an approximation, especially for moving legs.
    """
    mask = hard_mask(source_rgb)
    ys, xs = np.where(mask)
    meta = dict(enabled=False, source_pixels=int(len(xs)))
    if len(xs) < 20:
        return None, dict(meta, reason='source_not_detectable')
    h, w = mask.shape
    if xs.min() == 0 or xs.max() == w-1 or ys.min() == 0 or ys.max() == h-1:
        return None, dict(meta, reason='source_clipped')
    if not np.isfinite(source_theta) or not torch.isfinite(requested_pose).all():
        return None, dict(meta, reason='nonfinite_pose')
    tensor = torch.as_tensor(mask, device=dev, dtype=torch.float32)[None]
    target_cx = (requested_pose[0, 0]-coef_x[1])/coef_x[0]
    target_cy = (requested_pose[0, 1]-coef_y[1])/coef_y[0]
    delta = requested_pose[0, 2]-float(source_theta)
    c, s = torch.cos(delta), torch.sin(delta)
    x, y = coordinates(tensor)
    dx, dy = x-target_cx, y-target_cy
    # Inverse of the forward screen rotation [[cos,sin],[-sin,cos]].
    sx = c*dx-s*dy+float(xs.mean())
    sy = s*dx+c*dy+float(ys.mean())
    grid = torch.stack((2*(sx+.5)/w-1, 2*(sy+.5)/h-1), -1)[None]
    warped = F.grid_sample(tensor[:, None], grid, align_corners=False,
                           mode='bilinear', padding_mode='zeros')[:, 0]
    retained = float(warped.sum())/len(xs)
    meta.update(target_mass=float(warped.sum()), retained_fraction=retained)
    if warped.sum() < 20 or retained < .8:
        return None, dict(meta, reason='target_mostly_outside_frame')
    return warped.detach(), dict(meta, enabled=True, reason='valid')


def improve_clean(clean, ae, pose, norm, target, coef_x, coef_y, a):
    """Gradient through frozen decoder to clean normalized spatial grid only.

    Backtracking accepts only decreases in the soft combined objective. This
    does NOT guarantee improvements to hard-checker metrics or visual quality.
    """
    z = clean.detach().requires_grad_(True)
    image = base.decode(ae, pose, z*norm['std']+norm['mean'])
    loss, _ = guidance_terms(image, pose, target, coef_x, coef_y, a)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite guidance loss')
    gradient = torch.autograd.grad(loss, z)[0]
    if not torch.isfinite(gradient).all():
        raise RuntimeError('Nonfinite image-to-latent gradient')
    update = a.guidance_step*gradient
    rms = update.square().mean((1, 2, 3), keepdim=True).sqrt()
    update = update*(a.max_update_rms/rms.clamp_min(1e-12)).clamp(max=1)
    with torch.no_grad():
        for trial in range(a.backtracks+1):
            candidate = z-update*(.5**trial)
            if a.clip_x0 > 0:
                candidate = candidate.clamp(-a.clip_x0, a.clip_x0)
            value, _ = guidance_terms(base.decode(ae, pose, candidate*norm['std']+norm['mean']),
                                       pose, target, coef_x, coef_y, a)
            if torch.isfinite(value) and value < loss.detach():
                return candidate.detach(), True
    return clean.detach(), False


def refinement(model, ae, schedule, current, source, cond, pose, norm, target,
               coef_x, coef_y, noise, guided, a):
    """Correct skipped DDIM transitions; guides x0 before advancing to next t."""
    times = torch.linspace(a.start_t, 0, a.refine_steps).round().long().tolist()
    assert len(times) == len(set(times)), 'Repeated diffusion timestep'
    with torch.no_grad():
        z = schedule.add_noise(current, torch.tensor([a.start_t], device=current.device), noise)
    accepted = 0
    for index, time in enumerate(times):
        with torch.no_grad():
            tt = torch.full((len(z),), time, device=z.device, dtype=torch.long)
            alpha = schedule.alpha[time]
            previous = schedule.alpha[times[index+1]] if index+1 < len(times) else z.new_tensor(1.)
            epsilon = model(z, tt, source, cond)
            clean = (z-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
            if a.clip_x0 > 0:
                clean = clean.clamp(-a.clip_x0, a.clip_x0)
                epsilon = (z-alpha.sqrt()*clean)/(1-alpha).sqrt()
        if guided:
            clean, success = improve_clean(clean, ae, pose, norm, target, coef_x, coef_y, a)
            accepted += int(success)
        with torch.no_grad():
            # Retain the model's epsilon direction after editing the clean image.
            z = previous.sqrt()*clean+(1-previous).sqrt()*epsilon
        if not torch.isfinite(z).all():
            raise RuntimeError('Nonfinite refined latent')
    return z.detach(), accepted


def selection_key(metrics):
    """Uses requested-pose checker ONLY, never true future state or image."""
    errors = [metrics[k] for k in ('x_err', 'y_err', 'theta_err')]
    if any(v is None for v in errors):
        return (1, float('inf'))
    x, y, theta = errors
    return (int(not metrics['all_ok']), x/.06+y/.1+theta/.3)


def initialize_grid(model, schedule, batch, pose, norm, noise, a):
    """Source mode performs no denoising before iteration 0 is evaluated."""
    if a.initialization == 'source':
        return batch['source'].detach().clone()
    return base.sample_grid(model, schedule, batch['source'], batch['source_pose'], pose,
                            batch['context_raw'], batch['actions'], norm,
                            a.sample_steps, noise, a.clip_x0)[0]


def summarize(records, iterations):
    output = {}
    for group in ('all_pairs', 'future_detectable'):
        subset = records if group == 'all_pairs' else [r for r in records if r['future_detectable']]
        output[group] = {}
        for reference in ('true_future', 'requested'):
            result = {}
            variants = {f'{arm}_iter_{i}': [(r[arm][i], r) for r in subset]
                        for arm in ('off', 'on') for i in range(iterations+1)}
            for arm in ('off', 'on'):
                variants[arm+'_selected'] = [(r[arm][r[arm+'_selected']], r) for r in subset]
            variants['real_future'] = [(r['real_future'], r) for r in subset]
            variants['future_ae'] = [(r['future_ae'], r) for r in subset]
            variants['pose_swap'] = [(r['pose_swap'], r) for r in subset]
            for name, entries in variants.items():
                metrics = [h[reference] for h, _ in entries]
                n = len(metrics)
                v = dict(n=n)
                for key in ('xy_ok', 'theta_ok', 'all_ok'):
                    v[key] = sum(m[key] for m in metrics)
                    v[key+'_rate'] = v[key]/n if n else None
                for key in ('x_err', 'y_err', 'theta_err', 'future_mask_iou', 'components', 'largest_fraction'):
                    values = [m[key] for m in metrics if m[key] is not None]
                    v[key+'_finite'] = len(values)
                    v[key+'_mean'] = float(np.mean(values)) if values else None
                result[name] = v
            output[group][reference] = result
    return output


def run(a):
    import constraint_checker as checker_module
    checker = checker_module.constraint_checker
    coef_x = np.asarray(checker_module.COEF_X, dtype=float)
    coef_y = np.asarray(checker_module.COEF_Y, dtype=float)
    if not np.allclose(coef_x, EXPECTED_X, atol=1e-7, rtol=0) or not np.allclose(coef_y, EXPECTED_Y, atol=1e-7, rtol=0):
        raise ValueError('Expected your calibrated constraint_checker coefficients. '
                         f'Found X={coef_x}, Y={coef_y}; expected X={EXPECTED_X}, Y={EXPECTED_Y}.')
    base.seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    dev, out = base.device(), Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a fresh --out directory: {out}')
    (out/'images').mkdir(parents=True, exist_ok=True)
    saved = base.load_pt(a.diffusion_checkpoint)
    manifest = base.cache_manifest(a.cache)
    if saved.get('format') != base.FORMAT or saved['cache_sha256'] != file_hash(Path(a.cache)/'manifest.json'):
        raise ValueError('Denoiser/cache mismatch')
    base.verify_cache_files(a.cache, manifest, (a.split,))
    ae, ae_info = base.load_ae(a.ae_checkpoint, dev, saved['ae_sha256'])
    data = base.CachedPairs(Path(a.cache)/f'{a.split}.pt', ae_info['sha256']).data
    model = base.SpatialDenoiser(**saved['model_config']).to(dev)
    model.load_state_dict(saved['ema'], strict=True)
    model.eval().requires_grad_(False)
    norm = {k: v.to(dev) for k, v in saved['normalization'].items()}
    schedule = base.Schedule(saved['settings']['diffusion_steps'], dev)
    if not 0 <= a.start_t < schedule.steps or not 2 <= a.sample_steps <= schedule.steps:
        raise ValueError('Sampling timestep settings do not match trained schedule')
    dynamics = None
    if a.pose_mode == 'dynamics':
        if data['horizon'] != 2 or data['pose_columns'] != [0, 1, 4]:
            raise ValueError('Two-state dynamics requires horizon=2 and pose columns 0,1,4')
        dynamics = base.load_dynamics(a.dynamics_checkpoint, a.project, dev)
    ids = np.arange(len(data['source']))
    if a.max_pairs and len(ids) > a.max_pairs:
        ids = np.sort(np.random.default_rng(a.seed).choice(len(ids), a.max_pairs, replace=False))
    config = dict(settings=vars(a), ae=ae_info, denoiser_epoch=saved['epoch'], pairs=len(ids),
                  coef_x=coef_x.tolist(), coef_y=coef_y.tolist(),
                  script_sha256=file_hash(__file__), base_script_sha256=file_hash(base.__file__),
                  denoiser_sha256=file_hash(a.diffusion_checkpoint),
                  checker_sha256=file_hash(checker_module.__file__),
                  iteration_zero=('Source appearance + requested pose; no denoising yet'
                                  if a.initialization == 'source'
                                  else 'Initial full DDIM output, not source reconstruction'),
                  off='Same initial grid and re-noising, with no image guidance',
                  on='Calibrated xy + axis-only theta + transformed SOURCE mask',
                  selection='Requested-pose metrics only; no true future information',
                  shape='Source hard purple mask, bilinear pose transform; clipped/absent source disables shape only',
                  evaluation='All pairs AND real-future-detectable subset; missing outputs fail')
    if dynamics is not None:
        config['dynamics_sha256'] = file_hash(a.dynamics_checkpoint)
    base.write_json(out/'config.json', config)
    print(json.dumps(config, indent=2), flush=True)
    records, episodes, cached_path = [], None, None
    for number, i in enumerate(ids, 1):
        b = {k: data[k][i:i+1].to(dev) for k in ('source', 'source_pose', 'future_pose', 'context_raw', 'actions')}
        archive, episode, frame = map(int, data['identity'][i].tolist())
        path = data['files'][archive]['path']
        if path != cached_path:
            if file_hash(path) != data['files'][archive]['sha256']:
                raise ValueError(f'Trajectory changed: {path}')
            episodes, cached_path = load_episodes(path), path
        source_rgb = episodes[episode][0][frame].astype(np.float32)/255.
        source_true_pose = physical_pose(episodes[episode][1][frame:frame+1], data['pose_columns'],
                                         data['xy_scales'], data['xy_offsets'])[0]
        with torch.no_grad():
            if dynamics is None:
                pose = b['future_pose'].clone()
            else:
                prediction = dynamics(b['context_raw'][:, 0], b['context_raw'][:, 1], b['actions'][:, 1].long())
                pose = torch.from_numpy(physical_pose(prediction.cpu().numpy(), data['pose_columns'],
                                                       data['xy_scales'], data['xy_offsets'])).to(dev)
            if not torch.isfinite(pose).all():
                raise RuntimeError('Nonfinite requested pose')
        target, shape_meta = make_shape_target(source_rgb, source_true_pose[2], pose, coef_x, coef_y, dev)
        identity = f'{data["files"][archive]["sha256"]}:{episode}:{frame}'
        pair_seed = (a.seed+int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)) % (2**63-1)
        generator = torch.Generator(device=dev).manual_seed(pair_seed)
        noise = torch.randn(b['source'].shape, device=dev, generator=generator)
        # Reserve the same random draw in either mode, keeping refinement noise
        # matched if the two initialization modes are compared in separate runs.
        initial = initialize_grid(model, schedule, b, pose, norm, noise, a)
        current = {arm: (initial-norm['mean'])/norm['std'] for arm in ('off', 'on')}
        source = (b['source']-norm['mean'])/norm['std']
        cond = (base.condition(b['source_pose'], pose, b['context_raw'], b['actions'])-norm['cond_mean'])/norm['cond_std']
        # Evaluation references never enter initialization or refinement.
        future_rgb = data['future_rgb'][i].numpy().astype(np.float32)/255.
        true_pose, requested = data['future_pose'][i].numpy(), pose[0].cpu().numpy()
        def measurement(image, iteration=None, accepted=0):
            return dict(iteration=iteration, accepted_steps=accepted,
                        true_future=base.image_metrics(image, true_pose, future_rgb, checker),
                        requested=base.image_metrics(image, requested, future_rgb, checker))
        fm = hard_mask(future_rgb)
        record = dict(path=path, episode=episode, source_frame=frame, future_frame=frame+data['horizon'],
                      true_future_pose=true_pose.tolist(), requested_pose=requested.tolist(), seed=pair_seed,
                      future_detectable=bool(fm.sum() >= 20), initialization=a.initialization,
                      shape_target=shape_meta, off=[], on=[])
        keep = number <= a.save_n
        panels = []
        with torch.no_grad():
            source_ae = base.decode(ae, b['source_pose'], b['source'])[0]
            swap = base.decode(ae, pose, b['source'])[0]
            future_ae = base.decode(ae, data['future_encoded_pose'][i:i+1].to(dev), data['future'][i:i+1].to(dev))[0]
            initial_rgb = base.decode(ae, pose, initial)[0]
        record['real_future'], record['future_ae'], record['pose_swap'] = [measurement(im) for im in (future_rgb, future_ae, swap)]
        best_keys, best_images = {}, {}
        for arm in ('off', 'on'):
            entry = measurement(initial_rgb, 0)
            record[arm].append(entry)
            record[arm+'_selected'] = 0
            best_keys[arm] = selection_key(entry['requested'])
            best_images[arm] = initial_rgb.detach().cpu() if keep else None
        if keep:
            panels = [source_rgb, source_ae.cpu(), future_rgb, future_ae.cpu(), swap.cpu(), initial_rgb.cpu()]
            initial_title = ('Iteration 0: before denoising' if a.initialization == 'source'
                             else 'Iteration 0: initial diffusion')
            titles = ['Real source', 'Source spatial AE', 'Real future (evaluation only)',
                      'Future spatial AE (evaluation only)', 'Source grid + requested pose', initial_title]
        for iteration in range(1, a.iterations+1):
            # Identical random draw for OFF/ON, independent of guidance/selection.
            noise = torch.randn(b['source'].shape, device=dev, generator=generator)
            for arm in ('off', 'on'):
                current[arm], accepted = refinement(model, ae, schedule, current[arm], source, cond, pose,
                    norm, target, coef_x, coef_y, noise, arm == 'on', a)
                with torch.no_grad():
                    image = base.decode(ae, pose, current[arm]*norm['std']+norm['mean'])[0]
                entry = measurement(image, iteration, accepted)
                with torch.no_grad():
                    soft_total, soft_parts = guidance_terms(image[None], pose, target, coef_x, coef_y, a)
                entry['soft_guidance'] = dict(total=float(soft_total), **{k: float(v) for k, v in soft_parts.items()})
                entry['drift_rms_from_initial'] = float((current[arm]-(initial-norm['mean'])/norm['std']).square().mean().sqrt())
                record[arm].append(entry)
                key = selection_key(entry['requested'])
                if key < best_keys[arm]:
                    best_keys[arm], record[arm+'_selected'] = key, iteration
                    if keep:
                        best_images[arm] = image.cpu()
                if keep and (a.initialization == 'source' or iteration in {1, 5, a.iterations}):
                    panels.append(image.cpu())
                    m = entry['true_future']
                    titles.append(f'{arm.upper()} iteration {iteration}\nxy={m["xy_ok"]} theta={m["theta_ok"]}')
        if keep:
            for arm in ('off', 'on'):
                panels.append(best_images[arm])
                titles.append(f'{arm.upper()} selected iteration {record[arm+"_selected"]}')
            # Keep the matched OFF arm in measurements, but omit its images.
            visible = [(image, title) for image, title in zip(panels, titles)
                       if not title.startswith('OFF ')]
            base.render_panel(out/'images'/f'pair_{number:05d}.png',
                              [image for image, title in visible],
                              [title for image, title in visible])
            if target is not None:
                target_rgb = target[0, :, :, None].expand(-1, -1, 3).cpu().numpy()
                base.render_panel(out/'images'/f'target_{number:05d}.png', [source_rgb, target_rgb],
                                  ['Observed source', 'Transformed source mask (guidance target)'])
        records.append(record)
        if number % 10 == 0 or number == len(ids):
            base.write_json(out/'measurements.json', records)
            base.write_json(out/'summary.json', summarize(records, a.iterations))
            print(f'Evaluated {number}/{len(ids)}; source shape: {shape_meta["reason"]}', flush=True)
    assert all(p.grad is None for p in ae.parameters())
    assert all(p.grad is None for p in model.parameters())
    summary = summarize(records, a.iterations)
    for group, references in summary.items():
        for reference, variants in references.items():
            print(f'\n{group} | against {reference} | missing outputs fail', flush=True)
            for name, r in variants.items():
                if not r['n']:
                    print(f'{name}: n=0', flush=True)
                    continue
                print(f'{name}: x/y={r["xy_ok"]}/{r["n"]} ({100*r["xy_ok_rate"]:.1f}%) '
                      f'theta={r["theta_ok"]}/{r["n"]} ({100*r["theta_ok_rate"]:.1f}%) '
                      f'all={r["all_ok"]}/{r["n"]} ({100*r["all_ok_rate"]:.1f}%) '
                      f'future IoU={r["future_mask_iou_mean"]:.4f}', flush=True)
    with zipfile.ZipFile(out/'comparisons.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out/'images').glob('*.png')):
            z.write(p, p.relative_to(out))
    print(f'Finished: {out}\nImages: {out / "comparisons.zip"}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--initialization', choices=['source', 'diffusion'], default='diffusion',
                   help='source: measure iteration 0 before denoising; diffusion: preserve original full-sample start')
    p.add_argument('--ae-checkpoint', default=base.DEFAULT_AE)
    p.add_argument('--cache', default='./checkpoints/spatial_diffusion/data')
    p.add_argument('--diffusion-checkpoint', default='./checkpoints/spatial_diffusion/model/best.pt')
    p.add_argument('--dynamics-checkpoint', default='./dynamic/checkpoints/dynamics_discrete/best.pt')
    p.add_argument('--project', default='.')
    p.add_argument('--out', default='./checkpoints/spatial_guided_test')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--pose-mode', choices=['oracle', 'dynamics'], default='dynamics')
    p.add_argument('--max-pairs', type=int, default=400)
    p.add_argument('--save-n', type=int, default=40)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--start-t', type=int, default=20)
    p.add_argument('--refine-steps', type=int, default=21)
    p.add_argument('--guidance-step', type=float, default=1.)
    p.add_argument('--max-update-rms', type=float, default=.02)
    p.add_argument('--backtracks', type=int, default=4)
    p.add_argument('--theta-weight', type=float, default=.02)
    p.add_argument('--shape-weight', type=float, default=.10)
    p.add_argument('--temperature', type=float, default=.02)
    p.add_argument('--clip-x0', type=float, default=8.)
    p.add_argument('--seed', type=int, default=42)
    return p


def self_test():
    """CPU mathematical and frozen-model tests; no HPG data/checkpoints needed."""
    from types import SimpleNamespace
    from torch import nn
    from train_lander_spatial_ae import SpatialLanderAE
    base.seed_all(42)
    torch.set_num_threads(2)
    a = parser().parse_args([])
    dev = torch.device('cpu')
    rgb = np.zeros((96, 128, 3), np.float32)
    rgb[38:44, 50:64] = [120/255, 40/255, 200/255]
    rgb[44:48, 50:52] = [120/255, 40/255, 200/255]
    rgb[44:48, 62:64] = [120/255, 40/255, 200/255]
    mask = torch.as_tensor(hard_mask(rgb)).float()[None]
    _, cx, cy, *_ = mask_stats(mask)
    pose = torch.tensor([[float(EXPECTED_X[0]*cx+EXPECTED_X[1]),
                          float(EXPECTED_Y[0]*cy+EXPECTED_Y[1]), 0.]])
    target, meta = make_shape_target(rgb, 0., pose, EXPECTED_X, EXPECTED_Y, dev)
    assert meta['enabled'] and torch.allclose(target, mask, atol=2e-5)
    moved = pose.clone()
    moved[0, 0] += 8*EXPECTED_X[0]; moved[0, 1] -= 5*EXPECTED_Y[0]
    shifted, meta = make_shape_target(rgb, 0., moved, EXPECTED_X, EXPECTED_Y, dev)
    _, sx, sy, *_ = mask_stats(shifted)
    assert abs(float(sx-cx)-8) < .01 and abs(float(sy-cy)+5) < .01
    rotated_pose = pose.clone(); rotated_pose[0, 2] = np.pi/2
    rotated, _ = make_shape_target(rgb, 0., rotated_pose, EXPECTED_X, EXPECTED_Y, dev)
    # The top/body is above centroid before rotation and left afterward.
    x, y = coordinates(mask)
    old_skew = (mask*(y-cy[:, None, None]).pow(3)).sum()
    new_skew = (rotated*(x-cx[:, None, None]).pow(3)).sum()
    assert old_skew*new_skew > 0, 'Rotation sign inverted'
    clipped = rgb.copy(); clipped[0, 50:64] = [120/255, 40/255, 200/255]
    assert make_shape_target(clipped, 0., pose, EXPECTED_X, EXPECTED_Y, dev)[0] is None
    assert make_shape_target(np.zeros_like(rgb), 0., pose, EXPECTED_X, EXPECTED_Y, dev)[0] is None
    offscreen = pose.clone(); offscreen[0, 0] = 10.
    assert make_shape_target(rgb, 0., offscreen, EXPECTED_X, EXPECTED_Y, dev)[0] is None
    exact = shape_terms(mask, mask)[0]
    missing = mask.clone(); missing[:, 44:48] = 0
    satellite = mask.clone(); satellite[:, 70:75, 90:95] = 1
    assert shape_terms(missing, mask)[0] > exact
    assert shape_terms(satellite, mask)[0] > exact
    # Axis loss should agree at theta and theta+pi, reject the perpendicular.
    image = torch.as_tensor(rgb).permute(2, 0, 1)[None].requires_grad_(True)
    losses = []
    for angle in (0., np.pi, np.pi/2):
        p = pose.clone(); p[0, 2] = angle
        loss, terms = guidance_terms(image, p, target, EXPECTED_X, EXPECTED_Y, a)
        losses.append(float(terms['axis'].detach()))
    assert abs(losses[0]-losses[1]) < 1e-5 and losses[2] > losses[0]+.5
    gradient = torch.autograd.grad(loss, image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Verify skipped DDIM jumps including the final transition to clean t=-1.
    schedule = base.Schedule(100, dev)
    clean = torch.randn(1, 2, 6, 8)
    norm = dict(mean=torch.zeros(1, 2, 1, 1), std=torch.ones(1, 2, 1, 1))
    source, cond = torch.zeros_like(clean), torch.zeros(1, 32)
    # The source start must use an exact copy and never call the denoiser.
    class NeverDenoise(nn.Module):
        def forward(self, *args):
            raise AssertionError('Denoiser ran before source-start iteration 0')
    source_settings = SimpleNamespace(**vars(a)); source_settings.initialization = 'source'
    batch = dict(source=clean)
    initialized = initialize_grid(NeverDenoise(), schedule, batch, pose, norm,
                                  torch.randn_like(clean), source_settings)
    assert torch.equal(initialized, clean) and initialized.data_ptr() != clean.data_ptr()
    assert not initialized.requires_grad
    class PerfectNoise(nn.Module):
        def forward(self, noisy, time, src, condition):
            alpha = schedule.alpha[time].view(-1, 1, 1, 1)
            return (noisy-alpha.sqrt()*clean)/(1-alpha).sqrt()
    settings = SimpleNamespace(**vars(a)); settings.refine_steps = 6; settings.clip_x0 = 0
    recovered, accepted = refinement(PerfectNoise(), None, schedule, clean, source, cond, pose,
        norm, target, EXPECTED_X, EXPECTED_Y, torch.randn_like(clean), False, settings)
    assert torch.allclose(recovered, clean, atol=1e-5) and accepted == 0
    # Real spatial architecture: gradients reach appearance, frozen weights unchanged.
    ae = SpatialLanderAE(2).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in ae.state_dict().items()}
    z = clean.clone().requires_grad_(True)
    loss, _ = guidance_terms(base.decode(ae, pose, z), pose, target, EXPECTED_X, EXPECTED_Y, a)
    dz = torch.autograd.grad(loss, z)[0]
    assert dz.abs().sum() > 0 and torch.isfinite(dz).all()
    improved, accepted = improve_clean(clean, ae, pose, norm, target, EXPECTED_X, EXPECTED_Y, a)
    assert accepted, 'Synthetic gradient step failed to improve the soft objective'
    if accepted:
        end, _ = guidance_terms(base.decode(ae, pose, improved), pose, target, EXPECTED_X, EXPECTED_Y, a)
        assert end < loss and (improved-clean).square().mean().sqrt() <= a.max_update_rms+1e-6
    denoiser = base.SpatialDenoiser(2, 32, width=8, blocks=1).eval().requires_grad_(False)
    result, _ = refinement(denoiser, ae, schedule, clean, source, cond, pose, norm, target,
                            EXPECTED_X, EXPECTED_Y, torch.randn_like(clean), True, settings)
    assert result.shape == clean.shape and torch.isfinite(result).all()
    assert all(p.grad is None for p in ae.parameters()) and all(p.grad is None for p in denoiser.parameters())
    assert all(torch.equal(before[k], v) for k, v in ae.state_dict().items())
    assert selection_key(dict(x_err=None, y_err=None, theta_err=None, all_ok=False))[0] == 1
    print('GUIDED SELF-TEST PASSED: source iteration 0 has no denoising, calibration, translation/rotation, clipped-source handling, '
          'missing legs/satellites, axis-only loss, latent gradients, frozen weights, '
          'DDIM jumps, both refinement branches and accepted-update bound.', flush=True)


def main():
    p = parser(); a = p.parse_args()
    if min(a.max_pairs, a.save_n, a.backtracks, a.theta_weight, a.shape_weight, a.clip_x0) < 0:
        p.error('Counts and weights must be nonnegative')
    if a.iterations < 1 or not 2 <= a.refine_steps <= a.start_t+1 or min(a.guidance_step, a.max_update_rms, a.temperature) <= 0:
        p.error('Invalid refinement/guidance settings')
    if a.self_test:
        self_test()
    else:
        run(a)


if __name__ == '__main__':
    main()
