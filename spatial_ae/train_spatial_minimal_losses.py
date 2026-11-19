"""Minimal lander-specific image-loss comparison for SpatialLanderAE.

Imports the existing architecture/dataset/loss primitives, never pretrained weights.
Each seed uses identical random initial weights, data split, order and budget.
Selects checkpoints by validation eligible-frame theta, with IoU as a tie-break.
Test shards are deliberately not used for screening or checkpoint selection.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile
from types import SimpleNamespace

ARMS = ('control', 'generic', 'mask_only', 'boundary_only', 'crop_only', 'mask_boundary')
BASE = dict(global_weight=1., mask_bce_weight=0., dice_weight=0.,
            boundary_weight=0., moment_weight=0., rgb_weight=0.,
            near_weight=1., state_weight=1., lambda_chroma=4.)
SHAPE_TERMS = {
    'control': dict(mask_bce_weight=.25, boundary_weight=1., rgb_weight=.1),
    'generic': {},
    'mask_only': dict(mask_bce_weight=.25),
    'boundary_only': dict(boundary_weight=1.),
    'crop_only': dict(rgb_weight=.1),
    'mask_boundary': dict(mask_bce_weight=.25, boundary_weight=1.),
}


def coefficients(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(BASE, **SHAPE_TERMS[arm])


def init_runtime():
    global np, torch, base, DataLoader, Subset, checker
    import numpy as np
    import torch
    import train_lander_spatial_ae as base
    from torch.utils.data import DataLoader, Subset
    from constraint_checker import extract_theta as checker


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def chroma(pred, target):
    def contrasts(im):
        r,g,b = im.unbind(1)
        return torch.stack((r-g, r-b, g-b), 1)
    e = (contrasts(pred)-contrasts(target.detach())).abs().mean(1).flatten(1)
    k = max(1, math.ceil(e.shape[1]*.01))
    return e.mean(1)+e.topk(k, dim=1).values.mean(1)+10*(e-.02).clamp_min(0).mean(1)


def objective(pred, latent, batch, cfg, normalization):
    terms = base.loss_terms(pred, batch, cfg)
    physical = base.state_loss(latent, batch['state'])
    weights = torch.where(batch['near'], cfg.near_weight, 1.)/normalization
    # Equal frame weights for BOTH reconstruction and pose in every arm (norm=1).
    recon = weights*terms['total']
    pose = weights*cfg.state_weight*physical
    cleanup = cfg.lambda_chroma*chroma(pred, batch['image'])
    return dict(total=recon+pose+cleanup, reconstruction=recon, physical=pose, chroma=cleanup)


def summarize(rows):
    groups = {}
    for name, chosen in (
        ('all_labeled', rows), ('eligible', [r for r in rows if r['valid']]),
        ('eligible_near', [r for r in rows if r['valid'] and r['near']]),
        ('eligible_far', [r for r in rows if r['valid'] and not r['near']])):
        if not chosen:
            raise ValueError('Empty validation group: '+name)
        finite = [r for r in chosen if r['theta_error'] is not None]
        passed = sum(r['theta_error'] <= .30 for r in finite)
        groups[name] = dict(frames=len(chosen), theta_pass=passed,
            theta_rate=passed/len(chosen), missing_theta=len(chosen)-len(finite),
            mean_iou=float(np.mean([r['iou'] for r in chosen])),
            foreground_recall=float(np.mean([r['recall'] for r in chosen])),
            global_bce=float(np.mean([r['bce'] for r in chosen])),
            latent_state_mse=float(np.mean([r['latent_state_mse'] for r in chosen])))
    return groups


def evaluate(model, dataset, loader, device, path):
    model.eval()
    rows = []
    with torch.no_grad():
        for cpu in loader:
            b = base.to_device(cpu, device)
            pred,z = model(b['image'])
            if not torch.isfinite(pred).all() or not torch.isfinite(z).all():
                raise RuntimeError('Nonfinite validation output')
            bce = torch.nn.functional.binary_cross_entropy(pred,b['image'],reduction='none').mean((1,2,3)).cpu().tolist()
            state_mse = (z[:,:3]-b['state']).square().mean(1).cpu().tolist()
            for j, idx in enumerate(cpu['index'].tolist()):
                rgb = pred[j].permute(1,2,0).cpu().numpy()
                tm,pm = base.hard_mask(dataset.rgb(idx)),base.hard_mask(rgb)
                union = int((tm|pm).sum())
                error = base.angle_error(checker(rgb),float(dataset.theta[idx]))
                rows.append(dict(index=idx,near=bool(dataset.near[idx]),valid=bool(dataset.valid[idx]),
                    theta_error=float(error) if np.isfinite(error) else None,
                    iou=float((tm&pm).sum()/union) if union else 1.,
                    recall=float((tm&pm).sum()/tm.sum()) if tm.any() else 0.,
                    bce=bce[j],latent_state_mse=state_mse[j]))
    write_csv(path,rows)
    return summarize(rows)


def key(metrics):
    return (metrics['eligible']['theta_rate'],metrics['eligible']['mean_iou'])


def train(args):
    init_runtime()
    if args.epochs<1 or args.batch_size<1 or args.workers<0 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('Invalid training settings')
    cfg = SimpleNamespace(**vars(args), **coefficients(args.arm))
    if cfg.state_cols is None:
        raise ValueError('Explicit --state-cols required; intrinsic archive normally uses 0 1 2')
    root = Path(args.out)
    if root.exists():
        raise FileExistsError(root)
    for p in (args.data,args.split):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    dataset = base.StateFrames(cfg)
    with np.load(args.split,allow_pickle=False) as z:
        tr,va = z['train'],z['validation']
    joined = np.concatenate((tr,va))
    if (tr.ndim!=1 or va.ndim!=1 or not len(tr) or not len(va) or
        not np.issubdtype(joined.dtype,np.integer) or
        not np.array_equal(np.sort(joined),np.arange(len(dataset)))):
        raise ValueError('Split must partition the archive exactly')
    for ids in (tr,va):
        for near in (False,True):
            if not (dataset.valid[ids] & (dataset.near[ids]==near)).any():
                raise ValueError('Need eligible near/far frames in both subsets')
    # No load_reference, warm_start, load_state_dict, or pretrained checkpoint.
    base.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base.SpatialLanderAE(args.appearance_channels).to(device)
    initial_hash = model_hash(model)
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    train_loader = DataLoader(Subset(dataset,tr),batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,worker_init_fn=base.seed_worker,
        generator=torch.Generator().manual_seed(args.seed),pin_memory=device.type=='cuda')
    val_loader = DataLoader(Subset(dataset,va),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    norm = float(np.where(dataset.near[tr],cfg.near_weight,1.).mean())
    root.mkdir(parents=True)
    shutil.copy2(__file__,root/'training_script.py')
    np.savez(root/'split_indices.npz',train=tr,validation=va)
    settings = dict(vars(cfg), initialization='random',initial_model_sha256=initial_hash,
        split_sha256=hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        training_data_size=Path(args.data).stat().st_size,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_script_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        device=str(device),torch_version=str(torch.__version__),train_frames=len(tr),val_frames=len(va),
        reconstruction_normalization=norm,encoder_frozen=False,purple_leak_weight=0,
        chroma_formula='4*(mean + worst_1pct + 10*mean(relu(error-.02))) unless ablated',
        note='Frame-split validation screening; not an independent trajectory test. '
             'All arms keep latent state supervision and equal frame weights. Dice and moments are zero. Full-model training changes the latent space.')
    write_json(root/'settings.json',settings)
    print(json.dumps(settings,indent=2),flush=True)
    history = []
    best = None
    best_iou = -1.
    preview_ids = []
    for near in (False,True):
        preview_ids.extend(va[dataset.valid[va] & (dataset.near[va]==near)][:6].tolist())
    np.save(root/'preview_indices.npy',np.array(preview_ids))
    for epoch in range(args.epochs+1):
        totals = dict(total=0.,reconstruction=0.,physical=0.,chroma=0.)
        n = 0
        if epoch:
            model.train()
            for cpu in train_loader:
                batch = base.to_device(cpu,device)
                pred,latent = model(batch['image'])
                values = objective(pred,latent,batch,cfg,norm)
                loss = values['total'].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                count = len(pred)
                for term in totals:
                    totals[term] += float(values[term].detach().mean())*count
                n += count
        metrics = evaluate(model,dataset,val_loader,device,root/f'metrics_epoch_{epoch:03d}.csv')
        if epoch==0:
            base.save_spatial(root/'initial.pt',model,optimizer,epoch,metrics,cfg)
        # Only trained epochs compete for best.pt; initialization cannot masquerade as training.
        if epoch and (best is None or key(metrics)>key(best['metrics'])):
            best = dict(epoch=epoch,metrics=metrics)
            base.save_spatial(root/'best.pt',model,optimizer,epoch,metrics,cfg)
        if epoch and metrics['eligible']['mean_iou']>best_iou:
            best_iou = metrics['eligible']['mean_iou']
            base.save_spatial(root/'best_shape.pt',model,optimizer,epoch,metrics,cfg)
        if epoch:
            base.save_spatial(root/'last.pt',model,optimizer,epoch,metrics,cfg)
        row = dict(epoch=epoch,metrics=metrics,train={k:v/n for k,v in totals.items()} if n else None)
        history.append(row)
        write_json(root/'history.json',history)
        print(f"{args.arm} epoch={epoch}/{args.epochs}: eligible theta={100*metrics['eligible']['theta_rate']:.2f}% "
              f"IoU={metrics['eligible']['mean_iou']:.4f} best_epoch={best['epoch'] if best else None}",flush=True)
    base.save_panels(dataset,preview_ids,[('Last',base.preview_arrays(model,dataset,preview_ids,device))],
        root/'pictures_last',cfg,checker)
    selected,_ = base.load_spatial(root/'best.pt',device)
    base.save_panels(dataset,preview_ids,[('Validation best theta',base.preview_arrays(selected,dataset,preview_ids,device))],
        root/'pictures_best',cfg,checker)
    write_json(root/'summary.json',dict(arm=args.arm,seed=args.seed,best=best,last=history[-1],
        initialization=history[0],selection_uses_test=False,epochs=args.epochs))
    shutil.make_archive(str(root/'pictures'),'zip',root/'pictures_last')
    print('Finished '+str(root),flush=True)


def compare(args):
    root = Path(args.root)
    summaries = {}
    settings = {}
    rows = []
    for seed in args.seeds:
        seed_summaries = {}
        ref = None
        for arm in ARMS:
            folder = root/f'seed_{seed}'/arm
            cfg = json.loads((folder/'settings.json').read_text())
            summary = json.loads((folder/'summary.json').read_text())
            matched = {k:cfg[k] for k in ('initial_model_sha256','split_sha256','data',
                'training_data_size','script_sha256','base_script_sha256','epochs','seed','lr',
                'batch_size','workers','appearance_channels','state_cols','crop_size',
                'min_pixels','near_threshold','temperature')}
            if ref is None:
                ref = matched
            if matched!=ref or cfg['initialization']!='random' or cfg['encoder_frozen']:
                raise ValueError('Unmatched or non-random training: '+str(folder))
            for k,v in coefficients(arm).items():
                if cfg[k]!=v:
                    raise ValueError('Incorrect coefficient '+arm+' '+k)
            seed_summaries[arm] = summary
            for kind,m,epoch in (('best',summary['best']['metrics'],summary['best']['epoch']),
                                  ('last',summary['last']['metrics'],summary['last']['epoch'])):
                g=m['eligible']
                rows.append(dict(seed=seed,arm=arm,kind=kind,epoch=epoch,frames=g['frames'],
                    theta_pct=100*g['theta_rate'],iou=g['mean_iou'],recall=g['foreground_recall'],
                    near_theta_pct=100*m['eligible_near']['theta_rate'],
                    far_theta_pct=100*m['eligible_far']['theta_rate'],
                    latent_state_mse=g['latent_state_mse']))
        summaries[str(seed)]=seed_summaries
        settings[str(seed)]=ref
    suffix = '_seeds_'+'_'.join(map(str,args.seeds))
    write_json(root/('screening_summary'+suffix+'.json'),dict(seeds=summaries,matched_settings=settings,
        note='Validation-only, same frame split. Control excludes Dice, moments and near weighting. Read learning curves before inferring necessity. '
             'Loss interactions require joint removals; confirm selected candidates on unused trajectories.'))
    write_csv(root/('screening_table'+suffix+'.csv'),rows)
    print('VALIDATION ONLY — not comparable to the previous ~95% TEST result')
    for r in rows:
        print(f"seed={r['seed']} {r['arm']:28s} {r['kind']:4s} epoch={r['epoch']:2d} "
              f"theta={r['theta_pct']:6.2f}% IoU={r['iou']:.4f} latentMSE={r['latent_state_mse']:.5f}")
    print('Reports: '+str(root/('screening_table'+suffix+'.csv'))+' and screening_summary'+suffix+'.json')
    for seed in args.seeds:
        folder=root/f'seed_{seed}'
        with zipfile.ZipFile(folder/'comparisons.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for arm in ARMS:
                for label in ('pictures_best','pictures_last'):
                    panels=folder/arm/label
                    if not panels.is_dir() or not list(panels.glob('*.png')):
                        raise FileNotFoundError('Missing reconstruction panels: '+str(panels))
                    for picture in sorted(panels.glob('*.png')):
                        z.write(picture,Path(arm)/label/picture.name)
                for filename in ('settings.json','summary.json','history.json'):
                    z.write(folder/arm/filename,Path(arm)/filename)
            z.write(root/('screening_table'+suffix+'.csv'),'screening_table.csv')
            z.write(root/('screening_summary'+suffix+'.json'),'screening_summary.json')
        print('Pictures: '+str(folder/'comparisons.zip'))



def self_test():
    init_runtime()
    from scipy.ndimage import binary_erosion,distance_transform_edt
    torch.set_num_threads(2)
    target = torch.zeros(2,3,96,128)
    color = torch.tensor([.5,.25,.85])[None,:,None,None]
    target[:,:,62:69,59:68] = color
    target[:,:,69:74,58:60] = color
    target[:,:,69:74,67:69] = color
    masks = torch.stack([torch.tensor(base.hard_mask(im.permute(1,2,0).numpy()),dtype=torch.float32) for im in target])
    meta = base.mask_metadata(masks[0].numpy().astype(bool),32)
    x,y = meta['x0'],meta['y0']
    crop = masks[0,y:y+32,x:x+32].numpy().astype(bool)
    distance = torch.tensor(distance_transform_edt(~(crop & ~binary_erosion(crop))),dtype=torch.float32)
    batch = dict(image=target,mask=masks,distance=distance[None].repeat(2,1,1),
        x0=torch.tensor([x,x]),y0=torch.tensor([y,y]),near=torch.tensor([True,False]),
        valid=torch.tensor([True,True]),state=torch.tensor([[.1,.2,.3],[.3,.1,-.2]]))
    cfg_extra = dict(crop_size=32,temperature=.02)
    random_pred = (target*.6+.1).requires_grad_()
    latent = torch.ones(2,3+2*48,requires_grad=True)
    generic_cfg = SimpleNamespace(**coefficients('generic'),**cfg_extra)
    generic = objective(random_pred,latent,batch,generic_cfg,1.)['total']
    terms = base.loss_terms(random_pred,batch,generic_cfg)
    mapping = {'mask_bce_weight':'mask_bce','boundary_weight':'boundary','rgb_weight':'crop_rgb'}
    for arm in ARMS:
        cfg = SimpleNamespace(**coefficients(arm),**cfg_extra)
        values = objective(random_pred,latent,batch,cfg,1.)['total']
        expected = sum((getattr(cfg,k)*terms[v] for k,v in mapping.items()), torch.zeros_like(generic))
        assert torch.allclose(values-generic,expected,atol=3e-6),arm
        # Check the added gradient equals the gradient of the intended shape terms.
        actual_grad, = torch.autograd.grad(values.sum(), random_pred, retain_graph=True)
        expected_grad, = torch.autograd.grad((generic+expected).sum(), random_pred, retain_graph=True)
        assert torch.allclose(actual_grad,expected_grad,atol=3e-5),arm
        base.seed_all(7)
        model = base.SpatialLanderAE(2)
        h = model_hash(model)
        if arm=='control':
            first_hash=h
        assert h==first_hash
        pred,z = model(target)
        loss = objective(pred,z,batch,cfg,1.)['total'].mean()
        loss.backward()
        params = dict(model.named_parameters())
        for name in ('encoder.0.weight','state_head.weight','appearance_head.weight',
                     'decode_input.0.weight','decoder.0.weight'):
            grad=params[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,(arm,name)
    assert torch.allclose(chroma(target,target),torch.zeros(2))
    for arm in ARMS:
        cfg=coefficients(arm)
        assert cfg['near_weight']==1. and cfg['state_weight']==1.
        assert cfg['dice_weight']==0. and cfg['moment_weight']==0.
    print('SELF-TEST PASSED: matched random initialization, exact objectives/gradients, '
          'encoder/both heads/decoder gradients in all arms, chroma exact match.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    t = sub.add_parser('train')
    t.add_argument('--arm',choices=ARMS,required=True)
    t.add_argument('--out',required=True)
    t.add_argument('--data',required=True)
    t.add_argument('--split',required=True)
    t.add_argument('--state-cols',type=int,nargs=3,required=True)
    t.add_argument('--epochs',type=int,default=40)
    t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--batch-size',type=int,default=32)
    t.add_argument('--workers',type=int,default=4)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--appearance-channels',type=int,default=16)
    t.set_defaults(crop_size=32,min_pixels=20,near_threshold=.65,temperature=.02)
    c=sub.add_parser('compare')
    c.add_argument('--root',required=True)
    c.add_argument('--seeds',type=int,nargs='+',default=[42])
    sub.add_parser('self-test')
    args=p.parse_args()
    if args.command=='train':
        train(args)
    elif args.command=='compare':
        compare(args)
    else:
        self_test()

if __name__=='__main__':
    main()
