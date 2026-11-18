import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .occlude_bp import final_checkpoint_epochs, segment_importance, read_pids
from sehnet.evaluate import build_model
from sehnet.config import get_path
from sehnet.data import load_terms


def main():
    parser = argparse.ArgumentParser(description='Compare routing and gradient-times-input on a fixed BP cohort.')
    parser.add_argument('--run-dir', type=Path, required=True, help='BP analysis run directory containing the final five training checkpoints')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True, help='pair_results.csv produced by analysis.occlude_bp')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory to avoid overwriting results')
    start = time.monotonic()
    with args.pairs.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['pid'] for r in rows}) != len(rows):
        raise ValueError('Expected one nonempty protein–GO pair per protein')
    run = args.run_dir
    terms = load_terms(get_path('label_dir', args.data_root), 'bp')
    term_indices = {t: i for i, t in enumerate(terms)}
    with (args.data_root / 'features_3B/bp_test_esm2.pkl').open('rb') as f:
        global_features = pickle.load(f)
    local_dir = args.data_root / 'features_150M_multilayer_segments/bp'
    index = {p: i for i, p in enumerate(read_pids(local_dir / 'test_pids.txt'))}
    segments = np.load(local_dir / 'test_segments.npy', mmap_mode='r')
    anchors = np.load(local_dir / 'test_global.npy', mmap_mode='r')
    masks = np.load(local_dir / 'test_segment_mask.npy', mmap_mode='r')
    xs, ms, gs, targets = [], [], [], []
    for row in rows:
        i = index[row['pid']]
        mask = np.asarray(masks[i], dtype=bool)
        x = np.asarray(segments[i], dtype=np.float32).copy()
        x[:, mask] -= np.asarray(anchors[i], dtype=np.float32)[:, None, :]
        x[:, ~mask] = 0
        if mask.sum() != int(row['valid_segments']):
            raise ValueError(f"Segment mask disagrees for {row['pid']}")
        feature = global_features[row['pid']]
        if isinstance(feature, torch.Tensor): feature = feature.detach().cpu().numpy()
        xs.append(x); ms.append(mask); gs.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        targets.append(term_indices[row['target_term']])
    xs = torch.from_numpy(np.stack(xs)); ms = torch.from_numpy(np.stack(ms))
    gs = torch.from_numpy(np.stack(gs)); targets = torch.tensor(targets, dtype=torch.long)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    model = build_model(config, len(terms), terms, args.data_root, 'bp', device).to(device).eval()
    for param in model.parameters(): param.requires_grad_(False)
    n = len(rows)
    routing = np.zeros((n, 32), np.float64)
    gradient = np.zeros((n, 32), np.float64)
    baseline = np.zeros(n, np.float64)
    paths = [run / 'trajectory_checkpoints' / f'epoch_{e:03d}.pth' for e in final_checkpoint_epochs(run)]
    for path in paths:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state'])
        for begin in range(0, n, args.batch_size):
            end = min(begin + args.batch_size, n)
            x = xs[begin:end].to(device).detach().requires_grad_(True)
            m = ms[begin:end].to(device); g = gs[begin:end].to(device); t = targets[begin:end].to(device)
            routing[begin:end] += segment_importance(model, x.detach(), m, t, 'routing').cpu().numpy()
            probs = model(g, local_segments=x, local_mask=m).probs.float()
            selected = probs.gather(1, t[:, None]).squeeze(1)
            grad = torch.autograd.grad(selected.sum(), x)[0]
            gradient[begin:end] += (grad * x).sum(dim=(1, 3)).detach().cpu().numpy()
            baseline[begin:end] += selected.detach().cpu().numpy()
        print(f'[ranking] {path.name}: {time.monotonic()-start:.1f}s', flush=True)
    baseline /= len(paths); routing /= len(paths); gradient /= len(paths)
    old = np.asarray([float(r['baseline_target_raw_probability']) for r in rows])
    error = float(np.max(np.abs(baseline-old)))
    if error > 0.001:
        raise ValueError('Original baseline differs by >0.001; check model, data and checkpoints')
    valid = ms.numpy()
    choices = np.stack([np.argmax(np.where(valid, rank, -np.inf), axis=1) for rank in (routing, gradient)])
    masked = np.zeros((2,n), np.float64)
    for path in paths:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state'])
        for method in range(2):
            for begin in range(0,n,args.batch_size):
                end = min(begin+args.batch_size,n)
                x = xs[begin:end].to(device).clone(); m = ms[begin:end].to(device).clone()
                ids = torch.arange(end-begin, device=device)
                slots = torch.as_tensor(choices[method,begin:end], device=device)
                x[ids,:,slots,:] = 0; m[ids,slots] = False
                with torch.inference_mode():
                    prob = model(gs[begin:end].to(device), local_segments=x, local_mask=m).probs.float()
                    masked[method,begin:end] += prob.gather(1,targets[begin:end].to(device)[:,None]).squeeze(1).cpu().numpy()
        print(f'[occlusion] {path.name}: {time.monotonic()-start:.1f}s', flush=True)
    masked /= len(paths)
    random = np.asarray([float(r['mean_random_target_probability_drop']) for r in rows])
    original = np.asarray([float(r['top_target_probability_drop']) for r in rows])
    drops = np.vstack([original, baseline-masked[0], baseline-masked[1]])
    rng = np.random.default_rng(20260809)
    samples = rng.integers(0,n,size=(10000,n))
    summary = {}
    for name, values in zip(('signed_support','routing','gradient_x_input'), drops):
        diff = values-random
        ci = np.quantile(diff[samples].mean(axis=1),[0.025,0.975])
        summary[name] = {'mean_probability_drop':float(values.mean()), 'mean_minus_original_random':float(diff.mean()), 'bootstrap_95_ci':ci.tolist()}
    summary['original_random_mean_drop'] = float(random.mean())
    summary['n_pairs'] = n; summary['max_baseline_difference'] = error
    summary['seconds'] = time.monotonic()-start
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'pair_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['pair_id','pid','target_term','routing_top','gradient_x_input_top','signed_support_drop','routing_drop','gradient_x_input_drop','original_random_mean_drop'])
        for i,row in enumerate(rows): w.writerow([row['pair_id'],row['pid'],row['target_term'],int(choices[0,i]),int(choices[1,i]),*drops[:,i],random[i]])
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Results: {args.output_dir}',flush=True)

if __name__ == '__main__':
    main()
