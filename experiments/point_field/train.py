"""Train a PointField (point_field.py) on exact scene distances; write field.pt and point-level accuracy.

Each step draws a fresh batch: 50% sphere centers of training configurations, 25% the same kind of centers
jittered by N(0, s^2) with s in {1, 5} cm, 25% uniform points in the workspace box. Training configurations
are the FK pool's training rows (pair index modulo 100 != 0, the rows the speed networks fit) and, with
--sources pool+uniform, as many uniform joint draws |q_norm| <= 0.5. Held-out evaluation uses the centers of
the pool's held-out rows (modulo 100 == 0), jittered and uniform points.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'method'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from experiments.point_field.point_field import PointField, Hierarchical, scene_sdf

BANDS = [(-1., 0.), (0., .05), (.05, .15), (.15, .5), (.5, 9.)]


def pool(sampled):
    rows = []
    for kind in ['original', 'augmentation']:
        a = np.load(Path(sampled)/'inputs'/kind/'sampled_points.npy', mmap_mode='r'); ids = np.arange(len(a))
        rows.append((np.asarray(a[ids[ids % 100 != 0]], np.float32).reshape(-1, 6), np.asarray(a[ids[ids % 100 == 0]], np.float32).reshape(-1, 6)))
    return np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])


def point_metrics(pred, true):
    """pred, true: signed distances; errors on the clamped (release) distance in mm, by true signed distance."""
    e = (pred.clamp_min(0)-true.clamp_min(0)).abs()*1000; out = dict(n=int(len(e)), mae_mm=float(e.mean()), p95_mm=float(e.quantile(.95)) if len(e) <= 2**24 else None,
                                                              max_mm=float(e.max()), signed_mae_mm=float((pred-true).abs().mean()*1000), bands=[])
    for lo, hi in BANDS:
        m = (true >= lo) & (true < hi)
        out['bands'].append(dict(low=lo, high=hi, n=int(m.sum()), mae_mm=float(e[m].mean()) if m.any() else None, bias_mm=float(((pred-true)[m]).mean()*1000) if m.any() else None))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--sampled', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--sources', choices=['pool', 'pool+uniform'], default='pool+uniform')
    p.add_argument('--steps', type=int, default=40000); p.add_argument('--batch', type=int, default=65536); p.add_argument('--seed', type=int, default=20261025)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); dev = 'cuda:0'
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    steps = 50 if a.smoke else a.steps
    exact = Hierarchical(a.scene, None, dev); robot, keep, lo, hi = exact.robot, exact.keep, exact.lo, exact.hi
    train_q, held_q = pool(a.sampled); train_q = torch.as_tensor(train_q, device=dev); held_q = torch.as_tensor(held_q, device=dev)
    S = int(keep.sum())

    @torch.no_grad()
    def centers(q):
        return robot.link_spheres(robot.radians(q))[:, keep, :3].reshape(-1, 3)

    def configs(n):
        if a.sources == 'pool': return train_q[torch.randint(len(train_q), (n,), device=dev, generator=gen)]
        k = n//2; u = torch.rand(n-k, 6, device=dev, generator=gen)-.5
        return torch.cat([train_q[torch.randint(len(train_q), (k,), device=dev, generator=gen)], u])
    # Workspace box: every sphere center of the pool and of uniform draws, plus 10 cm.
    probe = torch.cat([centers(train_q[::50]), centers(torch.rand(100000, 6, device=dev, generator=gen)-.5)])
    box_lo, box_hi = (probe.amin(0)-.1).cpu(), (probe.amax(0)+.1).cpu(); del probe

    def batch(n):
        n_c, n_j = n//2, n//4; n_u = n-n_c-n_j
        c = centers(configs((n_c+n_j)//S+1))[torch.randperm(((n_c+n_j)//S+1)*S, device=dev, generator=gen)[:n_c+n_j]]
        sig = torch.where(torch.rand(n_j, 1, device=dev, generator=gen) < .5, .01, .05)
        c[n_c:] += torch.randn(n_j, 3, device=dev, generator=gen)*sig
        u = box_lo.to(dev)+(box_hi-box_lo).to(dev)*torch.rand(n_u, 3, device=dev, generator=gen)
        x = torch.cat([c, u]); return x, scene_sdf(x, lo, hi)

    field = PointField(box_lo.tolist(), box_hi.tolist()).to(dev)
    opt = torch.optim.AdamW(field.parameters(), lr=2e-3, weight_decay=0.); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=max(.02, 10/steps))
    history = []
    for step in range(1, steps+1):
        x, t = batch(a.batch); pred = field(x)
        loss = ((pred-t).abs()*(1+4*torch.exp(-t.abs()/.1))).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss), lr=sched.get_last_lr()[0], seconds=time.monotonic()-started)); print('POINT_FIELD', history[-1], flush=True)
    field.eval().requires_grad_(False)
    torch.save(dict(config=field.config, state_dict=field.state_dict(), sources=a.sources, steps=steps, batch=a.batch, seed=a.seed, scene=str(a.scene), sampled=str(a.sampled),
                    parameters=sum(v.numel() for v in field.parameters())), a.output/'field.pt')
    # Point-level held-out accuracy.
    g = torch.Generator(device=dev).manual_seed(a.seed+1); sets = {}
    hc = centers(held_q[torch.randperm(len(held_q), device=dev, generator=g)[:4000 if not a.smoke else 200]])
    sets['heldout_sphere_centers'] = hc
    sets['heldout_centers_jitter_1cm'] = hc+torch.randn_like(hc)*.01
    sets['heldout_centers_jitter_5cm'] = hc+torch.randn_like(hc)*.05
    sets['uniform_workspace_box'] = box_lo.to(dev)+(box_hi-box_lo).to(dev)*torch.rand(200000, 3, device=dev, generator=g)
    report = dict(sources=a.sources, steps=steps, batch=a.batch, seed=a.seed, parameters=sum(v.numel() for v in field.parameters()), box_lo=box_lo.tolist(), box_hi=box_hi.tolist(),
                  training_configs=int(len(train_q)), history=history, sets={})
    with torch.no_grad():
        for name, x in sets.items():
            preds = torch.cat([field(x[s:s+65536]) for s in range(0, len(x), 65536)])
            report['sets'][name] = point_metrics(preds, scene_sdf(x, lo, hi)); print('POINT_ACCURACY', name, round(report['sets'][name]['mae_mm'], 3), flush=True)
    report['elapsed_seconds'] = time.monotonic()-started
    (a.output/'point_accuracy.json').write_text(json.dumps(report, indent=1)+'\n'); print('POINT_FIELD_COMPLETE', flush=True)


if __name__ == '__main__': main()
