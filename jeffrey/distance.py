"""Train the hierarchical distance model's learned part, D_theta(p) = distance from a 3-D point to the scene, for
guarding (geometry.py ArmClearance), and calibrate its over-estimation bound eps.

Same PointField as experiments/point_field (Fourier features + 5 x 256 SiLU MLP), trained for the guard's needs:
  - points: centres of ALL 53 collision spheres (the scorer checks all of them) of FK-pool training rows and of
    uniform joint draws (the planner goes anywhere in the joint box), the same centres jittered by 1/3/5 cm,
    points within a few cm of the cuboid surfaces (where the guard decides), and uniform workspace points;
  - loss: L1 on the signed distance, weighted towards the surface, with over-estimation (pred > true, the unsafe
    direction) weighted --over times more than under-estimation, so the model errs on the safe side.
Calibration: on fresh points of every kind, the over-estimation (pred - true of the release's unsigned distance)
among points with true distance <= 0.10 m (where a guard threshold can bind); eps = its maximum over the sets the
guard can query (sphere centres of pool / uniform configurations, jittered centres), rounded up to 0.5 mm; uniform
workspace points are reported but not used (no sphere centre reaches most of them). The planner's obstacle threshold is then 5 mm + eps.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'method'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from experiments.point_field.point_field import PointField, scene_sdf
from experiments.point_field.train import pool
from jeffrey.geometry import scene_boxes
from experiments.face_speed import robot as R

NEAR = .10


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--sampled', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--steps', type=int, default=60000); p.add_argument('--batch', type=int, default=65536)
    p.add_argument('--width', type=int, default=256); p.add_argument('--depth', type=int, default=5); p.add_argument('--sigma', type=float, default=8.)
    p.add_argument('--over', type=float, default=4.); p.add_argument('--seed', type=int, default=20261026); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); dev = 'cuda:0'
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    steps = 50 if a.smoke else a.steps
    robot = R.Robot().to(dev); lo, hi = scene_boxes(a.scene, dev)
    train_q, held_q = pool(a.sampled); train_q = torch.as_tensor(train_q, device=dev); held_q = torch.as_tensor(held_q, device=dev)
    S = len(robot.sphere_radii)

    @torch.no_grad()
    def centers(q):
        return robot.link_spheres(robot.radians(q))[..., :3].reshape(-1, 3)

    def configs(n, g=gen, rows=train_q):
        k = n//2; return torch.cat([rows[torch.randint(len(rows), (k,), device=dev, generator=g)], torch.rand(n-k, 6, device=dev, generator=g)-.5])

    probe = torch.cat([centers(train_q[::50]), centers(torch.rand(100000, 6, device=dev, generator=gen)-.5)])
    box_lo, box_hi = (probe.amin(0)-.1).cpu(), (probe.amax(0)+.1).cpu(); del probe
    faces = (hi-lo)

    def surface(n, g=gen):
        """Points near the cuboid surfaces: a uniform point on a random cuboid face, offset N(0, 3 cm) along it."""
        k = torch.randint(len(lo), (n,), device=dev, generator=g); axis = torch.randint(3, (n,), device=dev, generator=g)
        side = torch.randint(2, (n,), device=dev, generator=g).float()
        x = lo[k]+faces[k]*torch.rand(n, 3, device=dev, generator=g)
        face = torch.where(side[:, None] > 0, hi[k], lo[k]).gather(1, axis[:, None])[:, 0]
        x[torch.arange(n, device=dev), axis] = face+torch.randn(n, device=dev, generator=g)*.03
        return x

    def batch(n, g=gen, rows=train_q):
        n_c, n_j, n_s = (3*n)//8, n//4, n//4; n_u = n-n_c-n_j-n_s
        c = centers(configs((n_c+n_j)//S+1, g, rows)); c = c[torch.randperm(len(c), device=dev, generator=g)[:n_c+n_j]]
        sig = torch.tensor([.01, .03, .05], device=dev)[torch.randint(3, (n_j,), device=dev, generator=g)][:, None]
        c[n_c:] += torch.randn(n_j, 3, device=dev, generator=g)*sig
        u = box_lo.to(dev)+(box_hi-box_lo).to(dev)*torch.rand(n_u, 3, device=dev, generator=g)
        x = torch.cat([c, surface(n_s, g), u]); return x, scene_sdf(x, lo, hi)

    field = PointField(box_lo.tolist(), box_hi.tolist(), a.width, a.depth, 128, a.sigma, a.seed).to(dev)
    opt = torch.optim.AdamW(field.parameters(), lr=2e-3, weight_decay=0.); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=max(.02, 10/steps))
    history = []
    for step in range(1, steps+1):
        x, t = batch(a.batch); err = field(x)-t
        w = (1+9*torch.exp(-t.abs()/.03))*torch.where(err > 0, a.over, 1.)
        loss = (err.abs()*w).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss), seconds=time.monotonic()-started)); print('DISTANCE', history[-1], flush=True)
    field.eval().requires_grad_(False)
    # Calibration on fresh points (another generator; held-out pool rows), the release's unsigned distance.
    g = torch.Generator(device=dev).manual_seed(a.seed+1); over, report = [], {}
    with torch.no_grad():
        for name, make in [('heldout_centers', lambda: centers(configs(65536//S, g, held_q))), ('uniform_q_centers', lambda: centers(torch.rand(65536//S, 6, device=dev, generator=g)-.5)),
                           ('jittered', lambda: batch(65536, g, held_q)[0][(3*65536)//8:(5*65536)//8]), ('surface', lambda: surface(65536, g)),
                           ('workspace', lambda: box_lo.to(dev)+(box_hi-box_lo).to(dev)*torch.rand(65536, 3, device=dev, generator=g))]:
            e_all, t_all = [], []
            for _ in range(2 if a.smoke else 150):
                x = make(); t = scene_sdf(x, lo, hi).clamp_min(0); e_all.append(field(x).clamp_min(0)-t); t_all.append(t)
            e, t = torch.cat(e_all), torch.cat(t_all); near = t <= NEAR
            report[name] = dict(n=int(len(e)), near=int(near.sum()), mae_mm=float(e.abs().mean()*1000), near_mae_mm=float(e[near].abs().mean()*1000),
                                near_over_max_mm=float(e[near].max()*1000), near_over_p9999_mm=float(e[near].quantile(.9999)*1000) if near.sum() < 2**24 else None,
                                near_under_max_mm=float(-e[near].min()*1000), near_bias_mm=float(e[near].mean()*1000))
            if name != 'workspace': over.append(float(e[near].max()))                     # the guard only queries FK sphere centres
            print('CALIBRATION', name, report[name], flush=True)
    eps = float(np.ceil(max(over)*2000)/2000)
    torch.save(dict(config=field.config, state_dict=field.state_dict(), eps=eps, steps=steps, batch=a.batch, over=a.over, seed=a.seed, scene=str(a.scene), sampled=str(a.sampled),
                    parameters=sum(v.numel() for v in field.parameters())), a.output/'distance.pt')
    (a.output/'calibration.json').write_text(json.dumps(dict(eps_m=eps, near_m=NEAR, sets=report, history=history, steps=steps, over_weight=a.over, width=a.width, depth=a.depth,
                                                              elapsed_seconds=time.monotonic()-started, smoke=a.smoke), indent=1)+'\n')
    print('DISTANCE_COMPLETE eps', eps, flush=True)


if __name__ == '__main__': main()
