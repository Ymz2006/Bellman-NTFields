"""Idea 1, Bellman-sharp: start from a trained P-NTFields field and refine it with Bellman updates only.

Generation k: a frozen copy T_k of the current field is the Bellman target and the field is trained on

    L = SmoothL1( T(q, g),  min_a [ c(q, a) + T_k(q'_a, g) ] )         (safe actions a only)

over the offline transition bank of the cabinet Bellman baseline (results/<run>/prepared/bank: 399,581 anchor
poses x 32 candidate actions, the exact collision/self-collision veto `safe`, goals = random anchors, as
method/training.py). No Eikonal, no direction term, no P-NTFields loss: the update is the Bellman backup only,
applied to a model that is already a good travel-time field and is wrong mainly where its straight-ish geodesics
cross obstacles (the safe veto removes those actions from the backup, so T rises there).

Local cost c(q, a) is in P-NTFields' own units so that the backup is consistent with the pretrained field:
coordinates x = q_rad / (2 pi), speed S = clip(d_env, m/10, m) / m with m = 0.5/12 m and d_env the exact
sphere-cuboid clearance without the shoulder spheres (experiments/pntfields_cabinet/pipeline.py's label), and
c = |dx| * trapezoid mean of 1/S over 17 points of the straight edge. Speed is queried from the exact scene
distance (the fastest option: no network).

Outputs: local_cost_pnt.npy, generation checkpoints gen_XX.pt in P-NTFields' format (loadable by
experiments/pntfields_cabinet/field.load_pnt / evaluate.py), progress.json with per-generation diagnostics.
"""
import argparse, copy, importlib.util, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'method'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from jeffrey.geometry import ArmClearance
from experiments.face_speed.robot import IGNORED_SPHERES

MARGIN = .5/12; OFFSET = MARGIN/10


def pnt_module():
    spec = importlib.util.spec_from_file_location('pntfields_cabinet_field', ROOT/'experiments/pntfields_cabinet/field.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Travel(torch.nn.Module):
    """P-NTFields' T = |x1 - x0| / tau(x0, x1) on release-normalized inputs (x = q_norm * q_range / 2 pi)."""
    def __init__(self, net, device):
        super().__init__(); self.net = net
        q_range = torch.tensor([12.5664, 12.5664, 6.2832, 12.5664, 12.5664, 12.5664], device=device)
        self.register_buffer('k', q_range/(2*math.pi))

    def forward(self, q, g):
        x0, x1 = q*self.k, g*self.k
        tau, _ = self.net.out(torch.cat([x0, x1], 1))
        return torch.linalg.vector_norm(x1-x0, dim=1)/tau[:, 0]


@torch.no_grad()
def recost(bank, scene, out, device, smoke, quadrature=17, chunk=96):
    source = np.load(bank/'source_q.npy', mmap_mode='r'); succ = np.load(bank/'successor_q.npy', mmap_mode='r')
    n = 4096 if smoke else len(source); A = succ.shape[1]
    clear = ArmClearance(scene, None, 0., device); keep = torch.ones(len(clear.robot.sphere_radii), dtype=torch.bool, device=device); keep[list(IGNORED_SPHERES)] = False
    k = torch.tensor([2, 2, 1, 2, 2, 2.], device=device)                                     # q_range / 2 pi
    cost = np.lib.format.open_memmap(out, mode='w+', dtype=np.float32, shape=(n, A))
    w = torch.linspace(0, 1, quadrature, device=device); trap = torch.ones(quadrature, device=device); trap[[0, -1]] = .5; trap /= trap.sum()
    started = time.monotonic()
    for s in range(0, n, chunk):
        e = min(n, s+chunk); q = torch.as_tensor(np.asarray(source[s:e]), device=device); z = torch.as_tensor(np.asarray(succ[s:e]), device=device)
        pts = q[:, None, None]+(z-q[:, None])[:, :, None]*w[None, None, :, None]                 # (b, A, Qd, 6)
        env, _ = clear.components(pts.reshape(-1, 6)); d = env[:, keep].amin(1)
        inv = (MARGIN/d.clamp(OFFSET, MARGIN)).reshape(e-s, A, quadrature)
        length = torch.linalg.vector_norm((z-q[:, None])*k, dim=-1)
        cost[s:e] = (length*(inv*trap).sum(-1)).cpu().numpy()
    cost.flush(); return n, time.monotonic()-started


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=10); p.add_argument('--epochs', type=float, default=1., help='bank epochs per generation')
    p.add_argument('--batch', type=int, default=256); p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--clip', type=float, default=2.); p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); dev = 'cuda:0'; started = time.monotonic()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); rng = np.random.default_rng(a.seed)
    n, recost_seconds = recost(a.bank, a.scene, a.output/'local_cost_pnt.npy', dev, a.smoke)
    print('RECOST', n, recost_seconds, flush=True)
    source = np.load(a.bank/'source_q.npy', mmap_mode='r'); succ = np.load(a.bank/'successor_q.npy', mmap_mode='r'); safe = np.load(a.bank/'safe.npy', mmap_mode='r')
    cost = np.load(a.output/'local_cost_pnt.npy', mmap_mode='r')
    md = pnt_module().pnt_module(); payload = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    net = md.NN(dev, 6, payload['B_state_dict']); net.load_state_dict(payload['model_state_dict'], strict=True); net.to(torch.device(dev)).float()
    model = Travel(net, dev).train(); target = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    generations = 2 if a.smoke else a.generations
    steps_per_gen = 3 if a.smoke else int(round(a.epochs*n/a.batch))
    probe = rng.choice(n, 4096, replace=False); probe_goal = rng.choice(n, 4096, replace=False)

    def batch(rows, goals):
        q, z, c, s = [torch.as_tensor(np.asarray(x[rows]).copy(), device=dev) for x in (source, succ, cost, safe)]
        return q, z, c, s.bool(), torch.as_tensor(np.asarray(source[goals]).copy(), device=dev)

    @torch.no_grad()
    def backup(tgt, q, z, c, s, g):
        tail = tgt(z.reshape(-1, 6), g[:, None].expand_as(z).reshape(-1, 6)).reshape(c.shape)
        return (c+tail).masked_fill(~s, torch.inf).amin(1)

    @torch.no_grad()
    def diagnose(tag):
        rows = []
        for s in range(0, len(probe), 512):
            q, z, c, sf, g = batch(probe[s:s+512], probe_goal[s:s+512]); ok = sf.any(1)
            v = model(q, g); b = backup(model, q, z, c, sf, g); rows.append(torch.stack([v[ok], b[ok]], 1))
        r = torch.cat(rows); r = r[r[:, 0] > 1e-2]; v, b = r[:, 0], r[:, 1]; rel = (b-v)/v
        return dict(tag=tag, mean_T=float(v.mean()), bellman_mae=float((v-b).abs().mean()), median_rel_gap=float(rel.median()), mean_rel_gap=float(rel.mean()), frac_backup_above_10pct=float((rel > .1).float().mean()),
                    frac_backup_below_10pct=float((rel < -.1).float().mean()))

    def save(gen):
        state = {k[len('net.'):]: v for k, v in model.state_dict().items() if k.startswith('net.')}
        torch.save({**payload, 'model_state_dict': state, 'bellman_sharp': dict(generation=gen, base_checkpoint=str(a.checkpoint), lr=a.lr, batch=a.batch,
                    steps_per_generation=steps_per_gen)}, a.output/f'gen_{gen:02d}.pt')

    history = [diagnose('generation 0 (pretrained P-NTFields)')]; print('BELLMAN_SHARP', history[-1], flush=True)
    step = 0
    for gen in range(1, generations+1):
        order = rng.permutation(n); pos = 0; losses = []
        for _ in range(steps_per_gen):
            if pos+a.batch > n: order = rng.permutation(n); pos = 0
            rows = np.sort(order[pos:pos+a.batch]); pos += a.batch
            q, z, c, s, g = batch(rows, rng.choice(n, len(rows), replace=False)); ok = s.any(1)
            b = backup(target, q, z, c, s, g)
            loss = F.smooth_l1_loss(model(q[ok], g[ok]), b[ok])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); step += 1; losses.append(float(loss))
            if not math.isfinite(losses[-1]): raise FloatingPointError('non-finite Bellman loss')
        target.load_state_dict(model.state_dict()); save(gen)
        history.append(dict(generation=gen, steps=step, loss_mean=float(np.mean(losses)), loss_last100=float(np.mean(losses[-100:])), **diagnose(f'generation {gen}'),
                            seconds=time.monotonic()-started))
        print('BELLMAN_SHARP', history[-1], flush=True)
        (a.output/'progress.json').write_text(json.dumps(dict(complete=False, history=history), indent=1)+'\n')
    (a.output/'progress.json').write_text(json.dumps(dict(complete=True, generations=generations, steps_per_generation=steps_per_gen, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay,
        base_checkpoint=str(a.checkpoint), bank=str(a.bank), recost_seconds=recost_seconds, history=history, elapsed_seconds=time.monotonic()-started, smoke=a.smoke), indent=1)+'\n')
    print('BELLMAN_SHARP_COMPLETE', flush=True)


if __name__ == '__main__': main()
