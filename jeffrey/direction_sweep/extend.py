"""Nested direction-pair extension of a frozen transition bank (same source poses, more random walk directions).

The frozen bank stores, for every source pose q, 4 random unit directions d_1..d_4 in both signs and 4 step sizes:
32 actions z = q + h * (+-d_k), index scale*8 + sign*4 + k. This script keeps every frozen source and the frozen
4 pairs verbatim and appends pairs 5..--pairs (new random unit directions, seed --seed), evaluated with the same
rules as the bank builder (experiments/no_wall_rebuild/bank.py): 17-point trapezoid integral of ds / S with the
frozen speed network, the domain box |z| <= 0.5, the scene's exact local-edge gate at 0.02 rad, minimum speed 0.015.

Saved arrays (ext/), layout [source, scale, sign, pair] so that pairs[:D] is the D-pair bank and D = 4 is the
frozen bank bit for bit:
  successor_q (N,4,2,P,6) float32   local_cost (N,4,2,P) float32   safe (N,4,2,P) bool
  in_domain   (N,4,2,P) bool        minimum_speed (N,4,2,P) float32
audit.json: the same code re-evaluated on the frozen 4 pairs of --audit sources, compared with the frozen arrays.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
from speed import load_speed

STEPS = (.01, .05, .15, .25)
NAMES = ('successor_q', 'local_cost', 'safe', 'in_domain', 'minimum_speed')


def load_gate(scene):
    # Both gate modules are called geometry.py; load the scene's one by path.
    path = '/clean/experiments/cabinet/geometry.py' if scene else '/clean/experiments/no_wall_rebuild/geometry.py'
    spec = importlib.util.spec_from_file_location('gate_geometry', path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.ClearanceGate(scene) if scene else module.ExactLocalGate(RELEASE/'configs/scene.yml')


@torch.no_grad()
def evaluate(source, direction, speed, gate, chunk=128):
    """source (n,6), direction (n,p,6) unit -> arrays of shape (n,4,2,p[,6]) in the bank's rules."""
    n, p = direction.shape[:2]
    sizes = np.asarray(STEPS, np.float32)[:, None, None, None]; signs = np.array([1, -1], np.float32)[None, :, None, None]
    z = (source[:, None, None, None, :] + sizes*signs*direction[:, None, None, :, :]).astype(np.float32)   # (n,4,2,p,6)
    flat = z.reshape(n, -1, 6); domain = ((flat >= -.5) & (flat <= .5)).all(2)
    cost = np.empty(domain.shape, np.float32); minima = np.empty_like(cost)
    alpha = torch.linspace(0, 1, 17, device='cuda:0')
    for b in range(0, n, chunk):
        e = min(b+chunk, n); q = torch.tensor(source[b:e], device='cuda:0'); zz = torch.tensor(flat[b:e], device='cuda:0')
        chord = q[:, None, None, :]+alpha[None, None, :, None]*(zz[:, :, None, :]-q[:, None, None, :])
        ss = speed(chord.reshape(-1, 6)).reshape(e-b, flat.shape[1], 17)
        cost[b:e] = (torch.linalg.vector_norm(zz-q[:, None, :], dim=2)*(torch.trapezoid(ss.clamp_min(1e-5).reciprocal(), dim=-1)/16)).cpu().numpy()
        minima[b:e] = ss.amin(2).cpu().numpy()
    exact = gate.edges(source, flat, domain)
    safe = domain & exact & (minima >= .015)
    shape = (n, 4, 2, p)
    return dict(successor_q=z, local_cost=cost.reshape(shape), safe=safe.reshape(shape), in_domain=domain.reshape(shape), minimum_speed=minima.reshape(shape))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--scene', type=Path, help='cabinet scene; omitted: no-wall')
    p.add_argument('--pairs', type=int, default=16); p.add_argument('--seed', type=int, default=20261028); p.add_argument('--block', type=int, default=8192)
    p.add_argument('--audit', type=int, default=16384); p.add_argument('--output', type=Path, required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    assert a.pairs > 4
    out = a.output/'ext'; out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    bank = a.prepared/'bank'; manifest = json.loads((bank/'manifest.json').read_text())
    assert manifest['direction_pairs'] == 4 and list(manifest['step_sizes']) == list(STEPS) and manifest['actions_per_anchor'] == 32
    frozen = {k: np.load(bank/(k+'.npy'), mmap_mode='r') for k in ('source_q',)+NAMES}
    source = frozen['source_q']; n = 2048 if a.smoke else len(source); P = a.pairs; extra = P-4
    speed = load_speed(a.prepared/'speed.pt', 'cuda:0'); gate = load_gate(a.scene)
    rng = np.random.default_rng(a.seed)
    shapes = dict(successor_q=(n, 4, 2, P, 6), local_cost=(n, 4, 2, P), safe=(n, 4, 2, P), in_domain=(n, 4, 2, P), minimum_speed=(n, 4, 2, P))
    dtypes = dict(successor_q=np.float32, local_cost=np.float32, safe=bool, in_domain=bool, minimum_speed=np.float32)
    arrays = {k: np.lib.format.open_memmap(out/(k+'.npy'), mode='w+', dtype=dtypes[k], shape=shapes[k]) for k in NAMES}
    progress = []
    for b in range(0, n, a.block):
        e = min(b+a.block, n); q = np.asarray(source[b:e], np.float32).copy()
        for k in NAMES:   # frozen pairs verbatim: index scale*8 + sign*4 + pair
            value = np.asarray(frozen[k][b:e]); arrays[k][b:e, :, :, :4] = value.reshape(e-b, 4, 2, 4, *value.shape[2:])
        d = rng.normal(size=(e-b, extra, 6)).astype(np.float32); d /= np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-8)
        new = evaluate(q, d, speed, gate)
        for k in NAMES: arrays[k][b:e, :, :, 4:] = new[k]
        progress.append(dict(rows=e, seconds=time.monotonic()-started, safe_fraction_new=float(new['safe'].mean())))
        print('EXTEND', progress[-1], flush=True); write(a.output/'progress.json', progress)
    for k in NAMES: arrays[k].flush()
    del arrays
    # Audit: the same evaluation applied to the frozen directions must reproduce the frozen arrays.
    ids = np.sort(np.random.default_rng(1).choice(n, min(a.audit, n), replace=False))
    q = np.asarray(source[ids], np.float32); z = np.asarray(frozen['successor_q'][ids]).reshape(len(ids), 4, 2, 4, 6)
    d = (z[:, 3, 0]-q[:, None, :])/np.float32(.25)
    again = evaluate(q, d, speed, gate)
    ref = {k: np.asarray(frozen[k][ids]).reshape(len(ids), 4, 2, 4, *(() if k != 'successor_q' else (6,))) for k in NAMES}
    audit = dict(sources=len(ids), successor_max_abs=float(np.abs(again['successor_q']-ref['successor_q']).max()),
                 cost_max_rel=float((np.abs(again['local_cost']-ref['local_cost'])/ref['local_cost']).max()),
                 safe_agreement=float((again['safe'] == ref['safe']).mean()), domain_agreement=float((again['in_domain'] == ref['in_domain']).mean()),
                 safe_fraction_frozen=float(ref['safe'].mean()), safe_fraction_recomputed=float(again['safe'].mean()))
    print('AUDIT', audit, flush=True)
    safe = np.load(out/'safe.npy', mmap_mode='r'); counts = {}
    for D in (1, 2, 4, 8, 16, 32):
        if D > P: break
        none = 0; mean = 0.
        for b in range(0, n, 65536):
            s = np.asarray(safe[b:b+65536, :, :, :D]).reshape(-1, 8*D); none += int((~s.any(1)).sum()); mean += float(s.sum())
        counts[D] = dict(actions=8*D, sources_without_safe_action=none, mean_safe_actions=mean/n)
    files = {k+'.npy': sha(out/(k+'.npy')) for k in NAMES}
    write(out/'manifest.json', dict(schema_version=1, sources=n, pairs=P, frozen_pairs=4, layout='[source, scale, sign, pair]', step_sizes=list(STEPS),
          seed=a.seed, frozen_bank_manifest_sha256=sha(bank/'manifest.json'), source_q_sha256=sha(bank/'source_q.npy'), speed_sha256=sha(a.prepared/'speed.pt'),
          scene=str(a.scene) if a.scene else 'no-wall release scene', scene_sha256=sha(a.scene) if a.scene else manifest['scene_sha256'],
          rules=dict(quadrature_points=17, minimum_speed=.015, domain='|z|<=0.5', gate='cabinet ClearanceGate' if a.scene else 'no_wall_rebuild ExactLocalGate'),
          audit=audit, by_pairs=counts, file_sha256=files, smoke=a.smoke, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__': main()
