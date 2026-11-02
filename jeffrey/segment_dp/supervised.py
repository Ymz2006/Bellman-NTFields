"""The "correct local field" (segment-DP task 8): specialists 1..K regressed on path-optimized local travel times.

Same networks, segments, pair distribution and optimizer as the eikonal specialists of eikonal.py (TravelCost, one
model per segment of 0.08 widened by the blend, batched as one Ensemble with separate batches / clipping / AdamW
state, AdamW 2e-4, weight decay 1e-6, clip 2, batch 256, 15,620 updates per model), but trained by regression on the
labels of local_truth.py instead of the eikonal residual:
    value      (log T(q, g) - log T*(q, g))^2
    gradient   --w-grad x 1/2 [ |S(q) (grad_q T - grad_q T*)|^2 + |S(g) (grad_g T - grad_g T*)|^2 ]
               (grad T* from the optimized path by the envelope theorem; S |grad T*| = 1, so the term is
               dimensionless: it pins both the norm and the direction of the field's gradient)
2 % of each segment's labels are held out. Writes field.pt: a segment field over the full EDGES whose models
1..K are the supervised specialists and whose models K+1..M are copies of model K, i.e. model K answers every
longer gap (as derived.py --first K): the frozen evaluation of this run is the SL1-K control, and a DP chain
(dp3.py --base-models K) replaces the copies by its own levels. progress.json, config.json.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import sha, write
from fields import Ensemble, build, clip_per_model, save_field
from sampler import load_pool
from eikonal import EDGES


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--labels', type=Path, nargs='+', required=True); p.add_argument('--models', type=int, default=2)
    p.add_argument('--edges', default=EDGES); p.add_argument('--blend', type=float, default=0.016); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--batch', type=int, default=256); p.add_argument('--w-grad', type=float, default=1.0); p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.time(); updates = 40 if a.smoke else a.updates
    edges = [float(x) for x in a.edges.split(',')]; K = a.models
    torch.manual_seed(a.seed); generator = torch.Generator(device='cuda:0').manual_seed(a.seed)
    pool, speeds, _, pool_info = load_pool(a.prepared, 'cuda:0')
    parts = [np.load(f) for f in a.labels]; meta = [json.loads(f.with_suffix('.json').read_text()) for f in a.labels]
    assert all(m['source_q_sha256'] == pool_info['source_q_sha256'] and m['speed_sha256'] == pool_info['speed_sha256'] for m in meta), 'labels of another pool / speed'
    assert all(m['edges'] == edges and m['blend'] == a.blend for m in meta)
    cat = {k: torch.as_tensor(np.concatenate([z[k] for z in parts]), device='cuda:0') for k in ('i', 'j', 'segment', 'T', 'grad_q', 'grad_g')}
    assert bool((torch.linalg.vector_norm(pool[cat['i']]-pool[cat['j']], dim=1) > 1e-3).all())
    train, hold = [], []
    for m in range(K):
        rows = torch.nonzero(cat['segment'] == m).flatten(); assert len(rows) > 1000, (m, len(rows))
        rows = rows[torch.randperm(len(rows), generator=generator, device='cuda:0')]; hold.append(rows[:len(rows)//50]); train.append(rows[len(rows)//50:])
    ensemble = Ensemble([build('pair').cuda() for _ in range(K)]).cuda().train()
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=2e-4, weight_decay=1e-6)
    config = dict(models=K, edges=edges, blend=a.blend, updates_per_model=updates, batch_per_model=a.batch, w_grad=a.w_grad, seed=a.seed, learning_rate=2e-4, weight_decay=1e-6,
                  gradient_clip=2.0, loss=f'(log T/T*)^2 + {a.w_grad:g} x 1/2 (|S(q)(dT/dq - dT*/dq)|^2 + |S(g)(dT/dg - dT*/dg)|^2)', labels=[str(f) for f in a.labels],
                  labels_sha256=[sha(f) for f in a.labels], label_settings=meta, train_rows=[len(t) for t in train], held_out_rows=[len(h) for h in hold], smoke=a.smoke,
                  filler='models K+1..M are copies of model K (model K answers every longer gap)', **pool_info)
    write(a.output/'config.json', config)

    def terms(rows, create_graph):
        i, j = cat['i'][rows], cat['j'][rows]
        q = pool[i].view(K, -1, 6).clone().requires_grad_(True); g = pool[j].view(K, -1, 6).clone().requires_grad_(True)
        value = ensemble(q, g)[..., 0]
        dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=create_graph)
        sq, sg = speeds[i].view(K, -1, 1), speeds[j].view(K, -1, 1)
        vterm = (value.clamp_min(1e-6).log()-cat['T'][rows].view(K, -1).log()).square().mean(1)
        gterm = 0.5*((sq*(dq-cat['grad_q'][rows].view(K, -1, 6))).square().sum(2)+(sg*(dg-cat['grad_g'][rows].view(K, -1, 6))).square().sum(2)).mean(1)
        cos = torch.nn.functional.cosine_similarity(dq, cat['grad_q'][rows].view(K, -1, 6), dim=2).mean(1)
        return vterm, gterm, cos, value

    history, acc = [], torch.zeros((3, K), device='cuda:0', dtype=torch.float64); n = 0
    for step in range(1, updates+1):
        rows = torch.cat([t[torch.randint(len(t), (a.batch,), generator=generator, device='cuda:0')] for t in train])
        vterm, gterm, cos, _ = terms(rows, True); per_model = vterm+a.w_grad*gterm; loss = per_model.sum()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = clip_per_model(ensemble.parameters(), 2.0)
        if not bool(torch.isfinite(norm).all()): raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        acc += torch.stack((vterm.detach(), gterm.detach(), cos.detach())).double(); n += 1
        if step % 500 == 0 or step == updates:
            row = dict(step=step, seconds=time.time()-started, value=(acc[0]/n).tolist(), gradient=(acc[1]/n).tolist(), grad_cosine=(acc[2]/n).tolist())
            history.append(row); print(json.dumps({k: [round(x, 5) for x in v] if isinstance(v, list) else round(v) for k, v in row.items()}), flush=True)
            acc.zero_(); n = 0; write(a.output/'progress.json', dict(complete=False, history=history))
    ensemble.eval()
    held = []
    for m in range(K):   # held-out fit, model by model (rows of model m only)
        h = hold[m][:(len(hold[m])//256)*256].view(-1, 256); stats = torch.zeros(4, device='cuda:0', dtype=torch.float64)
        for c in range(len(h)):
            rows = torch.cat([h[c]]*K)   # every model sees the same rows; read model m's terms
            vterm, gterm, cos, value = terms(rows, False)
            ratio = value[m].detach()/cat['T'][h[c]]
            stats += torch.stack((ratio.log().abs().mean(), gterm[m].detach(), cos[m].detach(), ratio.median())).double()
        stats /= max(1, len(h)); held.append(dict(segment=m+1, mean_abs_log=float(stats[0]), gradient=float(stats[1]), grad_cosine=float(stats[2]), median_ratio=float(stats[3])))
    states = [ensemble.state_dict_of(m) for m in range(K)]; states += [states[-1]]*(len(edges)-1-K)
    save_field(a.output/'field.pt', 'pair', edges, a.blend, states, mode='supervised', supervised_models=K, seed=a.seed, smoke=a.smoke)
    write(a.output/'progress.json', dict(complete=True, history=history, held_out=held, elapsed_seconds=time.time()-started)); print(json.dumps(held), flush=True)


if __name__ == '__main__': main()
