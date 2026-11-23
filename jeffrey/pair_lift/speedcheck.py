"""How accurate is the speed a trained field predicts, 1 / |grad_q T(q, g)|, and the step cost integrated from it? Read-only.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/pair_lift/speedcheck.py <run id> [<run id> ...]        -> report/speed.json

1. Point speed: S_true(q) |grad_q T(q, g)| (1 = exact) at configurations uniform in the C-space box, partner g a random free configuration,
   by the clearance of q: deep inside obstacles, inside, the skin at the floor speed, the speed band, open space.
2. Step cost: straight steps of 0.33 - 0.66 cells from uniform configurations, cost from the field's speed table (pairnet.SpeedTable, the
   one the backup reads with --bell-tau field) against the true step time (the true speed along the step; inside an obstacle that is
   the floor speed), for steps that stay out of obstacles and steps that cross one.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import forensics as FX  # noqa: E402
PT, PN = FX.PT, FX.PN
OFF, MAR = PT.OFFSET_CELLS, 0.175
BINS = [('inside, deeper than 0.1 cells', -9, -0.1), ('inside, 0.0175 - 0.1 cells deep', -0.1, -OFF), ('skin at the floor speed', -OFF, OFF), ('band, lower half', OFF, 0.5 * MAR),
        ('band, upper half', 0.5 * MAR, MAR), ('open space', MAR, 9)]


def quant(x):
    return [round(float(v), 3) for v in torch.quantile(x, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device))]


def main():
    out_f = HERE / 'report/speed.json'
    data = json.load(open(out_f)) if out_f.exists() else {}
    for spec in sys.argv[1:]:
        net, scene, prepared, res, row = FX.load(spec)
        dev = FX.DEV
        gen = torch.Generator(dev).manual_seed(11)
        q = scene.uniform(120_000, gen)
        d = scene.clearance(q)
        g = scene.free(len(q), gen, 0.1)
        slow = torch.empty(len(q), device=dev)
        for i in range(0, len(q), 40_000):
            with torch.enable_grad():
                tau, _, xp = net.out(torch.cat([q[i:i + 40_000], g[i:i + 40_000]], 1))
                gr, = torch.autograd.grad(tau.sum(), xp)
            slow[i:i + 40_000] = torch.linalg.vector_norm(gr[:, :3], dim=1)
        fit = scene.train_speed(d) * slow
        rec = dict(scene=res.get('scene_name'), label=res['args'].get('label'), point={}, step={})
        print(spec, rec['scene'])
        for name, lo, hi in BINS:
            m = (d > lo) & (d <= hi)
            if int(m.sum()) < 50:
                continue
            rec['point'][name] = dict(share=float(m.float().mean()), fit=quant(fit[m]), within10=float(((fit[m] - 1).abs() < 0.1).float().mean()), within25=float(((fit[m] - 1).abs() < 0.25).float().mean()))
            print(f"  S |grad T|  {name:32s} share {rec['point'][name]['share']:.3f}  10/25/50/75/90 % {rec['point'][name]['fit']}  within 10 % {rec['point'][name]['within10']:.2f}  within 25 % {rec['point'][name]['within25']:.2f}")
        table = PN.SpeedTable(net, scene, 32, 49, goals=4, seed=0)
        clear = PN.ClearTable(scene, 32, 49)
        N, K, r = 20_000, 8, 0.03
        a = scene.uniform(N, gen)
        a = a[scene.clearance(a) > -OFF]
        N = len(a)
        dirs = torch.randn(N, K, 3, generator=gen, device=dev)
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True)
        b = torch.minimum(torch.maximum(a[:, None] + dirs * r * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev)), scene.lo), scene.hi)
        A = a[:, None].expand(-1, K, -1)
        with torch.no_grad():
            tt, free = PT.step_time(scene, A, b, 137, clear=clear)
            tl, smax = table.step_time(A, b, 65, with_max=True)
            p = A[..., None, :] + torch.linspace(0, 1, 65, device=dev)[:, None] * (b - A)[..., None, :]
            dmin = clear(p.reshape(-1, 3)).reshape(N, K, 65).amin(-1)
        E = scene.E
        for name, m in (('free, open space', free & (dmin > MAR)), ('free, in the speed band', free & (dmin <= MAR) & (dmin > 0.05)), ('free, within 0.05 cells of an obstacle', free & (dmin <= 0.05)),
                        ('through an obstacle', ~free)):
            if int(m.sum()) < 50:
                continue
            ratio = (tl / tt)[m]
            rec['step'][name] = dict(share=float(m.float().mean()), ratio=quant(ratio), within10=float(((ratio - 1).abs() < 0.1).float().mean()), true_cells=float(tt[m].median() * E), table_cells=float(tl[m].median() * E))
            print(f"  step cost   {name:38s} share {rec['step'][name]['share']:.3f}  table / true 10/25/50/75/90 % {rec['step'][name]['ratio']}  within 10 % {rec['step'][name]['within10']:.2f}  "
                  f"median true {rec['step'][name]['true_cells']:.2f} table {rec['step'][name]['table_cells']:.2f} cells")
        # a collision check from the predicted speed: a step counts as blocked when the table's speed falls below thr at a sample
        rec['check'] = {}
        for thr in (0.02, 0.03, 0.04, 0.06, 0.08, 0.12):
            flag = smax > 1.0 / thr
            rec['check'][str(thr)] = dict(blocked_caught=float(flag[~free].float().mean()), free_flagged=float(flag[free].float().mean()),
                                          free_flagged_open=float(flag[free & (dmin > 0.05)].float().mean()))
            print(f"  speed check below {thr:.2f}: steps through an obstacle caught {rec['check'][str(thr)]['blocked_caught']:.3f}, free steps flagged {rec['check'][str(thr)]['free_flagged']:.3f} "
                  f"(more than 0.05 cells from an obstacle: {rec['check'][str(thr)]['free_flagged_open']:.3f})")
        data[row['run']] = rec
        del table, clear, net
        torch.cuda.empty_cache()
    out_f.write_text(json.dumps(data, indent=1) + '\n')


if __name__ == '__main__':
    main()
