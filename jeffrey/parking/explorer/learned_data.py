"""Learned-field data for the Parking Speed Explorer (explorer/speed_explorer.html).

For every registered parking run and training seed, evaluates the trained ntrl-demo field from the bay-A parked pose
qA = (0.29, y_A, 0) (the source used in training and in the pipeline's profile):
  - along the line x = 0.29, theta = 0, y in [-0.47, 0.47] step 0.001:
      T     = T(q, qA)                   learned travel time
      S     = 1 / |grad_q T(q, qA)|       learned speed (q = (x, y, theta / 2 pi), as in training and pipeline.py's speed fit)
      G     = -dT/dy                      learned slope along the line, toward bay B
  - bayInv = 1 / S at (0.10, -y_A, 0)     (car halfway into bay B, the explorer's "along the bay" slope)
  - a 121 x 121 plane (theta = 0) of learned speed, 0..1 as bytes, for the lot heat map.
Writes explorer/learned.js (window.LEARNED = {...}). Inference only; no training, no registry writes.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python explorer/learned_data.py
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import plan as P  # noqa: E402
import scene as SC  # noqa: E402
from report import PS, find_runs  # noqa: E402

LINE_Y = np.round(np.arange(-470, 471) / 1000.0, 3)
NPLANE = 121


def sig(a, n=5):
    return [float(f'{v:.{n}g}') for v in np.asarray(a, dtype=np.float64)]


def main():
    dev = 'cuda:0'
    out = {'line_y': {'start': -0.47, 'step': 0.001, 'n': len(LINE_Y)}, 'plane_n': NPLANE, 'fields': []}
    gx = torch.linspace(-0.5, 0.5, NPLANE, device=dev)
    # plane rows run top (y = 0.5) to bottom, columns left to right, like the explorer's heat image
    yy, xx = torch.meshgrid(-gx, gx, indexing='ij')
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(NPLANE * NPLANE, device=dev)], 1)
    for rid, run_dir, res in find_runs([]):
        car_w, wall, bay_w = float(res.get('car_width', SC.CAR_W)), float(res['wall']), float(res['bay_width'])
        fixed = bool(res.get('bay_width_fixed'))
        family = 'widening' if fixed else ('divider sweep' if abs(car_w - SC.CAR_W) < 1e-9 else 'car-width sweep')
        ya = SC.bay_y(wall, 'A', car_w, bay_w)
        qa = torch.tensor([[SC.PARK_X, ya, 0.0]], device=dev)
        line = torch.tensor(np.stack([np.full_like(LINE_Y, SC.PARK_X), LINE_Y, np.zeros_like(LINE_Y)], 1), device=dev, dtype=torch.float32)
        for seed, sr in res['seeds_results'].items():
            ck = run_dir / f'seed{seed}' / 'training' / 'checkpoint.pt'
            net = P.load_field(str(ck), res['variant'], dev)
            t, g = P.field_grad(net, torch.cat([line, qa.expand(len(line), 3)], 1))
            s = 1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)
            _, gb = P.field_grad(net, torch.tensor([[0.10, -ya, 0.0, SC.PARK_X, ya, 0.0]], device=dev))
            _, gp = P.field_grad(net, torch.cat([plane, qa.expand(len(plane), 3)], 1))
            sp = (1 / torch.linalg.vector_norm(gp, dim=1).clamp(min=1e-6)).clamp(0, 1)
            pl = sr['planning']
            succ = {st: sum(pl[f'{st}/spot_to_spot/{p}']['success'] for p in PS) for st in ('default', 'fine') if f'{st}/spot_to_spot/0' in pl}
            out['fields'].append(dict(
                run=rid, seed=seed, variant=res['variant'], family=family, carW=round(car_w, 6), div=wall, bay=round(bay_w, 6),
                margin=res['margin'], success=succ, t_over_tstar=sr['field']['t_over_tstar_median'],
                T=sig(t.cpu().numpy()), S=sig(s.cpu().numpy(), 4), G=sig(-g[:, 1].cpu().numpy(), 4),
                bayInv=float(1 / torch.linalg.vector_norm(gb[0]).clamp(min=1e-6)),
                plane=base64.b64encode(np.round(sp.cpu().numpy() * 255).astype(np.uint8).tobytes()).decode()))
            del net
            print(f"{rid} seed {seed} {res['variant']} car {car_w:.4f} div {wall} bay {bay_w:.4f}: T(bay B) "
                  f"{float(np.interp(-ya, LINE_Y, t.cpu().numpy())):.3f}, success {succ}", flush=True)
    path = HERE / 'learned.js'
    path.write_text('window.LEARNED = ' + json.dumps(out, separators=(',', ':')) + ';\n')
    print(f'{len(out["fields"])} fields -> {path} ({path.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
