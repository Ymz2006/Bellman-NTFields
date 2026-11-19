"""Every finished map of one scene, measured again with the final measure.py, so that a table never mixes versions of a measure. Read-only on the runs.

    remeasure.py --scene <spec> --res 32 --runs label=<run id>,... --out <dir>
"""
import argparse
import json
from pathlib import Path
import torch

torch.set_float32_matmul_precision('high')
import carto as C
import maps as M
import measure as ME
import nf as N
import scene as SC
import speedfit as SF


def load(rid, dev):
    """psi of a run of this study, whichever kind it was."""
    d = torch.load(SC.ROOT / 'results' / f'registered_{rid}' / 'psi.pt', map_location=dev, weights_only=False)
    if 'psi' in d:
        return C.TableMap(d['psi'].to(dev), d['phi'].to(dev)).to(dev)
    m = M.build(d['cfg']).to(dev)
    if d['cfg']['kind'] != 'identity':
        m.load_state_dict(d['state'])
    res = json.load(open(SC.ROOT / 'results' / f'registered_{rid}' / 'results.json'))
    return N.Inverted(m) if res.get('stage') == 'equal-runs-flow' else m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--res', type=int, default=32)
    p.add_argument('--headings', type=int, default=9)
    p.add_argument('--runs', required=True)
    p.add_argument('--wall-steps', type=int, default=3000, help='budget of the near-wall score (0: skip it)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)
    dev = 'cuda:0'
    scene = SC.build(a.scene, dev, a.headings, a.res)
    samples = max(2048, 32 * int(round(scene.E)))
    ident = M.Identity()
    res = dict(scene=a.scene, rows={}, identity=ME.all_measures(ident, scene, samples))
    if a.wall_steps:
        res['identity_wall'] = SF.near_wall(ident, ident, scene, 4.0 if scene.dim < 6 else 1.0, a.wall_steps)
    for pair in a.runs.split(','):
        label, rid = pair.split('=')
        m = load(rid, dev)
        res['rows'][label] = dict(run=rid, warp=ME.all_measures(m, scene, samples))
        if a.wall_steps:
            res['rows'][label]['wall'] = SF.near_wall(m, m.inverse, scene, 4.0 if scene.dim < 6 else 1.0, a.wall_steps)
        f.write(f'{label} done\n')
        (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    res['complete'] = True
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
