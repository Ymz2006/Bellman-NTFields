"""Interactive viewer of the grid-synced octave rows (jeffrey/grid_sync/features.py, mode `octave`), the grid-anchored rows (gridrows.py) and ntrl-demo's own draw, on the 13x13 and
22x22 mazes of jeffrey/maze_scale: report/octave_rows.html, built from octaveview_template.html. CPU only.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/octaveview.py
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import features as FT  # noqa: E402
sys.path.insert(0, str(HERE))
import gridrows as GR  # noqa: E402

PREPARED = {13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
SEEDS = [87358, 1, 2]


def main():
    data = dict(seeds=SEEDS, mazes={})
    for n, run in PREPARED.items():
        info = json.loads((ROOT / f'results/registered_{run}/maze.json').read_text())
        E = info['E_cells']
        Ls = FT.octaves(n)
        m = dict(n=n, E=E, rho=info['rho'], t_wall=info['t_wall'], Ls=Ls, V=[[int(bool(v)) for v in r] for r in info['V']], H=[[int(bool(v)) for v in r] for r in info['H']], seeds={})
        for s in SEEDS:
            draw = FT.draw(s)
            raw = draw.double().numpy()
            sets = dict(ntrl=dict(rows=np.c_[raw, np.zeros(128)], L=None))
            for mode in ['octave'] + GR.MODES:
                B, delta, L = GR.rows(raw, E, n, mode)
                if mode == 'octave':
                    assert np.allclose(B, FT.transform(draw, E, 'octave').double().numpy(), atol=1e-5), 'gridrows.rows differs from features.transform'
                sets[mode] = dict(rows=np.c_[B, np.mod(delta, 2 * np.pi)], L=L.tolist())
            m['seeds'][str(s)] = {k: dict(rows=np.round(v['rows'], 5).tolist(), L=v['L']) for k, v in sets.items()}
        data['mazes'][str(n)] = m
    page = (HERE / 'octaveview_template.html').read_text().replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
    (HERE / 'report/octave_rows.html').write_text(page)
    print('ok', len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
