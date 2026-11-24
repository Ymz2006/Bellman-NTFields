"""3-D viewer of the p_d055_s46_h60 re-creation: the study's viewer (viewer.py + viewer_template.html) with this one placement
shown three times, the original runs, the retraining here on the original data and queries (chain_frozen_state.json), and the
re-creation from scratch here (chain_state.json), plus the three sets of ten picked configurations.

    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/repro/viewer_repro.py

Reads only finished registered runs (no training, no planning). The original placement's files are copied from report/viewer/.
Plans shown per query: the `picks_*` runs (two-ended planner, 2,400 steps per end, planning seed 0, paths saved); every verdict is
recomputed from the checker's samples and asserted equal to the run's flag, as in viewer.py. Writes repro/viewer/.
"""
import base64
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
import viewer as V  # noqa: E402

NAME = 'p_d055_s46_h60'
RES = lambda r: ROOT / f'results/registered_{r}'  # noqa: E731
C2C = 'compartment_to_compartment'


def flags_row(run, key, task, c2c):
    z = np.load(RES(run) / 'flags.npz')
    f = [z[f'{key}|{k}'] for k in (0, 1, 2)]
    return [int(sum(x.sum() for x in f)), 3 * len(f[0]), int(sum(x[task == c2c].sum() for x in f)), run[-8:]]


def placement(tag, label, state, prep, orig_row, out):
    """One variant of the placement: its row for data.json and its pl/<tag>.json/.txt."""
    sc, _ = V.arm12.load_scene(prep, V.DEV)
    Q = np.load(prep / 'queries.npz', allow_pickle=True)
    names = [str(t) for t in Q['task_names']]
    task, c2c = Q['task'], names.index(C2C)
    cmp = json.loads((HERE / 'compare.json').read_text())[label_key[tag]]
    seeds = [s for s in ('s87358', 's87359') if state.get(f'lift_{s}')]
    runs = dict(lift_two_ended=[flags_row(state[f'budget_lift_{s}'], 'lift|bidir|2400|50|5', task, c2c) for s in seeds if state.get(f'budget_lift_{s}')],
                ntrl_two_ended=[flags_row(state['budget_ntrl_s87358'], 'ntrl|bidir|2400|50|5', task, c2c)])
    runs['lift_forward'] = [[*cmp['lift']['forward 600'][:1], 900, cmp['lift']['forward 600'][1], state['lift_s87358'][-8:]]]
    if 'forward 600' in cmp.get('lift2', {}):
        runs['lift_forward'].append([cmp['lift2']['forward 600'][0], 900, cmp['lift2']['forward 600'][1], state['lift_s87359'][-8:]])
    runs['ntrl_forward'] = [[cmp['ntrl']['forward 600'][0], 900, cmp['ntrl']['forward 600'][1], state['ntrl_s87358'][-8:]]]
    if 'forward 600' in cmp.get('ntrl2', {}):
        runs['ntrl_forward'].append([cmp['ntrl2']['forward 600'][0], 900, cmp['ntrl2']['forward 600'][1], state['ntrl_s87359'][-8:]])
    row = {k: orig_row[k] for k in ('dx', 's', 'h', 'bases', 'cuboids', 'zones', 'reach', 'zone_runs')}
    line = Q['line_here' if 'line_here' in Q else 'line'].astype(int)
    row.update(name=tag, key=tag, label=label, trained=True, runs=runs, euclid=[cmp['euclid']['euclid'][0], 900, state['euclid'][-8:]],
               line=[int(line.sum()), len(line)], queries=len(line))
    if tag == 'frozen':
        row['depth_cm'] = orig_row['depth_cm']                                  # same queries as the original
    # test queries and their plans
    n = len(Q['start'])
    xs, xg = torch.tensor(Q['start'], device=V.DEV), torch.tensor(Q['goal'], device=V.DEV)
    rows, at = [V.rows_of(sc, xs), V.rows_of(sc, xg)], 2 * n
    F = []
    for s in seeds:
        if state.get(f'budget_lift_{s}'):
            z = np.load(RES(state[f'budget_lift_{s}']) / 'flags.npz')
            F += [z[f'lift|bidir|2400|50|5|{k}'] for k in (0, 1, 2)]
    meta = dict(name=tag, n=n, tasks=names, task=task.tolist(), line=line.tolist(),
                dist=[round(float(v), 3) for v in torch.linalg.vector_norm((xg - xs) * sc.q_range, dim=1).cpu()], plans={},
                solved_runs=np.stack(F).sum(0).astype(int).tolist(), runs=len(F), run=state['picks_lift'])
    for field in ('lift', 'ntrl'):
        run = RES(state[f'picks_{field}'])
        paths = np.load(run / f'paths_{field}_bidir_2400_50_5_seed0.npy', allow_pickle=True)
        ok = np.load(run / 'flags.npz')[f'{field}|bidir|2400|50|5|0']
        plan, wrong = [], 0
        for i, pth in enumerate(paths):
            r, info = V.pack(sc, pth, 0.08 if field == 'lift' else 0.12, 12, 160 if field == 'lift' else 110)
            wrong += int(info['ok'] != bool(ok[i]))
            plan.append([at, info['n'], int(info['ok']), info.get('fail_at', -1), V.KIND.index(info['fail_kind']) if 'fail_kind' in info else -1, info['length'], info['waypoints']])
            rows.append(r)
            at += info['n']
        assert wrong == 0, (tag, field, wrong, 'verdicts differ from the registered run')
        meta['plans'][field] = plan
        meta[f'solved_{field}'] = int(ok.sum())
    arr = np.concatenate(rows).astype('<i2')
    (out / 'pl' / f'{tag}.txt').write_bytes(base64.b64encode(arr.tobytes()))
    (out / 'pl' / f'{tag}.json').write_text(json.dumps(meta, separators=(',', ':')))
    row['viewer_run'] = state['picks_lift'][-8:]
    print(tag, n, 'queries;', {k: f'{meta["solved_" + k]}/{n}' for k in meta['plans']}, f'{arr.nbytes / 1e6:.2f} MB', flush=True)
    return row, sc


def demo_set(k, tag, label, run, sc, dx, out):
    """The ten picked configurations of a registered demo run, as in viewer.py."""
    J = json.loads((RES(run) / 'results.json').read_text())
    Z = np.load(RES(run) / 'demo.npz', allow_pickle=True)
    rows, at, picks = [], 0, []
    for pk in J['picked']:
        ends = torch.tensor(np.array([pk['start_rad'], pk['goal_rad']]), device=V.DEV, dtype=torch.float32) / sc.q_range
        rows.append(V.rows_of(sc, ends))
        deep = float(sc.ee(ends)[..., 0].max()) - dx
        item = dict(query=pk['query'], line=bool(pk['straight_line_feasible']), task=pk['task'], deep_cm=round(100 * deep, 1), start=at, goal=at + 1,
                    start_zones=pk['start_zones'], goal_zones=pk['goal_zones'], dist=round(pk['joint_distance_rad'], 3),
                    start_rad=[round(v, 4) for v in pk['start_rad']], goal_rad=[round(v, 4) for v in pk['goal_rad']], plans={})
        at += 2
        for field in J['fields']:
            for seed in (0, 1, 2):
                r, info = V.pack(sc, Z[f'q{pk["query"]}_{field}_seed{seed}'], 0.045, 24, 360)
                assert info['ok'] == bool(pk['plans'][field]['solved_seeds'][seed]), (run, pk['query'], field, seed, 'verdict differs from the demo run')
                item['plans'].setdefault(field, []).append(dict(at=at, **info))
                rows.append(r)
                at += info['n']
        picks.append(item)
    arr = np.concatenate(rows).astype('<i2')
    (out / f'demo_{k}.txt').write_bytes(base64.b64encode(arr.tobytes()))
    print('set', k, tag, len(picks), 'configurations', f'{arr.nbytes / 1e6:.2f} MB', flush=True)
    return dict(placement=tag, label=label, file=f'demo_{k}.txt', fields=list(J['fields']), picks=picks,
                runs={f: v.split('registered_')[1].split(':')[0][-8:] for f, v in J['fields'].items()},
                candidates=J['hard']['n'], solved_all={f: v['all_seeds'] for f, v in J['hard']['solved'].items()})


label_key = dict(frozen='retrained here, original data + queries', scratch='re-created from scratch here (new suite)')
LABEL = dict(orig='Original (2× RTX 4090)', frozen='Retrained here on the original data', scratch='Re-created here from scratch (new queries)')


def main():
    out = HERE / 'viewer'
    (out / 'pl').mkdir(parents=True, exist_ok=True)
    src = STUDY / 'report/viewer'
    D = json.loads((src / 'data.json').read_text())
    o = next(p for p in D['placements'] if p['name'] == NAME)
    orig = dict(o, name='orig', key='orig', label=LABEL['orig'])
    for ext in ('json', 'txt'):
        shutil.copyfile(src / 'pl' / f'{NAME}.{ext}', out / 'pl' / f'orig.{ext}')
    m = json.loads((out / 'pl/orig.json').read_text())
    m['name'] = 'orig'
    (out / 'pl/orig.json').write_text(json.dumps(m, separators=(',', ':')))

    frozen_state = json.loads((HERE / 'chain_frozen_state.json').read_text())
    chain_state = json.loads((HERE / 'chain_state.json').read_text())
    frozen, sc_f = placement('frozen', LABEL['frozen'], frozen_state, HERE / 'prep_frozen_local', o, out)
    scratch, sc_s = placement('scratch', LABEL['scratch'], chain_state, RES(chain_state['prep']), o, out)

    s0 = next(s for s in D['sets'] if s['placement'] == NAME)
    shutil.copyfile(src / s0['file'], out / 'demo_0.txt')
    sets = [dict(s0, placement='orig', label=LABEL['orig'], file='demo_0.txt')]
    # the depth reference of viewer.py (cupboard front of models/index.json), recovered from the original set and checked on the frozen scene
    pk = s0['picks'][0]
    ends = torch.tensor(np.array([pk['start_rad'], pk['goal_rad']]), device=V.DEV, dtype=torch.float32) / sc_f.q_range
    dx = float(sc_f.ee(ends)[..., 0].max()) - pk['deep_cm'] / 100
    assert all(abs(float(sc_f.ee(torch.tensor(np.array([q['start_rad'], q['goal_rad']]), device=V.DEV, dtype=torch.float32) / sc_f.q_range)[..., 0].max()) - dx - q['deep_cm'] / 100) < 2e-3
               for q in s0['picks']), 'depth reference does not reproduce the original set'
    sets.append(demo_set(1, 'frozen', LABEL['frozen'], frozen_state['demo'], sc_f, dx, out))
    sets.append(demo_set(2, 'scratch', LABEL['scratch'], chain_state['demo'], sc_s, dx, out))

    data = dict(arm=D['arm'], best='orig', placements=[orig, frozen, scratch], sets=sets)
    (out / 'data.json').write_text(json.dumps(data, separators=(',', ':')))

    tpl = (STUDY / 'viewer_template.html').read_text()
    intro = ('The placement 0.55 m · ±0.46 m · 0.60 m high, three times: the original runs on the rented machine, the same pipeline '
             'retrained on this workstation (RTX 3090) on the original training data and test queries, and the whole pipeline re-created '
             'here from scratch, which builds new test queries (88 of 300 are the same as the original ones, because the GPU sampler is not '
             'bit-reproducible across cards). Each set of ten configurations is picked by the same rule from its own runs. The arms are drawn '
             'as the collision spheres the checker tests. Drag to turn, scroll or pinch to zoom, right-drag to move. '
             'The numbers are in jeffrey/arm_placement/README.md (last section) and repro/compare.md.')
    swaps = [
        ('<title>2026-11-23 Arm Placement Viewer</title>', '<title>2026-11-24 Arm Placement Re-creation</title>'),
        ('jeffrey/arm_placement · two UR5e arms, 12 joints · viewer', 'jeffrey/arm_placement/repro · two UR5e arms, 12 joints · re-creation of one placement'),
        ('<h1>Arm placement viewer</h1>', '<h1>Re-creating 0.55 m · ±0.46 m · 0.60 m high</h1>'),
        ('const plName = p => `', 'const plName = p => p.label || `'),
        ("(s.placement === DATA.best ? 'Best placement: ' : '') + plName(byName[s.placement])", "s.label || plName(byName[s.placement])"),
        ("html: plName(q) + (q.name === DATA.best ? '<span class=\"best\">best</span>' : '')", "html: plName(q)"),
        (">All placements</button>", ">Test queries</button>"),
        ("'Distance from the bases to the cupboard front · half the spacing between the bases · base height. The cupboard stays where it is; the bases move.'",
         "'The same placement in every row: the original runs and the two re-creations. Pick one to see its test queries and plans.'"),
        ("pane.append(el('h2', {}, `Placements (${PL.length})`), list);", "pane.append(el('h2', {}, 'Runs'), list);"),
        ("  if (t === 'placements' && !st.pins) $('#t-pins').click();\n", ''),
    ]
    for a, b in swaps:
        assert tpl.count(a) == 1, ('template text not found once', a)
        tpl = tpl.replace(a, b)
    tpl = tpl.replace(tpl[tpl.index('<p>Every base placement'):tpl.index('</p>', tpl.index('<p>Every base placement')) + 4], f'<p>{intro}</p>')
    (out / 'index.html').write_text(tpl)
    files = sorted(f for f in out.rglob('*') if f.is_file())
    print(f'{len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
