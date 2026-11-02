"""3D path viewer for the segment-DP cabinet fields, queries grouped by start-goal distance bin.

Two tabs on the cabinet scene:
  Frozen queries  the certified suites (0.06-0.57): witness path, every field's release-MPPI path (planning seed 0)
                  and native RK4 path. Bins of --bin in |q - g| (normalized joint units), up to --per-bin queries each.
  Long queries    the 400 uncertified long queries (0.80-1.48) of long.py run --long: RK4 paths of every field in it,
                  and MPPI seed 0 for the first 100 (run --long-mppi). No witness (not certified).
Queries in a bin are picked so B5 and DP+TD+BELL disagree in both directions, both solve and both fail (frozen: MPPI
seed 0; long: RK4), then filled. The totals panel shows each field's success over the whole set and in the selected
query's bin (the same counts as success_bins.py). Verdicts are the scorer's (.score.json); red frames come from the
release sphere model at the scorer's margins (cabinet < 5 mm, self < 2 mm), as in experiments/pntfields_cabinet.

Reuses experiments/pntfields_cabinet/visualizer (export helpers, template.html layout, kinematics.js) unchanged.
Read-only on results/; writes viewer.html (and viewer.json) next to this script, or to --output.

  ~/.venvs/bellman-ntfield/bin/python jeffrey/segment_dp/visualizer/build_viewer.py
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent)]
from experiments.face_speed import robot as R
from experiments.face_speed.clearance import PieceClearance
from experiments.pntfields_cabinet.visualizer.export_cabinet import urdf_joints, robot_meshes, judge, thin, segment_min, OBS_MARGIN, SELF_MARGIN
from success_bins import STRATA, evaluation_dir, frozen_gaps, frozen_successes, binned

OLD = ROOT/'experiments/pntfields_cabinet/visualizer'
FIELDS = [('B5', '20261027T044847Z-c42c6e8f', 'bellman', '--b5'), ('ES', '20261027T175328Z-b7ab5a49', 'bellman', '--es'),
          ('DP', '20261027T184605Z-f57e1332', 'bellman', '--dp'), ('DP+TD+BELL', '20261030T051716Z-502d7e3a', 'bellman', '--dptb'),
          ('ntrl-demo', '20261026T041234Z-664b5d50', 'metric', '--ntrl')]
PRETTY = {'free_to_free': 'free → free', 'free_to_compartment': 'free → compartment', 'compartment_to_free': 'compartment → free',
          'compartment_to_compartment': 'compartment → compartment', 'same_compartment': 'same compartment', 'anywhere': 'anywhere'}
TOKENS = {  # light, dark: the dataviz reference categorical order
    '--b5': ('#2a78d6', '#3987e5'), '--es': ('#eb6834', '#d95926'), '--dp': ('#1baf7a', '#199e70'),
    '--dptb': ('#eda100', '#c98500'), '--ntrl': ('#e87ba4', '#d55181'), '--witness': ('#78828a', '#87919a')}


def path_entry(pc, path, verdict):
    path = np.asarray(path, np.float64); env, slf = judge(pc, path)
    keep = thin(len(path), [int(np.argmin(env))])
    collide = segment_min(env, keep) < OBS_MARGIN; self_collide = segment_min(slf, keep) < SELF_MARGIN
    return dict(path=path[keep].round(5).tolist(), clearance=np.minimum(segment_min(env, keep), 1).round(4).tolist(),
                collide=collide.astype(int).tolist(), self_collide=self_collide.astype(int).tolist(), success=bool(verdict['success']),
                reasons=verdict.get('reasons', []), waypoints=int(verdict.get('n_waypoints', len(path))), min_clearance=float(env.min()))


def pick(ok_a, ok_b, members, per_bin):
    """Queries of one bin: a solves / b fails, a fails / b solves, both solve, both fail, then the rest in order."""
    out = []
    for want in [(True, False), (False, True), (True, True), (False, False)]:
        out += [int(i) for i in members if ok_a[i] == want[0] and ok_b[i] == want[1]][:1]
    out += [int(i) for i in members if i not in out]
    return sorted(out[:per_bin])


def bin_label(edges, b, n):
    return f'{edges[b]:.2f}–{edges[b+1]:.2f} · segment {b+1} · {n} queries'


def frozen_tab(pc, edges, per_bin, cuboids, agree):
    tests = ROOT/'experiments/cabinet/tests'; manifest = json.loads((tests/'suite_manifest.json').read_text())
    files = {s['file'].removeprefix('cabinet_').removesuffix('_100.npz'): s['file'] for s in manifest['suites']}
    suites = [np.load(tests/'suites'/files[s], allow_pickle=True) for s in STRATA]
    gaps = frozen_gaps(tests); bins = np.clip(np.digitize(gaps, edges)-1, 0, len(edges)-2)
    paths, scores = {}, {}
    for label, run, prefix, _ in FIELDS:
        e = evaluation_dir(run)
        for planner, tag in (('mppi', '_seed0'), ('rk4', '')):
            traj, sc = [], []
            for s in STRATA:
                name = f'{prefix}_{planner}_{s}{tag}'
                with np.load(e/f'{name}.npz', allow_pickle=True) as z:
                    t = list(z['trajectories']); ids = [int(x) for x in z['query_ids']] if 'query_ids' in z.files else list(range(len(t)))
                assert ids == list(range(100)), (name, ids[:5])
                traj += t; pq = json.loads((e/f'{name}.score.json').read_text())[name]['per_query']; sc += [pq[str(i)] for i in range(100)]
            paths[f'{planner}_{label}'] = traj; scores[f'{planner}_{label}'] = sc
    ok = {k: np.array([v['success'] for v in s]) for k, s in scores.items()}
    queries, per_bin_counts = [], {}
    for b in range(len(edges)-1):
        members = np.flatnonzero(bins == b)
        if not len(members):
            continue
        group = bin_label(edges, b, len(members))
        for i in pick(ok['mppi_DP+TD+BELL'], ok['mppi_B5'], members, per_bin):
            s, q = divmod(i, 100); z = suites[s]
            plans = {'witness': path_entry(pc, z['witness_trajectories'][q], dict(success=True, n_waypoints=len(z['witness_trajectories'][q])))}
            for key in paths:
                plans[key] = path_entry(pc, paths[key][i], scores[key][i])
                agree.append(any('collision' in r for r in plans[key]['reasons']) == bool(np.any(plans[key]['collide']) or np.any(plans[key]['self_collide'])))
            queries.append(dict(group=group, bin=str(b), suite=STRATA[s], name=f'{PRETTY[STRATA[s]]} #{q}',
                                detail=f'|q−g| {gaps[i]:.3f} · {z["start_region"][q]} → {z["goal_region"][q]}', plans=plans))
        print('frozen', group, flush=True)
    totals = []
    for planner, title, per, trials in (('mppi', 'Release MPPI, 600 queries × 3 planning seeds', 3, 1800), ('rk4', 'Native RK4, 600 queries', 1, 600)):
        rows = []
        for label, run, prefix, _ in FIELDS:
            mppi, rk4 = frozen_successes(run, prefix); wins = mppi if planner == 'mppi' else rk4
            k, n = binned(gaps, wins, per, edges)
            rows.append(dict(key=f'{planner}_{label}', label=f'{label} · {planner.upper()}', short=label, success=int(wins.sum()), n=trials,
                             per_bin={str(b): f'{k[b]}/{n[b]}' for b in range(len(k)) if n[b]}))
        totals.append(dict(label=title, rows=rows))
    order = ['witness']+[f'mppi_{f[0]}' for f in FIELDS]+[f'rk4_{f[0]}' for f in FIELDS]
    in_bin = [j for j, q in enumerate(queries) if q['bin'] == '3']   # the busiest bin, 0.24-0.32
    start = next((j for j in in_bin if queries[j]['plans']['mppi_DP+TD+BELL']['success'] and not queries[j]['plans']['mppi_ntrl-demo']['success']),
                 next((j for j in in_bin if queries[j]['plans']['mppi_DP+TD+BELL']['success']), in_bin[0] if in_bin else 0))
    return dict(id='frozen', tabKey='Frozen', tab='frozen queries · 0.06–0.57', title='Cabinet, frozen certified queries', robot='ur5e',
                obstacle=dict(cuboids=cuboids), sourceOrder=order, keySources=[f'mppi_{f[0]}' for f in FIELDS],
                defaultShown=['witness', 'mppi_B5', 'mppi_DP+TD+BELL', 'mppi_ntrl-demo'], defaultActive='mppi_DP+TD+BELL', startQuery=start,
                queries=queries, totals=totals,
                totalsNote='Bars: the whole frozen set. Under each bar: the same field in the selected query\'s distance bin (MPPI over 3 planning seeds). Paths shown are MPPI planning seed 0.',
                queriesNote=f'Grouped by |q − g| in normalized joint units (bins of {edges[1]-edges[0]:.2f}, the segment width). Up to {per_bin} per bin, picked so B5 and DP + TD + bell (MPPI seed 0) disagree both ways, both solve and both fail. Marks: MPPI seed 0 of B5, ES, DP, DP + TD + bell, ntrl-demo.',
                contactRule='release sphere model: cabinet < 5 mm or self < 2 mm, as the scorer', selfScored=True, contactMargin=OBS_MARGIN,
                clearanceCap=.1, clearanceTicks=[.025, .05, .1], clearanceLabel='release-sphere clearance to the cabinet',
                view=dict(eye=[-0.25, -1.5, 1.05], target=[0.55, 0.2, 0.25]))


def long_tab(pc, edges, per_bin, cuboids, long_run, mppi_run, agree):
    L = ROOT/'results'/f'registered_{long_run}'/'long'; M = ROOT/'results'/f'registered_{mppi_run}'/'long'
    lq = np.load(L/'long_queries.npz'); gaps = lq['gap']; bins = np.clip(np.digitize(gaps, edges)-1, 0, len(edges)-2)
    fields = [f for f in FIELDS if (L/f'rk4_{f[0]}.score.json').exists()]
    paths, scores, ok = {}, {}, {}
    for label, *_ in fields:
        with np.load(L/f'rk4_{label}.npz', allow_pickle=True) as z:
            assert [int(x) for x in z['query_ids']] == list(range(len(gaps))); paths[f'rk4_{label}'] = list(z['trajectories'])
        pq = json.loads((L/f'rk4_{label}.score.json').read_text())[label]['per_query']; scores[f'rk4_{label}'] = [pq[str(i)] for i in range(len(gaps))]
        mp = M/f'mppi_{label}_seed0.npz'
        if mp.exists():
            with np.load(mp, allow_pickle=True) as z:
                t = list(z['trajectories']); ids = [int(x) for x in z['query_ids']] if 'query_ids' in z.files else list(range(len(t)))
            pq = next(iter(json.loads((M/f'mppi_{label}_seed0.score.json').read_text()).values()))['per_query']
            paths[f'mppi_{label}'] = dict(zip(ids, t)); scores[f'mppi_{label}'] = {i: pq[str(i)] for i in ids}
    ok = {k: np.array([v['success'] for v in scores[k]]) for k in scores if k.startswith('rk4')}
    queries = []
    for b in range(len(edges)-1):
        members = np.flatnonzero(bins == b)
        if not len(members):
            continue
        members = sorted(members, key=lambda i: (i >= 100, i))   # queries with MPPI paths first
        group = bin_label(edges, b, len(members))
        for i in pick(ok['rk4_DP+TD+BELL'], ok['rk4_B5'], members, per_bin):
            plans = {}
            for key in paths:
                if key.startswith('mppi') and i not in paths[key]:
                    continue
                plans[key] = path_entry(pc, paths[key][i], scores[key][i])
                agree.append(any('collision' in r for r in plans[key]['reasons']) == bool(np.any(plans[key]['collide']) or np.any(plans[key]['self_collide'])))
            queries.append(dict(group=group, bin=str(b), name=f'long #{i}', detail=f'|q−g| {gaps[i]:.3f}' + (' · has MPPI' if i < 100 else ''), plans=plans))
        print('long', group, flush=True)
    rows_rk4, rows_mppi = [], []
    for label, *_ in fields:
        k, n = binned(gaps, ok[f'rk4_{label}'], 1, edges)
        rows_rk4.append(dict(key=f'rk4_{label}', label=f'{label} · RK4', short=label, success=int(ok[f'rk4_{label}'].sum()), n=len(gaps),
                             per_bin={str(b): f'{k[b]}/{n[b]}' for b in range(len(k)) if n[b]}))
        if f'mppi_{label}' in scores:
            ids = sorted(scores[f'mppi_{label}']); w = np.array([scores[f'mppi_{label}'][i]['success'] for i in ids])
            k, n = binned(gaps[ids], w, 1, edges)
            rows_mppi.append(dict(key=f'mppi_{label}', label=f'{label} · MPPI seed 0', short=label, success=int(w.sum()), n=len(ids),
                                  per_bin={str(b): f'{k[b]}/{n[b]}' for b in range(len(k)) if n[b]}))
    order = [f'rk4_{f[0]}' for f in fields]+[f'mppi_{f[0]}' for f in fields]
    start = next((j for j, q in enumerate(queries) if q['plans']['rk4_DP+TD+BELL']['success'] and 'mppi_B5' in q['plans']), 0)
    return dict(id='long', tabKey='Long', tab='long queries · 0.80–1.48', title='Cabinet, long uncertified queries', robot='ur5e',
                obstacle=dict(cuboids=cuboids), sourceOrder=order, keySources=[f'rk4_{f[0]}' for f in fields],
                defaultShown=['rk4_B5', 'rk4_DP+TD+BELL', 'rk4_ES'], defaultActive='rk4_DP+TD+BELL', startQuery=start, queries=queries,
                totals=[dict(label=f'Native RK4, {len(gaps)} long queries', rows=rows_rk4), dict(label='Release MPPI seed 0, the first 100 long queries', rows=rows_mppi)],
                totalsNote='Bars: the whole long set. Under each bar: the same field in the selected query\'s distance bin. ntrl-demo was not run on the long queries.',
                queriesNote=f'Not certified: no witness path, and some may have no collision-free path at all. Up to {per_bin} per bin, picked so B5 and DP + TD + bell (RK4) disagree both ways, both solve and both fail; queries with an MPPI path first. Marks: RK4 of each field.',
                contactRule='release sphere model: cabinet < 5 mm or self < 2 mm, as the scorer', selfScored=True, contactMargin=OBS_MARGIN,
                clearanceCap=.1, clearanceTicks=[.025, .05, .1], clearanceLabel='release-sphere clearance to the cabinet',
                view=dict(eye=[-0.25, -1.5, 1.05], target=[0.55, 0.2, 0.25]))


def template():
    page = (OLD/'template.html').read_text()
    light = ' '.join(f'{k}: {v[0]};' for k, v in TOKENS.items()); dark = ' '.join(f'{k}: {v[1]};' for k, v in TOKENS.items())
    swaps = [('<title>P-NTFields Path Viewer</title>', '<title>Segment DP Path Viewer</title>'),
             ('<h1>P-NTFields path viewer</h1>', '<h1>Segment DP path viewer</h1>'),
             ('--pnt: #2a78d6; --ntrl: #eb6834; --released: #1baf7a; --witness: #78828a;', light),
             ('--accent: #3987e5; --pnt: #3987e5; --ntrl: #d95926; --released: #199e70; --witness: #87919a;', '--accent: #3987e5; '+dark),
             # per-bin counts instead of per-suite counts under each total bar
             ("const suite = Q && r.per_suite && Q.suite in r.per_suite ? ` · suite ${r.per_suite[Q.suite]}/${r.n/6}` : '';",
              "const suite = Q && r.per_bin && Q.bin in r.per_bin ? ` · this bin ${r.per_bin[Q.bin]}` : '';"),
             ("${r.per_suite && Q ? ` · this suite ${r.per_suite[Q.suite]}/${r.n/6}` : ''}",
              "${r.per_bin && Q && Q.bin in r.per_bin ? ` · this bin ${r.per_bin[Q.bin]}` : ''}"),
             ("'pnt-viewer-scene'", "'sdp-viewer-scene'")]
    for old, new in swaps:
        n = page.count(old); assert n >= 1, old
        page = page.replace(old, new)
    assert page.count('/*DATA*/') == 1 and page.count('/*KINEMATICS_JS*/') == 1
    return page


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bin', type=float, default=0.08); p.add_argument('--per-bin', type=int, default=4)
    p.add_argument('--long', default='20261030T065351Z-87bf2327'); p.add_argument('--long-mppi', default='20261030T065714Z-61d3599d')
    p.add_argument('--output', type=Path, default=HERE); a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    scene = ROOT/'experiments/cabinet/scenes/cabinet.yml'; pc = PieceClearance(scene, 'split', 'cpu')
    doc = yaml.safe_load(scene.read_text()); cuboids = []
    for c in doc['cuboids']:
        assert c['pose'][3:] == [1., 0., 0., 0.], c['name']
        cuboids.append(dict(name=c['name'], lo=np.subtract(c['pose'][:3], np.divide(c['dims'], 2)).tolist(), hi=np.add(c['pose'][:3], np.divide(c['dims'], 2)).tolist(), color=c.get('color', [128, 128, 128])))
    edges = np.round(np.arange(0, 1.6+a.bin/2, a.bin), 6); agree = []
    scenes = [frozen_tab(pc, edges, a.per_bin, cuboids, agree), long_tab(pc, edges, a.per_bin, cuboids, a.long, a.long_mppi, agree)]
    rng = np.random.default_rng(5); ref_q = rng.uniform(-math.pi, math.pi, (16, 6)); meshes = robot_meshes(); links = sorted({m['link'] for m in meshes} | {R.EE_LINK})
    with torch.no_grad(): t = R.Kinematics().transforms(torch.tensor(ref_q, dtype=torch.float64), links)
    reference = dict(q=ref_q.round(6).tolist(), links=links, poses=torch.stack([t[l] for l in links], 1).reshape(16, -1, 16).numpy().round(7).tolist())
    names = {}
    for label, run, *_ in FIELDS:
        names[f'mppi_{label}'] = f'{label} · MPPI (seed 0)'; names[f'rk4_{label}'] = f'{label} · RK4'
    style = {'witness': dict(color='--witness', dash=[3, 2])}
    for label, run, prefix, token in FIELDS:
        style[f'mppi_{label}'] = dict(color=token, dash=None); style[f'rk4_{label}'] = dict(color=token, dash=[1, 3])
    data = dict(
        lede='Planned paths of the cabinet fields replayed on the real arm, with the queries grouped by start–goal distance |q − g| '
             '(normalized joint units, bins of the segment width). Red frames are where the arm touches the cabinet under the scorer\'s collision test.',
        robots=dict(ur5e=dict(name='UR5e + Robotiq 2F-140', base='base_link', joints=urdf_joints(R.URDF, R.JOINTS), meshes=meshes, ee_link=R.EE_LINK, reference=reference)),
        sources=dict(names, witness='Certified witness path'), sourceStyle=style, scenes=scenes,
        about=['Fields: B5 = original method (c42c6e8f); ES = 20 eikonal specialists (b7ab5a49); DP = DP chain (f57e1332); DP + TD + bell = 20-segment chain (502d7e3a); ntrl-demo = metric NTFields, best cabinet run (664b5d50).',
               f'Saved paths of the registered runs, verdicts from the scorer\'s .score.json. Red frames: release sphere model at the scorer\'s margins, sampled every 0.002 rad; on the {len(agree)} exported paths this agrees with the scorer\'s collision verdict on {sum(agree)}.',
               'Built by jeffrey/segment_dp/visualizer/build_viewer.py from the experiments/pntfields_cabinet/visualizer template.'])
    (a.output/'viewer.json').write_text(json.dumps(data, separators=(',', ':')))
    page = template().replace('/*KINEMATICS_JS*/', (OLD/'kinematics.js').read_text()).replace('/*DATA*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    (a.output/'viewer.html').write_text(page)
    print(f'collision agreement {sum(agree)}/{len(agree)}; {sum(len(s["queries"]) for s in scenes)} queries; wrote {a.output/"viewer.html"} ({len(page)/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
