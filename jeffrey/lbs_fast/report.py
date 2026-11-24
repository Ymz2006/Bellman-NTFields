"""Collect the LBS inference-time study from the queue's registered runs -> report/summary.json and report/tables.md.

    ladder    per scene and device job: every variant's ms per MPPI step, time per query, success; for the variants that draw the
              reference's random numbers, on how many queries they stop at the reference's step
    knobs     per scene: step multiplier / samples / bidirectional variants (GPU)
    dist      learned clearance fields: error statistics, and the planners with them next to the exact clearance
    final     every scene (first 250 queries): fast planners on the GPU and the native CPU kernel, with the reference: its step time
              re-measured on the idle machine (20 queries) times the step counts jeffrey/lbs_final recorded for the same queries, and
              the times jeffrey/lbs_final recorded itself (machine shared with trainings and other benches)
Families pool the three instances of a maze size / gap scale; houses are pooled over the ten houses.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q = HERE / 'queue'
SCENES = json.load(open(HERE / 'scenes.json'))
HOUSES = [k for k in SCENES if k.endswith('3d')]
FAMILIES = [(f'maze{n}', f'maze {n} × {n}', [f'maze{n}_{i}' for i in (1, 2, 3)]) for n in (10, 14, 18, 22)] + \
           [(f'gap{k}', f'gap ×{k}', [f'gap{k}_{i}' for i in (1, 2, 3)]) for k in (1, 2, 4, 8)] + [('houses', '3-D houses (10)', HOUSES)]


def run_dir(rid):
    return ROOT / 'results' / f'registered_{rid}'


def stats(rows, key='wall'):
    ok = [r for r in rows if r['success']]
    out = dict(n=len(rows), success=len(ok), pct=100 * len(ok) / max(len(rows), 1), steps_all=float(np.mean([r['steps'] for r in rows])) if rows else None,
               ms_per_step=1e3 * sum(r[key] for r in rows) / max(sum(r['steps'] for r in rows), 1) if rows and key in rows[0] else None)
    if rows and key in rows[0]:
        t = np.array([r[key] for r in rows])
        out['t_all'] = dict(mean=float(t.mean()), median=float(np.median(t)))
        if ok:
            to = np.array([r[key] for r in ok])
            out['t_ok'] = dict(mean=float(to.mean()), median=float(np.median(to)), std=float(to.std()), p90=float(np.quantile(to, 0.9)))
    if ok:
        out['steps_ok'] = float(np.mean([r['steps'] for r in ok]))
        out['len_ok'] = float(np.mean([r['length_cells'] for r in ok]))
    out['collisions'] = int(sum(bool(r.get('collision')) and bool(r.get('converged')) for r in rows))
    out['stalls'] = int(sum(not r.get('converged') for r in rows))
    return out


def load_bench(rid):
    return json.load(open(run_dir(rid) / 'bench.json'))


def main():
    state = json.load(open(Q / 'state.json'))
    done = state['done']
    out = dict(ladder={}, knobs={}, distcmp={}, dist={}, final={}, direction={}, families={}, runs=dict(done), failed=list(state.get('failed', {})), scenes=SCENES)
    for name, rid in done.items():
        kind, scene, tag = name.split(':')
        if kind == 'dist':
            r = json.load(open(run_dir(rid) / 'results.json'))
            out['dist'].setdefault(scene, {})[tag] = dict(run=rid, parameters=r['parameters'], train_seconds=r['train_seconds'], error=r['error'], cfg=r['cfg'], scene_name=r['scene_name'])
            continue
        b = load_bench(rid)
        entry = dict(run=rid, device=b['device'], gpu=b.get('gpu'), threads=b['threads'], scene_name=b['scene_name'], n=b['n_queries'], planner=b['planner'],
                     native_check=b.get('native_check'), field_check=b.get('field_check'), variants={})
        ref = b['variants'].get('ref', {}).get('rows')
        for label, v in b['variants'].items():
            s = stats(v['rows'])
            s['setup_seconds'] = v['setup_seconds']
            s['spec'] = v['spec']
            s['how'] = {h: sum(r['how'] == h for r in v['rows']) for h in sorted(set(r['how'] for r in v['rows']))}
            if ref is not None and label != 'ref' and v['spec'].get('n', 'ref') == 'ref' and v['spec'].get('v', 'goal') == 'goal' and v['spec'].get('c') != 'native':
                m = min(len(ref), len(v['rows']))
                s['same_as_ref'] = [int(sum(a['steps'] == c['steps'] and a['converged'] == c['converged'] for a, c in zip(ref[:m], v['rows'][:m]))), m]
            entry['variants'][label] = s
        entry['rows'] = {label: [dict(w=r['wall'], s=r['steps'], ok=r['success'], c=r['converged']) for r in v['rows']] for label, v in b['variants'].items()}
        out[kind].setdefault(scene, {})[tag] = entry
    # the reference of jeffrey/lbs_final on the same queries
    for scene, jobs in out['final'].items():
        fb = SCENES[scene].get('final_bench')
        if not fb:
            continue
        b = json.load(open(run_dir(fb) / 'bench.json'))
        n = max(e['n'] for e in jobs.values())
        rows = b['methods']['lbs']['rows'][:n]
        ref = jobs.get('cpu', {}).get('variants', {}).get('ref')
        ms = ref['ms_per_step'] if ref else None
        rr = [dict(wall=r['time'], steps=r['steps'], success=r['success'], converged=r['converged'], collision=r['collision'], length_cells=r['length_cells'],
                   clean=(r['steps'] * ms / 1e3 if ms else None)) for r in rows]
        e = dict(recorded=stats(rr, 'wall'), run=fb, ms_per_step_clean=ms)
        if ms:
            e['clean'] = stats(rr, 'clean')
        e['rows'] = [dict(w=r['wall'], s=r['steps'], ok=r['success'], c=r['converged'], clean=r['clean']) for r in rr]
        jobs['lbs_final'] = e
    # direction control: per query, one-way forward / backward / either against the bidirectional planners
    for scene, jobs in out['direction'].items():
        R = jobs['gpu']['rows']
        if all(k in R for k in ('goal', 'goal_rev', 'meet')):
            f = np.array([r['ok'] for r in R['goal']])
            b = np.array([r['ok'] for r in R['goal_rev']])
            m = np.array([r['ok'] for r in R['meet']])
            d = dict(n=int(len(f)), forward=int(f.sum()), backward=int(b.sum()), either=int((f | b).sum()), both_dirs=int((f & b).sum()), meet=int(m.sum()),
                     meet_and_either=int((m & (f | b)).sum()), meet_only=int((m & ~(f | b)).sum()), either_only=int(((f | b) & ~m).sum()))
            if 'both' in R:
                bo = np.array([r['ok'] for r in R['both']])
                d.update(both=int(bo.sum()), both_and_either=int((bo & (f | b)).sum()))
            jobs['gpu']['direction'] = d
    # families
    for key, title, scenes in FAMILIES:
        fam = dict(title=title, scenes=[s for s in scenes if s in out['final']], cells={})
        pools = {}
        for s in fam['scenes']:
            jobs = out['final'][s]
            for dev in ('gpu', 'cpu'):
                if dev in jobs:
                    for label, rows in jobs[dev]['rows'].items():
                        if label == 'ref':
                            continue
                        pools.setdefault(f'{dev}:{label}', []).extend(dict(wall=r['w'], steps=r['s'], success=r['ok'], converged=r['c'], length_cells=0.0) for r in rows)
            if 'lbs_final' in jobs:
                for r in jobs['lbs_final']['rows']:
                    pools.setdefault('ref:recorded', []).append(dict(wall=r['w'], steps=r['s'], success=r['ok'], converged=r['c'], length_cells=0.0))
                    if r['clean'] is not None:
                        pools.setdefault('ref:clean', []).append(dict(wall=r['clean'], steps=r['s'], success=r['ok'], converged=r['c'], length_cells=0.0))
        for k, rows in pools.items():
            c = stats(rows)
            c.pop('len_ok', None)
            fam['cells'][k] = c
        out['families'][key] = fam
    (HERE / 'report').mkdir(exist_ok=True)
    slim = json.loads(json.dumps(out))
    for sec in ('ladder', 'knobs', 'distcmp', 'final', 'direction'):
        for scene in slim[sec].values():
            for e in scene.values():
                e.pop('rows', None)
    json.dump(slim, open(HERE / 'report/summary.json', 'w'), indent=1)
    json.dump(out, open(HERE / 'report/full.json', 'w'))
    # tables
    L = ['# LBS inference time: tables', '']

    def table(title, entry_by_tag):
        L.append(f'### {title}')
        L.append('')
        L.append('| job | variant | solved | ms / step | steps (all) | time per solved query (ms) | time per query, all (ms) | same stop as ref | setup (s) |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for tag, e in entry_by_tag.items():
            if 'variants' not in e:
                continue
            for label, s in e['variants'].items():
                same = f"{s['same_as_ref'][0]}/{s['same_as_ref'][1]}" if 'same_as_ref' in s else ''
                tok = f"{1e3 * s['t_ok']['mean']:.1f}" if 't_ok' in s else '-'
                L.append(f"| {tag} | {label} | {s['success']}/{s['n']} | {s['ms_per_step']:.3f} | {s['steps_all']:.0f} | {tok} | {1e3 * s['t_all']['mean']:.1f} | {same} | {s['setup_seconds']:.0f} |")
        L.append('')
    for sec in ('ladder', 'knobs', 'distcmp', 'final', 'direction'):
        L.append(f'## {sec}')
        L.append('')
        for scene, jobs in out[sec].items():
            table(f'{scene}', jobs)
    L.append('## families (final, pooled)')
    L.append('')
    for key, fam in out['families'].items():
        L.append(f"### {fam['title']} ({len(fam['scenes'])} scenes)")
        L.append('')
        L.append('| planner | solved % | n | time per solved query (ms) | median | time per query, all (ms) | ms / step | steps (solved) |')
        L.append('|---|---|---|---|---|---|---|---|')
        for k, c in fam['cells'].items():
            if 't_ok' in c:
                L.append(f"| {k} | {c['pct']:.1f} | {c['n']} | {1e3 * c['t_ok']['mean']:.1f} | {1e3 * c['t_ok']['median']:.1f} | {1e3 * c['t_all']['mean']:.1f} | {c['ms_per_step']:.3f} | {c['steps_ok']:.0f} |")
        L.append('')
    L.append('## learned clearance fields')
    L.append('')
    L.append('| scene | model | parameters | train (s) | mean abs error in the lift band (cells) | 95 % | max | free called blocked | h error in band (cells) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for scene, d in out['dist'].items():
        for tag, r in d.items():
            e = r['error']
            L.append(f"| {scene} | {tag} | {r['parameters'] / 1e6:.2f} M | {r['train_seconds']:.0f} | {e['mae_band']:.5f} | {e['p95_band']:.5f} | {e['max_band']:.4f} | {100 * e['free_called_blocked']:.2f} % | {e['h_mae_band_cells']:.4f} |")
    (HERE / 'report/tables.md').write_text('\n'.join(L) + '\n')
    print('ladder', len(out['ladder']), 'knobs', len(out['knobs']), 'distcmp', len(out['distcmp']), 'final', len(out['final']), 'dist', len(out['dist']), 'failed', out['failed'])


if __name__ == '__main__':
    main()
