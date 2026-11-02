"""Tables of the direction sweep from registered results (markdown on stdout).

    python3 jeffrey/direction_sweep/report.py --ext nowall=<extend run> cabinet=<extend run>

Collects every complete, non-smoke train run under results/ whose --ext is one of the given extend runs.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
BANDS = dict(short=(0, .24), mid=(.24, .8), long=(.8, 1.61))
# Checkpoints of the original method with the same seed (D = 4 must reproduce them tensor for tensor).
REFERENCE = {('nowall', 87358): 'results/registered_20261027T045906Z-45797805/training/checkpoint.pt',   # same machine (the assets/ checkpoints come from the original GPU)
             ('nowall', 1): 'results/registered_20261027T183907Z-1e328ab5/training/checkpoint.pt',
             ('nowall', 2): 'results/registered_20261027T211213Z-13c28f64/training/checkpoint.pt',
             ('cabinet', 87358): 'results/registered_20261019T034213Z-f9c501f9/training/checkpoint.pt',
             ('cabinet', 1): 'results/registered_20261027T182552Z-1e04efd5/training/checkpoint.pt',
             ('cabinet', 2): 'results/registered_20261027T183810Z-3b2e6860/training/checkpoint.pt'}


def same_tensors(a, b):
    import torch
    x, y = (torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'] for p in (a, b))
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def load(path):
    return json.loads(Path(path).read_text())


def query_flags(run, scene):
    """{(method, task, planning seed): set of solved query ids}."""
    ev = load(run/'evaluation/evaluation.json'); out = {}
    for s in ev['scores']:
        key = (s['method'], s['task'], s['planning_seed'])
        if scene == 'cabinet': out[key] = set(s['success_ids'])
        else:
            label = ('bellman_rk4_'+s['task']) if s['method'] == 'native_rk4' else f'bellman_mppi_{s["task"]}_seed{s["planning_seed"]}'
            out[key] = set(load(run/'evaluation'/(label+'.score.json'))[label]['solved_ids'])
    return out, ev


def band_mean(fit, label, metric, edges):
    row = fit['fields'][label][metric]; centres = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
    return {b: float(np.mean([v for v, c in zip(row, centres) if lo <= c < hi and v is not None])) for b, (lo, hi) in BANDS.items()}


def failures(run, scene):
    """Failure reasons summed over queries (a query can count under several reasons)."""
    out = dict(rk4=Counter(), mppi=Counter())
    for f in (run/'evaluation').glob('*.score.json'):
        d = load(f); k = [x for x in d if not x.endswith('_groups')][0]; m = 'rk4' if '_rk4_' in k else 'mppi'
        if 'per_query' in d[k]:
            for q in d[k]['per_query'].values():
                if not q['success']: out[m].update({x.split(':')[0].split('(')[0].strip() for x in q['reasons']} or {'?'})
    if scene == 'cabinet':
        for s in load(run/'evaluation/evaluation.json')['scores']: out['rk4' if s['method'] == 'native_rk4' else 'mppi'].update(s.get('failures', {}))
    return out


def fmt_fail(c): return ', '.join(f'{k} {v}' for k, v in sorted(c.items()))


def fmt(x, d=3): return '–' if x is None else f'{x:.{d}f}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ext', nargs='+', required=True)
    p.add_argument('--posthoc', nargs='*', default=[], help='scene=<diag2 run>'); a = p.parse_args(); posthoc = dict(x.split('=') for x in a.posthoc)
    exts = dict(x.split('=') for x in a.ext); runs = defaultdict(dict); controls = defaultdict(dict)
    for st in sorted(ROOT.glob('results/registered_*/status.json')):
        try: s = load(st)
        except ValueError: continue
        cfg = s.get('settings', {})
        if not s.get('complete') or cfg.get('stage') != 'train' or cfg.get('smoke'): continue
        for scene, ext in exts.items():
            if not cfg.get('ext', '').rstrip('/').endswith(ext): continue
            if 'pairs' in cfg: runs[scene][(cfg['pairs'], cfg['seed'])] = st.parent
            else: controls[scene][(cfg['bellman_pairs'], cfg['direction_pairs'], cfg['seed'])] = st.parent
    for scene, ext in exts.items():
        em = load(ROOT/'results'/ext/'ext/manifest.json'); R = runs[scene]
        pairs = sorted({d for d, _ in R}); seeds = sorted({s for _, s in R}, key=lambda s: (s != 87358, s))
        print(f'\n## {scene} (extension {ext}; audit safe agreement {em["audit"]["safe_agreement"]:.4f}, cost max rel {em["audit"]["cost_max_rel"]:.1e})\n')
        any_diag = load(next(iter(R.values()))/'diag.json') if R else None
        print('| pairs D | actions | mean safe actions | sources w/o safe action | best cos to optimal direction |\n|---|---|---|---|---|')
        for d, b in em['by_pairs'].items():
            cov = any_diag['coverage'].get(str(d), {}).get('mean_best_cosine') if any_diag else None
            if cov is None and scene in posthoc: cov = next(iter(load(ROOT/'results'/posthoc[scene]/'diag2.json').values()))['coverage'][str(d)]['mean_best_cosine']
            print(f'| {d} | {b["actions"]} | {b["mean_safe_actions"]:.1f} | {b["sources_without_safe_action"]} ({b["sources_without_safe_action"]/em["sources"]:.1%}) | {fmt(cov)} |')
        # Planning.
        print('\n| D | seeds | RK4 /600 per seed | RK4 mean | MPPI /1800 per seed | MPPI mean | MPPI gained/lost vs D4 (sum over seeds) | checkpoint = baseline? |\n|---|---|---|---|---|---|---|---|')
        flags = {k: query_flags(v, scene) for k, v in R.items()}
        tasks = sorted({t for f, _ in flags.values() for (_, t, _) in f})
        for d in pairs:
            ss = [s for s in seeds if (d, s) in R]; rk = [flags[(d, s)][1]['native_success'] for s in ss]; mp = [sum(flags[(d, s)][1]['mppi_successes']) for s in ss]
            gained = lost = 0
            for s in ss:
                if (4, s) not in flags or d == 4: continue
                for key, solved in flags[(d, s)][0].items():
                    if key[0] != 'matched_mppi': continue
                    ref = flags[(4, s)][0][key]; gained += len(solved-ref); lost += len(ref-solved)
            same = '' if d != 4 else ', '.join(('yes' if same_tensors(R[(4, s)]/'training/checkpoint.pt', ROOT/REFERENCE[(scene, s)]) else 'NO')
                                               if (scene, s) in REFERENCE else '?' for s in ss)
            print(f'| {d} | {len(ss)} | {"/".join(map(str, rk))} | {np.mean(rk):.1f} | {"/".join(map(str, mp))} | {np.mean(mp):.1f} | {"" if d == 4 else f"+{gained}/−{lost}"} | {same} |')
        print('\nMPPI successes per task (mean over training seeds, of 300 = 100 queries x 3 planning seeds); RK4 per task of 100 in brackets\n')
        print('| D | ' + ' | '.join(tasks) + ' |\n|---|' + '---|'*len(tasks))
        for d in pairs:
            ss = [s for s in seeds if (d, s) in R]; cells = []
            for t in tasks:
                m = np.mean([sum(len(v) for k, v in flags[(d, s)][0].items() if k[0] == 'matched_mppi' and k[1] == t) for s in ss])
                r = np.mean([len(v) for s in ss for k, v in flags[(d, s)][0].items() if k[0] == 'native_rk4' and k[1] == t])
                cells.append(f'{m:.0f} ({r:.0f})')
            print(f'| {d} | ' + ' | '.join(cells) + ' |')
        # Accuracy.
        print('\nAccuracy against ground truth (truth.py optimized paths), mean over training seeds. rel err = mean |log T/T*|; bias = median T/T*; '
              'speed err = mean |log S|grad T|| (implied speed 1/|grad T| vs the speed net); short/mid/long gap = 0-0.24 / 0.24-0.8 / 0.8-1.6\n')
        print('| D | rel err short/mid/long | bias short/mid/long | speed err short/mid/long | cosine to optimal dir short/mid/long |\n|---|---|---|---|---|')
        for d in pairs:
            ss = [s for s in seeds if (d, s) in R]; cols = []
            for metric in ('rel_error', 'bias', 'eikonal', 'cosine'):
                vals = [band_mean(f, f'D{d}', metric, f['edges']) for f in (load(R[(d, s)]/'fit.json') for s in ss)]
                cols.append('/'.join(f'{np.mean([v[b] for v in vals]):.3f}' for b in BANDS))
            print(f'| {d} | ' + ' | '.join(cols) + ' |')
        print('\nOn 16,384 bank sources with random goals (mean over training seeds; diag2.py post-hoc run when given, else diag.py). speed ratio = S(q)|grad_q T| '
              '(1 = implied speed exact, >1 field predicts slower than S); backup ratio = T / min over D\' pairs of [c + T(z)] (self-consistency with a min over D\' '
              'directions); confident = rows the direction term uses (gap >= 10 %)\n')
        print('| D | speed ratio median | speed |log| mean | speed within 5 % | S|grad T| at S<0.3 | backup ratio own D | backup ratio D\'=4 | backup ratio D\'=32 | winner cosine own D | confident own D |\n|---|---|---|---|---|---|---|---|---|---|')
        post = load(ROOT/'results'/posthoc[scene]/'diag2.json') if scene in posthoc else {}
        for d in pairs:
            ss = [s for s in seeds if (d, s) in R]; dg = [post.get(f'D{d}s{s}') or load(R[(d, s)]/'diag.json') for s in ss]
            def m(f, digits):
                try: return f'{np.mean([f(x) for x in dg]):.{digits}f}'
                except (KeyError, TypeError): return '–'
            print(f'| {d} | {m(lambda x: x["speed_ratio"]["median"], 3)} | {m(lambda x: x["speed_ratio"]["mean_abs_log"], 3)} | {m(lambda x: 100*x["speed_ratio"]["within_5pct"], 1)} % | '
                  f'{m(lambda x: x["speed_ratio_by_speed"]["0-0.3"]["median"], 3)} | {m(lambda x: x["backup_ratio"][str(d)]["median"], 4)} | {m(lambda x: x["backup_ratio"]["4"]["median"], 4)} | '
                  f'{m(lambda x: x["backup_ratio"]["32"]["median"], 4)} | {m(lambda x: x["winner_cosine"][str(d)], 3)} | {m(lambda x: x["confident"][str(d)], 3)} |')
        C = controls[scene]
        if C:
            print('\nMechanism control: Bellman min over Db pairs, direction term (winner, runner-up, confidence gate) over Dd pairs; '
                  'MPPI gained/lost vs D4 of the same seed; failures summed over RK4 / MPPI\n')
            print('| Db | Dd | seed | RK4 /600 | MPPI /1800 | MPPI gained/lost vs D4 | RK4 failures | MPPI failures |\n|---|---|---|---|---|---|---|---|')
            rows = [((d, d, s), R[(d, s)]) for d, s in sorted(R) if d in (4, 32) or any(k[0] == d or k[1] == d for k in C)]
            for (b, dd, s), path in sorted(rows + list(C.items()), key=lambda x: (x[0][2] != 87358, x[0][2], x[0][0], x[0][1])):
                f, ev = query_flags(path, scene); gained = lost = 0
                if (4, s) in flags and (b, dd) != (4, 4):
                    for key, solved in f.items():
                        if key[0] == 'matched_mppi': ref = flags[(4, s)][0][key]; gained += len(solved-ref); lost += len(ref-solved)
                fr = failures(path, scene)
                print(f'| {b} | {dd} | {s} | {ev["native_success"]} | {sum(ev["mppi_successes"])} | {"" if (b, dd) == (4, 4) else f"+{gained}/−{lost}"} | '
                      f'{fmt_fail(fr["rk4"])} | {fmt_fail(fr["mppi"])} |')
        if post:
            print('\nDirection-term mechanism (post-hoc diag2, training seed 87358). confident = share of sources the direction term uses with its own Dd; '
                  'winner cos = cos(-grad T, winner step) at Dd; toward clearance = mean cos(-grad T, grad S) by speed band (S < 0.3 is next to obstacles)\n')
            print('| field | Db | Dd | RK4 | MPPI | confident | winner cos | toward clearance S 0-0.3 / 0.3-0.6 / 0.6-0.9 | S|grad T| at S<0.3 |\n|---|---|---|---|---|---|---|---|---|')
            entries = [((d, d), R[(d, 87358)]) for d in pairs if (d, 87358) in R] + [((b, dd), v) for (b, dd, s), v in C.items() if s == 87358]
            for (b, dd), path in sorted(entries, key=lambda x: -sum(load(x[1]/'evaluation/evaluation.json')['mppi_successes'])):
                label = (f'D{b}' if b == dd else f'B{b}D{dd}') + 's87358'
                if label not in post: continue
                x = post[label]; ev = load(path/'evaluation/evaluation.json'); t = x['descent_toward_speed']
                print(f'| {label[:-6]} | {b} | {dd} | {ev["native_success"]} | {sum(ev["mppi_successes"])} | {x["confident"][str(dd)]:.3f} | {x["winner_cosine"][str(dd)]:.3f} | '
                      f'{t.get("0-0.3", float("nan")):.3f} / {t.get("0.3-0.6", float("nan")):.3f} / {t.get("0.6-0.9", float("nan")):.3f} | {x["speed_ratio_by_speed"]["0-0.3"]["median"]:.3f} |')
        print('\nRuns: ' + ', '.join(f'D{d} s{s} {v.name}' for (d, s), v in sorted(R.items())) + ''.join(f', B{b}D{d} s{s} {v.name}' for (b, d, s), v in sorted(C.items())))


if __name__ == '__main__': main()
