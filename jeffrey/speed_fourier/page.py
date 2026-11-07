"""Findings page of the speed-Fourier study (report/speed_fourier.html), from the registered runs (report.collect()), report/fit.json, capture.json,
ablate.json, mechanism.json, profile.json and report/rows.png.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/page.py

Every number on the page is computed here: the prose in page_template.html and page_text/*.html uses {{name}} tokens from numbers(), and a token without a value
stops the build. Publish report/speed_fourier.html with rows.png next to it.
"""
import html
import json
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import report as R

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
S = R.SEEDS
# key in report.collect(), label on the page, family, walls
ARMS = [('ntrl-demo', 'ntrl-demo', 'ntrl', None), ('octave', '128 octave rows', 'base', None), ('topzero', '120 octaves + 16 empty', 'base', None),
        ('top line M30 flip', '+ 16 speed rows, flipped', 'speed', 'flip'), ('top line M30 true', '+ 16 speed rows', 'speed', 'true'),
        ('oracle flip', '+ exact speed, flipped', 'oracle', 'flip'), ('oracle true', '+ exact speed', 'oracle', 'true'),
        ('geo flip', '+ 16 distance rows, flipped', 'geo', 'flip'), ('geo true', '+ 16 distance rows', 'geo', 'true')]
LABEL = {a[0]: a[1] for a in ARMS}
LABEL.update({'half': '64 octaves + 128 empty', 'mix line M30 true': '64 octaves + 128 speed rows', 'mix line M30 flip': '64 octaves + 128 speed rows, flipped', 'mix line M5 true': '64 octaves + 128 speed rows, M = 5',
              'mix ridge M30 true': '64 octaves + 128 ridge rows', 'mix ridge M30 flip': '64 octaves + 128 ridge rows, flipped', 'mix terms M32 true': '64 octaves + 128 single terms',
              'mix terms M32 flip': '64 octaves + 128 single terms, flipped', 'mix map true': '64 octaves + 128 copies of the map', 'plain spectrum': '128 peaks of the maze spectrum (plain sin/cos)',
              'pure line M30 true': 'speed rows only, M = 30', 'pure line M5 true': 'speed rows only, M = 5', 'pure ridge M30 true': 'ridge rows only', 'pure ridge M30 flip': 'ridge rows only, flipped',
              'pure terms M30 true': 'single terms only', 'pure terms M30 flip': 'single terms only, flipped'})
LABEL.update({'top line M30 true slow': '+ 16 slowness rows (1 / speed), not integrated', 'top integral M30 true slow': '+ 16 integral rows, 30 terms', 'top integral M30 flip slow': '+ 16 integral rows, 30 terms, flipped',
              'top exact-integral true slow': '+ 16 integral rows, exact', 'top exact-integral flip slow': '+ 16 integral rows, exact, flipped', 'top share M30 true slow': '+ 16 integral rows, share of the line total',
              'top both M30 true slow': '+ 8 integral rows + 8 speed rows', 'top both M30 flip slow': '+ 8 integral rows + 8 speed rows, flipped', 'mix iterms M31 true slow': '64 octaves + 128 per-term integrals',
              'mix iterms M31 flip slow': '64 octaves + 128 per-term integrals, flipped'})
LABEL.update({'pure exact-integral true slow': 'exact integral only (no octaves)', 'pure integral M30 true slow': '30-term integral only (no octaves)', 'pure integral M30 flip slow': '30-term integral only, flipped',
              'pure exact-integral true unit': 'length along the line only (no octaves, no walls)', 'pure both M30 true slow': 'integral rows + speed rows (no octaves)',
              'pure iterms M29 true slow': 'per-term integral channels only (no octaves)', 'pure iterms M29 flip slow': 'per-term integral channels only, flipped'})
TERMS = [5, 15, 30, 60, 120]
# part 2 (integral rows): name in the tokens, key in report.collect(), control it is paired with, flipped twin
P2 = [('oct', 'octave', None, None), ('zero', 'topzero', None, None), ('true', 'top line M30 true', 'topzero', 'top line M30 flip'), ('slowline', 'top line M30 true slow', 'topzero', None),
      ('int', 'top integral M30 true slow', 'topzero', 'top integral M30 flip slow'), ('exact', 'top exact-integral true slow', 'topzero', 'top exact-integral flip slow'),
      ('share', 'top share M30 true slow', 'topzero', None), ('both', 'top both M30 true slow', 'topzero', 'top both M30 flip slow'), ('geo', 'geo true', 'topzero', 'geo flip'),
      ('half', 'half', None, None), ('iterms', 'mix iterms M31 true slow', 'half', 'mix iterms M31 flip slow')]


def load(name):
    return json.loads((OUT / name).read_text())


def pct(v):
    ok, tot, _ = R.pooled(v)
    return 100 * ok / tot


def mean(rows, *path):
    def get(x):
        for p in path:
            x = x[p]
        return x
    return float(np.mean([get(r) for r in rows]))


def success_data(t):
    out = {}
    for n in (22, 13):
        rows = []
        for key, label, fam, walls in ARMS:
            v = t[n].get(key, {})
            seeds = [dict(seed=s, success=100 * v[s]['success'], ok=v[s]['ok'], tot=v[s]['tot'], run=v[s]['id']) for s in S if s in v]
            rows.append(dict(key=key, label=label, family=fam, walls=walls, seeds=seeds, mean=pct(v) if seeds else None))
        out[n] = rows
    return out


def sweep_data(t, ab):
    out = {}
    for n in (22, 13):
        d = dict(zero=pct(t[n]['topzero']), octave=pct({s: t[n]['octave'][s] for s in S}), series={})
        for w in ('true', 'flip'):
            pts = []
            for m in TERMS:
                v = t[n].get(f'top line M{m} {w}', {})
                x = [100 * v[s]['success'] for s in S if s in v]
                if x:
                    rows = [ab[v[s]['id']]['intact'] for s in S if s in v and v[s]['id'] in ab]
                    pts.append(dict(M=m, mean=pct(v), seeds=x, sep=float(np.mean([r['sep'] for r in rows])) if rows else None, jump=float(np.mean([r['jump_median'] for r in rows])) if rows else None,
                                    rise=float(np.mean([r['rise_median'] for r in rows])) if rows else None, leak=float(np.mean([r['leak'] for r in rows])) if rows else None, sep_seeds=[r['sep'] for r in rows], jump_seeds=[r['jump_median'] for r in rows]))
            d['series'][w] = pts
        out[n] = d
    return out


def by_field(rows, n, ids):
    return [rows[i] for i in ids if i in rows]


def ids(t, n, key):
    return [t[n][key][s]['id'] for s in S if s in t[n].get(key, {})]


def mech_data(t, ab, me):
    out = {}
    for n in (22, 13):
        rows = []
        for key, label, fam, walls in ARMS:
            a = by_field(ab, n, ids(t, n, key))
            m = by_field(me, n, ids(t, n, key))
            if not a:
                continue
            r = dict(key=key, label=label, family=fam, walls=walls, seeds=len(a), success=pct({s: t[n][key][s] for s in S if s in t[n][key]}),
                     sep=mean(a, 'intact', 'sep'), rise=mean(a, 'intact', 'rise_median'), jump=mean(a, 'intact', 'jump_median'), leak=mean(a, 'intact', 'leak'), door=mean(a, 'intact', 'door'),
                     wrong=mean(a, 'intact', 'jump_wrong_sign'), bumps=mean(m, 'bumps', 'bumps_per_path') if m else None)
            if 'frozen' in a[0]:
                r.update(fsep=mean(a, 'frozen', 'sep'), frise=mean(a, 'frozen', 'rise_median'), fjump=mean(a, 'frozen', 'jump_median'), fleak=mean(a, 'frozen', 'leak'), fdoor=mean(a, 'frozen', 'door'))
            rows.append(r)
        out[n] = rows
    return out


def profile_data(t, pr):
    out = {}
    for n in (22, 13):
        d = {}
        for key in ('topzero', 'top line M30 true', 'oracle true', 'geo true'):
            rows = [pr[i] for i in ids(t, n, key) if i in pr]
            if rows:
                d[key] = dict(label=LABEL[key], field=np.mean([r['field'] for r in rows], 0).tolist(), seeds=len(rows))
                if 'frozen' in rows[0]:
                    d[key]['frozen'] = np.mean([r['frozen'] for r in rows], 0).tolist()
                if 'rowtrack' in rows[0]:
                    d[key]['rowtrack'] = float(np.mean([r['rowtrack'] for r in rows]))
        any_row = pr[ids(t, n, 'topzero')[0]]
        out[n] = dict(u=any_row['u'], truth=np.mean([pr[i]['truth'] for i in ids(t, n, 'topzero')], 0).tolist(), arms=d)
    return out


def capture_data(t, cap):
    cap = [c for c in cap if c['n'] in (13, 22) and c['seed'] in S]
    ids_of = {(n, r['id']): k for n in t for k, v in t[n].items() for s, r in v.items() if s in S}
    out = {}
    for n in (22, 13):
        g = {}
        for c in cap:
            k = ids_of.get((n, c['run']))
            if k is not None:
                g.setdefault(k, []).append(c)
        rows = []
        for k, v in g.items():
            fam = next((a[2] for a in ARMS if a[0] == k), 'pure' if k.startswith('pure') else 'other')
            rows.append(dict(key=k, label=LABEL.get(k, k), family=fam, seeds=len(v), capture=mean(v, 'capture'), layer=mean(v, 'layer'), cells=mean(v, 'cells'), resolve=mean(v, 'resolve'), rank=mean(v, 'rank'),
                             coord=mean(v, 'coord'), coordlayer=mean(v, 'coordlayer'), success=100 * mean(v, 'success')))
        out[n] = rows
    return out


def signed(x, d=1):
    return f'{x:.{d}f}'.lstrip('-') if round(x, d) == 0 else f'{x:+.{d}f}'.replace('-', '−')


def numbers(t, fit, mech, prof, cap, ab, raw, me):
    N = {}
    for n in (22, 13):
        for key, name in (('ntrl-demo', 'ntrl'), ('octave', 'octave'), ('topzero', 'zero'), ('top line M30 true', 'true'), ('top line M30 flip', 'flip'), ('oracle true', 'oracle'), ('oracle flip', 'oracleflip'),
                          ('geo true', 'geo'), ('geo flip', 'geoflip'), ('half', 'half')):
            v = {s: x for s, x in t[n].get(key, {}).items() if s in S}
            if v:
                N[f'{name}{n}'] = f'{pct(v):.1f}'
        for a, b, name in (('top line M30 true', 'topzero', 'truezero'), ('top line M30 flip', 'topzero', 'flipzero'), ('top line M30 true', 'top line M30 flip', 'trueflip'), ('oracle true', 'topzero', 'oraclezero'), ('oracle true', 'oracle flip', 'oracletf'),
                           ('geo true', 'topzero', 'geozero'), ('geo flip', 'topzero', 'geoflipzero'), ('geo true', 'octave', 'geooct'), ('geo true', 'geo flip', 'geotf')):
            d = R.paired(t, n, a, b)
            if d:
                N[f'd{name}{n}'] = signed(float(np.mean(d)))
                N[f'w{name}{n}'] = f'{sum(x > 0 for x in d)} of {len(d)}'
                N[f'l{name}{n}'] = f'{sum(x < 0 for x in d)} of {len(d)}'
                N[f'seeds{name}{n}'] = ' / '.join(signed(x) for x in d)
        for r in mech[n]:
            k = dict(zip([a[0] for a in ARMS], ['ntrl', 'octave', 'zero', 'flip', 'true', 'oracleflip', 'oracle', 'geoflip', 'geo']))[r['key']]
            for f in ('sep', 'jump', 'leak', 'bumps', 'fsep', 'fjump', 'fleak', 'door', 'fdoor'):
                if r.get(f) is not None:
                    N[f'{f}_{k}{n}'] = f'{r[f]:.2f}'
            for f in ('rise', 'frise'):
                if r.get(f) is not None:
                    N[f'{f}_{k}{n}'] = f'{r[f]:.1f}' if r[f] >= 1 else f'{r[f]:.2f}'
        for key, name in (('ntrl-demo', 'ntrl'), ('octave', 'octave'), ('topzero', 'zero'), ('top line M30 true', 'true'), ('top line M30 flip', 'flip'), ('oracle true', 'oracle'), ('geo true', 'geo')):
            v = {s: x for s, x in t[n].get(key, {}).items() if s in S}
            if v:
                N[f'coll_{name}{n}'] = f'{100 * sum(x["collisions"] for x in v.values()) / sum(x["tot"] for x in v.values()):.1f}'
                N[f'other_{name}{n}'] = f'{100 * sum(x["tot"] - x["ok"] - x["collisions"] for x in v.values()) / sum(x["tot"] for x in v.values()):.1f}'
        for k, a in (('true', 'top line M30 true'), ('oracle', 'oracle true'), ('geo', 'geo true')):
            x = prof[n]['arms'].get(a)
            if x and 'rowtrack' in x:
                N[f'track_{k}{n}'] = f'{x["rowtrack"]:.2f}'.replace('-', '−')
        by = {c['key']: c for c in cap[n]}
        for k, name in (('ntrl-demo', 'ntrl'), ('octave', 'octave'), ('top line M30 true', 'true'), ('top line M30 flip', 'flip'), ('oracle true', 'oracle'), ('oracle flip', 'oracleflip'), ('geo true', 'geo'), ('topzero', 'zero')):
            if k in by:
                N[f'layer_{name}{n}'] = f'{by[k]["layer"]:.2f}'
                N[f'cap_{name}{n}'] = f'{by[k]["capture"]:.2f}'
                N[f'cells_{name}{n}'] = f'{by[k]["cells"]:.2f}'
                N[f'coord_{name}{n}'] = f'{by[k]["coord"]:.2f}'
        rows = [c for c in raw if c['n'] == n and c['seed'] in S and not c['field'].endswith((' slow', ' unit'))]
        N[f'rho_layer{n}'] = signed(float(spearmanr([c['layer'] for c in rows], [c['success'] for c in rows]).statistic), 2)
        N[f'rho_cap{n}'] = signed(float(spearmanr([c['capture'] for c in rows], [c['success'] for c in rows]).statistic), 2)
        N[f'rho_capruns{n}'] = str(len(rows))
        if 'mix map true' in by:
            N[f'cap_map{n}'] = f'{by["mix map true"]["capture"]:.2f}'
        for k, name in (('pure line M30 true', 'pure'), ('mix line M30 true', 'mix')):
            if k in by:
                N[f'coord_{name}{n}'] = f'{by[k]["coord"]:.2f}'
                N[f'coordl_{name}{n}'] = f'{by[k]["coordlayer"]:.2f}'
                N[f'cells_{name}{n}'] = f'{by[k]["cells"]:.2f}'
                N[f'layer_{name}{n}'] = f'{by[k]["layer"]:.2f}'
        for m in fit[str(n)]['line']:
            N[f'r2_{m["M"]}_{n}'] = f'{m["r2"]:.2f}'
            N[f'rank_{m["M"]}_{n}'] = f'{m["rank"]:.1f}'
        for key, name in (('pure line M30 true', 'pure'), ('mix line M30 true', 'mix'), ('mix line M30 flip', 'mixflip'), ('mix line M5 true', 'mix5'), ('mix terms M32 flip', 'termsflip'), ('mix ridge M30 true', 'ridge'), ('mix ridge M30 flip', 'ridgeflip'), ('mix map true', 'map'), ('plain spectrum', 'plain'), ('mix terms M32 true', 'terms')):
            if key in t[n]:
                N[f'{name}{n}'] = f'{pct(t[n][key]):.1f}'
    for n, d in sweep_data(t, ab).items():
        for w, pts in d['series'].items():
            for p in pts:
                N[f'sw_{w}{p["M"]}_{n}'] = f'{p["mean"]:.1f}'
                if p['sep'] is not None:
                    N[f'swsep_{w}{p["M"]}_{n}'] = f'{p["sep"]:.2f}'
                    N[f'swjump_{w}{p["M"]}_{n}'] = f'{p["jump"]:.2f}'
                    N[f'swrise_{w}{p["M"]}_{n}'] = f'{p["rise"]:.1f}' if p['rise'] >= 1 else f'{p["rise"]:.2f}'
                    N[f'swleak_{w}{p["M"]}_{n}'] = f'{p["leak"]:.2f}'
    for n in (22, 13):
        sw = [v[s] for m in TERMS for v in [t[n][f'top line M{m} true']] for s in S if s in v]
        rho = lambda get: signed(float(spearmanr([get(r['id']) for r in sw], [r['ok'] for r in sw]).statistic), 2)
        N[f'rho_swsep{n}'], N[f'rho_swjump{n}'] = rho(lambda i: ab[i]['intact']['sep']), rho(lambda i: ab[i]['intact']['jump_median'])
        N[f'rho_swleak{n}'], N[f'rho_swbumps{n}'], N[f'rho_swruns{n}'] = rho(lambda i: ab[i]['intact']['leak']), rho(lambda i: me[i]['bumps']['bumps_per_path']), str(len(sw))
    for n in (22, 13):
        runs = [(k, v[s]) for k, v in t[n].items() for s in S if s in v and v[s]['id'] in ab and 'sep' in ab[v[s]['id']]['intact']]
        rho = lambda rows, f: float(spearmanr([ab[r['id']]['intact'][f] for _, r in rows], [r['ok'] for _, r in rows]).statistic)
        keep = [x for x in runs if x[0] not in ('ntrl-demo', 'plain spectrum') and not x[0].startswith('pure') and not x[0].endswith(' slow')]
        N[f'rho_sep{n}'], N[f'rho_jump{n}'], N[f'rho_runs{n}'] = f'{rho(keep, "sep"):.2f}', f'{rho(keep, "jump_median"):.2f}', str(len(keep))
        top = [x for x in keep if x[0].startswith(('top', 'oracle', 'geo', 'octave'))]
        N[f'rho_septop{n}'], N[f'rho_jumptop{n}'], N[f'rho_runstop{n}'] = f'{rho(top, "sep"):.2f}', f'{rho(top, "jump_median"):.2f}', str(len(top))
    N.update(part2_numbers(t, ab, me))
    N.update(part3_numbers(t, ab, me, raw))
    N.update(part4_numbers(t, ab, me))
    N['runs'] = str(sum(len(v) for n in t for k, v in t[n].items() if k not in ('ntrl-demo', 'octave')))
    return N


def part2_numbers(t, ab, me):
    N = {}
    sc = load('screen.json')
    for n in (22, 13):
        for k, name in (('slow integral M30', 'int'), ('slow integral M0', 'exact'), ('slow line M30', 'slowline'), ('obstacle line M30', 'true'), ('slow share M30', 'share'), ('geo', 'geo')):
            N[f'scr_{name}{n}'] = f'{sc[str(n)]["rows"][k]["ratio"]:.2f}'
        for name, key, ctrl, flip in P2:
            v = {s: x for s, x in t[n].get(key, {}).items() if s in S}
            if not v:
                continue
            N[f'p2_{name}{n}'], N[f'p2k_{name}{n}'] = f'{pct(v):.1f}', f'{len(v)} seed' + ('s' if len(v) > 1 else '')
            for other, oname in ((ctrl, 'ctrl'), (flip, 'flip'), ('octave', 'oct')):
                d = R.paired(t, n, key, other) if other and other != key else []
                if d:
                    same = {s: x for s, x in t[n][other].items() if s in v}
                    N[f'p2d_{name}{oname}{n}'], N[f'p2s_{name}{oname}{n}'] = signed(float(np.mean(d))), ' / '.join(signed(x) for x in d)
                    N[f'p2w_{name}{oname}{n}'], N[f'p2l_{name}{oname}{n}'] = f'{sum(x > 0 for x in d)} of {len(d)}', f'{sum(x < 0 for x in d)} of {len(d)}'
                    N[f'p2o_{name}{oname}{n}'] = f'{pct(same):.1f}'                                   # the other arm on the same seeds
                    N[f'p2self_{name}{oname}{n}'] = f'{pct({s: x for s, x in v.items() if s in same}):.1f}'
            a = [ab[x['id']] for x in v.values() if x['id'] in ab and 'sep' in ab[x['id']]['intact']]
            m = [me[x['id']] for x in v.values() if x['id'] in me]
            if a:
                N[f'p2jump_{name}{n}'], N[f'p2sep_{name}{n}'], N[f'p2leak_{name}{n}'] = f'{mean(a, "intact", "jump_median"):.2f}', f'{mean(a, "intact", "sep"):.2f}', f'{mean(a, "intact", "leak"):.2f}'
                if all('frozen' in r for r in a):
                    N[f'p2fjump_{name}{n}'] = f'{mean(a, "frozen", "jump_median"):.2f}'
            if m:
                N[f'p2bumps_{name}{n}'] = f'{mean(m, "bumps", "bumps_per_path"):.2f}'
            f = [x['fit']['r2_median'] for x in v.values() if x.get('fit') and 'r2_median' in x['fit']]
            if f and name in ('int', 'iterms'):
                N[f'p2r2_{name}{n}'] = f'{np.mean(f):.3f}'
    return N


# part 3 (no octaves): name in the tokens, key in report.collect(), label. Training seed 87358 only.
P3 = [('oct', 'octave', '128 octave rows (reference)'), ('ntrl', 'ntrl-demo', "ntrl-demo's 128 random rows (reference)"), ('exact', 'pure exact-integral true slow', 'exact integral only'),
      ('int', 'pure integral M30 true slow', '30-term integral only'), ('intflip', 'pure integral M30 flip slow', '30-term integral only, walls flipped (control)'),
      ('unit', 'pure exact-integral true unit', 'length along the line only (control: no walls in it)'), ('both', 'pure both M30 true slow', 'integral rows + speed rows'),
      ('iterms', 'pure iterms M29 true slow', 'one channel per sin/cos term (the matrix)'), ('itermsflip', 'pure iterms M29 flip slow', 'one channel per term, walls flipped (control)'),
      ('speed', 'pure line M30 true', 'speed rows only (first part)')]
SEED3 = 87358


def part3_numbers(t, ab, me, raw):
    N = {}
    cap = {c['run']: c for c in raw}
    z = lambda x, d=2: f'{0.0 if round(x, d) == 0 else x:.{d}f}'.replace('-', '−')
    for n in (22, 13):
        for name, key, _ in P3:
            r = t[n].get(key, {}).get(SEED3)
            if not r:
                continue
            N[f'p3_{name}{n}'], N[f'p3ok_{name}{n}'], N[f'p3t_{name}{n}'], N[f'p3rho_{name}{n}'] = f'{100 * r["success"]:.1f}', str(r['ok']), z(r['t_ratio']), z(r['spearman'])
            N[f'p3coll_{name}{n}'], N[f'p3fit_{name}{n}'] = f'{100 * r["collisions"] / r["tot"]:.0f}', z(r['speed_corr'])
            a, m, c = ab.get(r['id']), me.get(r['id']), cap.get(r['id'])
            if a and 'sep' in a['intact']:
                N[f'p3jump_{name}{n}'], N[f'p3sep_{name}{n}'], N[f'p3leak_{name}{n}'] = z(a['intact']['jump_median']), z(a['intact']['sep']), z(a['intact']['leak'])
            if m:
                N[f'p3bumps_{name}{n}'], N[f'p3cos_{name}{n}'] = z(m['bumps']['bumps_per_path'], 1), z(m['direction']['cos_median'])
            if c:
                N[f'p3rank_{name}{n}'], N[f'p3cells_{name}{n}'], N[f'p3coord_{name}{n}'], N[f'p3res_{name}{n}'] = f'{c["rank"]:.0f}' if c['rank'] >= 10 else f'{c["rank"]:.1f}', z(c['cells']), z(c['coord']), z(c['resolve'], 1)
        for a, b in (('exact', 'unit'), ('int', 'intflip'), ('iterms', 'itermsflip'), ('exact', 'ntrl'), ('exact', 'oct'), ('int', 'ntrl'), ('int', 'oct'), ('both', 'int')):
            if f'p3_{a}{n}' in N and f'p3_{b}{n}' in N:
                N[f'p3d_{a}{b}{n}'] = signed(float(N[f'p3_{a}{n}']) - float(N[f'p3_{b}{n}']))
    return N


def part3_table(N):
    g = lambda k: N.get(k, '–')
    two = lambda f, name: f'{g(f"{f}_{name}13")} <span class="mute">/</span> {g(f"{f}_{name}22")}'
    rows = []
    for name, key, lab in P3:
        if f'p3_{name}13' not in N and f'p3_{name}22' not in N:
            continue
        pc = lambda n: f'{N[f"p3_{name}{n}"]} %' if f'p3_{name}{n}' in N else '–'
        rows.append(f'<tr><td>{html.escape(lab)}</td><td class="num">{pc(13)}</td><td class="num">{pc(22)}</td><td class="num">{two("p3t", name)}</td><td class="num">{two("p3jump", name)}</td>'
                    f'<td class="num">{two("p3cos", name)}</td><td class="num">{two("p3bumps", name)}</td><td class="num">{two("p3rank", name)}</td></tr>')
    return '\n'.join(rows)


# part 4 (ray hit counts; a norm on the integral): name in the tokens, key in report.collect(), label, row of report/rayscreen.json
H = 'top exact-integral true slow +head '
P4 = [('zero', 'topzero', '120 octaves + 16 empty (control)', None), ('oct', 'octave', '128 octave rows', None),
      ('exact', 'top exact-integral true slow', '+ 16 integral rows, no norm (part 2)', 'integral of 1 / speed (part 2)'),
      ('hits', 'top hits true', '+ 16 ray hit counts (test 1)', 'hits to the edge'), ('hitsflip', 'top hits flip', '+ 16 ray hit counts, walls flipped (control)', None),
      ('hitsr2', 'top hits r2 true', '+ 16 ray hit counts within 2 cells', 'hits within 2 cells'),
      ('headl1', H + 'l1', 'integral rows + L1 norm in the distance, trained weight', None), ('headlinf', H + 'linf', 'integral rows + L∞ norm in the distance, trained weight', None),
      ('headl1f', H + 'l1 w0.2', 'integral rows + L1 norm in the distance, weight fixed at 0.2', None), ('headlinff', H + 'linf w0.2', 'integral rows + L∞ norm in the distance, weight fixed at 0.2', None),
      ('headl1one', H + 'l1 w1', 'integral rows + L1 norm in the distance, weight fixed at 1', None),
      ('mark', 'top markint l1 true', '+ 16 landmark rows: L1 norm of integral differences', 'landmark rows: L1 of integral differences (64 directions)'),
      ('markunit', 'top markint l1 true unit', 'same landmark rows without walls (control)', None), ('markflip', 'top markint l1 flip', 'same landmark rows, walls flipped (control)', None),
      ('markinf', 'top markint linf true', '+ 16 landmark rows: L∞ norm of integral differences', 'landmark rows: Linf of integral differences (64 directions)'),
      ('markhits', 'top markhits l1 true', '+ 16 landmark rows: L1 norm of hit-count differences', 'landmark rows: L1 of hit-count differences (64 rays)'),
      ('markhitsflip', 'top markhits l1 flip', 'same hit-count landmark rows, walls flipped (control)', None),
      ('intnl1', 'top intnorm l1 true', '+ 16 integral rows divided by their L1 norm over directions', 'integral / its L1 norm over directions'),
      ('intnlinf', 'top intnorm linf true', '+ 16 integral rows divided by their L∞ norm over directions', 'integral / its Linf norm over directions'),
      ('geo', 'geo true', '+ 16 corridor-distance rows (uses the maze graph)', 'corridor distance to 16 landmarks (geo, part 1)')]
LABEL.update({k: lab for _, k, lab, _ in P4 if k not in LABEL})


def part4_numbers(t, ab, me):
    N = {}
    sc, fz, hc = load('rayscreen.json'), load('frozenplan.json'), load('headcheck.json')
    for n in (22, 13):
        for kind in ('l1', 'linf'):
            for w, name in (('1.0', 'one'), ('0.2', 'fifth')):
                N[f'p4above_{kind}{name}{n}'] = f'{100 * hc[str(n)][kind]["above"][w]:.0f}' if hc[str(n)][kind]['above'][w] >= 0.01 else f'{100 * hc[str(n)][kind]["above"][w]:.1f}'
    z = lambda x, d=2: f'{0.0 if round(x, d) == 0 else x:.{d}f}'.replace('-', '−')
    for n in (22, 13):
        for name, key, _, scr in P4:
            if scr:
                N[f'p4scr_{name}{n}'], N[f'p4scr1_{name}{n}'] = f'{sc[str(n)][scr]["L2"]["sep"]:.2f}', f'{sc[str(n)][scr]["L1"]["sep"]:.2f}'
            v = {s: x for s, x in t[n].get(key, {}).items() if s in S}
            if not v:
                continue
            N[f'p4_{name}{n}'], N[f'p4k_{name}{n}'] = f'{pct(v):.1f}', f'{len(v)} seed' + ('s' if len(v) > 1 else '')
            N[f'p4per_{name}{n}'] = ' / '.join(f'{v[s]["ok"]}' if s in v else '–' for s in S)
            N[f'p4t_{name}{n}'] = z(float(np.mean([x['t_ratio'] for x in v.values()])))
            for other, oname in (('topzero', 'zero'), ('octave', 'oct'), ('top exact-integral true slow', 'exact'), ('top markint l1 true unit', 'markunit'), ('top markint l1 flip', 'markflip'),
                                 ('top hits flip', 'hitsflip'), ('geo true', 'geo'), ('top markint linf true', 'markinf'), ('top markhits l1 flip', 'markhitsflip')):
                d = R.paired(t, n, key, other) if other != key else []
                if d:
                    N[f'p4d_{name}{oname}{n}'], N[f'p4s_{name}{oname}{n}'] = signed(float(np.mean(d))), ' / '.join(signed(x) for x in d)
                    N[f'p4w_{name}{oname}{n}'], N[f'p4l_{name}{oname}{n}'] = f'{sum(x > 0 for x in d)} of {len(d)}', f'{sum(x < 0 for x in d)} of {len(d)}'
                    N[f'p4o_{name}{oname}{n}'] = f'{pct({s: x for s, x in t[n][other].items() if s in v}):.1f}'
            a = [ab[x['id']] for x in v.values() if x['id'] in ab and 'sep' in ab[x['id']]['intact']]
            m = [me[x['id']] for x in v.values() if x['id'] in me]
            if a:
                N[f'p4jump_{name}{n}'], N[f'p4sep_{name}{n}'], N[f'p4leak_{name}{n}'] = z(mean(a, 'intact', 'jump_median')), z(mean(a, 'intact', 'sep')), z(mean(a, 'intact', 'leak'))
                if all('frozen' in r for r in a):
                    N[f'p4fjump_{name}{n}'] = z(mean(a, 'frozen', 'jump_median'))
            if m:
                N[f'p4bumps_{name}{n}'], N[f'p4cos_{name}{n}'] = z(mean(m, 'bumps', 'bumps_per_path'), 1), z(mean(m, 'direction', 'cos_median'))
            w = [x['head_weight'] for x in v.values() if x.get('head_weight') is not None]
            if w:
                N[f'p4hw_{name}{n}'] = ' / '.join(f'{x:.4f}' for x in w)
            f = [(fz[x['id']]['frozen'][0], x['ok']) for x in v.values() if x['id'] in fz]
            if f and len(f) == len(v):                                                       # planning with the added rows pinned (frozenplan.py)
                N[f'p4fz_{name}{n}'] = f'{100 * (1 - sum(a for a, _ in f) / sum(b for _, b in f)):.0f}'
                N[f'p4fzok_{name}{n}'] = f'{100 * sum(a for a, _ in f) / sum(x["tot"] for x in v.values()):.1f}'
            N[f'p4ok_{name}{n}'] = str(sum(x['ok'] for x in v.values()))
            if 87358 in v:
                N[f'p4first_{name}{n}'] = str(v[87358]['ok'])
            N[f'p4conv_{name}{n}'] = f'{100 * np.mean([x["converged"] for x in v.values()]):.0f}'
    return N


def part4_table(N):
    g = lambda k: N.get(k, '–')
    rows = []
    for name, key, lab, _ in P4:
        if f'p4_{name}22' not in N:
            continue
        few = '' if g(f'p4k_{name}22') == '3 seeds' else f' <span class="mute">{g(f"p4k_{name}22")}</span>'
        c13 = f'{N[f"p4_{name}13"]} %' if f'p4_{name}13' in N else '–'
        rows.append(f'<tr><td>{html.escape(lab)}</td><td class="num">{g(f"p4_{name}22")} %{few}</td><td class="num">{g(f"p4per_{name}22")}</td><td class="num">{g(f"p4s_{name}zero22")}</td>'
                    f'<td class="num">{c13}</td><td class="num">{g(f"p4fz_{name}22") + " %" if f"p4fz_{name}22" in N else "–"}</td><td class="num">{g(f"p4jump_{name}22")}</td><td class="num">{g(f"p4bumps_{name}22")}</td><td class="num">{g(f"p4scr_{name}22")}</td></tr>')
    return '\n'.join(rows)


def part2_table(t, N):
    rows = []
    g = lambda k: N.get(k, '–')
    for name, key, ctrl, flip in P2:
        v22 = {s: x for s, x in t[22].get(key, {}).items() if s in S}
        if not v22:
            continue
        lab = dict(oct='128 octave rows', zero='120 octaves + 16 empty (control)', true='+ 16 speed rows, 1 − speed (first part)', geo='+ 16 distance rows (uses the maze graph)', half='64 octaves + 128 empty (control)').get(name, LABEL[key])
        per = ' / '.join(f'{v22[s]["ok"]}' if s in v22 else '–' for s in S)
        fl = f'{g(f"p2o_{name}flip22")} % <span class="mute">{g(f"p2s_{name}flip22")}</span>' if f'p2o_{name}flip22' in N else '–'
        few = lambda n: '' if g(f'p2k_{name}{n}') == '3 seeds' else f' <span class="mute">{g(f"p2k_{name}{n}")}</span>'
        c13 = f'{g(f"p2_{name}13")} %{few(13)}' if f'p2_{name}13' in N else '–'
        rows.append(f'<tr><td>{html.escape(lab)}</td><td class="num">{g(f"p2_{name}22")} %{few(22)}</td><td class="num">{per}</td><td class="num">{g(f"p2s_{name}ctrl22")}</td><td class="num">{fl}</td>'
                    f'<td class="num">{g(f"p2jump_{name}22")}</td><td class="num">{g(f"p2fjump_{name}22")}</td><td class="num">{g(f"p2sep_{name}22")}</td><td class="num">{g(f"p2bumps_{name}22")}</td></tr>')
    return '\n'.join(rows)


def run_table(t):
    rows = []
    for n in (22, 13):
        for key, v in sorted(t[n].items(), key=lambda kv: -pct({s: x for s, x in kv[1].items() if s in S} or kv[1])):
            v = {s: x for s, x in v.items() if s in S}
            if not v:
                continue
            per = ' / '.join(f'{v[s]["ok"]}' if s in v else '–' for s in S)
            run = ' '.join(f'<code>{v[s]["id"]}</code>' for s in S if s in v)
            lab = LABEL.get(key) or key.replace('top line ', '120 octaves + 16 speed rows, ').replace(' true', '').replace(' flip', ', flipped').replace('M', 'M = ')
            tot = next(iter(v.values()))['tot']
            rows.append(f'<tr><td class="num">{n}×{n}</td><td>{html.escape(lab)}</td><td class="num">{pct(v):.1f} %</td><td class="num">{per} <span class="mute">of {tot}</span></td><td class="num">{np.mean([x["speed_corr"] for x in v.values()]):.3f}</td><td class="num">{np.mean([x["t_ratio"] for x in v.values()]):.2f}</td><td class="runs">{run}</td></tr>')
    return '\n'.join(rows)


def mech_table(mech):
    rows = []
    f = lambda r, k, d=2: '–' if r.get(k) is None else f'{r[k]:.{d}f}'
    for n in (22, 13):
        for r in mech[n]:
            rows.append(f'<tr><td class="num">{n}×{n}</td><td>{html.escape(r["label"])}</td><td class="num">{r["success"]:.1f}</td><td class="num">{f(r, "leak")}</td><td class="num">{f(r, "rise")}</td><td class="num">{f(r, "frise")}</td><td class="num">{f(r, "jump")}</td><td class="num">{f(r, "fjump")}</td><td class="num">{f(r, "sep")}</td><td class="num">{f(r, "fsep")}</td><td class="num">{f(r, "bumps")}</td></tr>')
    return '\n'.join(rows)


def main():
    t = R.collect()
    fit, cap, ab, me, pr = load('fit.json'), load('capture.json'), load('ablate.json'), load('mechanism.json'), load('profile.json')
    mech, prof, capd = mech_data(t, ab, me), profile_data(t, pr), capture_data(t, cap)
    N = numbers(t, fit, mech, prof, capd, ab, cap, me)
    data = dict(success=success_data(t), sweep=sweep_data(t, ab), mech=mech, profile=prof, capture=capd, fit={n: fit[n]['line'] for n in fit})
    page = (HERE / 'page_template.html').read_text()
    for f in sorted((HERE / 'page_text').glob('*.html')):
        page = page.replace(f'<!--{f.stem.upper()}-->', f.read_text())
    page = page.replace('<!--RUNS-->', run_table(t)).replace('<!--MECHROWS-->', mech_table(mech)).replace('<!--P2ROWS-->', part2_table(t, N)).replace('<!--P3ROWS-->', part3_table(N)).replace('<!--P4ROWS-->', part4_table(N))
    page = page.replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
    missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', page)) - set(N))
    if missing:
        raise SystemExit(f'no value for {missing}')
    page = re.sub(r'\{\{(\w+)\}\}', lambda m: N[m.group(1)], page)
    (OUT / 'speed_fourier.html').write_text(page)
    (OUT / 'page_numbers.json').write_text(json.dumps(N, indent=0, sort_keys=True) + '\n')
    print('ok', len(page) // 1024, 'KB', len(N), 'numbers')


if __name__ == '__main__':
    main()
