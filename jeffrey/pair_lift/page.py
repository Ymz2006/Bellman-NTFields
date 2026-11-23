"""Findings page of the pair-lift study: report/pair_lift.html from report/runs.json (report.py), report/forensics.json (forensics.py),
report/jumps.json (pairjumps.py), jeffrey/lift_scale/report/runs.json (the reference arms on the other scenes) and the figures of figs.py.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/pair_lift/page.py

Chart code and style sheet are jeffrey/lift_scale/page_assets (the lift pages' own). Published as an artifact with report/*.png as img/<name>.png.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
LS = ROOT / 'jeffrey/lift_scale'
_spec = importlib.util.spec_from_file_location('lift_scale_pagelib', LS / 'pagelib.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
f, hbar, line, table, details, chart, figure, esc = L.f, L.hbar, L.line, L.table, L.details, L.chart, L.figure, L.esc

TITLE_TAG = '2026-11-19 Pair Lift'
# obstacle-space sampling: (group title, ((label, scene), name) of the free-space control, [((label, scene), name), ...])
INSIDE = [('22-cell maze, lift + backup', (('bxt1', 'plus22'), 'free-space pairs'),
           [(('bxt1_in', 'plus22in'), 'obstacle pairs'), (('bxt1_inF', 'plus22in'), 'backup on free pairs'), (('bxt1_inW', 'plus22in'), 'obstacle loss x 0.02')]),
          ('Apartment, lift + octaves + backup', (('rec', 'apt64'), 'free-space pairs'), [(('rec_in', 'apt64in'), 'obstacle pairs'), (('rec_inF', 'apt64in'), 'backup on free pairs')]),
          ('40-cell maze, lift + octaves + backup', (('bxt1nL_oct', 'plus40'), 'free-space pairs'), [(('bxt1nL_oct_in', 'plus40in'), 'obstacle pairs'), (('bxt1nL_oct_inF', 'plus40in'), 'backup on free pairs')])]
# Part 6. Models trained with pairs inside obstacles: W20 / W68 = flat-top lift on whole-interior pairs (20 % / 68 % of the pairs with an end inside),
# R20 = jeffrey/lift_scale's lift_in: the lift continued 0.3 cells into the walls, 20 % of the pairs with the second end inside
W20, W68, R20, LIFT_FREE = '20261119T203113Z-543f4eb9', '20261119T195829Z-e518c164', '20261117T075605Z-a19f8f34', '20261117T055249Z-d953394e'
SPEED_RUNS = [(LIFT_FREE, 'lift, free pairs', 3), (W20, 'flat top, 20 % inside', 4), (W68, 'flat top, 68 % inside', 5), (R20, 'into walls, 20 %', 6)]
R40, R40O = '20261119T221757Z-d6b6c73f', '20261119T224932Z-fa50bf49'          # the same model on the 40-cell maze (plus40in pairs), without / with octaves
SPEED_RUNS40 = [(R40, '40-cell maze: lift into walls, 20 % inside', 6), (R40O, '40-cell maze: the same with octaves', 6)]
SPEED_BINS = ['inside, deeper than 0.1 cells', 'inside, 0.0175 - 0.1 cells deep', 'skin at the floor speed', 'band, lower half', 'band, upper half', 'open space']
WMODEL = [('ref:plus22:lift', 'free-space pairs', 3), (('liftW', 'plus22w20'), 'flat top, 20 % inside', 4), (('liftW', 'plus22w'), 'flat top, 68 % inside', 5), ('ref:plus22in:lift_in', 'into walls, 20 %', 6)]
T1ARMS = [('ref:plus22:ntrl', 'ntrl-demo', 1), ('ref:plus22:lift', 'lift, no backup', 3), (('bxt1', 'plus22'), 'true speed + check', 4), (('bxt1_nofree', 'plus22'), 'true speed, no check', 4),
          (('bxt1_sR_scheck', 'plus22'), 'model speed + check', 6), (('bxt1_sR_scheck_tdall', 'plus22'), 'same, plain TD term', 6), (('bxt1_sR_soft', 'plus22'), 'model speed, no check', 6),
          (('bxt1_s20_scheck', 'plus22'), 'flat top 20 % + check', 5), (('bxt1_sA_soft', 'plus22'), 'flat 68 %, no check', 5)]
T1ARMS40 = [('ref:plus40:ntrl', 'ntrl-demo', 1), (('liftR_oct', 'plus40in'), 'model alone', 3),
            (('bxt1nL_oct', 'plus40'), 'true speed + check', 4), (('bxt1nL_oct_sR_scheck', 'plus40'), 'model speed + check', 6)]
T2ARMS = [(('bxt1', 'plus22'), 'free pairs + backup', 3), ('ref:plus22in:lift_in', 'model alone', 4), (('bxt1_inF', 'plus22in'), '+ backup, skip inside', 6),
          (('bxt1_inF_init', 'plus22in'), 'same, warm start', 6), (('bxt1_in_soft', 'plus22in'), 'backup all, no check', 6),
          (('bxt1_in', 'plus22in'), 'backup all + check', 5), (('liftW', 'plus22w20'), 'flat top 20 % alone', 4), (('liftW_bellF', 'plus22w20'), 'flat 20 % + backup', 6)]
SCENES = ['plus22', 'plus30', 'plus40', 'apt32', 'Allensville', 'apt64', 'Lakeville', 'Klickitat']
# arm label -> (name on the page, colour slot). Slots follow the lift pages: 1 ntrl-demo, 2 lift, 3 lift + push, 6 lift + push + octaves; 5 = pair input, 4 = Bellman backup
ARM = {
    'ntrl-demo': ('ntrl-demo', 1), 'lift': ('lift', 2), 'lift + push': ('lift + push', 3), 'lift + push + octaves': ('lift + push + octaves', 6),
    'lift + strong push + octaves': ('lift + octaves + constant strong push', 6),
    'bit': ('flag −1 / +1 alone', 5), 'bit_push': ('flag + push, no cap', 5), 'bit_push_tri': ('flag', 5),
    'ud_push_tri': ('length, saturated', 5), 'count_f025': ('wall count', 5),
    'udraw_f025': ('length, unsaturated', 5), 'len_push_tri': ('length excess, saturated', 5),
    'rows_ud': ('length in own Fourier rows', 5), 'same_len': ('length, same at both ends', 5),
    'post32': ('32 copies after the norm', 5), 'pre8': ('8 copies before the norm', 5), 'pre128': ('128 copies before the norm', 5),
    'bit_in_zero': ('flag, pairs inside obstacles', 5), 'lift_push_tri': ('no pair input (control)', 3),
    'bxt1': ('lift + Bellman backup', 4), 'bxt1L': ('lift + Bellman backup, long steps', 4), 'bxt3': ('weight 3', 4), 'bxt1_r15': ('step radius 0.33 cells', 4), 'bxt1_k8': ('8 steps, none around the gradient', 4),
    'bxt1_tdall': ('wall-crossing TD steps kept', 4), 'ntrl_bxt1': ('no lift', 1), 'tdfree': ('no backup, TD steps dropped', 2),
    'bxt1_count': ('+ wall count as input', 5), 'bxt1_bit': ('+ blocked bit as input', 5),
    'bell1': ('weight 1', 4), 'bell01': ('weight 0.1', 4), 'bell1_count': ('+ wall count as input', 5),
    'ntrl_bell1': ('no lift', 1),
    'bxt1K': ('long steps, short ones kept around the gradient', 4),
    'bxt1_pre8': ('+ 8 copies before the norm', 5), 'bxt1_pre32': ('+ 32 copies before the norm', 5), 'bxt1_post32': ('+ 32 copies after the norm', 5),
}
FIT = {'fit_lift': 'lift', 'fit_ntrl': 'ntrl-demo', 'fit_bit': 'lift + blocked bit', 'fit_count_f025': 'lift + wall count (frequency 0.25)', 'fit_count_f1': 'lift + wall count (frequency 1)',
       'fit_udraw_f025': 'lift + lifted length, unsaturated', 'fit_lenraw_box': 'lift + whole lifted length', 'fit_count_same': 'lift + wall count, same at both ends',
       'fit_ud_tanh': 'lift + lifted length, saturated', 'fit_vec': 'vector lift', 'fit_lift_oct8': 'lift + octaves',
       'fit_lift_30k': 'lift', 'fit_count_f1_30k': 'lift + wall count (frequency 1)', 'fit_liftB': 'lift', 'fit_countB_f1': 'lift + wall count (frequency 1)',
       'fit_countB_f025': 'lift + wall count (frequency 0.25)', 'fit_udrawB_f025': 'lift + lifted length, unsaturated', 'fit_lenrawB_box': 'lift + whole lifted length',
       'fit_bitB': 'lift + blocked bit', 'fit_lift_oct8B': 'lift + octaves', 'fit_ntrlB': 'ntrl-demo'}
FIT.update({f'fit_pre{k}B': f'{k} copies, before' for k in (2, 8, 32, 128)})
FIT.update({f'fit_post{k}B': f'{k} copies, after' for k in (2, 8, 32, 128)})
FIT.update({'fit_lift_octB': 'lift + octaves'})
COPY_FITS = [('no pair input', ['fit_liftB']), ('lifted length appended to the embedding before the instance norm', [f'fit_pre{k}B' for k in (2, 8, 32, 128)]),
             ('appended after the instance norm (read directly by the metric)', [f'fit_post{k}B' for k in (2, 8, 32, 128)])]
SHORT_FITS = ['fit_ntrl', 'fit_lift', 'fit_lift_oct8', 'fit_vec', 'fit_bit', 'fit_ud_tanh', 'fit_udraw_f025', 'fit_lenraw_box', 'fit_count_f025', 'fit_count_f1', 'fit_count_same']
LONG_FITS = ['fit_ntrlB', 'fit_liftB', 'fit_lift_30k', 'fit_lift_oct8B', 'fit_bitB', 'fit_udrawB_f025', 'fit_lenrawB_box', 'fit_countB_f025', 'fit_countB_f1', 'fit_count_f1_30k']


class Data:
    def __init__(self):
        d = json.load(open(OUT / 'runs.json'))
        self.runs, self.failed = d['runs'], d['failed']
        self.forensics = json.load(open(OUT / 'forensics.json')) if (OUT / 'forensics.json').exists() else {}
        self.jumps = json.load(open(OUT / 'jumps.json')) if (OUT / 'jumps.json').exists() else []
        self.ls = json.load(open(LS / 'report/runs.json'))['runs']
        self.speed = json.load(open(OUT / 'speed.json')) if (OUT / 'speed.json').exists() else {}

    def sel(self, label, scene='plus22', seed=None):
        return [r for r in self.runs if r['label'] == label and r['scene'] == scene and (seed is None or r['seed'] == seed)]

    def one(self, label, scene='plus22', seed=87358):
        rs = self.sel(label, scene, seed)
        return rs[0] if rs else None

    def pool(self, label, scene='plus22'):
        rs = [r for r in self.sel(label, scene) if 'success' in r]
        if not rs:
            return None
        out = dict(seeds=len(rs), solved=100 * sum(r['success'][0] for r in rs) / sum(r['success'][1] for r in rs), each=[r['solved'] for r in sorted(rs, key=lambda r: r['seed'])])
        for k in ('within10', 'ratio', 'los_within10', 'rho', 'seen_through', 'wall', 'stall'):
            v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            out[k] = float(np.mean(v)) if v else None
        return out

    def ref(self, scene, label):
        """Pooled jeffrey/lift_scale arm on a scene (its own labels: ntrl, lift, liftpush, liftpush_oct / liftpush_oct8)."""
        labels = {label} | ({'liftpush_oct8'} if (scene, label) == ('plus22', 'liftpush_oct') else set())
        rs = [r for r in self.ls if r['scene'] == scene and r['label'] in labels]
        if not rs:
            return None
        out = dict(seeds=len(rs), solved=100 * sum(r['success'][0] for r in rs) / sum(r['success'][1] for r in rs), each=[r['solved'] for r in sorted(rs, key=lambda r: r['seed'])])
        for k in ('within10', 'seen_through', 'wall', 'stall', 'rho', 'ratio'):
            v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            out[k] = float(np.mean(v)) if v else None
        return out

    def jump(self, rid, epoch=None):
        for j in self.jumps:
            if j['run'] == rid and j['epoch'] == epoch:
                return j

    def fx(self, section, rid, epoch=None):
        for r in self.forensics.get(section, []):
            if r.get('run') == rid and r.get('epoch') == epoch:
                return r


# the backup's variants on the scene charts (on the 22-cell maze the radius is 0.03 box lengths and the scaled weight is 1, so bxt1L is bxt1nL there)
BACKUPS = [('bxt1', 'backup, short steps', 'steps of 0.33–0.66 cells, weight 1'), ('bxt1n', 'backup, scaled weight', 'steps of 0.33–0.66 cells, weight × (0.03 / radius)²'),
           ('bxt1b', 'backup, box steps', 'steps of 0.03 box lengths, weight 1'), ('bxt1nL', 'backup, long steps', 'scaled weight, steps up to 5.3 cells'),
           ('bxt1nK', 'backup, long + short', 'scaled weight, steps up to 5.3 cells, short around the descent direction'),
           ('bxt1nLB', 'backup, box-long steps', 'scaled weight, steps up to 0.24 box lengths'),
           ('bxt1nL_oct', 'backup + octaves', 'lift + octaves, scaled weight, steps up to 5.3 cells'),
           ('rec', 'backup + octaves, box', 'lift + octaves, scaled weight, steps up to 0.24 box lengths')]
ALIAS = {'bxt1nL': 'bxt1L', 'bxt1nK': 'bxt1K'}    # on the 22-cell maze the radius is 0.03 box lengths and the scaled weight is 1


def seeds_note(p):
    return f"{p['seeds']} training seed{'s' if p['seeds'] > 1 else ''}" + (': ' + ' / '.join(f(v) for v in p['each']) if p['seeds'] > 1 else '')


GROUPS = [('References (jeffrey/lift_scale, same pipeline)', ['ntrl-demo', 'lift', 'lift + push']),
          ('Pair feature as a network input (with the push and the cap unless named)', ['bit', 'bit_push', 'bit_push_tri', 'bit_in_zero', 'ud_push_tri', 'len_push_tri', 'udraw_f025', 'count_f025', 'rows_ud', 'same_len', 'post32', 'pre8', 'pre128', 'lift_push_tri']),
          ('The lifted length as the cost of a straight step (Bellman backup)', ['bxt1', 'bxt1L'])]
ABLATION = [('The recipe and what each piece does', ['bxt1', 'bxt1L', 'bxt1K', 'bxt3', 'bxt1_r15', 'bxt1_k8', 'bxt1_tdall', 'tdfree', 'ntrl_bxt1']),
            ('A pair input on top of the recipe', ['bxt1_bit', 'bxt1_count', 'bxt1_pre8', 'bxt1_pre32', 'bxt1_post32']),
            ('First version: step time by a 6-point trapezoid rule', ['bell1', 'bell01', 'ntrl_bell1', 'bell1_count'])]


def build_charts(D):
    C = {}

    def groups(spec, key, scene_of=None):
        out = []
        for name, labels in spec:
            rows = []
            for lab in labels:
                scene = 'plus22in' if lab == 'bit_in_zero' else 'plus22'
                p = D.pool(lab, scene)
                if p and p.get(key) is not None:
                    rows.append((ARM[lab][0], p[key], ARM[lab][1], seeds_note(p)))
            if rows:
                out.append((name, rows))
        return out
    leg = [('ntrl-demo', 1), ('lift', 2), ('lift + push', 3), ('pair feature as input', 5), ('Bellman backup over free steps', 4)]
    sub = '22 × 22 plus maze. Share of MPPI plans (240 queries × planning seeds 0 / 1 / 2) that arrive with the exact clearance never negative. Training seed 87358 unless the tooltip lists more.'
    C['solved'] = hbar('Queries solved', sub, groups(GROUPS, 'solved'), legend=leg, value_name='solved')
    C['within'] = hbar('Travel times within 10 % of the truth', 'Random (free configuration, goal) pairs against the grid solver’s travel time. A bar near 0 is a field whose times are several times too long.',
                       groups(GROUPS, 'within10'), legend=leg, value_name='within 10 %')
    C['seen'] = hbar('Detour pairs seen through', 'Pairs whose true path is at least 30 % longer than the straight segment: share whose learned detour is under half the true one.',
                     groups(GROUPS, 'seen_through'), legend=leg, value_name='seen through')
    C['abl_solved'] = hbar('Queries solved: the backup, its pieces, and pair inputs on top of it', sub, groups(ABLATION, 'solved'),
                           legend=[('lift + backup', 4), ('no lift', 1), ('no backup', 2), ('with a pair input', 5)], value_name='solved')
    C['abl_within'] = hbar('Travel times within 10 %: same arms', 'The first version’s step time was too long near walls, and its travel times came out about 28 % long.', groups(ABLATION, 'within10'),
                           legend=[('lift + backup', 4), ('no lift', 1), ('no backup', 2), ('with a pair input', 5)], value_name='within 10 %')

    # the step across a wall
    ser = []
    for lab, name, slot in (('ntrl-demo', 'ntrl-demo', 1), ('lift', 'lift', 2), ('lift + push', 'lift + push', 3), ('bit_push_tri', 'blocked bit + push + cap', 5), ('bxt1', 'lift + Bellman backup', 4)):
        r = D.one(lab)
        j = D.jump(r['id']) if r else None
        if j:
            ser.append((name, slot, [(v['true'], v['learned']) for v in j['by_true'].values()], ''))
    if ser:
        ser.append(('truth', 6, [(0, 0), (16, 16)], '4 3'))
        C['jumps'] = line('The step in travel time a field holds across a wall', 'Mirror configurations on the two sides of a wall, 0.8 cells apart: the true difference of their travel times to a goal on the horizontal axis, the field’s on the vertical (medians by class of true step).',
                          ser, xlabel='true step across the wall (cells)', ylabel='learned step (cells)', xticks=[0, 2, 4, 6, 8, 10, 12, 14], yticks=[0, 4, 8, 12, 16], digits=2, xname='true step')

    # supervised fits
    def fit_rows(labels, key, scale=1.0, scene='plus22'):
        rows = []
        for lab in labels:
            r = D.one(lab, scene)
            if r and r.get(key) is not None:
                slot = 1 if 'ntrl' in lab else 6 if 'oct' in lab else 2 if FIT[lab] in ('lift', 'vector lift') else 5
                rows.append((FIT[lab] + (f" ({r['steps'] // 1000}k steps)" if lab in ('fit_lift_30k', 'fit_count_f1_30k') else ''), r[key] * scale, slot, f"{r['steps']:,} steps, run {r['id']}"))
        return rows
    fleg = [('ntrl-demo', 1), ('lift', 2), ('lift + octaves', 6), ('lift + a pair input', 5)]
    fg = [('6,000 steps, configurations uniform', SHORT_FITS), ('20,000 steps with 60 % of the configurations in the speed bands (30,000 uniform where marked)', LONG_FITS)]
    C['fit_jump'] = hbar('Fitted to the truth: share of the true step held across walls', 'Each network regressed on the true travel time of 600,000 (configuration, goal) pairs. Median of learned step / true step over mirror pairs whose true step is at least 3 cells. 1 = the step is fully represented.',
                         [(n, fit_rows(ls, 'jump_big')) for n, ls in fg], xmax=1.1, ticks=(0, 0.25, 0.5, 0.75, 1.0), digits=2, suffix='', value_name='step held', legend=fleg, unit=True)
    C['fit_solved'] = hbar('Fitted to the truth: queries solved', 'The same fields planned with ntrl-demo’s MPPI. Every one of them has at least 95 % of its travel times within 10 %.',
                           [(n, fit_rows(ls, 'solved')) for n, ls in fg], legend=fleg, value_name='solved')

    if any(D.one(l) for _, ls in COPY_FITS[1:] for l in ls):
        cleg = [('lift', 2), ('lift + copies of the lifted length', 5)]
        C['copies_jump'] = hbar('Copies of the lifted length on the embedding: share of the true step held across walls', 'Fitted to the truth, 20,000 steps, 60 % of the configurations in the speed bands. The feature is tanh of the up-and-down part of the lifted curve (cells), −ψ at one end and +ψ at the other.',
                                [(n, fit_rows(ls, 'jump_big')) for n, ls in COPY_FITS], xmax=1.1, ticks=(0, 0.25, 0.5, 0.75, 1.0), digits=2, suffix='', value_name='step held', legend=cleg, unit=True)
        C['copies_solved'] = hbar('Copies of the lifted length on the embedding: queries solved', 'The same fitted fields planned with ntrl-demo’s MPPI.', [(n, fit_rows(ls, 'solved')) for n, ls in COPY_FITS], legend=cleg, value_name='solved')
    sf = [(L.TITLE[sc], fit_rows(['fit_liftB', 'fit_lift_octB' if sc != 'plus22' else 'fit_lift_oct8B'], 'solved', scene=sc)) for sc in ('plus22', 'plus40', 'apt64')]
    if any(rows for n, rows in sf[1:]):
        C['fit_scenes'] = hbar('Fitted to the truth on the larger scenes: queries solved', 'The lifted network regressed on the true travel times (20,000 steps): what the network can plan when the loss is not the limit.', [g for g in sf if g[1]],
                               legend=[('lift', 2), ('lift + octaves', 6)], value_name='solved')
        C['fit_scenes_jump'] = hbar('Fitted to the truth on the larger scenes: share of the true step held across walls', 'Same fits.',
                                    [g for g in [(L.TITLE[sc], fit_rows(['fit_liftB', 'fit_lift_octB' if sc != 'plus22' else 'fit_lift_oct8B'], 'jump_big', scene=sc)) for sc in ('plus22', 'plus40', 'apt64')] if g[1]],
                                    xmax=1.1, ticks=(0, 0.25, 0.5, 0.75, 1.0), digits=2, suffix='', value_name='step held', legend=[('lift', 2), ('lift + octaves', 6)], unit=True)

    # cliffs in open space
    rows = []
    for lab in ('fit_liftB', 'fit_lift_30k', 'fit_bitB', 'fit_udrawB_f025', 'fit_lenrawB_box', 'fit_countB_f1', 'fit_count_f1_30k'):
        r = D.one(lab)
        c = D.fx('cliffs', r['id']) if r else None
        if c:
            slot = 2 if FIT[lab] == 'lift' else 5
            rows.append((FIT[lab] + (' (30k steps)' if lab.endswith('30k') else ''), 100 * c['changes']['over2'], slot, f"median |dT| / step {c['changes']['median']:.2f}, 90th percentile {c['changes']['q90']:.2f}; elsewhere {100 * c['same']['over2']:.2f} % above 2"))
    if rows:
        C['cliffs'] = hbar('Cliffs in open space where the wall count changes', 'Steps of 0.1 cells between configurations at least 0.2 cells from every obstacle, on the fields fitted to the truth: share of the steps across a line where the number of obstacles on the straight segment to the goal changes on which the travel time changes by more than twice the step (the truth never exceeds 1.1).',
                            [('', rows)], legend=[('lift', 2), ('lift + a pair input', 5)], value_name='steps with a cliff')

    # training course
    def course(lab, key, scale=1.0):
        r = D.one(lab)
        if not r or not r.get('course'):
            return []
        return [(int(e), (c.get(key) * scale if c.get(key) is not None else None)) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0]))]

    def course_solved(lab):
        r = D.one(lab)
        if not r or not r.get('course'):
            return []
        return [(int(e), 100 * c['success'][0] / c['success'][1]) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0])) if 'success' in c] + [(5000, r['solved'])]
    arms = [('lift + push', 'lift + push', 3), ('bell1', 'backup, trapezoid step time', 2), ('bxt1', 'backup, exact step time', 4), ('bxt1L', 'backup, long steps', 6)]
    xt = [0, 1000, 2000, 3000, 4000, 5000]
    C['course_solved'] = line('Queries solved during training', 'Planning seed 0 at epochs 1,000 / 2,000 / 3,000 and the final field (all three planning seeds).', [(n, s, course_solved(l), '') for l, n, s in arms],
                              xlabel='epoch', ylabel='solved %', xticks=xt, yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, digits=1, xname='epoch', height=260)
    C['course_los'] = line('Line-of-sight travel time over the truth', 'Median learned / true travel time on pairs with a free straight path. 1 = right; the trapezoid step time settles 20–30 % long.',
                           [(n, s, course(l, 'los_ratio'), '') for l, n, s in arms], xlabel='epoch', ylabel='learned / true', xticks=xt, yticks=[0.9, 1.0, 1.1, 1.2, 1.3], refs=[(1.0, 'truth')], digits=3, xname='epoch', height=260)
    C['course_seen'] = line('Detour pairs seen through during training', '', [(n, s, course(l, 'seen_through', 100), '') for l, n, s in arms], xlabel='epoch', ylabel='seen through %', xticks=xt,
                            yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, digits=1, xname='epoch', height=260)

    # other scenes
    def scene_groups(key):
        out = []
        for s in SCENES:
            rows = []
            for lab, name, slot in (('ntrl', 'ntrl-demo', 1), ('liftpush', 'lift + push', 3), ('liftpush_oct', 'lift + push + octaves', 6)):
                p = D.ref(s, lab)
                if p and p.get(key) is not None:
                    rows.append((name, p[key], slot, seeds_note(p)))
            got = False
            for lab, name, _ in BACKUPS:
                p = D.pool(lab, s) or (D.pool(ALIAS[lab], s) if lab in ALIAS else None)
                if p and p.get(key) is not None:
                    rows.append((name, p[key], 4, seeds_note(p)))
                    got = True
            if got:
                out.append((L.TITLE[s], rows))
        return out
    sleg = [('ntrl-demo', 1), ('lift + push', 3), ('lift + push + octaves', 6), ('lift + Bellman backup (variant on the label)', 4)]
    C['scenes_solved'] = hbar('Queries solved on every scene', 'Yellow bars: lift + Bellman backup, one bar per variant that was run on the scene. Reference arms from the lift-scale study (pooled over its training seeds).',
                              scene_groups('solved'), legend=sleg, value_name='solved')
    C['scenes_within'] = hbar('Travel times within 10 % on every scene', 'Same arms.', scene_groups('within10'), legend=sleg, value_name='within 10 %')
    C['scenes_wall'] = hbar('Plans that arrive through an obstacle', 'Share of plans that reach the goal with negative clearance somewhere on the path.', scene_groups('wall'), legend=sleg, value_name='through a wall')
    # obstacle-space sampling: each run next to the same recipe on free-space pairs
    def inside_groups(key):
        out = []
        for title, free, runs in INSIDE:
            rows = []
            p = D.pool(*free[0])
            if p and p.get(key) is not None:
                rows.append((free[1], p[key], 1, seeds_note(p)))
            for (lab, sc), name in runs:
                p = D.pool(lab, sc)
                if p and p.get(key) is not None:
                    rows.append((name, p[key], 4, seeds_note(p)))
            if len(rows) > 1:
                out.append((title, rows))
        return out
    ileg = [('free-space pairs only', 1), ('20 % of the pairs with one end inside an obstacle', 4)]
    if inside_groups('solved'):
        C['inside_solved'] = hbar('Queries solved with obstacle-space samples', 'Same recipe and scene; grey: the training pairs of every other run (both ends in free space).', inside_groups('solved'), legend=ileg, value_name='solved')
        C['inside_seen'] = hbar('Detours seen through with obstacle-space samples', 'Share of the pairs with a true detour whose learned time is close to the straight-line time.', inside_groups('seen_through'), legend=ileg, value_name='seen through')
    # Part 6: the flat-top lift on whole-interior pairs (the Lift Variants page's model), its speed as the backup's step cost, the backup on it
    def arm_groups(key, arms, title='22-cell maze'):
        rows = []
        for src, name, slot in arms:
            p = D.ref(*src[4:].split(':')) if isinstance(src, str) and src.startswith('ref:') else D.pool(*src)
            if p and p.get(key) is not None:
                rows.append((name, p[key], slot, seeds_note(p)))
        return [(title, rows)] if rows else []
    if D.pool('liftW', 'plus22w') or D.pool('liftW', 'plus22w20'):
        wleg = [('free-space pairs', 3), ('flat-top lift, pairs inside obstacles', 4), ('the same, 68 % of the pairs', 5), ('lift continued into the walls, 20 % of the pairs', 6)]
        C['w_solved'] = hbar('The lift alone with pairs inside obstacles: queries solved', 'No push, no backup. The share of pairs with an end inside an obstacle is on the label.',
                             arm_groups('solved', WMODEL), legend=wleg, value_name='solved')
        C['w_seen'] = hbar('The lift alone with pairs inside obstacles: detours seen through', 'Share of the pairs with a true detour whose learned time is close to the straight-line time.', arm_groups('seen_through', WMODEL), legend=wleg, value_name='seen through')
        C['w_within'] = hbar('The lift alone with pairs inside obstacles: travel times within 10 %', 'Random (free configuration, goal) pairs against the grid solver’s travel time.', arm_groups('within10', WMODEL), legend=wleg, value_name='within 10 %')
    if D.speed:
        groups = []
        for b in SPEED_BINS:
            rows = [(name, D.speed[rid]['point'][b]['fit'][2], slot, '') for rid, name, slot in SPEED_RUNS if rid in D.speed and b in D.speed[rid]['point']]
            if rows:
                groups.append((b, rows))
        C['w_speed'] = hbar('Predicted speed against the true speed, by clearance', 'Median of S |∇T| at random configurations (1 = the predicted speed is exact; below 1 the field is too fast there). Partner: a random free configuration.',
                            groups, xmax=1.5, ticks=(0, 0.5, 1, 1.5), digits=2, suffix='', value_name='S |∇T|', legend=[(n, sl) for _, n, sl in SPEED_RUNS if _ in D.speed])
    t1leg = [('ntrl-demo', 1), ('lift, no backup', 3), ('backup, step cost from the true speed (lifted length)', 4), ('backup, step cost from the speed of the lift-into-walls model', 6), ('backup, step cost from a flat-top model’s speed', 5)]
    if any(D.pool(*src) for src, _, _ in T1ARMS[4:]):
        C['t1_solved'] = hbar('Idea 1: queries solved', 'Free-space pairs, same recipe; only the cost of a candidate step and the check of the steps change. “check”: geometric for the true speed, made from the same speed table for a model’s speed.', arm_groups('solved', T1ARMS), legend=t1leg, value_name='solved')
        C['t1_seen'] = hbar('Idea 1: detours seen through', 'Same arms.', arm_groups('seen_through', T1ARMS), legend=t1leg, value_name='seen through')
        C['t1_within'] = hbar('Idea 1: travel times within 10 %', 'Same arms.', arm_groups('within10', T1ARMS), legend=t1leg, value_name='within 10 %')
        C['t1_stall'] = hbar('Idea 1: plans that stall', 'Share of plans that do not reach the goal.', arm_groups('stall', T1ARMS), legend=t1leg, value_name='stalled')
    if D.pool('bxt1nL_oct_sR_scheck', 'plus40'):
        t40 = [('ntrl-demo', 1), ('the frozen model alone', 3), ('backup, step cost from the true speed', 4), ('backup, step cost and check from the model’s speed', 6)]
        C['t1_40_solved'] = hbar('Idea 1 on the 40-cell maze: queries solved', 'Lift + octaves + backup with long steps (the recipe of Part 4); only the step cost and the check change.', arm_groups('solved', T1ARMS40, '40-cell maze'), legend=t40, value_name='solved')
        C['t1_40_seen'] = hbar('Idea 1 on the 40-cell maze: detours seen through', 'Same arms.', arm_groups('seen_through', T1ARMS40, '40-cell maze'), legend=t40, value_name='seen through')
    t2leg = [('free pairs + backup', 3), ('pairs inside obstacles, no backup', 4), ('+ backup (skipped inside obstacles, or everywhere without a collision check)', 6), ('+ backup everywhere with the collision check', 5)]
    if D.pool('bxt1_inF', 'plus22in'):
        C['t2_solved'] = hbar('Idea 2: queries solved', 'The backup added to the training of a model with pairs inside obstacles. “skip inside”: no backup for a pair with an end inside an obstacle.', arm_groups('solved', T2ARMS), legend=t2leg, value_name='solved')
        C['t2_seen'] = hbar('Idea 2: detours seen through', 'Same arms.', arm_groups('seen_through', T2ARMS), legend=t2leg, value_name='seen through')
        C['t2_within'] = hbar('Idea 2: travel times within 10 %', 'Same arms.', arm_groups('within10', T2ARMS), legend=t2leg, value_name='within 10 %')
        C['t2_stall'] = hbar('Idea 2: plans that stall', 'Share of plans that do not reach the goal.', arm_groups('stall', T2ARMS), legend=t2leg, value_name='stalled')
    return {k: v for k, v in C.items() if v}


def main():
    import pagetext as T
    D = Data()
    C = build_charts(D)
    body = T.body(D, C)
    css = (LS / 'page_assets/style.css').read_text()
    js = (LS / 'page_assets/charts.js').read_text()
    page = (f'<title>{TITLE_TAG}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">\n'
            f'<style>\n{css}\n{T.EXTRA_CSS}</style>\n<div class="wrap">\n{body}\n</div>\n<div class="tip" id="tip" hidden></div>\n'
            f'<script id="data" type="application/json">{json.dumps(dict(charts=C))}</script>\n<script>\n{js}\n</script>\n')
    (OUT / 'pair_lift.html').write_text(page)
    print(f'{OUT / "pair_lift.html"}: {len(page) / 1024:.0f} KB, {len(C)} charts')


if __name__ == '__main__':
    main()
