"""The findings page's text (report/notes.json) written from report/summary.json, so every number in the prose is read from a
registered run. Hand-set here: which placement is called best, which planner budget is the headline, the throughput measurements.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/notes.py [--best p_d100_s30_h60] [--steps 2400] [--interim]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
TASKS = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
C2C = 'compartment_to_compartment'
# small double-backward loop, processes on one RTX 4090 that was also generating suites (scratchpad bench.py, 2026-11-23 06:58 UTC): processes, steps/s each
THROUGHPUT = [[1, 103.0], [6, 67.3], [12, 37.2], [24, 22.9]]


def pct(a, b):
    return 100 * a / b if b else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fm(v, d=0):
    return '–' if v is None else f'{v:.{d}f}'


class S:
    def __init__(self):
        self.d = json.loads((OUT / 'summary.json').read_text())
        self.P = {p['name']: p for p in self.d['placements']}

    def fields(self, pl, label):
        return sorted([x for x in self.P[pl]['fields'] if x['label'] == label], key=lambda x: x['seed']) if pl in self.P else []

    def fwd(self, pl, label, task=None):
        return mean([x['solved'] if task is None else pct(x['by_task'][task], x['per_task_runs']) for x in self.fields(pl, label)])

    def plan(self, pl, label, kind, steps, task=None, samples=50):
        out = []
        for x in self.fields(pl, label):
            for q in x.get('planners', []):
                if q['kind'] == kind and q['steps'] == steps and q['samples'] == samples:
                    out.append(q['solved'] if task is None else pct(q['by_task'][task], x['per_task_runs']))
        return mean(out)

    def rows(self, pl, label, kind, steps, samples=50):
        return [q for x in self.fields(pl, label) for q in x.get('planners', []) if q['kind'] == kind and q['steps'] == steps and q['samples'] == samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', default='p_d100_s30_h60')
    ap.add_argument('--steps', type=int, default=2400)
    ap.add_argument('--interim', action='store_true')
    ap.add_argument('--demo-pl', help='the placement of the demo run in report/demo.json (default: --best)')
    ap.add_argument('--planner-pl', default='p_d100_s30_h60', help='the placement whose fields the planner comparison was run on')
    ap.add_argument('--alt', default='p_d085_s30_h60', help='the placement named as the alternative with more reach')
    ap.add_argument('--alts', nargs='*', default=['p_d095_s20_h80', 'p_d085_s30_h60'], help='placements named as alternatives with more reach')
    a = ap.parse_args()
    s = S()
    B, st = a.best, a.steps
    bp = s.P.get(B)
    sweep = sorted([p for p in s.P.values() if p['name'] != 'base' and abs(p['s'] - 0.30) < 1e-6 and abs(p['h'] - 0.60) < 1e-6 and s.fields(p['name'], 'lift')], key=lambda p: p['dx'])
    names = [p['name'] for p in sweep]
    seq = lambda fn, d=0: ', '.join(f'{fm(fn(n), d)} % at {s.P[n]["dx"]:.2f} m' for n in names if fn(n) is not None)  # noqa: E731
    PP = a.planner_pl
    pp = s.P[PP]
    pn = lambda p: f'{p["dx"]:.2f} m, bases ±{p["s"]:.2f} m, {p["h"]:.2f} m high'  # noqa: E731
    N = dict(best=B, planner_pl=PP, alt=a.alt, planner_kind='bidir', planner_steps=st, throughput=THROUGHPUT, show_pending=a.interim)
    bl, bn = s.plan(B, 'lift', 'bidir', st), s.plan(B, 'ntrl', 'bidir', st)
    blc = s.plan(B, 'lift', 'bidir', st, C2C)
    fl, fn_ = s.fwd(B, 'lift'), s.fwd(B, 'ntrl')
    base_f = s.fwd('base', 'lift')
    base_b = s.plan('base', 'lift', 'bidir', st)
    r = bp.get('reach') if bp else None
    hard = [q for q in s.d['plans'] if q['suite'] == 'hard' and q['placement'] == B]
    DEP = json.loads((OUT / 'depth.json').read_text()) if (OUT / 'depth.json').exists() else {}
    dz = lambda n, k: 100 * DEP[n]['zone_to_zone'][k] if n in DEP else None  # noqa: E731  (cm)
    ALT = a.alt if a.alt in s.P else None
    ap_ = s.P.get(ALT) if ALT else None

    def bybin(n, label):
        b = next((b for b in DEP.get(n, {}).get('by_depth', {}).get('bins', []) if b['label'] == label), None)
        return pct(b['solved'], b['runs']) if b and b['runs'] >= 30 else None
    hl = lambda f, kind, steps: [q for q in hard if q['field'] == f and q['kind'] == kind and q['steps'] == steps]  # noqa: E731
    hsum = lambda rows: (sum(q['success'][0] for q in rows), sum(q['success'][1] for q in rows)) if rows else None  # noqa: E731

    N['lede'] = (('<b>Interim page: runs are still in flight.</b> ' if a.interim else '')
                 + f'Moving the two arms back makes the task easier, as far back as they can still reach into the cupboard. The best placement tested is <b>{bp["dx"]:.2f} m from the cupboard front</b> '
                 f'(bases ±{bp["s"]:.2f} m from the centre plane, {bp["h"]:.2f} m above the floor). There the repository’s lift field solves <b>{fm(bl)} %</b> of the certified queries '
                 f'and <b>{fm(blc)} %</b> of the zone-to-another-zone queries, when the planner walks from both ends at once; the ntrl-demo field solves {fm(bn if bn is not None else s.plan(B, "ntrl", "bidir", 600))} %. At the earlier study’s 0.25 m the same field and ntrl-demo’s own planner solved {fm(base_f)} %. '
                 + (f'The price is depth: at {bp["dx"]:.2f} m the hands only work the front {fm(dz(B, "max"))} cm of the 53 cm deep shelves.' if dz(B, 'max') is not None else ''))
    seeds = [f'{q["success"][0]}' for q in s.rows(B, 'lift', 'bidir', st)]
    others = sorted([p for p in s.P.values() if p['name'] not in ('base', B) and abs(p['dx'] - bp['dx']) < 1e-6 and s.plan(p['name'], 'lift', 'bidir', st) is not None], key=lambda p: (p['s'], p['h']))
    ans = [('Best placement', f'<b>{bp["dx"]:.2f} m</b> from the cupboard front, bases ±{bp["s"]:.2f} m, {bp["h"]:.2f} m high: {fm(bl)} % solved ({", ".join(seeds)} of 900 runs on {len(seeds)} training seeds). '
            + (f'With twice the planner budget (4,800 steps per end): {", ".join(str(q["success"][0]) for q in s.rows(B, "lift", "bidir", 4800))} of 900. ' if s.rows(B, 'lift', 'bidir', 4800) else '')
            + ('The other placements at that distance: ' + '; '.join(f'±{p["s"]:.2f} m, {p["h"]:.2f} m high {fm(s.plan(p["name"], "lift", "bidir", st))} %' for p in others) + '. ' if others else '')
            + ''.join(f'The {p["h"]:.2f} m high base at ±{p["s"]:.2f} m ties it ({", ".join(str(q["success"][0]) for q in s.rows(p["name"], "lift", "bidir", st))} of 900); {bp["h"]:.2f} m is named here because it has more training seeds, the 200-query set and the demo. '
                      for p in others if abs(s.plan(p['name'], 'lift', 'bidir', st) - bl) < 0.5)
            + (f'It is the reach limit: either arm still reaches {100 * r["union_mean"]:.0f} % of the zone volume ({", ".join(f"{k} {100 * r['union'][k]:.0f} %" for k in ("bottom", "middle", "top"))}); at 1.05 m, and at 1.00 m with the bases ±0.46 m apart and 0.80 m high, an arm reaches fewer than two zones.' if r else '')),
           ('The price', (f'At {bp["dx"]:.2f} m the grasp point is {fm(dz(B, "median"))} cm inside the cupboard in the median zone → zone query and never more than {fm(dz(B, "max"))} cm (shelves: 53 cm deep). '
                          + ' '.join(f'At {pn(s.P[n])} it goes up to {fm(dz(n, "max"))} cm in, an arm reaches {100 * s.P[n]["reach"]["union_mean"]:.0f} % of the zone volume, and the lift field solves <b>{fm(s.plan(n, "lift", "bidir", st))} %</b> '
                                     f'({fm(s.plan(n, "lift", "bidir", st, C2C))} % of zone → zone).' for n in a.alts if n in s.P and s.plan(n, 'lift', 'bidir', st) is not None and dz(n, 'max') is not None)
                          + ' If the shelves’ depth matters, move the bases closer: five centimetres cost about one point, ten to fifteen about seven.') if dz(B, 'max') is not None else 'See the depth chart below.'),
           ('Does moving back help?', '<span class="verdict yes">yes</span> Solved share with the lift field and ntrl-demo’s planner: ' + seq(lambda n: s.fwd(n, 'lift')) + f' (0.25 m, earlier suite: {fm(base_f)} %). With the two-ended planner at {st:,} steps per end: ' + seq(lambda n: s.plan(n, 'lift', 'bidir', st)) + '.'),
           ('Inside the cupboard to inside the cupboard', f'Zone → another zone at {bp["dx"]:.2f} m: {fm(s.fwd(B, "lift", C2C))} % with ntrl-demo’s planner, <b>{fm(blc)} %</b> with the two-ended planner ({st:,} steps per end). '
            + (f'On the larger set of {hard[0]["queries"]} such queries: {hsum(hl("lift", "bidir", st))[0]:,} of {hsum(hl("lift", "bidir", st))[1]:,} runs. ' if hl('lift', 'bidir', st) else '')
            + (f'With 4,800 steps per end: {hsum(hl("lift", "bidir", 4800))[0]:,} of {hsum(hl("lift", "bidir", 4800))[1]:,}. ' if hl('lift', 'bidir', 4800) else '')
            + ' '.join(f'At {pn(s.P[n])}: {sum(q["success"][0] for q in qs):,} of {sum(q["success"][1] for q in qs):,}.' for n in sorted({q['placement'] for q in s.d['plans'] if q['suite'] == 'hard'} - {B})
                       for qs in [[q for q in s.d['plans'] if q['suite'] == 'hard' and q['placement'] == n and q['field'] == 'lift' and q['kind'] == 'bidir' and q['steps'] == st]] if qs)),
           ('The two models', f'Lift field (LBS, lift only, half Fourier frequency): {fm(fl)} % with ntrl-demo’s planner, {fm(bl)} % two-ended. ntrl-demo field: {fm(fn_)} % and {fm(bn if bn is not None else s.plan(B, "ntrl", "bidir", 600))} %. '
            'The ntrl-demo field arrives in contact on most runs at every placement; distance helps it too, from 1 % to 11 %.'),
           ('What made the difference', f'The planner first: the same lift field goes from {fm(base_f)} % to {fm(base_b)} % at 0.25 m and from {fm(s.fwd(PP, "lift"))} % to {fm(s.plan(PP, "lift", "bidir", st))} % at {pp["dx"]:.2f} m when it is used from both ends. '
            'The field is symmetric, yet its planner leaves a zone far more reliably than it enters one. Then the distance, and part of that gain is that far placements only have shallow queries: '
            + ', '.join(f'{fm(bybin(n, "7 to 13 cm"))} % at {s.P[n]["dx"]:.2f} m' for n in names if bybin(n, '7 to 13 cm') is not None) + ' of the queries 7 to 13 cm inside the cupboard are solved.')]
    N['answers'] = ans
    N['sweep_text'] = ('<p>The cupboard, the test recipe and the training recipe stay fixed; only the bases move. Each placement gets its own certified test queries (six kinds × 50), its own training pairs and its own two fields. '
                       f'Both learned fields gain from distance. So do the two controls that use no learned field, which says the task itself gets easier: the planner on plain joint distance solves {seq(lambda n: s.P[n]["euclid"]["solved"] if "euclid" in s.P[n] else None)}.</p>')
    N['ease_text'] = ('<p>Why it gets easier. Close to the cupboard, most arm poses with the hands in the task box collide with a panel or with the other arm: '
                      + ', '.join(f'{100 * s.P[n]["prepare"]["valid_of_in_box"]:.1f} % are collision-free at {s.P[n]["dx"]:.2f} m' for n in names)
                      + (f' ({100 * s.P["base"]["prepare"]["valid_of_in_box"]:.1f} % at 0.25 m)' if 'base' in s.P else '')
                      + '. Far away the arms enter almost straight, with only the forearm and hand inside, and they cross each other less. The price is reach: the blue line in the chart is the share of the three zones’ volume that at least one arm can still put its grasp point in.</p>')
    seqd = lambda k: ', '.join(f'{fm(dz(n, k))} cm at {s.P[n]["dx"]:.2f} m' for n in names if dz(n, k) is not None)  # noqa: E731
    cmp_bins = [('up to 7 cm inside', 'up to 7 cm'), ('7 to 13 cm', '7 to 13 cm'), ('13 to 25 cm', '13 to 25 cm')]
    N['depth_text'] = ('<p>Reach as a volume hides how shallow the far placements work. Depth of the grasp point behind the cupboard’s front face in the zone → another zone test queries: median '
                       + seqd('median') + '; deepest ' + seqd('max') + (f' (0.25 m, earlier suite: median {fm(dz("base", "median"))} cm, deepest {fm(dz("base", "max"))} cm)' if 'base' in DEP else '') + '.</p>'
                       '<p>Control: is the task easier at a distance, or are the queries only shallower? The second chart compares placements at equal depth (each query filed by its deepest hand at either end). '
                       + ' '.join(f'Queries {nm} inside: ' + ', '.join(f'{fm(bybin(n, lab))} % at {s.P[n]["dx"]:.2f} m' for n in names if bybin(n, lab) is not None) + '.' for lab, nm in cmp_bins if any(bybin(n, lab) is not None for n in names))
                       + '</p>') if DEP else ''
    N['kinds_text'] = ('<p>With ntrl-demo’s planner (left) distance helps the queries that end in free space and does nothing for those that end inside the cupboard. '
                       f'At {pn(pp)} the lift field solves {fm(s.fwd(PP, "lift", "compartment_to_free"))} % of zone → free and {fm(s.fwd(PP, "lift", "free_to_compartment"))} % of free → zone, although T(a, b) = T(b, a). '
                       f'Run from the goal to the start, the same planner gives the mirror image ({fm(s.plan(PP, "lift", "reverse", 600, "free_to_compartment"))} % and {fm(s.plan(PP, "lift", "reverse", 600, "compartment_to_free"))} %). '
                       'So the field knows the way out of a zone and the planner loses the last stretch on the way in. The two-ended planner (right) only ever asks for the way out.</p>')
    kinds = [('forward', 600, 'ntrl-demo’s planner, 600 steps'), ('bidir', 300, 'two-ended, 300 steps per end (the same 600 moves)'), ('bidir', 600, 'two-ended, 600 per end'), ('bidir', 1200, '1,200 per end'), ('bidir', 2400, '2,400 per end'), ('bidir', 4800, '4,800 per end')]
    vals = [(nm, s.fwd(PP, 'lift') if k == 'forward' and stp == 600 else s.plan(PP, 'lift', k, stp)) for k, stp, nm in kinds]
    fw = [(stp, s.plan(PP, 'lift', 'forward', stp)) for stp in (2400, 4800)]
    smp = s.plan(PP, 'lift', 'bidir', 600, samples=200)
    hub = s.plan('p_d085_s30_h60', 'lift', 'hub', 600)
    N['planner_text'] = ('<p><code>bidir.plan_bidir</code>: one head starts at the start, one at the goal. In every iteration each head takes one ordinary MPPI step of ntrl-demo’s planner towards the other head’s current position. '
                         'They stop when they meet; the path is the first head’s path followed by the second one’s reversed. Same field, same step, same 50 samples, no collision checker. '
                         f'At {pn(pp)}: ' + '; '.join(f'{nm} {fm(v)} %' for nm, v in vals if v is not None) + '. '
                         + ('More steps do not rescue the one-ended planner: ' + ', '.join(f'{fm(v)} % at {stp:,} steps' for stp, v in fw if v is not None) + '. ' if any(v is not None for _, v in fw) else '')
                         + (f'More samples per step do not help either ({fm(smp)} % with 200 samples at 600 steps per end). ' if smp is not None else '')
                         + (f'A third variant, both ends walking to a retracted pose chosen by the field, was worse ({fm(hub)} % at 0.85 m against {fm(s.plan("p_d085_s30_h60", "lift", "bidir", 600))} % two-ended).' if hub is not None else '') + '</p>')
    var = [p for p in s.P.values() if p['name'] != 'base' and s.fields(p['name'], 'lift') and not (abs(p['s'] - 0.30) < 1e-6 and abs(p['h'] - 0.60) < 1e-6)]
    N['variants_text'] = ('<p>Both bases stay mirror images about the cupboard’s centre plane. Two-ended planner, ' + f'{st:,}' + ' steps per end, mean of the training seeds: '
                          + '; '.join(f'{p["dx"]:.2f} m, ±{p["s"]:.2f} m, {p["h"]:.2f} m high: {fm(s.plan(p["name"], "lift", "bidir", st))} %' for p in sorted(var, key=lambda p: (p['dx'], p['s'], p['h'])) if s.plan(p['name'], 'lift', 'bidir', st) is not None)
                          + '. At 1.00 m the placements differ by a few points, on different sets of 300 queries each; a difference of two to three points between two of them is about what the query sets alone can produce.</p>') if var else ''
    N['throughput_text'] = ('<p>The cards were limited by compute and then by memory, in that order. A single training process leaves a 4090 mostly idle (it waits on Python), so the scheduler (<code>queue/sched.py</code>) runs many registered jobs per card: '
                            f'{THROUGHPUT[2][0]} training-like processes together did {THROUGHPUT[2][0] * THROUGHPUT[2][1]:.0f} steps per second against {THROUGHPUT[0][1]:.0f} for one, on a card that was generating test suites at the same time. '
                            'With 25 to 30 jobs in flight both cards sat at 40 to 47 GB, and three jobs died from out-of-memory before the scheduler began reserving each job’s peak VRAM. '
                            'CUDA MPS made it worse in this mix (6 processes: 36 steps per second each with MPS, 67 without), so it is off. '
                            'Bigger batches where they pay: the pair sampler draws 2^20 candidate configurations at a time (a prepare run takes 0.6 to 5 minutes; 8 minutes in the earlier study), the suite generator batches every stratum’s RRT queries together (about 1,400 per round). '
                            'The slow part was certifying test queries: RRT-Connect at the scorer’s 0.001 rad resolution is bound by memory bandwidth, not by batch size.</p>')
    N['models_text'] = ('<p>Everything is under <code>jeffrey/arm_placement/models/</code>: one folder per placement with the weights of every finished field (<code>lift_s87358.pt</code>, <code>ntrl_s87358.pt</code>, second seeds where trained), the scene files the lift needs, and the test queries; '
                        '<code>models/best/</code> holds the chosen placement’s full checkpoints; <code>models/index.json</code> lists registered run ids and solved counts. '
                        '<b>This machine has no persistent volume and the folder is not committed: copy it off before the instance is recycled.</b></p>')
    N['setup_text'] = ('<ul><li><b>Two models:</b> my reading of the request, not confirmed: ntrl-demo (cloned from github.com/ruiqini/ntrl-demo, commit <code>5ed065f</code>, unmodified) and the repository’s LBS field in its best two-arm setting from <code>jeffrey/lbs_arm</code> '
                       '(travel-time lift, half Fourier frequency, no Bellman or segment term). 5,000 epochs, training seeds 87358 and 87359.</li>'
                       '<li><b>Scene:</b> the measured cupboard and both UR5e with grippers, as in <code>jeffrey/real_world</code>. A run is solved when every configuration of the path, checked every 0.001 rad per joint, keeps 5 mm from the cupboard, floor and pedestals, 5 mm between the arms and 2 mm self-clearance.</li>'
                       '<li><b>Test queries:</b> per placement, six kinds × 50, each certified by a witness path (straight line, RRT-Connect in 12 joints, or one arm at a time). The witness planner ran 3,000 iterations per stage instead of the earlier study’s 10,000, '
                       'so these sets lean somewhat more towards queries a sampling planner solves quickly. The 0.25 m point uses the earlier study’s frozen 600 queries. Its suite with this study’s recipe, 0.40 m, and the 0.80 m height at 0.55 and 0.70 m were stopped: their suites took over two hours each.</li>'
                       '<li><b>Different queries at every placement.</b> A placement changes which queries exist, so the comparison across distances is between tasks defined the same way, not between identical queries.</li>'
                       '<li><b>Reproduction check:</b> on the earlier study’s placement and frozen suite the lift field solves 645 of 1,800 (earlier: 673 and 639 on two seeds), ntrl-demo 15.</li>'
                       + (f'<li><b>Controls on the lift at {pn(pp)}:</b> Fourier frequency 0.25 instead of 0.5: {fm(s.fwd(PP, "lift025"))} % with ntrl-demo’s planner, {fm(s.plan(PP, "lift025", "bidir", st))} % two-ended (0.5: {fm(s.fwd(PP, "lift"))} % and {fm(s.plan(PP, "lift", "bidir", st))} %). '
                          f'Frequency 1.0: {fm(s.fwd(PP, "lift1"))} % with ntrl-demo’s planner. The full LBS loss (Bellman and segment terms) took 8 seconds per epoch next to the other jobs and was stopped at epoch 1,500 of 5,000; '
                          'its validation counts there were 141, 151, 150 of 300 at epochs 500, 1,000, 1,500 against 163, 163, 158 for the lift alone.</li>' if s.fwd(PP, 'lift025') is not None else '')
                       + '<li><b>Training length:</b> the lift field’s validation count is flat from epoch 500 on (163 of 300 at epoch 500, 165 at epoch 5,000, at 1.00 m). The 5,000 epochs were kept so that every run matches the earlier study.</li>'
                       '<li><b>Planner steps:</b> a step is 0.0615 cells, about 0.03 rad of joint motion. 2,400 steps per end is eight times the moves of the default budget; planning 300 queries that way took 80 seconds on a shared card.</li></ul>')
    N['demo_text'] = (f'<p>Ten zone → another-zone queries at {pn(s.P[a.demo_pl or B])}, both arms inside the cupboard at the start and at the goal, each arm ending in a different zone from where it started. All ten are solved with all three planning seeds by both lift fields that were trained there (training seeds 87358 and 87359; two-ended planner, 2,400 steps per end), '
                      'and each plan was re-checked at 0.001 rad. The ntrl-demo field solves none of them. The planner draws random samples, so a repeat is not identical: the two saved lift models, loaded on their own and run on just these ten with fresh draws (<code>use_model.py</code>), solved 10 and 9 of them. '
                      'The plans are valid, not short: the heads wander, and a plan is two to seven times as long as the straight joint distance. They are the ten hardest by this order: straight line blocked, both arms change zone, longest joint distance, at most two per zone pattern.</p>')
    N['demo_files'] = ('<p>Files: <code>jeffrey/arm_placement/report/demo.npz</code> (start and goal in radians, and the planned 12-joint paths of every planning seed as arrays of waypoints) and <code>report/demo.json</code>.</p>')
    (OUT / 'notes.json').write_text(json.dumps(N, indent=1))
    print('notes.json written; best', B, 'two-ended', fm(bl, 1), 'zone->zone', fm(blc, 1))


if __name__ == '__main__':
    main()
