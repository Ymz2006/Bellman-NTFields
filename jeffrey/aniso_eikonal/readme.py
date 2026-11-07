"""README.md of the study from the page's text blocks (text_blocks.py) and tables (page.py), so both carry the same numbers.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/readme.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUEST = '''Request (2026-11-05, about 22:10 EST): "Here is the issue I have with the mazes example. I want a diffeomorphism on the current maze to
'compress' some features, check the plus maze study. However, I can plan on this warped space fine and get a path but it is no
longer the minimal travel time path since the diffeomorphism ruins the travel time. With this diffeomorphism, the position alone is
not enough to determine the speed, but the direction is required as well. Moving in certain directions should cost more. The solution
is an anisotropic formulation, in order to compress the space while keeping an accurate travel time. I think you can implement this
through an inner product on the Lie gradient of T instead of just the magnitude of it. Verify that this idea makes sense, then try
this on the plus mazes. The goal is for this idea to get similar results on just planning on the warped space. Use all GPUs, try for
12 hours." Follow-ups in the same session: start from the optimal constructed warp for the plus mazes and use only constructed warps
(the learned warps are not good yet); use ntrl-demo with no TD / normal loss as the base; 4,000 epochs only.'''

REDIRECT = '''Redirect (2026-11-06, about 00:45 EST), after a first pass had found that a closed-form "deficit" term added to the stretched-space field
reproduces the travel time: "The idea I want is for this method to be used eventually in conjunction with a learned warp, and you will not
find success with the deficit term because the learned warp can be anisotropic, nonuniform and a lot messier. Treat this like a research
task. If it doesn't work with the biggest gap, try smaller gaps. However, the method you come up with must generalize. Stop the tests
with the deficit term. First get this idea to work, do not be stopped by the neural network's inability to represent large travel times.
Work with the smaller gaps first until you have a result similar to just planning on the stretched. Next, you can try to expand the
output of the neural network [the scalar on the distance head, a constant on the embeddings, anything that raises its maximum travel
time]."'''

VARIANTS = '''Queries solved (%), pooled over training seeds and planning seeds 0/1/2. Variant names: `ziso` trained on the stretched space (plain length);
`zaniso-U<unit>` the inner product with that time unit (cells per unit; `quad` = the warp's largest stretch, 3.7 cells; `box` = the maze's side);
`-open` with the open-start pairs; `-lb10` lower-bound hinge, weight 10; `-pwtauz` pair weight exp(-T rho / 2) (`-pwz`: on the warp distance alone);
`-gs8x1+8x4` head with 8 groups as they are and 8 multiplied by 4 (`-hs4`: a scalar 4 on the whole head); `-messy0.6` the messy warp;
`zres-Uquad-open-far-lb10` the two-part field (head on the plain length, far term on the inner product of the sum; `-full`: the head taken, frozen, from the gap study's full-loss field);
`-far` a far term added to the single field; `-sum2` two head groups added up instead of entering the soft maximum; `id` no warp. First-pass variants (not pursued): `zsplit-...-posthoc` a stretched-space field plus the deficit term, `zsplit-...` / `zboth-...`
trained with it, `-sh` / `-lo0.05` other band densities, `-full` the gap study's full-loss fields, `-tw0` no pair weight.'''

DEFICIT = '''## Appendix: the deficit term (first pass, set aside)

For this constructed warp the maze's travel time splits as T*(a, b) ≈ T_w(φ(a), φ(b)) / m + |u(a) − u(b)|, u(x) = x − φ(x) / m: the stretched maze's
travel time plus the straight-line distance in the coordinate the warp removed. On exact fields at gap ×12 the sum is within 5 % of T* on 97 % of
queries (run `verify`), and added to a stretched-space field without retraining it gave the same planning with true travel times (variants
`zsplit-Uquad-posthoc`, `...-posthoc-full` in the table above). It needs the compressed region to be open and the warp to be separable with a uniform
reference magnification, which a learned warp will not give, so it was dropped at the user's direction; the code (`aniso.split_class`, modes `zsplit`,
`zboth`) and the runs remain.'''

CODE = '''## Code

| file | what |
|---|---|
| `aniso.py` | `metric_full` (M = J Jᵀ of any warp, autograd), `eikonal_loss` (ntrl-demo's eikonal term with the inner product, a time unit and the pair weights), `residual_loss` (two-part field), `FarMetric` and `far_scale`, the field class (head scalar, embedding factor, group factors, additive groups; first-pass deficit term), `MessyWarp`, `load_field`, the checks |
| `atrain.py` | ntrl-demo's training loop with the eikonal term only; modes `id`, `ziso`, `zaniso`, `zres` (first pass: `zsplit`, `zboth`); the lower-bound hinge; `head_from` (train the correction on a frozen field) |
| `wallfeel.py` | post-hoc on registered checkpoints: where T rises into the quadrants' walls by goal class, and where failed plans first touch a wall (`report/wallfeel/<run id>.json`) |
| `pipeline.py` | one run: train (or `--checkpoint`), plan the gap study's queries on the maze and on the warp, travel-time accuracy, implied speed by region |
| `verify.py` | exact fields: the equations by region, the price of planning on the warped maze, the split identity |
| `run.py` | registers a run (stages `train`, `evalq`, `verify`); sources are frozen in `snap/<hash>/` |
| `report.py`, `figs.py`, `page.py`, `text_blocks.py`, `page_template.html`, `readme.py` | summary, figures, the findings page and this file |

Runs use the cross-warp study's prepared gap mazes, pairs and queries (`jeffrey/cross_warp`, stage `gapprepare --z-draw`) and its warp, planner and truth modules, copied into each snapshot.

## Reproduce

```bash
PY=~/.venvs/bellman-ntfield/bin/python
OPEN=results/registered_20261106T061244Z-8a3753d8     # gap x4, 22x22 quadrants: recipe pairs + open-start pairs (cross_warp gapprepare --z-draw --open-pairs 250000)
# the full recipe, and the stretched-space reference on the same pairs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4recipe \\
    --extra '--mode zaniso --unit-cells quad --group-scales 8x1,8x4 --lb-weight 10 --pair-weight tauz'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4zisoopen --extra '--mode ziso'
# the two-part field (plans with the sum and with its head alone), and its correction alone on an existing field
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4zres \\
    --extra '--mode zres --unit-cells quad --far auto --lb-weight 10 --planners z'
... --extra '--mode zres --unit-cells quad --far auto --lb-weight 10 --planners z --tag full --head-from results/registered_<field>/training/checkpoint.pt'
# the checks on exact fields
... --stage verify --prepared results/registered_20261105T070343Z-fda24e40 --label 'verify c22 k12'
$PY jeffrey/aniso_eikonal/report.py && $PY jeffrey/aniso_eikonal/wallfeel.py --all && $PY jeffrey/aniso_eikonal/figs.py && $PY jeffrey/aniso_eikonal/page.py && $PY jeffrey/aniso_eikonal/readme.py
```
'''


def md(s):
    """The page's HTML fragments as Markdown (the few tags text_blocks.py and page.table use)."""
    s = s.replace('</p><p class="lede">', '\n\n')
    s = re.sub(r'<figure>.*?</figure>', '', s, flags=re.S)
    s = re.sub(r'<div class="eq">(.*?)</div>', lambda m: '\n\n    ' + re.sub(r'<[^>]+>', '', re.sub(r'<sub>(.*?)</sub>', r'_\1', m.group(1))).replace('&nbsp;', ' ').strip() + '\n\n', s, flags=re.S)

    def tab(m):
        rows = re.findall(r'<tr>(.*?)</tr>', m.group(0), flags=re.S)
        out = []
        for i, r in enumerate(rows):
            cells = [re.sub(r'<[^>]+>', '', c).replace('|', '\\|').strip() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.S)]
            out.append('| ' + ' | '.join(cells) + ' |')
            if i == 0:
                out.append('|' + '---|' * len(cells))
        return '\n\n' + '\n'.join(out) + '\n\n'
    s = re.sub(r'<table.*?</table>', tab, s, flags=re.S)
    s = re.sub(r'<li>(.*?)</li>', lambda m: '\n* ' + m.group(1).strip(), s, flags=re.S)
    s = re.sub(r'</?ul[^>]*>', '\n', s)
    s = re.sub(r'<p[^>]*>(.*?)</p>', lambda m: '\n\n' + m.group(1).strip() + '\n\n', s, flags=re.S)
    s = re.sub(r'<strong>(.*?)</strong>', r'**\1**', s, flags=re.S)
    s = re.sub(r'<code>(.*?)</code>', r'`\1`', s, flags=re.S)
    s = re.sub(r'<sub>(.*?)</sub>', r'_\1', s)
    s = re.sub(r'<small>(.*?)</small>', r'\1', s)
    s = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', s, flags=re.S)
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    return re.sub(r'\n{3,}', '\n\n', s).strip()


def feel_table(PG):
    """The wall chart's numbers: share (%) of near-wall poses where T rises into the wall, goal in the same / another quadrant."""
    rows = []
    for base in PG.BASE_NAME:
        for key, label, _, _, _ in PG.FIELDS:
            if key == 'RESH':
                continue
            for goals, gname in ((['same quadrant'], 'same quadrant'), (PG.FAR_GOALS, 'another quadrant')):
                vals = [PG.feel(base, k, PG.resolve(base, k, key), goals) for k in PG.GAPS]
                if any(v is not None for v in vals):
                    rows.append([f'{PG.BASE_NAME[base]} · {label} · goal in {gname}'] + [PG.f1(v, 0) for v in vals])
    return PG.table(['field'] + [f'×{k}' for k in PG.GAPS], rows) if rows else ''


def main():
    import sys
    sys.path.insert(0, str(HERE))
    import page as PG
    import text_blocks as TB
    b = TB.blocks(dict(P=PG.P, plan=PG.plan, pc=PG.pc, f1=PG.f1, run_of=PG.run_of, rows_of=PG.rows_of, VER=PG.VER, POOL=PG.POOL, RUNS=PG.RUNS, BASE_NAME=PG.BASE_NAME, CATS=PG.CATS,
                       GAPS=PG.GAPS, table=PG.table, recipe=PG.recipe, feel=PG.feel, fail_where=PG.fail_where, feel_raw=PG.feel_raw, arrival=PG.arrival, stalled_open=PG.stalled_open, have_paths=False, have_fields=False, have_deficit=False))
    link = (HERE / 'report/artifact_url.txt').read_text().strip() if (HERE / 'report/artifact_url.txt').exists() else ''
    parts = ['# Anisotropic eikonal on the warp (`jeffrey/aniso_eikonal`)', REQUEST, REDIRECT,
             ('Findings page: ' + link + ' (`report/aniso_eikonal.html`, from `page.py`).') if link else 'Findings page: `report/aniso_eikonal.html` (from `page.py`).',
             '## Summary', md(b['LEDE']), md(b['VERDICT']),
             '## 1. The idea, and that it is right', md(b['IDEATEXT']), '(Figure: `report/indicatrix.png`.)', md(b['IDEATEXT2']), md(PG.eq_table()),
             'S × gradient norm at random grid nodes of the gap ×12 maze (22×22 quadrants), by region; 1 = the equation holds. T*: the maze’s travel time; T_w: the stretched maze’s.',
             '## 2. What the warp costs when it is ignored', md(b['PRICETEXT']), '(Figure: `report/paths.png`.)',
             '## 3. Getting the inner product to train', md(b['STEPS0']),
             '### 3.1 The unit of time', md(b['UNITTEXT']),
             '### 3.2 Pairs that start in the open', md(b['OPENTEXT']),
             '### 3.3 A lower bound the equation does not contain', md(b['LBTEXT']), '(Figure: `report/fields.png`.)',
             '### 3.4 A pair weight that sees across the band', md(b['PWTEXT']),
             '### 3.5 A distance head with more than one scale', md(b['RANGETEXT']), md(b['ABLTABLE']), md(b['ABLNOTE']),
             '## 4. Against planning on the stretched space', md(b['MAINTEXT']), md(b['MAINNOTE']),
             '## 5. Where the plans between quadrants fail: walls are not felt from afar', md(b['FEELTEXT']), md(feel_table(PG)), md(b['FEELNOTE']), md(b['FEELFIX']),
             '## 6. The other route: the stretched space’s field plus a learned correction', md(b['RESTEXT']), md(PG.res_table()), md(b['RESNOTE']), md(b['RESTEXT2']), md(PG.full_table()), md(b['FULLNOTE']),
             '## 7. Does the field tell the maze’s travel time?', md(b['TIMETEXT']), md(b['PATHTEXT']),
             '## 8. Nothing here uses the constructed warp’s structure', md(b['GENTEXT']), md(b['GENTABLE']),
             '## 9. What this says', md(b['VERDICT']),
             '## 10. How it was measured', md(b['SETUP']),
             '## 11. Every field at every gap', md(PG.all_table()), VARIANTS, DEFICIT,
             CODE, '## Registered runs', md(PG.run_table())]
    (HERE / 'README.md').write_text('\n\n'.join(p for p in parts if p) + '\n')
    print(f'wrote {HERE / "README.md"}')


if __name__ == '__main__':
    main()
