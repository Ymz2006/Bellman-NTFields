# Lift + midpoint: the travel-time midpoint loss on top of the travel-time lift (no push)

Study of 2026-11-17. Request:

> I want you to try the recursive travel time midpoint loss with the travel time lift (no push). Do only a few tests, and show me the
> results. Show me like 8 scenarios. You can try increasing maze sizes from 10x10 and onward, a few plus mazes with the expanding plus
> area, and also some gibson. 2d for now. https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d this one is for lift.
> https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U this one for midpoint. See if the midpoint loss can mitigate the problems that lift
> alone could not solve, so your tests should be on those where the lift itself was struggling compare to the baseline.

## What was combined

* **Lift** (jeffrey/lift_scale, the local rebuild of the 2026-11-11 lift page): ntrl-demo's network gets one more input column, the
  travel-time height h(q) = G(d(q)) / E of the configuration, G(d) = int_d^margin sqrt(1 / S^2 - 1), d the exact clearance (lift.py).
  No push, no octaves, gain 1.
* **Midpoint loss** (jeffrey/midpoint_dp, its MAIN setting unchanged): from epoch 2,000, every 100 epochs 8,192 random pairs of
  configurations (clearance >= 0.1 cells) are driven by ntrl-demo's MPPI on the current field ('safe': a rollout whose first state comes
  within 0.02 cells of a wall gets +100); each drive is cut where it first collides or stalls, split recursively at travel-time midpoints
  down to one cell of travel time; a leaf is worth min(T(leaf ends), the time the drive took over it); every node of the recursion and an
  equal number of random sub-paths get the sum of their leaves as target; |log T - log target| on 8,192 nodes per update, weight ramped
  0 -> 1 over epochs 2,000-2,500, to epoch 5,000.
* **Scenes**: the SE(2) rectangle robot of every local maze study (0.5 x 0.2 cells, q = (x, y, phi)) in 2-D scenes: maze_scale's
  10 x 10, 13 x 13, 22 x 22 mazes; plus mazes 22 / 30 / 40 with a centred + of 6 / 8 / 12 cells; Gibson houses Allensville (43 cells)
  and Lakeville (83 cells) as floor plans (lift_scale). "2d for now" was read as these 2-D floor-plan scenes with the robot every local
  study uses, rather than a point robot, because both pages being combined were reproduced here on this setup.

## Arms (training seed 87358, planning seeds 0 / 1 / 2)

| arm | what | where it comes from |
|---|---|---|
| ntrl-demo | the baseline | maze_scale ntrl runs (mazes), lift_scale `--lift none` runs (plus mazes, houses) |
| lift alone | `--lift time`, no push | lift_scale runs (plus mazes, houses); new lift_scale runs on the three mazes |
| lift + midpoint | the test | this study, `pipeline.py --lift time` |
| midpoint alone | `--lift none` + the same midpoint term | midpoint_dp MAIN runs (mazes); this study, `pipeline.py --lift none` (plus mazes, houses) |
| lift + push | lift_scale's page schedule | reference only (lift_scale runs) |
| driver control | lift + midpoint, but every training drive is made by the frozen ntrl-demo field of the scene (`--driver`); leaves, targets and the rest unchanged | this study (plus 22, Allensville) |

The lift + midpoint run is the lift-alone run of the same seed up to epoch 2,000, bit for bit: the trainer is lift_scale's loop with no
push, and the midpoint machinery draws only from its own generator (checked on the smoke run: losses equal to every digit through the
fork epoch, different after it; the plus-maze run plans 129 / 240 at epoch 2,000 like the registered lift-alone field). The driver
control is not identical up to the fork: loading the frozen driver network consumes random draws, so its first 2,000 epochs are another
draw of the lift-alone training (plus maze: 133 / 240 at epoch 2,000).

## Files

| file | what |
|---|---|
| `liftfast.py` | the lifted network as one matrix product on rows [q, h]: compiled no-grad evaluator, batched MPPI traversals (midpoint_dp/traverse.mppi with the clearance of both rollout ends computed once per step for the lift and the safety probe), differentiable T for training; `check()` against NN.out (4e-7 eager, 1e-5 compiled) |
| `trainer.py` | lift_scale's ntrl-demo loop (no push) + midpoint_dp's bank, refresh and term |
| `pipeline.py` | train, plan (maze_scale MPPI, 3 planning seeds, exact scoring), lift_scale diagnostics (wall fit, travel times against the truth, line of sight, detour index, stalls), field measures at saved epochs |
| `run.py` | registry wrapper; sources frozen in `snap/<hash>/`; checks that the six reused modules are verbatim copies |
| `lift.py`, `scenes.py`, `diag.py` | verbatim copies of jeffrey/lift_scale's |
| `fastnet.py`, `traverse.py`, `tree.py` | verbatim copies of jeffrey/midpoint_dp's |
| `posthoc.py` | lift_scale's field measures for the maze_scale ntrl and midpoint_dp runs (same 24,000 truth pairs) |
| `leafcheck.py` | read-only: 4,096 training-style drives of a final network (or of its driver): share that arrive, detour share of the arrivals, leaf ratio T_net / drive time and how often the min takes the network; one process per scene (plus.install patches once per process) |
| `page.py`, `pagetext.py`, `page_assets/charts.js`, `pagecheck.py` | findings page (inline SVG charts) and its headless-Chrome check |
| `readme.py` | this README: README_setup.md + the page's text + report/tables.md |
| `report.py` | `report/runs.json`, `report/tables.md` from the registered runs |
| `queue/` | runner (at most two of this study's heavy runs at once, RAM guard), jobs, state |

## Findings

Findings page: https://claude.ai/artifact/DdUvv75Ld972r86nabE8kT (`report/lift_midpoint.html`, built by `page.py` from `report/runs.json`). `report/tables.md` (`report.py`) is regenerated from the registry and is authoritative for every number. Training seed 87358 everywhere; one training of the same recipe moves by 3-6 points between seeds.

Added to the travel-time lift without the push, the recursive midpoint loss fixes the lift's short far field wherever it shows up, and its see-through detours only where the lifted field can already drive around obstacles. On the plus mazes and the houses, travel times along free lines of sight go from 0.54–0.88 of the truth to 0.82–0.97. Detours are learned in Gibson Allensville (43 cells): 84.6 % of the queries solved against 66.5 % for the lift alone and 83.9 % for ntrl-demo. On the plus mazes the field keeps going through walls and gains +4.3 / +0.8 / +1.4 points (plus maze 22 x 22 / plus maze 30 x 30 / plus maze 40 x 40). The loss learns only from drives that the current field makes. A field that sees through a wall drives into it and stalls, and its drives that do arrive are almost all straight lines.

1. **Where does the lift alone fall behind ntrl-demo?** **not everywhere.** On the open scenes: plus maze 22 x 22, plus maze 40 x 40, Gibson Allensville, Gibson Lakeville, Gibson Klickitat (55.0 / 56.2 / 66.5 / 54.2 / 60.4 % against 65.0 / 59.2 / 83.9 / 69.4 / 72.5 %). On the regular mazes it is ahead (95.5 / 86.3 / 28.8 % against 83.3 / 57.8 / 21.1 % on maze 10 x 10, maze 13 x 13, maze 22 x 22), because ntrl-demo arrives through the thin walls there (14.2 / 41.3 / 67.9 % of its runs).
2. **Does the midpoint loss fix the short far field?** **yes.** Line-of-sight travel times on the plus mazes and the houses, learned over true: 0.54–0.88 with the lift alone, 0.82–0.97 with the midpoint loss. In the configuration space the lifted field puts part of its slope into the heading coordinate and comes out short far away; drives along free lines of sight are what the loss sees most, and it corrects them. On the mazes the lift alone is already right along lines of sight (0.99 / 1.00).
3. **Does it fix the see-through detours?** **only where drives find them.** Detour index (0 = straight through the obstacle, 1 = the true detour), lift alone → lift + midpoint. Houses (Allensville / Lakeville / Klickitat): -0.43 / -0.86 / -0.60 → 0.37 / -0.37 / 0.11, queries solved +18.1 / +6.5 / +4.6 points; Allensville reaches ntrl-demo (84.6 % against 83.9 %), the larger houses stay below it. Plus mazes (22 / 30 / 40 cells): -0.20 / -0.67 / -0.84 → 0.01 / -0.41 / -0.64, queries solved +4.3 / +0.8 / +1.4 points. 22-cell maze: -0.63 → 0.00, queries solved +2.9 points. The 13-cell maze is the contrast: the lift alone is not see-through there (detour index 1.18), and lift + midpoint plans 88.7 %, level with the midpoint alone (89.3 %).
4. **Why not on the plus mazes?** **no detours in the drives.** The loss trains on the drives the current field makes. On the 22-cell plus maze the lifted field's drives reach their goal 53 % of the time, and 1.0 % of those are detours (more than 30 % longer than the straight line); ntrl-demo's drives: 83 % and 6.8 %. In Gibson Allensville (43 cells) the lifted field's own drives already hold some detours (6.1 % of the arrivals) and the loss grows them: 76 % of its drives arrive at the end, 16.7 % of them detours. The same limit holds without the lift: on the 40-cell plus maze even ntrl-demo's drives arrive almost only along straight lines (0.3 % detours), and the midpoint loss on ntrl-demo's network stays at 60.7 % (ntrl-demo 59.2 %).
5. **And when the drives do hold detours?** **in the house, mostly yes; on the plus maze, partly.** Driver control: the same lift + midpoint training with every drive made by the frozen ntrl-demo field of the scene, which arrives with detours (6.8 % of its arrivals on the 22-cell plus maze, 23.4 % in Gibson Allensville). In Gibson Allensville the lifted network then learns the long detours almost fully: a true detour of 17.1 cells is learned as 15.5 cells, against 5.7 when the lifted field drives itself and 17.7 for ntrl-demo's network with the midpoint loss; detour index 0.84 against 0.37. On the 22-cell plus maze it learns part of them: detour index 0.39 against 0.01 (ntrl-demo's network 1.11), 8.0-cell detours learned as 3.2 cells (ntrl-demo's network 8.8). Queries solved with ntrl-demo's drives: 87.5 % and 61.9 % (self-driven 84.6 % and 59.3 %). So in the house the missing piece is the detours in the data; on the plus maze it is also the representation: the lift gives the two faces of a 0.1-cell wall the same height, and the step in travel time across the wall has to come from the planar features alone (the lift-scale study's finding).
6. **Is the lift worth adding to the midpoint loss?** **no.** Queries solved, midpoint alone against lift + midpoint: maze 13 x 13 89.3 % against 88.7 %; maze 22 x 22 39.2 % against 31.7 %; plus maze 22 x 22 76.4 % against 59.3 %; plus maze 40 x 40 60.7 % against 57.6 %; Gibson Allensville 92.8 % against 84.6 %. Midpoint alone is ahead by more than 5 points on maze 22 x 22, plus maze 22 x 22, Gibson Allensville and within 5 points on maze 13 x 13 (the lifted field is not see-through there), plus maze 40 x 40 (neither field's drives hold detours there). The lift removes ntrl-demo's wall crossings, but the midpoint loss needs a field whose drives go around walls, and ntrl-demo's smeared walls do that where the lift sees through.
7. **What would make the combination work?** Both missing pieces are known from the lift-scale study: drives that hold detours from the start (a push, or a driver field that is not see-through) and planar resolution at the cell scale, so a thin wall can carry a step (octaves of the position features scaled with the scene took the houses to 98–100 % with lift + push). Neither was combined with the midpoint loss here.

### Mechanism

Every 100 epochs the loss drives 8,192 random pairs with the current field and learns travel times only along those drives. A pair whose straight line crosses a wall reaches its goal only if the field leads the drive around the wall. A field that sees through the wall leads it into the wall, where the drive stalls; the part before the stall is kept, and it says nothing about the way around. How many random pairs need a detour at all differs by scene: maze 10 x 10 64 %, maze 13 x 13 71 %, maze 22 x 22 65 %, plus maze 22 x 22 11 %, plus maze 30 x 30 10 %, plus maze 40 x 40 8 %, Gibson Allensville 22 %, Gibson Lakeville 12 %, Gibson Klickitat 13 %.

Per field, from its own drives (leafcheck.py, 4,096 drives of the final network): on the 22-cell plus maze the lifted field arrives 53 % of the time with 1.0 % detours, ntrl-demo 83 % with 6.8 %; on the 40-cell plus maze both fields' arrivals are straight (0.3 % and 0.3 % detours). In Gibson Allensville (43 cells) the lifted field arrives 31 % of the time with 6.1 % detours and, after the midpoint loss, 76 % with 16.7 %. On the 22-cell maze, where every corridor wall is seen through (detour index -0.63), only 6 % of the first refresh's drives arrive at all. On the 13-cell maze the lifted field is not see-through at all (detour index 1.18) and 72 % of its arrivals are detours.

A second, smaller effect is in the leaves. A leaf is worth min(network time, time the drive took); that minimum was built for ntrl-demo, whose field is long at the one-cell scale, so the drive's time usually wins and anchors the targets. The lifted field in the configuration space is short locally on the 22-cell plus maze: the network's time over a leaf is 0.98 of the drive's before the midpoint loss and 0.93 after it (ntrl-demo 1.09), and the minimum takes the network's own value 59 % and 72 % of the time. The targets inherit that shortfall, so the field ends at 0.91 of the truth along lines of sight, consistent but short. In Gibson Allensville (43 cells) it went the other way (0.88 → 1.03), once the drives there held enough detours.

The midpoint loss narrows the first fault on every open scene (0.54–0.88 → 0.82–0.97 of the truth): drives along free lines of sight are the drives a see-through field completes, and their travel times are what the loss pulls the field towards. The detour index moves only where detours are among those drives.

### Caveats

* One training seed per arm. Between seeds the same recipe moves by 3–6 points on these scenes (lift_scale, midpoint_dp), so differences
below about 5 points are not resolved.

* "2-D" was read as 2-D scenes with the robot every local study uses. With a point robot the lift is exact and the heading shortcut
(the reason the lifted field is short far away) does not exist.

* The midpoint setting was chosen on ntrl-demo fields (midpoint_dp, 13 × 13 maze) and is used unchanged; nothing was tuned for the lift.

* In Allensville the lift + midpoint field was still improving at epoch 5,000 (80.4 % at 4,500, 84.6 % at 5,000 with planning seed 0);
longer training was not run.

* Field measures of the maze_scale ntrl-demo and midpoint_dp runs were computed afterwards on their final networks (posthoc.py), on the
same truth pairs as every other arm.

## Tables

| scene | arm | solved % | wall % | stall % | T within 10 % | median T/T* | line-of-sight within 10 % | LOS T/T* | detour index | seen through % | S x slope innermost | run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| maze 10 x 10 | ntrl-demo | 83.3 | 14.2 | 2.5 | 11.3 | 1.318 | 62.4 | 1.058 | 2.10 | 0.0 | 0.25 | 20261102T032216Z-e76476bf |
| maze 10 x 10 | lift alone | 95.5 | 0.0 | 4.5 | 61.7 | 1.041 | 91.5 | 0.965 | 1.23 | 0.6 | 0.98 | 20261117T174808Z-04004e68 |
| maze 10 x 10 | midpoint alone | 97.7 | 1.8 | 0.5 | 94.5 | 1.014 | 79.3 | 0.963 | 1.05 | 0.1 | 0.31 | 20261102T081615Z-0adefad6 |
| maze 13 x 13 | ntrl-demo | 57.8 | 41.3 | 0.8 | 6.4 | 1.730 | 46.5 | 1.100 | 2.86 | 0.1 | 0.16 | 20261101T203437Z-ebc95225 |
| maze 13 x 13 | lift alone | 86.3 | 0.0 | 13.7 | 50.4 | 1.049 | 86.5 | 0.988 | 1.18 | 6.3 | 0.97 | 20261117T180223Z-c09b69b8 |
| maze 13 x 13 | lift + midpoint | 88.7 | 0.0 | 11.3 | 90.2 | 1.015 | 92.9 | 0.989 | 1.05 | 3.8 | 0.97 | 20261117T214618Z-3444441f |
| maze 13 x 13 | midpoint alone | 89.3 | 8.2 | 2.5 | 69.3 | 1.073 | 85.3 | 1.005 | 1.18 | 0.1 | 0.18 | 20261102T081356Z-c2f195aa |
| maze 22 x 22 | ntrl-demo | 21.1 | 67.9 | 11.0 | 9.7 | 1.338 | 8.6 | 1.314 | 1.74 | 7.0 | 0.06 | 20261101T193703Z-dd44f02b |
| maze 22 x 22 | lift alone | 28.8 | 0.0 | 71.2 | 1.8 | 0.398 | 38.5 | 1.001 | -0.63 | 99.9 | 0.96 | 20261117T182020Z-bfba8bc8 |
| maze 22 x 22 | lift + midpoint | 31.7 | 0.0 | 68.3 | 14.9 | 0.676 | 60.1 | 1.075 | -0.00 | 94.4 | 0.96 | 20261117T202828Z-666895f4 |
| maze 22 x 22 | midpoint alone | 39.2 | 43.5 | 17.4 | 31.9 | 1.118 | 38.7 | 1.127 | 1.28 | 3.9 | 0.08 | 20261102T084156Z-6809fed7 |
| plus maze 22 x 22 (+ of 6) | ntrl-demo | 65.0 | 15.1 | 19.9 | 46.3 | 1.104 | 50.6 | 1.099 | 1.37 | 2.8 | 0.08 | 20261117T054519Z-9a3bc556 |
| plus maze 22 x 22 (+ of 6) | lift alone | 55.0 | 0.0 | 45.0 | 25.5 | 0.821 | 43.5 | 0.884 | -0.20 | 98.3 | 1.00 | 20261117T055249Z-d953394e |
| plus maze 22 x 22 (+ of 6) | lift + midpoint | 59.3 | 0.0 | 40.7 | 40.7 | 0.882 | 60.7 | 0.914 | 0.01 | 85.5 | 0.97 | 20261117T175835Z-8c936298 |
| plus maze 22 x 22 (+ of 6) | midpoint alone | 76.4 | 4.2 | 19.4 | 92.3 | 0.997 | 98.4 | 0.977 | 1.11 | 1.4 | 0.10 | 20261117T193157Z-04660c58 |
| plus maze 22 x 22 (+ of 6) | lift + push (reference) | 68.2 | 1.0 | 30.8 | 89.4 | 1.002 | 99.4 | 1.015 | 0.47 | 52.9 | 0.86 | 20261117T061710Z-c95ca37c |
| plus maze 22 x 22 (+ of 6) | lift + midpoint, drives by the ntrl-demo field (control) | 61.9 | 0.0 | 38.1 | 69.6 | 0.937 | 87.6 | 0.954 | 0.39 | 60.7 | 0.97 | 20261117T195718Z-0501b704 |
| plus maze 30 x 30 (+ of 8) | ntrl-demo | 56.0 | 32.4 | 11.7 | 54.9 | 1.091 | 62.9 | 1.082 | 1.22 | 4.3 | 0.06 | 20261117T120340Z-057a1ff3 |
| plus maze 30 x 30 (+ of 8) | lift alone | 57.2 | 0.0 | 42.8 | 11.4 | 0.628 | 21.7 | 0.735 | -0.67 | 99.9 | 0.96 | 20261117T120955Z-aa3b4619 |
| plus maze 30 x 30 (+ of 8) | lift + midpoint | 58.1 | 0.0 | 41.9 | 38.6 | 0.835 | 70.9 | 0.952 | -0.41 | 99.4 | 0.95 | 20261117T214028Z-41832502 |
| plus maze 30 x 30 (+ of 8) | lift + push (reference) | 63.8 | 0.0 | 36.2 | 80.2 | 0.975 | 99.6 | 1.000 | 0.25 | 95.8 | 0.84 | 20261117T121530Z-ea7160ea |
| plus maze 40 x 40 (+ of 12) | ntrl-demo | 59.2 | 40.8 | 0.0 | 49.6 | 1.096 | 56.8 | 1.093 | 0.73 | 19.1 | 0.05 | 20261117T070240Z-9926f070 |
| plus maze 40 x 40 (+ of 12) | lift alone | 56.2 | 0.0 | 43.8 | 12.0 | 0.641 | 19.6 | 0.727 | -0.84 | 99.9 | 0.97 | 20261117T070630Z-a905919d |
| plus maze 40 x 40 (+ of 12) | lift + midpoint | 57.6 | 0.0 | 42.4 | 38.8 | 0.829 | 63.4 | 0.947 | -0.64 | 100.0 | 0.96 | 20261117T183848Z-fec23ff1 |
| plus maze 40 x 40 (+ of 12) | midpoint alone | 60.7 | 37.1 | 2.1 | 67.3 | 1.030 | 86.9 | 1.007 | 0.90 | 9.9 | 0.06 | 20261117T221238Z-093cd9c0 |
| plus maze 40 x 40 (+ of 12) | lift + push (reference) | 63.6 | 0.0 | 36.4 | 78.5 | 0.970 | 99.4 | 0.988 | 0.03 | 100.0 | 0.83 | 20261117T071105Z-e272648c |
| Gibson Allensville (43 cells) | ntrl-demo | 83.9 | 15.6 | 0.6 | 66.5 | 1.064 | 82.2 | 1.047 | 1.26 | 0.3 | 0.07 | 20261117T121605Z-9c558da3 |
| Gibson Allensville (43 cells) | lift alone | 66.5 | 0.0 | 33.5 | 4.9 | 0.617 | 12.7 | 0.730 | -0.43 | 99.9 | 1.00 | 20261117T122340Z-11cef54b |
| Gibson Allensville (43 cells) | lift + midpoint | 84.6 | 0.0 | 15.4 | 61.1 | 0.932 | 87.4 | 0.973 | 0.37 | 69.4 | 0.88 | 20261117T182927Z-0a0eca6d |
| Gibson Allensville (43 cells) | midpoint alone | 92.8 | 6.8 | 0.4 | 98.3 | 1.005 | 99.1 | 0.998 | 1.03 | 0.6 | 0.06 | 20261117T205218Z-8f115b21 |
| Gibson Allensville (43 cells) | lift + push (reference) | 99.6 | 0.0 | 0.4 | 87.1 | 1.043 | 97.8 | 1.012 | 1.20 | 0.2 | 0.78 | 20261117T122655Z-ff2c4b49 |
| Gibson Allensville (43 cells) | lift + midpoint, drives by the ntrl-demo field (control) | 87.5 | 0.0 | 12.5 | 65.1 | 0.933 | 76.7 | 0.960 | 0.84 | 15.9 | 0.95 | 20261117T210608Z-44ad666e |
| Gibson Lakeville (83 cells) | ntrl-demo | 69.4 | 7.0 | 23.6 | 75.3 | 0.961 | 83.1 | 0.982 | 0.72 | 31.5 | 0.06 | 20261117T094420Z-31c51802 |
| Gibson Lakeville (83 cells) | lift alone | 54.2 | 0.0 | 45.8 | 1.7 | 0.442 | 4.9 | 0.538 | -0.86 | 100.0 | 1.00 | 20261117T094615Z-a1c691d1 |
| Gibson Lakeville (83 cells) | lift + midpoint | 60.7 | 0.0 | 39.3 | 14.3 | 0.690 | 33.6 | 0.821 | -0.37 | 97.8 | 0.97 | 20261117T191038Z-55bb04e3 |
| Gibson Lakeville (83 cells) | lift + push (reference) | 77.0 | 0.0 | 23.0 | 86.4 | 0.984 | 99.7 | 1.004 | 0.43 | 55.0 | 0.75 | 20261117T104235Z-b6379951 |
| Gibson Klickitat (96 cells) | ntrl-demo | 72.5 | 3.2 | 24.3 | 72.8 | 0.940 | 85.9 | 0.958 | 0.75 | 28.0 | 0.05 | 20261117T104445Z-0c50416c |
| Gibson Klickitat (96 cells) | lift alone | 60.4 | 0.0 | 39.6 | 2.9 | 0.483 | 7.2 | 0.553 | -0.60 | 100.0 | 0.96 | 20261117T105600Z-740e99b6 |
| Gibson Klickitat (96 cells) | lift + midpoint | 65.0 | 0.0 | 35.0 | 23.3 | 0.759 | 38.6 | 0.839 | 0.11 | 75.6 | 0.95 | 20261117T200908Z-309ea4ed |
| Gibson Klickitat (96 cells) | lift + push (reference) | 85.5 | 0.0 | 14.5 | 94.6 | 0.991 | 99.2 | 0.989 | 0.95 | 15.0 | 0.87 | 20261117T110015Z-d7107d88 |

## Success by true distance (cells), % of runs

### maze 10 x 10

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 |
|---|---:|---:|---:|---:|---:|
| ntrl-demo | 98 | 88 | 98 | 82 | 51 |
| lift alone | 100 | 98 | 98 | 90 | 92 |
| midpoint alone | 100 | 100 | 100 | 97 | 92 |

### maze 13 x 13

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 |
|---|---:|---:|---:|---:|---:|
| ntrl-demo | 92 | 75 | 48 | 48 | 27 |
| lift alone | 100 | 95 | 75 | 90 | 72 |
| lift + midpoint | 100 | 100 | 82 | 91 | 70 |
| midpoint alone | 100 | 81 | 88 | 92 | 87 |

### maze 22 x 22

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 75 | 22 | 10 | 15 | 5 | 0 |
| lift alone | 94 | 60 | 15 | 3 | 0 | 0 |
| lift + midpoint | 94 | 69 | 18 | 9 | 0 | 0 |
| midpoint alone | 82 | 53 | 33 | 38 | 28 | 1 |

### plus maze 22 x 22 (+ of 6)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 100 | 98 | 91 | 65 | 37 | 0 |
| lift alone | 100 | 95 | 85 | 40 | 10 | 0 |
| lift + midpoint | 100 | 100 | 85 | 56 | 15 | 0 |
| midpoint alone | 100 | 100 | 98 | 90 | 71 | 0 |
| lift + push (reference) | 100 | 100 | 93 | 81 | 35 | 0 |
| lift + midpoint, drives by the ntrl-demo field (control) | 100 | 98 | 88 | 65 | 22 | 0 |

### plus maze 30 x 30 (+ of 8)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 85 | 88 | 78 | 57 | 28 | 0 |
| lift alone | 95 | 95 | 83 | 52 | 18 | 0 |
| lift + midpoint | 95 | 95 | 88 | 58 | 13 | 0 |
| lift + push (reference) | 95 | 98 | 89 | 60 | 38 | 2 |

### plus maze 40 x 40 (+ of 12)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 100 | 80 | 88 | 78 | 52 | 18 | 0 |
| lift alone | 100 | 88 | 88 | 78 | 32 | 8 | 0 |
| lift + midpoint | 100 | 88 | 90 | 85 | 36 | 5 | 0 |
| midpoint alone | 100 | 82 | 88 | 82 | 58 | 15 | 0 |
| lift + push (reference) | 100 | 88 | 90 | 89 | 52 | 26 | 0 |

### Gibson Allensville (43 cells)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 100 | 100 | 95 | 82 | 74 | 52 |
| lift alone | 100 | 100 | 85 | 67 | 48 | 0 |
| lift + midpoint | 100 | 100 | 98 | 88 | 90 | 32 |
| midpoint alone | 100 | 100 | 92 | 90 | 89 | 85 |
| lift + push (reference) | 100 | 100 | 100 | 100 | 98 | 100 |
| lift + midpoint, drives by the ntrl-demo field (control) | 100 | 100 | 88 | 80 | 92 | 65 |

### Gibson Lakeville (83 cells)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 100 | 100 | 95 | 78 | 51 | 35 | 27 |
| lift alone | 100 | 100 | 95 | 72 | 12 | 0 | 0 |
| lift + midpoint | 100 | 100 | 95 | 85 | 32 | 10 | 2 |
| lift + push (reference) | 100 | 100 | 100 | 98 | 47 | 57 | 38 |

### Gibson Klickitat (96 cells)

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl-demo | 100 | 95 | 85 | 95 | 62 | 52 | 18 |
| lift alone | 100 | 98 | 90 | 92 | 40 | 2 | 0 |
| lift + midpoint | 100 | 98 | 85 | 95 | 50 | 22 | 5 |
| lift + push (reference) | 100 | 100 | 95 | 97 | 78 | 82 | 47 |

## Lift + midpoint during training (planning seed 0 on the prepared queries; refresh = 8,192 traversals of the current field)

* maze 13 x 13: validate 2000: 68.0 % (stall 62, wall 2), 2500: 75.0 % (stall 50, wall 0), 3000: 87.5 % (stall 23, wall 2), 3500: 83.0 % (stall 33, wall 1), 4000: 87.5 % (stall 24, wall 1), 4500: 88.0 % (stall 22, wall 2)
  refresh 2001: reached 4520, detour share 0.660, 2501: reached 5014, detour share 0.654, 3001: reached 6389, detour share 0.708, 3501: reached 6062, detour share 0.680, 4001: reached 6452, detour share 0.699, 4501: reached 6775, detour share 0.708, 4901: reached 6874, detour share 0.697
  training 26 min, whole run 27 min on NVIDIA GeForce RTX 3090
* maze 22 x 22: validate 2000: 30.4 % (stall 167, wall 0), 2500: 28.3 % (stall 172, wall 0), 3000: 29.2 % (stall 170, wall 0), 3500: 29.2 % (stall 170, wall 0), 4000: 29.2 % (stall 170, wall 0), 4500: 28.8 % (stall 171, wall 0)
  refresh 2001: reached 483, detour share 0.048, 2501: reached 521, detour share 0.050, 3001: reached 533, detour share 0.056, 3501: reached 540, detour share 0.059, 4001: reached 634, detour share 0.069, 4501: reached 635, detour share 0.077, 4901: reached 677, detour share 0.072
  training 23 min, whole run 23 min on NVIDIA GeForce RTX 5070
* plus maze 22 x 22 (+ of 6): validate 2000: 53.8 % (stall 111, wall 0), 2500: 55.8 % (stall 106, wall 0), 3000: 57.1 % (stall 103, wall 0), 3500: 57.5 % (stall 102, wall 0), 4000: 59.6 % (stall 97, wall 0), 4500: 59.2 % (stall 98, wall 0)
  refresh 2001: reached 4062, detour share 0.002, 2501: reached 4170, detour share 0.005, 3001: reached 4420, detour share 0.004, 3501: reached 4493, detour share 0.007, 4001: reached 4645, detour share 0.009, 4501: reached 4746, detour share 0.009, 4901: reached 4860, detour share 0.014
  training 30 min, whole run 30 min on NVIDIA GeForce RTX 5070
* plus maze 30 x 30 (+ of 8): validate 2000: 58.8 % (stall 99, wall 0), 2500: 59.2 % (stall 98, wall 0), 3000: 57.5 % (stall 102, wall 0), 3500: 57.5 % (stall 102, wall 0), 4000: 59.2 % (stall 98, wall 0), 4500: 57.5 % (stall 102, wall 0)
  refresh 2001: reached 2958, detour share 0.005, 2501: reached 3773, detour share 0.004, 3001: reached 3617, detour share 0.006, 3501: reached 3574, detour share 0.004, 4001: reached 3689, detour share 0.004, 4501: reached 3761, detour share 0.005, 4901: reached 3731, detour share 0.003
  training 31 min, whole run 32 min on NVIDIA GeForce RTX 5070
* plus maze 40 x 40 (+ of 12): validate 2000: 58.9 % (stall 115, wall 0), 2500: 59.6 % (stall 113, wall 0), 3000: 57.5 % (stall 119, wall 0), 3500: 57.5 % (stall 119, wall 0), 4000: 57.5 % (stall 119, wall 0), 4500: 57.5 % (stall 119, wall 0)
  refresh 2001: reached 3522, detour share 0.000, 2501: reached 4057, detour share 0.000, 3001: reached 3866, detour share 0.000, 3501: reached 3890, detour share 0.000, 4001: reached 3887, detour share 0.000, 4501: reached 3900, detour share 0.001, 4901: reached 3860, detour share 0.001
  training 51 min, whole run 53 min on NVIDIA GeForce RTX 3090
* Gibson Allensville (43 cells): validate 2000: 66.7 % (stall 80, wall 0), 2500: 73.3 % (stall 64, wall 0), 3000: 75.8 % (stall 58, wall 0), 3500: 76.7 % (stall 56, wall 0), 4000: 78.8 % (stall 51, wall 0), 4500: 80.4 % (stall 47, wall 0)
  refresh 2001: reached 2817, detour share 0.051, 2501: reached 4463, detour share 0.081, 3001: reached 4802, detour share 0.086, 3501: reached 5063, detour share 0.075, 4001: reached 5210, detour share 0.091, 4501: reached 5578, detour share 0.108, 4901: reached 5837, detour share 0.129
  training 40 min, whole run 41 min on NVIDIA GeForce RTX 5070
* Gibson Lakeville (83 cells): validate 2000: 55.4 % (stall 125, wall 0), 2500: 58.2 % (stall 117, wall 0), 3000: 58.6 % (stall 116, wall 0), 3500: 60.4 % (stall 111, wall 0), 4000: 60.4 % (stall 111, wall 0), 4500: 60.4 % (stall 111, wall 0)
  refresh 2001: reached 978, detour share 0.013, 2501: reached 1695, detour share 0.022, 3001: reached 1911, detour share 0.026, 3501: reached 1939, detour share 0.034, 4001: reached 2012, detour share 0.039, 4501: reached 2059, detour share 0.034, 4901: reached 2157, detour share 0.037
  training 45 min, whole run 46 min on NVIDIA GeForce RTX 5070
* Gibson Klickitat (96 cells): validate 2000: 59.3 % (stall 114, wall 0), 2500: 62.5 % (stall 105, wall 0), 3000: 64.3 % (stall 100, wall 0), 3500: 64.6 % (stall 99, wall 0), 4000: 65.0 % (stall 98, wall 0), 4500: 66.1 % (stall 95, wall 0)
  refresh 2001: reached 1331, detour share 0.021, 2501: reached 2017, detour share 0.043, 3001: reached 2298, detour share 0.049, 3501: reached 2390, detour share 0.063, 4001: reached 2385, detour share 0.065, 4501: reached 2482, detour share 0.067, 4901: reached 2507, detour share 0.071
  training 56 min, whole run 57 min on NVIDIA GeForce RTX 3090

