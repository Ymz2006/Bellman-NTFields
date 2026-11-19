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
