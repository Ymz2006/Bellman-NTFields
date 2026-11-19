# Lift scale: the travel-time lift and the push on larger plus mazes and Gibson floor plans

Study of 2026-11-17. Request (attached page: "2026-11-11 Travel-Time Lift", https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d):

> explore this lift idea. I want you to do more testing with the plus maze and gibson environments, especially the larger ones. Try the
> lift push and also lift approaches. I also want you to see why it doesn't work well in certain scenarios. Use all gpus, 12 hours

The study folders of that page (`jeffrey/lift_warp`, `warp_ntrl`, `chord_warp`, `maxmin_warp`) are not on this workstation (the registry
has no runs between 2026-11-08 and 11-11), so the lift and the push were rebuilt here from the page's description, on the scenes and the
planner every local maze study uses. Two things differ from the page and are part of what was tested:

* **Configuration space instead of the plane.** The local scenes are jeffrey/maze_scale's: a 0.5 x 0.2-cell rectangle robot,
  q = (x, y, phi) with phi = 0.25 theta / E, the exact signed clearance of the rectangle, ntrl-demo's speed label (margin 0.175 cells,
  floor 0.012). The page's scenes were point robots in the plane. In C-space the clearance gradient is not a unit vector
  (nu = |grad_q d| / E, median 1.07-1.16, 90th percentile 1.35-2.1), the case the page lists as not tested.
* **Scenes.** Plus mazes of 22, 30 and 40 cells a side (maze_scale's generator with a centred + of 6, 8 and 12 cells; the 22-cell one
  is the registered prepare of the earlier studies), the P-NTFields apartment at 32 and 64 cells a side (embed_warp's prepares) and three
  Gibson houses as floor plans at 0.25 m per cell: Allensville (43 cells), Lakeville (83), Klickitat (96).

## Method as implemented

    phi(q) = (q, h(q))        h = gain * G(d(q)) / E        G(d) = int_d^margin sqrt(1 / S(d')^2 - 1) dd'    (cells; 0 for d >= margin;
                                                                continued at the floor slope 83.3 down to d = -depth, default -offset)
    T = net(fourier([q0, h0]), fourier([q1, h1]))                one more column in ntrl-demo's Fourier matrix (private generator; B itself is
                                                                drawn from the global RNG as ntrl-demo draws it)
    loss = beta [ ntrl-demo's loss - lambda(epoch) mean(T over blocked pairs with T < bound) + mu ramp mean(relu(sqrt(S |grad T|) - 1)^2 at both ends) ]

`lift.py`: G is a table (G(offset) = 0.685 cells, G(-offset) = 3.60 cells, band length int dd / S = 0.748 cells). h enters the graph as
h(q0) + G'(d) grad_q d . (q - q0) (value and first derivative exact; no second derivatives are needed by the loss). Kinds: `none`
(ntrl-demo), `time` (the lift), `clear` / `dead` (controls), `vec` (the vector lift h n, three extra columns, full Jacobian by autograd).

`trainer.py`: ntrl-demo's loop step for step (as jeffrey/embed_warp/trainer.py). With `--lift none` and no push the loss of the first 60
epochs equals the registered ntrl-demo run of the same seed (20261112T063256Z-fd5d28b2) to every printed digit. Push schedule `page`:
lambda 0 at epoch 200, 1e-2 from 600 to 2,000, 3e-3 from 3,000 on; mu = 0.1; bound 3 box lengths (the largest true travel time between
configurations with clearance >= 0.1 cells is 0.9-1.9 box lengths on these scenes, so the bound never acts on a connected pair). The
one-sided term is faded in with the push's first ramp: at full weight from the first push epoch (`--mu-abrupt`, the first two runs) the
epoch-repeat rule rolled back 1,323 times in 100 epochs (8 minutes). `--mu1` adds a linear (exact) one-sided penalty.

`blocked.py` (stage `blocked`): for every training pair, whether its straight C-space segment crosses an obstacle (clearance below
-offset at a sample every 0.1 cells). 62-85 % of the training pairs are blocked. The page's 48-point check misses 0.2-1.6 % of them.

`scenes.py`: `House` = a Gibson house (~/Downloads/gibson_tiny/<name>/mesh_z_up.obj) cut by five horizontal planes 0.4-1.6 m above the
main floor, rasterized conservatively, gaps under 0.3 m sealed to find the inside, everything else an obstacle; signed distance table,
the rectangle's clearance as the minimum over a 9 x 5 lattice (as embed_warp's apartment). Lindenwood leaks through a wider gap and
was not used.

`house.py` (stage `house`) = jeffrey/embed_warp/prepare.py on a House. `pipeline.py` (stage `field`): train, plan (maze_scale's MPPI,
planning seeds 0 / 1 / 2, exact scoring), then `diag.py`: wall-layer fit S |grad T| by speed bin on held-out band points; travel times
against the prepare run's ground truth on 24,000 random (free configuration, goal) pairs; line-of-sight pairs (T* <= 1.05 D); the
detour index rho = (T - D) / (T* - D) on pairs with T* >= 1.3 D, by detour length; where the runs that did not arrive stopped; the same
field measures at epochs 100 ... 5,000 with planning seed 0 at 1,000 / 2,000 / 3,000.

Read-only forensics on registered checkpoints: `jumps.py` (the step in travel time a field holds across a wall), `heading.py` (radial,
planar and heading slope on line-of-sight pairs), `losscomp.py` (ntrl-demo's loss term by term). `run.py` registers the stages (sources
frozen in `snap/<hash>/`), `queue/runner.py` runs `queue/jobs.txt` (GPU 0 one run, GPU 1 two; shared with the octave_detune session),
`report.py` -> `report/runs.json`, `report/tables.md`; `figs.py`, `page.py` + `pagetext.py` + `pagelib.py` -> `report/lift_scale.html`.

## Findings

Findings page: https://claude.ai/artifact/3NQ9XSVrVgRCMBjaugePSM (`report/lift_scale.html`, built by `page.py`). `report/tables.md` (`report.py`) is regenerated from the
registry and is authoritative for every number; the pooled table is copied at the end of this file. Single-run numbers below are
seed 87358 on the 22-cell plus maze unless a scene is named. One training of the same recipe moves by 3-6 points between seeds.

1. **The lift's own claim holds in the C-space and at every size.** S x |grad T| on held-out band points is 0.93-1.02 in the innermost
   band on every scene with the lift alone, within the first 100 epochs (ntrl-demo 0.05-0.11, worse the larger the scene). nu = |grad_q d| / E
   up to 1.4-2.1 costs nothing measurable.
2. **The lift alone plans below ntrl-demo on 7 of 8 scenes** (55-67 % solved against 56-90 %). Obstacles are seen through (98-100 % of detour pairs), as on the lift page,
   and the far field is short even along free lines of sight, which is new: learned / true 0.88 (plus 22), 0.65 (plus 40), 0.42-0.54
   (apartment, houses). `heading.py`: the field keeps |grad T| = 1 by putting slope into the heading coordinate (radial slope 0.63,
   |dT/dphi| 0.55 beyond 16 cells; ntrl-demo 0.93 and 0.15). Control `--zero-heading` (the network does not see phi directly): line of
   sight learned / true 1.00, 97 % within 10 % with no push (it plans 34 %: a control, not a recipe).
3. **Lift + push (the page's schedule)** fixes the far field at once (line-of-sight pairs within 10 %: 98-100 %) and never arrives
   through a wall; failures are stalls at walls with the goal behind them. Solved %, lift + push vs ntrl-demo (seeds pooled: 3 on plus 30 /
   plus 40 / Lakeville / Klickitat, 2 on Allensville / apartment 64, 1 on plus 22 / apartment 32): plus 22 68.2 vs 65.0, plus 30 64.2
   vs 56.8, plus 40 63.2 vs 59.2, apartment 32 75.4 vs 76.8, Allensville 99.6 vs 90.3, apartment 64 73.0 vs 66.3, Lakeville 76.5 vs
   69.1, Klickitat 85.5 vs 71.2; travel times within 10 %: 89 / 81 / 79 / 38 / 88 / 37 / 87 / 94 % vs 46 / 61 / 52 / 47 / 78 / 20 /
   74 / 77 %. On the thin-walled scenes (plus mazes, apartment) the detours are seen through at the end
   (53-100 % of detour pairs); on the houses mostly not (Allensville 0 %, Klickitat 20 %, Lakeville 55 %).
   On the 22-cell maze they exist at the end of the strong push (detour index 1.64, 87 % solved at epoch 2,000, planning seed 0) and
   are gone 1,000 epochs later (0.45; lift page: 1.66 -> 0.50). On the 40-cell maze they never form (0.22 at epoch 2,000).
4. **Why (four tests).**
   * Constant strong push (`lift_c10`): the index swings 1.02 / 0.67 / 0.54 / 1.17 / 1.01 / 0.33 between checkpoints, line-of-sight
     times 8-13 % long, 404 rollbacks. The quadratic cap's gradient vanishes at the constraint: a push buys a violation proportional
     to its strength, and the detours are made of it.
   * Exact cap (`--mu1 0.05`, `lift_x`) or stiff quadratic cap (`--mu 10`, `liftq`): detour index 0.00 / -0.02, 100 % seen through.
     The largest field that respects the cap at the trained points is the see-through one.
   * Inside pairs (embed_warp data variant 20261112T065945Z-91ae7981: one end of 20 % of the pairs inside obstacles) with the lift
     continued through the walls (`--depth 0.3`): S x slope 1.00-1.03 at every depth inside the walls, line of sight right without a
     push, detour index creeping to 0.45 (5,000 epochs) and 0.3-0.6 (12,500 epochs: 68.1 % solved, 84 % within 10 %, wall layer 0.98).
     Yet `inside.py`: the learned time between mirror configurations across a wall (0.8 cells apart) is 1.3 cells where the straight
     crossing costs 55. The wall is a symmetric bump in the lifted input, T ~ T0(x) + h; the truth needs a step in T0 across the wall's
     thickness at equal height, i.e. planar resolution. Detours saturate near 4 cells.
   * `jumps.py`: ntrl-demo holds a 13.5-cell step across a wall on the 22-cell maze (with its smeared wall layer, times 10 % long) and
     about 1 cell on the 40-cell maze.
5. **Planar resolution is the missing piece.** Octaves (embed_warp's whitened x2, x4, x8 rows; `--octaves cell` scales them with the
   box side): on the 22-cell maze under the strong push the detour index is stable (1.72-1.75 from epoch 600 to 2,000), 91 % solved at
   epoch 2,000, 83 % at 3,000 with 88 % of times within 10 %, then lost by 5,000 (68.1 %). Push kept at 1e-2 (`lift_c10_oct8`): 83.8 %
   solved, times 14 % long, 13.6 % wall hits. Push at 6e-3: 72.4 % (index 0.8-1.5 between checkpoints, 95 % solved at epoch 2,000).
   Controls: ntrl-demo + octaves 68.9 %, lift + octaves without push 58.1 %. 40-cell maze with scaled octaves: lift + push 67.5 %
   solved, 92 % within 10 %; strong push 71.3 % (ntrl-demo 59.2 % / 50 %, ntrl-demo + octaves 60.0 % / 72 %, lift + push without octaves
   63.6 % / 79 %). 30-cell maze: 76.2 % (74.3 / 77.9 / 76.4), 95 % within 10 % (ntrl-demo 56.8 %); 40-cell maze over
   three seeds 68.8 % (69.9 / 68.9 / 67.5).
   **Houses with scaled octaves: Klickitat 98.7 % solved (97.9 / 98.6 / 99.6), 98 % of times within 10 % (ntrl-demo 71.2 %,
   ntrl-demo + octaves 71.4 %, lift + push 85.5 %); Lakeville 97.9 % (97.7 / 97.9 / 98.2), 98 % (ntrl-demo 69.1 %, ntrl-demo + octaves
   65.6 %, lift + push 76.5 %); Allensville 99.9 % (two seeds; lift + push without octaves already 99.6 %; ntrl-demo 90.3 %).** Constant strong push with
   octaves: plus 22 89.4 % (95.1 / 83.8), plus 40 71.3 %, Klickitat 100.0 %, times 9-14 % long.
   The 64-cell apartment does not move (72.8 % with or without octaves, 73.6 % with the strong push; every detour seen through).
   `thick.py`: in the houses lift + push without octaves sees through obstacles thinner than about 1.5-2.5 cells on the straight line
   and respects thicker ones (Klickitat: 100 % seen through at 1-1.5 cells, 14 % at 2.5-4, 4 % at 4-8): wall thickness in cells, not
   scene size, decides where the page's recipe works; the octaves move that threshold below the wall thickness of the houses.
6. **Not adopted:** vector lift h n (62.5 % with the push; with inside pairs 58.6 % / 59.4 %), push scaled with the scene (plus 40:
   index 0.42 at epoch 2,000 with times 15 % long, 64.3 %), base frequencies scaled with the scene (64.9 %, still see-through), a denser
   segment check (the 48-point check misses 0.2-1.6 %), a larger bound (largest T* between free configurations 0.9-1.9 box lengths).
7. **Floor plans.** ntrl-demo is itself see-through on the apartment (65 % / 100 % of detour pairs at 32 / 64 cells) and fails by
   stalling; every lifted arm stays see-through there as well (octaves, one octave higher, strong push, inside pairs). The houses,
   with walls of real thickness and a closed outside, are where the recipe works best (item 3 and item 5).
8. **Open.** Why the apartment (a thin scan outline with free space outside) stays see-through under every arm, strong push and
   octaves included; a push schedule that keeps the octave fields' detours without the 14 % inflation on thin-walled mazes (the state
   at epoch 3,000 of `liftpush_oct8` has both, and is transient); the longest plus-maze queries (32-64 cells) are solved only by
   strong-push fields.

Pitfalls: the one-sided term must be faded in with the push (1,323 rollbacks in 100 epochs otherwise); `plus.install` is once per
process (one plus maze per process in figures and forensics); `report` / `prepare` module names exist in jeffrey/maze_scale, which
`scenes.py` puts first on `sys.path`; the registry-declared inside-pair variants of embed_warp only know its own scenes (no houses).

# Tables (report/tables.md)

## Scenes

| scene | E (cells) | blocked training pairs % | missed by 48 points % | largest T* (box lengths) | nu median / q90 | prepare | blocked |
|---|---:|---:|---:|---:|---:|---|---|
| plus maze 22 x 22 | 22.1 | 75.7 | 0.2 | 3.03 | 1.07 / 1.35 | 20261102T042454Z-62864e61 | 20261117T054324Z-5d148fc2 |
| plus maze 30 x 30 | 30.1 | 81.6 | 0.2 | 2.85 | 1.07 / 1.35 | 20261117T053658Z-9fd2caea | 20261117T055154Z-cfaabec8 |
| plus maze 40 x 40 | 40.1 | 84.5 | 0.3 | 2.49 | 1.07 / 1.35 | 20261117T053700Z-41060313 | 20261117T060610Z-f818c76e |
| apartment, 32 cells | 32.0 | 70.1 | 0.9 | 2.56 | 1.11 / 1.39 | 20261112T081337Z-6b5d9544 | 20261117T054339Z-e11680af |
| apartment, 64 cells | 64.0 | 75.4 | 1.6 | 2.15 | 1.16 / 2.12 | 20261112T063206Z-6e271a95 | 20261117T054355Z-f92d3460 |
| house Allensville, 43 cells | 43.0 | 62.2 | 0.3 | 22.66 | 1.09 / 1.46 | 20261117T054449Z-0144646b | 20261117T055619Z-82b0a941 |
| house Lakeville, 83 cells | 83.0 | 72.8 | 0.3 | 34.47 | 1.09 / 1.43 | 20261117T054504Z-e60b96b5 | 20261117T060740Z-ec2f04f3 |
| house Klickitat, 96 cells | 96.0 | 66.5 | 0.5 | 32.22 | 1.10 / 1.82 | 20261117T063709Z-61a4f754 | 20261117T070515Z-c2424771 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | 22.1 | 80.2 | 0.2 | 3.03 | 1.07 / 1.36 | 20261112T065945Z-91ae7981 | 20261117T075055Z-ad16e810 |
| plus maze 40 x 40, 20 % inside pairs | 40.1 | 87.4 | 0.4 | 2.49 | 1.07 / 1.35 | 20261117T081804Z-72d00fe8 | 20261117T082205Z-01c9ed95 |
| apartment, 64 cells, 20 % inside pairs | 64.0 | 80.1 | 2.6 | 2.15 | 1.16 / 2.11 | 20261117T081808Z-316b6497 | 20261117T082300Z-5bf4cb84 |

## Pooled over training seeds

| scene | arm | seeds | solved % (each seed) | wall % | stall % | T within 10 % | median T/T* | line-of-sight within 10 % | detour index | seen through % | S x slope innermost |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plus maze 22 x 22 | ntrl | 1 | 65.0 (65.0) | 15.1 | 19.9 | 46.3 | 1.104 | 50.6 | 1.37 | 2.8 | 0.08 |
| plus maze 22 x 22 | lift | 1 | 55.0 (55.0) | 0.0 | 45.0 | 25.5 | 0.821 | 43.5 | -0.20 | 98.3 | 1.00 |
| plus maze 22 x 22 | push | 1 | 54.2 (54.2) | 45.8 | 0.0 | 88.2 | 1.031 | 96.6 | 0.73 | 25.7 | 0.04 |
| plus maze 22 x 22 | liftpush | 1 | 68.2 (68.2) | 1.0 | 30.8 | 89.4 | 1.002 | 99.4 | 0.47 | 52.9 | 0.86 |
| plus maze 22 x 22 | lift_push | 1 | 66.4 (66.4) | 0.1 | 33.5 | 88.9 | 1.007 | 99.2 | 0.32 | 66.5 | 0.85 |
| plus maze 22 x 22 | pushonly | 1 | 53.3 (53.3) | 46.7 | 0.0 | 89.1 | 1.029 | 97.8 | 0.52 | 47.2 | 0.04 |
| plus maze 22 x 22 | vecpush | 1 | 62.5 (62.5) | 0.0 | 37.5 | 86.1 | 0.991 | 99.4 | 0.22 | 78.6 | 0.92 |
| plus maze 22 x 22 | lift_c10 | 1 | 68.2 (68.2) | 0.8 | 31.0 | 51.5 | 1.074 | 51.9 | 0.34 | 55.8 | 0.81 |
| plus maze 22 x 22 | lift_x | 1 | 60.3 (60.3) | 0.0 | 39.7 | 69.8 | 0.940 | 98.5 | 0.00 | 100.0 | 0.76 |
| plus maze 22 x 22 | lift_x3 | 1 | 59.7 (59.7) | 0.0 | 40.3 | 62.3 | 0.923 | 96.5 | -0.03 | 99.5 | 0.76 |
| plus maze 22 x 22 | liftq | 1 | 59.0 (59.0) | 0.0 | 41.0 | 67.0 | 0.937 | 98.9 | -0.02 | 100.0 | 0.72 |
| plus maze 22 x 22 | liftpush_oct8 | 1 | 68.1 (68.1) | 0.0 | 31.9 | 68.4 | 0.963 | 80.2 | -0.02 | 77.0 | 0.83 |
| plus maze 22 x 22 | lift_nohead | 1 | 33.9 (33.9) | 0.0 | 66.1 | 78.3 | 0.970 | 97.1 | 0.13 | 97.5 | 0.98 |
| plus maze 22 x 22 | lift_c6_oct8 | 1 | 72.4 (72.4) | 2.5 | 25.1 | 76.5 | 1.059 | 85.1 | 0.79 | 29.8 | 0.87 |
| plus maze 22 x 22 | ntrl_oct8 | 1 | 68.9 (68.9) | 9.6 | 21.5 | 48.4 | 1.100 | 52.4 | 1.61 | 6.2 | 0.20 |
| plus maze 22 x 22 | liftpush_oct | 1 | 69.9 (69.9) | 0.1 | 30.0 | 83.1 | 0.992 | 95.9 | 0.57 | 40.4 | 0.83 |
| plus maze 22 x 22 | lift_r6_oct8 | 1 | 70.7 (70.7) | 0.4 | 28.9 | 78.2 | 1.049 | 88.9 | 0.83 | 31.3 | 0.86 |
| plus maze 22 x 22 | lift_oct8 | 1 | 58.1 (58.1) | 0.0 | 41.9 | 54.0 | 0.910 | 80.8 | 0.08 | 92.5 | 1.00 |
| plus maze 22 x 22 | lift_c10_oct8 | 2 | 89.4 (95.1 / 83.8) | 8.3 | 2.3 | 15.0 | 1.160 | 19.9 | 1.85 | 0.8 | 0.85 |
| plus maze 30 x 30 | ntrl | 3 | 56.8 (56.4 / 58.1 / 56.0) | 28.1 | 15.1 | 61.1 | 1.081 | 65.8 | 1.14 | 6.6 | 0.07 |
| plus maze 30 x 30 | lift | 1 | 57.2 (57.2) | 0.0 | 42.8 | 11.4 | 0.628 | 21.7 | -0.67 | 99.9 | 0.96 |
| plus maze 30 x 30 | liftpush | 3 | 64.2 (64.3 / 64.4 / 63.8) | 0.1 | 35.7 | 81.4 | 0.979 | 99.5 | 0.24 | 90.7 | 0.84 |
| plus maze 30 x 30 | liftpush_oct | 3 | 76.2 (74.3 / 77.9 / 76.4) | 0.7 | 23.1 | 94.9 | 1.017 | 99.2 | 0.87 | 12.6 | 0.86 |
| plus maze 40 x 40 | ntrl | 3 | 59.2 (58.9 / 59.6 / 59.2) | 40.4 | 0.4 | 51.8 | 1.092 | 60.1 | 0.74 | 21.0 | 0.05 |
| plus maze 40 x 40 | lift | 1 | 56.2 (56.2) | 0.0 | 43.8 | 12.0 | 0.641 | 19.6 | -0.84 | 99.9 | 0.97 |
| plus maze 40 x 40 | liftpush | 3 | 63.2 (62.9 / 63.2 / 63.6) | 0.0 | 36.8 | 79.3 | 0.974 | 99.5 | 0.04 | 100.0 | 0.84 |
| plus maze 40 x 40 | pushonly | 1 | 57.1 (57.1) | 42.9 | 0.0 | 89.0 | 1.011 | 99.5 | 0.30 | 89.3 | 0.04 |
| plus maze 40 x 40 | liftpush_cell | 1 | 64.3 (64.3) | 0.0 | 35.7 | 84.8 | 1.004 | 99.3 | 0.13 | 97.3 | 0.81 |
| plus maze 40 x 40 | liftpush_b | 1 | 64.9 (64.9) | 0.0 | 35.1 | 81.5 | 0.983 | 99.9 | 0.06 | 99.9 | 0.83 |
| plus maze 40 x 40 | ntrl_b | 1 | 60.2 (60.2) | 20.6 | 19.2 | 59.7 | 1.075 | 60.4 | 0.54 | 43.8 | 0.06 |
| plus maze 40 x 40 | liftpush_oct | 3 | 68.8 (69.9 / 68.9 / 67.5) | 0.4 | 30.9 | 92.8 | 1.004 | 99.0 | 0.54 | 41.2 | 0.85 |
| plus maze 40 x 40 | ntrl_oct | 1 | 60.0 (60.0) | 8.7 | 31.3 | 72.2 | 0.970 | 85.4 | 0.13 | 91.9 | 0.11 |
| plus maze 40 x 40 | lift_c10_oct | 1 | 71.3 (71.3) | 0.1 | 28.6 | 31.4 | 1.128 | 34.1 | 1.34 | 3.9 | 0.85 |
| plus maze 40 x 40 | liftpush_oct2 | 1 | 66.4 (66.4) | 0.1 | 33.5 | 89.4 | 0.996 | 99.1 | 0.31 | 80.2 | 0.87 |
| apartment, 32 cells | ntrl | 1 | 76.8 (76.8) | 2.5 | 20.7 | 46.8 | 0.899 | 87.4 | 0.25 | 65.4 | 0.11 |
| apartment, 32 cells | lift | 1 | 62.9 (62.9) | 0.0 | 37.1 | 4.6 | 0.416 | 16.6 | -0.73 | 100.0 | 1.02 |
| apartment, 32 cells | liftpush | 1 | 75.4 (75.4) | 0.0 | 24.6 | 38.1 | 0.820 | 99.2 | 0.02 | 100.0 | 0.82 |
| apartment, 32 cells | liftpush_oct | 1 | 76.9 (76.9) | 1.1 | 22.0 | 40.8 | 0.837 | 99.1 | 0.06 | 98.1 | 0.81 |
| apartment, 64 cells | ntrl | 2 | 66.3 (66.0 / 66.5) | 1.3 | 32.4 | 20.0 | 0.705 | 69.9 | -0.16 | 99.9 | 0.06 |
| apartment, 64 cells | lift | 1 | 62.1 (62.1) | 0.0 | 37.9 | 5.0 | 0.503 | 18.2 | -0.65 | 100.0 | 0.93 |
| apartment, 64 cells | liftpush | 2 | 73.0 (73.2 / 72.8) | 0.0 | 27.0 | 37.1 | 0.776 | 99.3 | 0.02 | 99.8 | 0.84 |
| apartment, 64 cells | pushonly | 1 | 66.0 (66.0) | 34.0 | 0.0 | 40.5 | 0.804 | 99.7 | 0.08 | 99.8 | 0.04 |
| apartment, 64 cells | liftpush_oct | 1 | 72.8 (72.8) | 0.0 | 27.2 | 35.7 | 0.773 | 97.5 | 0.01 | 99.9 | 0.87 |
| apartment, 64 cells | ntrl_oct | 1 | 68.5 (68.5) | 0.0 | 31.5 | 19.0 | 0.706 | 67.9 | -0.15 | 100.0 | 0.10 |
| apartment, 64 cells | lift_c10_oct | 1 | 73.6 (73.6) | 0.1 | 26.3 | 40.3 | 0.823 | 94.2 | 0.12 | 99.7 | 0.88 |
| apartment, 64 cells | liftpush_oct2 | 1 | 73.7 (73.7) | 0.0 | 26.3 | 34.5 | 0.764 | 96.5 | -0.01 | 99.9 | 0.86 |
| house Allensville, 43 cells | ntrl | 2 | 90.3 (96.8 / 83.9) | 9.0 | 0.7 | 78.1 | 1.041 | 88.5 | 1.19 | 0.6 | 0.07 |
| house Allensville, 43 cells | lift | 1 | 66.5 (66.5) | 0.0 | 33.5 | 4.9 | 0.617 | 12.7 | -0.43 | 99.9 | 1.00 |
| house Allensville, 43 cells | liftpush | 2 | 99.6 (99.6 / 99.6) | 0.0 | 0.4 | 87.7 | 1.040 | 98.2 | 1.19 | 0.2 | 0.78 |
| house Allensville, 43 cells | liftpush_oct | 2 | 99.9 (100.0 / 99.9) | 0.0 | 0.1 | 82.8 | 1.046 | 97.9 | 1.22 | 0.1 | 0.83 |
| house Lakeville, 83 cells | ntrl | 3 | 69.1 (69.0 / 68.8 / 69.4) | 7.8 | 23.1 | 74.3 | 0.958 | 83.8 | 0.67 | 36.0 | 0.06 |
| house Lakeville, 83 cells | lift | 1 | 54.2 (54.2) | 0.0 | 45.8 | 1.7 | 0.442 | 4.9 | -0.86 | 100.0 | 1.00 |
| house Lakeville, 83 cells | liftpush | 3 | 76.5 (76.8 / 75.8 / 77.0) | 0.0 | 23.5 | 86.7 | 0.985 | 99.7 | 0.45 | 54.4 | 0.79 |
| house Lakeville, 83 cells | liftpush_oct | 3 | 97.9 (97.7 / 97.9 / 98.2) | 0.0 | 2.1 | 98.1 | 1.015 | 98.0 | 1.04 | 1.1 | 0.84 |
| house Lakeville, 83 cells | ntrl_oct | 1 | 65.6 (65.6) | 1.7 | 32.7 | 44.1 | 0.889 | 88.0 | 0.01 | 99.4 | 0.07 |
| house Klickitat, 96 cells | ntrl | 3 | 71.2 (69.3 / 71.7 / 72.5) | 5.4 | 23.5 | 77.2 | 0.952 | 87.3 | 0.80 | 22.1 | 0.05 |
| house Klickitat, 96 cells | lift | 1 | 60.4 (60.4) | 0.0 | 39.6 | 2.9 | 0.483 | 7.2 | -0.60 | 100.0 | 0.96 |
| house Klickitat, 96 cells | liftpush | 3 | 85.5 (85.5 / 85.6 / 85.5) | 0.0 | 14.5 | 93.7 | 0.994 | 99.5 | 0.97 | 21.2 | 0.84 |
| house Klickitat, 96 cells | pushonly | 1 | 60.6 (60.6) | 39.3 | 0.1 | 95.3 | 1.021 | 99.2 | 1.06 | 22.1 | 0.04 |
| house Klickitat, 96 cells | liftpush_oct | 3 | 98.7 (97.9 / 98.6 / 99.6) | 0.0 | 1.3 | 97.8 | 1.022 | 98.3 | 1.10 | 0.9 | 0.87 |
| house Klickitat, 96 cells | ntrl_oct | 1 | 71.4 (71.4) | 0.7 | 27.9 | 53.0 | 0.908 | 64.9 | 0.51 | 48.2 | 0.09 |
| house Klickitat, 96 cells | lift_c10_oct | 1 | 100.0 (100.0) | 0.0 | 0.0 | 30.5 | 1.141 | 55.7 | 1.52 | 0.1 | 0.89 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | liftx_in | 1 | 64.4 (64.4) | 0.0 | 35.6 | 83.6 | 0.962 | 99.0 | 0.31 | 64.2 | 0.76 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | lift_in | 1 | 63.3 (63.3) | 0.0 | 36.7 | 80.8 | 1.007 | 92.3 | 0.45 | 54.4 | 0.96 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | liftpush_in | 1 | 65.3 (65.3) | 0.0 | 34.7 | 87.6 | 1.003 | 99.3 | 0.29 | 77.2 | 0.85 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | lift_in_long | 1 | 68.1 (68.1) | 0.0 | 31.9 | 83.8 | 1.004 | 92.9 | 0.54 | 45.0 | 0.98 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | liftq_in | 1 | 59.0 (59.0) | 0.0 | 41.0 | 66.2 | 0.936 | 99.0 | -0.02 | 100.0 | 0.72 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | liftq3_in | 1 | 60.4 (60.4) | 0.0 | 39.6 | 72.3 | 0.953 | 99.4 | 0.01 | 100.0 | 0.71 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | vecpush_in | 1 | 59.4 (59.4) | 0.0 | 40.6 | 82.2 | 0.989 | 99.4 | 0.16 | 100.0 | 0.94 |
| plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle | vec_in | 1 | 58.6 (58.6) | 0.0 | 41.4 | 82.4 | 0.991 | 97.2 | 0.16 | 97.8 | 0.99 |
| apartment, 64 cells, 20 % inside pairs | lift_in | 1 | 63.6 (63.6) | 0.0 | 36.4 | 6.5 | 0.567 | 23.9 | -0.50 | 100.0 | 0.97 |

## Solved % by true travel distance (cells), pooled


**plus maze 22 x 22**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 98 | 91 | 65 | 37 | 0 |
| lift | 100 | 95 | 85 | 40 | 10 | 0 |
| push | 95 | 92 | 74 | 48 | 16 | 0 |
| liftpush | 100 | 100 | 93 | 81 | 35 | 0 |
| lift_push | 100 | 100 | 95 | 70 | 33 | 0 |
| pushonly | 95 | 89 | 77 | 45 | 14 | 0 |
| vecpush | 100 | 100 | 86 | 65 | 24 | 0 |
| lift_c10 | 100 | 100 | 92 | 77 | 40 | 0 |
| lift_x | 100 | 100 | 87 | 60 | 15 | 0 |
| lift_x3 | 100 | 100 | 85 | 56 | 18 | 0 |
| liftq | 100 | 100 | 83 | 55 | 16 | 0 |
| liftpush_oct8 | 100 | 100 | 96 | 78 | 35 | 0 |
| lift_nohead | 55 | 64 | 47 | 28 | 9 | 0 |
| lift_c6_oct8 | 100 | 100 | 93 | 93 | 48 | 0 |
| ntrl_oct8 | 100 | 95 | 91 | 72 | 36 | 19 |
| liftpush_oct | 100 | 100 | 94 | 80 | 36 | 9 |
| lift_r6_oct8 | 100 | 100 | 89 | 90 | 45 | 0 |
| lift_oct8 | 100 | 100 | 88 | 45 | 16 | 0 |
| lift_c10_oct8 | 100 | 100 | 98 | 100 | 88 | 51 |

**plus maze 30 x 30**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl | 86 | 89 | 79 | 59 | 27 | 1 |
| lift | 95 | 95 | 83 | 52 | 18 | 0 |
| liftpush | 95 | 98 | 91 | 62 | 37 | 2 |
| liftpush_oct | 95 | 99 | 92 | 88 | 64 | 19 |

**plus maze 40 x 40**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 80 | 88 | 78 | 53 | 16 | 0 |
| lift | 100 | 88 | 88 | 78 | 32 | 8 | 0 |
| liftpush | 100 | 86 | 90 | 87 | 54 | 25 | 0 |
| pushonly | 100 | 80 | 88 | 72 | 45 | 15 | 0 |
| liftpush_cell | 100 | 87 | 91 | 88 | 54 | 30 | 0 |
| liftpush_b | 100 | 88 | 90 | 88 | 60 | 28 | 0 |
| ntrl_b | 100 | 85 | 90 | 80 | 52 | 15 | 0 |
| liftpush_oct | 100 | 91 | 93 | 93 | 67 | 37 | 0 |
| ntrl_oct | 100 | 88 | 90 | 80 | 48 | 15 | 0 |
| lift_c10_oct | 100 | 92 | 95 | 92 | 79 | 40 | 0 |
| liftpush_oct2 | 100 | 90 | 92 | 92 | 60 | 30 | 0 |

**apartment, 32 cells**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 98 | 87 | 82 | 41 | 0 |
| lift | 100 | 100 | 82 | 49 | 0 | 0 |
| liftpush | 100 | 100 | 92 | 80 | 27 | 0 |
| liftpush_oct | 100 | 100 | 92 | 81 | 35 | 0 |

**apartment, 64 cells**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 100 | 94 | 76 | 34 | 10 | 0 |
| lift | 100 | 98 | 91 | 70 | 22 | 8 | 0 |
| liftpush | 100 | 100 | 100 | 85 | 54 | 18 | 0 |
| pushonly | 100 | 100 | 95 | 79 | 28 | 10 | 0 |
| liftpush_oct | 100 | 100 | 100 | 85 | 52 | 18 | 0 |
| ntrl_oct | 100 | 100 | 98 | 78 | 43 | 10 | 0 |
| lift_c10_oct | 100 | 100 | 100 | 88 | 52 | 21 | 0 |
| liftpush_oct2 | 100 | 100 | 100 | 82 | 58 | 21 | 0 |

**house Allensville, 43 cells**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 100 | 93 | 87 | 87 | 75 |
| lift | 100 | 100 | 85 | 67 | 48 | 0 |
| liftpush | 100 | 100 | 100 | 100 | 98 | 100 |
| liftpush_oct | 100 | 100 | 100 | 100 | 100 | 100 |

**house Lakeville, 83 cells**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 99 | 90 | 77 | 50 | 37 | 31 |
| lift | 100 | 100 | 95 | 72 | 12 | 0 | 0 |
| liftpush | 100 | 100 | 100 | 99 | 47 | 59 | 30 |
| liftpush_oct | 100 | 100 | 100 | 98 | 92 | 98 | 98 |
| ntrl_oct | 100 | 100 | 98 | 100 | 40 | 16 | 6 |

**house Klickitat, 96 cells**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntrl | 100 | 96 | 87 | 94 | 60 | 47 | 14 |
| lift | 100 | 98 | 90 | 92 | 40 | 2 | 0 |
| liftpush | 100 | 100 | 95 | 98 | 79 | 79 | 49 |
| pushonly | 100 | 95 | 82 | 80 | 42 | 25 | 0 |
| liftpush_oct | 100 | 100 | 100 | 100 | 97 | 99 | 95 |
| ntrl_oct | 100 | 100 | 92 | 92 | 62 | 40 | 12 |
| lift_c10_oct | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

**plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---:|---:|---:|---:|---:|---:|
| liftx_in | 100 | 100 | 88 | 73 | 26 | 0 |
| lift_in | 100 | 100 | 88 | 68 | 25 | 0 |
| liftpush_in | 100 | 100 | 92 | 70 | 29 | 0 |
| lift_in_long | 100 | 100 | 90 | 80 | 38 | 0 |
| liftq_in | 100 | 100 | 84 | 55 | 15 | 0 |
| liftq3_in | 100 | 100 | 88 | 57 | 18 | 0 |
| vecpush_in | 100 | 100 | 82 | 59 | 15 | 0 |
| vec_in | 100 | 100 | 78 | 57 | 18 | 0 |

**apartment, 64 cells, 20 % inside pairs**

| arm | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lift_in | 100 | 100 | 92 | 72 | 28 | 5 | 0 |

## Every run

| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | S x slope innermost | rollbacks | train min | extra |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | ntrl | 87358 | 20261117T054519Z-9a3bc556 | 65.0 | 15.1 | 19.9 | 46.3 | 1.104 | 50.6 | 1.37 | 0.08 | 8 | 5.5 | `--lift none` |
| plus22 | lift | 87358 | 20261117T055249Z-d953394e | 55.0 | 0.0 | 45.0 | 25.5 | 0.821 | 43.5 | -0.20 | 1.00 | 0 | 11.4 | `--lift time` |
| plus22 | push | 87358 | 20261117T060704Z-ebccda2a | 54.2 | 45.8 | 0.0 | 88.2 | 1.031 | 96.6 | 0.73 | 0.04 | 6 | 9.2 | `--lift none --push page` |
| plus22 | liftpush | 87358 | 20261117T061710Z-c95ca37c | 68.2 | 1.0 | 30.8 | 89.4 | 1.002 | 99.4 | 0.47 | 0.86 | 149 | 13.0 | `--lift time --push page` |
| plus22 | lift_push | 87358 | 20261117T055734Z-8c721107 | 66.4 | 0.1 | 33.5 | 88.9 | 1.007 | 99.2 | 0.32 | 0.85 | 1378 | 32.6 | `--lift time --push page` |
| plus22 | pushonly | 87358 | 20261117T062104Z-ea118172 | 53.3 | 46.7 | 0.0 | 89.1 | 1.029 | 97.8 | 0.52 | 0.04 | 0 | 14.8 | `--lift none --push page` |
| plus22 | vecpush | 87358 | 20261117T064144Z-f4a69b84 | 62.5 | 0.0 | 37.5 | 86.1 | 0.991 | 99.4 | 0.22 | 0.92 | 51 | 18.7 | `--lift vec --push page` |
| plus22 | lift_c10 | 87358 | 20261117T063455Z-3b9b5dce | 68.2 | 0.8 | 31.0 | 51.5 | 1.074 | 51.9 | 0.34 | 0.81 | 404 | 31.7 | `--lift time --push const:1e-2` |
| plus22 | lift_x | 87358 | 20261117T072635Z-103dd2fe | 60.3 | 0.0 | 39.7 | 69.8 | 0.940 | 98.5 | 0.00 | 0.76 | 39 | 12.8 | `--lift time --push const:1e-2 --mu1 0.05` |
| plus22 | lift_x3 | 87358 | 20261117T073009Z-ffd1f750 | 59.7 | 0.0 | 40.3 | 62.3 | 0.923 | 96.5 | -0.03 | 0.76 | 0 | 21.8 | `--lift time --push const:3e-3 --mu1 0.05` |
| plus22 | liftq | 87358 | 20261117T085500Z-cabdf4bb | 59.0 | 0.0 | 41.0 | 67.0 | 0.937 | 98.9 | -0.02 | 0.72 | 0 | 13.3 | `--lift time --push const:1e-2 --mu 10` |
| plus22 | liftpush_oct8 | 87358 | 20261117T093110Z-3477083e | 68.1 | 0.0 | 31.9 | 68.4 | 0.963 | 80.2 | -0.02 | 0.83 | 276 | 13.5 | `--lift time --push page --octaves 2,4,8` |
| plus22 | lift_nohead | 87358 | 20261117T094225Z-95aa5afb | 33.9 | 0.0 | 66.1 | 78.3 | 0.970 | 97.1 | 0.13 | 0.98 | 0 | 18.5 | `--lift time --zero-heading` |
| plus22 | lift_c6_oct8 | 87358 | 20261117T095830Z-fd6bf53d | 72.4 | 2.5 | 25.1 | 76.5 | 1.059 | 85.1 | 0.79 | 0.87 | 303 | 13.1 | `--lift time --push const:6e-3 --octaves 2,4,8` |
| plus22 | ntrl_oct8 | 87358 | 20261117T100520Z-398c93e3 | 68.9 | 9.6 | 21.5 | 48.4 | 1.100 | 52.4 | 1.61 | 0.20 | 42 | 9.2 | `--lift none --octaves 2,4,8` |
| plus22 | liftpush_oct | 1 | 20261117T132550Z-58584406 | 69.9 | 0.1 | 30.0 | 83.1 | 0.992 | 95.9 | 0.57 | 0.83 | 306 | 13.3 | `--lift time --push page --octaves 2,4,8` |
| plus22 | lift_r6_oct8 | 87358 | 20261117T100205Z-025079d3 | 70.7 | 0.4 | 28.9 | 78.2 | 1.049 | 88.9 | 0.83 | 0.86 | 304 | 24.0 | `--lift time --push 200:0,600:1e-2,2000:1e-2,3000:6e-3 --octaves 2,4,8` |
| plus22 | lift_oct8 | 87358 | 20261117T102925Z-a51eb1ca | 58.1 | 0.0 | 41.9 | 54.0 | 0.910 | 80.8 | 0.08 | 1.00 | 0 | 11.6 | `--lift time --octaves 2,4,8` |
| plus22 | lift_c10_oct8 | 1 | 20261117T131035Z-0889732b | 95.1 | 2.9 | 1.9 | 13.6 | 1.163 | 18.2 | 1.85 | 0.84 | 374 | 13.7 | `--lift time --push const:1e-2 --octaves 2,4,8` |
| plus22 | lift_c10_oct8 | 87358 | 20261117T101750Z-b9e07f67 | 83.8 | 13.6 | 2.6 | 16.3 | 1.157 | 21.5 | 1.85 | 0.86 | 353 | 23.4 | `--lift time --push const:1e-2 --octaves 2,4,8` |
| plus30 | ntrl | 1 | 20261117T144100Z-3a738b11 | 56.4 | 28.1 | 15.6 | 57.8 | 1.083 | 59.4 | 0.97 | 0.07 | 0 | 7.3 | `--lift none` |
| plus30 | ntrl | 2 | 20261117T154704Z-5db8406b | 58.1 | 23.8 | 18.2 | 70.6 | 1.067 | 75.1 | 1.22 | 0.07 | 0 | 9.4 | `--lift none` |
| plus30 | ntrl | 87358 | 20261117T120340Z-057a1ff3 | 56.0 | 32.4 | 11.7 | 54.9 | 1.091 | 62.9 | 1.22 | 0.06 | 0 | 8.7 | `--lift none` |
| plus30 | lift | 87358 | 20261117T120955Z-aa3b4619 | 57.2 | 0.0 | 42.8 | 11.4 | 0.628 | 21.7 | -0.67 | 0.96 | 0 | 11.4 | `--lift time` |
| plus30 | liftpush | 1 | 20261117T144635Z-6fcd6d04 | 64.3 | 0.3 | 35.4 | 82.8 | 0.983 | 99.6 | 0.28 | 0.84 | 0 | 14.8 | `--lift time --push page` |
| plus30 | liftpush | 2 | 20261117T160244Z-082e04bc | 64.4 | 0.0 | 35.6 | 81.2 | 0.980 | 99.4 | 0.18 | 0.84 | 0 | 12.3 | `--lift time --push page` |
| plus30 | liftpush | 87358 | 20261117T121530Z-ea7160ea | 63.8 | 0.0 | 36.2 | 80.2 | 0.975 | 99.6 | 0.25 | 0.84 | 0 | 23.3 | `--lift time --push page` |
| plus30 | liftpush_oct | 1 | 20261117T142355Z-d7376fc7 | 74.3 | 1.4 | 24.3 | 94.7 | 1.014 | 99.2 | 0.88 | 0.86 | 65 | 22.5 | `--lift time --push page --octaves cell` |
| plus30 | liftpush_oct | 2 | 20261117T152524Z-c6aa3c0b | 77.9 | 0.3 | 21.8 | 96.0 | 1.021 | 99.2 | 0.97 | 0.86 | 110 | 13.0 | `--lift time --push page --octaves cell` |
| plus30 | liftpush_oct | 87358 | 20261117T124440Z-32e35c1d | 76.4 | 0.4 | 23.2 | 94.0 | 1.015 | 99.2 | 0.76 | 0.87 | 53 | 20.0 | `--lift time --push page --octaves cell` |
| plus40 | ntrl | 1 | 20261117T132840Z-70ef00da | 58.9 | 41.1 | 0.0 | 43.8 | 1.104 | 50.9 | 0.76 | 0.05 | 0 | 9.0 | `--lift none` |
| plus40 | ntrl | 2 | 20261117T154029Z-275fd1a2 | 59.6 | 39.3 | 1.1 | 62.1 | 1.076 | 72.7 | 0.73 | 0.05 | 0 | 7.2 | `--lift none` |
| plus40 | ntrl | 87358 | 20261117T070240Z-9926f070 | 59.2 | 40.8 | 0.0 | 49.6 | 1.096 | 56.8 | 0.73 | 0.05 | 0 | 7.4 | `--lift none` |
| plus40 | lift | 87358 | 20261117T070630Z-a905919d | 56.2 | 0.0 | 43.8 | 12.0 | 0.641 | 19.6 | -0.84 | 0.97 | 0 | 16.9 | `--lift time` |
| plus40 | liftpush | 1 | 20261117T134015Z-18ee0433 | 62.9 | 0.0 | 37.1 | 79.4 | 0.978 | 99.6 | 0.04 | 0.84 | 0 | 20.6 | `--lift time --push page` |
| plus40 | liftpush | 2 | 20261117T155909Z-cd7d27bf | 63.2 | 0.0 | 36.8 | 80.1 | 0.975 | 99.6 | 0.06 | 0.84 | 0 | 17.5 | `--lift time --push page` |
| plus40 | liftpush | 87358 | 20261117T071105Z-e272648c | 63.6 | 0.0 | 36.4 | 78.5 | 0.970 | 99.4 | 0.03 | 0.83 | 0 | 12.5 | `--lift time --push page` |
| plus40 | pushonly | 87358 | 20261117T071140Z-28649db0 | 57.1 | 42.9 | 0.0 | 89.0 | 1.011 | 99.5 | 0.30 | 0.04 | 0 | 16.1 | `--lift none --push page` |
| plus40 | liftpush_cell | 87358 | 20261117T073145Z-8ed28572 | 64.3 | 0.0 | 35.7 | 84.8 | 1.004 | 99.3 | 0.13 | 0.81 | 0 | 21.6 | `--lift time --push page --lam-scale cell` |
| plus40 | liftpush_b | 87358 | 20261117T080615Z-fd5a32ff | 64.9 | 0.0 | 35.1 | 81.5 | 0.983 | 99.9 | 0.06 | 0.83 | 0 | 12.5 | `--lift time --push page --b-scale cell` |
| plus40 | ntrl_b | 87358 | 20261117T081650Z-33b64b44 | 60.2 | 20.6 | 19.2 | 59.7 | 1.075 | 60.4 | 0.54 | 0.06 | 0 | 9.1 | `--lift none --b-scale cell` |
| plus40 | liftpush_oct | 1 | 20261117T132625Z-7f8503a3 | 69.9 | 0.8 | 29.3 | 93.7 | 1.011 | 98.5 | 0.67 | 0.86 | 16 | 22.4 | `--lift time --push page --octaves 3.6,7.3,14.5` |
| plus40 | liftpush_oct | 2 | 20261117T150949Z-990c1e77 | 68.9 | 0.2 | 30.8 | 92.3 | 0.997 | 99.3 | 0.47 | 0.84 | 9 | 12.3 | `--lift time --push page --octaves 3.6,7.3,14.5` |
| plus40 | liftpush_oct | 87358 | 20261117T101315Z-a6b50606 | 67.5 | 0.0 | 32.5 | 92.4 | 1.005 | 99.3 | 0.47 | 0.86 | 30 | 12.8 | `--lift time --push page --octaves 3.6,7.3,14.5` |
| plus40 | ntrl_oct | 87358 | 20261117T103020Z-ddb04bcd | 60.0 | 8.7 | 31.3 | 72.2 | 0.970 | 85.4 | 0.13 | 0.11 | 0 | 9.0 | `--lift none --octaves 3.6,7.3,14.5` |
| plus40 | lift_c10_oct | 87358 | 20261117T112255Z-51fbe28f | 71.3 | 0.1 | 28.6 | 31.4 | 1.128 | 34.1 | 1.34 | 0.85 | 57 | 21.3 | `--lift time --push const:1e-2 --octaves cell` |
| plus40 | liftpush_oct2 | 87358 | 20261117T124305Z-4468d75d | 66.4 | 0.1 | 33.5 | 89.4 | 0.996 | 99.1 | 0.31 | 0.87 | 26 | 20.0 | `--lift time --push page --octaves 7.3,14.5,29` |
| apt32 | ntrl | 87358 | 20261117T060834Z-9f5630ef | 76.8 | 2.5 | 20.7 | 46.8 | 0.899 | 87.4 | 0.25 | 0.11 | 158 | 9.6 | `--lift none` |
| apt32 | lift | 87358 | 20261117T063139Z-f6d1d0a8 | 62.9 | 0.0 | 37.1 | 4.6 | 0.416 | 16.6 | -0.73 | 1.02 | 1 | 8.7 | `--lift time` |
| apt32 | liftpush | 87358 | 20261117T104430Z-8cfce33d | 75.4 | 0.0 | 24.6 | 38.1 | 0.820 | 99.2 | 0.02 | 0.82 | 526 | 19.7 | `--lift time --push page` |
| apt32 | liftpush_oct | 87358 | 20261117T125750Z-f64b6b8c | 76.9 | 1.1 | 22.0 | 40.8 | 0.837 | 99.1 | 0.06 | 0.81 | 724 | 11.1 | `--lift time --push page --octaves cell` |
| apt64 | ntrl | 1 | 20261117T135640Z-aa85dfe7 | 66.0 | 0.8 | 33.2 | 19.4 | 0.709 | 68.5 | -0.15 | 0.06 | 0 | 8.7 | `--lift none` |
| apt64 | ntrl | 87358 | 20261117T074120Z-8be2ce34 | 66.5 | 1.9 | 31.6 | 20.5 | 0.701 | 71.3 | -0.16 | 0.06 | 0 | 7.4 | `--lift none` |
| apt64 | lift | 87358 | 20261117T085615Z-a115c756 | 62.1 | 0.0 | 37.9 | 5.0 | 0.503 | 18.2 | -0.65 | 0.93 | 2 | 11.8 | `--lift time` |
| apt64 | liftpush | 1 | 20261117T140335Z-cd2b2f8e | 73.2 | 0.0 | 26.8 | 36.8 | 0.777 | 99.2 | 0.02 | 0.85 | 0 | 10.3 | `--lift time --push page` |
| apt64 | liftpush | 87358 | 20261117T085650Z-09015182 | 72.8 | 0.0 | 27.2 | 37.4 | 0.775 | 99.4 | 0.02 | 0.84 | 0 | 19.0 | `--lift time --push page` |
| apt64 | pushonly | 87358 | 20261117T092335Z-f4dfe69e | 66.0 | 34.0 | 0.0 | 40.5 | 0.804 | 99.7 | 0.08 | 0.04 | 0 | 16.6 | `--lift none --push page` |
| apt64 | liftpush_oct | 87358 | 20261117T110805Z-0d2f51e8 | 72.8 | 0.0 | 27.2 | 35.7 | 0.773 | 97.5 | 0.01 | 0.87 | 418 | 11.1 | `--lift time --push page --octaves cell` |
| apt64 | ntrl_oct | 87358 | 20261117T114700Z-265cf951 | 68.5 | 0.0 | 31.5 | 19.0 | 0.706 | 67.9 | -0.15 | 0.10 | 0 | 7.6 | `--lift none --octaves cell` |
| apt64 | lift_c10_oct | 87358 | 20261117T112430Z-aa09ca34 | 73.6 | 0.1 | 26.3 | 40.3 | 0.823 | 94.2 | 0.12 | 0.88 | 584 | 17.8 | `--lift time --push const:1e-2 --octaves cell` |
| apt64 | liftpush_oct2 | 87358 | 20261117T123430Z-9054c5dd | 73.7 | 0.0 | 26.3 | 34.5 | 0.764 | 96.5 | -0.01 | 0.86 | 200 | 10.5 | `--lift time --push page --octaves 11.6,23.2,46.4` |
| Allensville | ntrl | 1 | 20261117T143525Z-fdfc256c | 96.8 | 2.4 | 0.8 | 89.6 | 1.017 | 94.7 | 1.12 | 0.07 | 0 | 8.9 | `--lift none` |
| Allensville | ntrl | 87358 | 20261117T121605Z-9c558da3 | 83.9 | 15.6 | 0.6 | 66.5 | 1.064 | 82.2 | 1.26 | 0.07 | 1 | 9.0 | `--lift none` |
| Allensville | lift | 87358 | 20261117T122340Z-11cef54b | 66.5 | 0.0 | 33.5 | 4.9 | 0.617 | 12.7 | -0.43 | 1.00 | 1 | 8.8 | `--lift time` |
| Allensville | liftpush | 1 | 20261117T142930Z-33b3f06f | 99.6 | 0.0 | 0.4 | 88.2 | 1.037 | 98.7 | 1.19 | 0.78 | 74 | 10.4 | `--lift time --push page` |
| Allensville | liftpush | 87358 | 20261117T122655Z-ff2c4b49 | 99.6 | 0.0 | 0.4 | 87.1 | 1.043 | 97.8 | 1.20 | 0.78 | 47 | 15.4 | `--lift time --push page` |
| Allensville | liftpush_oct | 1 | 20261117T144930Z-b6ef0ddf | 100.0 | 0.0 | 0.0 | 82.8 | 1.046 | 97.9 | 1.22 | 0.84 | 0 | 9.8 | `--lift time --push page --octaves cell` |
| Allensville | liftpush_oct | 87358 | 20261117T130845Z-1fda1189 | 99.9 | 0.0 | 0.1 | 82.8 | 1.045 | 97.9 | 1.22 | 0.83 | 1 | 17.8 | `--lift time --push page --octaves cell` |
| Lakeville | ntrl | 1 | 20261117T140830Z-1e68d6a5 | 69.0 | 7.0 | 23.9 | 71.5 | 0.955 | 82.5 | 0.56 | 0.06 | 0 | 9.2 | `--lift none` |
| Lakeville | ntrl | 2 | 20261117T153034Z-153e533a | 68.8 | 9.4 | 21.8 | 76.2 | 0.958 | 85.9 | 0.74 | 0.05 | 0 | 10.8 | `--lift none` |
| Lakeville | ntrl | 87358 | 20261117T094420Z-31c51802 | 69.4 | 7.0 | 23.6 | 75.3 | 0.961 | 83.1 | 0.72 | 0.06 | 0 | 9.7 | `--lift none` |
| Lakeville | lift | 87358 | 20261117T094615Z-a1c691d1 | 54.2 | 0.0 | 45.8 | 1.7 | 0.442 | 4.9 | -0.86 | 1.00 | 0 | 8.8 | `--lift time` |
| Lakeville | liftpush | 1 | 20261117T141125Z-43f2550a | 76.8 | 0.0 | 23.2 | 86.7 | 0.985 | 99.6 | 0.44 | 0.81 | 19 | 17.0 | `--lift time --push page` |
| Lakeville | liftpush | 2 | 20261117T154854Z-ca32082b | 75.8 | 0.0 | 24.2 | 86.8 | 0.987 | 99.8 | 0.47 | 0.80 | 32 | 10.5 | `--lift time --push page` |
| Lakeville | liftpush | 87358 | 20261117T104235Z-b6379951 | 77.0 | 0.0 | 23.0 | 86.4 | 0.984 | 99.7 | 0.43 | 0.75 | 21 | 10.3 | `--lift time --push page` |
| Lakeville | liftpush_oct | 1 | 20261117T141640Z-8edc7161 | 97.7 | 0.0 | 2.3 | 98.1 | 1.015 | 97.7 | 1.05 | 0.85 | 1 | 10.4 | `--lift time --push page --octaves cell` |
| Lakeville | liftpush_oct | 2 | 20261117T150934Z-37e15b74 | 97.9 | 0.0 | 2.1 | 98.2 | 1.017 | 98.0 | 1.04 | 0.83 | 8 | 16.7 | `--lift time --push page --octaves cell` |
| Lakeville | liftpush_oct | 87358 | 20261117T115130Z-4d5a41b5 | 98.2 | 0.0 | 1.8 | 98.2 | 1.014 | 98.3 | 1.04 | 0.84 | 30 | 17.8 | `--lift time --push page --octaves cell` |
| Lakeville | ntrl_oct | 87358 | 20261117T124735Z-44651d4b | 65.6 | 1.7 | 32.7 | 44.1 | 0.889 | 88.0 | 0.01 | 0.07 | 3 | 7.4 | `--lift none --octaves cell` |
| Klickitat | ntrl | 1 | 20261117T134051Z-0075f71c | 69.3 | 6.1 | 24.6 | 75.3 | 0.947 | 85.9 | 0.78 | 0.06 | 0 | 7.4 | `--lift none` |
| Klickitat | ntrl | 2 | 20261117T153019Z-18f6316c | 71.7 | 6.9 | 21.4 | 83.4 | 0.967 | 90.3 | 0.86 | 0.05 | 0 | 10.8 | `--lift none` |
| Klickitat | ntrl | 87358 | 20261117T104445Z-0c50416c | 72.5 | 3.2 | 24.3 | 72.8 | 0.940 | 85.9 | 0.75 | 0.05 | 0 | 9.6 | `--lift none` |
| Klickitat | lift | 87358 | 20261117T105600Z-740e99b6 | 60.4 | 0.0 | 39.6 | 2.9 | 0.483 | 7.2 | -0.60 | 0.96 | 0 | 8.7 | `--lift time` |
| Klickitat | liftpush | 1 | 20261117T135026Z-f929b0d1 | 85.5 | 0.0 | 14.5 | 93.0 | 0.999 | 99.6 | 1.00 | 0.83 | 29 | 10.2 | `--lift time --push page` |
| Klickitat | liftpush | 2 | 20261117T154720Z-7e775064 | 85.6 | 0.0 | 14.4 | 93.5 | 0.991 | 99.7 | 0.95 | 0.82 | 4 | 17.8 | `--lift time --push page` |
| Klickitat | liftpush | 87358 | 20261117T110015Z-d7107d88 | 85.5 | 0.0 | 14.5 | 94.6 | 0.991 | 99.2 | 0.95 | 0.87 | 12 | 15.9 | `--lift time --push page` |
| Klickitat | pushonly | 87358 | 20261117T110730Z-676b1fb1 | 60.6 | 39.3 | 0.1 | 95.3 | 1.021 | 99.2 | 1.06 | 0.04 | 0 | 14.4 | `--lift none --push page` |
| Klickitat | liftpush_oct | 1 | 20261117T115705Z-1a049ef2 | 97.9 | 0.0 | 2.1 | 98.0 | 1.013 | 98.4 | 1.08 | 0.85 | 0 | 9.9 | `--lift time --push page --octaves cell` |
| Klickitat | liftpush_oct | 2 | 20261117T150919Z-90bec7a7 | 98.6 | 0.0 | 1.4 | 97.6 | 1.027 | 98.0 | 1.11 | 0.85 | 0 | 16.7 | `--lift time --push page --octaves cell` |
| Klickitat | liftpush_oct | 87358 | 20261117T112200Z-7fa766e3 | 99.6 | 0.0 | 0.4 | 97.9 | 1.024 | 98.4 | 1.11 | 0.90 | 0 | 10.2 | `--lift time --push page --octaves cell` |
| Klickitat | ntrl_oct | 87358 | 20261117T114735Z-91b46baa | 71.4 | 0.7 | 27.9 | 53.0 | 0.908 | 64.9 | 0.51 | 0.09 | 2 | 9.4 | `--lift none --octaves cell` |
| Klickitat | lift_c10_oct | 87358 | 20261117T113425Z-cb165da0 | 100.0 | 0.0 | 0.0 | 30.5 | 1.141 | 55.7 | 1.52 | 0.89 | 0 | 10.3 | `--lift time --push const:1e-2 --octaves cell` |
| plus22in | liftx_in | 87358 | 20261117T075149Z-b6173dbb | 64.4 | 0.0 | 35.6 | 83.6 | 0.962 | 99.0 | 0.31 | 0.76 | 6 | 12.5 | `--lift time --depth 0.3 --push const:1e-2 --mu1 0.05` |
| plus22in | lift_in | 87358 | 20261117T075605Z-a19f8f34 | 63.3 | 0.0 | 36.7 | 80.8 | 1.007 | 92.3 | 0.45 | 0.96 | 1 | 16.5 | `--lift time --depth 0.3` |
| plus22in | liftpush_in | 87358 | 20261117T080140Z-8fdea6c6 | 65.3 | 0.0 | 34.7 | 87.6 | 1.003 | 99.3 | 0.29 | 0.85 | 41 | 25.1 | `--lift time --depth 0.3 --push page` |
| plus22in | lift_in_long | 87358 | 20261117T082355Z-441a3707 | 68.1 | 0.0 | 31.9 | 83.8 | 1.004 | 92.9 | 0.54 | 0.98 | 25 | 28.7 | `--lift time --depth 0.3 --epochs 12500` |
| plus22in | liftq_in | 87358 | 20261117T083030Z-4c43e0a1 | 59.0 | 0.0 | 41.0 | 66.2 | 0.936 | 99.0 | -0.02 | 0.72 | 0 | 21.5 | `--lift time --depth 0.3 --push const:1e-2 --mu 10` |
| plus22in | liftq3_in | 87358 | 20261117T083105Z-2481144a | 60.4 | 0.0 | 39.6 | 72.3 | 0.953 | 99.4 | 0.01 | 0.71 | 0 | 21.7 | `--lift time --depth 0.3 --push const:3e-2 --mu 10` |
| plus22in | vecpush_in | 87358 | 20261117T090945Z-a7809efc | 59.4 | 0.0 | 40.6 | 82.2 | 0.989 | 99.4 | 0.16 | 0.94 | 76 | 19.1 | `--lift vec --depth 0.3 --push page` |
| plus22in | vec_in | 87358 | 20261117T091540Z-f0633d57 | 58.6 | 0.0 | 41.4 | 82.4 | 0.991 | 97.2 | 0.16 | 0.99 | 102 | 23.7 | `--lift vec --depth 0.3` |
| apt64in | lift_in | 87358 | 20261117T130920Z-f9e9ce99 | 63.6 | 0.0 | 36.4 | 6.5 | 0.567 | 23.9 | -0.50 | 0.97 | 15 | 11.4 | `--lift time --depth 0.3` |

## Failed jobs

* `house|Klickitat|--name Klickitat` -> 20261117T055524Z-d8fade8c
