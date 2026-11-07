# Maze scale: how ntrl-demo copes with bigger mazes, and whether segment DP helps long trajectories

Request (2026-11-01): test ntrl-demo on small and big environments. Five 2-D mazes, planning in SE(2) with rotation,
mazes as `.obj`; query ntrl-demo with queries binned by distance; then try segment DP and see whether it helps long
trajectories. Follow-ups (2026-11-01/02): replace the 40 × 40 maze by 10 × 10 and drop the zoomed-input control; fewer
DP levels; more split candidates; higher Fourier frequencies; segment DP on a frequency-scaled base; a plus-shaped
opening in the 22 × 22 maze; u-turns around wall ends.

Everything below is from registered runs (run index at the end); success = ntrl-demo's MPPI over planning seeds 0/1/2,
pooled over training seeds 87358/1/2 unless a cell says otherwise. Full tables: `report/tables.md`; near-wall
mechanism: `report/wallfeel.md`; failure geometry: `report/posthoc.md`; follow-ups: `report/extras.md`. Findings page:
https://claude.ai/artifact/JuQxHyTA2siUza9a7ESHRv (built by `page.py`).

## Summary

![success by distance](report/success.png)

*MPPI success by true travel distance; ntrl-demo pools training seeds 87358 / 1 / 2 (whisker = seed range), so do
segment DP + wall term (weight 0.3) on 10×10 and 13×13 and plain segment DP on 13×13; every other point is training
seed 87358.*

1. **ntrl-demo is perfect up to the size of its own demo maze and degrades beyond it.** On 4×4 and 7×7 mazes it
   solves 1,077/1,080 and 1,437/1,440 runs. On 10×10 it solves 88 % (1,583/1,800; training seeds 83–93 %), on 13×13
   59 % (1,060/1,800; 53–66 %), on 22×22 23 % (505/2,160). The oracle (the same MPPI on the true travel time) solves
   100 % of every bin of every maze, so the losses are the field's.
2. **Distance matters only once the maze is big, and even short queries suffer there.** At 7×7 success is flat in
   distance. At 13×13 it falls from 94 % (0.5–2 cells) to 19 % (16–32 cells); at 22×22 from 75 % to 0 %. A 2–4-cell
   query is solved 100 % on 7×7, 92 % on 10×10, 78 % on 13×13 and 32 % on 22×22: the field degrades everywhere as the
   maze grows, and long queries add up local errors.
3. **ntrl-demo fails by hitting walls, and the walls fade out of its field.** Planning seed 0, training seed 87358:
   every failed run on 13×13 touches a wall while driving, and 174 of 190 on 22×22 (the other 16 stall, in the two
   longest bins). The learned speed 1/|∇T| matches the true speed less and less (log-correlation on held-out pairs
   0.96, 0.92, 0.81, 0.69, 0.32 from 4 to 22 cells). At near-wall poses (gap 2–8 cells) the field's rise into the wall
   (S·|∇T|, target 1) falls 0.89, 0.86, 0.62, 0.48, 0.24 and its direction (cosine with "into the wall") 0.99, 0.98,
   0.96, 0.92, 0.56.
4. **Control: input frequency explains much of it.** Multiplying ntrl-demo's Fourier matrix B by s (same random draw,
   nothing else changed; `fourier.py`) gives 95 % on 10×10 at s = 3, 90–91 % on 13×13 at s = 3–6 and 43 % on 22×22
   at s = 6 (still rising). Too high a frequency for the maze breaks planning by stalls (7×7 at s = 3: 78 %; 10×10 at
   s = 6: 51 %), although the speed fit keeps rising.
5. **Segment DP as a value chain makes the field better and the planner worse.** Interval DP over geometric gap
   segments (0.5 cells up to the maze diameter, one ntrl-demo network per segment, each fit to min-over-split-point
   targets built from the shorter segments, ntrl-demo's own field below 0.5 cells) fixes the long-range travel times
   (13×13, 16–32 cells: 1.36 × T* vs 2.08) and descent directions, yet plans worse on every maze that is not already
   solved: 13×13 262 / 280 / 263 vs 347 / 320 / 393 of 600 (training seeds 87358 / 1 / 2), 10×10 421 vs 500/600,
   22×22 118 vs 152/720, 7×7 459 vs 480/480 (training seed 87358). Mechanism: the DP levels are fit to values only, so
   they lose the wall barrier that ntrl-demo's near-wall terms give its own field (13×13 near-wall slope 0.24–0.29 vs
   0.41–0.48, direction 0.64–0.73 vs 0.89–0.92), and MPPI collides sooner. Excluding through-wall base pieces from the
   DP (22×22: 132/720) does not change that.
6. **With ntrl-demo's wall term in every DP level, segment DP helps on 13×13 and 22×22.** Fitting ntrl-demo's normal
   (wall-direction) term at near-wall poses in each level (`segdp3.py`) brings the barrier back and the chains stop
   hitting walls: their failures become stalls in local minima of the long-range field. At weight 1 the long-range
   values drift low and long queries stall; at weight 0.3 the chain solves 451 / 456 / 420 of 600 on 13×13 against
   ntrl-demo's 347 / 320 / 393 (74 % vs 59 %), better at every distance (16–32 cells 31 % vs 19 %), and 308 vs 152/720
   on 22×22 (seed 87358). Where ntrl-demo is already good it costs a little: 10×10 504 / 476 / 467 vs 500 / 523 / 560
   (80 % vs 88 %, losing at 16–32 cells: 32 % vs 66 %), 7×7 454 vs 480.
7. **Follow-ups** (section 4): doubling the gap per level does not work (6 networks: 58 %); more split candidates make
   the targets more accurate but planning saturates at 512 (76 %); the DP chain on a frequency-scaled base has more
   accurate values but more local minima and plans worse than either parent (13×13 66 %, 22×22 37 %); opening a
   6-cell + in the 22×22 maze lifts ntrl-demo to 67 % (99 % for queries inside the +); a 2-cell u-turn around a wall
   end degrades with maze size like everything else (ntrl-demo 100 → 3 %), and the scaled DP chain solves it best on
   the large mazes (89 % on 13×13, 73 % on 22×22).

So, for the question: segment DP as a pure value chain does not help long trajectories in these mazes, because a long
trajectory fails by hitting a wall somewhere along the way, which needs a sharp local barrier that value regression does
not keep. With ntrl-demo's own wall term in every level (weight 0.3) it does help where ntrl-demo struggles (74 % vs
59 % on 13×13, 43 % vs 23 % on 22×22). The longest trajectories are then limited by stalls in local minima of the
long-range field, which neither more split candidates nor a higher-frequency base removes; a higher input frequency for
ntrl-demo alone is the larger single gain on 10×10 and 13×13.

<!--SETUP-->

## Results

### 1. ntrl-demo by maze size and distance

MPPI success (%), pooled over 3 training seeds × 3 planning seeds (per-seed values in `report/tables.md`):

<!--TABLE1-->

Field diagnostics (median per bin, mean over training seeds; `report/tables.md`): T(start, goal)/T* is 0.91–0.93
at 4×4 and 0.96–1.05 at 7×7, then 1.05–1.80 at 10×10, 1.12–2.08 at 13×13 (the long bins overestimate by 2×) and
1.01–1.43 at 22×22. The cosine between the field's descent direction at the start and the truth's is 0.91–0.97 at 4–7
cells, 0.82–0.95 at 10–13 cells and falls with distance at 22×22, to 0.06 in the longest bin. Spearman correlation of T
with T* over all queries stays 0.97–0.99 at every size: the field orders goals by distance even when it cannot plan to
them.

### 2. Mechanism: the walls fade out

![speed maps](report/speedmaps.png)

The learned speed 1/|∇_x T(x, goal)| at heading 0 (middle row) redraws every wall of the 7×7 maze, most of the 10×10
and 13×13 walls and a connected subset of the 22×22 walls; the bottom row is ntrl-demo with B × 3. Held-out near-wall
poses (S < 0.3; median S·|∇T| / cosine of ∇T with the into-wall direction, `report/wallfeel.md`):

<!--TABLE2-->

Failure geometry (`report/posthoc.md`, planning seed 0, training seed 87358; the goal the scorer appends to a run
that never converged is excluded): on 22×22 the first wall contact comes after a median 0.4–0.7 cells for queries up to
4 cells and 1.1–3.5 cells for longer ones; on 13×13 after 0.8–4.3 cells (22–51 % of the route).

### 3. Segment DP

`segdp.py`: the interval-DP transition of `jeffrey/segment_dp` on maze C-space gaps,
T_(m+1)(i, j) = min_k T_≤m(i, k) + T_≤m(k, j) with |i − k|, |k − j| ≤ e_m (1 + blend). Edges are geometric in cells
(0.5, ×1.35, … to the 99.9th-percentile pool gap; 8 levels at 4×4, 14 at 22×22, blend ±5 %); below 0.5 cells the chain
is ntrl-demo's trained field; each level is an ntrl-demo `metric` network warm-started from the previous level and fit
for 6,000 steps (batch 2,000, AdamW 5·10⁻⁴) with |log T − log y| to the DP targets of 100,000 pool pairs (200,000
collision-free pool poses; 128 split candidates per pair, half nearest the straight line, half uniform in the lens).
The chain is planned with the same MPPI and diagnosed the same way. Variants: `--check-pieces` (a base piece counts
only if its straight segment is collision-free), `segdp3.py` (ntrl-demo's normal term at
near-wall poses paired with goals in each level's segment, weight 1 as `jeffrey/segment_dp` dp3 `--normal-weight`,
and 0.3).

<!--TABLE3-->

Per-level truth probes (pool pose → truth goal pairs in each segment): the DP targets track the trained levels to
0.2–3 % (held-out |log|) and sit at 1.07–1.36 × T* on 13×13 and 22×22 (ntrl-demo's own short-range bias, 1.03–1.2,
plus split-point detours), against 1.03–1.96 × for ntrl-demo itself at the same gaps on 13×13.

### 4. Follow-ups

- **Fewer networks** (`--ratio`, 13×13, weight 0.3): r = 2 (6 networks) 58 %, 1.641 (8) 60 %, 1.470 (10) 69 %, 1.35
  (13) 74 %. At the top of a band the lens of admissible split points has half-width R√(1 − (r/2)²), zero at r = 2, so
  a detour around a wall has no split point; targets overestimate and compound (top level 1.42 × T* with 6 networks,
  1.11 × with 13).
- **Split candidates** (`--candidates`, 13×13, weight 0.3): 32 → 69 %, 128 → 74 %, 512 → 76 % (three seeds), 2,048 →
  453/600 (seed 87358; 465 at 512). Top-level target / T* 1.22, 1.12, 1.08, 1.04. Failures remain stalls (143 of 147 at
  2,048 never reach the goal).
- **Fourier scale** (`fourier.py`): see summary item 4 and the table below; the over-scaled failures (7×7 s = 3,
  10×10 s = 6) are all stalls.
- **Segment DP on the scaled base** (`segdp3.py --base` a `fourier.py` checkpoint; every level inherits the scaled B):
  13×13 B × 3 66 % (394 / 378 / 416) vs 74 % plain-base chain and 90 % ntrl-demo B × 3; 22×22 B × 6 37 % (263 / 256 /
  282 of 720) vs 43 % for both. Its values are more accurate (13×13 top target 1.07 × T* vs 1.11–1.14; near-wall slope
  0.87–0.93 vs 0.75–0.82) but it has more local minima of T(x, goal) along true paths of ≥ 8 cells (13×13: 5.6–7.6 per
  path vs 3.4–4.7 for the plain-base chain and 0–0.1 for ntrl-demo; 22×22: 11–13 vs 6.7 and 0.3–1.9), not
  concentrated in blend zones (33 % of minima in 32 % of path length). The DP levels are fit to values only, so
  nothing constrains the gradient between samples; a higher-frequency embedding turns the residual error into short
  ripples, and MPPI stalls in them. ntrl-demo's eikonal loss constrains |∇T| and keeps its field monotone at these
  scales.
- **Plus maze** (`plusprepare.py`, `plusntrl.py`; `plus.py` patches the maze generator in-process): a centred + of
  width 6 cells, interior walls removed (168 of 458; 228 of 484 cells). ntrl-demo 67 % (468 / 479 / 490 of 720) vs
  23 % on the original maze; speed fit 0.53 vs 0.32. Both ends inside the + 99 %, one end 45 %, neither 18 % (mostly
  32–64-cell queries, 0 % for every field). The straight line also gains (49 % vs 13 %), but ntrl-demo's margin over
  it grows too (15–34 vs 5–22 points per bin from 2 to 32 cells).
- **U-turns** (`uturn.py`, `uturneval.py`): start and goal in the two cells beside a wall's last segment (1 cell apart,
  path round the end 2.2–2.3 cells), 40 per maze. ntrl-demo 100 / 99 / 83 / 30 / 3 % on 4 / 7 / 10 / 13 / 22 cells,
  mostly by cutting through the wall end; B × 3 96 % (10×10), 81 % (13×13); DP + wall term 97 / 79 / 29 %; DP on the
  scaled base 89 % (13×13, B × 3) and 73 % (22×22, B × 6).
- **Level viewer** (`levelexplorer.py` → `report/levelexplorer/`, artifact https://claude.ai/artifact/XunBq8wPMFfm2LPgk9jwbs;
  static version `levelfields.py`): the travel-time field of each chain level on 13×13 and 22×22 next to the truth,
  for one query per level band.

<!--EXTRAS-->

## Caveats

- ntrl-demo has three training seeds on every maze; segment DP + wall term (weight 0.3) on 10×10 and 13×13, plain
  segment DP and weight 1 on 13×13, and the 13×13 follow-up variants (unless their table says otherwise) have three;
  every other DP number is training seed 87358 only. ntrl-demo's 13×13 seeds differ by up to 73 runs of 600 (320–393), mostly in the
  long bins, so single-seed differences smaller than that are not separations (the 22×22 wall-term gain,
  +156, is larger than the seed spread there: 152–186).
- The wall-term weight was tried at 1 and 0.3 only, on 13×13 first (one seed) and then carried to the other mazes.
- One maze per size (seed 1000 + n); a different layout of the same size would move the numbers.
- 40 queries per distance bin; bins a maze cannot reach are absent (4×4 reaches 4–8 cells, 22×22 32–64).
- The planner is ntrl-demo's MPPI scaled per cell; a gradient-descent planner or a wider MPPI was not tried here.
- ntrl-demo's `metric` package was run unchanged except for the data recipe being applied in the unit cube the
  C-space box maps to (see Setup); its TD step (0.03 of the box) spans 0.7 cells at 22×22.

## Reproduce

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py prepare --n 13
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py ntrl --prepared results/registered_<prepare id> --seed 87358
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py fourier --prepared ... --seed 87358 --b-scale 3.0
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py segdp --prepared ... --base results/registered_<ntrl id>/training/checkpoint.pt --seed 87358
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py segdp3 --prepared ... --base ... --seed 87358
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/report.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/wallfeel.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/posthoc.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/figs.py
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/extras.py
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/assemble.py
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/explorer/fragment.py && ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/page.py

Superseded: the first four prepare runs (20261101T192118Z-57dea88a, -645cf7ff, 192127Z-848660ef, 192138Z-d3554001) had
a truth-seeding bug (fixed in `truth.py`; registry notes). Cancelled: 20261101T204642Z-c422c40e (DP on the zoomed 40×40
base, stopped at level 1 because that base was as wall-blind as the default).

Dropped runs (on request, 2026-11-01): the 40×40 maze (replaced by 10×10) and the zoomed-input control (`zoom.py`,
`segdp2.py`). Their registered runs stay on disk and `report.py` skips them (`DROPPED_N`, `DROPPED_STAGES`). Earlier
results: ntrl-demo 268/2,520 on 40×40 (= the straight line), DP + wall term 155/840; zoom 13×13 444/600, 22×22
213/720, 40×40 108/840.

## Run index

<!--RUNS-->
