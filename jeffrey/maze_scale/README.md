# Maze scale: how ntrl-demo copes with bigger mazes, and whether segment DP helps long trajectories

Request (2026-11-01): test ntrl-demo on small and big environments. Five 2-D mazes (the last 10× larger), planning in
SE(2) with rotation, mazes as `.obj`; query ntrl-demo with queries binned by distance; then try segment DP and see
whether it helps long trajectories.

Everything below is from registered runs (run index at the end); success = ntrl-demo's MPPI over planning seeds 0/1/2,
pooled over training seeds 87358/1/2 unless a cell says otherwise. Full tables: `report/tables.md`; near-wall
mechanism: `report/wallfeel.md`; failure geometry: `report/posthoc.md`.

## Summary

![success by distance](report/success.png)

*MPPI success by true travel distance; ntrl-demo pools training seeds 87358 / 1 / 2 (whisker = seed range), so do
segment DP and segment DP + wall term on 13×13; every other point is training seed 87358.*

1. **ntrl-demo is perfect up to the size of its own demo maze and falls apart beyond it.** On 4×4 and 7×7 mazes it
   solves 1,077/1,080 and 1,437/1,440 runs: every distance bin, including the longest (8–16 cells). On 13×13 it solves
   59 % (1,060/1,800; training seeds 53–66 %), on 22×22 23 % (505/2,160), on 40×40 11 % (268/2,520). The oracle (the
   same MPPI on the true travel time) solves 100 % of every bin of every maze, so the losses are the field's.
2. **Distance matters only once the maze is big, and even short queries suffer there.** At 7×7 success is flat in
   distance. At 13×13 it falls from 94 % (0.5–2 cells) to 19 % (16–32 cells; training seeds 3–28 %); at 22×22 from
   75 % to 5 %; at 40×40 from 66 % to 0 % beyond 4 cells. But a 2–4-cell query is solved 100 % in the 7×7 maze, 78 % in 13×13, 32 % in
   22×22 and 8 % in 40×40: the field degrades everywhere as the maze grows, and long queries add up local errors.
   On the 40×40 maze ntrl-demo does no better than driving straight at the goal in any bin (66 vs 62 %, 8 vs 5 %, then 0).
3. **ntrl-demo fails by hitting walls, and the walls fade out of its field.** Every failed ntrl-demo run touches a wall
   while driving (planning seed 0: all of them on 13×13 and 40×40; at most 3 of 40 per bin stall instead). The learned speed 1/|∇T| matches the true
   speed less and less (log-correlation on held-out pairs 0.96, 0.92, 0.69, 0.32, 0.16 from 4 to 40 cells). At
   near-wall poses the field's rise into the wall (S·|∇T|, target 1) falls from 0.84–0.89 to 0.20 and its direction
   (cosine with "into the wall") from 0.98 to 0.07–0.10 at 40 cells. On the 40×40 maze MPPI hits the first wall it
   meets, a median 0.4–0.8 cells after the start.
4. **Control: input resolution explains part of it, capacity the rest.** Training ntrl-demo on zoomed coordinates
   (one cell spans the same input length as in the 7×7 maze; equivalent to scaling its Fourier matrix B) recovers a
   lot at 13×13 (444 vs 347/600 for the same training seed, 74 %; better in every bin) and some at 22×22 (213 vs
   152/720), but not at 40×40 (108 vs 89/840, speed fit unchanged at 0.15): one network does not hold ~1,400 walls
   even at the right frequency.
5. **Segment DP as a value chain makes the field better and the planner worse.** Interval DP over geometric gap
   segments (0.5 cells up to the maze diameter, one ntrl-demo network per segment, each fit to min-over-split-point
   targets built from the shorter segments, ntrl-demo's own field below 0.5 cells) fixes the long-range travel times
   (13×13, 16–32 cells: 1.35 × T* vs 2.04) and descent directions (cosine 0.92 vs 0.88), yet plans worse on every maze
   that is not already solved: 13×13 262 / 280 / 263 vs 347 / 320 / 393 of 600 (training seeds 87358 / 1 / 2), 22×22
   118 vs 152/720, 40×40 85 vs 89/840, 7×7 459 vs 480/480 (4×4 360 = 360; training seed 87358). Mechanism: the DP levels are fit to values only, so they lose the wall barrier
   that ntrl-demo's near-wall terms give its own field (13×13 near-wall slope 0.24–0.29 vs 0.41–0.48, direction
   0.64–0.73 vs 0.89–0.92), and MPPI collides sooner. Neither excluding through-wall base pieces from the DP (22×22:
   132/720) nor a zoomed base (22×22: 169 vs its own 213/720) changes that.
6. **With ntrl-demo's wall term in every DP level, segment DP helps, on short and long trajectories, in the 13- and 22-cell mazes.**
   Fitting ntrl-demo's normal (wall-direction) term at near-wall poses in each level (`segdp3.py`) brings the barrier
   back (13×13 near-wall slope 0.58–0.82, direction 0.97–0.99) and the chains stop hitting walls: their failures become
   stalls, MPPI stopping in a local minimum of the long-range field (13×13, planning seed 0: 3–4 wall contacts in 48–74
   failed runs; `report/paths13.png`, query 161). The term's weight matters. At weight 1 (the cabinet study's setting)
   the long-range values drift low (0.86 × T* at the last level) and long queries stall: 13×13 379 / 400 vs ntrl-demo
   347 / 320 (training seeds 87358 / 1), 16–32 cells 3 / 34 vs 32 / 4 of 120. At weight 0.3 the values stay calibrated
   (1.12 × T*) and the chain is the best field of the study on 13×13: 451 / 456 / 420 of 600 runs against ntrl-demo's
   347 / 320 / 393 on the same training seeds (74 % vs 59 % pooled), better at every distance: 4–8 cells 114 / 107 / 96
   vs 58 / 79 / 77 of 120 and, on the longest trajectories (16–32 cells), 32 / 51 / 29 vs 32 / 4 / 34 (mean 37 vs 23).
   On the bigger mazes (training seed 87358): 22×22 308 vs 152/720 (2–4 cells 84 vs 26, 8–16 cells 36 vs 18, 16–32
   cells 12 vs 6 of 120); 40×40 155 vs 89/840, all of it below 8 cells (2–4 cells 58 vs 11 of 120; nothing beyond 8 cells
   for either). The wall-term levels even give the 40×40 chain the barrier its base lacks (near-wall direction 0.69–0.94
   vs 0.07–0.10). Where ntrl-demo is already perfect the chain costs a little (7×7: 454 vs 480/480, 4–16-cell queries).
   The long-range failures that remain are stalls (22×22: 34 of 36 failures at 16–32 cells; 40×40: 38–40 of 40 in every
   bin from 8 cells).

So, for the question: segment DP as a pure value chain does not help long trajectories in these mazes, because a long
trajectory fails by hitting a wall somewhere along the way, which needs a sharp local barrier that value regression does
not keep. With ntrl-demo's own wall term in every level (weight 0.3) it does help: it removes most wall hits and beats
ntrl-demo at every distance on 13×13 (74 % vs 59 % over three training seeds; longest bin 31 % vs 19 %) and doubles it
on 22×22 (43 % vs 21 %), with the gains largest at 2–16 cells. The longest trajectories are now limited by stalls in
local minima of the long-range field, and on the 40×40 maze nothing tested here gets past 8 cells.

## Setup

![mazes](report/mazes.png)

**Mazes** (`maze.py`). Five square mazes of n = 4, 7, 13, 22 and 40 cells a side (about ×1.78 per step; the last is
10× the first in each direction, 100× in area). Cell pitch 1, walls 0.1 thick on the grid lines. Layout: a perfect
maze (randomized Kruskal), then 15 % of the remaining interior walls removed to open loops, as in ntrl-demo's own demo
maze (`datasets/test/maze`, 7 × 7 cells, open). Every maze is written as an `.obj` (walls extruded to height 0.5) and
normalized exactly as ntrl-demo's `to_off` does (centre on the bounding box, divide by its largest extent n + 0.1), so
the maze fills [−0.5, 0.5]² and a cell is 1/(n + 0.1) of the box: a bigger maze is the same world with more cells,
not a bigger cell.

**Robot and C-space.** A 0.5 × 0.2-cell rectangle with heading θ ∈ [−π/2, π/2] (the rectangle is π-symmetric, so
this holds every orientation; no wrap-around). Configuration q = (x, y, φ) with x, y normalized and φ = 0.25 θ / (n +
0.1): rotating by dθ costs like moving 0.25 dθ cells at every maze size. On a corridor's centre line the robot can turn
in place (half-diagonal 0.27 < 0.45); driving along a corridor broadside leaves 0.2 cells at its ends (slow), aligned
0.35 (fast). Clearance = exact signed distance between the rectangle and the wall boxes (checked against shapely's
polygon distance to 4·10⁻⁶ cells on 20,000 random poses).

**Speed** = ntrl-demo's point recipe clip(d / margin, offset / margin, 1) with margin 0.175 cells (ntrl-demo's
`test`-maze ratio: margin limit/20 = 0.025 at pitch 1/7) and offset margin / 10, stretched by the `metric` trainer to
1.025 s²(2 − s)² − 0.025 (slowest 0.012).

**ntrl-demo** (`data.py`, `train.py`). Its point recipe in the robot's C-space, 500,000 pairs per maze (data seed 0):
x0 within the margin of a wall, x1 = x0 + unit(rand − 0.5) · U(0, √3) not inside a wall, normals = unit C-space
gradient of the clearance. The recipe assumes the unit cube, so it is applied in the cube the C-space box maps to
(φ scaled to [−0.5, 0.5]); applied directly to the thin φ slab every long step would leave the box (median pair gap
0.3 cells at every size). Training = ntrl-demo's `models/metric` package through its own `Model.train()` with
dim = 3, unchanged (5,000 epochs, AdamW 5·10⁻⁴, weight decay 0.5, epoch rollback); training seeds 87358 / 1 / 2.

**Planner** (`plan.py`). ntrl-demo's MPPI (`tests/arm_plan_stat.py`: 50 samples, horizon 5, cost 10 T(first) +
T(last), softmax(−50 cost), stop within tolerance), batched over queries. Its step 0.015 and tolerance 0.01 hold for the
4-cell maze; on a maze with pitch p they are multiplied by p / p₄ and the temperature divided by it (cost differences
between rollouts scale with the step), so the planner is the same in cell units at every size. Step budget per query:
3 × the true path's C-space length / step (at least ntrl-demo's 200). Planning seeds 0 / 1 / 2. A path succeeds if it
reaches the goal and the robot never touches a wall (exact clearance every 0.005 cells).

**Ground truth and queries** (`truth.py`, `prepare.py`). T*(·, goal) on a C-space grid (1/15 cell, 13 headings,
98-neighbourhood shortest paths with the trained speed; within 1 % median / 3 % p99 of a 4× finer grid on the 4- and
7-cell mazes) for 48 goal poses per maze. Queries: 200,000 random start poses (clearance ≥ 0.1 cells) paired with the
goals and binned by the true travel distance D = T* / pitch (cells at full speed): 0.5–2, 2–4, 4–8, … up to the
largest distance the maze has; 40 queries per bin, spread over goals, each certified by its truth path (exact
collision check). Controls: the straight segment, and the *oracle* = the same MPPI driven by T* — 100 % in every bin of
every maze, so every failure below is the learned field's.

## Results

### 1. ntrl-demo by maze size and distance

MPPI success (%), pooled over 3 training seeds × 3 planning seeds (per-seed values in `report/tables.md`):

| maze | field | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 | all |
|---|---|---|---|---|---|---|---|---|---|
| 4×4 | oracle | 100 | 100 | 100 |  |  |  |  | 100 |
| | straight line | 62 | 5 | 0 |  |  |  |  | 22 |
| | **ntrl-demo** | **100** | **99** | **100** |  |  |  |  | **100** |
| 7×7 | oracle | 100 | 100 | 100 | 100 |  |  |  | 100 |
| | straight line | 78 | 18 | 0 | 0 |  |  |  | 24 |
| | **ntrl-demo** | **100** | **100** | **100** | **99** |  |  |  | **100** |
| 13×13 | oracle | 100 | 100 | 100 | 100 | 100 |  |  | 100 |
| | straight line | 70 | 10 | 0 | 0 | 0 |  |  | 16 |
| | **ntrl-demo** | **94** | **78** | **59** | **44** | **19** |  |  | **59** |
| 22×22 | oracle | 100 | 100 | 100 | 100 | 100 | 100 |  | 100 |
| | straight line | 65 | 10 | 0 | 0 | 0 | 0 |  | 12 |
| | **ntrl-demo** | **75** | **32** | **16** | **12** | **5** | **0** |  | **23** |
| 40×40 | oracle | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| | straight line | 62 | 5 | 0 | 0 | 0 | 0 | 0 | 10 |
| | **ntrl-demo** | **66** | **8** | **0** | **0** | **0** | **0** | **0** | **11** |

Field diagnostics (median per bin, mean over training seeds; `report/tables.md`): T(start, goal)/T* is 0.91–0.93
at 4×4 and 0.96–1.05 at 7×7, then 1.12–2.08 at 13×13 (the long bins overestimate by 2×), 1.01–1.43 at 22×22 and
1.11–1.28 at 40×40. The cosine between the field's descent direction at the start and the truth's is 0.91–0.97 at 4–7
cells and 0.82–0.95 at 13×13; at 22×22 and 40×40 it falls with distance to 0.06 in the longest bins. Spearman
correlation of T with T* over all queries stays 0.97–0.99 at every size: the field orders goals by distance even
when it cannot plan to them.

### 2. Mechanism: the walls fade out

![speed maps](report/speedmaps.png)

The learned speed 1/|∇_x T(x, goal)| at heading 0 (middle row) redraws every wall of the 7×7 maze, most of the 13×13,
a connected subset of the 22×22 walls and almost none of the 40×40. Held-out near-wall poses (S < 0.3; median
S·|∇T| / cosine of ∇T with the into-wall direction, `report/wallfeel.md`):

| maze | field | 0-0.5 | 0.5-2 | 2-8 | >8 |
|---|---|---|---|---|---|
| 4 | ntrl-demo | 0.83 / 0.98 | 0.88 / 0.99 | 0.89 / 0.99 | - |
| 4 | segment DP | 0.83 / 0.98 | 0.81 / 0.99 | 0.81 / 0.99 | - |
| 7 | ntrl-demo | 0.76 / 0.97 | 0.84 / 0.98 | 0.86 / 0.98 | - |
| 7 | segment DP | 0.76 / 0.97 | 0.65 / 0.97 | 0.62 / 0.96 | - |
| 7 | segment DP + wall term | 0.77 / 0.97 | 0.84 / 0.99 | 0.92 / 0.99 | - |
| 7 | segment DP + wall term (weight 0.3) | 0.77 / 0.97 | 0.84 / 0.99 | 0.90 / 0.99 | - |
| 13 | ntrl-demo | 0.38 / 0.86 | 0.41 / 0.89 | 0.48 / 0.92 | 0.48 / 0.91 |
| 13 | segment DP | 0.38 / 0.86 | 0.29 / 0.73 | 0.27 / 0.64 | 0.24 / 0.65 |
| 13 | segment DP + wall term | 0.39 / 0.86 | 0.58 / 0.97 | 0.77 / 0.99 | 0.82 / 0.99 |
| 13 | segment DP + wall term (weight 0.3) | 0.39 / 0.85 | 0.57 / 0.97 | 0.76 / 0.98 | 0.79 / 0.99 |
| 13 | zoom x1.85 | 0.45 / 0.90 | 0.50 / 0.93 | 0.55 / 0.94 | 0.50 / 0.93 |
| 22 | ntrl-demo | 0.21 / 0.57 | 0.23 / 0.35 | 0.24 / 0.56 | 0.24 / 0.64 |
| 22 | segment DP | 0.21 / 0.57 | 0.23 / 0.19 | 0.23 / 0.20 | 0.22 / 0.25 |
| 22 | segment DP + wall term | 0.22 / 0.58 | 0.47 / 0.96 | 0.60 / 0.97 | 0.60 / 0.98 |
| 22 | segment DP + wall term (weight 0.3) | 0.21 / 0.57 | 0.35 / 0.89 | 0.57 / 0.96 | 0.61 / 0.97 |
| 22 | segment DP, checked pieces | 0.21 / 0.57 | 0.24 / 0.17 | 0.23 / 0.17 | 0.22 / 0.24 |
| 22 | zoom x3.11 | 0.28 / 0.67 | 0.31 / 0.61 | 0.35 / 0.70 | 0.35 / 0.72 |
| 40 | ntrl-demo | 0.18 / 0.48 | 0.20 / 0.07 | 0.21 / 0.09 | 0.20 / 0.10 |
| 40 | segment DP | 0.18 / 0.48 | 0.22 / 0.05 | 0.24 / 0.06 | 0.23 / 0.04 |
| 40 | segment DP + wall term (weight 0.3) | 0.18 / 0.48 | 0.23 / 0.69 | 0.37 / 0.91 | 0.42 / 0.94 |
| 40 | zoom x5.65 | 0.24 / 0.47 | 0.27 / 0.05 | 0.28 / 0.15 | 0.23 / 0.17 |

Failure geometry (`report/posthoc.md`, planning seed 0, training seed 87358; the goal the scorer appends to a run
that never converged is excluded): every failed ntrl-demo run touches a wall while driving; the first contact comes
after a median 0.4–0.9 cells at 22×22 and 40×40 for short queries, within one cell of the start in 22–33 of 40 runs
per bin at 40×40, and after 2.4–4.3 cells (22–41 % of the route) on 13×13.

Zoom control (`zoom.py`): training pairs' coordinates multiplied by s = (n + 0.1)/7.1 (1.845, 3.113, 5.648), the
trained network used on the original coordinates as the NN with B' = s B and output / s (exact: 0.0 difference in
value and gradient). Success 13×13 444/600 (74 %; ntrl-demo 320–393), 22×22 213/720 (ntrl-demo 152–186), 40×40
108/840 (86–93). The bottom row of the speed maps shows the zoomed fields: walls over the whole 22×22 maze (speed fit
0.42 vs 0.31), fine structure everywhere on 40×40 but only a fraction of the walls in place (0.15).

### 3. Segment DP

`segdp.py`: the interval-DP transition of `jeffrey/segment_dp` on maze C-space gaps,
T_(m+1)(i, j) = min_k T_≤m(i, k) + T_≤m(k, j) with |i − k|, |k − j| ≤ e_m (1 + blend). Edges are geometric in cells
(0.5, ×1.35, … to the 99.9th-percentile pool gap; 8 levels at 4×4, 16 at 40×40, blend ±5 %); below 0.5 cells the chain
is ntrl-demo's trained field; each level is an ntrl-demo `metric` network warm-started from the previous level and fit
for 6,000 steps (batch 2,000, AdamW 5·10⁻⁴) with |log T − log y| to the DP targets of 100,000 pool pairs (200,000
collision-free pool poses; 128 split candidates per pair, half nearest the straight line, half uniform in the lens).
The chain is planned with the same MPPI and diagnosed the same way. Variants: `--check-pieces` (a base piece counts
only if its straight segment is collision-free), `segdp2.py` (zoomed base), `segdp3.py` (ntrl-demo's normal term at
near-wall poses paired with goals in each level's segment, weight 1 as `jeffrey/segment_dp` dp3 `--normal-weight`,
and 0.3).

MPPI success (%) by true travel distance, per training seed (3 planning seeds each):

| maze | field (training seed) | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 | all |
|---|---|---|---|---|---|---|---|---|---|
| 4×4 | ntrl-demo (87358) | 100 | 100 | 100 |  |  |  |  | 100 (360/360) |
|  | segment DP (87358) | 100 | 100 | 100 |  |  |  |  | 100 (360/360) |
| 7×7 | ntrl-demo (87358) | 100 | 100 | 100 | 100 |  |  |  | 100 (480/480) |
|  | segment DP (87358) | 100 | 100 | 95 | 88 |  |  |  | 96 (459/480) |
|  | segment DP + wall term (87358) | 100 | 100 | 91 | 81 |  |  |  | 93 (446/480) |
|  | segment DP + wall term (weight 0.3) (87358) | 100 | 100 | 90 | 88 |  |  |  | 95 (454/480) |
| 13×13 | ntrl-demo (87358) | 92 | 75 | 48 | 48 | 27 |  |  | 58 (347/600) |
|  | ntrl-demo (1) | 92 | 78 | 66 | 28 | 3 |  |  | 53 (320/600) |
|  | ntrl-demo (2) | 98 | 81 | 64 | 57 | 28 |  |  | 66 (393/600) |
|  | ntrl-demo, zoomed input (87358) | 97 | 88 | 78 | 62 | 45 |  |  | 74 (444/600) |
|  | segment DP (87358) | 96 | 61 | 38 | 23 | 1 |  |  | 44 (262/600) |
|  | segment DP (1) | 95 | 72 | 46 | 15 | 6 |  |  | 47 (280/600) |
|  | segment DP (2) | 96 | 69 | 30 | 22 | 2 |  |  | 44 (263/600) |
|  | segment DP + wall term (87358) | 100 | 88 | 82 | 42 | 2 |  |  | 63 (379/600) |
|  | segment DP + wall term (1) | 100 | 90 | 78 | 38 | 28 |  |  | 67 (400/600) |
|  | segment DP + wall term (2) | 98 | 91 | 78 | 42 | 5 |  |  | 63 (377/600) |
|  | segment DP + wall term (weight 0.3) (87358) | 100 | 99 | 95 | 55 | 27 |  |  | 75 (451/600) |
|  | segment DP + wall term (weight 0.3) (1) | 100 | 98 | 89 | 51 | 42 |  |  | 76 (456/600) |
|  | segment DP + wall term (weight 0.3) (2) | 100 | 95 | 80 | 51 | 24 |  |  | 70 (420/600) |
| 22×22 | ntrl-demo (87358) | 75 | 22 | 10 | 15 | 5 | 0 |  | 21 (152/720) |
|  | ntrl-demo, zoomed input (87358) | 78 | 58 | 30 | 12 | 1 | 0 |  | 30 (213/720) |
|  | segment DP (87358) | 74 | 17 | 4 | 1 | 2 | 0 |  | 16 (118/720) |
|  | segment DP, checked pieces (87358) | 76 | 21 | 5 | 6 | 2 | 0 |  | 18 (132/720) |
|  | segment DP, zoomed base (87358) | 77 | 47 | 13 | 4 | 0 | 0 |  | 23 (169/720) |
|  | segment DP + wall term (87358) | 86 | 58 | 41 | 8 | 0 | 0 |  | 32 (232/720) |
|  | segment DP + wall term (weight 0.3) (87358) | 90 | 70 | 57 | 30 | 10 | 0 |  | 43 (308/720) |
| 40×40 | ntrl-demo (87358) | 65 | 9 | 0 | 0 | 0 | 0 | 0 | 11 (89/840) |
|  | ntrl-demo, zoomed input (87358) | 68 | 20 | 2 | 0 | 0 | 0 | 0 | 13 (108/840) |
|  | segment DP (87358) | 62 | 9 | 0 | 0 | 0 | 0 | 0 | 10 (85/840) |
|  | segment DP + wall term (weight 0.3) (87358) | 73 | 48 | 8 | 0 | 0 | 0 | 0 | 18 (155/840) |

Per-level truth probes (pool pose → truth goal pairs in each segment): the DP targets track the trained levels to
0.2–3 % (held-out |log|) and sit at 1.07–1.36 × T* on 13×13 and 22×22 (ntrl-demo's own short-range bias, 1.03–1.2,
plus split-point detours), against 1.03–1.96 × for ntrl-demo itself at the same gaps on 13×13.

## Caveats

- On 13×13 ntrl-demo, segment DP and both wall-term weights have three training seeds each; every other DP and zoom
  number is training seed 87358 only. ntrl-demo's 13×13 seeds differ by up to 73 runs of 600 (320–393), mostly in the
  long bins, so single-seed differences smaller than that are not separations (the 22×22 and 40×40 wall-term gains,
  +156 and +66, are larger than the seed spread there: 152–186 and 86–93).
- The wall-term weight was tried at 1 and 0.3 only, on 13×13 first (one seed) and then carried to the other mazes.
- One maze per size (seed 1000 + n); a different layout of the same size would move the numbers.
- 40 queries per distance bin; bins a maze cannot reach are absent (4×4 reaches 4–8 cells, 40×40 64–128).
- The planner is ntrl-demo's MPPI scaled per cell; a gradient-descent planner or a wider MPPI was not tried here.
- Zoom changes three things at once (Fourier frequency per cell, TD step 0.03/s, time units); the planner sees the
  same time units because the output is divided by s.
- ntrl-demo's `metric` package was run unchanged except for the data recipe being applied in the unit cube the
  C-space box maps to (see Setup); its TD step (0.03 of the box) spans 1.2 cells at 40×40.

## Reproduce

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py prepare --n 13
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py ntrl --prepared results/registered_<prepare id> --seed 87358
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py zoom --prepared ... --seed 87358 --zoom 1.8450704225352113
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py segdp --prepared ... --base results/registered_<ntrl id>/training/checkpoint.pt --seed 87358
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py segdp3 --prepared ... --base ... --seed 87358
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/report.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/wallfeel.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/posthoc.py
    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/figs.py
    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/page.py

Superseded: the first four prepare runs (20261101T192118Z-57dea88a, -645cf7ff, 192127Z-848660ef, 192138Z-d3554001) had
a truth-seeding bug (fixed in `truth.py`; registry notes). Cancelled: 20261101T204642Z-c422c40e (DP on the zoomed 40×40
base, stopped at level 1 because that base was as wall-blind as the default).

## Run index

| maze | stage | training seed | run id (results/registered_&lt;id&gt;) |
|---|---|---|---|
| 4×4 | prepare | - | 20261101T192432Z-423319dc |
| | ntrl-demo | 1 | 20261101T194708Z-15deb708 |
| | ntrl-demo | 2 | 20261101T195648Z-927151d4 |
| | ntrl-demo | 87358 | 20261101T193251Z-f43026ee |
| | segment DP | 87358 | 20261101T203439Z-6ed89e28 |
| 7×7 | prepare | - | 20261101T192446Z-9b1569db |
| | ntrl-demo | 1 | 20261101T201244Z-1ae74c20 |
| | ntrl-demo | 2 | 20261101T202527Z-32f7835e |
| | ntrl-demo | 87358 | 20261101T200440Z-b920137d |
| | segment DP | 87358 | 20261101T201621Z-4d135bc3 |
| | segment DP + wall term | 87358 | 20261101T212502Z-d09fb16b |
| | segment DP + wall term (weight 0.3) | 87358 | 20261101T230634Z-024a6522 |
| 13×13 | prepare | - | 20261101T192504Z-13cf1d04 |
| | ntrl-demo | 1 | 20261101T204630Z-58badec5 |
| | ntrl-demo | 2 | 20261101T205900Z-c57aa0f1 |
| | ntrl-demo | 87358 | 20261101T203437Z-ebc95225 |
| | zoom x1.845 | 87358 | 20261101T204839Z-df3ab517 |
| | segment DP | 1 | 20261101T221325Z-26a627cd |
| | segment DP | 2 | 20261101T212901Z-f4ac5cd0 |
| | segment DP | 87358 | 20261101T204654Z-6719910c |
| | segment DP + wall term | 1 | 20261101T214011Z-1684e9fd |
| | segment DP + wall term | 2 | 20261101T215413Z-4f22c760 |
| | segment DP + wall term | 87358 | 20261101T210413Z-ae790e10 |
| | segment DP + wall term (weight 0.3) | 1 | 20261101T222804Z-9ddcbd27 |
| | segment DP + wall term (weight 0.3) | 2 | 20261101T224644Z-59272aa7 |
| | segment DP + wall term (weight 0.3) | 87358 | 20261101T215932Z-1bbbfbc7 |
| 22×22 | prepare | - | 20261101T193403Z-cea05796 |
| | ntrl-demo | 1 | 20261101T194908Z-fa2678e3 |
| | ntrl-demo | 2 | 20261101T200614Z-92e8b351 |
| | ntrl-demo | 87358 | 20261101T193703Z-dd44f02b |
| | zoom x3.113 | 87358 | 20261101T202555Z-61b47892 |
| | segment DP | 87358 | 20261101T202506Z-fbbff9fe |
| | segment DP, checked pieces | 87358 | 20261101T204940Z-0ab9e137 |
| | segment DP, zoomed base | 87358 | 20261101T204927Z-51fa1b1c |
| | segment DP + wall term | 87358 | 20261101T212214Z-6b29cad2 |
| | segment DP + wall term (weight 0.3) | 87358 | 20261101T223027Z-dd0f1f6b |
| 40×40 | prepare | - | 20261101T192432Z-a7d34464 |
| | ntrl-demo | 1 | 20261101T194659Z-7d97259d |
| | ntrl-demo | 2 | 20261101T200157Z-9b30f307 |
| | ntrl-demo | 87358 | 20261101T193402Z-ffc2decc |
| | zoom x5.648 | 87358 | 20261101T202151Z-8e48de41 |
| | segment DP | 87358 | 20261101T195644Z-ac24eb74 |
| | segment DP + wall term (weight 0.3) | 87358 | 20261101T223506Z-f2a8699a |
