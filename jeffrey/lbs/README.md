# LBS: lift + Bellman + segment, benchmarked with one part removed at a time

Study of 2026-11-19/20. Request (21:36 EST, 2026-11-19):

> Stop everything. Here is what I want. My new method is called lbs, it stands for lift, bellman segment. lift + bellman you already
> know., segment stands for the midpoint loss https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U over the next 12 hours work on this new
> idea. For now, use the ground truth for bellman update. For this method, you will always train 4 methods, one with everything, and the
> other 3 have one of either lift, bellman, or segment ablated. I want the following tests: 10x10, 13x13, 18x18, mazes for these you
> will report the success rate as time goes on. You will also try 10x10 gap maze, from (x1,x2,x4 gaps). Finally, you will try 2d gibson,
> pick 3 envs. Don't worry about multiple seeds for now [...]. Create a new report, call it the lbs bellman benchmarking. Oh and also,
> include the original ntrl-demo baseline. You can see if some of these models already exist and just run the evalaute on them to
> reduce training time. Your deadline is to have it ready in 12 hours.

## What is combined

* **L, lift** (jeffrey/lift_scale): one more input column, the travel-time height h(q) = G(d(q)) / E. No push, no octaves.
* **B, Bellman** (jeffrey/pair_lift, recipe `bxt1` / `bxt1nL`): for each end q of a pair (q, g), y = min over 16 straight collision-free
  steps of [tau(q, q') + T(q', g)], loss (T - y)^2, tau from the TRUE speed along the step (its lifted length), the true geometry for the
  collision check ("use the ground truth for bellman update"). Radius 0.663 cells, weight x (0.03 / radius in box lengths)^2, 6 of the 16
  candidates around the descent direction; step lengths up to 8 radii on scenes above 24 cells (`--bell-long 8`: gap x4, the houses),
  0.5-1 radius otherwise. ntrl-demo's TD term skips steps that cross an obstacle.
* **S, segment** (jeffrey/midpoint_dp MAIN setting, as in jeffrey/lift_midpoint): the travel-time midpoint loss from epoch 2,000.

## Arms (one training seed, 87358)

| arm | pipeline arguments |
|---|---|
| ntrl-demo | `--lift none --weight 0` |
| LBS | `--lift time --bell 1` |
| no lift | `--lift none --bell 1` |
| no Bellman | `--lift time` |
| no segment | `--lift time --bell 1 --weight 0` |

## Scenes

maze_scale mazes 10 x 10 (prepare b0baa8e9), 13 x 13 (13cf1d04), 18 x 18 (5c83b203, prepared for this study); the 10 x 10-base gap mazes
of jeffrey/cross_warp x1 / x2 / x4 (5877176f, 01c9f536, 25e73d68: 16 / 22 / 34 cells); Gibson floor plans Allensville, Lakeville,
Klickitat (jeffrey/lift_scale prepares 0144646b, e60b96b5, 61a4f754). The SE(2) rectangle robot of every local study.
On the houses ntrl-demo, no Bellman and no segment are the registered runs of jeffrey/lift_scale (`--lift none`), jeffrey/lift_midpoint
(`liftmid`) and jeffrey/pair_lift (`bxt1nL`): same loop, data, queries, planner, scorer (report.py REUSE).

## Files

| file | what |
|---|---|
| `trainer.py` | jeffrey/lift_midpoint/trainer.py + the Bellman term and the free-step TD option |
| `bell.py` | the Bellman backup, copied from jeffrey/pair_lift (ClearTable, Slowness, step_time, bellman_term) |
| `pipeline.py` | train (planning seed 0 on the test queries every `--validate-every` epochs), plan with 3 planning seeds, diagnostics (`--lean`: only the quick ones) |
| `run.py` | registry wrapper; sources frozen in `snap/<hash>/`; the reused modules are checked to be verbatim copies |
| `lift.py`, `scenes.py`, `diag.py`, `fastnet.py`, `traverse.py`, `tree.py`, `liftfast.py` | verbatim copies (lift_scale, midpoint_dp, lift_midpoint) |
| `queue/` | runner (one run per GPU), jobs.txt, scenes.json, state.json, runner.log |
| `report.py`, `page.py`, `pagecheck.py` | `report/runs.json`, `report/tables.md`, the report page `report/lbs.html` |

Results: `report/tables.md` (regenerated from the registry) and the report page.

## Findings (2026-11-20, 08:45; one training seed unless said; all numbers in `report/tables.md`, run ids there)

Queries solved, % (ntrl-demo's MPPI, planning seeds 0 / 1 / 2, exact scoring, network of epoch 5,000):

| scene | LBS | no lift | no Bellman | no segment | ntrl-demo |
|---|---|---|---|---|---|
| maze 10 x 10 | 100.0 | 92.0 | 94.3 | 94.5 | 83.3 |
| maze 13 x 13 | 99.5 | 69.2 | 88.7 | 98.3 | 64.2 |
| maze 18 x 18 | 94.6 | 49.4 | 46.9 | 86.5 | 35.4 |
| gap x1 | 100.0 | 81.8 | 71.0 | 100.0 | 70.2 |
| gap x2 | 99.9 | 78.9 | 41.6 | 99.8 | 45.8 |
| gap x4 | 99.3 | 63.9 | 27.0 | 94.1 | 33.0 |
| Allensville | 92.5, seed 1: 97.1 | 98.8 | 84.6 (reused) | 94.2, seed 1: 99.6 | 83.9 (reused) |
| Lakeville | 90.8 | 44.9 | 60.7 (reused) | 97.0 (own; reused run 97.9) | 69.4 (reused) |
| Klickitat | 95.7 | 58.5 | 65.0 (reused) | 90.1 (own; reused run 94.0) | 72.5 (reused) |

- LBS is ahead of ntrl-demo on every scene, by 11 to 66 points.
- Without the lift the failures are plans through a wall (7.5 to 44.6 % of the runs); without Bellman they are stalls (5.7 to 58.4 %).
  Within the regular mazes and within the gap mazes both parts are worth more as the scene grows.
- Segment is the smallest part: +5.5 / +1.2 / +8.1 on the mazes, 0 on gap x1 and x2 (the arm without it already solves everything),
  +5.2 on gap x4, and on Allensville LBS ends 4 and 6 queries of 240 lower in two matched seed pairs.
- Travel-time accuracy does not follow planning success: on the mazes the arm without Bellman has the most accurate times
  (90 to 96 % within 10 % on 10 x 10 and 13 x 13) and LBS 31 to 56 %.

### Allensville: is the segment loss harmful? (asked 2026-11-20 01:13)

The first LBS run solved 92.5 % against 100.0 % for the no-segment run reused from jeffrey/pair_lift (`dfeaeac0`). Re-run of the
no-segment arm with this study's code, the same seed and GPU (`7b0ee268`): 94.2 %; identical to the LBS run (`a97e6309`) through
epoch 2,000. Second seed: LBS 97.1 (`ef11536e`), no segment 99.6 (`6b3e7bec`). `allens.py` -> `report/allensville.json`:

- All 14 failures without segment and 15 of the 18 with it start in one room (x < -2, y > 11 cells; 23 of the 240 queries), on every
  planning seed, and stop at one spot just outside it: a local minimum of the field on the way out of that room.
- That room opens and closes between saved epochs in every run: both runs solve 236 of 240 at epoch 4,000 and 222 / 226 at 5,000; the
  reused run failed on the same room at epochs 1,000 to 3,000 and had it open at 5,000. The last-epoch score on this house moves by
  about 6 points with it, whichever arm is trained. A reused run of another code path is therefore not a matched control.
- The gap to 100 % is not the segment loss. LBS still ends 4 and 6 queries lower in the two matched pairs: small against swings of up
  to 17 queries between saved epochs, same direction twice, cause not tested. A third pair is queued.
- Follow-up queued: the no-segment arm on Lakeville and Klickitat with this study's code, for matched pairs on all three houses.

### Lakeville: the matched pair (09:25)

No segment with this study's code, seed and GPU: 97.0 % (reused pair_lift run 97.9 %); LBS 90.8 %. Identical through epoch 2,000 (43 and
20 failed of 280, planning seed 0); after the segment loss starts: no segment 9 / 10 / 8 failures at epochs 3,000 / 4,000 / 5,000, LBS
22 / 14 / 27. So on this house the segment loss costs about 6 points in one seed pair, separated at every saved epoch after it starts;
with Allensville's two pairs the houses point the same way three times. Travel times are more accurate with segment (75.3 against
66.9 % within 10 %), the loss is more stalls (9.2 against 3.0 %). Cause not tested. gap x4 without Bellman: 27.0 %.

### Klickitat: the matched pair (09:40), and the houses together

No segment with this study's code, seed and GPU: 90.1 % (reused pair_lift run 94.0 %); LBS 95.7 %. Failed of 280 at epochs 3,000 /
4,000 / 5,000 (planning seed 0): LBS 15 / 23 / 12, no segment 41 / 19 / 27. So the matched pairs on the houses are Allensville -1.7 and
-2.5, Lakeville -6.2, Klickitat +5.6 points for LBS: no consistent sign. They do not show that the segment loss is harmful on the
houses and do not show that it helps; with one seed pair per house its effect is within about +-6 points, the size of the movement
between saved epochs of one run. (This replaces the reading "the houses point the same way three times" of the Lakeville note above.)

### Pictures (10:00)

`envpics.py` -> `report/env/<scene>.png`, `report/env.json`: each scene from above with the true path of its longest test query;
`Allensville_room.png`: starts of the failed queries of the same-seed pair and the spots their plans stop at.

### No lift on the houses (12:30): needed on the two larger ones

All 45 scene x arm results are in. No lift (Bellman + segment on ntrl-demo's input): Allensville 98.8 % (`5cdf0c70`, above LBS), Lakeville
**44.9 %** (`a3c68687`), Klickitat **58.5 %** (`57bfc23b`), both below ntrl-demo itself (69.4 / 72.5 %, reused runs); gap x4 63.9 % (`b2939fac`).

- The failures on the two larger houses are stalls (54.6 / 41.2 % of runs), not walls (0.5 / 0.4 %).
- The travel times stop growing with distance: median learned / true by true distance (0-4, 4-8, 8-16, 16-32, 32-64, 64-128 cells) is
  1.04 / 0.91 / 0.54 / 0.27 / 0.15 / 0.10 on Lakeville (largest learned time 0.18 against a true 1.11) and 1.03 / 0.96 / 0.91 / 0.51 / 0.26 /
  0.18 on Klickitat. No query longer than 16 cells is solved on Lakeville (0 of 360).
- ntrl-demo alone and LBS keep the far field (farthest bin: 0.95 and 0.89 on Lakeville), so it is the Bellman term without lift on these
  houses, not the plain input and not the Bellman term as such.
- It is there before the segment loss starts (queries > 16 cells solved at epochs 1,000..5,000: Lakeville 0 / 3 / 2 / 0 / 0 of 120), so
  segment is not the cause. On Allensville the same arm also starts with nothing beyond 16 cells and builds the far field outward
  (0 / 1 / 21 / 60 / 80 of 80).
- What differs: scene size in cells and with it the Bellman weight (0.03 / step radius)^2 = 3.8 / 14.1 / 18.9 on Allensville / Lakeville /
  Klickitat. Controls queued 12:13 on GPU 0 (`queue/jobs.txt`): `ntrl|Lakeville` trained by this code with its course, and `noLw|Lakeville`
  = no lift with `--bell 0.27` (the Allensville weight). Jobs whose label is not one of the five arms go to `controls` in `report/runs.json`.

### Segment variants inside LBS (asked 2026-11-20 12:20)

> I want you to do this test right now. Reference this again https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U ... Try plain, min leaves,
> tree and safe, min leaves, tree + sub-paths. These are 2 tests, just plug this in for the segment part in lbs. No need for ablation.

- **safe, min leaves, tree + sub-paths** = the midpoint artifact's main setting = the pipeline defaults (`--traverse safe --leaf min --nodes both`)
  = what every LBS run of this benchmark already uses (checked against midpoint_dp run `c2f195aa`). No new runs; the LBS column is that test.
- **plain, min leaves, tree** = `--traverse plain --nodes tree` (midpoint_dp run `3c09dbc5`). Queue label `lbsP`, nine scenes, seed 87358, each
  pinned to the GPU its LBS run had (5070: maze13, maze18, gap1, gap2, gap4; 3090: maze10 and the houses) so the pair is identical through
  epoch 2,000. Put at the front of the queue; results land in `controls` of `report/runs.json` and the page section "Segment variants inside LBS".
- To start it at once, the just-started control `noLw|Lakeville` (`--bell 0.27`, run `64c9c7fe`, one minute in) was stopped and re-queued
  last as `--bell 0.2684`; own-code ntrl-demo on Lakeville (`f77982a7`): 72.7 %, far field present at epoch 1,000 (37 of 120 queries > 16 cells).

Results so far (14:25, 3 of 9; solved % at epoch 5,000 over 3 planning seeds, then the mean of the saved epochs >= 3,000 on planning seed 0):

| scene | plain, min leaves, tree (`lbsP`) | safe, min leaves, tree + sub-paths (LBS) | no segment |
|---|---|---|---|
| maze 10x10 | 88.0 / 97.6 (`6a238a9c`) | 100.0 / 100.0 (`e62e1e8c`) | 94.5 / 96.9 |
| maze 13x13 | 94.5 / 97.1 (`7dbca24f`) | 99.5 / 98.7 (`84d35844`) | 98.3 / 98.5 |
| maze 18x18 | 91.9 / 88.2 (`b587cb68`) | 94.6 / 90.1 (`a02310b0`) | 86.5 / 86.0 |

Plain is below safe on all three, by 3 to 12 points at the last epoch and by about 2 points in the saved-epoch mean. The last-epoch
gap on 10x10 is one bad epoch: 100 % at every saved epoch from 2,500 to 4,500, then 88 % at 5,000 (the logged ntrl loss of that epoch is
5.6e-4 against a median of 4.5e-4; the far field is short, t_max 2.11 against 2.53 true, 60 of 120 queries of 16-32 cells solved).
The training drives are not the cause there (8,188 of 8,192 reached at the last refresh).

### Segment variants inside LBS: all nine in (18:00)

Solved % at epoch 5,000 over 3 planning seeds / mean of the saved epochs >= 3,000 on planning seed 0:

| scene | plain, min leaves, tree (`lbsP`) | safe, min leaves, tree + sub-paths (LBS) | no segment |
|---|---|---|---|
| gap x1 | 100.0 / 99.9 (`32364492`) | 100.0 / 99.9 | 100.0 / 99.8 |
| gap x2 | 100.0 / 99.7 (`14e899e7`) | 99.9 / 99.3 | 99.8 / 99.7 |
| gap x4 | 93.4 / 96.9 (`1843335d`) | 99.3 / 98.0 | 94.1 / 97.1 |
| Allensville | 88.6 / 91.5 (`cd4d99a6`) | 92.5 / 94.6 | 94.2 / 95.6 |
| Lakeville | 95.5 / 96.0 (`49ce09a5`) | 90.8 / 92.5 | 97.0 / 96.8 |
| Klickitat | 89.3 / 94.4 (`8d07cff5`) | 95.7 / 94.0 | 90.1 / 89.6 |

Plain is below safe on 6 of 9 scenes at the last epoch (mazes -12.0 / -5.0 / -2.6, gap x4 -5.9, Allensville -3.9, Klickitat -6.4), level on
gap x1 / x2 and above on Lakeville (+4.7). In the saved-epoch mean the gaps are about 1 to 3 points and Lakeville (+3.5) and Klickitat
(+0.4) go the other way. One seed per variant; the safe variant (the pipeline default) stays the segment part of LBS.

### 3-D Gibson, point robot in R^3 (asked 2026-11-20 18:00)

Request: "I want to try first 3d gibson R3 planning. Select 5 gibson environments and run it. The lift should be onto 4 dimensions in
this scenario. You can actually try scaling the lift dimension based on the number of input dimensions too. Remove klickitat. I want
these new tests. lbs vs original method only for now." Plus a time estimate and a summary table grouped by type (maze2d, mazegap2d,
gibson2d, gibson3d; baseline and LBS only; empty where not done) in the artifact.

- **Scene** (`scenes3.py`, `House3`): the house mesh at 2.5 cm voxels (10 per 0.25 m cell) in a slab 0.15 to 2.15 m above the main
  floor (8 cells high). Obstacles: voxels holding mesh, the outside of the house (footprint from the 2-D slices, as the 2-D `House`),
  below and above the slab. Free space: the largest 6-connected component. Signed distance by distance transforms, trilinear table.
  Configuration (x, y, z) / box side; the interface's `phi_max` is the half-range of z. The robot is a point.
- **Lift onto 4 dimensions**: the scalar travel-time lift (`--lift time`), input (x, y, z, h). My reading of "scaling the lift dimension
  with the input dimensions": lifted dimension = input dimensions + 1, which is also what the 2-D tests do ((x, y, heading) -> 4).
  A lift with more than one extra coordinate was not run.
- **Prepare** (`prepare3.py`, `run.py --house3 <name>`, action `lbs-house3`): embed_warp's prepare unchanged on the House3; truth grid
  5 nodes per cell in x, y and z; every grid edge and seed segment that touches an obstacle is removed (clearance < 0.02 cells at
  samples <= 0.07 cells apart). Without this the truth leaked through the scans' one-voxel walls (smoke: 23 of 78 query paths
  uncertified; with it 0 of 78). Unreached nodes store 1000 (finite, so the planner control can interpolate). The table is stored as
  `house3_table.npy` and declared as an input of the field runs.
- **Runs**: `lbs|<house>3d` (`--lift time --bell 1 --bell-long 8 --lean --validate-every 1000`) and `ntrl|<house>3d`
  (`--lift none --weight 0 ...`), seed 87358, at the front of the queue (`queue/add3d.py`, `queue/prep3d.sh`). Houses: Allensville,
  Mifflinburg, Noxapater, Markleeville, Lakeville.
- **Klickitat** was taken out of the benchmark (report.py `SCENES`, page.py `FAMILIES`); its runs stay registered and in `runs.json`.
- Not adapted to 3-D (statistics only, no effect on training or scoring): `lift.segment_blocked` counts samples from the x, y length,
  the drive statistics `L_path` / `L_line` and `plan.score`'s `length_cells` are x, y lengths.
- Smoke runs: prepare `963ef834` (oracle planner 36 / 35 / 36 of 36, straight line 20 of 36), field `fd78491f`.

Controls that finished at 18:22 / 18:49 (11-20):
- `noLw|Lakeville` (`f16d151d`, no lift, Bellman weight x 0.2684 = Allensville's 3.8): 48.9 % solved, 48.6 % stalled, median T/T* 0.285,
  queries > 16 cells solved 1 / 1 / 1 / 8 / 6 of 120 at epochs 1,000-5,000 (full weight: 44.9 %). The size of the Bellman weight is
  not why no-lift loses the far field on Lakeville; what is left (scene size in cells itself) is untested.
- Third Allensville seed pair (seed 2): LBS 90.7 (`04b846cf`), no segment 91.2 (`68b6362a`); pairs now -1.7 / -2.5 / -0.6 for LBS.

3-D prepare runs (all `--label v1`): Allensville `90ac97fb`, Mifflinburg `fcaf1d95`, Noxapater `eecfd42b`, Markleeville `13be5158`,
Lakeville `c9cf7b86`. Controls on the test queries, straight line / planner on the true field (3 planning seeds):
Allensville 55.8 / 95.8-96.7, Mifflinburg 52.1 / 97.9-99.2, Noxapater 58.3 / 98.8, Markleeville 64.6 / 97.9, Lakeville 53.6 / 93.2-93.6.
First field result: ntrl-demo on Mifflinburg 3-D 68.2 % (`2727e491`; 28.9 % through a wall, 2.9 % stalled; 5 min of training).

3-D field results so far (20:05, 11-20; queries solved over planning seeds 0, 1, 2; through a wall / stalled as % of runs):
- Mifflinburg 3-D: ntrl-demo 68.2 (`2727e491`; 28.9 / 2.9), LBS 90.6 (`6dfa59f7`; 0.0 / 9.4; 86 min training beside two prepares).
- Allensville 3-D: ntrl-demo 64.7 (`585c4d0f`; 35.3 / 0.0), LBS 78.3 (`71d6a2c5`; 0.1 / 21.5; T within 10 % 67.2; paths of 32-64 cells 28 %).
- Noxapater 3-D: ntrl-demo 75.6 (`d6eee0aa`).
LBS removes the wall crossings and leaves stalls on the long queries; the cause of the stalls is not tested.
Queue order changed at 20:03 so that LBS on Lakeville 3-D (the longest run, pinned to GPU 1) starts before Noxapater and Markleeville.
- Markleeville 3-D: ntrl-demo 79.3 (`51711f85`; 17.6 / 3.1). Lakeville 3-D: ntrl-demo 58.2 (`6ac4101f`; 41.8 / 0.0). Noxapater ntrl split: 18.2 / 6.2.
  LBS on Lakeville 3-D started 20:08 (GPU 1), on Noxapater 3-D 20:09 (GPU 0); Markleeville LBS follows on the first free GPU.
- Lakeville 3-D: LBS 68.0 (`3fc6f1af`; 0.2 / 31.8; T within 10 % 63.5; 66 min) vs ntrl-demo 58.2.

Accuracy pairs without a true path (found 21:30, 11-20). `prepare3.py` stores 1000 at truth-grid nodes that no free edge reaches, and
`diag.truth_pairs` keeps every finite value, so the accuracy statistics of the 3-D runs included pairs with no true travel time
(`t_star_max` 1000, a '>128' distance bin with median ratio ~0). Shares: Lakeville 3-D 12.2-12.5 % of the 24,000 pairs, Markleeville 3-D
0.1 %, Allensville 3-D under 20 pairs, Mifflinburg and Noxapater none. On Lakeville three of the 48 truth goals reach only 2.0 % of the
grid nodes (the other 45 reach 49.9 %): a part of the free space (about 4 %) that is connected at the 2.5 cm voxels but not in the truth
grid; 4 of the 280 test queries lie inside it, none crosses into it. `report.py:no_path_pairs` takes these pairs out (within-10 %
recomputed from the other bins: Lakeville 3-D ntrl 59.3 -> 67.5, LBS 63.5 -> 72.5; median ratio and detour statistics dropped when
more than 1 % of the pairs are affected). Solved rates are unaffected (every query has a certified path). Training does not use the truth.
A proper fix (not done while runs are in flight): truth_pairs should drop t >= 100 for house3 scenes.

3-D Gibson test complete (22:27, 11-20): queries solved, ntrl-demo -> LBS (LBS run; through a wall / stalled % of LBS runs):
Allensville 64.7 -> 78.3 (`71d6a2c5`; 0.1 / 21.5), Mifflinburg 68.2 -> 90.6 (`6dfa59f7`; 0.0 / 9.4), Noxapater 75.6 -> 95.4 (`fd0e606d`; 0.0 / 4.6),
Markleeville 79.3 -> 92.5 (`9645d217`; 0.0 / 7.5), Lakeville 58.2 -> 68.0 (`3fc6f1af`; 0.2 / 31.8). Average 69.2 -> 85.0. One training seed.
LBS is ahead on 5 of 5 (+9.8 to +22.4); ntrl-demo fails through walls (17.6-41.8 % of runs), LBS by stalling on the long queries
(32-64 cells solved: 28 / 70 / 88 / 62 / 23 %). Planner on the true field: 93-99 %. Why Allensville and Lakeville stall more is untested.
