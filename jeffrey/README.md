# jeffrey/: Bellman-sharp and the guarded hierarchical-distance planner

Two ideas for the cabinet Bellman task (`experiments/cabinet`: 6 frozen certified suites x 100 queries, release
scorer in the scene with the tilt gate off: obstacle clearance >= 5 mm and self clearance >= 2 mm at every
0.001 rad of the path, joint limits, exact endpoints). Planning seeds 0/1/2 are planner seeds, not training seeds.

| Idea | What | Result (cabinet, 1800 executions) |
|---|---|---|
| 2 (main) | learned hierarchical distance model + guarded bidirectional planner | **1793/1800 (99.6%)** (v2; v1 1778), 0 collisions on certified paths; plus10 1791, plus20 1794 |
| 1 | Bellman-only refinement of a pretrained P-NTFields field | **no gain**: release MPPI 646 -> 605, P-NTFields planner 228 -> 220 /600 |

For reference on the same suites: Bellman baseline release MPPI 682/1800 and RK4 228/600
(`registered_20261025T070429Z-b9a2383c`); the best earlier guard (`experiments/face_speed`, per-piece step filter on
the face-trained field) 1217/1800 ntrl-MPPI and 375/600 RK4; metric NTFields best 596/1800.

## Idea 2: hierarchical distance model and guards

**Distance model** (`distance.py`, `geometry.py`). The hierarchy is q -> exact forward kinematics -> the 53 release
collision spheres -> learned point distance D_theta(c_i) -> sphere clearance D_theta(c_i) - r_i -> min. Only the
3-D function D (distance from a point to the scene) is learned (PointField: Fourier features + 5 x 256 SiLU MLP);
self-collision is exact (sphere pairs with the release mask). Trained 60k steps on fresh points every step (sphere
centres of pool and uniform configurations, jittered centres, points near the cuboid faces, workspace points) with a
surface-weighted L1 that penalizes over-estimation 4x. Calibration on 32M fresh points: near-surface MAE 0.09-0.33 mm
on sphere centres; worst over-estimation 3.8 mm, p99.99 1.2-1.5 mm. The guard threshold is 5 mm + eps with
eps = 4.0 mm (worst case, rounded up). The geometry matches the release scorer's own functions to 6e-8 m.

**Certified segments.** Rotating joint j moves sphere i by at most |dq_j| R_ij, with R_ij a chain-length bound valid
at every configuration (checked: worst observed ratio 0.999). Along a straight joint-space segment the planner
advances by the largest step for which no sphere can lose its clearance surplus (conservative advancement); a
certified segment is collision-free whenever the model error is within eps. Self pairs use only the joints that
move one sphere of the pair.

**Guards** (`planner.py`):
1. *Departure*: 1 in 5 test endpoints is closer than 5 mm + eps (compartment endpoints have median clearance 13 mm).
   They are moved out first; every scorer-resolution point must keep each tight sphere above the threshold or no
   closer than at the (valid) endpoint. Candidates: soft-min ascent, its noisy versions, random directions and (v2)
   the ascent projected onto "every tight sphere moves outward" (Hildreth).
2. *Barrier*: candidate steps ranked by V(candidate, target) + 2 relu(1 cm - surplus) / 1 cm.
3. *Certified step*: a step is taken only if its whole segment certifies (best candidate, second best, half step).
4. *Connection*: 4 particles per query, each with a forward branch from the start and a backward branch from the
   goal descending the field towards each other; heads are joined to heads and random branch nodes of every particle
   of the query whenever the joining segment certifies. The first join ends the query, so a returned path consists
   of certified segments only.
5. *Escape*: random jump + persistent random direction after 12 iterations without progress.

### Results (registered, host backend, RTX 3090)

Field: the cabinet Bellman baseline TravelCost (`b9a2383c`). Runs `registered_20261026T062255Z-97e4c3cc` (distance
model + main variants) and `registered_20261026T064855Z-37ec7a76` (guard ablations, same distance model).

|Variant|Field|Distance, eps|Success /1800|Per seed|Connected|Certified paths that failed|s/query|Path rad / m|
|---|---|---|---:|---|---:|---:|---:|---|
|`full`|Bellman|learned, 4.0 mm|**1778 (98.8%)**|593, 594, 591|1777|0|0.43|4.58 / 1.23|
|`eps1p5`|Bellman|learned, 1.5 mm (p99.99)|1790 (99.4%)|596, 598, 596|1790|0|0.32|4.51 / 1.21|
|`exact` (control)|Bellman|exact distance|1795 (99.7%)|599, 596, 600|1795|0|0.31|4.44 / 1.19|
|`euclid` (control)|none (joint distance)|learned, 4.0 mm|1752 (97.3%)|588, 583, 581|1749|0|0.52|5.49 / 1.52|
|`bi_k4_noescape`|Bellman|learned, 4.0 mm|1778 (98.8%)|594, 596, 588|1776|0|0.44|5.30 / 1.47|
|`bi_k1`|Bellman|learned, 4.0 mm|1328 (73.8%)|445, 436, 447|1319|0|0.37|3.39 / 0.78|
|`uni_k1` (certified steps only)|Bellman|learned, 4.0 mm|1095 (60.8%)|369, 362, 364|1086|0|0.34|3.15 / 0.70|

|Variant|free→free|free→comp.|comp.→free|comp.→comp.|same comp.|anywhere|
|---|---:|---:|---:|---:|---:|---:|
|`full`|300|300|291|294|294|299|
|`eps1p5`|300|298|296|299|297|300|
|`exact`|300|300|296|300|299|300|
|`euclid`|299|297|288|280|288|300|
|`bi_k4_noescape`|300|300|290|296|293|299|
|`bi_k1`|297|229|206|119|183|294|
|`uni_k1`|288|122|219|54|125|287|
|Bellman baseline, release MPPI|268|67|64|3|22|258|

What the table says:
- **Collisions are prevented before they happen.** Across all 12,600 executions, not one certified path was
  rejected by the scorer. Every failure is a query that did not connect within 300 iterations (its fallback path,
  start branch + straight line to the goal, then usually collides). Even at eps 1.5 mm, below the measured worst
  error, no certified path failed.
- **Learned vs exact distance**: 1778 vs 1795. The gap is the 4 mm safety margin, not accuracy: at eps 1.5 mm the
  learned model gets 1790. 16 of the 22 `full` failures are queries whose endpoint departure got stuck (4 endpoints x
  ~3 seeds). Departure v2 (projected cone) frees 3 of those 4 endpoints (1 of 1200 endpoints left, at 5.0 mm
  clearance); its registered rerun is listed below.
- **Which guards matter**: certified steps alone (`uni_k1`, essentially the earlier step filter made exact) 60.8%;
  bidirectional search +13 points; 4 particles with cross-particle connection +25 points, the largest single gain;
  escape adds nothing (`bi_k4_noescape` = `full`).
- **What the learned field contributes**: with the joint-space distance as the field (`euclid`) the guards still
  solve 97.3%, so success comes mainly from the distance model and the search. The Bellman field adds 26 executions
  (mostly compartment-to-compartment, 294 vs 280), 17% shorter joint paths, 19% shorter end-effector paths and 17%
  less planning time.

### Departure v2, shortcutting and other scenes

|Run|Scene|Variant|Field|eps|Success /1800|Per seed|Certified paths that failed|s/query|Path rad / m|
|---|---|---|---|---:|---:|---|---:|---:|---|
|`090311Z-34199422`|cabinet|`full_v2` (departure v2)|Bellman|4.0 mm|**1793 (99.6%)**|598, 597, 598|0|0.77*|4.58 / 1.24|
|`090311Z-34199422`|cabinet|`full_v2_shortcut`|Bellman|4.0 mm|1791 (99.5%)|597, 598, 596|0|0.34|**3.44 / 0.76**|
|`090311Z-34199422`|cabinet|`eps1p5_v2`|Bellman|1.5 mm|1794 (99.7%)|598, 598, 598|0|0.28|4.53 / 1.22|
|`075256Z-0c0895e4`|cabinet_plus10|`full`|Bellman plus10 (`c5ebfc5b`)|3.5 mm|**1791 (99.5%)**|597, 595, 599|0|0.61*|4.67 / 1.30|
|`075256Z-0c0895e4`|cabinet_plus10|`euclid`|none|3.5 mm|1785 (99.2%)|596, 594, 595|0|0.50|5.22 / 1.46|
|`090317Z-ff0408d7`|cabinet_plus20|`full`|Bellman plus20 (`c41ae437`)|3.5 mm|**1794 (99.7%)**|597, 598, 599|0|0.25|4.40 / 1.22|
|`090317Z-ff0408d7`|cabinet_plus20|`euclid`|none|3.5 mm|1781 (98.9%)|593, 593, 595|0|0.31|5.25 / 1.51|

|`100515Z-7dbeaefa`|cabinet|`pnt_full`|P-NTFields (`41e5b5e9`, epoch 10,000)|4.0 mm|**1794 (99.7%)**|598, 599, 597|0|0.08†|4.22 / 1.19|
|`100515Z-7dbeaefa`|cabinet|`pnt_full_shortcut`|P-NTFields|4.0 mm|1793 (99.6%)|599, 596, 598|0|0.08†|3.52 / 0.87|

\* wall-clock per query depends on how many jobs shared the 3090 at the time (up to four); compare times only
within one run. † run G had the 3090 to itself: 0.08 s per query is the planner's actual speed
(600 queries in about 50 s per seed, batched).

With the P-NTFields field as the planner's value field, the guards give the same success as with the Bellman field
(1794 vs 1793), with slightly shorter paths (4.22 vs 4.58 rad). On its own, the same P-NTFields field scores 646/1800
with the release MPPI and 228/600 with its own planner.

|Variant|free→free|free→comp.|comp.→free|comp.→comp.|same comp.|anywhere|
|---|---:|---:|---:|---:|---:|---:|
|cabinet `full_v2`|300|300|298|298|297|300|
|cabinet `full_v2_shortcut`|300|300|297|298|297|299|
|cabinet `eps1p5_v2`|300|300|298|300|297|299|
|plus10 `full`|300|299|299|296|297|300|
|plus20 `full`|300|295|300|299|300|300|
|plus20 `euclid`|300|295|298|289|299|300|

- Departure v2 adds the direction projected onto "every tight sphere moves outward" to the departure candidates. It
  frees 3 of the 4 endpoints the first version could not leave (1 of 1200 endpoints left, at 5.0 mm exact
  clearance), and the cabinet goes from 1778 to 1793. The remaining 7 failures: 3 executions of that one query, the
  rest are queries that did not connect in 300 iterations.
- Certified shortcutting (greedy jump to the farthest later waypoint whose segment certifies) leaves success
  unchanged and makes paths 25% shorter in joint space and 39% shorter for the end effector.
- The distance model and guards carry over unchanged to the 10% and 20% larger cupboards (each with its own distance
  model from the same recipe, eps 3.5 mm): 99.5% and 99.7%, against 546 and 558 /1800 for those scenes' Bellman
  release MPPI. The field again adds a little over the joint-distance control (plus20: 1794 vs 1781, mostly
  compartment-to-compartment) and shortens paths by about 16%.
- Run `075251Z-39686215` (registry: failed) completed an identical `full_v2` (1792/1800) before crashing in its
  shortcut variant on an index bug in `GuardedPlanner.shortcut` for 2-waypoint paths (fixed); run E reran it.

Development history of Idea 2 (unregistered, seed 0, scratchpad): first version with the simple departure 562/600 (eps 4 mm)
and 578/600 (eps 1.5 mm); candidate-based departure 595/600 and 597/600.

## Idea 1: Bellman-sharp

`bellman_sharp.py`: start from a P-NTFields field trained on the cabinet (`run.py pntfields-cabinet`,
registered `20261026T053609Z-41e5b5e9`); each generation freezes a copy T_k and trains the field for one bank epoch on
SmoothL1(T(q, g), min over safe actions a of c(q, a) + T_k(q'_a, g)), Bellman term only, over the cabinet Bellman
bank (399,581 anchors x 32 actions, exact collision veto). Local costs are in P-NTFields' units (x = q / 2 pi,
speed clip(d_env, m/10, m)/m, m = 0.5/12 m) and use the exact distance query (157 s for the 217 M speed evaluations).

Registered run `registered_20261026T092413Z-9eb5fd8c`: the final P-NTFields checkpoint (epoch 10,000), 10
generations x 1 bank epoch (15,610 updates), lr 1e-4, batch 256; generation 10 evaluated with the unchanged P-NTFields
cabinet protocol (`experiments/pntfields_cabinet/evaluate.py`); generation 0 is the P-NTFields run's own evaluation.

|Field|RK4 /600|P-NTFields planner /600|ntrl-demo MPPI /1800|Release MPPI /1800|
|---|---:|---:|---:|---:|
|P-NTFields, generation 0 (`41e5b5e9`)|197|228|633|646|
|Bellman-sharp, generation 10|190|220|631|605|

Per suite (release MPPI, gen 0 -> gen 10): free→free 260 -> 255, free→comp. 58 -> 52, comp.→free 63 -> 45,
comp.→comp. 15 -> 8, same comp. 15 -> 18, anywhere 235 -> 227. Failures stay collisions (1154 -> 1195 of 1800).

Why it does not work (`sharp_diagnose.py`, 8192 probe pairs of bank anchors, 7568 of whose straight segments collide):

|Checkpoint|T/T0, collision-free pairs|T/T0, colliding pairs|Bellman gap, free|Bellman gap, colliding|
|---|---:|---:|---:|---:|
|generation 0|1.000|1.000|0.44%|0.18%|
|generation 5|1.016|1.000|0.45%|0.26%|
|generation 10|1.033|1.007|0.45%|0.27%|

- The pretrained field is already Bellman-consistent on the bank: the median one-step gap (backup - T)/T is below
  0.5%, and no probe pair is above 10%.
- The update raises T on collision-free pairs (+3.3%) more than on colliding ones (+0.7%), the opposite of a
  correction where the field tunnels through the cupboard. The rise on free pairs is the bias of a min over 32
  discrete actions (backup >= continuous optimum).
- Mechanism: the backup compares a state only with its neighbours one action away, where the frozen field is wrong
  in the same way. The exact veto removes only the 1-2 actions that cross the obstacle next to it, and the remaining
  actions are almost as good, so the detour cost enters at about one action step per generation and stays below the
  fitting noise. Making Bellman fix collisions would need multi-step backups or online transitions along the field's
  own failing paths. Both go beyond "Bellman update only" and were not run.
- Development run on the epoch-2000 checkpoint (unregistered): the same picture (RK4 193 -> 185, P-NTFields planner
  216 -> 217; T +4.6% on free vs +3.9% on colliding pairs).

## Commands

```bash
export BELLMAN600_BACKEND=host PY=~/.venvs/bellman-ntfield/bin/python
# Idea 2: train the distance model and evaluate variants (each variant = evaluate.py options)
$PY jeffrey/run.py guard --field bellman --field-run registered_20261025T070429Z-b9a2383c --train-distance \
    --variants '[{"name": "full", "args": ""}, {"name": "euclid", "args": "--field euclid"}]' --gpu 1
# reuse a distance model: --distance-run registered_<guard run>; other scenes: --scene cabinet_plus10 ...
# Idea 1
$PY jeffrey/run.py sharp --pnt-run registered_20261026T053609Z-41e5b5e9 --sharp-args "--generations 10" --eval-generations 10 --gpu 1
# tables
$PY jeffrey/report.py --guard results/registered_<A> results/registered_<B> --sharp results/registered_<S>
```

`jeffrey/run.py` registers through `tools/experiment_registry` like the top-level `run.py` (duplicate check,
declared inputs hashed at launch and at the end, outputs in `results/registered_<id>/`). Its command is
`jeffrey/launch.py`, which reuses `launch.py`'s mounts and host/Docker command builders by import. The top-level
`run.py`/`launch.py` were not edited because they are declared inputs of other sessions' queued runs. Wiring a
`jeffrey` action into them is a small additive change once nothing is in flight.

Environment: host backend (bwrap + venv, torch 2.7.1+cu128), not the pinned image; every run records
`environment_override.json`. Single training seed for every learned model; no claim of statistical significance.
