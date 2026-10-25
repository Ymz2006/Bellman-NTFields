# Cabinet task: Bellman NTFields from source vs metric NTFields

The user asked for a scene with only the cabinet, motion planning in and around it, with no
end-effector orientation rule (no cupping/pouring) and a speed that depends on clearance only; the
full Bellman pipeline run on it; and the metric NTFields baseline (`/media/4tbssd/jeffrey/ntrl-demo`)
run on the same task, comparing success rates.

## Task

`scenes/cabinet.yml`: the table plus the standing cupboard of `obstacle_scenes/scenes/shelf.yml`,
cuboids copied verbatim; no other obstacle, no zones, no task axes. The table stays because it is
the cupboard's floor (the bottom compartment has no floor panel) and the frozen FK sampler requires
a table cuboid. Cupboard outer box x 0.56–0.88, y 0.13–0.53, z −0.035–0.665 (0.32 deep, 0.40 wide,
0.70 tall); three compartments 0.30 × 0.36, open towards the robot.

Sampling box (grasp point), from the cupboard's outer box:

|Axis|Rule|Range (m)|
|---|---|---|
|z (height)|equal to the cupboard|−0.035 to 0.665|
|y (width)|1.5 × 0.40 = 0.60, centred on the cupboard|0.03 to 0.63|
|x (depth)|2.5 × 0.32 = 0.80, from the cupboard's back face towards the robot|0.08 to 0.88|

The depth runs from the back face towards the robot rather than being centred, so the extra depth
is the approach space in front of the opening and not space behind the closed back panel.

Speed: the original physical components without any tilt term,
`min(clip(d_obstacle, 0.005, 0.05), clip(0.5 d_self, 0.005, 0.05)) / 0.05` in [0.1, 1].
Validity (scorer, gates, certificates): obstacle clearance ≥ 0.005 m, self clearance ≥ 0.002 m,
joint limits; no orientation rule. The unchanged release scorer runs with `--tilt_deg 180`, which
switches its tilt gate off.

## Test suites (`tests/`, frozen)

`make_tests.py` through `run.py cabinet-tests`, then `freeze_tests.py`; `verify_tests.py` replays
every witness. Same certified recipe as the obstacle scenes (`obstacle_scenes/tests/README.md`):
endpoints from uniform joint samples that pass the endpoint rule, random pairs 0.02–4.0 rad apart,
a witness path (straight line or RRT-Connect, never a learned method) that the release scorer
accepts, and a seeded random choice of 100 certified queries per suite. Regions of the grasp point:
`free` = sampling box outside the cupboard's outer box; `compartment` = one of the three
compartment interiors (`compartments:` in the scene, 0.04 m past the opening); `box` = anywhere in
the sampling box.

|Suite|Start|Goal|
|---|---|---|
|free_to_free|free|free|
|free_to_compartment|free|compartment|
|compartment_to_free|compartment|free|
|compartment_to_compartment|compartment|another compartment|
|same_compartment|compartment|the same compartment|
|anywhere|box|box|

## Bellman pipeline (`run.py cabinet`)

`pipeline.py --method bellman`: the from-source recipe with the orientation removed.

1. `generate.py`: the frozen FK sampler (seed 42) collects 2,000,000 collision-free configurations
   with the grasp point in the sampling box (`--task cupping` without `--max_tilt_deg` is its
   unfiltered mode); 1,000,000 original pairs (one seeded permutation of the pool); clearance labels
   with the original label code; coverage augmentation (seed 20260912) in 15 strata, cupboard / box
   outside the cupboard / outside the box (within 0.15 m) × clearance-speed bins 0.1, (0.1, 0.4],
   (0.4, 0.7], (0.7, 1), 1, with 133,334 + 13,334 configurations each and 1,000,005 training pairs.
   Half of the perturbed bases come from pool configurations in the cupboard; with the original
   uniform bases the high-clearance cupboard strata appear at 26 per million proposals.
2. `prepare.py`: speed network and offline bank as `experiments/from_source/prepare.py` (same
   architecture, seeds, steps and bank builder). Both speed heads fit the same clearance label, so
   the network's FK-y blend of its heads is the identity. The bank's edge veto is
   `geometry.py ClearanceGate`: collision and joint domain only.
3. `experiments/from_source/training.py`, unchanged: 338,177-parameter TravelCost, losses 6/2/2,
   seed 87358, 5 rounds × 2 epochs, batch 256, round 5 evaluated.
4. `evaluate.py --method bellman`: native RK4 and the release MPPI (original settings, planning
   seeds 0/1/2) on the frozen suites.

## Metric NTFields baseline (`run.py cabinet-metric`)

The baseline's files are vendored unmodified in `metric/ntrl_demo/` (commit
`5ed065f6040bac0003fc0cf69874e1bdfbc50a6a`, hashes in `PROVENANCE.json`).

1. `metric/data.py`: the ntrl-demo arm recipe (`dataprocessing/speed_sampling_arm_normal.py`,
   500,000 pairs from `configs/arm.txt`): x0 near an obstacle (0 < d − 0.01 < 0.05 m), x1 a random
   step of length U(0, 0.5) from x0, speed `clip(d, 0.005, 0.05) / 0.05` and the unit clearance
   gradient as normal. Adapted to the task: x0 has its grasp point in the sampling box (no IK
   orientation cone), coordinates are the release joint normalization, d is the Bellman clearance
   with exact sphere–cuboid distances, so both methods learn the same speed.
2. `metric/train.py`: ntrl-demo's `model_train_metric.Model.train()` with its own settings
   (5,000 epochs, AdamW lr 5e-4, weight decay 0.5, alpha 1.025, TD and normal weights 1e-3).
   Shims only: `pickle5` → `pickle`, `igl` → an OBJ writer for the progress plots. Final epoch.
3. `evaluate.py --method metric`: native RK4 and the release MPPI exactly as for Bellman, plus the
   method's own MPPI (`tests/arm_plan_stat.py`: 50 samples, horizon 5, 200 steps, step 0.015,
   temperature 50), planning seeds 0/1/2.

## Commands

```bash
export BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86
./docker-python run.py cabinet-tests --gpu 2
python3 experiments/cabinet/freeze_tests.py results/registered_<cabinet-tests run>
setsid nohup ./docker-python run.py cabinet --gpu 2 > cabinet.log 2>&1 < /dev/null &
setsid nohup ./docker-python run.py cabinet-metric --gpu 2 > cabinet-metric.log 2>&1 < /dev/null &
./docker-python run.py cabinet-evaluate --run registered_<cabinet or cabinet-metric run> --gpu 2   # re-score a finished checkpoint
python3 experiments/cabinet/report.py --bellman results/registered_<cabinet run> --bellman-eval results/registered_<evaluate run> \
    --metric results/registered_<cabinet-metric run> --metric-eval results/registered_<evaluate run>
```

`diagnose.py` (read-only, pinned environment with the `launch.py` mounts) writes the diagnostics that
`report.py` adds to the comparison when `evidence/cabinet/diagnostics.json` exists.

## Results (2026-10-19, one RTX 3090)

|Run|Registered id|
|---|---|
|Test suites (frozen into `tests/`)|`20261019T033428Z-1342d9d2`|
|Bellman pipeline (sample, prepare, training)|`20261019T034213Z-f9c501f9`|
|Metric NTFields pipeline (data, training)|`20261019T034237Z-30d4d872`|
|Bellman evaluation of the round-5 checkpoint|`20261019T041026Z-39d9dafe`|
|Metric NTFields evaluation of the final epoch|`20261019T041045Z-9baa751f`|

Both pipeline runs finished training and then failed at their own evaluation stage on an inverted
assertion in `evaluate.py` (it required smoke suites for a full evaluation). After the fix (that
assertion, plus `--run`), `run.py cabinet-evaluate` scored the unchanged checkpoints; the report
checks that the evaluated checkpoint hashes are the pipelines' checkpoints.

All 600 certified queries (MPPI: 3 planning seeds, 1,800 executions):

|Method|RK4 (native)|Release MPPI|ntrl-demo MPPI (metric's own)|
|---|---:|---:|---:|
|Bellman NTFields|229/600 (38.2%)|688/1800 (38.2%)|—|
|Metric NTFields|182/600 (30.3%)|595/1800 (33.1%)|509/1800 (28.3%)|
|Straight joint-space line (reference)|202/600 (33.7%)|||

|Suite|Bellman RK4|Bellman MPPI|Metric RK4|Metric MPPI|Metric ntrl-demo MPPI|Straight line|
|---|---:|---:|---:|---:|---:|---:|
|free_to_free|91%|89.3%|81%|84.7%|78.3%|82%|
|free_to_compartment|22%|22.0%|10%|16.3%|8.0%|17%|
|compartment_to_free|22%|21.0%|10%|9.7%|9.7%|11%|
|compartment_to_compartment|3%|1.0%|1%|1.0%|0.3%|1%|
|same_compartment|5%|8.0%|9%|11.3%|3.0%|17%|
|anywhere|86%|88.0%|71%|75.3%|70.3%|74%|

With the same planner on the same queries, Bellman alone solves 65 RK4 queries that metric NTFields
fails (18 the other way) and 144 MPPI executions (51 the other way). Both methods do well in free
space and fail most compartment queries, almost always by hitting a cupboard panel (95–97% of the collision
failures for Bellman, 89–90% for metric NTFields). Along the certified witness paths the true clearance speed inside the cupboard averages
0.25–0.30; Bellman's travel cost implies 0.29–0.68 there and metric NTFields' 0.69–0.94, so both fields
underestimate the cost of moving inside the cupboard, metric NTFields more so. Training states in the
cupboard: 4.7% of the Bellman bank sources, 0.8% of the metric NTFields pairs. Full tables:
[`evidence/cabinet/COMPARISON.md`](../../evidence/cabinet/COMPARISON.md).

One training seed per method; planning seeds 0/1/2 are not training seeds; no significance claim.

## Clearance variants: 10% and 20% more clearance

The user then asked for slightly easier cabinet tasks with 10% and 20% more clearance, prepared and run
with everything above. `make_scenes.py` derives `scenes/cabinet_plus10.yml` and `scenes/cabinet_plus20.yml`
from `scenes/cabinet.yml` (with scale 1 it reproduces `cabinet.yml`; `--check` re-derives the files):
every compartment opening is 1.1x / 1.2x as wide and each compartment 1.1x / 1.2x as tall, so the arm has
that much more room in every compartment. The panels (0.02 m), the floor (table top), the centre line
(y 0.33), the back face (x 0.88) and the depth (0.30 m inside) are unchanged: depth adds reach, not room
around the arm, and a deeper cupboard would push the 2.5x-depth sampling box into the robot base. The
sampling box follows the same rule from the new cupboard, and the compartment boxes are the new interiors.

|Scene|Opening width|Compartment heights bottom / middle / top|Cupboard outer d x w x h|Sampling box y / z|
|---|---:|---:|---:|---|
|`cabinet`|0.36|0.225 / 0.21 / 0.205|0.32 x 0.40 x 0.70|0.03–0.63 / −0.035–0.665|
|`cabinet_plus10`|0.396|0.2475 / 0.231 / 0.2255|0.32 x 0.436 x 0.764|0.003–0.657 / −0.035–0.729|
|`cabinet_plus20`|0.432|0.27 / 0.252 / 0.246|0.32 x 0.472 x 0.828|−0.024–0.684 / −0.035–0.793|

Everything else is the cabinet recipe unchanged, run per scene with `--scene`: certified suites from the
same generator and seeds, frozen into `tests_plus10/` and `tests_plus20/` (all witnesses replayed by
`verify_tests.py`, `evidence/cabinet/plus*/tests_verify.json`); the Bellman pipeline (`generate.py` now
takes the FK sampler's `--height` from the scene, 0.70 / 0.764 / 0.828 m) and the metric NTFields
pipeline, each trained from scratch in its scene and evaluated on that scene's frozen suites.

```bash
python3 experiments/cabinet/make_scenes.py --check
./docker-python run.py cabinet-tests --scene cabinet_plus10 --gpu 0
python3 experiments/cabinet/freeze_tests.py results/registered_<cabinet-tests run>
setsid nohup ./docker-python run.py cabinet --scene cabinet_plus10 --gpu 0 > cabinet_plus10.log 2>&1 < /dev/null &
setsid nohup ./docker-python run.py cabinet-metric --scene cabinet_plus10 --gpu 2 > cabinet_metric_plus10.log 2>&1 < /dev/null &
python3 experiments/cabinet/report.py --scene cabinet_plus10 --bellman results/registered_<cabinet run> --metric results/registered_<cabinet-metric run>
python3 experiments/cabinet/clearance_report.py   # all three scenes -> evidence/cabinet/CLEARANCE.md
```

|Run|cabinet_plus10|cabinet_plus20|
|---|---|---|
|Test suites (frozen)|`20261019T171426Z-986a0e88` → `tests_plus10/`|`20261019T171445Z-37602704` → `tests_plus20/`|
|Bellman pipeline (sample, prepare, training, evaluation)|`20261019T171925Z-c5ebfc5b`|`20261019T171949Z-c41ae437`|
|Metric NTFields pipeline (data, training, evaluation)|`20261019T172013Z-4aa0e26a`|`20261019T172037Z-75a8efe5`|

Results (2026-10-19; the Bellman runs on GPUs 0 and 1, both metric NTFields runs sharing GPU 2), each scene on
its own 600 certified queries (MPPI: 3 planning seeds, 1,800 executions):

|Scene|Bellman RK4|Bellman release MPPI|Metric RK4|Metric release MPPI|Metric ntrl-demo MPPI|Straight line|
|---|---:|---:|---:|---:|---:|---:|
|`cabinet`|229/600 (38.2%)|688/1800 (38.2%)|182/600 (30.3%)|595/1800 (33.1%)|509/1800 (28.3%)|202/600 (33.7%)|
|`cabinet_plus10`|224/600 (37.3%)|682/1800 (37.9%)|183/600 (30.5%)|600/1800 (33.3%)|513/1800 (28.5%)|186/600 (31.0%)|
|`cabinet_plus20`|223/600 (37.2%)|646/1800 (35.9%)|194/600 (32.3%)|615/1800 (34.2%)|549/1800 (30.5%)|208/600 (34.7%)|

More clearance did not make either method more successful. The rooms are roomier (true clearance speed at
the compartment endpoints 0.33 → 0.37 → 0.41), but the queries are not easier (straight line 31–35%; more of
every compartment becomes reachable, and the endpoints lie slightly deeper), and the two limits of both
fields are unchanged: 89–98% of collision failures hit a cupboard panel, and inside the cupboard Bellman's
travel time implies about 2x and metric NTFields' about 3x the true clearance speed. On the four compartment
suites Bellman's release MPPI rate drifts from 13.0% to 12.1% and 10.3% and metric NTFields' from 9.6% to
9.8% and 10.8%, so Bellman's overall lead with the same planner shrinks from +5.2 to +4.6 and +1.7 points;
these drifts are about one standard error of the query draw, and one training seed per run cannot separate
them from seed variation. Full tables and diagnostics: [`evidence/cabinet/CLEARANCE.md`](../../evidence/cabinet/CLEARANCE.md),
per scene `evidence/cabinet/plus10/` and `evidence/cabinet/plus20/`. `diagnose.py` now also records query
difficulty (section 5); its re-run on the cabinet scene reproduced the earlier diagnostics exactly.
