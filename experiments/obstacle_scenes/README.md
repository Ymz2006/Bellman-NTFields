# Obstacle scenes, trained from source

The user requested three more obstacle scenes beyond the transition walls, reviewed them in a viser
preview before any training, and asked for the clutter scene to keep one object per task region.
Each scene only appends boxes to the frozen no-wall scene. Table, zones, task axes, sampling bounds,
normalization, the 600 queries, the 20° OR rule, 0.005/0.002 m margins and 0.001 rad dense scoring
are unchanged. The user then asked for each scene's testing data to be recomputed and frozen: every
scene also has its own 600 certified queries in `tests/<scene>/` (see [tests/README.md](tests/README.md)).

|Scene|Added boxes|Frozen 600 queries with valid endpoints|
|---|---|---:|
|`scenes/platforms.yml`|Two 0.50 × 0.40 × 0.15 m platforms under the task regions: x 0.40–0.90, y +0.15–+0.55 (cupping) and −0.55–−0.15 (pouring), top z = 0.115. The gap \|y\| ≤ 0.15 stays at table height.|438/600|
|`scenes/shelf.yml`|Standing cupboard on the cupping side, open towards the robot (−x): outer x 0.56–0.88, y 0.13–0.53, z −0.035–0.665, 0.02 m panels. Three compartments 0.30 deep × 0.36 wide: bottom z −0.035–0.19, middle 0.21–0.42, top 0.44–0.645.|456/600|
|`scenes/clutter.yml`|One object per task region: bottle 0.08 × 0.08 × 0.30 at (0.48, +0.28) (cupping), jar 0.10 × 0.10 × 0.16 at (0.62, 0.00) (transition), kettle 0.18 × 0.16 × 0.24 at (0.70, −0.33) (pouring).|497/600|

Obstacle names must not contain `table`: the FK sampler treats such cuboids as the table.

## Pipeline

`pipeline.py` runs the transition-wall recipe with the scene passed to every stage: endpoint audit,
FK sampling and labels (seeds 42/42/20260912), speed network and bank (seeds 87358/87344),
the unchanged 338,177-parameter TravelCost (losses 6/2/2, seed 87358, 5 rounds), and round-5
evaluation on the fixed 600 queries (MPPI planning seeds 0/1/2, native RK4, strict scorer in the scene).
The audit and evaluator are `experiments/transition_wall/audit.py` and `evaluate.py`, unchanged;
their `wall_*` fields and labels cover every added obstacle. A last stage, `evaluate_tests.py`, runs
the same evaluation on the scene's frozen test suites. The platforms run predates that stage; its
checkpoint was scored on the platforms suites with `obstacle-evaluate`.

```bash
export BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86
./docker-python run.py obstacle-tests --scene clutter --gpu 2      # generate suites; then freeze into tests/clutter/
./docker-python run.py obstacle-scene --scene clutter --gpu 2 --smoke
setsid nohup ./docker-python run.py obstacle-scene --scene clutter --gpu 2 > clutter.log 2>&1 < /dev/null &
./docker-python run.py obstacle-evaluate --scene platforms --run registered_20261013T194621Z-1978fc0c --gpu 2
./docker-python experiments/obstacle_scenes/report.py . results/registered_<scene run>... --no-wall results/registered_20261009T042727Z-c1d4e622 --tests-eval results/registered_<obstacle-evaluate run>
```

Start full runs detached: an earlier wall run was killed when the Claude Code session that started it
ended. The spec declares the scene file, `pipeline.py` and `run.py` individually rather than this
folder, so editing another scene, `targets.yml` or the preview while one scene runs does not
invalidate that run's input identity. Outputs go to registered `results/registered_<id>/`;
`report.py` writes `evidence/obstacle_scenes/`.

## Results

Compared with the no-wall from-source run `registered_20261009T042727Z-c1d4e622` (same pipeline,
seeds and machine). Endpoint-valid queries are those whose start and goal pass the strict check in
the scene; every success falls in that subset. Full tables: `evidence/obstacle_scenes/COMPARISON.md`.

Frozen 600 queries:

|Scene|Run|MPPI, all 600|RK4, all 600|MPPI, endpoint-valid|No-wall MPPI, same queries|
|---|---|---:|---:|---:|---:|
|platforms|`registered_20261013T194621Z-1978fc0c`|1163/1800 (64.61%)|396/600 (66.00%)|1163/1314 (88.51%)|1204/1314 (91.63%)|
|clutter|`registered_20261013T210624Z-f04a0fc2`|1232/1800 (68.44%)|420/600 (70.00%)|1232/1491 (82.63%)|1347/1491 (90.34%)|

The scene's own certified suites (MPPI over 3 seeds; the platforms checkpoint was scored by
`registered_20261013T210736Z-4a8e8aed`, the clutter checkpoint by its pipeline's last stage):

|Scene|All 600: MPPI|RK4|General cupping / pouring / cross (MPPI)|Scene suites: cupping / pouring / cross (MPPI)|
|---|---:|---:|---|---|
|platforms|1301/1800 (72.28%)|456/600 (76.00%)|91.3% / 99.0% / 36.3%|platform tops 91.7% / 94.0% / 21.3%|
|clutter|1209/1800 (67.17%)|414/600 (69.00%)|88.0% / 93.3% / 41.3%|low endpoints 67.0% / 96.0% / 17.3%|

For reference, the no-wall run scores cupping 84.4%, pouring 96.7% and cross 40.8% (MPPI) on the
frozen certified `mixed` suite, which was built the same way without obstacles. Certified random
cross pairs are much harder than the task suites' cross segments, so suite totals are not
comparable with the frozen 600 totals.

Replay a run in the scene:
`./docker-python gui/viewer.py --evaluation-dir results/registered_<id>/evaluation --scene experiments/obstacle_scenes/scenes/<scene>.yml --port 8904`.

## Test suites and design tools

- `make_tests.py` (`run.py obstacle-tests`): the scene's certified test suites; see
  [tests/README.md](tests/README.md). Copied verbatim into `tests/<scene>/` once, then frozen.
- `targets.yml`: grasp-frame regions of the scene suites' endpoints (platform tops, shelf
  compartments, object height). Not read by training or the frozen-600 evaluation.

The tools below are not used by training or evaluation.

- `design.py`: frozen-600 endpoint validity per scene and valid example poses per target region,
  from 400 million uniform joint samples (seed 20261013, about 2 minutes on one RTX 3090). It runs in
  the pinned GPU environment with the same mounts as `launch.py`.
- `preview.py`: viser preview of the scenes, target regions, example poses and blocked frozen
  endpoints: `./docker-python experiments/obstacle_scenes/preview.py --data <design.py output> --port 8903`.

In the frozen 600, valid endpoints do not prove that a collision-free path exists; every scene-suite
query does have one (its witness). One training seed per scene supports no significance claims.
