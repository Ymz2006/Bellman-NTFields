# Frozen obstacle-scene test suites

Each obstacle scene has its own testing data: 600 certified queries in six suites of 100, generated
once with fixed seeds by `make_tests.py` through `run.py obstacle-tests` and frozen here. Never
resample them; the frozen 600 no-wall queries in `runtime/tests/` are unchanged and still evaluated.

Every query was built like the certified frozen `mixed` suite:

- **Endpoints** come from uniform joint samples (seed 20261014) whose grasp point lies in the
  stratum's region and that pass the strict endpoint rule in the scene: obstacle clearance
  ≥ 0.005 m, self clearance ≥ 0.002 m, the 20° orientation rule of the endpoint's region, and the
  normalization joint bounds. Endpoints never lie in the transition band, as in the frozen suites.
- **Pairs** are random (seed 20261015) with joint-space distance 0.02–4.0 rad. Cross pairs run in
  both directions with equal probability; the shelf strata always end inside the cupboard.
- **Certificate**: a witness path that the unchanged release scorer (`score_cross_suite.py`)
  accepts in the scene at 0.001 rad resolution. Witnesses come from an independent planner, the
  straight joint-space line or bidirectional RRT-Connect (seed 20261016), never from the Bellman
  method. Candidates without a certificate are dropped.
- **Selection**: 100 queries per suite, a seeded random choice among the certified candidates
  (seed 20261017), not a choice by difficulty.

|Scene|General suites (all heights, whole regions)|Scene suites|
|---|---|---|
|platforms|general_cupping, general_pouring, general_cross|platform_cupping, platform_pouring (both endpoints within 0.15 m above that platform), platform_cross (one on each platform)|
|shelf|general_cupping, general_pouring, general_cross|shelf_from_cupping (cupping region outside the cupboard → compartment), shelf_from_pouring (pouring region → compartment), shelf_to_shelf (compartment → another compartment)|
|clutter|general_cupping, general_pouring, general_cross|low_cupping, low_pouring (both endpoints at most 0.25 m high in that region), low_cross (one low endpoint in each region)|

Each `<scene>/` holds `suite_manifest.json` (suite files and sha256, scene sha256, seeds),
`generation.json` (candidate and certification counts per suite) and `suites/*.npz` in the frozen
suite format, plus `witness_*`, `direct_line_strict_feasible` and `query_group_label`.
`evaluate_tests.py` refuses suites whose hashes or scene file differ from the manifest.

`freeze_tests.py` copied each registered generation here verbatim after checking every hash, the
scene file and the generator against the manifest, and wrote `FROZEN.json`. `verify_tests.py`
replays the stored witnesses through the release scorer: all 1,800 were accepted, and no query
repeats within or across suites. The clutter suites were frozen before the clutter training run
started, and the shelf suites before any shelf training.

## Certification

RRT-Connect: 0.2 rad steps, 10,000 iterations, 10% goal bias, targets drawn from valid configurations near the table.
Uncertified candidates are not necessarily infeasible, so the kept queries lean towards ones the planner could solve;
the certified share per suite shows how strong that selection was.

|Scene|Suite|Candidates|Certified|Straight line in the 100 kept|Generating run|
|---|---|---:|---:|---:|---|
|platforms|general_cupping|160|143 (89%)|75|`registered_20261013T205443Z-cfbb59fe`|
|platforms|general_pouring|160|151 (94%)|78|`registered_20261013T205443Z-cfbb59fe`|
|platforms|general_cross|160|131 (82%)|8|`registered_20261013T205443Z-cfbb59fe`|
|platforms|platform_cupping|160|127 (79%)|59|`registered_20261013T205443Z-cfbb59fe`|
|platforms|platform_pouring|160|157 (98%)|56|`registered_20261013T205443Z-cfbb59fe`|
|platforms|platform_cross|160|105 (66%)|1|`registered_20261013T205443Z-cfbb59fe`|
|shelf|general_cupping|160|136 (85%)|56|`registered_20261013T210448Z-651bed01`|
|shelf|general_pouring|160|148 (92%)|74|`registered_20261013T210448Z-651bed01`|
|shelf|general_cross|160|114 (71%)|16|`registered_20261013T210448Z-651bed01`|
|shelf|shelf_from_cupping|160|108 (68%)|16|`registered_20261013T210448Z-651bed01`|
|shelf|shelf_from_pouring|320|150 (47%)|1|`registered_20261013T210448Z-651bed01`|
|shelf|shelf_to_shelf|320|165 (52%)|0|`registered_20261013T210448Z-651bed01`|
|clutter|general_cupping|160|144 (90%)|63|`registered_20261013T205357Z-a7474a7c`|
|clutter|general_pouring|160|149 (93%)|66|`registered_20261013T205357Z-a7474a7c`|
|clutter|general_cross|160|132 (82%)|13|`registered_20261013T205357Z-a7474a7c`|
|clutter|low_cupping|160|112 (70%)|48|`registered_20261013T205357Z-a7474a7c`|
|clutter|low_pouring|160|157 (98%)|83|`registered_20261013T205357Z-a7474a7c`|
|clutter|low_cross|320|183 (57%)|3|`registered_20261013T205357Z-a7474a7c`|
