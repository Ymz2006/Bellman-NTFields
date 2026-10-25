# Cabinet clearance sweep: Bellman NTFields vs metric NTFields

The cabinet task (`experiments/cabinet/README.md`) in three scenes: the original cupboard and two variants with 10% and 20% more clearance, where every compartment opening is 1.1x / 1.2x as wide and each compartment 1.1x / 1.2x as tall (`make_scenes.py`; panels, floor, back face and depth unchanged; the sampling box follows the same rule from the new cupboard). Each scene was prepared and trained from scratch with both methods and scored on its own frozen, certified suites: 6 suites x 100 queries; MPPI rows count 3 planning seeds per query (1,800 executions).

## Headline (cabinet -> +10% -> +20% clearance)

- Release MPPI, all 600 queries: Bellman 38.2% -> 37.9% -> 35.9%; metric NTFields 33.1% -> 33.3% -> 34.2%.
- RK4, all 600 queries: Bellman 38.2% -> 37.3% -> 37.2%; metric NTFields 30.3% -> 30.5% -> 32.3%.
- Release MPPI, the four compartment suites: Bellman 13.0% -> 12.1% -> 10.3%; metric NTFields 9.6% -> 9.8% -> 10.8%; straight line 11.5% -> 10.2% -> 12.0%.
- Bellman minus metric NTFields, release MPPI: +5.2 -> +4.6 -> +1.7 points.

## Reading

- The geometry is roomier, as intended: the true clearance speed at the compartment endpoints rises from 0.33 to 0.37 and 0.41, and along the certified witness paths inside the cupboard from 0.27 to 0.29 and 0.32.
- The queries did not get easier: the straight line solves 31-35% of each scene's queries and 10-12% of its compartment queries. More room also makes more of every compartment reachable (test-suite compartment pools 1.8x and 3.0x larger), and the compartment endpoints lie slightly deeper (0.086 -> 0.093 -> 0.098 m past the opening).
- Neither method turns the extra room into success. Overall rates move by at most 2.3 points, about one standard error of the query draw. On the compartment suites Bellman drifts down (release MPPI 13.0 -> 12.1 -> 10.3%) and metric NTFields up (9.6 -> 9.8 -> 10.8%), so Bellman's lead with the same planner shrinks from +5.2 to +4.6 and +1.7 points overall. Bellman still solves more queries alone than metric NTFields in every scene (RK4: 65 vs 18, 61 vs 20, 50 vs 21).
- What limits both methods is unchanged by the clearance: 89-98% of the collision failures hit a cupboard panel, mostly early in the path, and inside the cupboard both travel-time fields imply about 2x (Bellman) and 3x (metric NTFields) the true clearance speed. The extra room raises that true speed by at most 20%, much less than these errors.
- One training seed per method and scene: the drift of Bellman's compartment rate is within what training-seed and query-draw variation could produce; separating them needs repeated training seeds.

## Scenes

||cabinet (x1.0)|cabinet_plus10 (x1.1)|cabinet_plus20 (x1.2)|
|---|---:|---:|---:|
|Compartment opening width (m)|0.36|0.396|0.432|
|Compartment heights bottom / middle / top (m)|0.225 / 0.21 / 0.205|0.2475 / 0.231 / 0.2255|0.27 / 0.252 / 0.246|
|Cupboard outer depth x width x height (m)|0.32 x 0.4 x 0.7|0.32 x 0.436 x 0.764|0.32 x 0.472 x 0.828|
|Sampling box y (m)|0.03 to 0.63|0.003 to 0.657|-0.024 to 0.684|
|Sampling box z (m)|-0.035 to 0.665|-0.035 to 0.729|-0.035 to 0.793|
|Test-suite compartment pools bottom / middle / top|5,539 / 10,536 / 16,189|10,438 / 19,131 / 27,650|17,871 / 33,280 / 44,117|
|Candidates certified (direct line or RRT-Connect)|759/960 (79.1%)|797/960 (83.0%)|802/960 (83.5%)|
|Frozen suites|`experiments/cabinet/tests/`|`experiments/cabinet/tests_plus10/`|`experiments/cabinet/tests_plus20/`|

## Success rate, all 600 queries

|Method and planner|cabinet (x1.0)|cabinet_plus10 (x1.1)|cabinet_plus20 (x1.2)|
|---|---:|---:|---:|
|Bellman, RK4 (native)|229/600 (38.2%)|224/600 (37.3%)|223/600 (37.2%)|
|Bellman, release MPPI|688/1800 (38.2%)|682/1800 (37.9%)|646/1800 (35.9%)|
|Metric NTFields, RK4 (native)|182/600 (30.3%)|183/600 (30.5%)|194/600 (32.3%)|
|Metric NTFields, release MPPI|595/1800 (33.1%)|600/1800 (33.3%)|615/1800 (34.2%)|
|Metric NTFields, ntrl-demo MPPI|509/1800 (28.3%)|513/1800 (28.5%)|549/1800 (30.5%)|
|Straight joint-space line (reference)|202/600 (33.7%)|186/600 (31.0%)|208/600 (34.7%)|

Compartment suites (free_to_compartment, compartment_to_free, compartment_to_compartment, same_compartment; 400 queries) and open-space suites (free_to_free, anywhere; 200 queries):

|Method and planner|cabinet compartment|cabinet open|cabinet_plus10 compartment|cabinet_plus10 open|cabinet_plus20 compartment|cabinet_plus20 open|
|---|---:|---:|---:|---:|---:|---:|
|Bellman, RK4 (native)|13.0%|88.5%|12.5%|87.0%|11.5%|88.5%|
|Bellman, release MPPI|13.0%|88.7%|12.1%|89.5%|10.3%|87.0%|
|Metric NTFields, RK4 (native)|7.5%|76.0%|6.8%|78.0%|9.5%|78.0%|
|Metric NTFields, release MPPI|9.6%|80.0%|9.8%|80.3%|10.8%|80.8%|
|Metric NTFields, ntrl-demo MPPI|5.2%|74.3%|5.4%|74.7%|7.2%|77.2%|
|Straight joint-space line|11.5%|78.0%|10.2%|72.5%|12.0%|80.0%|

## Release MPPI per suite (same planner for both methods)

|Suite|Bellman cabinet|Bellman plus10|Bellman plus20|Metric cabinet|Metric plus10|Metric plus20|
|---|---:|---:|---:|---:|---:|---:|
|free_to_free|89.3%|88.3%|91.0%|84.7%|81.3%|83.7%|
|free_to_compartment|22.0%|14.3%|13.3%|16.3%|15.0%|13.3%|
|compartment_to_free|21.0%|16.3%|15.7%|9.7%|11.3%|14.7%|
|compartment_to_compartment|1.0%|0.0%|0.0%|1.0%|0.0%|0.0%|
|same_compartment|8.0%|17.7%|12.3%|11.3%|13.0%|15.3%|
|anywhere|88.0%|90.7%|83.0%|75.3%|79.3%|78.0%|

## RK4 per suite

|Suite|Bellman cabinet|Bellman plus10|Bellman plus20|Metric cabinet|Metric plus10|Metric plus20|
|---|---:|---:|---:|---:|---:|---:|
|free_to_free|91.0%|83.0%|91.0%|81.0%|78.0%|83.0%|
|free_to_compartment|22.0%|15.0%|15.0%|10.0%|9.0%|12.0%|
|compartment_to_free|22.0%|16.0%|19.0%|10.0%|7.0%|17.0%|
|compartment_to_compartment|3.0%|0.0%|3.0%|1.0%|0.0%|0.0%|
|same_compartment|5.0%|19.0%|9.0%|9.0%|11.0%|9.0%|
|anywhere|86.0%|91.0%|86.0%|71.0%|78.0%|73.0%|

## ntrl-demo MPPI (metric NTFields' own planner) and the straight line per suite

|Suite|ntrl MPPI cabinet|ntrl MPPI plus10|ntrl MPPI plus20|Line cabinet|Line plus10|Line plus20|
|---|---:|---:|---:|---:|---:|---:|
|free_to_free|78.3%|75.0%|79.3%|82.0%|72.0%|83.0%|
|free_to_compartment|8.0%|9.3%|10.0%|17.0%|10.0%|15.0%|
|compartment_to_free|9.7%|6.3%|13.7%|11.0%|13.0%|12.0%|
|compartment_to_compartment|0.3%|0.0%|0.0%|1.0%|0.0%|0.0%|
|same_compartment|3.0%|6.0%|5.0%|17.0%|18.0%|21.0%|
|anywhere|70.3%|74.3%|75.0%|74.0%|73.0%|77.0%|

## Same planner, same queries (per scene)

|Scene|Planner|Both solve|Bellman only|Metric only|Neither|
|---|---|---:|---:|---:|---:|
|cabinet|RK4 (600)|164|65|18|353|
|cabinet|Release MPPI (600 x 3 seeds)|544|144|51|1061|
|cabinet_plus10|RK4 (600)|163|61|20|356|
|cabinet_plus10|Release MPPI (600 x 3 seeds)|546|136|54|1064|
|cabinet_plus20|RK4 (600)|173|50|21|356|
|cabinet_plus20|Release MPPI (600 x 3 seeds)|558|88|57|1097|

## Diagnostics (`diagnose.py`, read-only)

First scorer violation on paths rejected for collision (seed-0 MPPI and RK4 paths):

|Scene|Method and planner|Collision paths|Cupboard panel|Table|Self|
|---|---|---:|---:|---:|---:|
|cabinet|bellman/rk4|367|354|12|1|
|cabinet|bellman/mppi_seed0|370|352|14|4|
|cabinet|metric/rk4|410|366|30|14|
|cabinet|metric/mppi_seed0|403|362|28|13|
|cabinet|metric/ntrl_mppi_seed0|427|380|33|14|
|cabinet_plus10|bellman/rk4|376|354|22|0|
|cabinet_plus10|bellman/mppi_seed0|370|351|18|1|
|cabinet_plus10|metric/rk4|412|374|22|16|
|cabinet_plus10|metric/mppi_seed0|402|365|22|15|
|cabinet_plus10|metric/ntrl_mppi_seed0|424|379|28|17|
|cabinet_plus20|bellman/rk4|375|367|8|0|
|cabinet_plus20|bellman/mppi_seed0|384|374|7|3|
|cabinet_plus20|metric/rk4|402|365|26|11|
|cabinet_plus20|metric/mppi_seed0|395|366|20|9|
|cabinet_plus20|metric/ntrl_mppi_seed0|414|379|27|8|

Implied speed 1/|grad_q T(q, goal)| along the certified witness paths, mean over the points inside the cupboard, against the true clearance speed there (the four compartment suites pooled; Eikonal: equal for an exact travel time):

|Scene|True speed in cupboard|Bellman implied|Metric implied|Bellman log-ratio RMSE|Metric log-ratio RMSE|
|---|---:|---:|---:|---:|---:|
|cabinet|0.27|0.57|0.83|1.09|1.46|
|cabinet_plus10|0.29|0.61|0.84|1.11|1.42|
|cabinet_plus20|0.32|0.59|0.88|1.01|1.37|

How hard the frozen compartment queries are (`diagnose.py` section 5; the four compartment suites pooled):

|Measure|cabinet|cabinet_plus10|cabinet_plus20|
|---|---:|---:|---:|
|True clearance speed at the compartment endpoints (mean)|0.332|0.369|0.405|
|Compartment endpoint depth past the opening (m, mean)|0.086|0.093|0.098|
|Lowest clearance speed along the certified witness path (mean)|0.122|0.115|0.126|
|Witness joint-space length (rad, median per suite)|5.5 / 5.8 / 11.4 / 7.6|5.8 / 5.8 / 11.0 / 4.5|5.4 / 5.5 / 9.5 / 4.1|
|Queries certified by RRT-Connect, not the straight line|354/400|359/400|352/400|

Speeds are the label `min(clip(d_obstacle, 0.005, 0.05), clip(0.5 d_self, 0.005, 0.05)) / 0.05`: 0.1 is 5 mm clearance or less, 1 is 5 cm or more.

Training states with the grasp point in the cupboard (outer box) / in a compartment interior:

|States|cabinet|cabinet_plus10|cabinet_plus20|
|---|---:|---:|---:|
|bellman_original_endpoints|1.02% / 0.33%|1.24% / 0.48%|1.50% / 0.66%|
|bellman_coverage_endpoints|33.33% / 14.46%|33.26% / 16.62%|33.29% / 18.23%|
|bellman_bank_sources|4.65% / 2.65%|4.81% / 3.03%|4.81% / 3.20%|
|metric_x0|0.83% / 0.26%|1.06% / 0.39%|1.32% / 0.55%|
|metric_x1|0.86% / 0.44%|1.05% / 0.57%|1.25% / 0.70%|

## Training inputs and runs

|Scene|Bellman run|Metric NTFields run|FK pool in cupboard|Speed MAE original / augmentation|Bank anchors|Metric x0 in cupboard|Metric final loss|
|---|---|---|---:|---:|---:|---:|---:|
|cabinet|`registered_20261019T034213Z-f9c501f9`|`registered_20261019T034237Z-30d4d872`|1.00%|0.0358 / 0.0510|399,581|0.83%|0.00277|
|cabinet_plus10|`registered_20261019T171925Z-c5ebfc5b`|`registered_20261019T172013Z-4aa0e26a`|1.23%|0.0359 / 0.0483|399,616|1.06%|0.00256|
|cabinet_plus20|`registered_20261019T171949Z-c41ae437`|`registered_20261019T172037Z-75a8efe5`|1.50%|0.0352 / 0.0476|399,700|1.32%|0.00255|

Separate evaluation runs (scene cabinet only; its pipeline evaluation stage had failed on an inverted assertion): `registered_20261019T041026Z-39d9dafe`, `registered_20261019T041045Z-9baa751f`.

Wall time (seconds):

|Scene|Bellman: sample / prepare / training / evaluate|Metric: data / training / evaluate|
|---|---|---|
|cabinet|206 / 628 / 598 / 473 (total 1905)|35 / 1425 / 632 (total 2092)|
|cabinet_plus10|73 / 534 / 681 / 243 (total 1531)|23 / 1313 / 540 (total 1876)|
|cabinet_plus20|51 / 539 / 691 / 241 (total 1521)|25 / 1326 / 562 (total 1913)|

## Caveats

- Every scene has its own certified suites (endpoints valid in that scene, regions from its own compartments), so the scenes share no queries; changes across scenes mix the easier geometry with different queries. The straight-line row is the per-scene reference.
- Resolution: drawing 600 queries alone gives a success rate near 35% a standard error of about 1.9 points (about 1.8 points for the 400 compartment queries near 15%); MPPI repeats are correlated per query and do not shrink it. Training-seed variation comes on top and is unmeasured.
- One training seed per method and scene; planning seeds 0/1/2 are not training seeds. No significance claim.
- Per-scene details: `evidence/cabinet/COMPARISON.md`, `evidence/cabinet/plus10/COMPARISON.md`, `evidence/cabinet/plus20/COMPARISON.md`.
