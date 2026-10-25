# Obstacle scenes: from-source retraining

Every run used the same from-source pipeline, seeds and machine (RTX 3090, derived sm_86 image), so they differ only in the scene.
Scene files: `experiments/obstacle_scenes/scenes/`.

## Scenes

|Scene|Valid endpoints|Blocked: cupping / pouring / cross|
|---|---:|---|
|platforms|438/600|42 / 46 / 74|
|clutter|497/600|44 / 11 / 48|

|Scene|Box|x (m)|y (m)|z (m)|
|---|---|---|---|---|
|platforms|cup_platform|0.40–0.90|+0.15–+0.55|-0.035–0.115|
|platforms|pour_platform|0.40–0.90|-0.55–-0.15|-0.035–0.115|
|clutter|bottle_left|0.44–0.52|+0.24–+0.32|-0.035–0.265|
|clutter|jar_center|0.57–0.67|-0.05–+0.05|-0.035–0.125|
|clutter|kettle_right|0.61–0.79|-0.41–-0.25|-0.035–0.205|

## Success rates (all 600 queries)

|Run|MPPI seeds 0/1/2|MPPI|RK4|
|---|---|---:|---:|
|no-wall (registered_20261009T042727Z-c1d4e622)|[533, 531, 533]|1597/1800 (88.72%)|535/600 (89.17%)|
|platforms (registered_20261013T194621Z-1978fc0c)|[388, 388, 387]|1163/1800 (64.61%)|396/600 (66.00%)|
|clutter (registered_20261013T210624Z-f04a0fc2)|[412, 410, 410]|1232/1800 (68.44%)|420/600 (70.00%)|

## Scene test suites

Each scene's own frozen, certified queries (`experiments/obstacle_scenes/tests/<scene>/`): every query has a witness path that the release scorer accepts in the scene, so no query is impossible. Direct line: queries whose straight joint-space line already passes.

|Scene|Suite|Pair type|Direct line|RK4|MPPI, 3 seeds|
|---|---|---|---:|---:|---:|
|platforms|general_cupping|cupping|75/100|90/100 (90.00%)|274/300 (91.33%)|
|platforms|general_pouring|pouring|78/100|97/100 (97.00%)|297/300 (99.00%)|
|platforms|general_cross|cross|8/100|49/100 (49.00%)|109/300 (36.33%)|
|platforms|platform_cupping|cupping|59/100|91/100 (91.00%)|275/300 (91.67%)|
|platforms|platform_pouring|pouring|56/100|96/100 (96.00%)|282/300 (94.00%)|
|platforms|platform_cross|cross|1/100|33/100 (33.00%)|64/300 (21.33%)|
|**platforms**|**all**||277/600|**456/600 (76.00%)**|**1301/1800 (72.28%)**|
|clutter|general_cupping|cupping|63/100|86/100 (86.00%)|264/300 (88.00%)|
|clutter|general_pouring|pouring|66/100|90/100 (90.00%)|280/300 (93.33%)|
|clutter|general_cross|cross|13/100|49/100 (49.00%)|124/300 (41.33%)|
|clutter|low_cupping|cupping|48/100|67/100 (67.00%)|201/300 (67.00%)|
|clutter|low_pouring|pouring|83/100|94/100 (94.00%)|288/300 (96.00%)|
|clutter|low_cross|cross|3/100|28/100 (28.00%)|52/300 (17.33%)|
|**clutter**|**all**||276/600|**414/600 (69.00%)**|**1209/1800 (67.17%)**|

For reference, the no-wall run on the frozen certified `mixed` suite (same construction, no obstacles; 30 cupping, 30 pouring, 40 cross queries): cupping MPPI 76/90 (84.44%), RK4 22/30 (73.33%), pouring MPPI 87/90 (96.67%), RK4 29/30 (96.67%), cross MPPI 49/120 (40.83%), RK4 26/40 (65.00%). Certified random cross pairs are much harder than the cross segments of the five task suites, so these totals are not comparable with the frozen 600.

## Same queries with and without the obstacles

Restricted to the queries whose endpoints are valid in each scene, so impossible queries do not count against the scene runs.

|Subset|Planner|No wall|With obstacles|
|---|---|---:|---:|
|platforms valid (438 queries)|MPPI|1204/1314 (91.63%)|1163/1314 (88.51%)|
|platforms valid (438 queries)|RK4|396/438 (90.41%)|396/438 (90.41%)|
|clutter valid (497 queries)|MPPI|1347/1491 (90.34%)|1232/1491 (82.63%)|
|clutter valid (497 queries)|RK4|448/497 (90.14%)|420/497 (84.51%)|

## MPPI by query type, endpoint-valid queries

|Scene|Query type|No wall|With obstacles|
|---|---|---:|---:|
|platforms|cupping|403/414 (97.34%)|396/414 (95.65%)|
|platforms|pouring|398/402 (99.00%)|399/402 (99.25%)|
|platforms|cross|403/498 (80.92%)|368/498 (73.90%)|
|clutter|cupping|399/408 (97.79%)|374/408 (91.67%)|
|clutter|pouring|500/507 (98.62%)|488/507 (96.25%)|
|clutter|cross|448/576 (77.78%)|370/576 (64.24%)|

## Paired outcomes, endpoint-valid queries

Same query and planning seed with and without the obstacles. Lost: solved without them, failed with them.

|Scene|Planner|Both succeed|Lost|Gained|Both fail|Failure reasons with obstacles|Failure reasons without|
|---|---|---:|---:|---:|---:|---|---|
|platforms|MPPI|1155|49|8|102|tilt 114, collision 55|tilt 98, collision 27|
|platforms|RK4|383|13|13|29|endpoint 26, tilt 17, collision 10|endpoint 20, tilt 20, collision 4|
|clutter|MPPI|1225|122|7|137|collision 154, tilt 147|tilt 123, collision 42|
|clutter|RK4|413|35|7|42|collision 38, tilt 29, endpoint 22|endpoint 24, tilt 23, collision 6|

## MPPI by task (/300, all queries)

|Task|no-wall|platforms|clutter|
|---|---:|---:|---:|
|cutting|288|222|227|
|hammering|273|177|181|
|mixed|212|156|160|
|screwdriver|270|185|193|
|stirring|276|186|213|
|wiping|278|237|258|

## Failure reasons (all queries, including impossible ones; a run can have several)

|Run|MPPI|RK4|
|---|---|---|
|no-wall|tilt 167, collision 69|endpoint 34, tilt 27, collision 11|
|platforms|collision 541, tilt 189|collision 165, endpoint 41, tilt 26|
|clutter|collision 463, tilt 194|collision 139, tilt 34, endpoint 30|

## Data, training and timing

|Run|Bank rows|Bank safe-move fraction|Speed log-RMSE (orig / aug)|Stage seconds|Total|
|---|---:|---:|---|---|---:|
|no-wall|399752|0.283|0.125 / 0.107|sample 314, prepare 901, training 977, evaluate 840|50.5 min|
|platforms|399661|0.282|0.129 / 0.117|audit 7, sample 411, prepare 554, training 493, evaluate 265|28.8 min|
|clutter|399585|0.281|0.146 / 0.138|audit 6, sample 637, prepare 574, training 528, evaluate 272, scene_tests 292|38.5 min|

## Caveats

- Modified environment (cuRobo rebuilt for sm_86, host CUDA driver libraries): independent samples, not bitwise replays of the original machine.
- One training seed per scene; the 600 queries are a reused development set. No significance or generalization claims.
- T(q,g) takes only q and g. The obstacles reach it only through the scene-specific training data, speed labels and bank.
- In the frozen 600, valid endpoints do not prove a collision-free path exists; every scene-suite query has a witness path.
- The scene suites keep only queries the certifying planner solved, so they lean towards queries it finds easier.
- The 600 queries were drawn for the empty table, not for these obstacles: they test planning around the obstacles, not starts and goals placed on or inside them.
- The frozen no-wall baseline and committed evidence are unchanged.
