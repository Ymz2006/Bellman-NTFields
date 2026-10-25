# Transition-wall scenes: from-source retraining

All three runs used the same from-source pipeline, seeds and machine (RTX 3090, derived sm_86 image), so they differ only in the scene.

## Scenes

|Scene|Wall x × y × z (m)|Centre (m)|Valid endpoints|Blocked: cupping / pouring / cross|
|---|---|---|---:|---|
|wall40|0.40 × 0.05 × 0.40|(0.57, 0.0, 0.165)|528/600|31 / 9 / 32|
|wall80|0.40 × 0.05 × 0.80|(0.57, 0.0, 0.365)|477/600|44 / 20 / 59|

## Success rates (all 600 queries)

|Run|MPPI seeds 0/1/2|MPPI|RK4|
|---|---|---:|---:|
|no-wall (registered_20261009T042727Z-c1d4e622)|[533, 531, 533]|1597/1800 (88.72%)|535/600 (89.17%)|
|wall40 (registered_20261010T205523Z-0a390379)|[425, 427, 425]|1277/1800 (70.94%)|430/600 (71.67%)|
|wall80 (registered_20261010T205528Z-978ed9d1)|[298, 298, 299]|895/1800 (49.72%)|297/600 (49.50%)|

## Same queries with and without the wall

Restricted to the queries whose endpoints are valid in each wall scene, so impossible queries do not count against the wall runs.

|Subset|Planner|No wall|With wall|
|---|---|---:|---:|
|wall40 valid (528 queries)|MPPI|1422/1584 (89.77%)|1277/1584 (80.62%)|
|wall40 valid (528 queries)|RK4|478/528 (90.53%)|430/528 (81.44%)|
|wall80 valid (477 queries)|MPPI|1296/1431 (90.57%)|895/1431 (62.54%)|
|wall80 valid (477 queries)|RK4|433/477 (90.78%)|297/477 (62.26%)|

## MPPI by query type

The walls sit in the transition band, so cross queries (one side to the other) are the ones that must get past them.

|Query type|no-wall|wall40|wall80|
|---|---:|---:|---:|
|cupping|523/540 (96.85%)|435/540 (80.56%)|400/540 (74.07%)|
|pouring|533/540 (98.70%)|513/540 (95.00%)|477/540 (88.33%)|
|cross|541/720 (75.14%)|329/720 (45.69%)|18/720 (2.50%)|

## MPPI by task (/300)

|Task|no-wall|wall40|wall80|
|---|---:|---:|---:|
|cutting|288|244|156|
|hammering|273|216|165|
|mixed|212|151|129|
|screwdriver|270|177|115|
|stirring|276|238|159|
|wiping|278|251|171|

## Failure reasons (a run can have several)

|Run|MPPI|RK4|
|---|---|---|
|no-wall|tilt 167, collision 69|endpoint 34, tilt 27, collision 11|
|wall40|collision 435, tilt 208|collision 131, endpoint 44, tilt 37|
|wall80|collision 877, tilt 337|collision 246, endpoint 113, tilt 59|

## Data, training and timing

|Run|Bank rows|Bank safe-move fraction|Speed log-RMSE (orig / aug)|Stage seconds|Total|
|---|---:|---:|---|---|---:|
|no-wall|399752|0.283|0.125 / 0.107|sample 314, prepare 901, training 977, evaluate 840|50.5 min|
|wall40|399575|0.280|0.147 / 0.133|audit 9, sample 814, prepare 1109, training 779, evaluate 455|52.8 min|
|wall80|398673|0.266|0.156 / 0.134|audit 8, sample 924, prepare 1163, training 795, evaluate 328|53.6 min|

## Caveats

- Modified environment (cuRobo rebuilt for sm_86, host CUDA driver libraries): independent samples, not bitwise replays of the original machine.
- One training seed per scene; the 600 queries are a reused development set. No significance or generalization claims.
- T(q,g) takes only q and g. Each wall reaches it only through the scene-specific training data, speed labels and bank.
- Valid endpoints do not prove a collision-free path exists, especially for cross queries past the 0.80 m wall.
- The frozen no-wall baseline and committed evidence are unchanged.
