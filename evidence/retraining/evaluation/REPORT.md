# Clean Bellman600 baseline

Fixed checkpoint; three planning seeds; Bellman + Eikonal + direction.

Native RK4: 548/600.
Matched MPPI successes: [532, 533, 532]; 1597/1800 (88.72%).
Frozen Bellman reference: 532/533/532, 1597/1800 (88.72%). Tolerance: minimum83.50% under cumulative budget; minimum1503/1800.

|Task|Seed|Ablation|Bellman reference|Rescued|Regressed|
|---|---:|---:|---:|---:|---:|
|mixed|0|68|68|0|0|
|screwdriver|0|89|89|0|0|
|cutting|0|97|97|0|0|
|hammering|0|92|92|0|0|
|stirring|0|93|93|0|0|
|wiping|0|93|93|0|0|
|mixed|1|69|69|0|0|
|screwdriver|1|89|89|0|0|
|cutting|1|97|97|0|0|
|hammering|1|92|92|0|0|
|stirring|1|93|93|0|0|
|wiping|1|93|93|0|0|
|mixed|2|68|69|0|1|
|screwdriver|2|89|89|0|0|
|cutting|2|97|97|0|0|
|hammering|2|92|92|0|0|
|stirring|2|93|92|1|0|
|wiping|2|93|93|0|0|
