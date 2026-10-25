# Wall25: retraining speed, offline bank and TravelCost

Official run: `registered_20260930T013432Z-7905f45e`. Round 5 is fixed; the checkpoint was not selected by evaluation.

|Method|No-wall training → wall evaluation|Wall training → wall evaluation|Valid-endpoint subset: no-wall training|Valid-endpoint subset: wall training|
|---|---:|---:|---:|---:|
|MPPI|1494/1800 (83.00%)|1475/1800 (81.94%)|1494/1716 (87.06%)|1475/1716 (85.96%)|
|RK4|508/600 (84.67%)|494/600 (82.33%)|508/572 (88.81%)|494/572 (86.36%)|

MPPI for the three planning seeds: [492, 491, 492]. 600 distinct queries, of which 28 have a start or goal blocked by the wall and 572 have valid endpoints; the three planning seeds are not three training seeds.

|Task|Direct transfer /300|Wall retraining /300|Difference|
|---|---:|---:|---:|
|cutting|282|273|-9|
|hammering|246|249|+3|
|mixed|194|185|-9|
|screwdriver|239|240|+1|
|stirring|260|261|+1|
|wiping|273|267|-6|

## Paired changes

- MPPI: 18 rescued, 37 new failures. Of the 31 executions that succeeded without the wall, failed with it and have valid endpoints, retraining rescued 6.
- RK4: 16 rescued, 30 new failures. Of the 14 executions that succeeded without the wall, failed with it and have valid endpoints, retraining rescued 5.

## Training and timing

- speed: 20000 steps, fresh initialization; an independent training pose pool was relabeled, and no evaluation queries were read.
- Offline bank: 399674 source poses × 32 candidates, generated in 276.97 seconds.
- TravelCost: 338177 parameters, losses 6/2/2, 5 rounds × 2 epochs, 15620 steps.
- Stage timing: {"prepare": 359.8970802251715, "training": 256.31706811115146, "evaluate": 161.55906646791846}; total 777.78 seconds.

Training and evaluation use a fixed wall; wall information enters supervision through speed and bank, and the model still takes only q,g as input. This is not generalization to arbitrary new walls. Valid endpoints do not prove that a path exists. Single training seed and a reused development query set; no significance or independent-generalization claims. The no-wall baseline was not replaced.
