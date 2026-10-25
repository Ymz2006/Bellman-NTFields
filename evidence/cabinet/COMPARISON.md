# Cabinet task: Bellman NTFields vs metric NTFields

Training runs: Bellman `registered_20261019T034213Z-f9c501f9`, metric NTFields `registered_20261019T034237Z-30d4d872`; evaluation runs `registered_20261019T041026Z-39d9dafe` and `registered_20261019T041045Z-9baa751f`. Both scored on the same frozen suites of this scene (`experiments/cabinet/tests/`, from `registered_20261019T033428Z-1342d9d2`): 6 suites x 100 certified queries, release scorer in the cabinet scene with the tilt gate off. MPPI rows count 3 planning seeds per query.

## Success rate, all 600 queries

|Method and planner|Success|
|---|---:|
|Bellman, RK4 (native)|229/600 (38.2%)|
|Bellman, release MPPI|688/1800 (38.2%)|
|Metric NTFields, RK4|182/600 (30.3%)|
|Metric NTFields, release MPPI|595/1800 (33.1%)|
|Metric NTFields, ntrl-demo MPPI (native)|509/1800 (28.3%)|

## Per suite

|Suite|Bellman, RK4 (native)|Bellman, release MPPI|Metric NTFields, RK4|Metric NTFields, release MPPI|Metric NTFields, ntrl-demo MPPI (native)|
|---|---:|---:|---:|---:|---:|
|free_to_free|91/100 (91.0%)|268/300 (89.3%)|81/100 (81.0%)|254/300 (84.7%)|235/300 (78.3%)|
|free_to_compartment|22/100 (22.0%)|66/300 (22.0%)|10/100 (10.0%)|49/300 (16.3%)|24/300 (8.0%)|
|compartment_to_free|22/100 (22.0%)|63/300 (21.0%)|10/100 (10.0%)|29/300 (9.7%)|29/300 (9.7%)|
|compartment_to_compartment|3/100 (3.0%)|3/300 (1.0%)|1/100 (1.0%)|3/300 (1.0%)|1/300 (0.3%)|
|same_compartment|5/100 (5.0%)|24/300 (8.0%)|9/100 (9.0%)|34/300 (11.3%)|9/300 (3.0%)|
|anywhere|86/100 (86.0%)|264/300 (88.0%)|71/100 (71.0%)|226/300 (75.3%)|211/300 (70.3%)|

## Same planner, same queries

|Planner|Both solve|Bellman only|Metric only|Neither|
|---|---:|---:|---:|---:|
|RK4 (deterministic, 600)|164|65|18|353|
|Release MPPI (600 x 3 seeds)|544|144|51|1061|

## Failure reasons (scorer)

|Method and planner|collision|endpoint|joint_limits|
|---|---:|---:|---:|
|Bellman, RK4 (native)|367|6|0|
|Bellman, release MPPI|1112|0|0|
|Metric NTFields, RK4|410|26|0|
|Metric NTFields, release MPPI|1205|0|0|
|Metric NTFields, ntrl-demo MPPI (native)|1275|0|62|

`endpoint`: the path does not end at the goal (planner did not converge and its last segment was not appended or failed).

## Training inputs

- Bellman: FK pool 2,000,000 configurations in the sampling box (1.00% with the grasp point in the cupboard), 1,000,000 original + 1,000,005 coverage pairs; speed network held-out MAE original 0.0358, augmentation 0.0510; bank 399,581 anchors.
- Metric NTFields: 500,000 pairs (0.83% of x0 in the cupboard), 5000 epochs, final-epoch loss 0.00277 (ntrl-demo loss_n).

## Wall time (seconds; RTX 3090, GPU of each registered run in brackets)

- bellman [device=2]: sample 206, prepare 628, training 598, evaluate 473; total 1905
- metric [device=2]: data 35, training 1425, evaluate 632; total 2092

## Diagnostics (`diagnose.py`, read-only; `evidence/cabinet/diagnostics.json`)

Straight joint-space line (the suites' own certificates): 202/600 (33.7%); per suite free_to_free 82, free_to_compartment 17, compartment_to_free 11, compartment_to_compartment 1, same_compartment 17, anywhere 74.

First scorer violation on every path rejected for collision (0.001 rad densification):

|Method and planner|Collision paths|Cupboard panel|Table|Self|Grasp point at first contact: free box / compartment / other cupboard / outside box|
|---|---:|---:|---:|---:|---|
|bellman/rk4|367|354|12|1|117 / 158 / 74 / 18|
|bellman/mppi_seed0|370|352|14|4|107 / 172 / 75 / 16|
|metric/rk4|410|366|30|14|107 / 206 / 69 / 28|
|metric/mppi_seed0|403|362|28|13|104 / 223 / 57 / 19|
|metric/ntrl_mppi_seed0|427|380|33|14|114 / 221 / 66 / 26|

Training states with the grasp point in the cupboard (outer box) and in a compartment interior:

|States|Cupboard|Compartments|
|---|---:|---:|
|bellman_original_endpoints|1.02%|0.33%|
|bellman_coverage_endpoints|33.33%|14.46%|
|bellman_bank_sources|4.65%|2.65%|
|metric_x0|0.83%|0.26%|
|metric_x1|0.86%|0.44%|

Implied speed 1/|grad_q T(q, goal)| along the certified witness paths (points more than 0.02 from the goal), against the true clearance speed:

|Suite|True, in cupboard|Bellman|Metric|True, outside|Bellman|Metric|Log-ratio RMSE in cupboard: Bellman / metric|
|---|---:|---:|---:|---:|---:|---:|---|
|free_to_free|0.30|0.29|0.69|0.68|0.73|0.76|0.15 / 0.85|
|free_to_compartment|0.29|0.68|0.74|0.62|0.65|0.72|1.21 / 1.34|
|compartment_to_free|0.26|0.49|0.89|0.69|0.72|0.85|0.96 / 1.55|
|compartment_to_compartment|0.25|0.59|0.79|0.66|0.62|0.81|1.13 / 1.41|
|same_compartment|0.27|0.56|0.87|0.62|0.66|0.76|1.08 / 1.49|
|anywhere|0.28|0.57|0.94|0.62|0.71|0.68|1.00 / 1.53|

One training seed per method; planning seeds are not training seeds. No significance claim.
