# Cabinet task (cabinet_plus20: 20% more clearance): Bellman NTFields vs metric NTFields

Training runs: Bellman `registered_20261019T171949Z-c41ae437`, metric NTFields `registered_20261019T172037Z-75a8efe5`; scored by their own evaluation stage. Both scored on the same frozen suites of this scene (`experiments/cabinet/tests_plus20/`, from `registered_20261019T171445Z-37602704`): 6 suites x 100 certified queries, release scorer in the cabinet scene with the tilt gate off. MPPI rows count 3 planning seeds per query.

## Success rate, all 600 queries

|Method and planner|Success|
|---|---:|
|Bellman, RK4 (native)|223/600 (37.2%)|
|Bellman, release MPPI|646/1800 (35.9%)|
|Metric NTFields, RK4|194/600 (32.3%)|
|Metric NTFields, release MPPI|615/1800 (34.2%)|
|Metric NTFields, ntrl-demo MPPI (native)|549/1800 (30.5%)|

## Per suite

|Suite|Bellman, RK4 (native)|Bellman, release MPPI|Metric NTFields, RK4|Metric NTFields, release MPPI|Metric NTFields, ntrl-demo MPPI (native)|
|---|---:|---:|---:|---:|---:|
|free_to_free|91/100 (91.0%)|273/300 (91.0%)|83/100 (83.0%)|251/300 (83.7%)|238/300 (79.3%)|
|free_to_compartment|15/100 (15.0%)|40/300 (13.3%)|12/100 (12.0%)|40/300 (13.3%)|30/300 (10.0%)|
|compartment_to_free|19/100 (19.0%)|47/300 (15.7%)|17/100 (17.0%)|44/300 (14.7%)|41/300 (13.7%)|
|compartment_to_compartment|3/100 (3.0%)|0/300 (0.0%)|0/100 (0.0%)|0/300 (0.0%)|0/300 (0.0%)|
|same_compartment|9/100 (9.0%)|37/300 (12.3%)|9/100 (9.0%)|46/300 (15.3%)|15/300 (5.0%)|
|anywhere|86/100 (86.0%)|249/300 (83.0%)|73/100 (73.0%)|234/300 (78.0%)|225/300 (75.0%)|

## Same planner, same queries

|Planner|Both solve|Bellman only|Metric only|Neither|
|---|---:|---:|---:|---:|
|RK4 (deterministic, 600)|173|50|21|356|
|Release MPPI (600 x 3 seeds)|558|88|57|1097|

## Failure reasons (scorer)

|Method and planner|collision|endpoint|joint_limits|
|---|---:|---:|---:|
|Bellman, RK4 (native)|375|5|0|
|Bellman, release MPPI|1154|0|0|
|Metric NTFields, RK4|402|12|0|
|Metric NTFields, release MPPI|1185|0|0|
|Metric NTFields, ntrl-demo MPPI (native)|1239|0|49|

`endpoint`: the path does not end at the goal (planner did not converge and its last segment was not appended or failed).

## Training inputs

- Bellman: FK pool 2,000,000 configurations in the sampling box (1.50% with the grasp point in the cupboard), 1,000,000 original + 1,000,005 coverage pairs; speed network held-out MAE original 0.0352, augmentation 0.0476; bank 399,700 anchors.
- Metric NTFields: 500,000 pairs (1.32% of x0 in the cupboard), 5000 epochs, final-epoch loss 0.00255 (ntrl-demo loss_n).

## Wall time (seconds; RTX 3090, GPU of each registered run in brackets)

- bellman [device=1]: sample 51, prepare 539, training 691, evaluate 241; total 1521
- metric [device=2]: data 25, training 1326, evaluate 562; total 1913

## Diagnostics (`diagnose.py`, read-only; `evidence/cabinet/plus20/diagnostics.json`)

Straight joint-space line (the suites' own certificates): 208/600 (34.7%); per suite free_to_free 83, free_to_compartment 15, compartment_to_free 12, compartment_to_compartment 0, same_compartment 21, anywhere 77.

First scorer violation on every path rejected for collision (0.001 rad densification):

|Method and planner|Collision paths|Cupboard panel|Table|Self|Grasp point at first contact: free box / compartment / other cupboard / outside box|
|---|---:|---:|---:|---:|---|
|bellman/rk4|375|367|8|0|111 / 181 / 63 / 20|
|bellman/mppi_seed0|384|374|7|3|106 / 194 / 59 / 25|
|metric/rk4|402|365|26|11|94 / 230 / 54 / 24|
|metric/mppi_seed0|395|366|20|9|95 / 230 / 50 / 20|
|metric/ntrl_mppi_seed0|414|379|27|8|99 / 238 / 50 / 27|

Training states with the grasp point in the cupboard (outer box) and in a compartment interior:

|States|Cupboard|Compartments|
|---|---:|---:|
|bellman_original_endpoints|1.50%|0.66%|
|bellman_coverage_endpoints|33.29%|18.23%|
|bellman_bank_sources|4.81%|3.20%|
|metric_x0|1.32%|0.55%|
|metric_x1|1.25%|0.70%|

Implied speed 1/|grad_q T(q, goal)| along the certified witness paths (points more than 0.02 from the goal), against the true clearance speed:

|Suite|True, in cupboard|Bellman|Metric|True, outside|Bellman|Metric|Log-ratio RMSE in cupboard: Bellman / metric|
|---|---:|---:|---:|---:|---:|---:|---|
|free_to_free|0.25|0.58|0.88|0.71|0.75|0.85|1.20 / 1.55|
|free_to_compartment|0.30|0.65|0.82|0.67|0.65|0.82|1.10 / 1.36|
|compartment_to_free|0.33|0.48|0.91|0.70|0.71|0.87|0.81 / 1.36|
|compartment_to_compartment|0.31|0.61|0.89|0.59|0.59|0.74|1.07 / 1.41|
|same_compartment|0.34|0.61|0.87|0.56|0.62|0.71|0.99 / 1.32|
|anywhere|0.34|0.65|0.97|0.70|0.73|0.83|1.01 / 1.41|

One training seed per method; planning seeds are not training seeds. No significance claim.
