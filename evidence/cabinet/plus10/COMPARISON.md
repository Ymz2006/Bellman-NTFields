# Cabinet task (cabinet_plus10: 10% more clearance): Bellman NTFields vs metric NTFields

Training runs: Bellman `registered_20261019T171925Z-c5ebfc5b`, metric NTFields `registered_20261019T172013Z-4aa0e26a`; scored by their own evaluation stage. Both scored on the same frozen suites of this scene (`experiments/cabinet/tests_plus10/`, from `registered_20261019T171426Z-986a0e88`): 6 suites x 100 certified queries, release scorer in the cabinet scene with the tilt gate off. MPPI rows count 3 planning seeds per query.

## Success rate, all 600 queries

|Method and planner|Success|
|---|---:|
|Bellman, RK4 (native)|224/600 (37.3%)|
|Bellman, release MPPI|682/1800 (37.9%)|
|Metric NTFields, RK4|183/600 (30.5%)|
|Metric NTFields, release MPPI|600/1800 (33.3%)|
|Metric NTFields, ntrl-demo MPPI (native)|513/1800 (28.5%)|

## Per suite

|Suite|Bellman, RK4 (native)|Bellman, release MPPI|Metric NTFields, RK4|Metric NTFields, release MPPI|Metric NTFields, ntrl-demo MPPI (native)|
|---|---:|---:|---:|---:|---:|
|free_to_free|83/100 (83.0%)|265/300 (88.3%)|78/100 (78.0%)|244/300 (81.3%)|225/300 (75.0%)|
|free_to_compartment|15/100 (15.0%)|43/300 (14.3%)|9/100 (9.0%)|45/300 (15.0%)|28/300 (9.3%)|
|compartment_to_free|16/100 (16.0%)|49/300 (16.3%)|7/100 (7.0%)|34/300 (11.3%)|19/300 (6.3%)|
|compartment_to_compartment|0/100 (0.0%)|0/300 (0.0%)|0/100 (0.0%)|0/300 (0.0%)|0/300 (0.0%)|
|same_compartment|19/100 (19.0%)|53/300 (17.7%)|11/100 (11.0%)|39/300 (13.0%)|18/300 (6.0%)|
|anywhere|91/100 (91.0%)|272/300 (90.7%)|78/100 (78.0%)|238/300 (79.3%)|223/300 (74.3%)|

## Same planner, same queries

|Planner|Both solve|Bellman only|Metric only|Neither|
|---|---:|---:|---:|---:|
|RK4 (deterministic, 600)|163|61|20|356|
|Release MPPI (600 x 3 seeds)|546|136|54|1064|

## Failure reasons (scorer)

|Method and planner|collision|endpoint|joint_limits|
|---|---:|---:|---:|
|Bellman, RK4 (native)|376|6|0|
|Bellman, release MPPI|1118|0|0|
|Metric NTFields, RK4|412|19|0|
|Metric NTFields, release MPPI|1200|0|0|
|Metric NTFields, ntrl-demo MPPI (native)|1281|0|66|

`endpoint`: the path does not end at the goal (planner did not converge and its last segment was not appended or failed).

## Training inputs

- Bellman: FK pool 2,000,000 configurations in the sampling box (1.23% with the grasp point in the cupboard), 1,000,000 original + 1,000,005 coverage pairs; speed network held-out MAE original 0.0359, augmentation 0.0483; bank 399,616 anchors.
- Metric NTFields: 500,000 pairs (1.06% of x0 in the cupboard), 5000 epochs, final-epoch loss 0.00256 (ntrl-demo loss_n).

## Wall time (seconds; RTX 3090, GPU of each registered run in brackets)

- bellman [device=0]: sample 73, prepare 534, training 681, evaluate 243; total 1531
- metric [device=2]: data 23, training 1313, evaluate 540; total 1876

## Diagnostics (`diagnose.py`, read-only; `evidence/cabinet/plus10/diagnostics.json`)

Straight joint-space line (the suites' own certificates): 186/600 (31.0%); per suite free_to_free 72, free_to_compartment 10, compartment_to_free 13, compartment_to_compartment 0, same_compartment 18, anywhere 73.

First scorer violation on every path rejected for collision (0.001 rad densification):

|Method and planner|Collision paths|Cupboard panel|Table|Self|Grasp point at first contact: free box / compartment / other cupboard / outside box|
|---|---:|---:|---:|---:|---|
|bellman/rk4|376|354|22|0|111 / 186 / 64 / 15|
|bellman/mppi_seed0|370|351|18|1|104 / 191 / 60 / 15|
|metric/rk4|412|374|22|16|101 / 231 / 56 / 24|
|metric/mppi_seed0|402|365|22|15|105 / 246 / 35 / 16|
|metric/ntrl_mppi_seed0|424|379|28|17|109 / 237 / 51 / 27|

Training states with the grasp point in the cupboard (outer box) and in a compartment interior:

|States|Cupboard|Compartments|
|---|---:|---:|
|bellman_original_endpoints|1.24%|0.48%|
|bellman_coverage_endpoints|33.26%|16.62%|
|bellman_bank_sources|4.81%|3.03%|
|metric_x0|1.06%|0.39%|
|metric_x1|1.05%|0.57%|

Implied speed 1/|grad_q T(q, goal)| along the certified witness paths (points more than 0.02 from the goal), against the true clearance speed:

|Suite|True, in cupboard|Bellman|Metric|True, outside|Bellman|Metric|Log-ratio RMSE in cupboard: Bellman / metric|
|---|---:|---:|---:|---:|---:|---:|---|
|free_to_free|0.68|0.61|0.85|0.69|0.78|0.77|0.42 / 0.50|
|free_to_compartment|0.29|0.82|0.83|0.66|0.65|0.85|1.37 / 1.41|
|compartment_to_free|0.29|0.48|0.86|0.67|0.68|0.85|0.92 / 1.43|
|compartment_to_compartment|0.27|0.61|0.85|0.59|0.59|0.86|1.15 / 1.47|
|same_compartment|0.32|0.62|0.82|0.61|0.60|0.76|1.05 / 1.35|
|anywhere|0.50|0.68|0.96|0.65|0.71|0.83|0.62 / 1.11|

One training seed per method; planning seeds are not training seeds. No significance claim.
