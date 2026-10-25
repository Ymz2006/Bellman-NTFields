# Wall25: direct transfer of the frozen no-wall checkpoint

Run: `registered_20260930T003519Z-f50d55a8`. No retraining and no planner parameter changes; only a 25 cm wall is added.

|Method|No wall, all 600 queries|Wall, all 600 queries|No wall, valid-endpoint subset|Wall, valid-endpoint subset|
|---|---:|---:|---:|---:|
|MPPI|1597/1800 (88.72%)|1494/1800 (83.00%)|1525/1716 (88.87%)|1494/1716 (87.06%)|
|RK4|549/600 (91.50%)|508/600 (84.67%)|522/572 (91.26%)|508/572 (88.81%)|

MPPI successes for the three planning seeds: [498, 498, 498]. There are 600 distinct queries; 572 have valid endpoints, and the added wall invalidates the start or goal of 28 queries.

|Task|MPPI no wall /300|MPPI wall /300|Valid-endpoint queries /100|
|---|---:|---:|---:|
|cutting|291|282|98|
|hammering|276|246|94|
|mixed|206|194|93|
|screwdriver|267|239|95|
|stirring|278|260|94|
|wiping|279|273|98|

## Saved trajectories and changes in failures

- MPPI: saved trajectories identical point for point in 1800/1800; 103 new failures, 0 rescued. Of the new failures, 72 have invalid endpoints and 31 still have valid endpoints.
- RK4: saved trajectories identical point for point in 600/600; 41 new failures, 0 rescued. Of the new failures, 27 have invalid endpoints and 14 still have valid endpoints.

T(q,g) receives no obstacle information, and the default MPPI does not enable geometric safety re-ranking. This result must not be read as a success rate after training with the wall, nor as evidence of adaptation to new obstacles. The valid-endpoint subset has no full path-feasibility certificate. The frozen no-wall baseline remains at 88.72% and was not replaced.

The scene keeps the same normalization as the original 600 queries, a ±0.10 m transition zone, the 20-degree OR constraint, 0.005/0.002 m collision margins and 0.001 rad dense scoring; this is not the historical Suite-safe180 protocol.
