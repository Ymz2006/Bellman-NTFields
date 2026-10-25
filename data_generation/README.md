# Training data generation (no evaluation inputs)

`source/sample_fk_ur5e.py`: original uniform normalized joint-limit sampling, FK,
workspace focus, physical collision checks and task-tilt filtering. Fixed seed42,
50000proposals per chunk; collect2000000cup and2000000pour configurations.
`source/pkl_to_training_data_multi_ur5e.py`: original seed42pair construction,
1000000pairs,80%intra/20%cross, same region filters and normalization.
`generate.py`: reconstruct tilt20/transition10physical labels using the original
signed-distance conversion; then run `prepare_coverage.py`.

Coverage preserves the original seed20260912recipe: perturb original endpoints
with sigma0.02/0.08/0.20/0.50rad;10%uniform joint-limit draws; reject joint-bound
violations; fill15region×active-tilt strata with133334training and13334validation
configurations each. Angle edges0/10/18/22/35/181degrees; in overlap, use min task
tilt. Focus x0.15–1m,y±0.7m,z−0.1–0.8m is sampling focus, not a new planning
constraint. Train augmentation has1000005pairs.

All stages record commands, RNGseeds, counts, and array hashes in registered
outputs. No evaluation endpoint, path or success label is read. Original pose
pools and their labels are newly generated: `training_inputs/` is not required.
Preparation/training preserve the already verified speed/bank/TravelCost recipes.
