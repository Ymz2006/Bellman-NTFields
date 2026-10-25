# Full no-wall preparation reconstruction: passed

Run `registered_20261002T194107Z-f3a425b2`. Validated in an isolated standalone copy and copied into this repository.
Pinned Docker image and original GPU environment, one training seed87358.

- Speed:20000updates; all58state tensors exactly equal to the frozen speed.
- Bank:399725sources×32actions; all11original arrays numerically identical,
  including dtype and shape. Original exact_safe.npy equals safe.npy in this
  protocol; the rebuilt bank uses safe.npy for this historical alias.
- TravelCost:5rounds,15620updates; checkpoint SHA256 matches original retraining:
  `6a4f28557b1ec655a0eb813f00774f6ed02586ad93410a282efe705ee3294cab`.
- MPPI532/533/532,1597/1800(88.72%); RK4548/600(91.33%).
  Every success flag matches the prior standalone retraining.
- Full physical labels regenerated for1000000original and1000005augmentation
  pose pairs; maximum absolute error1.15633011e-05,
  threshold2e-5. Labels agree within tolerance, not bitwise. Exact historical
  speed training uses original dataset labels. Pose pools remain source data.

|Stage|Seconds|
|---|---:|
|Speed/label audit/seed states/bank preparation|380.95|
|TravelCost training|272.50|
|Complete evaluation|177.83|
|Pipeline total|833.37|

Includes all failures and three MPPI planning seeds. Evaluation loading, warmup,
scoring and serialization are included in pipeline time; this is not inference
latency. Shared GPU load affects timing. Cross-GPU equality is unverified.

Files: rebuild_validation.json, physical_label_audit.json,
end_to_end_validation.json, status.json and registered_spec.json.
Frozen assets and baseline pointers were not replaced. Source input package:
`dist/no-wall-training-inputs.tar.gz` and its SHA256 file.

Installed repository smoke `20261002T195541Z-80fc1bed` succeeded under the
shared default registry. All113frozen asset files were verified after installation.
