# From-source no-wall reproduction: passed

Full run: `registered_20261008T185137Z-054b2203`. Only frozen runtime and pinned GPU environment were supplied.
The test checkout had no `assets/` or `training_inputs/` directory; historical
references were read only by the post-run comparison, not by sampling/training.

- Uniform task-filtered FK pose sampling and original pair/coverage construction
  newly produced all4original training arrays with historical SHA256 equality.
- All58speed-state tensors and all11bank arrays match historical data exactly.
  Historical exact_safe.npy is an alias of safe.npy under this fixed protocol.
- TravelCost fifth-round checkpoint SHA256:
  `6a4f28557b1ec655a0eb813f00774f6ed02586ad93410a282efe705ee3294cab`.
- MPPI532/533/532,1597/1800(88.72%); RK4548/600(91.33%).
- All1800MPPI and600RK4successflags match prior standalone retraining.
- Clean-clone runtime-only smoke `20261008T185816Z-d381e5e2` succeeded;
  generated4trainingarrays and finalsmokecheckpoint match the first smoke.
- Runtime-only clean clone loads real UR5e Viser meshes and GUI controls.
- Runtime archive unpack/checksum and refusal to overwrite changed files passed.
- Exported Docker archive was loaded successfully and retained the pinned image ID.
  All5downloadparts passed SHA256 checks.

|Stage|Seconds|
|---|---:|
|Original FK poses, pairs, labels, coverage|239.09|
|Speed, safe seeds, transition bank|349.31|
|TravelCost training|243.10|
|Complete600evaluation|144.05|
|Pipeline total|975.57|

Preparation+training: 13.86minutes;
with evaluation: 16.26minutes. Earlier~10.9minutes began
with pre-existing source pose pools; this run includes their regeneration.
One training seed and the same development benchmark; cross-GPU equality is
unverified. Fixed scene, normalized speed coefficient, no arbitrary new-wall input.

Entry: `python3 run.py from-source`. Required download: bellman600-runtime.tar.gz
plus the pinned Docker parts. Historical assets/source pools are optional.
Full instructions: [COLLEAGUE_QUICKSTART.md](../../COLLEAGUE_QUICKSTART.md).
