# Speed-network test

Compares a run's trained speed network (`prepared/speed.pt`) with ground-truth speed on sampled
configurations. Read-only: no training and no planning.

```bash
BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py speed-test --run registered_<id> --gpu 2 [--smoke]
```

Ground truth is `PhysicalSpeed` (`experiments/from_source/physical.py`), the obstacle distance,
self-collision distance and cup/pour tilt that produced the training labels, in the run's own scene.
Both sides give the scalar speed (two task speeds blended by the end effector's y) and the two task
speeds separately. Speeds lie in [0.1, 1]; 0.1 is the floor.

Sample sets (20,000 each, seed 20261018):

- `uniform`: uniform over the training domain |q_norm| <= 0.5. Mostly in collision or tilted, so mostly at the floor.
- `uniform_valid`: the same draw kept only where the release gate accepts it (collision-free, tilt within limits).
- `heldout_original`, `heldout_augmentation`: the run's validation endpoints (pair index modulo 100 == 0), never fitted.

Checks, which fail the registered run:

- every speed lies inside [0.1, 1];
- the checkpoint's training contract names this scene's sha256;
- the held-out task-speed MAE and p95 reproduce the run's `prepared/speed_audit.json`.

Outputs in `speed_test/`: `speed_test.json` (all metrics), `REPORT.md`, and `samples.npz` with every
sampled configuration and its true and predicted speeds.
