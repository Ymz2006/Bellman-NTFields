# Transition-wall scenes, trained from source

The user requested two new scenes, each adding one wall in the middle of the red transition band
(x 0.29–0.85 m, y −0.10–0.10 m, centre (0.57, 0.00)), and a complete retrain for each.
The walls differ only in height:

|Scene|Dims x × y × z (m)|Centre (m)|Top (m)|
|---|---|---|---:|
|`scenes/wall40.yml`|0.40 × 0.05 × 0.40|(0.57, 0.00, 0.165)|0.365|
|`scenes/wall80.yml`|0.40 × 0.05 × 0.80|(0.57, 0.00, 0.365)|0.765|

Each wall stands on the table top (z = −0.035 m) and separates the cupping side from the pouring side
along y. Table, zones, task axes, sampling bounds, normalization, the 600 queries, the 20° OR rule,
0.005/0.002 m margins and 0.001 rad dense scoring are unchanged.

## Pipeline

`pipeline.py` runs the from-source recipe with the scene passed to every stage:

1. **audit**: endpoint validity of the fixed 600 queries in the scene (runs first; seconds).
2. **sample**: `data_generation/generate.py --scene`: new FK pose pools, pairs, physical labels and
   coverage augmentation, collision-filtered and labelled against the wall (seeds 42/42/20260912).
3. **prepare**: `experiments/from_source/prepare.py --scene`: speed network (20000 steps), label audit,
   offline bank with the wall in the 0.02 rad veto (seed 87344).
4. **training**: `experiments/from_source/training.py`: unchanged 338,177-parameter TravelCost,
   losses 6/2/2, seed 87358, 5 rounds × 2 epochs; updates = 10 × ceil(bank rows / 256).
5. **evaluate**: `evaluate.py` (copy of `wall25_train/evaluate.py` with scene and audit arguments):
   round 5 only, original MPPI settings, planning seeds 0/1/2, native RK4, strict scorer in the scene.

The `--scene` arguments added to `generate.py`, `prepare_coverage.py` and `from_source/prepare.py`
default to the frozen no-wall scene, so existing pipelines are unchanged.

```bash
python3 run.py transition-wall --scene wall40 [--gpu N] [--smoke]
python3 run.py transition-wall --scene wall80 [--gpu N] [--smoke]
python3 experiments/transition_wall/report.py . <wall40 run> <wall80 run> --no-wall <from-source run>
```

`--gpu N` pins the container to one GPU (recorded in the spec) so both scenes can run in parallel.
Outputs go to registered `results/registered_<id>/`; the no-wall baseline is never replaced.
Valid endpoints do not prove a path exists; one training seed supports no significance claims.
