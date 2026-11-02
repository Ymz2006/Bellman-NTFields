# P-NTFields on the cabinet scenes

P-NTFields (Ni & Qureshi, RSS 2023; `baselines/P-NTFields`) trained from scratch on each cabinet scene and scored on
the frozen suites with the release scorer, for comparison with ntrl-demo (`experiments/ntrl_cabinet`).

The method is unchanged: `models/model_res_sigmoid.py`, trained as `train/train_arm.py` does (10,000 epochs,
2,000,000 pairs, margin 0.5/12). It gets the two adaptations ntrl-demo got: start points are sampled in the scene's
sampling box, and clearance is the exact sphere–cuboid distance of the release arm. Four planners are evaluated on
the field: P-NTFields' own gradient planner, ntrl-demo's MPPI, the release MPPI (planning seeds 0/1/2) and the
release native RK4.

```bash
BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python run.py pntfields-cabinet --scene cabinet_w15 --gpu 0   # -> results/registered_<id>/
```

|File|What|
|---|---|
|`pipeline.py`|data (`experiments/ntrl_cabinet/data.py`, P-NTFields recipe) → `train.py` → `evaluate.py`|
|`train.py`|P-NTFields training, unchanged apart from the loader (no obstacle mesh), pickle5 → pickle and seeds|
|`field.py`|the checkpoint as a travel-time field in release coordinates, plus P-NTFields' planner (`test/arm_plan.py`)|
|`evaluate.py`|the four planners on the frozen suites, scored by `score_cross_suite.py` (tilt gate off)|
|`analysis/`|`line_overlap.py` (per-execution overlap with the straight line and between fields), `contact_analysis.py` (first contact and detour per path; output `contacts.json`)|
|`visualizer/`|3D path viewer of Task 1 and all five scenes (`export_cabinet.py`, `build.py` → `viewer.html`)|

## Results (successes; MPPI out of 1800 = 600 queries × 3 seeds, the others out of 600)

|Scene|P-NTFields run|RK4|own planner|ntrl-demo MPPI|release MPPI|ntrl-demo run: RK4 / release / ntrl-demo MPPI|straight line|
|---|---|---:|---:|---:|---:|---|---:|
|cabinet|`20261026T080627Z-53c1dd59`|199|227|635|635|`664b5d50`: 174 / 581 / 596|202|
|cabinet_pi|`20261026T110153Z-44e64765`|230|236|635|678|`379bcde5`: 186 / 622 / 588|173|
|cabinet_w25|`20261026T140052Z-b8746f48`|201|213|626|599|`b25f368e`: 179 / 600 / 517|215|
|cabinet_w20|`20261026T165453Z-9a9b9b4d`|204|220|641|635|`c32709b5`: 201 / 600 / 544|205|
|cabinet_w15|`20261026T194913Z-656fe377`|220|223|633|676|`b00193b9`: 183 / 588 / 532|204|

Over the five scenes, under ntrl-demo's MPPI P-NTFields solves 3170/9000 (35.2%) against ntrl-demo's 2777 (30.9%).
The straight joint-space line solves 999/3000 (33.3%). The lead comes from free space and from moving into or out
of a compartment. Within a compartment ntrl-demo does better (136 vs 69 of 1500), and both are below the straight
line (345). Compartment poses sit about 2 cm from the panels, inside the 4.2 cm margin where the training label is
already slow. As a result the field gives no direction there. One training seed per run.

Full report: https://claude.ai/artifact/GFEX6UonLjv2cfSMfvff2P
