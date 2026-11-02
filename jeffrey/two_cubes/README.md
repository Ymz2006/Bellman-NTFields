# Two cubes: ntrl-demo baseline (2026-11-01)

User request: "make a simple environment, like 2 cubes that are far apart. Run the ntrl-demo baseline and plan some
pairs, then show me the results."

- Scene `experiments/cabinet/scenes/cabinet_twocubes.yml` (hand-written, cabinet_clutter layout): table + two 0.20 m cubes
  at (0.58, +-0.36), 0.52 m of free table between them. Compartments = near-cube regions (footprint + 8 cm, up to 5 cm
  above the top); focus box = their bounding box (free = outside it).
- Suites: `run.py cabinet-tests --scene cabinet_twocubes` -> registered 20261101T204007Z-c026d7a8, frozen with
  `experiments/cabinet/freeze_tests.py` into `experiments/cabinet/tests_twocubes/` (600 pairs, 249 straight-line feasible).
- Baseline: `run.py ntrl-cabinet --scene cabinet_twocubes` (defaults = registered metric recipe) ->
  registered 20261101T210049Z-91b929e7 (Docker, 3090; data 2 min, training 27 min, planning 29 min).
- Viewer: `export.py --run results/registered_20261101T210049Z-91b929e7 --output viewer/data.json` (host venv,
  CUDA_DEVICE_ORDER=PCI_BUS_ID) writes data.json + paths_<group>.json; page `viewer/index.html`, published as
  https://claude.ai/artifact/HiZdES2ncBwoazwKaEM7iU

## Results (release scorer, tilt gate off)

| group | straight line works | RK4 /100 | ntrl-demo MPPI /300 | release MPPI /300 |
|---|---|---|---|---|
| free to free | 64 | 77 | 212 | 239 |
| free to cube | 38 | 51 | 139 | 152 |
| cube to free | 40 | 55 | 147 | 157 |
| cube to other cube | 31 | 29 | 82 | 95 |
| same cube | 26 | 45 | 112 | 120 |
| anywhere | 50 | 62 | 170 | 179 |
| **total** | **249 / 600** | **319 / 600 (53 %)** | **862 / 1800 (48 %)** | **942 / 1800 (52 %)** |

Split by the straight line: pairs where it works RK4 85 %, ntrl MPPI 79 %, release MPPI 92 %; pairs that need a detour
RK4 30 %, ntrl MPPI 26 %, release MPPI 24 %. Failures are almost all collisions (ntrl MPPI also 77 joint-limit exits);
the colliding paths hit cube_left / cube_right / table about equally often (table ~30-40 %). Release MPPI never used its
linear fallback. Clearance check of export.py (pure-torch release spheres) agrees with the scorer: 0 of 2,123 successes
below 4.5 mm; 1,969 of 2,034 collision failures below it (the rest self-collision or contact between check points).
