# P-NTFields path viewer

3D page replaying planned paths on the real arm geometry:

- Task 1: P-NTFields on its own UR5 cabinet, released vs retrained checkpoint, 16 queries.
- Task 2: P-NTFields vs ntrl-demo on `cabinet`, `cabinet_pi`, `cabinet_w25`, `cabinet_w20` and `cabinet_w15`, 4 queries per suite. Each query has the witness path plus both fields under ntrl-demo's MPPI, the release MPPI (seed 0) and the native RK4, and P-NTFields' own planner.

Red frames mark contact under each scene's collision test.
- Task 1 uses exact FCL link-mesh contact. Self contact is shown but not scored, as in the paper.
- Task 2 uses the release sphere model at the scorer's margins (cabinet < 5 mm, self < 2 mm). On all 840 exported planner paths this agrees with the scorer's collision verdict.

Published: https://claude.ai/artifact/BN95zNGmWN9mnQ8gyyaN2N

```bash
cd baselines/P-NTFields && sh docker/run.sh python eval/visualize_export.py && cd -                      # Task 1 json (GPU 0)
~/.venvs/bellman-ntfield/bin/python experiments/pntfields_cabinet/visualizer/export_cabinet.py           # Task 2 json
python3 experiments/pntfields_cabinet/visualizer/build.py                                                # -> viewer.html
```

`kinematics.js` (forward kinematics) is checked in the page itself against the Python link poses in both exports. The footer prints the error, about 1e-6 m.
