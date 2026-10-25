# Wall25 with frozen600 geometry and protocol

Add only `cup_left_wall` to the frozen no-wall scene. Dimensions0.24×0.04×0.25m,
center(0.65,0.28,0.09)m; topz0.215m, bottomz-0.035m. Wall geometry comes from
`ablation600-handoff-20260915/assets/runtime/evaluation/upstream/curobo/scene_ur5e_tabletop_wall25.yml`.
Keep current overlap±0.10m, task axes, table, normalization,600 queries,20deg OR
constraint, collision margins0.005/0.002m and dense0.001rad scoring.
Do not reuse historical Suite-safe180 success rates: that suite has a different
query set and acceptance contract.

First audit endpoint validity under the new obstacle. Invalid endpoints remain
in the full600 denominator, with a separate eligible-endpoint subtotal.
Endpoint validity is not a route-feasibility certificate.

The initial evaluation uses the unchanged frozen no-wall checkpoint, without
training or planner tuning. T(q,g) has no obstacle/scene input. Default MPPI has
no safety rerank, geometric collision-cost tuning, or linear fallback enabled.
This measures direct transfer, not a wall-trained method or learned avoidance
of an unseen obstacle.

All actions go through the repository entry point and shared experiment registry:

```bash
python3 run.py wall-audit
python3 run.py wall-evaluate
```

The frozen no-wall baseline and its evidence remain unchanged.

## Completed direct-transfer evaluation

Registered run `20260930T003519Z-f50d55a8` succeeded. MPPI498/498/498,
1494/1800=83.00%; RK4508/600. Wall-valid endpoints:572/600 queries.
On this same572-query subset, MPPI drops from1525/1716=88.87% without wall
to1494/1716=87.06% with wall. All1800MPPI and600RK4 saved paths are exactly
unchanged; only the added-obstacle scoring differs. No wall-specific training
was performed, and no baseline was replaced.

Report: [COMPARISON.md](../../evidence/wall25/COMPARISON.md). Full trajectories
remain in the registered results directory. The endpoint audit was registered
as `20260930T003413Z-9ba5c691`.

Rebuild the comparison with a NumPy-enabled host Python:

```bash
python3 experiments/wall25/report.py . results/registered_20260930T003519Z-f50d55a8
```
