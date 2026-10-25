# Scene-specific Wall25 offline retraining

User requested wall-specific speed,bank and TravelCost training after the frozen
no-wall direct-transfer test. Keep the scene and endpoints used in that test;
do not select a new wall position or discard queries using model results.

1. Use only the original and augmentation training pose pools (about1M pairs
   each). Recompute physical task speeds for the wall scene. Check the no-wall
   formula against frozen original labels and assert added wall cannot increase
   physical speed. Evaluation endpoints/paths never enter preparation/training.
2. Train the same two-output SpeedField for20000updates, seed87358,50/50source
   sampling, pair-index%100==0 validation split, AdamW3e-4,weight_decay1e-6,
   gradclip2, log-speed MSE. This model is newly initialized, not fine-tuned.
3. Select100000wall-safe training poses, roll50000walkers over8generations;
   sample4paired directions ×4step sizes, integrate17points, veto unsafe edges
   at0.02rad interpolation, choose a safe scale then direction uniformly.
   Keep the original batched ExactLocalGate implementation. No online sampling.
4. Train the unchanged338177-parameter TravelCost from scratch with losses6/2/2,
   seed87358,5rounds×2epochs,batch256. Each round hard-updates target. Since the
   wall changes how many rows have any safe action, update count is computed
   before training as10*ceil(bank_rows/256), rather than forcing15620by duplication.
5. Evaluate only the fifth round, original MPPI settings,seeds0/1/2,strict0.001rad
   scorer. Report full600and the predeclared572endpoint-valid query subset.

Source pools are copied into ignored `training_inputs/wall25_seed_pool`, with
hashes in source_manifest.json. Newly prepared labels/speed/bank/checkpoints
stay in the registered output directory. The frozen no-wall assets and runtime
remain untouched. No archive code is imported: geometry.py preserves the
original gate with a scene argument; physical.py preserves the original formula;
bank.py implements the fixed uniform rollforward branch without legacy switches.

```bash
python3 run.py wall-reproduce --smoke
python3 run.py wall-reproduce
```

The smoke has12speed updates,512walkers×2generations,4TravelCost updates and
2evaluation queries, solely for end-to-end validation. It is not a benchmark.
New outputs are registered; exact repeats require a concrete --repeat-reason.

## Completed full run

`20260930T013432Z-7905f45e` succeeded in777.78seconds. Bank399674rows,
TravelCost15620updates, MPPI492/491/492, RK4494/600.
See [comparison](../../evidence/wall25_train/COMPARISON.md).
The new wall result does not outperform direct transfer; no baseline promotion.
Supplemental pose inputs are packaged separately in `dist/wall25-training-inputs.tar.gz`;
extract at repository root before running on another machine.

## Visual replay

```bash
/home/exx/miniforge3/bin/python gui/viewer.py --wall --port 8902
# For RK4 comparison, add --planner rk4 and use another port.
```

Open http://localhost:8902. Blue is wall-retrained; orange is frozen direct
transfer, both scored in Wall25. Filters show rescued and regressed queries.
Example MPPI seed0: hammering #33 rescued; cutting #7 regressed.
The displayed lines are end-effector traces; strict scores include full-arm
collision and orientation checks. Playback does not recompute paths.
