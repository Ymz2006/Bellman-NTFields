# ntrl-demo (metric NTFields) on the cabinet task: can it reach the paper's success rate?

The user asked (2026-10-26) to make the ntrl-demo method (`/home/jeffrey/bellman-ntfield/ntrl-demo`, commit
`5ed065f6`, paper arXiv 2505.05691, ICLR 2025) reach 80%+ on the cabinet scenario instead of ~30%. They allowed
changes to preprocessing, cabinet size, parameters and planner settings, not to the method, and later also changes
that bring the setup in line with the paper.

**Result: no setting got above 33% (596/1800 ntrl-demo MPPI executions) on any cabinet's six certified suites.**
Free-space suites run at 75–84%. The four compartment suites run at 0–22%, and almost every failure is the gripper
passing through a cupboard panel. The trained field does not learn the slow regions near the panels: its
speed correlates about 0.1 with its own training labels. Every variant behaves like the straight joint-space line
(202/600 on the cabinet suites).

What the paper actually reports: the cabinet result is a **12-DOF dual-arm** robot in a real cabinet, with 91% on
"100 randomly sampled start and goal pairs" (Sec. 5.2). There is no single-UR5e cabinet result, and the queries are
not stratified into compartments. Random pairs correspond to our `anywhere` and `free_to_free` suites, where the
method already reaches 75–84% (the straight line reaches 74–82%).

## Code

All files are new; nothing in ntrl-demo, the runtime or the other experiments was changed.

|File|What|
|---|---|
|`data.py`|ntrl-demo arm recipe (as `experiments/cabinet/metric/data.py`, chunked so it fits next to other jobs). The defaults reproduce the registered baseline data bit for bit. Options: `--coords release\|ntrl` (q/q_range or ntrl-demo's q/2π), `--label release\|env` (with/without the 0.5·self term), `--margin/--offset`, `--pairs`, `--window`, `--step` (`0` = paper C.1 random configuration pairs), `--focus F` (fraction of x0 inside the cupboard), `--joint-limit` (ntrl-demo's ±π).|
|`train.py`|The user's ntrl-demo trainer, unchanged on disk. `--variant metric\|metric_arm` (metric_arm is what ntrl-demo's own `train_arm.py` uses), `--epochs`. Paper-alignment options are applied by substituting the constants at load time: `--td-step`, `--normal-weight` and `--td-weight` (paper C.2 manipulator values 0.005 / 2e-4; the demo hard-codes 0.02 / 1e-3), `--raw-speed` (paper Eq. 3: S* without the demo's s²(2−s)² stretch), `--metric-scale` (paper Eq. 10 has no 0.1 factor), `--weight-decay`, `--lr`.|
|`field.py`, `evaluate.py`|The field in release coordinates for native RK4 and the release MPPI, and ntrl-demo's own MPPI in the training coordinates (`--ntrl-*` planner settings). Same protocol and unchanged scorer as `experiments/cabinet/evaluate.py`.|
|`pipeline.py`, `run.py`|`run.py ntrl-cabinet --scene S --ntrl-args="..." [--data run]` registers a run (the new action is added to `run.py`/`launch.py`). `--ntrl-args=--make-tests` generates a scene's suites in its `joint_limit` domain (`make_tests_pi.py`).|
|`make_scenes.py`|Larger cupboards `cabinet_w15/w20/w25` (compartments 1.5/2/2.5× as wide and 1.2/1.25/1.3× as tall, centre line moved towards the robot so they stay reachable; w25 has the paper cabinet's ~0.9 m compartment width).|
|`make_tests_pi.py`, `experiments/cabinet/scenes/cabinet_pi.yml`|cabinet geometry with every joint restricted to \|q\| ≤ π (ntrl-demo's IK domain); suites frozen in `experiments/cabinet/tests_pi/`.|
|`diagnose.py`|Speed fit of a checkpoint on its training pairs (inside/outside the cupboard).|
|`failures.py`|Per-path failure breakdown: first contact (cuboid or self, robot link, gripper position, position along the path), convergence, clearance.|

Frozen suites added: `tests_w15` (`20261026T015134Z-54e20869`), `tests_w20` (`20261026T014113Z-94900c1a`),
`tests_w25` (`20261026T014217Z-7e59beb7`), `tests_pi` (`20261026T032734Z-34cb5d44`). The straight joint-space line
solves 205 / 215 / 202 / 173 of 600 on w20 / w25 / w15 / pi (cabinet 202). Bigger cupboards do not make the
queries easier for a straight line.

## Results (ntrl-demo MPPI executions out of 1800 = 600 queries × planning seeds 0/1/2; RK4 out of 600)

Registered runs (host backend, one training seed each):

|Run|Scene|Setting|RK4|Release MPPI|ntrl-demo MPPI|
|---|---|---|---:|---:|---:|
|`registered_20261025T065945Z-3c53b360` (earlier)|cabinet|baseline recipe|181|613|520|
|`20261026T023140Z-b00193b9`|cabinet_w15|baseline recipe|183|588|532|
|`20261026T022740Z-c32709b5`|cabinet_w20|baseline recipe|201|600|544|
|`20261026T022529Z-b25f368e`|cabinet_w25|baseline recipe|179|600|517|
|`20261026T041234Z-664b5d50`|cabinet|best: metric_arm, ntrl coords, env label, MPPI 200 samples / horizon 10 / 400 steps|174|581|596|
|`20261026T041504Z-379bcde5`|cabinet_pi|same, ±π data and suites|186|622|588|
|`20261031T075552Z-9d927f48`|cabinet|best setting, bare eikonal loss: TD weight 0 and normal weight 0 (`--td-weight 0 --normal-weight 0`, data of `664b5d50`)|173|568|541|

Exploration (unregistered, scratchpad; data/training/evaluation identical to the pipeline). The control A reproduces
the registered baseline exactly (181 / 613 / 520):

|Run|Change from the baseline recipe|RK4|Release MPPI|ntrl-demo MPPI|
|---|---|---:|---:|---:|
|A|none (control)|181|613|520|
|B|metric_arm network (ntrl-demo's arm package)|199|599|542|
|C|metric_arm + ntrl coordinates q/2π|174|568|534|
|D|C + environment-only label|174|581|561|
|E|D with the `metric` network (B ~ N(0,1))|146|589|472|
|F|D + 50% of x0 inside the cupboard|175|559|503|
|G|D + x1 step ≤ 0.2|174|572|536|
|H|D + 10,000 epochs|171|575|537|
|I1 / I2|D, ntrl MPPI 500 samples / 400 steps; 200 samples, horizon 10, 400 steps|174|—|580 / **596**|
|K / L|D on cabinet_w25 / + 50% cupboard x0|189 / 195|605 / 606|580 / 569|
|P1 / P2|D + paper TD step 0.005, λN 2e-4 (5,000 / 9,500 epochs)|177 / 180|558 / 568|548 / 545|
|Q1 / Q2|P2 + random configuration pairs, raw S* / demo stretch|135 / 144|451 / 463|415 / 421|
|W0 / W1|D, weight decay 0 / 0.01|176 / 178|574 / 603|527 / 525|
|M03 / M10p|D, metric scale 0.3 / 1.0 + paper loss settings|188 / 183|593 / 574|555 / 592|
|R|D, ±π data, on the ±π suites|186|622|575|

M10 (metric scale 1.0 with the demo's speed stretch) diverged in training (field speed ≈ 0); its evaluation was stopped at the user's request.

## Bare eikonal loss (2026-10-31)

The best setting retrained with only the eikonal loss LE (with the demo's causality weight exp(−0.5T)); the TD loss
and the normal loss are switched off by setting their weights to 0 at load time. Same data, seed and planners.

|Suite (ntrl-demo MPPI /300)|Full loss (`664b5d50`)|Eikonal only (`9d927f48`)|
|---|---:|---:|
|free_to_free|234|233|
|free_to_compartment|43|28|
|compartment_to_free|67|47|
|compartment_to_compartment|1|1|
|same_compartment|26|17|
|anywhere|225|215|
|total /1800|596|541|

RK4 is unchanged (174 → 173), and the release MPPI drops 581 → 568. Free space is unchanged, and the compartment suites lose 44
executions (137 → 93). The speed fit is almost the same: correlation with the label is 0.087 vs 0.101 on the training pairs,
the medians are 0.70 vs 0.69 at S* 0.2–0.3 and 0.85 vs 0.83 in free space, and 36% vs 41% of in-collision fresh configurations read as fast.
The TD and normal losses add about 2–3 points, all in the compartment suites. The field's failure to resolve the walls
comes from the eikonal fit itself, not from the extra losses.

## Why it fails (best model I2, `failures.py`)

- All 1,204 failures are collisions (32 also leave the joint limits). The planner reaches the goal in 1,198 of them:
  it follows the field to the goal along a path that cuts through obstacles. No contact is on the final segment.
- First contact: cupboard panels 86%, self 9%, table 5%. In ~85% of contacts the first sphere in contact belongs to the
  gripper (finger pads, fingers, knuckles). The median minimum clearance is −0.025 m, so the path crosses the panel.
- Starting in a compartment, the first contact comes after a median 2.6–4% of the path (leaving the compartment). Going in,
  the median is 57%. The free-space failures are mostly self-collisions (the env-only label has no self term).
- The method succeeds on 596 executions, the straight line on 606; it wins 113 where the line fails and loses 123
  where the line succeeds.
- Field fit (`diagnose.py`): on its own training pairs the field's speed 1/|∇T| correlates 0.03–0.24 with the label.
  Its 5th percentile is ~0.45 against label speeds down to 0.15. Inside the cupboard it reads ~0.6–0.7 against
  true 0.25–0.45. None of the changes above improved this. Raising the network's metric scale lets |∇T| grow
  (or diverge) but does not make it follow the labels.

One training seed per run; planning seeds 0/1/2 are not training seeds. Differences of ~±20 executions out of 1800
are within the binomial noise of the query draw.
