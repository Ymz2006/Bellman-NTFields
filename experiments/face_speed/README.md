# Per-piece (box-face) speed on the cabinet task

The user's formulation (`cspace_face_filter.pdf`, Sections 1–2): cover the robot with boxes, give every box
face its own clearance speed, and take the robot's speed as the minimum over the faces, since the true
clearance speed *is* a minimum over the robot's surfaces. Three tasks, all on the plain `cabinet` scene and
its frozen, certified 600-query suites (`experiments/cabinet/tests/`), scored by the unchanged release scorer:

1. **Metric NTFields with the face speed** (`--method metric --cover faces`): the ntrl-demo baseline trained on
   labels whose clearance is the box-face minimum, against the same baseline retrained in the same environment
   on its own release labels (`--cover release --data <baseline data>`).
2. **Speed accuracy** (`face-speed-accuracy`): the Bellman pipeline's learned speed, a single network for the
   whole body, against the per-piece network whose body speed is the minimum over its outputs.
3. **Bellman with the per-piece speed** (`--method bellman --cover faces|spheres|links`): the unchanged
   TravelCost recipe with the per-piece network as its frozen speed coefficient, against the Bellman pipeline
   rerun in the same environment with the release speed (`--cover release`).

Everything runs on this workstation through the rebuilt environment (`environment/local/README.md`): the
host backend records `environment_override.json`; the pure-torch kinematics used for the face geometry
reproduce the cuRobo labels of the earlier runs to 1e-6 m, and the venv's cuRobo reproduces them bitwise
(`tests/geometry_checks.json`, `tests/curobo_host_check.py`).

## Geometry

|Piece|Definition|
|---|---|
|Box cover|One axis-aligned box per rigid body in that body's link frame, the smallest box containing all of the body's collision spheres (radii padded by cuRobo's 0.005 m buffer): upper arm 0.58 × 0.17 × 0.18 m, forearm 0.52 × 0.15 × 0.18, wrist 1 0.10 × 0.14 × 0.16, wrist 2 0.11 × 0.14 × 0.14, hand (wrist 3 + gripper, rigid) 0.22 × 0.10 × 0.29. The shoulder is left out as the release label leaves out its two spheres.|
|Face clearance|Signed distance from the face rectangle to the nearest scene cuboid: exact for disjoint pairs (closest feature pair of two convex polytopes: rectangle vertex–cuboid, cuboid vertex–rectangle, edge–edge), minus the separating-axis penetration depth when they intersect. `faces.py`, checked against brute force.|
|Pieces|30 faces + the release self-collision term (0.5 d_self, release mask). Piece speed clip(d, 0.005, 0.05)/0.05; body speed = min over pieces.|
|Soundness|Boxes contain the spheres, so the face clearance never exceeds the release clearance: the face speed is a conservative surrogate of the label the scorer's safety rule is built on. On the FK pool it averages 0.50 against 0.58 and is at the floor for 22% of configurations against 8% (`tests/geometry_checks.json`).|
|Controls|`spheres` (one piece per sphere) and `links` (one per link) are exact decompositions: their minimum equals the release label, which separates the effect of the min structure from the effect of the box geometry.|

## Pipelines

|Method|Stages|Unchanged pieces|
|---|---|---|
|metric|`metric_data.py` (the cabinet ntrl-demo recipe, 500,000 pairs, seed 20261019, with the cover clearance and its autograd gradient as the normal) → `experiments/cabinet/metric/train.py` (ntrl-demo trainer, 5,000 epochs) → `experiments/cabinet/evaluate.py --method metric`|trainer, network, evaluation, scorer|
|bellman|`prepare_pieces.py` (per-piece labels of the shared FK pool and coverage set of `registered_20261019T034213Z-f9c501f9`; `speed_net.PieceSpeedField`, the release trunk with one output per piece, 20,000 steps, seed 87358; the offline bank with the release gate, seed 87344) → `training_pieces.py` (338,177-parameter TravelCost, losses 6/2/2, seed 87358, 5 rounds; the TravelCost initialization and minibatch stream are bitwise those of the baseline) → `experiments/cabinet/evaluate.py --method bellman`|poses, gate, bank walk (its transitions do not depend on the speed), TravelCost, losses, evaluation, scorer|

```bash
export BELLMAN600_BACKEND=host PY=~/.venvs/bellman-ntfield/bin/python
$PY run.py face-speed --method metric  --cover faces   --gpu 0
$PY run.py face-speed --method metric  --cover release --data registered_20261019T034237Z-30d4d872 --gpu 1
$PY run.py face-speed --method bellman --cover release --sampled registered_20261019T034213Z-f9c501f9 --gpu 1
$PY run.py face-speed --method bellman --cover faces   --sampled registered_20261019T034213Z-f9c501f9 --gpu 0
$PY run.py face-speed --method bellman --cover spheres --sampled registered_20261019T034213Z-f9c501f9 --gpu 1
$PY run.py face-speed-accuracy --runs registered_<bellman runs> registered_20261019T034213Z-f9c501f9 --gpu 1
python3 experiments/face_speed/report.py --runs registered_<...> --accuracy registered_<accuracy run>
```

## Results (2026-10-25, this workstation; frozen `cabinet` suites, 600 certified queries; release MPPI counts 3 planning seeds)

Registered runs of this experiment: 7 pipelines, 1 accuracy run and 16 filter evaluations (4 of them cross-GPU
repeats of the same configuration, listed in the comparison as a reproducibility check), all through the host
backend; the registry database is `~/.local/share/ntfields-experiments`.

Same-environment baselines were retrained here with the unchanged recipes; they reproduce the original
machine's numbers closely (metric 613/1800 vs 595, Bellman 682/1800 vs 688). Full tables, paired
comparisons and failure reasons: [`evidence/face_speed/COMPARISON.md`](../../evidence/face_speed/COMPARISON.md).

|Run|Field|Speed used for training|RK4 (native)|Release MPPI|per seed|
|---|---|---|---:|---:|---|
|`registered_20261025T065945Z-3c53b360`|metric NTFields|release label (baseline rerun)|181/600 (30.2%)|613/1800 (34.1%)|204/204/205|
|`registered_20261025T065830Z-8e5788c0`|metric NTFields|**box-face minimum** (Task 1)|173/600 (28.8%)|618/1800 (34.3%)|206/206/206|
|`registered_20261025T072528Z-2e175159`|metric NTFields|box-face minimum, x0 filter on the release clearance (Task 1, same x0 distribution as the baseline)|184/600 (30.7%)|626/1800 (34.8%)|211/206/209|
|`registered_20261025T070429Z-b9a2383c`|Bellman|release two-output SpeedField (baseline rerun)|228/600 (38.0%)|682/1800 (37.9%)|227/227/228|
|`registered_20261025T070501Z-aaa42561`|Bellman|**31-output per-face network, min** (Task 3)|**248/600 (41.3%)**|**722/1800 (40.1%)**|238/242/242|
|`registered_20261025T072358Z-932f514d`|Bellman|52-output per-sphere network, min (control: same label as the baseline)|231/600 (38.5%)|684/1800 (38.0%)|229/229/226|
|`registered_20261025T074323Z-331f4da7`|Bellman|18-output per-link network, min (control: same label as the baseline)|231/600 (38.5%)|681/1800 (37.8%)|228/227/226|

- **Task 1 (metric NTFields).** The box-face labels do not change the overall rate (618 vs 613 MPPI executions,
  RK4 173 vs 181); they move successes from the open-space suites (free_to_free 253 vs 262, anywhere 229 vs 239)
  into the compartment entries and exits (free_to_compartment 63 vs 53, compartment_to_free 36 vs 22). The
  ntrl-demo recipe's own x0 filter (0 < d − 0.01 < 0.05) applied to the conservative face clearance leaves almost
  no training pairs inside the cupboard (0.04% vs 0.83%). Keeping the baseline's x0 filter (release clearance)
  with the face labels restores that distribution (0.90% in the cupboard) and gives a small gain over the
  baseline, 626 vs 613 executions and RK4 184 vs 181, with the ntrl-demo MPPI lower (502 vs 520): within what
  one training seed can resolve. For metric NTFields the formulation only changes the labels, and the field
  network still has to fit the same kind of kinked speed, so no large effect was expected.
- **Task 3 (Bellman).** With the per-face speed network as the frozen coefficient, the unchanged TravelCost
  recipe gains on every planning seed: 722 vs 682 executions (+2.2 points) and 248 vs 228 RK4 queries; per suite
  it wins free_to_free (279 vs 268), compartment_to_free (83 vs 64), anywhere (273 vs 258) and loses
  same_compartment (12 vs 22). Paired on the same queries, 108 executions are solved only by the face-speed
  field and 68 only by the baseline. The bank's transitions are identical in all three Bellman runs (399,581
  anchors; the exact veto does not depend on the speed), so the difference is the local costs and the Eikonal
  coefficient alone. The per-sphere control, whose minimum equals the baseline label exactly, stays at the
  baseline (684 vs 682, 231 vs 228), and so does the per-link control (681, 231): **the gain comes from the
  box-face geometry of the speed, not from the min-structured network by itself.**
- **Task 2 (speed accuracy; `registered_20261025T073447Z-a5d29435`).** MAE of the predicted body speed
  against the release clearance speed (the scorer's rule): single two-output net 0.036 (held-out pool),
  0.051 (held-out coverage set), 0.085 inside the cupboard along the witness paths, 0.165 on valid uniform
  draws; per-sphere min net 0.040 / 0.047 / 0.082 / 0.144. Each of its 52 outputs fits its own sphere to
  0.003, but the minimum of 52 outputs accumulates the per-output errors, so in-distribution the single net
  is as good while off-distribution the min net generalizes better. The per-face net fits its own label to
  0.031 / 0.024 and to 0.003 inside the cupboard (where the box faces, not the spheres, define its speed);
  against the release truth it is off by construction (0.097 / 0.184), being the conservative lower bound.

- **Step filter at planning time (the PDF's own mechanism, `guard.py`, `face-speed-guard`).** The trained
  field is left alone and every planner step is projected onto the half-spaces of the active pieces
  (Hildreth, 30 sweeps, joint limits as rows, post-step halving). Evaluated with the native RK4 and the
  ntrl-demo MPPI settings for either field; the release MPPI planner is frozen and not filtered, so these
  rows are not comparable with the release-MPPI column above. On the same-environment Bellman baseline field
  the one-speed filter (one row, the gradient of the body speed; alpha 0.5, floor 0.15) lifts RK4 from
  228/600 to 311/600 and the ntrl-style MPPI from 641/1800 to 992/1800; the per-piece filter (one row per
  sphere, 52 rows) lifts them to **354/600 (59.0%)** and **1147/1800 (63.7%)**, so splitting the speed into
  pieces adds another +43 RK4 queries and +155 MPPI executions over the one-speed filter, the PDF's second
  lesson reproduced on the arm. Compartment suites move most: compartment_to_free 74/100 (20 without a
  filter), same_compartment 40/100 (7), compartment_to_compartment 16/100 (2). Costs: the filter stalls some
  rollouts (RK4 converged 382/600 against 595 unfiltered; the stalls are counted as endpoint failures) and the
  halvings show the linear model breaking for the arm's lever arms. The two parts of the formulation stack:
  the face-trained Bellman field with the per-piece filter reaches **375/600 (62.5%) RK4 and 1217/1800
  (67.6%) MPPI**, the best configuration measured (compartment_to_compartment 33/100 against 16 for the
  baseline field with the same filter and 2 without); with the box faces as the filter's pieces instead it
  falls to 263/600 and 1064/1800. The metric NTFields field gains the same way: with the
  per-piece filter it goes from 181/600 to 317/600 RK4 and from 527/1800 to 1050/1800 MPPI, against 282/600
  and 919/1800 with the one-speed filter. The filter's geometry matters the other way round from the
  training labels: with the box faces as the filter's pieces (conservative, 22% of valid poses already below
  the box floor) the metric field drops to 243/600 RK4 (only 272 rollouts converge) and 970/1800 MPPI, and a
  floor of 0.1 is slightly behind 0.15 on both fields (Bellman 337/1095, metric 306/998). The same filter
  configuration evaluated on the RTX 3090 and on the RTX 5070 gives identical RK4 results (267/600, 288
  converged) and 1028 vs 1030 MPPI executions, so the numbers are reproducible across the two GPUs. Full table: the guard section of
  `evidence/face_speed/COMPARISON.md` (remaining configurations: floor 0.1, the box-face cover as the filter
  geometry, the metric field, the face-trained field with the filter).

Findings write-up (per task, with the explanation of why the min formulation does not move the planner numbers and where the per-piece structure pays off): https://claude.ai/artifact/KAHnkrVHamSkb95GTeaGS4

## Follow-up 2026-10-25: one model for environment collision, one for self-collision

`--cover split` gives the speed two pieces, the environment clearance (min over the spheres) and the self piece
(0.5 d_self); their min is exactly the release label (checked to 2e-5). `--speed-model separate` trains one full
speed network per piece (own Fourier features, tower, head, AdamW and gradient clip; 6.6 M parameters together),
`shared` the usual one trunk with two outputs (the control). Bellman pipeline otherwise unchanged.

|Run|Speed model|RK4 /600|Release MPPI /1800 (per seed)|vs baseline, MPPI gained / lost|
|---|---|---:|---|---|
|registered_20261025T070429Z-b9a2383c|release baseline|228|682 (227, 227, 228)|–|
|registered_20261025T175901Z-4ef50717|two separate networks|227|669 (224, 223, 222)|25 / 38|
|registered_20261025T175946Z-fe060952|one network, two outputs|227|690 (230, 229, 231)|36 / 28|

Accuracy (registered_20261025T181538Z-bb96e76d, MAE of the body speed vs the release label): separate 0.027 held-out
original, 0.045 augmentation, 0.093 witness paths inside the cupboard, 0.112 gate-accepted uniform draws; shared
0.033 / 0.048 / 0.089 / 0.124; baseline 0.036 / 0.051 / 0.085 / 0.165. The separate networks are the most accurate
speed model measured, except inside the cupboard, where they are slightly optimistic (bias +0.012).

Why it does not move the planner: the self term sets the label at 78% of the original-pool endpoints (39% of the
augmentation endpoints), mostly finger <-> wrist_1 and gripper base <-> forearm pairs (median self speed 0.78,
median environment speed 1.0), yet only 12 of the baseline's 1118 release-MPPI collision failures (1 of 368 RK4)
involve self-collision; the rest hit the cabinet. Better self-collision modelling improves the label fit where the
planner does not fail.

## Follow-up 2026-10-25: metric NTFields without the self-collision speed, three cupboards

`--method metric --cover environment` labels the data with the environment clearance only (min over the spheres,
no 0.5 d_self term); `--cover spheres` is the same generator and seed with the full release label (the control);
`--cover release --data <pinned run>` reruns the original baseline data in this environment. The scorer is
unchanged and still requires 2 mm self clearance.

|Cupboard|Label|Run|RK4 /600|Release MPPI /1800|ntrl-demo MPPI /1800|
|---|---|---|---:|---:|---:|
|standard|env + self (control)|registered_20261025T182703Z-1cbd3744|181|603|519|
|standard|env only|registered_20261025T182603Z-750fdcca|187|623|536|
|standard|baseline rerun|registered_20261025T065945Z-3c53b360|181|613|520|
|+10%|env + self (control)|registered_20261025T184017Z-be897341|186|590|518|
|+10%|env only|registered_20261025T183842Z-fdc70d65|191|596|550|
|+10%|baseline rerun|registered_20261025T190635Z-678cce4e|184|594|521|
|+20%|env + self (control)|registered_20261025T185336Z-4891523d|188|614|516|
|+20%|env only|registered_20261025T185115Z-22004ad8|191|618|549|
|+20%|baseline rerun|registered_20261025T190343Z-36ff1869|186|588|542|

Env-only beats its control in all 9 cupboard x planner cells; summed over the three cupboards RK4 569 vs 555,
release MPPI 1837 vs 1807, ntrl-demo MPPI 1635 vs 1553 (+82 of 5400). Per query vs the control, ntrl-demo MPPI
gains 121/131/131 and loses 104/99/98. Failure split (densified paths, any self contact <2 mm counted as self):
cabinet hits fall (ntrl-demo MPPI 1196 -> 1099, 1203 -> 1090, 1213 -> 1115) but self-collisions roughly double
(68 -> 155, 62 -> 155, 42 -> 121), taking back part of the gain. One training seed per run.

## Margin/offset sweep (metric NTFields, spheres label, cabinet, 1M pairs)

`run.py face-speed --method metric --cover spheres --margin M --offset O --pairs 1000000` (new flags; defaults
0.05 / 0.005 / 500000 keep earlier runs unchanged). The margin also sets the x0 window 0 < d - 0.01 < margin.
The unmodified trainer maps the label s to 1.025 s^2 (2-s)^2 - 0.025 before sqrt(speed |grad tau|), so a floor
offset/margin below ~0.082 gives a negative speed and NaN loss (offset 0.001 at margin 0.05 is infeasible).

| margin / offset | floor | run | RK4 /600 | MPPI /1800 | ntrl MPPI /1800 |
|---|---:|---|---:|---:|---:|
| 0.025 / 0.0025 | 0.1 | 5ea7b288 | 192 | 630 | 525 |
| 0.05 / 0.005 (control) | 0.1 | b2db1e91 | 187 | 621 | 535 |
| 0.05 / 0.01 | 0.2 | b6559514 | 186 | 634 | 549 |
| 0.05 / 0.02 | 0.4 | 93f5d4c7 | 195 | 641 | 569 |
| 0.075 / 0.0075 | 0.1 | 54ad5ea8 | 192 | 625 | 504 |
| 0.10 / 0.01 | 0.1 | 40981308 | 174 | 583 | 494 |
| 0.15 / 0.015 | 0.1 | ee154d72 | 162 | 569 | 452 |

500k-pair control 1cbd3744: 181 / 603 / 519.
