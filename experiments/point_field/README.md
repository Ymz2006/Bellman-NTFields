# Hierarchical point-field clearance and speed on the cabinet task

Every speed model so far learns q -> speed directly in the 6-D joint space (`experiments/face_speed`): one
two-output net (release), one output per sphere / link / box face combined by a min, or one model for
environment collision and one for self-collision (split). The release label itself is

    d_env(q) = min_i [ D(c_i(q)) - r_i ],   d_body = min(d_env, 0.5 d_self),   speed = clip(d_body, 5 mm, 50 mm) / 50 mm

with D the distance from a 3-D point to the nearest cuboid, c_i(q) the collision sphere centres from forward
kinematics and r_i their padded radii. The sphere radius is only an offset and the spheres' positions are
exact, so the only thing the scene has to supply is the 3-D function D. This experiment learns D and gets the
clearance and speed of the arm through the exact kinematics:

1. `PointField` (`point_field.py`): Fourier features of the point, a 5 x 256 SiLU MLP (329,985 parameters),
   trained on the exact signed scene distance (positive part = the release's point-to-box distance). Labels
   are exact and cheap, so every step draws fresh points (`train.py`): sphere centres of training
   configurations, the same centres jittered by 1 or 5 cm, and uniform points in the workspace box.
2. `Hierarchical`: FK sphere centres -> D_theta(c_i) - r_i -> min over the 50 non-shoulder spheres = learned
   environment clearance; the self term 0.5 d_self is computed exactly from the same spheres with the release
   mask. Body clearance and speed exactly as the label.

With the exact D in place of D_theta, the hierarchy reproduces the release label to 6e-8 m
(`hierarchical:exact`, a check, not a model).

## Variants and controls

|Model|What is learned|Control for|
|---|---|---|
|`hierarchical:mixed`|D_theta on centres of FK-pool training rows + uniform joint draws|main model|
|`hierarchical:pool`|D_theta on centres of FK-pool training rows only (the configurations the speed nets fit)|more training configurations|
|`...+learned-self`|D_theta for the environment, self piece of the split/separate run's self network|exact self term (everything learned)|
|speed networks|q -> speed (release, spheres, links, faces, split shared, split separate)|baselines, unchanged checkpoints|

Distance is compared as environment clearance (the learned part) and as body clearance. A speed network
encodes distance only inside the label band, so its distance is 0.05 s and the common metric is the
band MAE |clip(d_hat) - clip(d)| with clip to [5, 50] mm; the hierarchical models also report unclipped error.
Speed is compared with `face_speed/speed_accuracy.py`'s metrics on its sample sets (same seed and draws).

```bash
export BELLMAN600_BACKEND=host PY=~/.venvs/bellman-ntfield/bin/python
$PY run.py point-field --sampled registered_20261019T034213Z-f9c501f9 --self-net registered_20261025T175901Z-4ef50717 \
  --runs registered_20261025T070429Z-b9a2383c registered_20261025T072358Z-932f514d registered_20261025T074323Z-331f4da7 \
         registered_20261025T070501Z-aaa42561 registered_20261025T175901Z-4ef50717 registered_20261025T175946Z-fe060952 --gpu 0
```

## Results

Registered run `registered_20261025T213813Z-e167a13e` (host backend, RTX 5070, 15 min; full tables in its
`comparison/REPORT.md`). Point level, held-out sphere centres: 0.36 mm MAE (mixed) / 0.26 mm (pool), 0.86 / 0.72 mm
within 5 cm of a surface.

|Model|Env clearance band MAE, mm (held-out orig / aug / in cupboard / valid uniform)|Body speed MAE (same sets)|Configs/s|
|---|---|---|---:|
|hierarchical, pool-only field, exact self|0.06 / 0.14 / 0.62 / 0.03|0.001 / 0.003 / 0.012 / 0.000|94k|
|hierarchical, mixed field, exact self|0.08 / 0.20 / 0.78 / 0.02|0.001 / 0.004 / 0.015 / 0.000|94k|
|hierarchical, pool-only field, learned self (split/separate self net)|same as above|0.010 / 0.006 / 0.012 / 0.009|75k|
|split, separate nets (`4ef50717`)|1.66 / 2.65 / 4.83 / 8.37|0.027 / 0.045 / 0.093 / 0.112|382k|
|split, shared (`fe060952`)|1.50 / 2.54 / 4.60 / 8.24|0.032 / 0.048 / 0.089 / 0.124|763k|
|per-link (`331f4da7`)|1.17 / 1.84 / 3.79 / 8.73|0.034 / 0.040 / 0.073 / 0.137|759k|
|per-sphere (`932f514d`)|1.34 / 2.13 / 4.23 / 8.71|0.040 / 0.047 / 0.082 / 0.144|755k|
|release two-output (`b9a2383c`)|– (no environment output)|0.035 / 0.051 / 0.085 / 0.165|716k|
|per-face (`aaa42561`, conservative by construction)|6.55 / 10.20 / 8.48 / 12.49|0.097 / 0.184 / 0.165 / 0.186|758k|

- The learned environment clearance is 6-25x more accurate than the best 6-D network's environment output on
  every set, and the body speed error drops from 0.027-0.165 to 0.000-0.015 MAE (p95 from 0.13-0.73 to <= 0.05).
- Pool-only control: the field fitted only on the configurations the speed nets saw is as good or better, so the
  gain is not from extra training configurations; uniform-q centres only help on the uniform-q sets.
- Learned-self control: replacing the exact self term with the split run's self network keeps a 3-10x gain
  (0.005-0.015 MAE); the remaining error is the self net's.
- Off-distribution (valid uniform draws) the 6-D nets are biased pessimistic (-0.10 to -0.15 speed); the
  hierarchical model is unbiased there, since any configuration's spheres lie in the 3-D region the field covers.
- Hardest set for the field: inside the cupboard (0.6-0.8 mm, slight pessimistic bias -0.4 mm), where centres sit
  near panel edges and corners of the box union.
- Cost: 50 point queries per configuration make it ~8x slower than one 6-D net (94k vs ~760k configs/s at batch
  4096). For this cuboid scene the exact point distance is both exact and fastest (1.7M configs/s); the learned
  D is what carries over to scenes without an analytic distance (meshes, point clouds).

## Bellman pipeline with the hierarchical speed (2026-10-26)

`run.py point-field-bellman --speed learned|exact` (`bellman.py`, `bellman_pipeline.py`): the cabinet Bellman recipe
with the speed coefficient replaced, everything else the baseline's. The prepare stage replays the baseline's random
stream, and the run checks that its seed states and bank `source_q`/`successor_q`/`safe` are bitwise those of the
baseline rerun `b9a2383c` (they are); only `local_cost` (integral of 1/speed) and the Eikonal coefficient change.

|Run|Speed|RK4 /600|Release MPPI /1800 (per seed)|MPPI gained / lost vs baseline|Eikonal residual (last 200 steps)|
|---|---|---:|---|---|---:|
|`registered_20261025T070429Z-b9a2383c`|release two-output net (baseline)|228|682 (227, 227, 228)|–|0.119|
|`registered_20261025T175901Z-4ef50717`|split, separate nets|227|669 (224, 223, 222)|25 / 38|–|
|`registered_20261026T010842Z-face045d`|**exact label** (hierarchy with exact distance)|212|647 (213, 217, 217)|36 / 71|0.085|
|`registered_20261026T012043Z-4ecda217`|**hierarchical, learned point field** (pool)|210|647 (214, 216, 217)|37 / 72|0.087|

Per suite (MPPI) the hierarchical/exact runs lose free_to_free (255 vs 268), free_to_compartment (59 vs 67) and
compartment_to_free (41 vs 64) and gain only same_compartment (34 vs 22).

Why a perfect speed plans worse (`bellman_diagnostics.py`, read-only):

- Not a fitting problem: with the exact speed the TravelCost fits its losses better (Eikonal 0.085 vs 0.119,
  Bellman 0.0007 vs 0.0013).
- The failures are gross penetrations, not millimetre misses: 99% of failures are collisions, and in every run about
  two-thirds of all paths reach a minimum body clearance below 5 mm, with the median path minimum at -15 mm (a sphere
  centre inside a panel). Speed errors of 1-5 mm inside the 5-50 mm band are not what decides these queries.
- The speed networks' errors are a systematic, smooth pessimism near obstacles. Mean speed on the bank states by true
  clearance: 35-50 mm: net 0.76, split 0.80, exact 0.86; 50-75 mm: net 0.88, split 0.92, exact 1.00. The networks
  raise the travel cost earlier as the arm approaches an obstacle, which acts as a wider margin than the label's 5 cm.
  The exact label is flat at 1 beyond 50 mm. Planner success follows that pessimism, not accuracy: baseline 682 >
  split 669 > exact 647 = learned 647.
- The learned hierarchy reproduces the exact run (647 vs 647, 210 vs 212), as its accuracy predicts.

So better speed accuracy cannot help while the label's own 5 cm margin is narrower than what the planner needs. In the
hierarchy the margin is a parameter like the radius (no retraining of the point field), so the natural next tests
are the hierarchical speed with a wider margin (for example 0.10 m) and with the environment term only (the
env-only label helped metric NTFields). Both change the speed label and need the user's go-ahead.

Cost: the bank's 217 M speed evaluations take 28 min with the learned hierarchy (prepare 1,691 s) against 8 min for
the baseline net and 6 min with the exact distance.
