# jeffrey/easy_cabinet: representability test, easier cabinets, local models and few-level segment DP

*2026-11-01, 06:00-18:00 UTC on the RTX 3090 and RTX 5070. Every planning number is from a registered run (index at
the end): MPPI = release MPPI successes /1800 (600 frozen queries x planning seeds 0/1/2), RK4 = native RK4 /600,
cabinet scorer (tilt gate off), per-suite columns = MPPI successes per 100 queries summed over the 3 planning seeds
(max 300). Training seed 87358 unless stated. Repeat noise of the MPPI total is about +-27 (same weights, planning
seeds), the training-seed noise about +-40 (ES-2 n1: 798 vs 760 with seed 1). Scenes: cabinet (the original),
cabinet_w20 (2 x wide compartments), cabinet_strong (0.10 m panels), cabinet_easy (both), cabinet_shallow (wide,
0.20 m deep). Distances |q - g| are in normalised joint units (1 = 12.57 rad on five joints, 6.28 rad on the third).*

The user's tasks (2026-11-01): (0) test whether the networks can represent a wall's slope at all, on one block with
samples only along the +y joint direction; (1) a model that fits an easier cupboard better, ~60 % as the bar; (1.5) a
model for close query pairs (the first level of the segment DP); (2) segment DP with fewer levels (down to log2 of
the range), five level counts, the new model per level, then per-level variations; (3) two or three cupboards in a
square around the robot, only once (1) has a good baseline.

## Summary

- **Representability (0).** The ntrl-demo metric network *can* represent the slope on one line (contact speed
  0.0128 vs 0.012 target, T through the wall 3.9 vs 3.5 with the log eikonal loss). The embedding stretch and higher
  Fourier frequencies make it worse (collapse to slow-everywhere). The original method's TravelCost gets |grad T|
  right at contact but puts the slope *orthogonal* to the approach direction, so T through the wall is 15x too small;
  ntrl-demo's direction term (1.001 - s')|s' grad T + n|^2 fixes the slope along the line but T still saturates.
- **Easier cabinets (1).** None helps the original method: cabinet 694, w20 662, strong 625, easy 604, shallow 601
  /1800. The failure is the compartment suites (0-23 per 100) and it is the same on every geometry.
- **Local model (1.5).** The best local model is the 2-level eikonal field (edges 0, 0.08, 0.16) with the direction
  term at weight 1 on the wide cabinet: 798/1800 (44 %) at the standard 15,620 updates, 845 / 881 / 916 (51 %) / 897
  with 2x / 4x / 8x / 16x updates (+35-50 per doubling up to 8x on both seeds: 760 -> 814 -> 860 -> 883 on seed 1;
  flat after). The weight sweep is monotone up to 1 (741 / 746 / 752 / 798 / 773 for 0 / 0.1 / 0.3 / 1 / 10). RK4
  collapses with the term (endpoint failures), MPPI gains. The 60 % bar (1080) was not reached; ~50 % is the plateau.
- **Few-level segment DP (2).** With 4, 5, 6, 8 and 10 levels the DP + TD + bell chain never beats its own eikonal
  specialists: on the easy cabinet all ten fields sit in 628-657 (noise); on w20 with the direction term the chain
  *loses* 20-85 (773 -> 721, 775 -> 706, 794 -> 709, 784 -> 738, 736 -> 714). Mechanism: the DP levels are trained
  without the direction term, and their gradients at wall poses fall back to the untrained alignment (cos 0.46-0.49
  with the away-from-wall normal, 13-15 % pointing into the wall, vs 0.64-0.71 and 5-9 % for the specialists).
  Controls: without the term the chain is neutral (688 -> 697); adding the term to the DP levels (new
  `dp3.py --normal-weight`) recovers most of the loss (721 -> 760, 706 -> 739) but still does not beat the eikonal
  field. Per-level architecture variation (16 harmonics in the chain levels: 742; in the specialists: 783 / chain
  751) does not change this.
- **Square of cupboards (3).** Not run: the scene (`cabinet_square.yml`, three easy cupboards at +x, +y, -y with the
  robot in the middle) is derived and checked, but the baseline never reached the bar the user set for it.
- **What the queries are.** Every frozen query is 0.05-0.56 normalised units apart (0.6-4.0 rad, median 3.2 rad), also
  in the "same compartment" suite: the suites test long joint-space moves, not short ones. The compartment suites'
  certified witness paths are 1.75-3.9x longer than the straight line (compartment-to-compartment: 10.9 rad median,
  no straight-line-feasible query). The 2-level local field, trained only up to 0.16, extrapolates on most queries
  and still beats every longer-range field.

## 0. Representability: one panel, one line (`jeffrey/representability/line_test.py`)

Setup. q* = the middle-compartment pose of the Test 4 viewer grid; u = the least-norm IK joint direction moving the
gripper in +y there (release units). Training points lie on the joint-space line x(s) = x* + s u, s in [-0.05, 0.05]
(the gripper travels 6.2 m per unit of s, so the line spans about +-0.3 m), or in a tube of radius 0.02 around it. The
only obstacle is a 0.40 x 0.02 x 0.70 m slab centred on the gripper position at s = 0.03 (plus the arm's self distance);
moving along +y hits it from s = 0.009 on. Truth: the release clearance label on the line, target speed s'(d), travel
time by exact 1-D integration (from s_ref = -0.04 to the far end: 3.54). Pairs: s0, s1 uniform on the line (200k,
41 % of the ends in contact), or only collision-free ends (`--free-only`, the original method's pool rule). Each network
family trains with its own loss: ntrl-demo metric NTFields (its Loss, AdamW 5e-4, 4,000 updates of 2,000 pairs) and
the original method's TravelCost with the eikonal specialist loss (15,620 updates of 256 pairs). Knobs: ntrl `f4/f16`
(Fourier B scale x4/x16), `s1/s5` (the metric factor 1 or 5 instead of 0.2 = the embedding stretch), `nolip`
(no Lipschitz normalisation), `log` (the unbounded log eikonal instead of the bounded sqrt loss); TravelCost `h4/h16`
(harmonics) and `n1` (ntrl-demo's direction term (1.001 - s')|s' grad T + n|^2 added with weight 1). Full tables:
`jeffrey/representability/*/results.json` (line_v1, line_v2, line_free, tube_v1, tube_v2, tube_free).

| data | variant | T at s_end (truth) | T at s_hit (truth) | max slope (truth 83) | median slope in band | speed at contact (target) | speed in band (target) | speed free | RMS log T over pairs |
|---|---|---|---|---|---|---|---|---|---|
| line_v1 | ntrl-base | 5.16 (3.54) | 4.58 (1.87) | 1300 | 33.1 | 0.0128 (0.012) | 0.83 (0.82) | 1.00 | 0.68 |
| line_v1 | ntrl-f4 | 4.71 (3.54) | 4.41 (1.87) | 1076 | 58.7 | 0.0117 (0.012) | 0.76 (0.82) | 0.98 | 0.63 |
| line_v1 | ntrl-f16 | 13.48 (3.54) | 15.25 (1.87) | 8793 | 32.1 | 0.0086 (0.012) | 0.85 (0.82) | 0.99 | 1.42 |
| line_v1 | ntrl-s1 | 32.43 (3.54) | 25.49 (1.87) | 2298 | 105.7 | 0.0034 (0.012) | 0.11 (0.82) | 0.00 | 4.58 |
| line_v1 | ntrl-s5 | 184.97 (3.54) | 114.87 (1.87) | 154101 | 5206.0 | 0.0001 (0.012) | 0.00 (0.82) | 0.00 | 7.40 |
| line_v1 | ntrl-nolip | 1.93 (3.54) | 1.58 (1.87) | 118 | 40.7 | 0.0126 (0.012) | 0.76 (0.82) | 0.97 | 0.44 |
| line_v1 | ntrl-log | 3.87 (3.54) | 2.23 (1.87) | 172 | 81.0 | 0.0115 (0.012) | 0.89 (0.82) | 0.98 | 0.38 |
| line_v1 | tc-h4 | 0.23 (3.54) | 0.16 (1.87) | 71 | 3.2 | 0.0113 (0.012) | 0.79 (0.82) | 1.00 | 1.57 |
| line_v1 | tc-h16 | 0.17 (3.54) | 0.15 (1.87) | 21 | 1.5 | 0.0116 (0.012) | 0.81 (0.82) | 1.00 | 1.58 |
| line_v2 | tc-h4-n1 | 0.25 (3.54) | 0.54 (1.87) | 219 | 41.2 | 0.0129 (0.012) | 0.77 (0.82) | 0.98 | 1.15 |
| line_free | tc-h4 | 0.47 (3.54) | 0.35 (1.87) | 220 | 5.8 | 0.1605 (0.012) | 0.82 (0.82) | 1.00 | 0.01 |
| line_free | ntrl-base | 0.18 (3.54) | 0.10 (1.87) | 5 | 2.5 | 0.2292 (0.012) | 0.79 (0.82) | 1.01 | 0.41 |
| line_free | ntrl-log | 1.29 (3.54) | 1.32 (1.87) | 132 | 9.9 | 0.0627 (0.012) | 0.81 (0.82) | 1.00 | 0.44 |
| tube_v1 | ntrl-base | 6.17 (3.54) | 3.47 (1.87) | 466 | 99.0 | 0.0077 (0.012) | 0.79 (0.82) | 1.00 | 1.03 |
| tube_v1 | ntrl-log | 2.92 (3.54) | 1.39 (1.87) | 122 | 58.9 | 0.0125 (0.012) | 0.78 (0.82) | 1.05 | 0.95 |
| tube_v1 | tc-h4 | 0.25 (3.54) | 0.73 (1.87) | 226 | 46.7 | 0.0128 (0.012) | 0.80 (0.82) | 1.00 | 1.32 |

Findings.
- The ntrl architecture can represent the slope: with the data on the line it reaches the true contact speed
  (0.0128 vs 0.012) and a slope above the true 83, and overestimates T through the wall (5.2 vs 3.5). The unbounded log
  eikonal loss is the best variant (3.9 vs 3.5, RMS log error over pairs 0.38 vs 0.68 for the bounded loss). Raising the
  input frequencies (x4, x16) or stretching the embedding (metric scale 1 or 5 instead of 0.2) does not help: the
  stretched variants collapse to slow-everywhere (T 30-250), the same collapse as the strong-cabinet trainings.
  The embedding stretch does not work because dT/dj = dT/de * de/dj is raised everywhere, not only at the wall: the
  bounded loss then clips the free-space error and the network settles slow everywhere.
- TravelCost gets the gradient *magnitude* right at contact (speed 0.011-0.013) but not T: 0.2 against 3.5 through
  the wall. In 6-D the eikonal loss fixes only |grad T|; the network puts the slope orthogonal to the approach
  direction (median slope along the line 1.5-4 of 83). The direction term raises the slope along the line (25-47) but T
  still saturates: with ends inside the panel the clearance gradient points to the nearest exit, so the term pulls T
  *down* across the second half of the panel. With collision-free ends only (the pool rule) TravelCost fits the free
  side and the slow band exactly (RMS log 0.01); the wall itself is then only an extrapolation, which is what the
  planners run into.
- In the tube (6-D neighbourhood of the line) the picture is the same; ntrl-log 2.9 vs 3.5 through the wall.

## 1. Easier cabinets: original-method baselines (`run.py cabinet`, pool + speed network + 5-round Bellman)

`make_easy.py` (2 x wide, 0.10 m panels, height scale 1.0), `make_shallow.py` (w20 with depth 0.20 instead of 0.40),
suites frozen with `run.py cabinet-tests` + `freeze_tests.py` (tests_easy, tests_shallow). ntrl rows: ntrl-demo metric
NTFields on cabinet_easy (`jeffrey/joint_range/run4.py data/train`, bounded loss vs `--log-eikonal`).

| run | label | scene | RK4 /600 | MPPI /1800 | free to free | free to compartment | compartment to free | compartment to compartment | same compartment | anywhere (MPPI, per 100 x 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261027T205902Z-79094a5f | B5-cabinet | cabinet | 220 | 694 | 276 | 82 | 44 | 14 | 23 | 255 |
| 20261101T063501Z-d7787d27 | B5-w20 | cabinet_w20 | 215 | 662 | 258 | 40 | 58 | 3 | 67 | 236 |
| 20261101T065007Z-9964aee5 | B5-strong | cabinet_strong | 209 | 625 | 258 | 27 | 25 | 0 | 37 | 278 |
| 20261101T070553Z-1969114c | B5-easy | cabinet_easy | 188 | 604 | 236 | 30 | 32 | 3 | 62 | 241 |
| 20261101T093234Z-e7b6223e | B5-shallow | cabinet_shallow | 205 | 601 | 259 | 28 | 56 | 0 | 26 | 232 |
| 20261101T075618Z-f3d663de | ntrl-easy-bounded | cabinet_easy | 25 | 223 | 89 | 9 | 8 | 0 | 19 | 98 |
| 20261101T083044Z-ab966038 | ntrl-easy-log | cabinet_easy | 18 | 237 | 93 | 21 | 10 | 0 | 15 | 98 |

None of the easier geometries helps the baseline: the free suites stay at 78-93 per 100 and the compartment suites
at 0-23. Thick panels make it worse (strong 625, easy 604 vs cabinet 694); a shallower cabinet does not help either
(601). ntrl-demo on the easy cabinet: 223 bounded, 237 log eikonal (the compartment suites 0-21) - the log loss that
won the line test changes nothing on the real scene.

## 1.5. Local models (`jeffrey/segment_dp/run.py --trainer eikonal --mode specialists --edges 0,0.08,0.16 --normal-weight w`)

ES-2 = two eikonal specialists (|q - g| in [0, 0.08] and [0.08, 0.16], the second one answers beyond 0.16), trained on
pool pairs with the frozen speed network; `-n w` = the direction term with weight w (the speed network's unit
gradient as the normal, at both ends); `-h16` = 16 harmonics (`--arch pair-h16`); `-2x` = 31,240 updates.

| run | label | scene | RK4 /600 | MPPI /1800 | free to free | free to compartment | compartment to free | compartment to compartment | same compartment | anywhere (MPPI, per 100 x 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261101T070811Z-91aa968b | ES2-w20-n0 | cabinet_w20 | 254 | 741 | 282 | 62 | 73 | 0 | 52 | 272 |
| 20261101T090315Z-eddb55ef | ES2-w20-n0.1 | cabinet_w20 | 242 | 746 | 288 | 54 | 73 | 0 | 53 | 278 |
| 20261101T091422Z-69b20fe8 | ES2-w20-n0.3 | cabinet_w20 | 35 | 752 | 296 | 48 | 73 | 7 | 57 | 271 |
| 20261101T072612Z-cf54c14d | ES2-w20-n1 | cabinet_w20 | 21 | 798 | 288 | 61 | 83 | 6 | 87 | 273 |
| 20261101T134218Z-0fb15314 | ES2-w20-n1-seed1 | cabinet_w20 | 23 | 760 | 285 | 66 | 63 | 14 | 61 | 271 |
| 20261101T074147Z-c9ad9a90 | ES2-w20-n10 | cabinet_w20 | 31 | 773 | 290 | 50 | 61 | 5 | 91 | 276 |
| 20261101T115053Z-ba02aa2e | ES2-w20-n1-h16 | cabinet_w20 | 9 | 749 | 282 | 62 | 70 | 3 | 62 | 270 |
| 20261101T130202Z-055cd855 | ES2-w20-n1-2x | cabinet_w20 | 20 | 845 | 288 | 55 | 103 | 21 | 107 | 271 |
| 20261101T140718Z-e7119990 | ES2-w20-n1-2x-seed1 | cabinet_w20 | 25 | 814 | 285 | 63 | 84 | 15 | 90 | 277 |
| 20261101T140718Z-62f5287c | ES2-w20-n1-4x | cabinet_w20 | 2 | 881 | 288 | 70 | 105 | 26 | 117 | 275 |
| 20261101T144948Z-60e4d54e | ES2-w20-n1-4x-seed1 | cabinet_w20 | 10 | 860 | 288 | 59 | 104 | 32 | 105 | 272 |
| 20261101T144948Z-9563c30b | ES2-w20-n1-8x | cabinet_w20 | 2 | 916 | 287 | 61 | 130 | 31 | 126 | 281 |
| 20261101T152745Z-d668c84d | ES2-w20-n1-8x-seed1 | cabinet_w20 | 1 | 883 | 279 | 77 | 111 | 28 | 116 | 272 |
| 20261101T152745Z-7c3064fd | ES2-w20-n1-16x | cabinet_w20 | 3 | 897 | 282 | 71 | 124 | 26 | 117 | 277 |
| 20261101T140632Z-16de57f4 | ES3-w20-n1 | cabinet_w20 | 27 | 791 | 288 | 61 | 84 | 3 | 86 | 269 |
| 20261101T095127Z-19a38671 | ES2-shallow-n0 | cabinet_shallow | 220 | 664 | 282 | 40 | 61 | 3 | 12 | 266 |
| 20261101T100055Z-b61dee43 | ES2-shallow-n1 | cabinet_shallow | 25 | 736 | 273 | 59 | 71 | 15 | 48 | 270 |

- The direction term is the one change that moves the compartment suites: same-compartment 52 -> 87 (n1), 91 (n10),
  compartment-to-free 73 -> 83; the weight sweep is monotone up to 1. The term costs RK4 (endpoint failures: the
  gradient flow no longer reaches the goal), which MPPI's sampling does not need.
- More updates is the lever that paid most: 1x 798, 2x 845, 4x 881, 8x 916 (51 %), 16x 897 on seed 87358, and
  760 -> 814 -> 860 -> 883 (1x, 2x, 4x, 8x) on seed 1: +35-50 per doubling on both seeds up to 8x, then flat (16x
  897 is within the repeat noise of 916). Training takes 6-45 min of each run; the rest is the evaluation. The 8x
  field has the best compartment suites of the day (compartment-to-free 130, compartment-to-compartment 31,
  same-compartment 126 per 300; the baseline: 58, 3, 67). The plateau at ~50 % is the ceiling of this local model
  on these suites, not a training-length artefact.
- 16 harmonics: 749 (no gain); three specialists (edges 0, 0.04, 0.08, 0.16): 791 (same as two). On the shallow cabinet the same model gives 736 (n1) vs 664 (n0).
- ES-2 answers every suite query by extrapolation of the 0.08-0.16 specialist (the queries are 0.05-0.56 apart) and
  still beats every field trained on the full range.

## 2. Few-level segment DP (`chain_queue.sh`: `run.py` specialists on the edges, then `run3.py --losses dp,td,bell`)

Edges (normalised |q - g|): 4 levels 0, 0.2, 0.4, 0.8, 1.6 (chain on the first model); 5: 0, 0.1, 0.2, 0.4, 0.8, 1.6;
6: 0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6; 8: 0, 0.08, 0.123, ..., 1.6 (geometric); 10: 0, 0.1, 0.136, ..., 1.6 (geometric). The
chain starts from the first two specialists (one for 4 levels) and trains every further level with DP + TD + bell
(`dp3.py`, 15,620 updates per level, fresh init). ESk = the k eikonal specialists routed by |q - g| (the chain's
starting point and its control), DPTBk = the chain. `-n1` = specialists with the direction term, `-dpn1` = direction
term also at the DP levels (new `dp3.py --normal-weight 1`), `-h16levels` = chain levels with 16 harmonics
(`--new-arch pair-h16`), `ES5-w20-n1-h16` = specialists with 16 harmonics.

| run | label | scene | RK4 /600 | MPPI /1800 | free to free | free to compartment | compartment to free | compartment to compartment | same compartment | anywhere (MPPI, per 100 x 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261101T091922Z-f7340615 | ES4-easy | cabinet_easy | 210 | 648 | 258 | 21 | 41 | 6 | 49 | 273 |
| 20261101T092608Z-6078ff43 | DPTB4-easy | cabinet_easy | 207 | 656 | 267 | 24 | 44 | 7 | 41 | 273 |
| 20261101T072126Z-23f264f7 | ES5-easy | cabinet_easy | 227 | 657 | 258 | 31 | 44 | 9 | 39 | 276 |
| 20261101T072849Z-ec6707b6 | DPTB5-easy | cabinet_easy | 221 | 647 | 253 | 30 | 36 | 12 | 42 | 274 |
| 20261101T075024Z-b6fab83f | ES6-easy | cabinet_easy | 213 | 637 | 248 | 32 | 34 | 12 | 44 | 267 |
| 20261101T075827Z-fd92c68e | DPTB6-easy | cabinet_easy | 212 | 645 | 264 | 26 | 34 | 7 | 50 | 264 |
| 20261101T082846Z-7071f469 | ES8-easy | cabinet_easy | 195 | 655 | 260 | 27 | 36 | 9 | 53 | 270 |
| 20261101T083918Z-e67935b0 | DPTB8-easy | cabinet_easy | 204 | 628 | 251 | 21 | 35 | 9 | 45 | 267 |
| 20261101T094706Z-1022b0c2 | ES10-easy | cabinet_easy | 204 | 636 | 252 | 31 | 31 | 9 | 43 | 270 |
| 20261101T095748Z-0093d27b | DPTB10-easy | cabinet_easy | 225 | 653 | 266 | 31 | 28 | 12 | 46 | 270 |
| 20261101T111925Z-7259a50f | ES4-w20-n1 | cabinet_w20 | 23 | 736 | 284 | 48 | 80 | 2 | 55 | 267 |
| 20261101T113016Z-c776eaba | DPTB4-w20-n1 | cabinet_w20 | 48 | 714 | 280 | 50 | 76 | 0 | 54 | 254 |
| 20261101T090416Z-3c6e59f3 | ES5-w20-n1 | cabinet_w20 | 14 | 773 | 288 | 60 | 84 | 6 | 70 | 265 |
| 20261101T092948Z-394403e2 | DPTB5-w20-n1 | cabinet_w20 | 23 | 721 | 285 | 51 | 67 | 3 | 55 | 260 |
| 20261101T115602Z-00252614 | DPTB5-w20-n1-dpn1 | cabinet_w20 | 14 | 760 | 287 | 55 | 86 | 6 | 65 | 261 |
| 20261101T130407Z-00445db2 | DPTB5-w20-n1-h16levels | cabinet_w20 | 24 | 742 | 284 | 52 | 75 | 7 | 56 | 268 |
| 20261101T133411Z-4e13f911 | ES5-w20-n1-h16 | cabinet_w20 | 8 | 783 | 288 | 52 | 87 | 3 | 86 | 267 |
| 20261101T135217Z-9811ef24 | DPTB5-w20-n1-h16 | cabinet_w20 | 12 | 751 | 282 | 49 | 79 | 5 | 74 | 262 |
| 20261101T101612Z-de68a590 | ES6-w20-n1 | cabinet_w20 | 3 | 775 | 289 | 58 | 88 | 8 | 60 | 272 |
| 20261101T104036Z-d8d8fa3e | DPTB6-w20-n1 | cabinet_w20 | 58 | 706 | 285 | 58 | 69 | 6 | 29 | 259 |
| 20261101T125708Z-07273f44 | DPTB6-w20-n1-dpn1 | cabinet_w20 | 6 | 739 | 282 | 58 | 71 | 5 | 59 | 264 |
| 20261101T102510Z-49f441a8 | ES8-w20-n1 | cabinet_w20 | 15 | 794 | 285 | 48 | 93 | 8 | 87 | 273 |
| 20261101T105823Z-5f7d5f3b | DPTB8-w20-n1 | cabinet_w20 | 54 | 709 | 277 | 51 | 64 | 0 | 56 | 261 |
| 20261101T114818Z-136660c4 | ES10-w20-n1 | cabinet_w20 | 20 | 784 | 294 | 59 | 84 | 6 | 69 | 272 |
| 20261101T120418Z-ce121e37 | DPTB10-w20-n1 | cabinet_w20 | 75 | 738 | 279 | 59 | 75 | 5 | 54 | 266 |
| 20261101T115119Z-7868f7fe | ES5-w20-n0 | cabinet_w20 | 201 | 688 | 279 | 56 | 57 | 0 | 35 | 261 |
| 20261101T122019Z-c09c4e32 | DPTB5-w20-n0 | cabinet_w20 | 230 | 697 | 279 | 48 | 67 | 0 | 34 | 269 |

Findings.
- Level count does not matter: on the easy cabinet the ten fields (5 level counts x ES/DPTB) all lie in 628-657,
  within the repeat noise of each other and 25-50 above the baseline 604. On w20 the ES fields with the direction
  term lie in 736-794 and every chain is below its ES (721, 706, 709, 738, 714).
- Mechanism of the chain's loss: the direction term lives only in the specialists. `direction_residual.py` measures,
  at pool poses with speed < 0.9 (near an obstacle), the cosine between -grad_q T(q, g) and the away-from-wall normal
  and the share of poses whose gradient points into the wall, for goals in each distance bin:

  | bin | pairs | ES5n1 cos / into-wall % | DPTB5n1 cos / into-wall % | ES2n1 cos / into-wall % | ES2n0 cos / into-wall % | B5 cos / into-wall % |
  |---|---|---|---|---|---|---|
  | 0.16-0.4 | 4876 | 0.64 / 9 | 0.49 / 14 | 0.50 / 12 | 0.46 / 17 | 0.35 / 24 |
  | 0.4-0.8 | 20000 | 0.68 / 7 | 0.46 / 15 | 0.45 / 16 | 0.51 / 14 | 0.38 / 22 |
  | 0.8-1.6 | 20000 | 0.71 / 5 | 0.49 / 13 | 0.46 / 16 | 0.56 / 12 | 0.40 / 21 |

  The specialists with the term (ES5n1) point away from the wall at 0.64-0.71 cosine with 5-9 % into the wall; the
  chain's DP levels (DPTB5n1, levels 0.2-1.6) fall back to 0.46-0.49 and 13-15 %, the same as the fields without the
  term (ES2n0, and ES2n1 beyond its trained range), and the original baseline is worse still (0.35-0.40, 21-24 %).
- Controls: without the term the chain is neutral (ES5-w20-n0 688 -> DPTB5-w20-n0 697, same as on the easy cabinet).
  With the term added to the DP levels the chain recovers (721 -> 760, 706 -> 739) to the ES level, not above it.
- `chain_vs_es.py` (implied speed 1/|grad T| along the suites' certified witness paths against the speed network):
  the chains are not worse than the specialists on magnitude (RMS log ratio 0.65-0.70 vs 0.76-0.81) and have fewer
  rising steps towards the goal; the chain's loss is a direction effect, not a magnitude effect
  (`chain_vs_es_w20.json`).
- Per-level architecture: 16 harmonics in the chain levels 742 (vs 721 with 4), in the specialists 783 (vs 773),
  chain on them 751: within noise.

## 3. Square of cupboards (`make_square.py` -> `experiments/cabinet/scenes/cabinet_square.yml`)

Three easy cupboards (2 x wide, 0.10 m panels) at +x, +y and -y of the robot with their own table slabs, sampling box
x, y in [-0.96, 0.96], focus = the cupboards' bounding box, nine compartments (`*_px`, `*_py`, `*_ny`). Derived and
checked (`--check`), suites not generated: the user's condition for task 3 was a good baseline on task 1 (~60 %), and
the best field is 51 %.

## Files and commands

- `make_easy.py`, `make_shallow.py`, `make_square.py`: scene derivations (checked with `--check`); suites frozen with
  `run.py cabinet-tests` + `freeze_tests.py`.
- `chain_queue.sh SCENE PREPARED_RUN GPU ARCH NEWARCH BASE "EDGES"...` (env NORMAL=w): specialists on the edges, then the
  DP + TD + bell chain on the first BASE models.
- `jeffrey/segment_dp/fields.py`: `pair-hH[-iI]` architectures (H harmonics, input scale I) usable in `eikonal.py`
  (`--arch`) and `dp3.py` (`--new-arch`); `eikonal.py --normal-weight w` and `dp3.py --normal-weight w` (via
  `run.py` / `run3.py`): the direction term with the speed network's unit gradient as the normal.
  `experiments/ntrl_cabinet/train.py --log-eikonal`, `jeffrey/joint_range/run4.py train --log-eikonal`: the unbounded
  eikonal loss for ntrl-demo.
- `chain_vs_es.py`, `direction_residual.py` (host venv, `CUDA_DEVICE_ORDER=PCI_BUS_ID`): the two diagnostics above;
  outputs `chain_vs_es_w20.json`, `direction_residual_w20.json`.
- `tables.py lines|runs`: the tables of this file.

## Run index

All runs are in the tables above (first column). Scene preparation: cabinet_w20 pipeline 20261101T063501Z-d7787d27
(pool, speed network and B5 baseline), cabinet_easy 20261101T070553Z-1969114c, cabinet_shallow
20261101T093234Z-e7b6223e (tests 20261101T075742Z-88fa14a3), cabinet_strong 20261101T065007Z-9964aee5; ntrl easy
data 20261101T075527Z-2564f140; dp3 direction-term smoke 20261101T115455Z-38400d48.
