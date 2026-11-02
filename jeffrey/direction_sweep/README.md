# Direction sweep: does a Bellman min over more random-walk directions help?

Question (2026-10-28): the original method's offline bank gives every source pose 4 random unit directions, each in
both signs, at 4 step sizes: 32 candidate actions. The Bellman target is the min over these candidates. Does a min
over more directions make the field more accurate (travel time, implied speed) and the planners more successful on
the frozen suites?

## Summary

**More directions do not make the Bellman update help.** A min over more random-walk directions is a better local
min: the best safe action's cosine with the optimal-path direction rises from 0.57 (4 pairs) to 0.82 (32 pairs),
and each field's gap to its own backup shrinks from 0.7 % to 0.2 %. The trained field and the planners do not
improve, and the two scenes move in opposite directions:

| MPPI /1800 (RK4 /600), mean of 3 training seeds | D=1 | D=2 | **D=4 (original)** | D=8 | D=16 | D=32 |
|---|---|---|---|---|---|---|
| no-wall (frozen 600) | 1559 (521) | 1581 (536) | **1589 (541)** | 1584 (531) | 1582 (532) | 1581 (523) |
| cabinet (frozen suites) | 709 (239) | 701 (234) | **693 (230)** | 688 (230) | 681 (224) | 680 (225) |

- **Accuracy does not improve.** Implied speed (1/|grad T| vs the speed net) is flat on no-wall: mean |log| 0.244–0.254 for
  D ≥ 2, 0.272 at D = 1. On the cabinet it gets worse (short pairs 0.295 at D = 1, 0.313 at D = 4, 0.346 at D = 32).
  Travel time against the ground truth gets slightly *less* accurate: more directions remove the min's zig-zag
  inflation, the field drops, and it already under-reads (short-pair T/T* no-wall 0.99 / 0.94 / 0.90 and cabinet
  0.89 / 0.85 / 0.83 at D = 1 / 4 / 32).
- **The Bellman min itself is neutral.** With the min over 32 pairs and everything else at 4, the result equals D = 4 in
  both scenes (no-wall 547/1596 vs 550/1598; cabinet 229/689 vs 229/688); so does a min over 1 pair (547/1592; 229/692).
- **Every planning change goes through the direction term.** The bank is used a second time: the gradient is pulled
  toward the Bellman winner wherever the winner leads the runner-up by 10 %. Giving only that term 32 pairs reproduces
  D = 32 (527/1583; 213/662). Giving it 1 pair reproduces D = 1 (521/1562) and is the best cabinet field
  (MPPI 727/704/712 vs 688/697/698 over three seeds).

## Design

- **Nested extension banks** (`extend.py`): the same 399,725 / 399,581 source poses as the frozen no-wall and cabinet
  banks. The frozen 4 direction pairs are copied verbatim; pairs 5–32 are new random unit directions, evaluated under
  the bank builder's rules: 17-point trapezoid cost with the frozen speed net, |z| ≤ 0.5, the scene's exact
  edge gate at 0.02 rad, and minimum speed 0.015. Layout `[source, scale, sign, pair]`, so pairs `[:D]` is the D-pair
  bank. Audit: re-evaluating the frozen 4 pairs with the same code reproduces the frozen safe flags exactly (16,384
  sources, both scenes) and the costs to 1e-5.
- **Training** (`train.py --pairs D`): the original method, unchanged except for the candidate set: 6/2/2 losses,
  seed, 5 rounds, 15,620 updates, round 5 evaluated. D = 4 reproduces the original method tensor for tensor (seed
  87358: no-wall = gap-curriculum B5 run 45797805, cabinet = f9c501f9; the `assets/` checkpoints come from the original
  lab GPU). With D < 4 some sources have no safe action (no-wall 1.7 % at D = 1, 0.4 % at D = 2; cabinet 1.1 % / 0.3 %).
  Those rows leave the Bellman and direction terms of their batch.
- **D ∈ {1, 2, 4, 8, 16, 32} pairs** = 8 … 256 actions; training seeds 87358, 1, 2 (the original method's own seed
  set); both scenes.
- **Controls** (`train2.py --bellman-pairs Db --direction-pairs Dd`): the bank enters the loss twice, as the Bellman
  target (min over candidates) and as the direction term (gradient pulled toward the winning action, only where the
  winner beats the runner-up by ≥ 10 % of its step cost). The controls give the two terms different direction counts.
- **Metrics**: frozen benchmarks (no-wall: 600 queries, RK4 + release MPPI × 3 planning seeds; cabinet: 6 suites ×
  100, same planners). Ground-truth fit (`segment_dp/fit.py` against the path-optimized truth runs 181431Z-bf1c963f /
  175501Z-5150664b). Implied speed 1/|∇T| against the speed net, and Bellman self-consistency (`diag.py`, post-hoc
  `diag2.py`).

## Results


### nowall (extension registered_20261028T183647Z-c06520bb; audit safe agreement 1.0000, cost max rel 8.7e-06)

| pairs D | actions | mean safe actions | sources w/o safe action | best cos to optimal direction |
|---|---|---|---|---|
| 1 | 8 | 2.3 | 6721 (1.7%) | 0.259 |
| 2 | 16 | 4.5 | 1412 (0.4%) | 0.428 |
| 4 | 32 | 9.1 | 0 (0.0%) | 0.567 |
| 8 | 64 | 18.1 | 0 (0.0%) | 0.675 |
| 16 | 128 | 36.3 | 0 (0.0%) | 0.757 |
| 32 | 256 | 72.5 | 0 (0.0%) | 0.818 |

| D | seeds | RK4 /600 per seed | RK4 mean | MPPI /1800 per seed | MPPI mean | MPPI gained/lost vs D4 (sum over seeds) | checkpoint = baseline? |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 523/521/519 | 521.0 | 1555/1562/1560 | 1559.0 | +36/−125 |  |
| 2 | 3 | 530/538/540 | 536.0 | 1580/1580/1582 | 1580.7 | +39/−63 |  |
| 4 | 3 | 550/538/535 | 541.0 | 1598/1585/1583 | 1588.7 |  | yes, NO, NO |
| 8 | 3 | 534/529/530 | 531.0 | 1580/1592/1579 | 1583.7 | +36/−51 |  |
| 16 | 3 | 532/530/534 | 532.0 | 1575/1584/1586 | 1581.7 | +39/−60 |  |
| 32 | 3 | 523/523/522 | 522.7 | 1586/1576/1581 | 1581.0 | +54/−77 |  |

MPPI successes per task (mean over training seeds, of 300 = 100 queries x 3 planning seeds); RK4 per task of 100 in brackets

| D | cutting | hammering | mixed | screwdriver | stirring | wiping |
|---|---|---|---|---|---|---|
| 1 | 287 (97) | 273 (90) | 196 (70) | 260 (88) | 267 (86) | 277 (90) |
| 2 | 289 (98) | 272 (91) | 204 (74) | 264 (91) | 275 (91) | 277 (91) |
| 4 | 289 (98) | 277 (94) | 203 (75) | 267 (91) | 276 (90) | 277 (93) |
| 8 | 288 (96) | 276 (93) | 204 (72) | 264 (91) | 275 (90) | 277 (89) |
| 16 | 287 (97) | 275 (93) | 204 (73) | 265 (90) | 275 (90) | 276 (89) |
| 32 | 289 (96) | 278 (92) | 200 (68) | 263 (89) | 275 (89) | 275 (89) |

Accuracy against ground truth (truth.py optimized paths), mean over training seeds. rel err = mean |log T/T*|; bias = median T/T*; speed err = mean |log S|grad T|| (implied speed 1/|grad T| vs the speed net); short/mid/long gap = 0-0.24 / 0.24-0.8 / 0.8-1.6

| D | rel err short/mid/long | bias short/mid/long | speed err short/mid/long | cosine to optimal dir short/mid/long |
|---|---|---|---|---|
| 1 | 0.183/0.572/0.901 | 0.991/0.657/0.402 | 0.386/0.294/0.263 | 0.897/0.589/0.457 |
| 2 | 0.185/0.591/0.918 | 0.954/0.633/0.395 | 0.378/0.272/0.242 | 0.895/0.588/0.449 |
| 4 | 0.183/0.599/0.921 | 0.936/0.624/0.394 | 0.378/0.269/0.234 | 0.896/0.596/0.463 |
| 8 | 0.184/0.614/0.933 | 0.912/0.609/0.389 | 0.378/0.264/0.235 | 0.897/0.601/0.470 |
| 16 | 0.189/0.610/0.924 | 0.914/0.615/0.392 | 0.386/0.270/0.239 | 0.900/0.606/0.482 |
| 32 | 0.193/0.613/0.918 | 0.902/0.612/0.394 | 0.396/0.276/0.244 | 0.902/0.607/0.487 |

On 16,384 bank sources with random goals (mean over training seeds; diag2.py post-hoc run when given, else diag.py). speed ratio = S(q)|grad_q T| (1 = implied speed exact, >1 field predicts slower than S); backup ratio = T / min over D' pairs of [c + T(z)] (self-consistency with a min over D' directions); confident = rows the direction term uses (gap >= 10 %)

| D | speed ratio median | speed |log| mean | speed within 5 % | S|grad T| at S<0.3 | backup ratio own D | backup ratio D'=4 | backup ratio D'=32 | winner cosine own D | confident own D |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.044 | 0.272 | 12.5 % | 0.828 | 0.9882 | 0.9936 | 0.9980 | 0.318 | 0.750 |
| 2 | 1.041 | 0.250 | 13.4 % | 0.854 | 0.9909 | 0.9934 | 0.9978 | 0.450 | 0.819 |
| 4 | 1.040 | 0.246 | 13.7 % | 0.855 | 0.9933 | 0.9933 | 0.9978 | 0.566 | 0.733 |
| 8 | 1.032 | 0.244 | 14.0 % | 0.856 | 0.9951 | 0.9931 | 0.9977 | 0.665 | 0.629 |
| 16 | 1.032 | 0.247 | 13.9 % | 0.846 | 0.9967 | 0.9931 | 0.9978 | 0.743 | 0.517 |
| 32 | 1.046 | 0.254 | 13.0 % | 0.844 | 0.9980 | 0.9932 | 0.9980 | 0.799 | 0.389 |

Mechanism control: Bellman min over Db pairs, direction term (winner, runner-up, confidence gate) over Dd pairs; MPPI gained/lost vs D4 of the same seed; failures summed over RK4 / MPPI

| Db | Dd | seed | RK4 /600 | MPPI /1800 | MPPI gained/lost vs D4 | RK4 failures | MPPI failures |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 87358 | 523 | 1555 | +3/−46 | collision 8, endpoint 25, tilt 59 | collision 57, tilt 212 |
| 1 | 4 | 87358 | 547 | 1592 | +1/−7 | collision 8, endpoint 27, tilt 28 | collision 64, tilt 177 |
| 4 | 1 | 87358 | 521 | 1562 | +3/−39 | collision 11, endpoint 27, tilt 55 | collision 57, tilt 205 |
| 4 | 4 | 87358 | 550 | 1598 |  | collision 7, endpoint 28, tilt 25 | collision 64, tilt 171 |
| 4 | 32 | 87358 | 527 | 1583 | +7/−22 | collision 11, endpoint 32, tilt 44 | collision 62, tilt 181 |
| 32 | 4 | 87358 | 547 | 1596 | +1/−3 | collision 7, endpoint 28, tilt 25 | collision 68, tilt 173 |
| 32 | 32 | 87358 | 523 | 1586 | +8/−20 | collision 10, endpoint 31, tilt 50 | collision 61, tilt 179 |
| 1 | 1 | 1 | 521 | 1562 | +22/−45 | collision 8, endpoint 25, tilt 57 | collision 46, tilt 213 |
| 4 | 4 | 1 | 538 | 1585 |  | collision 6, endpoint 35, tilt 33 | collision 54, tilt 191 |
| 32 | 32 | 1 | 523 | 1576 | +24/−33 | collision 11, endpoint 32, tilt 43 | collision 51, tilt 200 |
| 1 | 1 | 2 | 519 | 1560 | +11/−34 | collision 13, endpoint 35, tilt 44 | collision 54, tilt 204 |
| 4 | 4 | 2 | 535 | 1583 |  | collision 10, endpoint 36, tilt 28 | collision 49, tilt 190 |
| 32 | 32 | 2 | 522 | 1581 | +22/−24 | collision 11, endpoint 32, tilt 46 | collision 46, tilt 191 |

Direction-term mechanism (post-hoc diag2, training seed 87358). confident = share of sources the direction term uses with its own Dd; winner cos = cos(-grad T, winner step) at Dd; toward clearance = mean cos(-grad T, grad S) by speed band (S < 0.3 is next to obstacles)

| field | Db | Dd | RK4 | MPPI | confident | winner cos | toward clearance S 0-0.3 / 0.3-0.6 / 0.6-0.9 | S|grad T| at S<0.3 |
|---|---|---|---|---|---|---|---|---|
| D4 | 4 | 4 | 550 | 1598 | 0.730 | 0.565 | 0.739 / 0.787 / 0.535 | 0.847 |
| B32D4 | 32 | 4 | 547 | 1596 | 0.731 | 0.566 | 0.741 / 0.791 / 0.539 | 0.850 |
| B1D4 | 1 | 4 | 547 | 1592 | 0.733 | 0.566 | 0.738 / 0.785 / 0.530 | 0.846 |
| D32 | 32 | 32 | 523 | 1586 | 0.393 | 0.798 | 0.723 / 0.756 / 0.484 | 0.867 |
| B4D32 | 4 | 32 | 527 | 1583 | 0.399 | 0.798 | 0.722 / 0.754 / 0.480 | 0.875 |
| D2 | 2 | 2 | 530 | 1580 | 0.819 | 0.450 | 0.737 / 0.795 / 0.553 | 0.865 |
| D8 | 8 | 8 | 534 | 1580 | 0.635 | 0.666 | 0.735 / 0.784 / 0.527 | 0.881 |
| D16 | 16 | 16 | 532 | 1575 | 0.513 | 0.743 | 0.727 / 0.762 / 0.489 | 0.862 |
| B4D1 | 4 | 1 | 521 | 1562 | 0.749 | 0.318 | 0.708 / 0.768 / 0.551 | 0.846 |
| D1 | 1 | 1 | 523 | 1555 | 0.750 | 0.319 | 0.707 / 0.766 / 0.547 | 0.840 |

Runs: D1 s1 registered_20261028T213408Z-1aa70bff, D1 s2 registered_20261028T232237Z-5a263a8d, D1 s87358 registered_20261028T194029Z-d883963f, D2 s1 registered_20261028T224758Z-10f1dd5c, D2 s2 registered_20261028T235932Z-20869aa5, D2 s87358 registered_20261028T203011Z-67aca319, D4 s1 registered_20261029T002111Z-837c22d9, D4 s2 registered_20261029T003150Z-51dc56f1, D4 s87358 registered_20261028T193347Z-a7c6f944, D8 s1 registered_20261028T230036Z-b69029bc, D8 s2 registered_20261029T001327Z-bf4619e1, D8 s87358 registered_20261028T203955Z-15f7c5e8, D16 s1 registered_20261028T214718Z-c6887b7d, D16 s2 registered_20261028T233546Z-bd2adcc0, D16 s87358 registered_20261028T194717Z-4c5046a3, D32 s1 registered_20261028T220605Z-6782554f, D32 s2 registered_20261028T234532Z-7cecb989, D32 s87358 registered_20261028T200603Z-dcae4f29, B1D4 s87358 registered_20261028T223314Z-2b19eb09, B4D1 s87358 registered_20261028T224048Z-51df69f2, B4D32 s87358 registered_20261028T210740Z-a2ff90f8, B32D4 s87358 registered_20261028T205421Z-2cdac4b3

### cabinet (extension registered_20261028T174007Z-39a5d8ec; audit safe agreement 1.0000, cost max rel 9.4e-06)

| pairs D | actions | mean safe actions | sources w/o safe action | best cos to optimal direction |
|---|---|---|---|---|
| 1 | 8 | 4.8 | 4334 (1.1%) | 0.304 |
| 2 | 16 | 9.6 | 1262 (0.3%) | 0.450 |
| 4 | 32 | 19.1 | 0 (0.0%) | 0.579 |
| 8 | 64 | 38.3 | 0 (0.0%) | 0.680 |
| 16 | 128 | 76.5 | 0 (0.0%) | 0.757 |
| 32 | 256 | 153.1 | 0 (0.0%) | 0.817 |

| D | seeds | RK4 /600 per seed | RK4 mean | MPPI /1800 per seed | MPPI mean | MPPI gained/lost vs D4 (sum over seeds) | checkpoint = baseline? |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 240/244/234 | 239.3 | 710/705/713 | 709.3 | +167/−118 |  |
| 2 | 3 | 234/230/239 | 234.3 | 711/689/702 | 700.7 | +110/−87 |  |
| 4 | 3 | 229/226/236 | 230.3 | 688/693/698 | 693.0 |  | yes, NO, NO |
| 8 | 3 | 231/227/231 | 229.7 | 680/706/679 | 688.3 | +96/−110 |  |
| 16 | 3 | 217/230/224 | 223.7 | 681/686/676 | 681.0 | +101/−137 |  |
| 32 | 3 | 222/229/224 | 225.0 | 674/692/675 | 680.3 | +129/−167 |  |

MPPI successes per task (mean over training seeds, of 300 = 100 queries x 3 planning seeds); RK4 per task of 100 in brackets

| D | anywhere | compartment_to_compartment | compartment_to_free | free_to_compartment | free_to_free | same_compartment |
|---|---|---|---|---|---|---|
| 1 | 267 (89) | 6 (3) | 65 (25) | 79 (26) | 276 (92) | 16 (5) |
| 2 | 264 (89) | 3 (1) | 65 (26) | 74 (23) | 271 (90) | 23 (6) |
| 4 | 262 (88) | 2 (2) | 67 (22) | 69 (23) | 268 (90) | 25 (7) |
| 8 | 260 (88) | 3 (2) | 63 (22) | 68 (23) | 270 (89) | 23 (6) |
| 16 | 258 (87) | 3 (2) | 59 (19) | 63 (19) | 270 (88) | 28 (8) |
| 32 | 257 (86) | 3 (1) | 55 (19) | 68 (22) | 264 (88) | 33 (9) |

Accuracy against ground truth (truth.py optimized paths), mean over training seeds. rel err = mean |log T/T*|; bias = median T/T*; speed err = mean |log S|grad T|| (implied speed 1/|grad T| vs the speed net); short/mid/long gap = 0-0.24 / 0.24-0.8 / 0.8-1.6

| D | rel err short/mid/long | bias short/mid/long | speed err short/mid/long | cosine to optimal dir short/mid/long |
|---|---|---|---|---|
| 1 | 0.216/0.303/0.371 | 0.885/0.848/0.678 | 0.295/0.196/0.173 | 0.923/0.742/0.681 |
| 2 | 0.239/0.313/0.376 | 0.865/0.832/0.674 | 0.309/0.196/0.170 | 0.923/0.741/0.684 |
| 4 | 0.243/0.317/0.376 | 0.851/0.828/0.674 | 0.313/0.196/0.169 | 0.921/0.737/0.684 |
| 8 | 0.256/0.326/0.381 | 0.845/0.817/0.671 | 0.323/0.198/0.170 | 0.921/0.737/0.685 |
| 16 | 0.267/0.330/0.381 | 0.836/0.812/0.671 | 0.334/0.200/0.171 | 0.920/0.737/0.687 |
| 32 | 0.277/0.334/0.381 | 0.826/0.808/0.671 | 0.346/0.204/0.177 | 0.919/0.735/0.684 |

On 16,384 bank sources with random goals (mean over training seeds; diag2.py post-hoc run when given, else diag.py). speed ratio = S(q)|grad_q T| (1 = implied speed exact, >1 field predicts slower than S); backup ratio = T / min over D' pairs of [c + T(z)] (self-consistency with a min over D' directions); confident = rows the direction term uses (gap >= 10 %)

| D | speed ratio median | speed |log| mean | speed within 5 % | S|grad T| at S<0.3 | backup ratio own D | backup ratio D'=4 | backup ratio D'=32 | winner cosine own D | confident own D |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.028 | 0.181 | 23.5 % | 0.946 | 0.9903 | 0.9944 | 0.9980 | 0.327 | 0.887 |
| 2 | 1.021 | 0.179 | 23.9 % | 0.919 | 0.9922 | 0.9942 | 0.9979 | 0.454 | 0.767 |
| 4 | 1.028 | 0.178 | 23.8 % | 0.926 | 0.9942 | 0.9942 | 0.9979 | 0.576 | 0.671 |
| 8 | 1.023 | 0.179 | 23.9 % | 0.923 | 0.9957 | 0.9941 | 0.9979 | 0.673 | 0.548 |
| 16 | 1.028 | 0.182 | 23.3 % | 0.919 | 0.9970 | 0.9941 | 0.9979 | 0.748 | 0.416 |
| 32 | 1.032 | 0.186 | 22.5 % | 0.928 | 0.9979 | 0.9941 | 0.9979 | 0.803 | 0.278 |

Mechanism control: Bellman min over Db pairs, direction term (winner, runner-up, confidence gate) over Dd pairs; MPPI gained/lost vs D4 of the same seed; failures summed over RK4 / MPPI

| Db | Dd | seed | RK4 /600 | MPPI /1800 | MPPI gained/lost vs D4 | RK4 failures | MPPI failures |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 87358 | 240 | 710 | +59/−37 | collision 714, endpoint 10 | collision 2180 |
| 1 | 4 | 87358 | 229 | 692 | +15/−11 | collision 736, endpoint 8 | collision 2216 |
| 4 | 1 | 87358 | 238 | 727 | +67/−28 | collision 716, endpoint 14 | collision 2146 |
| 4 | 4 | 87358 | 229 | 688 |  | collision 734, endpoint 12 | collision 2224 |
| 4 | 32 | 87358 | 213 | 662 | +32/−58 | collision 768, endpoint 8 | collision 2276 |
| 32 | 4 | 87358 | 229 | 689 | +15/−14 | collision 736, endpoint 8 | collision 2222 |
| 32 | 32 | 87358 | 222 | 674 | +46/−60 | collision 750, endpoint 8 | collision 2252 |
| 1 | 1 | 1 | 244 | 705 | +53/−41 | collision 708, endpoint 6 | collision 2190 |
| 4 | 1 | 1 | 241 | 704 | +49/−38 | collision 714, endpoint 8 | collision 2192 |
| 4 | 4 | 1 | 226 | 693 |  | collision 746, endpoint 10 | collision 2214 |
| 32 | 32 | 1 | 229 | 692 | +46/−47 | collision 740, endpoint 6 | collision 2216 |
| 1 | 1 | 2 | 234 | 713 | +55/−40 | collision 728, endpoint 6 | collision 2174 |
| 4 | 1 | 2 | 235 | 712 | +56/−42 | collision 726, endpoint 6 | collision 2176 |
| 4 | 4 | 2 | 236 | 698 |  | collision 722, endpoint 8 | collision 2204 |
| 32 | 32 | 2 | 224 | 675 | +37/−60 | collision 748, endpoint 6 | collision 2250 |

Direction-term mechanism (post-hoc diag2, training seed 87358). confident = share of sources the direction term uses with its own Dd; winner cos = cos(-grad T, winner step) at Dd; toward clearance = mean cos(-grad T, grad S) by speed band (S < 0.3 is next to obstacles)

| field | Db | Dd | RK4 | MPPI | confident | winner cos | toward clearance S 0-0.3 / 0.3-0.6 / 0.6-0.9 | S|grad T| at S<0.3 |
|---|---|---|---|---|---|---|---|---|
| B4D1 | 4 | 1 | 238 | 727 | 0.889 | 0.327 | 0.618 / 0.629 / 0.205 | 0.947 |
| D2 | 2 | 2 | 234 | 711 | 0.771 | 0.454 | 0.620 / 0.624 / 0.198 | 0.914 |
| D1 | 1 | 1 | 240 | 710 | 0.885 | 0.327 | 0.617 / 0.620 / 0.196 | 0.958 |
| B1D4 | 1 | 4 | 229 | 692 | 0.668 | 0.575 | 0.621 / 0.607 / 0.175 | 0.928 |
| B32D4 | 32 | 4 | 229 | 689 | 0.669 | 0.574 | 0.622 / 0.610 / 0.179 | 0.920 |
| D4 | 4 | 4 | 229 | 688 | 0.670 | 0.575 | 0.620 / 0.610 / 0.180 | 0.917 |
| D16 | 16 | 16 | 217 | 681 | 0.415 | 0.747 | 0.614 / 0.596 / 0.165 | 0.923 |
| D8 | 8 | 8 | 231 | 680 | 0.545 | 0.672 | 0.615 / 0.596 / 0.162 | 0.924 |
| D32 | 32 | 32 | 222 | 674 | 0.277 | 0.803 | 0.608 / 0.593 / 0.164 | 0.949 |
| B4D32 | 4 | 32 | 213 | 662 | 0.277 | 0.803 | 0.605 / 0.584 / 0.166 | 0.925 |

Runs: D1 s1 registered_20261028T214052Z-06395556, D1 s2 registered_20261028T232922Z-2e3e879c, D1 s87358 registered_20261028T193015Z-870e5fc3, D2 s1 registered_20261028T225454Z-d7669d7f, D2 s2 registered_20261029T000641Z-a92adc46, D2 s87358 registered_20261028T203316Z-89b03865, D4 s1 registered_20261029T002745Z-73a9e47a, D4 s2 registered_20261029T003403Z-dcf42c06, D4 s87358 registered_20261028T183645Z-c99380d4, D8 s1 registered_20261028T230142Z-77caa4c3, D8 s2 registered_20261029T001711Z-7220f27d, D8 s87358 registered_20261028T204712Z-c2cac0a6, D16 s1 registered_20261028T215649Z-670a041e, D16 s2 registered_20261028T234311Z-bb103382, D16 s87358 registered_20261028T195647Z-66cdea84, D32 s1 registered_20261028T221410Z-af960cd5, D32 s2 registered_20261028T235907Z-735bacdd, D32 s87358 registered_20261028T201926Z-2851790a, B1D4 s87358 registered_20261028T221924Z-c94f6c3b, B4D1 s1 registered_20261028T230855Z-d994f5b9, B4D1 s2 registered_20261028T231549Z-5226da49, B4D1 s87358 registered_20261028T222617Z-694be39a, B4D32 s87358 registered_20261028T212849Z-797e170a, B32D4 s87358 registered_20261028T212100Z-909b2a74

Post-hoc diagnostics: no-wall registered_20261029T004520Z-4534e09d, cabinet registered_20261029T004059Z-d0eff0ed (diag2.py).
"checkpoint = baseline?": D = 4 is compared tensor for tensor with the original method's run of the same seed; "NO" for seeds 1/2 only because those
reference runs used the other GPU type (5070 vs 3090); same-GPU-type runs are identical (seed 87358, both scenes). Regenerate with
`python3 jeffrey/direction_sweep/report.py --ext nowall=registered_20261028T183647Z-c06520bb cabinet=registered_20261028T174007Z-39a5d8ec
--posthoc nowall=registered_20261029T004520Z-4534e09d cabinet=registered_20261029T004059Z-d0eff0ed`.

## Findings

1. **Why the Bellman min's direction count does not matter.** The winner is almost always the smallest step (mean
   winner step 0.010–0.017 rad for every D; steps are 0.01/0.05/0.15/0.25). The backup is then a local slope
   condition: T(q) ≤ 0.01/S + T(q + 0.01 d). Every trained field sits 0.2–1.2 % below its backup. The Bellman term
   ends training at 6 × 0.0003, against 2 × 0.07–0.13 for the eikonal term and 2 × 0.18–0.66 for the direction term.
   Earlier ablations (jeffrey/segment_dp) found that removing the Bellman term entirely (0/2/2) leaves both
   benchmarks unchanged (cabinet 690 vs 694, no-wall 1588 vs 1590). A better min in a near-inert term cannot move the
   result. Measured directly, every field is equally consistent with the dense 32-pair backup (T/B ≈ 0.998), whatever
   D it was trained with.
2. **What more directions do to the field.** The fixed point of a min over D directions with best cosine c has slope
   about 1/(S c): sparse directions zig-zag. More directions remove that inflation, so the field drops at short range.
   Because the field already under-reads short pairs, removing the inflation moves it away from the truth. This is
   small on no-wall (short rel err 0.183 → 0.193) and clear on the cabinet (0.216 → 0.277, monotone in D).
3. **The direction term is the channel, and it has two opposing dependencies on D.** More directions make the winner
   a better descent direction (its cosine with −grad T is 0.32 / 0.57 / 0.80 at D = 1 / 4 / 32). They also make the
   10 % confidence gate pass less often: cabinet 89 % / 67 % / 28 % of rows, no-wall 75 % / 73 % / 39 %. With many
   directions the winner nearly copies the field's own gradient on fewer rows, so the term carries little
   information. With few, it acts on most rows but points in a crude direction.
4. **No-wall: the direction term carries the tilt manifold, and D = 4 is its optimum.** The losses on both sides are
   tilt failures: RK4 tilt 25 at D = 4, 59 at D = 1, 50 at D = 32, 55 / 44 for the direction-only controls. This
   matches the earlier ablation, where the direction term was critical on no-wall. The method's 6/2/2 weights were
   developed at D = 4, which may be why the optimum sits there.
5. **Cabinet: fewer directions help, through the direction term.** Only safe actions can win the min, and few
   sampled directions near the cupboard give few safe ones. A direction term with 1 pair passes the gate on 89 % of
   rows and pulls the descent direction toward clearance. At moderate clearance (speed 0.6–0.9), mean
   cos(−grad T, grad S) orders exactly with MPPI: B4D1 0.205 (727), D1 0.196 (710), D4 0.180 (688), D32 0.164 (674),
   B4D32 0.166 (662). Planning changes are collisions only (cabinet failures are all collisions). Refuted alternative:
   inflation near walls. The D = 1 Bellman min gives the steepest near-wall slope (S|grad T| 0.958 vs 0.917 at
   S < 0.3), but the Bellman-only control B1D4 plans like D = 4 (692), and D32 is also steep there (0.949) yet plans
   worst.
6. **Size of the effects against noise.** The original method's three seeds span 688–698 (cabinet MPPI) and 1583–1598
   (no-wall MPPI); the same seed on a different GPU type (5070 vs 3090) moves results by ≤ 4. Cabinet D = 1 (705–713)
   and B4D1 (704–727) are above every D = 4 seed; cabinet D = 16 / 32 (674–692) are mostly below. No-wall D = 1
   (1555–1562) and every no-wall D ≥ 8 RK4 (522–534 vs 535–550) are below every D = 4 seed. No-wall MPPI at D ≥ 8 is
   −5 to −8 on average, inside the seed spread.
7. **Answer.** Random walks in more directions make the Bellman min itself more accurate, but this method barely uses
   the Bellman target. Neither field accuracy (travel time, implied speed) nor success improves. The direction count
   matters only through the direction term. There, 4 pairs is best on no-wall, and 1 pair is better on the cabinet
   (+20 MPPI, three seeds). A practical follow-up, which needs approval because it changes the method: decouple the
   two uses of the bank (Bellman over any D, direction term over Dd tuned per scene), or replace the 10 % confidence
   gate so that many-direction winners are not discarded.

## Caveats

- The source poses are fixed: the rollforward that produced them chose among the frozen 4 pairs. A bank generated
  from scratch with D pairs would also change where the sources lie.
- Training seeds 87358/1/2 only; the original method's seed-to-seed spread is ±6 MPPI on the cabinet (688–698) and
  ±7 on no-wall (1585–1598).
- Most cabinet runs on the 3090 shared the GPU with other sessions' jobs (slower, same results: training is
  deterministic per GPU type; the 5070 and 3090 runs are mixed across D values).
- Environment: Docker `bellman600-env:local` (not the pinned image), recorded in every run's environment_override.json.

## Files

`extend.py` (extension bank), `train.py` / `train2.py` (training, controls), `diag.py` / `diag2.py` (diagnostics),
`pipeline.py` / `pipeline2.py` (stages), `run.py` / `run2.py` / `posthoc.py` (registry entries), `report.py` (tables).
