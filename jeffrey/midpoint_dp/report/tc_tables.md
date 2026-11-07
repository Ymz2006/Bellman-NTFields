## 13×13: success

MPPI success, planning seeds 0/1/2 pooled, per training seed.

| field | seed 87358 | seed 1 | seed 2 | all | % |
|---|---|---|---|---|---|
| control | 334/600 | 337/600 | 408/600 | 1079/1800 | 60 |
| plain, net leaves, tree | 446/600 |  |  | 446/600 | 74 |
| safe, min leaves, tree + sub-paths | 536/600 | 559/600 | 543/600 | 1638/1800 | 91 |
| Tc (a): d 0.02, 200k pairs | 419/600 |  |  | 419/600 | 70 |
| Tc (a): d 0.05, 1M pairs | 556/600 | 558/600 | 547/600 | 1661/1800 | 92 |
| Tc (a): d 0.05, 200k pairs | 488/600 | 545/600 | 516/600 | 1549/1800 | 86 |
| Tc (a): d 0.05, 200k pairs, no truth in loop, plain | 418/600 |  |  | 418/600 | 70 |
| Tc (a): d 0.05, 30k pairs | 502/600 |  |  | 502/600 | 84 |
| Tc (a): d 0.1, 200k pairs | 424/600 |  |  | 424/600 | 71 |
| Tc (a): d 0.2, 200k pairs | 354/600 |  |  | 354/600 | 59 |
| Tc (b): epoch 1000, tau 1, 200k pairs | 553/600 | 523/600 | 533/600 | 1609/1800 | 89 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 529/600 |  |  | 529/600 | 88 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 537/600 | 532/600 | 551/600 | 1620/1800 | 90 |
| Tc (b): epoch 2000, tau 1, 200k pairs, direct leaves | 517/600 |  |  | 517/600 | 86 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 512/600 | 522/600 | 516/600 | 1550/1800 | 86 |
| Tc (b): epoch 3000, tau 1, 200k pairs | 541/600 |  |  | 541/600 | 90 |

## 13×13: field, Tc and ground-truth use (training seed 87358)

T/T*: median learned over true travel time per distance bin. Tc: pairs it was trained on, exact-clearance queries spent on them (sampling filter + labels) and its held-out |log error|. Loop queries: exact-clearance queries during the 30 refreshes (traversal endpoints, wall-avoiding MPPI moves, collision cut); not counted in runs before this test. Leaf error: median |log(Tc time / exact time)| over the leaves of the diagnostic refreshes (the exact time is computed for this column only); target bias: median log(target / target with exact times) at the last diagnostic refresh.

| field | success | T/T* 0.5-2 / 2-4 / 4-8 / 8-16 / 16-32 | Tc pairs | Tc queries (M) | Tc held-out err | loop queries (M) | leaf err | target bias | train min |
|---|---|---|---|---|---|---|---|---|---|
| control | 334/600 | 1.13 / 1.19 / 1.26 / 1.92 / 2.05 |  |  |  |  |  |  | 9 |
| plain, net leaves, tree | 446/600 | 1.37 / 1.41 / 1.45 / 1.61 / 1.67 |  |  |  |  |  |  | 20 |
| safe, min leaves, tree + sub-paths | 536/600 | 1.05 / 1.03 / 1.06 / 1.06 / 1.08 |  |  |  |  |  |  | 42 |
| Tc (a): d 0.02, 200k pairs | 419/600 | 1.06 / 1.05 / 1.10 / 1.10 / 1.13 | 200000 | 1.4 | 0.071 | 3048 | 0.003 | 0.002 | 22 |
| Tc (a): d 0.05, 1M pairs | 556/600 | 1.02 / 1.03 / 1.04 / 1.05 / 1.06 | 1000000 | 8.8 | 0.021 | 3059 | 0.003 | 0.001 | 23 |
| Tc (a): d 0.05, 200k pairs | 488/600 | 1.03 / 1.04 / 1.05 / 1.05 / 1.07 | 200000 | 1.7 | 0.035 | 3046 | 0.003 | 0.001 | 23 |
| Tc (a): d 0.05, 200k pairs, no truth in loop, plain | 418/600 | 1.05 / 1.03 / 1.04 / 1.04 / 1.03 | 200000 | 1.7 | 0.035 | 0 | 0.004 | 0.000 | 21 |
| Tc (a): d 0.05, 30k pairs | 502/600 | 1.05 / 1.02 / 1.05 / 1.05 / 1.05 | 30000 | 0.5 | 0.120 | 3048 | 0.019 | 0.003 | 23 |
| Tc (a): d 0.1, 200k pairs | 424/600 | 1.01 / 1.00 / 1.02 / 1.02 / 1.02 | 200000 | 2.8 | 0.025 | 3004 | 0.006 | -0.005 | 22 |
| Tc (a): d 0.2, 200k pairs | 354/600 | 1.02 / 0.98 / 1.00 / 0.99 / 1.00 | 200000 | 5.8 | 0.000 | 3012 | 0.008 | -0.016 | 22 |
| Tc (b): epoch 1000, tau 1, 200k pairs | 553/600 | 1.05 / 1.04 / 1.07 / 1.07 / 1.08 | 200000 | 8.2 | 0.205 | 3064 | 0.025 | 0.002 | 31 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 529/600 | 1.03 / 1.03 / 1.05 / 1.07 / 1.06 | 200000 | 6.2 | 0.147 | 3052 | 0.008 | 0.002 | 22 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 537/600 | 1.02 / 1.04 / 1.07 / 1.08 / 1.08 | 200000 | 8.2 | 0.221 | 3065 | 0.010 | 0.004 | 28 |
| Tc (b): epoch 2000, tau 1, 200k pairs, direct leaves | 517/600 | 1.03 / 1.04 / 1.04 / 1.05 / 1.06 | 200000 | 8.2 | 0.215 | 3066 | 0.012 | -0.004 | 29 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 512/600 | 1.04 / 1.03 / 1.04 / 1.05 / 1.07 | 200000 | 8.2 | 0.215 | 0 | 0.008 | 0.004 | 26 |
| Tc (b): epoch 3000, tau 1, 200k pairs | 541/600 | 1.06 / 1.03 / 1.07 / 1.08 / 1.08 | 200000 | 8.2 | 0.218 | 3065 | 0.009 | 0.004 | 30 |

## 13×13: near-wall descent (training seed 87358)

Share of near-wall poses whose descent direction points into the wall, by the goal's true distance (%).

| field | 0.5-2 | 2-8 | >8 |
|---|---|---|---|
| control | 4.2 | 3.5 | 3.9 |
| plain, net leaves, tree | 5.2 | 3.2 | 3.4 |
| safe, min leaves, tree + sub-paths | 2.8 | 1.8 | 2.1 |
| Tc (a): d 0.02, 200k pairs | 4.2 | 3.3 | 3.9 |
| Tc (a): d 0.05, 1M pairs | 2.3 | 1.9 | 2.0 |
| Tc (a): d 0.05, 200k pairs | 3.4 | 2.8 | 1.8 |
| Tc (a): d 0.05, 200k pairs, no truth in loop, plain | 3.2 | 1.9 | 2.1 |
| Tc (a): d 0.05, 30k pairs | 3.0 | 2.2 | 2.0 |
| Tc (a): d 0.1, 200k pairs | 2.9 | 2.3 | 2.5 |
| Tc (a): d 0.2, 200k pairs | 6.3 | 4.6 | 2.2 |
| Tc (b): epoch 1000, tau 1, 200k pairs | 2.5 | 1.8 | 1.9 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 3.1 | 1.6 | 1.7 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 2.5 | 2.0 | 2.1 |
| Tc (b): epoch 2000, tau 1, 200k pairs, direct leaves | 2.4 | 2.1 | 1.9 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 2.8 | 2.0 | 1.8 |
| Tc (b): epoch 3000, tau 1, 200k pairs | 2.5 | 1.8 | 2.1 |

## 22×22: success

MPPI success, planning seeds 0/1/2 pooled, per training seed.

| field | seed 87358 | seed 1 | seed 2 | all | % |
|---|---|---|---|---|---|
| control | 172/720 | 174/720 | 165/720 | 511/2160 | 24 |
| safe, min leaves, tree + sub-paths | 282/720 | 281/720 | 257/720 | 820/2160 | 38 |
| Tc (a): d 0.05, 1M pairs | 225/720 |  |  | 225/720 | 31 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 248/720 |  |  | 248/720 | 34 |
| Tc (b): epoch 2000, tau 1, 1M pairs | 209/720 |  |  | 209/720 | 29 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 137/720 | 251/720 | 229/720 | 617/2160 | 29 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 160/720 | 181/720 | 188/720 | 529/2160 | 24 |
| Tc (b): epoch 5000, tau 1, 200k pairs | 181/720 |  |  | 181/720 | 25 |

## 22×22: field, Tc and ground-truth use (training seed 87358)

T/T*: median learned over true travel time per distance bin. Tc: pairs it was trained on, exact-clearance queries spent on them (sampling filter + labels) and its held-out |log error|. Loop queries: exact-clearance queries during the 30 refreshes (traversal endpoints, wall-avoiding MPPI moves, collision cut); not counted in runs before this test. Leaf error: median |log(Tc time / exact time)| over the leaves of the diagnostic refreshes (the exact time is computed for this column only); target bias: median log(target / target with exact times) at the last diagnostic refresh.

| field | success | T/T* 0.5-2 / 2-4 / 4-8 / 8-16 / 16-32 | Tc pairs | Tc queries (M) | Tc held-out err | loop queries (M) | leaf err | target bias | train min |
|---|---|---|---|---|---|---|---|---|---|
| control | 172/720 | 1.20 / 1.33 / 1.35 / 1.38 / 1.32 |  |  |  |  |  |  | 4 |
| safe, min leaves, tree + sub-paths | 282/720 | 1.16 / 1.10 / 1.08 / 1.14 / 1.12 |  |  |  |  |  |  | 52 |
| Tc (a): d 0.05, 1M pairs | 225/720 | 1.07 / 1.12 / 1.12 / 1.11 / 1.11 | 1000000 | 8.8 | 0.058 | 3407 | 0.039 | 0.001 | 26 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 248/720 | 1.13 / 1.10 / 1.11 / 1.18 / 1.19 | 200000 | 6.3 | 0.431 | 3350 | 1.194 | 0.000 | 24 |
| Tc (b): epoch 2000, tau 1, 1M pairs | 209/720 | 1.22 / 1.18 / 1.16 / 1.24 / 1.25 | 1000000 | 41.0 | 0.334 | 3279 | 0.638 | 0.000 | 24 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 137/720 | 1.23 / 1.14 / 1.15 / 1.24 / 1.22 | 200000 | 8.2 | 0.517 | 3352 | 1.206 | 0.029 | 25 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 160/720 | 1.05 / 1.05 / 1.06 / 1.19 / 1.23 | 200000 | 8.2 | 0.527 | 0 | 0.108 | 0.000 | 24 |
| Tc (b): epoch 5000, tau 1, 200k pairs | 181/720 | 1.15 / 1.15 / 1.16 / 1.19 / 1.17 | 200000 | 8.2 | 0.530 | 3432 | 1.143 | 0.018 | 24 |

## 22×22: near-wall descent (training seed 87358)

Share of near-wall poses whose descent direction points into the wall, by the goal's true distance (%).

| field | 0.5-2 | 2-8 | >8 |
|---|---|---|---|
| control | 15.3 | 16.1 | 12.1 |
| safe, min leaves, tree + sub-paths | 9.4 | 7.2 | 6.2 |
| Tc (a): d 0.05, 1M pairs | 12.8 | 9.8 | 7.4 |
| Tc (b): epoch 2000, tau 0.5, 200k pairs | 12.0 | 9.5 | 6.5 |
| Tc (b): epoch 2000, tau 1, 1M pairs | 13.3 | 12.1 | 10.0 |
| Tc (b): epoch 2000, tau 1, 200k pairs | 17.0 | 20.6 | 16.7 |
| Tc (b): epoch 2000, tau 1, 200k pairs, no truth in loop, plain | 16.7 | 16.0 | 13.3 |
| Tc (b): epoch 5000, tau 1, 200k pairs | 13.3 | 12.1 | 10.1 |

