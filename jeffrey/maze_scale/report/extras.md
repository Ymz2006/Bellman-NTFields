**Fewer networks (13 × 13, segment DP + wall term weight 0.3, same ntrl-demo base per seed)**

| networks | edge ratio | runs solved per seed (87358 / 1 / 2, of 600) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells | T/T* 16–32 | minutes per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | – | 347 / 320 / 393 | 59 % (1060/1800) | 94 | 78 | 59 | 44 | 19 | 2.08 | – |
| 6 | 2.000 | 346 / 354 / 338 | 58 % (1038/1800) | 100 | 94 | 77 | 17 | 1 | 1.59 | 8 / 8 / 10 |
| 8 | 1.641 | 399 / 309 / 366 | 60 % (1074/1800) | 99 | 96 | 77 | 22 | 4 | 1.25 | 25 / 14 / 17 |
| 10 | 1.470 | 420 / 418 / 399 | 69 % (1237/1800) | 100 | 97 | 85 | 44 | 19 | 1.22 | 19 / 20 / 19 |
| 13 | 1.350 | 451 / 456 / 420 | 74 % (1327/1800) | 100 | 97 | 88 | 52 | 31 | 1.16 | 19 / 19 / 20 |

**Fourier scale (ntrl-demo with B × s, same seed draw; s = 1 is plain ntrl-demo)**

| maze | B scale | runs solved per seed (87358 / 1 / 2) | all | speed fit |
|---|---|---|---|---|
| 7×7 | 1 | 480 / 480 / 477 (of 480) | 100 % (1437/1440) | 0.91 |
| 7×7 | 1.5 | 477 / 480 / 477 (of 480) | 100 % (1434/1440) | 0.93 |
| 7×7 | 3 | 391 / 345 / 381 (of 480) | 78 % (1117/1440) | 0.96 |
| 10×10 | 1 | 500 / 523 / 560 (of 600) | 88 % (1583/1800) | 0.81 |
| 10×10 | 1.5 | 513 (of 600) | 86 % (513/600) | 0.85 |
| 13×13 | 1 | 347 / 320 / 393 (of 600) | 59 % (1060/1800) | 0.69 |
| 13×13 | 1.5 | 441 / 450 / 473 (of 600) | 76 % (1364/1800) | 0.75 |
| 13×13 | 3 | 554 / 529 / 537 (of 600) | 90 % (1620/1800) | 0.84 |
| 13×13 | 6 | 548 (of 600) | 91 % (548/600) | 0.89 |
| 22×22 | 1 | 152 / 186 / 167 (of 720) | 23 % (505/2160) | 0.32 |
| 22×22 | 1.5 | 159 / 179 / 175 (of 720) | 24 % (513/2160) | 0.39 |
| 22×22 | 3 | 212 / 175 / 201 (of 720) | 27 % (588/2160) | 0.52 |
| 22×22 | 6 | 302 (of 720) | 42 % (302/720) | 0.68 |

**U-turns around a wall end (40 queries per maze × 3 planning seeds per training seed)**

| maze | field | runs solved per seed (87358 / 1 / 2, of 120) | all |
|---|---|---|---|
| 4×4 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 120 / 120 / 120 | 100 % (360/360) |
| | segment DP | 120 | 100 % (120/120) |
| 7×7 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 118 / 120 / 119 | 99 % (357/360) |
| | segment DP | 116 | 97 % (116/120) |
| | segment DP + wall term | 115 | 96 % (115/120) |
| | segment DP + wall term (weight 0.3) | 120 | 100 % (120/120) |
| 10×10 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| 13×13 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 24 / 39 / 45 | 30 % (108/360) |
| | segment DP | 13 / 18 / 9 | 11 % (40/360) |
| | segment DP + wall term | 73 / 90 / 68 | 64 % (231/360) |
| | segment DP + wall term (weight 0.3) | 99 / 89 / 96 | 79 % (284/360) |
| | segment DP + wall term (weight 0.3), 5 levels (ratio 2.000) | 78 / 85 / 85 | 69 % (248/360) |
| | segment DP + wall term (weight 0.3), 7 levels (ratio 1.641) | 102 / 96 | 82 % (198/240) |
| | ntrl-demo, B × 1.5 | 62 | 52 % (62/120) |
| 22×22 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 4 / 3 / 3 | 3 % (10/360) |
| | segment DP | 0 | 0 % (0/120) |
| | segment DP, checked pieces | 0 | 0 % (0/120) |
| | segment DP + wall term | 44 | 37 % (44/120) |
| | segment DP + wall term (weight 0.3) | 35 | 29 % (35/120) |

