**Fewer networks (13 × 13, segment DP + wall term weight 0.3, same ntrl-demo base per seed)**

| networks | edge ratio | runs solved per seed (87358 / 1 / 2, of 600) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells | T/T* 16–32 | minutes per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | – | 347 / 320 / 393 | 59 % (1060/1800) | 94 | 78 | 59 | 44 | 19 | 2.08 | – |
| 6 | 2.000 | 346 / 354 / 338 | 58 % (1038/1800) | 100 | 94 | 77 | 17 | 1 | 1.59 | 8 / 8 / 10 |
| 8 | 1.641 | 399 / 309 / 366 | 60 % (1074/1800) | 99 | 96 | 77 | 22 | 4 | 1.25 | 25 / 14 / 17 |
| 10 | 1.470 | 420 / 418 / 399 | 69 % (1237/1800) | 100 | 97 | 85 | 44 | 19 | 1.22 | 19 / 20 / 19 |
| 13 | 1.350 | 451 / 456 / 420 | 74 % (1327/1800) | 100 | 97 | 88 | 52 | 31 | 1.16 | 19 / 19 / 20 |

**Split candidates per DP pair (13 × 13, segment DP + wall term weight 0.3, 13 networks)**

| candidates | runs solved per seed (87358 / 1 / 2, of 600) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells | top-level target / T* | target minutes | minutes per run |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 440 / 400 / 397 | 69 % (1237/1800) | 99 | 97 | 85 | 43 | 19 | 1.222 | 5 | 35 / 18 / 19 |
| 128 | 451 / 456 / 420 | 74 % (1327/1800) | 100 | 97 | 88 | 52 | 31 | 1.124 | 6 | 19 / 19 / 20 |
| 512 | 465 / 461 / 442 | 76 % (1368/1800) | 100 | 98 | 89 | 56 | 38 | 1.081 | 21 | 41 / 55 / 30 |
| 2048 | 453 | 76 % (453/600) | 100 | 98 | 90 | 51 | 39 | 1.037 | 58 | 71 |

**Segment DP + wall term (weight 0.3) on the B × 6 base, 22 × 22**

| field | runs solved per seed (87358 / 1 / 2) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells | 32-64 cells |
|---|---|---|---|---|---|---|---|---|
| ntrl-demo | 152 / 186 / 167 (of 720) | 23 % (505/2160) | 75 | 32 | 16 | 12 | 5 | 0 |
| ntrl-demo, B × 6 | 302 / 229 / 395 (of 720) | 43 % (926/2160) | 86 | 64 | 42 | 35 | 22 | 8 |
| segment DP + wall term (weight 0.3) | 308 (of 720) | 43 % (308/720) | 90 | 70 | 57 | 30 | 10 | 0 |
| segment DP + wall term (weight 0.3) on the B × 6 base | 263 / 256 / 282 (of 720) | 37 % (801/2160) | 98 | 82 | 35 | 6 | 1 | 0 |

**Segment DP + wall term (weight 0.3) on the B × 3 base, 13 × 13**

| field | runs solved per seed (87358 / 1 / 2) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells |
|---|---|---|---|---|---|---|---|
| ntrl-demo | 347 / 320 / 393 (of 600) | 59 % (1060/1800) | 94 | 78 | 59 | 44 | 19 |
| ntrl-demo, B × 3 | 554 / 529 / 537 (of 600) | 90 % (1620/1800) | 99 | 97 | 89 | 91 | 75 |
| segment DP + wall term (weight 0.3) | 451 / 456 / 420 (of 600) | 74 % (1327/1800) | 100 | 97 | 88 | 52 | 31 |
| segment DP + wall term (weight 0.3) on the B × 3 base | 394 / 378 / 416 (of 600) | 66 % (1188/1800) | 100 | 95 | 79 | 39 | 17 |

**Plus maze: 22 × 22 with a 6-cell + opened (228 of 484 cells, 168 walls removed; prepare run 20261102T042454Z-62864e61)**

| field | runs solved per seed (87358 / 1 / 2) | all | 0.5-2 cells | 2-4 cells | 4-8 cells | 8-16 cells | 16-32 cells | 32-64 cells | speed fit |
|---|---|---|---|---|---|---|---|---|---|
| oracle on the plus maze | – | – | 100 | 100 | 100 | 100 | 100 | 100 | – |
| straight line on the plus maze | – | – | 95 | 82 | 68 | 38 | 12 | 0 | – |
| ntrl-demo, original 22 × 22 | 152 / 186 / 167 (of 720) | 23 % (505/2160) | 75 | 32 | 16 | 12 | 5 | 0 | 0.32 |
| ntrl-demo, plus maze | 468 / 479 / 490 (of 720) | 67 % (1437/2160) | 100 | 98 | 89 | 71 | 41 | 0 | 0.53 |

Plus maze by where the query ends lie (runs solved): neither in the + 80/441 (18 %, 49 queries), one end 294/648 (45 %, 72 queries), both ends 1063/1071 (99 %, 119 queries)

**Fourier scale (ntrl-demo with B × s, same seed draw; s = 1 is plain ntrl-demo)**

| maze | B scale | runs solved per seed (87358 / 1 / 2) | all | speed fit |
|---|---|---|---|---|
| 7×7 | 1 | 480 / 480 / 477 (of 480) | 100 % (1437/1440) | 0.91 |
| 7×7 | 1.5 | 477 / 480 / 477 (of 480) | 100 % (1434/1440) | 0.93 |
| 7×7 | 3 | 391 / 345 / 381 (of 480) | 78 % (1117/1440) | 0.96 |
| 10×10 | 1 | 500 / 523 / 560 (of 600) | 88 % (1583/1800) | 0.81 |
| 10×10 | 1.5 | 513 / 581 / 585 (of 600) | 93 % (1679/1800) | 0.85 |
| 10×10 | 3 | 592 / 575 / 545 (of 600) | 95 % (1712/1800) | 0.90 |
| 10×10 | 6 | 304 (of 600) | 51 % (304/600) | 0.93 |
| 13×13 | 1 | 347 / 320 / 393 (of 600) | 59 % (1060/1800) | 0.69 |
| 13×13 | 1.5 | 441 / 450 / 473 (of 600) | 76 % (1364/1800) | 0.75 |
| 13×13 | 3 | 554 / 529 / 537 (of 600) | 90 % (1620/1800) | 0.84 |
| 13×13 | 6 | 548 (of 600) | 91 % (548/600) | 0.89 |
| 22×22 | 1 | 152 / 186 / 167 (of 720) | 23 % (505/2160) | 0.32 |
| 22×22 | 1.5 | 159 / 179 / 175 (of 720) | 24 % (513/2160) | 0.39 |
| 22×22 | 3 | 212 / 175 / 201 (of 720) | 27 % (588/2160) | 0.52 |
| 22×22 | 6 | 302 / 229 / 395 (of 720) | 43 % (926/2160) | 0.66 |

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
| | ntrl-demo, B × 1.5 | 120 / 120 / 120 | 100 % (360/360) |
| | ntrl-demo, B × 3 | 120 / 120 / 120 | 100 % (360/360) |
| 10×10 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 87 / 117 / 95 | 83 % (299/360) |
| | segment DP | 63 | 52 % (63/120) |
| | ntrl-demo, B × 1.5 | 113 / 111 / 113 | 94 % (337/360) |
| | segment DP + wall term (weight 0.3) | 113 / 117 / 120 | 97 % (350/360) |
| | ntrl-demo, B × 3 | 113 / 115 / 119 | 96 % (347/360) |
| | ntrl-demo, B × 6 | 105 | 88 % (105/120) |
| 13×13 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 24 / 39 / 45 | 30 % (108/360) |
| | segment DP | 13 / 18 / 9 | 11 % (40/360) |
| | segment DP + wall term | 73 / 90 / 68 | 64 % (231/360) |
| | segment DP + wall term (weight 0.3) | 99 / 89 / 96 | 79 % (284/360) |
| | segment DP + wall term (weight 0.3), 5 levels (ratio 2.000) | 78 / 85 / 85 | 69 % (248/360) |
| | segment DP + wall term (weight 0.3), 7 levels (ratio 1.641) | 102 / 96 / 94 | 81 % (292/360) |
| | ntrl-demo, B × 1.5 | 62 / 72 / 72 | 57 % (206/360) |
| | ntrl-demo, B × 3 | 93 / 112 / 85 | 81 % (290/360) |
| | segment DP + wall term (weight 0.3), 9 levels (ratio 1.470) | 98 / 96 / 83 | 77 % (277/360) |
| | ntrl-demo, B × 6 | 109 | 91 % (109/120) |
| | segment DP + wall term (weight 0.3), 32 split candidates | 89 / 87 / 94 | 75 % (270/360) |
| | segment DP + wall term (weight 0.3), 512 split candidates | 97 / 89 / 99 | 79 % (285/360) |
| | segment DP + wall term (weight 0.3), B × 3 base | 107 / 111 / 104 | 89 % (322/360) |
| | segment DP + wall term (weight 0.3), 2048 split candidates | 99 | 82 % (99/120) |
| 22×22 | oracle (MPPI on T*) | – | 100 % (120/120) |
| | straight line | – | 0 % (0/40) |
| | ntrl-demo | 4 / 3 / 3 | 3 % (10/360) |
| | segment DP | 0 | 0 % (0/120) |
| | segment DP, checked pieces | 0 | 0 % (0/120) |
| | segment DP + wall term | 44 | 37 % (44/120) |
| | segment DP + wall term (weight 0.3) | 35 | 29 % (35/120) |
| | ntrl-demo, B × 1.5 | 0 / 12 / 9 | 6 % (21/360) |
| | ntrl-demo, B × 3 | 5 / 3 / 10 | 5 % (18/360) |
| | ntrl-demo, B × 6 | 32 / 12 / 48 | 26 % (92/360) |
| | segment DP + wall term (weight 0.3), B × 6 base | 91 / 86 / 84 | 72 % (261/360) |

