# pair_lift tables (report.py; every number from registered runs)


## References (jeffrey/lift_scale, same pipeline)

| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | seen through % | S x slope innermost | rollbacks | minutes | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | ntrl-demo | 87358 | 20261117T054519Z-9a3bc556 | 65.0 | 15.1 | 19.9 | 46.3 | 1.104 | 50.6 | 1.37 | 2.8 | 0.08 | 8 | 6.4 | `` |
| plus22 | lift | 87358 | 20261117T055249Z-d953394e | 55.0 | 0.0 | 45.0 | 25.5 | 0.821 | 43.5 | -0.20 | 98.3 | 1.00 | 0 | 13.1 | `--lift time` |
| plus22 | lift + push | 87358 | 20261117T061710Z-c95ca37c | 68.2 | 1.0 | 30.8 | 89.4 | 1.002 | 99.4 | 0.47 | 52.9 | 0.86 | 149 | 14.3 | `--lift time --push page` |
| plus22 | lift + push + octaves | 87358 | 20261117T093110Z-3477083e | 68.1 | 0.0 | 31.9 | 68.4 | 0.963 | 80.2 | -0.02 | 77.0 | 0.83 | 276 | 15.0 | `--lift time --push page --octaves 2,4,8` |
| plus22 | lift + strong push + octaves | 87358 | 20261117T101750Z-b9e07f67 | 83.8 | 13.6 | 2.6 | 16.3 | 1.157 | 21.5 | 1.85 | 0.8 | 0.86 | 353 | 26.3 | `--lift time --push const:1e-2 --octaves 2,4,8` |

## Fields trained without the truth

| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | seen through % | S x slope innermost | rollbacks | minutes | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | bit | 87358 | 20261119T045331Z-3a5d2cfd | 40.7 | 1.8 | 57.5 | 0.0 | 29.530 | 0.0 | 81.75 | 0.0 | 1.08 | 0 | 10.4 | `--lift time --pair bit` |
| plus22 | bit_push | 87358 | 20261119T045311Z-00d33089 | 53.8 | 0.3 | 46.0 | 23.3 | 25.147 | 43.9 | 87.31 | 0.0 | 0.32 | 761 | 22.2 | `--lift time --pair bit --push page` |
| plus22 | ud_push_tri | 87358 | 20261119T050421Z-bfb97974 | 58.2 | 0.3 | 41.5 | 78.1 | 0.974 | 99.5 | 0.09 | 99.5 | 0.85 | 0 | 18.4 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-nograd --push page --tri1 0.5 --tri 10` |
| plus22 | bit_push_tri | 87358 | 20261119T045251Z-ffba52df | 59.2 | 0.0 | 40.8 | 81.0 | 0.978 | 99.7 | 0.16 | 90.9 | 0.85 | 0 | 33.5 | `--lift time --pair bit --push page --tri1 0.5 --tri 10` |
| plus22 | count_f025 | 87358 | 20261119T054033Z-7869b4de | 56.8 | 0.0 | 43.2 | 74.7 | 0.971 | 99.2 | 0.06 | 99.7 | 0.87 | 16 | 17.0 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --push page --tri1 0.5 --tri 10` |
| plus22 | tdfree | 87358 | 20261119T055803Z-84929489 | 49.3 | 0.0 | 50.7 | 8.7 | 0.622 | 15.7 | -0.56 | 99.8 | 0.99 | 0 | 9.5 | `--lift time --td-free` |
| plus22 | bell1 | 87358 | 20261119T054423Z-f493358c | 96.8 | 0.4 | 2.8 | 6.5 | 1.276 | 12.2 | 2.12 | 0.0 | 1.23 | 404 | 47.9 | `--lift time --bell 1 --td-free` |
| plus22 | bell01 | 87358 | 20261119T054443Z-d462e732 | 95.6 | 0.6 | 3.9 | 1.1 | 1.282 | 1.1 | 2.18 | 0.7 | 1.06 | 569 | 48.8 | `--lift time --bell 0.1 --td-free` |
| plus22 | bell1_count | 87358 | 20261119T062503Z-b895cf51 | 61.2 | 0.6 | 38.2 | 50.5 | 1.096 | 73.6 | 1.46 | 1.9 | 1.03 | 67 | 31.9 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --bell 1 --td-free` |
| plus22 | ntrl_bell1 | 87358 | 20261119T063223Z-72f35cfb | 72.6 | 24.6 | 2.8 | 36.1 | 1.170 | 60.8 | 2.22 | 0.0 | 0.08 | 129 | 41.0 | `--lift none --bell 1 --td-free` |
| plus22 | bxt1 | 87358 | 20261119T065723Z-b148dcbc | 97.5 | 0.8 | 1.7 | 84.5 | 1.026 | 95.7 | 1.31 | 0.1 | 0.99 | 483 | 32.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1_r15 | 87358 | 20261119T073003Z-d342af01 | 79.3 | 2.8 | 17.9 | 94.8 | 1.013 | 96.9 | 1.06 | 5.7 | 1.00 | 221 | 31.9 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-r 0.015 --bell-table` |
| plus22 | bxt3 | 87358 | 20261119T071343Z-95f04a0c | 85.7 | 0.0 | 14.3 | 82.4 | 0.974 | 84.3 | 1.12 | 0.2 | 1.01 | 136 | 54.9 | `--lift time --bell 3 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| apt64 | bxt1 | 87358 | 20261119T080213Z-45290212 | 64.4 | 0.0 | 35.6 | 19.9 | 0.699 | 69.6 | -0.17 | 99.9 | 1.00 | 5 | 27.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| plus40 | bxt1 | 87358 | 20261119T073553Z-b75cfe2c | 64.3 | 0.0 | 35.7 | 71.0 | 1.051 | 68.8 | 0.38 | 69.0 | 0.99 | 0 | 58.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| Klickitat | bxt1 | 87358 | 20261119T080904Z-37a261b0 | 67.7 | 0.0 | 32.3 | 33.4 | 0.859 | 43.5 | 0.49 | 51.1 | 0.95 | 0 | 44.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| plus22 | bxt1 | 1 | 20261119T082923Z-d2107e1a | 97.6 | 1.9 | 0.4 | 77.4 | 1.050 | 83.9 | 1.20 | 0.0 | 1.02 | 518 | 33.3 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1 | 2 | 20261119T083413Z-cad0f4e4 | 100.0 | 0.0 | 0.0 | 88.0 | 1.019 | 93.4 | 1.04 | 0.1 | 0.98 | 469 | 56.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1L | 87358 | 20261119T090253Z-f1b7016b | 100.0 | 0.0 | 0.0 | 66.7 | 1.029 | 91.3 | 1.65 | 0.0 | 1.02 | 522 | 46.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus40 | bxt1n | 87358 | 20261119T085333Z-aba5379e | 68.2 | 0.1 | 31.7 | 83.3 | 1.037 | 94.0 | 0.70 | 32.7 | 1.00 | 0 | 57.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm` |
| apt64 | bxt1nL | 87358 | 20261119T094903Z-7dc8cc67 | 76.5 | 0.0 | 23.5 | 34.7 | 0.844 | 76.8 | 0.20 | 84.2 | 1.06 | 1095 | 47.8 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus40 | bxt1nL | 87358 | 20261119T093043Z-7840e590 | 71.5 | 0.2 | 28.2 | 55.2 | 1.079 | 81.7 | 1.92 | 0.0 | 1.06 | 140 | 72.3 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus40 | bxt1b | 87358 | 20261119T095053Z-b953c20d | 68.7 | 0.0 | 31.3 | 15.2 | 1.167 | 20.6 | 1.76 | 3.5 | 1.01 | 157 | 60.9 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| Klickitat | bxt1nL | 87358 | 20261119T103714Z-b462ad74 | 94.0 | 0.0 | 6.0 | 21.0 | 0.820 | 25.7 | 0.48 | 52.4 | 1.15 | 90 | 40.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| Allensville | bxt1nL | 87358 | 20261119T111813Z-dfeaeac0 | 100.0 | 0.0 | 0.0 | 94.5 | 0.976 | 94.7 | 0.92 | 0.2 | 0.98 | 21 | 39.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| Lakeville | bxt1nL | 87358 | 20261119T105154Z-d5ee7cbe | 97.9 | 0.0 | 2.1 | 82.2 | 0.962 | 87.1 | 0.87 | 1.2 | 1.07 | 56 | 67.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus30 | bxt1nL | 87358 | 20261119T104333Z-1723575f | 81.5 | 0.6 | 17.9 | 57.9 | 1.061 | 88.2 | 1.81 | 0.0 | 1.06 | 372 | 83.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus22 | bxt1L | 1 | 20261119T115733Z-800e84b9 | 93.1 | 5.0 | 1.9 | 59.8 | 1.063 | 86.8 | 1.54 | 0.0 | 1.06 | 601 | 46.8 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus22 | bxt1_tdall | 87358 | 20261119T120743Z-1f8414c4 | 98.5 | 0.7 | 0.8 | 87.8 | 1.005 | 94.6 | 1.24 | 0.1 | 0.96 | 368 | 57.8 | `--lift time --bell 1 --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1L | 2 | 20261119T115923Z-a989e6a9 | 99.2 | 0.0 | 0.8 | 62.8 | 1.057 | 88.7 | 1.52 | 0.0 | 1.03 | 492 | 75.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus40 | bxt1nK | 87358 | 20261119T124434Z-58b1f4e1 | 66.7 | 0.0 | 33.3 | 67.4 | 1.062 | 79.2 | 0.83 | 20.0 | 1.01 | 0 | 42.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --bell-keep` |
| Klickitat | bxt1nLB | 87358 | 20261119T130554Z-6a9ff52a | 100.0 | 0.0 | 0.0 | 82.0 | 0.937 | 86.7 | 0.82 | 4.9 | 1.02 | 145 | 68.5 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| apt64 | bxt1nL_oct | 87358 | 20261119T133953Z-64e2c6ec | 74.4 | 0.5 | 25.1 | 23.1 | 0.819 | 44.8 | 0.14 | 75.9 | 1.00 | 590 | 44.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --octaves cell` |
| plus22 | bxt1K | 87358 | 20261119T131443Z-8d5d2253 | 95.6 | 0.8 | 3.6 | 79.1 | 1.025 | 94.1 | 1.31 | 0.1 | 1.00 | 501 | 86.4 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8 --bell-keep` |
| apt64 | bxt1nLB | 87358 | 20261119T142403Z-68ac3e37 | 77.6 | 0.0 | 22.4 | 43.2 | 0.896 | 84.5 | 0.30 | 74.0 | 1.06 | 1385 | 51.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus40 | bxt1nL_oct | 87358 | 20261119T141444Z-818d4b0f | 92.1 | 0.0 | 7.9 | 62.7 | 1.003 | 76.9 | 1.71 | 0.0 | 0.96 | 38 | 67.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --octaves cell` |
| plus30 | rec | 87358 | 20261119T153624Z-59bc35da | 92.4 | 0.0 | 7.6 | 58.4 | 1.041 | 79.8 | 1.75 | 0.0 | 1.03 | 77 | 43.5 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus40 | rec | 87358 | 20261119T153604Z-aa584d33 | 81.4 | 0.7 | 17.9 | 52.7 | 1.003 | 65.4 | 1.89 | 0.0 | 0.98 | 50 | 70.2 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |

## Supervised fits to the true travel time (representation test)

| scene | arm | seed | id | steps | band share | solved % | wall % | stall % | T within 10 % | T/T* | detour index | seen through % | step across walls / true (true >= 3 cells) | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | fit_lift | 87358 | 20261119T051532Z-bb58a134 | 6000 | 0.0 | 66.7 | 0.0 | 33.3 | 96.6 | 0.980 | 0.89 | 8.9 | 0.29 | `--lift time` |
| plus22 | fit_count_f025 | 87358 | 20261119T051752Z-0744cf09 | 6000 | 0.0 | 60.6 | 0.0 | 39.4 | 95.9 | 1.001 | 0.96 | 7.7 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25` |
| plus22 | fit_ntrl | 87358 | 20261119T052308Z-3456d8e3 | 6000 | 0.0 | 51.2 | 48.8 | 0.0 | 95.8 | 1.014 | 1.02 | 4.6 | 0.30 | `--lift none` |
| plus22 | fit_bit | 87358 | 20261119T052113Z-100d3f8e | 6000 | 0.0 | 64.4 | 0.1 | 35.4 | 96.5 | 1.002 | 0.95 | 6.4 | 0.29 | `--lift time --pair bit` |
| plus22 | fit_count_f1 | 87358 | 20261119T052358Z-5c15bcc3 | 6000 | 0.0 | 62.5 | 0.0 | 37.5 | 95.3 | 1.002 | 0.97 | 5.9 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1` |
| plus22 | fit_lenraw_box | 87358 | 20261119T052538Z-ce29b9ea | 6000 | 0.0 | 60.3 | 1.0 | 38.8 | 95.1 | 1.001 | 0.95 | 6.5 | 0.33 | `--lift time --pair len --pair-what length --pair-norm raw --pair-scale box` |
| plus22 | fit_udraw_f025 | 87358 | 20261119T052448Z-e443510f | 6000 | 0.0 | 59.3 | 0.0 | 40.7 | 95.8 | 1.000 | 0.96 | 7.1 | 0.32 | `--lift time --pair len --pair-what updown --pair-norm raw --pair-scale 7.2 --pair-freq 0.25` |
| plus22 | fit_lift_oct8 | 87358 | 20261119T052723Z-c0689232 | 6000 | 0.0 | 68.5 | 0.0 | 31.5 | 97.7 | 1.000 | 0.98 | 1.6 | 0.50 | `--lift time --octaves 2,4,8` |
| plus22 | fit_count_same | 87358 | 20261119T052633Z-b8f27f92 | 6000 | 0.0 | 59.3 | 0.0 | 40.7 | 95.6 | 1.004 | 0.97 | 7.4 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --pair-sign same` |
| plus22 | fit_lift_30k | 87358 | 20261119T052903Z-f76e30f7 | 30000 | 0.0 | 86.8 | 10.8 | 2.4 | 99.8 | 1.000 | 1.00 | 0.0 | 0.94 | `--lift time --steps 30000` |
| plus22 | fit_liftB | 87358 | 20261119T053023Z-fbac8fa0 | 20000 | 0.6 | 83.9 | 7.1 | 9.0 | 99.8 | 1.007 | 1.02 | 0.1 | 0.95 | `--lift time --band 0.6 --steps 20000` |
| plus22 | fit_countB_f1 | 87358 | 20261119T053313Z-cd5cac82 | 20000 | 0.6 | 63.5 | 2.1 | 34.4 | 99.4 | 1.004 | 1.01 | 0.2 | 0.81 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1 --band 0.6 --steps 20000` |
| plus22 | fit_count_f1_30k | 87358 | 20261119T052843Z-bec17a64 | 30000 | 0.0 | 66.1 | 1.1 | 32.8 | 99.5 | 1.007 | 1.01 | 0.2 | 0.88 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1 --steps 30000` |
| plus22 | fit_ud_tanh | 87358 | 20261119T053723Z-30de5ac0 | 6000 | 0.0 | 63.8 | 4.4 | 31.8 | 96.1 | 1.000 | 0.97 | 5.8 | 0.28 | `--lift time --pair len --pair-what updown --pair-scale 1` |
| plus22 | fit_countB_f025 | 87358 | 20261119T053703Z-5ecaf932 | 20000 | 0.6 | 64.4 | 5.8 | 29.7 | 99.6 | 1.003 | 1.00 | 0.2 | 0.88 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --band 0.6 --steps 20000` |
| plus22 | fit_vec | 87358 | 20261119T053943Z-57d9ef4c | 6000 | 0.0 | 66.0 | 0.0 | 34.0 | 96.6 | 1.002 | 1.00 | 7.5 | 0.31 | `--lift vec` |
| plus22 | fit_udrawB_f025 | 87358 | 20261119T060753Z-0bb19ddd | 20000 | 0.6 | 63.2 | 4.9 | 31.9 | 99.6 | 1.012 | 1.03 | 0.0 | 0.90 | `--lift time --pair len --pair-what updown --pair-norm raw --pair-scale 7.2 --pair-freq 0.25 --band 0.6 --steps 20000` |
| plus22 | fit_lenrawB_box | 87358 | 20261119T061143Z-97f421cd | 20000 | 0.6 | 61.8 | 1.4 | 36.8 | 99.5 | 0.997 | 0.97 | 0.1 | 0.80 | `--lift time --pair len --pair-what length --pair-norm raw --pair-scale box --band 0.6 --steps 20000` |
| plus22 | fit_bitB | 87358 | 20261119T061533Z-40b85496 | 20000 | 0.6 | 85.3 | 1.1 | 13.6 | 99.8 | 1.004 | 1.02 | 0.0 | 0.97 | `--lift time --pair bit --band 0.6 --steps 20000` |
| plus22 | fit_lift_oct8B | 87358 | 20261119T061923Z-2413d0b1 | 20000 | 0.6 | 84.6 | 13.1 | 2.4 | 99.9 | 0.997 | 1.00 | 0.0 | 0.99 | `--lift time --octaves 2,4,8 --band 0.6 --steps 20000` |
| plus22 | fit_ntrlB | 87358 | 20261119T062243Z-472600a2 | 20000 | 0.6 | 58.8 | 33.9 | 7.4 | 99.7 | 0.995 | 0.98 | 0.0 | 0.94 | `--lift none --band 0.6 --steps 20000` |
| plus40 | fit_liftB | 87358 | 20261119T132734Z-79b0398f | 20000 | 0.6 | 70.5 | 0.0 | 29.5 | 99.5 | 0.993 | 0.96 | 0.7 | 0.55 | `--lift time --band 0.6 --steps 20000` |
| apt64 | fit_liftB | 87358 | 20261119T133053Z-9120bb20 | 20000 | 0.6 | 83.1 | 0.0 | 16.9 | 97.5 | 1.001 | 1.00 | 0.3 | 0.86 | `--lift time --band 0.6 --steps 20000` |
| plus40 | fit_lift_octB | 87358 | 20261119T133344Z-4f5a9775 | 20000 | 0.6 | 74.9 | 0.0 | 25.1 | 99.7 | 1.007 | 1.02 | 0.1 | 0.94 | `--lift time --octaves 3.62,7.24,14.48 --band 0.6 --steps 20000` |
| apt64 | fit_lift_octB | 87358 | 20261119T133703Z-0236554c | 20000 | 0.6 | 71.3 | 0.1 | 28.5 | 99.8 | 1.000 | 1.00 | 0.0 | 0.99 | `--lift time --octaves 5.79,11.58,23.17 --band 0.6 --steps 20000` |
| plus22 | fit_pre2B | 87358 | 20261119T144123Z-ac36c9f0 | 20000 | 0.6 | 86.2 | 7.4 | 6.4 | 99.8 | 1.000 | 1.00 | 0.0 | 0.98 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 2 --band 0.6 --steps 20000` |
| plus22 | fit_pre8B | 87358 | 20261119T145213Z-4427705b | 20000 | 0.6 | 86.2 | 0.1 | 13.6 | 99.7 | 1.000 | 1.00 | 0.0 | 0.99 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 8 --band 0.6 --steps 20000` |
| plus22 | fit_pre32B | 87358 | 20261119T150303Z-eb8f94ed | 20000 | 0.6 | 77.2 | 2.4 | 20.4 | 99.3 | 1.000 | 1.00 | 0.0 | 0.98 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 32 --band 0.6 --steps 20000` |
| plus22 | fit_post2B | 87358 | 20261119T151543Z-54fa631c | 20000 | 0.6 | 83.8 | 10.6 | 5.7 | 99.6 | 0.993 | 0.99 | 0.0 | 0.95 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 2 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_pre128B | 87358 | 20261119T151353Z-15181721 | 20000 | 0.6 | 40.8 | 0.7 | 58.5 | 90.3 | 1.001 | 0.98 | 1.0 | 0.91 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 128 --band 0.6 --steps 20000` |
| plus22 | fit_post8B | 87358 | 20261119T151933Z-6ed591c5 | 20000 | 0.6 | 87.2 | 6.0 | 6.8 | 98.8 | 1.009 | 1.03 | 0.0 | 0.96 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 8 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_post32B | 87358 | 20261119T152223Z-e2b820a9 | 20000 | 0.6 | 54.3 | 0.3 | 45.4 | 92.8 | 0.993 | 0.98 | 0.2 | 0.90 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 32 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_post128B | 87358 | 20261119T152243Z-9d706657 | 20000 | 0.6 | 53.2 | 0.0 | 46.8 | 92.6 | 0.991 | 0.96 | 0.1 | 0.88 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 128 --pair-amp 0.075 --band 0.6 --steps 20000` |

## Jobs that did not finish

* `bell03_k16|plus22|87358|--lift time --bell 0.3 --bell-k 16 --td-free` -> 20261119T063343Z-fbef26cb
* `bx1|plus22|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6` -> 20261119T064233Z-a3d6270c
* `bxt1nLB|plus40|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` -> 20261119T152334Z-44d3ecfa
* `bxt1nLB|plus30|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` -> 20261119T153123Z-d85dea7e
