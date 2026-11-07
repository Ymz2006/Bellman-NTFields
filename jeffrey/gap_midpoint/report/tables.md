# Midpoint loss on the gap mazes

MPPI success (%), 300 queries × planning seeds 0/1/2 per training seed. Same queries for every row.

## All queries, per training seed

| gap | field | seed 87358 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|---|
| ×1 | ntrl-demo | 40.8 | 40.4 | 50.2 | 43.8 |
| ×1 | control (resumed, no term) | 39.4 | 51.1 | 43.0 | 44.5 |
| ×1 | midpoint loss | 59.2 | 58.1 | 60.8 | 59.4 |
| ×1 | segment DP + wall term 0.3 | 54.9 | 56.2 | 63.4 | 58.2 |
| ×1 | constructed | 66.4 | 68.2 | 58.9 | 64.5 |
| ×1 | warp + scaling | 58.3 | 60.3 | 56.9 | 58.5 |
| ×1 | README warp | 49.4 | 45.7 | 43.7 | 46.3 |
| ×2 | ntrl-demo | 34.8 | 29.8 | 32.3 | 32.3 |
| ×2 | control (resumed, no term) | 37.0 | 33.8 | 34.2 | 35.0 |
| ×2 | midpoint loss | 36.6 | 40.7 | 36.1 | 37.8 |
| ×2 | segment DP + wall term 0.3 | 39.7 | 36.0 | 38.4 | 38.0 |
| ×2 | constructed | 50.7 | 36.0 | 38.7 | 41.8 |
| ×2 | warp + scaling | 37.7 | 40.3 | 31.9 | 36.6 |
| ×2 | README warp | 31.3 | 40.6 | 41.1 | 37.7 |
| ×4 | ntrl-demo | 26.0 | 26.7 | 27.8 | 26.8 |
| ×4 | control (resumed, no term) | 26.4 | 26.9 | 26.2 | 26.5 |
| ×4 | midpoint loss | 29.2 | 29.2 | 28.9 | 29.1 |
| ×4 | segment DP + wall term 0.3 | 26.8 | 28.4 | 28.4 | 27.9 |
| ×4 | constructed | 25.1 | 22.0 | 16.1 | 21.1 |
| ×4 | warp + scaling | 31.3 | 21.4 | 26.8 | 26.5 |
| ×4 | README warp | 26.9 | 28.7 | 27.3 | 27.6 |
| ×8 | ntrl-demo | 26.3 | 26.4 | 25.0 | 25.9 |
| ×8 | control (resumed, no term) | 26.1 | 26.0 | 25.0 | 25.7 |
| ×8 | midpoint loss | 26.7 | 27.1 | 25.0 | 26.3 |
| ×8 | segment DP + wall term 0.3 | 20.6 | 20.0 |  | 20.3 |
| ×8 | constructed | 11.3 | 7.7 | 11.6 | 10.2 |
| ×8 | warp + scaling | 19.1 | 8.6 | 19.2 | 15.6 |
| ×8 | README warp | 27.0 | 22.3 | 23.4 | 24.3 |
| ×12 | ntrl-demo | 24.2 | 24.4 | 23.9 | 24.2 |
| ×12 | control (resumed, no term) | 24.0 | 25.3 | 24.1 | 24.5 |
| ×12 | midpoint loss | 24.9 | 24.6 | 24.7 | 24.7 |
| ×12 | segment DP + wall term 0.3 | 18.4 | 17.6 |  | 18.0 |
| ×12 | constructed | 10.0 | 7.3 | 12.0 | 9.8 |
| ×12 | warp + scaling | 11.7 | 3.9 | 11.0 | 8.9 |
| ×12 | README warp | 22.7 | 22.2 | 22.7 | 22.5 |

## By region pair (pooled over the training seeds listed)

| gap | field | seeds | same | adjacent | diagonal | quad–open | open–open |
|---|---|---|---|---|---|---|---|
| ×1 | ntrl-demo | 3 | 33 | 25 | 23 | 43 | 96 |
| ×1 | control (resumed, no term) | 3 | 34 | 19 | 29 | 47 | 93 |
| ×1 | midpoint loss | 3 | 48 | 40 | 51 | 58 | 100 |
| ×1 | segment DP + wall term 0.3 | 3 | 57 | 30 | 44 | 62 | 99 |
| ×1 | constructed | 3 | 64 | 50 | 48 | 64 | 97 |
| ×1 | warp + scaling | 3 | 55 | 42 | 43 | 57 | 96 |
| ×1 | README warp | 3 | 29 | 21 | 37 | 49 | 97 |
| ×2 | ntrl-demo | 3 | 22 | 10 | 8 | 25 | 97 |
| ×2 | control (resumed, no term) | 3 | 23 | 10 | 16 | 29 | 97 |
| ×2 | midpoint loss | 3 | 22 | 11 | 22 | 34 | 99 |
| ×2 | segment DP + wall term 0.3 | 3 | 32 | 6 | 19 | 36 | 97 |
| ×2 | constructed | 3 | 42 | 21 | 16 | 39 | 91 |
| ×2 | warp + scaling | 3 | 34 | 15 | 18 | 30 | 86 |
| ×2 | README warp | 3 | 26 | 11 | 19 | 34 | 98 |
| ×4 | ntrl-demo | 3 | 15 | 9 | 0 | 12 | 98 |
| ×4 | control (resumed, no term) | 3 | 13 | 12 | 0 | 11 | 97 |
| ×4 | midpoint loss | 3 | 14 | 12 | 5 | 16 | 99 |
| ×4 | segment DP + wall term 0.3 | 3 | 17 | 6 | 2 | 17 | 97 |
| ×4 | constructed | 3 | 36 | 1 | 0 | 19 | 49 |
| ×4 | warp + scaling | 3 | 23 | 8 | 7 | 23 | 72 |
| ×4 | README warp | 3 | 19 | 6 | 6 | 11 | 96 |
| ×8 | ntrl-demo | 3 | 12 | 3 | 2 | 13 | 100 |
| ×8 | control (resumed, no term) | 3 | 12 | 2 | 2 | 13 | 100 |
| ×8 | midpoint loss | 3 | 13 | 2 | 2 | 14 | 100 |
| ×8 | segment DP + wall term 0.3 | 2 | 11 | 0 | 0 | 5 | 85 |
| ×8 | constructed | 3 | 26 | 1 | 0 | 3 | 22 |
| ×8 | warp + scaling | 3 | 25 | 2 | 2 | 7 | 42 |
| ×8 | README warp | 3 | 14 | 1 | 1 | 10 | 95 |
| ×12 | ntrl-demo | 3 | 14 | 0 | 0 | 7 | 100 |
| ×12 | control (resumed, no term) | 3 | 14 | 0 | 0 | 8 | 100 |
| ×12 | midpoint loss | 3 | 15 | 0 | 0 | 9 | 100 |
| ×12 | segment DP + wall term 0.3 | 2 | 6 | 0 | 0 | 0 | 84 |
| ×12 | constructed | 3 | 19 | 0 | 0 | 1 | 29 |
| ×12 | warp + scaling | 3 | 16 | 0 | 1 | 3 | 24 |
| ×12 | README warp | 3 | 13 | 1 | 1 | 2 | 95 |

## Runs: failures, field and training drives

Failures out of 900 runs. T/T*: median learned over true travel time on the queries. Last refresh: drives reaching their goal of 8,192, and mean steps.

| gap | field | seed | run | success | collisions | stalls without collision | T/T* | Spearman | descent cos | last refresh reached | steps | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ×1 | control (resumed, no term) | 87358 | `20261105T025447Z-d6f0425f` | 355 | 539 | 6 | 1.07 | 0.986 | 0.88 |  | 0 | 7 |
| ×1 | control (resumed, no term) | 1 | `20261105T080028Z-6a0c2b6d` | 460 | 411 | 29 | 1.09 | 0.991 | 0.90 |  | 0 | 4 |
| ×1 | control (resumed, no term) | 2 | `20261105T124308Z-db683db3` | 387 | 511 | 2 | 1.08 | 0.991 | 0.88 |  | 0 | 4 |
| ×1 | midpoint loss | 87358 | `20261105T033827Z-518d4078` | 533 | 366 | 1 | 1.02 | 0.994 | 0.92 | 7137 | 189 | 29 |
| ×1 | midpoint loss | 1 | `20261105T082808Z-c7a08db2` | 523 | 361 | 16 | 1.02 | 0.995 | 0.91 | 7250 | 191 | 27 |
| ×1 | midpoint loss | 2 | `20261105T131108Z-56ab1623` | 547 | 350 | 3 | 1.02 | 0.995 | 0.92 | 7292 | 190 | 27 |
| ×2 | control (resumed, no term) | 87358 | `20261105T030207Z-dbcb8e53` | 333 | 561 | 6 | 1.00 | 0.986 | 0.84 |  | 0 | 6 |
| ×2 | control (resumed, no term) | 1 | `20261105T080508Z-25a4d2e7` | 304 | 586 | 10 | 1.02 | 0.968 | 0.88 |  | 0 | 5 |
| ×2 | control (resumed, no term) | 2 | `20261105T124748Z-696b266b` | 308 | 587 | 5 | 1.01 | 0.965 | 0.85 |  | 0 | 5 |
| ×2 | midpoint loss | 87358 | `20261105T040727Z-05470613` | 329 | 568 | 3 | 0.99 | 0.973 | 0.87 | 7128 | 225 | 32 |
| ×2 | midpoint loss | 1 | `20261105T085448Z-1068669c` | 366 | 534 | 0 | 1.01 | 0.983 | 0.91 | 7357 | 231 | 31 |
| ×2 | midpoint loss | 2 | `20261105T133749Z-08d357dd` | 325 | 572 | 3 | 1.01 | 0.984 | 0.91 | 7222 | 227 | 31 |
| ×4 | control (resumed, no term) | 87358 | `20261105T030847Z-4818d726` | 238 | 662 | 0 | 1.03 | 0.980 | 0.86 |  | 0 | 7 |
| ×4 | control (resumed, no term) | 1 | `20261105T081028Z-05996ee9` | 242 | 654 | 4 | 1.03 | 0.978 | 0.86 |  | 0 | 5 |
| ×4 | control (resumed, no term) | 2 | `20261105T125248Z-36ea4286` | 236 | 664 | 0 | 0.99 | 0.972 | 0.83 |  | 0 | 6 |
| ×4 | midpoint loss | 87358 | `20261105T043928Z-92e01995` | 263 | 637 | 0 | 1.00 | 0.981 | 0.87 | 7588 | 332 | 43 |
| ×4 | midpoint loss | 1 | `20261105T092628Z-c05f727f` | 263 | 637 | 0 | 0.99 | 0.979 | 0.88 | 7551 | 332 | 43 |
| ×4 | midpoint loss | 2 | `20261105T140929Z-d5196a0e` | 260 | 640 | 0 | 0.99 | 0.975 | 0.89 | 7548 | 333 | 42 |
| ×8 | control (resumed, no term) | 87358 | `20261105T031607Z-6a3d0480` | 235 | 665 | 0 | 1.06 | 0.994 | 0.90 |  | 0 | 12 |
| ×8 | control (resumed, no term) | 1 | `20261105T081608Z-72b348bf` | 234 | 666 | 0 | 1.06 | 0.993 | 0.91 |  | 0 | 6 |
| ×8 | control (resumed, no term) | 2 | `20261105T125849Z-731e9e7a` | 225 | 675 | 0 | 1.05 | 0.993 | 0.91 |  | 0 | 6 |
| ×8 | midpoint loss | 87358 | `20261105T052248Z-fcdeffdb` | 240 | 660 | 0 | 1.04 | 0.994 | 0.91 | 7943 | 564 | 66 |
| ×8 | midpoint loss | 1 | `20261105T100928Z-890eaeb5` | 244 | 656 | 0 | 1.04 | 0.994 | 0.92 | 7965 | 558 | 64 |
| ×8 | midpoint loss | 2 | `20261105T145209Z-0bdb0250` | 225 | 675 | 0 | 1.05 | 0.991 | 0.92 | 7994 | 562 | 64 |
| ×12 | control (resumed, no term) | 87358 | `20261105T032807Z-cf649b48` | 216 | 684 | 0 | 1.06 | 0.995 | 0.93 |  | 0 | 10 |
| ×12 | control (resumed, no term) | 1 | `20261105T082148Z-a1da0e28` | 228 | 672 | 0 | 1.05 | 0.994 | 0.92 |  | 0 | 6 |
| ×12 | control (resumed, no term) | 2 | `20261105T130449Z-7537edae` | 217 | 683 | 0 | 1.06 | 0.992 | 0.90 |  | 0 | 6 |
| ×12 | midpoint loss | 87358 | `20261105T062928Z-2db6963e` | 224 | 676 | 0 | 1.03 | 0.994 | 0.92 | 8077 | 794 | 91 |
| ×12 | midpoint loss | 1 | `20261105T111348Z-fcb3fcd1` | 221 | 679 | 0 | 1.03 | 0.995 | 0.92 | 8087 | 786 | 89 |
| ×12 | midpoint loss | 2 | `20261105T155629Z-6e4fb4ea` | 222 | 678 | 0 | 1.02 | 0.993 | 0.91 | 8073 | 792 | 89 |

