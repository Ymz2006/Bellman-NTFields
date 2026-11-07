# Segment DP on the gap mazes

MPPI success (%), 300 queries × planning seeds 0/1/2 per training seed. ntrl-demo is the base every DP chain starts from; warps from the gap study for reference.

## All queries, per training seed

| gap | field | seed 87358 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|---|
| ×1 | ntrl-demo | 40.8 | 40.4 | 50.2 | 43.8 |
| ×1 | segment DP + wall term 0.3 | 54.9 | 56.2 | 63.4 | 58.2 |
| ×1 | constructed | 66.4 | 68.2 | 58.9 | 64.5 |
| ×1 | warp + scaling | 58.3 | 60.3 | 56.9 | 58.5 |
| ×1 | README warp | 49.4 | 45.7 | 43.7 | 46.3 |
| ×2 | ntrl-demo | 34.8 | 29.8 | 32.3 | 32.3 |
| ×2 | segment DP + wall term 0.3 | 39.7 | 36.0 | 38.4 | 38.0 |
| ×2 | constructed | 50.7 | 36.0 | 38.7 | 41.8 |
| ×2 | warp + scaling | 37.7 | 40.3 | 31.9 | 36.6 |
| ×2 | README warp | 31.3 | 40.6 | 41.1 | 37.7 |
| ×4 | ntrl-demo | 26.0 | 26.7 | 27.8 | 26.8 |
| ×4 | segment DP + wall term 0.3 | 26.8 | 28.4 | 28.4 | 27.9 |
| ×4 | constructed | 25.1 | 22.0 | 16.1 | 21.1 |
| ×4 | warp + scaling | 31.3 | 21.4 | 26.8 | 26.5 |
| ×4 | README warp | 26.9 | 28.7 | 27.3 | 27.6 |
| ×8 | ntrl-demo | 26.3 | 26.4 | 25.0 | 25.9 |
| ×8 | segment DP + wall term 0.3 | 20.6 | 20.0 |  | 20.3 |
| ×8 | constructed | 11.3 | 7.7 | 11.6 | 10.2 |
| ×8 | warp + scaling | 19.1 | 8.6 | 19.2 | 15.6 |
| ×8 | README warp | 27.0 | 22.3 | 23.4 | 24.3 |
| ×12 | ntrl-demo | 24.2 | 24.4 | 23.9 | 24.2 |
| ×12 | segment DP + wall term 0.3 | 18.4 | 17.6 |  | 18.0 |
| ×12 | constructed | 10.0 | 7.3 | 12.0 | 9.8 |
| ×12 | warp + scaling | 11.7 | 3.9 | 11.0 | 8.9 |
| ×12 | README warp | 22.7 | 22.2 | 22.7 | 22.5 |

## By region pair (pooled over training seeds)

| gap | field | seeds | same | adjacent | diagonal | quad–open | open–open |
|---|---|---|---|---|---|---|---|
| ×1 | ntrl-demo | 3 | 33 | 25 | 23 | 43 | 96 |
| ×1 | segment DP + wall term 0.3 | 3 | 57 | 30 | 44 | 62 | 99 |
| ×1 | constructed | 3 | 64 | 50 | 48 | 64 | 97 |
| ×1 | warp + scaling | 3 | 55 | 42 | 43 | 57 | 96 |
| ×1 | README warp | 3 | 29 | 21 | 37 | 49 | 97 |
| ×2 | ntrl-demo | 3 | 22 | 10 | 8 | 25 | 97 |
| ×2 | segment DP + wall term 0.3 | 3 | 32 | 6 | 19 | 36 | 97 |
| ×2 | constructed | 3 | 42 | 21 | 16 | 39 | 91 |
| ×2 | warp + scaling | 3 | 34 | 15 | 18 | 30 | 86 |
| ×2 | README warp | 3 | 26 | 11 | 19 | 34 | 98 |
| ×4 | ntrl-demo | 3 | 15 | 9 | 0 | 12 | 98 |
| ×4 | segment DP + wall term 0.3 | 3 | 17 | 6 | 2 | 17 | 97 |
| ×4 | constructed | 3 | 36 | 1 | 0 | 19 | 49 |
| ×4 | warp + scaling | 3 | 23 | 8 | 7 | 23 | 72 |
| ×4 | README warp | 3 | 19 | 6 | 6 | 11 | 96 |
| ×8 | ntrl-demo | 3 | 12 | 3 | 2 | 13 | 100 |
| ×8 | segment DP + wall term 0.3 | 2 | 11 | 0 | 0 | 5 | 85 |
| ×8 | constructed | 3 | 26 | 1 | 0 | 3 | 22 |
| ×8 | warp + scaling | 3 | 25 | 2 | 2 | 7 | 42 |
| ×8 | README warp | 3 | 14 | 1 | 1 | 10 | 95 |
| ×12 | ntrl-demo | 3 | 14 | 0 | 0 | 7 | 100 |
| ×12 | segment DP + wall term 0.3 | 2 | 6 | 0 | 0 | 0 | 84 |
| ×12 | constructed | 3 | 19 | 0 | 0 | 1 | 29 |
| ×12 | warp + scaling | 3 | 16 | 0 | 1 | 3 | 24 |
| ×12 | README warp | 3 | 13 | 1 | 1 | 2 | 95 |

## Segment DP runs: failures and field

Failures out of 900 runs (a run can both stall and collide). T/T*: median learned over true travel time on the queries; top-level target/T*: the last DP level's targets over the truth on probe pairs; base/T* the same probes on the base.

| gap | seed | run | success | collisions | stalls without collision | T/T* | Spearman | descent cos | levels | top target/T* | base/T* there | train min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ×1 | 87358 | `20261104T195052Z-5695c6bf` | 494 | 383 | 23 | 1.07 | 0.989 | 0.89 | 14 | 1.07 | 1.07 | 30 |
| ×1 | 1 | `20261104T221552Z-be32944b` | 506 | 377 | 17 | 1.08 | 0.980 | 0.88 | 14 | 1.07 | 1.07 | 22 |
| ×1 | 2 | `20261105T004253Z-7f44dcbd` | 571 | 314 | 15 | 1.10 | 0.994 | 0.88 | 14 | 1.08 | 1.10 | 22 |
| ×2 | 87358 | `20261104T202052Z-a9ed04cd` | 357 | 541 | 2 | 1.02 | 0.966 | 0.87 | 14 | 1.03 | 0.98 | 23 |
| ×2 | 1 | `20261104T223832Z-70068132` | 324 | 561 | 15 | 1.04 | 0.959 | 0.86 | 14 | 1.06 | 1.01 | 23 |
| ×2 | 2 | `20261105T010533Z-036fb401` | 346 | 552 | 2 | 1.03 | 0.967 | 0.88 | 14 | 1.05 | 1.01 | 25 |
| ×4 | 87358 | `20261104T204353Z-07de6fa2` | 241 | 616 | 43 | 1.01 | 0.967 | 0.84 | 16 | 1.07 | 1.02 | 27 |
| ×4 | 1 | `20261104T230153Z-44248aa3` | 256 | 625 | 19 | 1.01 | 0.965 | 0.85 | 16 | 1.06 | 1.01 | 27 |
| ×4 | 2 | `20261105T013053Z-fb2b1f53` | 256 | 626 | 18 | 1.01 | 0.964 | 0.81 | 16 | 1.07 | 1.00 | 28 |
| ×8 | 87358 | `20261104T211113Z-baca88d7` | 185 | 676 | 39 | 0.86 | 0.902 | 0.69 | 17 | 1.07 | 1.05 | 30 |
| ×8 | 1 | `20261104T232934Z-31055d3a` | 180 | 664 | 56 | 0.85 | 0.883 | 0.71 | 17 | 1.06 | 1.06 | 35 |
| ×12 | 87358 | `20261104T214154Z-2b5fd960` | 166 | 678 | 56 | 0.72 | 0.719 | 0.56 | 18 | 0.97 | 1.04 | 34 |
| ×12 | 1 | `20261105T000515Z-6f439932` | 158 | 679 | 63 | 0.70 | 0.692 | 0.60 | 18 | 0.95 | 1.04 | 38 |

