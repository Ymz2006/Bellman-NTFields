# Part 4 screen (rayscreen.py). sep: row difference across a wall / across an open passage; rank: Spearman of row distance and T* over the queries; rough: see the script.

| rows | maze | sep L1 | sep L2 | sep Linf | rank L1 | rank L2 | rank Linf | rough L2 |
|---|---|---|---|---|---|---|---|---|
| hits to the edge | 22×22 | 1.18 | 1.05 | 0.99 | 0.91 | 0.90 | 0.82 | 2.4 |
| hits to the edge | 13×13 | 1.27 | 1.10 | 1.01 | 0.87 | 0.85 | 0.69 | 2.4 |
| hits within 1 cells | 22×22 | 0.99 | 1.00 | 1.02 | 0.02 | 0.03 | 0.06 | 2.8 |
| hits within 1 cells | 13×13 | 0.99 | 0.99 | 0.98 | 0.17 | 0.15 | 0.18 | 2.7 |
| hits within 2 cells | 22×22 | 1.03 | 1.00 | 0.98 | 0.05 | 0.06 | 0.03 | 3.0 |
| hits within 2 cells | 13×13 | 1.04 | 1.01 | 0.98 | 0.12 | 0.12 | 0.13 | 3.1 |
| hits within 4 cells | 22×22 | 1.05 | 1.01 | 0.98 | 0.23 | 0.20 | 0.11 | 2.7 |
| hits within 4 cells | 13×13 | 1.13 | 1.06 | 0.99 | 0.43 | 0.38 | 0.21 | 2.7 |
| depth, at most 1 cells | 22×22 | 1.01 | 1.01 | 1.01 | 0.11 | 0.09 | 0.02 | 2.0 |
| depth, at most 1 cells | 13×13 | 0.99 | 0.99 | 0.99 | 0.12 | 0.07 | -0.04 | 1.9 |
| depth, at most 2 cells | 22×22 | 0.92 | 0.98 | 1.02 | 0.03 | 0.04 | 0.06 | 2.2 |
| depth, at most 2 cells | 13×13 | 0.92 | 0.97 | 1.00 | 0.13 | 0.12 | 0.07 | 2.0 |
| depth, at most 4 cells | 22×22 | 0.91 | 1.00 | 1.03 | -0.06 | -0.03 | -0.01 | 2.3 |
| depth, at most 4 cells | 13×13 | 0.90 | 0.98 | 1.01 | 0.07 | 0.09 | 0.07 | 2.1 |
| integral of 1 / speed (part 2) | 22×22 | 1.09 | 1.04 | 1.01 | 0.88 | 0.87 | 0.78 | 2.2 |
| integral of 1 / speed (part 2) | 13×13 | 1.12 | 1.06 | 1.01 | 0.80 | 0.77 | 0.64 | 2.1 |
| unit ramps (length along the line, no walls) | 22×22 | 0.95 | 0.95 | 0.92 | 0.91 | 0.92 | 0.87 | 1.0 |
| unit ramps (length along the line, no walls) | 13×13 | 0.97 | 0.98 | 0.97 | 0.85 | 0.86 | 0.82 | 1.0 |
| integral / its L1 norm over directions | 22×22 | 1.01 | 0.99 | 0.98 | 0.86 | 0.85 | 0.75 | 2.3 |
| integral / its L1 norm over directions | 13×13 | 1.04 | 1.01 | 1.01 | 0.78 | 0.77 | 0.68 | 2.2 |
| integral / its Linf norm over directions | 22×22 | 1.07 | 1.04 | 1.02 | 0.84 | 0.85 | 0.76 | 2.3 |
| integral / its Linf norm over directions | 13×13 | 1.12 | 1.07 | 1.01 | 0.74 | 0.73 | 0.64 | 2.3 |
| landmark rows: L1 of integral differences (64 directions) | 22×22 | 1.38 | 1.37 | 1.31 | 0.92 | 0.92 | 0.91 | 1.4 |
| landmark rows: L1 of integral differences (64 directions) | 13×13 | 1.46 | 1.45 | 1.37 | 0.87 | 0.87 | 0.85 | 1.3 |
| landmark rows: L1 of hit-count differences (64 rays) | 22×22 | 1.66 | 1.59 | 1.46 | 0.94 | 0.94 | 0.93 | 2.0 |
| landmark rows: L1 of hit-count differences (64 rays) | 13×13 | 1.82 | 1.72 | 1.51 | 0.91 | 0.91 | 0.90 | 1.7 |
| landmark rows: Linf of integral differences (64 directions) | 22×22 | 1.13 | 1.09 | 1.04 | 0.80 | 0.81 | 0.78 | 2.7 |
| landmark rows: Linf of integral differences (64 directions) | 13×13 | 1.25 | 1.15 | 1.07 | 0.69 | 0.70 | 0.63 | 2.8 |
| landmark rows: Linf of hit-count differences (64 rays) | 22×22 | 1.06 | 1.04 | 1.04 | 0.85 | 0.87 | 0.86 | 3.2 |
| landmark rows: Linf of hit-count differences (64 rays) | 13×13 | 1.21 | 1.15 | 1.08 | 0.79 | 0.81 | 0.75 | 3.4 |
| corridor distance to 16 landmarks (geo, part 1) | 22×22 | 3.81 | 4.27 | 5.30 | 0.95 | 0.97 | 0.98 | 0.6 |
| corridor distance to 16 landmarks (geo, part 1) | 13×13 | 4.00 | 4.40 | 6.08 | 0.94 | 0.96 | 0.98 | 0.6 |
