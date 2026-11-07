# Segment DP on the gap (plus) mazes

Request (2026-11-04): "try to run my segment dp on the plus maze scenarios" (reference: the gap study's warp-space
viewer, https://claude.ai/artifact/Y6J6umPzgXSKs6uNZqr87m).

## Setup

- **Mazes and queries**: the gap study's (jeffrey/cross_warp, `README_gap.md`): the 22×22 cross maze's four 8×8 quadrant
  mazes pushed apart by an open band of 6k cells, k = 1 / 2 / 4 / 8 / 12 (n = 22 / 28 / 40 / 64 / 88), 300 certified queries
  per maze, 60 in each region pair (same quadrant, adjacent quadrants, diagonal quadrants, quadrant – open, open – open),
  MPPI planning seeds 0/1/2 (900 runs per field).
- **Segment DP**: jeffrey/maze_scale's best recipe, unchanged (`segdp3.py`, normal weight 0.3): a chain of networks
  over geometric distance segments (0.5 cells × 1.35^m up to the 99.9th-percentile pool gap), each level fit to DP
  targets min_k T(i, k) + T(k, j) over 128 pool split candidates from the previous levels, warm-started from the previous
  level, plus ntrl-demo's wall-direction (normal) term at near-wall poses on every level; base = ntrl-demo trained on that
  maze (the gap study's own ntrl-demo runs, training seeds 87358 / 1 / 2). Pool: 200,000 collision-free poses, as in
  maze_scale (so 413 per cell at k = 1 down to 26 per cell at k = 12).
- **Control**: the base ntrl-demo field itself on the same queries (from the gap study's tables).

## Code

- `prep.py`: lays a gap-study prepare run out as segdp3.py expects (copies maze, goals, queries, held-out data, adds the
  pool and the truth fields; for k ≥ 8 the truth fields are recomputed with cross_warp's truthfast.FastGrid and checked
  against every query's certified T*: max relative error 4.6e-4).
- `gapsegdp.py`: installs the gap maze patch, swaps truth.Grid for truthfast.FastGrid (only used for the per-level truth
  probes; identical query_time without the 18 GB cost arrays), checks the base is plain ntrl-demo (identity warp), runs
  segdp3.py / segdp.py's main() unchanged, adds the by-category breakdown.
- `run.py`: registry entry (stages `prep`, `segdp`), freezes the sources into `snap/<hash>/`.
- `report.py`: tables against the gap study's fields.

## Runs

Prepare runs: k1 `20261104T194755Z-2a41a007` (queries: gap study k=1 set `20261103T233014Z-ae84b379`, held-out data
from the maze_scale plus prepare `20261102T042454Z-62864e61`), k2 `20261104T194759Z-d67216a6`, k4 `20261104T194803Z-e648c682`,
k8 `20261104T194807Z-de3bf2e6`, k12 `20261104T194907Z-0c615578`.

## Findings

(pending)
