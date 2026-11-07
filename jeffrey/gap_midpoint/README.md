# Midpoint loss on the gap (plus) mazes

Request (2026-11-04): "on the plus experiments, try this midpoint loss" (reference: the midpoint study's page,
https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U).

## Setup

- **Mazes and queries**: the gap study's (jeffrey/cross_warp, `README_gap.md`), laid out by jeffrey/gap_segdp's prep runs:
  the 22×22 cross maze's four 8×8 quadrants pushed apart by an open band of 6k cells, k = 1 / 2 / 4 / 8 / 12
  (n = 22 / 28 / 40 / 64 / 88), 300 certified queries per maze (60 per region pair), MPPI planning seeds 0/1/2.
- **Midpoint loss**: jeffrey/midpoint_dp's main recipe, unchanged (`--weight 1 --leaf min --traverse safe --nodes both`):
  resume the gap study's ntrl-demo run at epoch 2,000 (training seeds 87358 / 1 / 2) and train to 5,000 with the term;
  every 100 epochs 8,192 wall-avoiding MPPI drives between random free poses, split at travel-time midpoints down to
  1-cell leaves, weight ramped 0 → 1 over 500 epochs.
- **Controls**: the base ntrl-demo field at epoch 5,000 (gap study) and the same resumed loop with `--weight 0`.
- **For reference**: segment DP (jeffrey/gap_segdp) and the gap study's warps on the same queries.

## Code

- `gapmid.py`: installs the gap maze patch, builds two small adapter folders (symlinks) and runs midpoint_dp's
  `pipeline.py` main() unchanged, then adds the by-category breakdown.
- `run.py`: registry entry; freezes gapmid.py, midpoint_dp's sources and cross_warp's gap.py into `snap/<hash>/`.
- `report.py`: tables against the base, the control, segment DP and the warps.

## Runs

Smoke (k = 1): `20261105T025038Z-63e35f05`. All 30 runs (15 midpoint, 15 control) succeeded; ids per run in
`report/tables.md`. Prepare runs: see jeffrey/gap_segdp. Bases: the gap study's ntrl-demo runs.

## Findings

MPPI success, 900 runs per training seed, tables in `report/tables.md` (`report.py`). Per seed 87358 / 1 / 2 and pooled (%):

| gap | ntrl-demo | control | midpoint loss | midpoint per seed | control per seed | segment DP |
|---|---|---|---|---|---|---|
| ×1 | 43.8 | 44.5 | **59.4** | 59.2 / 58.1 / 60.8 | 39.4 / 51.1 / 43.0 | 58.2 |
| ×2 | 32.3 | 35.0 | **37.8** | 36.6 / 40.7 / 36.1 | 37.0 / 33.8 / 34.2 | 38.0 |
| ×4 | 26.8 | 26.5 | **29.1** | 29.2 / 29.2 / 28.9 | 26.4 / 26.9 / 26.2 | 27.9 |
| ×8 | 25.9 | 25.7 | **26.3** | 26.7 / 27.1 / 25.0 | 26.1 / 26.0 / 25.0 | 20.3 (2 seeds) |
| ×12 | 24.2 | 24.5 | **24.7** | 24.9 / 24.6 / 24.7 | 24.0 / 25.3 / 24.1 | 18.0 (2 seeds) |

1. **The gain shrinks with the gap.** Over the control: +14.9 points at ×1 (every seed), +2.8 at ×2 (two seeds of three),
   +2.6 at ×4 (every seed, small but consistent), +0.6 at ×8 and +0.2 at ×12 (no effect).
2. **Where it helps.** ×1: every region pair (same 34 → 48, adjacent 19 → 40, diagonal 29 → 51, quadrant–open 47 → 58,
   open–open 93 → 100). ×4: diagonal 0 → 5, quadrant–open 11 → 16. ×12: same 14 → 15, adjacent and diagonal stay 0,
   quadrant–open 8 → 9.
3. **It does no harm.** Open–open stays at 99–100 % at every gap. Segment DP falls to 85 % (×8) and 84 % (×12), which is
   why it ends below ntrl-demo there; the midpoint loss stays level with ntrl-demo.
4. **At large gaps value error is not the failure.** ×12: wall hits 676–679 of 900 against 672–684 for the control, no
   stalls; T/T* 1.02–1.03 against 1.05–1.06, so the field's values were already close on these queries.

**Likely cause, not tested.** Drive endpoints are uniform over free poses, and the quadrants' share of the maze falls
quickly: by cell count both ends lie in a quadrant for 28 / 11 / 2.6 / 0.4 / 0.1 % of drives at ×1 / 2 / 4 / 8 / 12 (upper
bounds; near-wall poses are excluded). In the last refresh 8,073–8,087 of 8,192 drives reach their goal at ×12, against
7,137–7,292 at ×1. Region-balanced endpoints would test this; not run.
