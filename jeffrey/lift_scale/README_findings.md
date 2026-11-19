## Findings

Findings page: ARTIFACT_URL (`report/lift_scale.html`, built by `page.py`). `report/tables.md` (`report.py`) is regenerated from the
registry and is authoritative for every number; the pooled table is copied at the end of this file. Single-run numbers below are
seed 87358 on the 22-cell plus maze unless a scene is named. One training of the same recipe moves by 3-6 points between seeds.

1. **The lift's own claim holds in the C-space and at every size.** S x |grad T| on held-out band points is 0.93-1.02 in the innermost
   band on every scene with the lift alone, within the first 100 epochs (ntrl-demo 0.05-0.11, worse the larger the scene). nu = |grad_q d| / E
   up to 1.4-2.1 costs nothing measurable.
2. **The lift alone plans below ntrl-demo on 7 of 8 scenes** (55-67 % solved against 56-90 %). Obstacles are seen through (98-100 % of detour pairs), as on the lift page,
   and the far field is short even along free lines of sight, which is new: learned / true 0.88 (plus 22), 0.65 (plus 40), 0.42-0.54
   (apartment, houses). `heading.py`: the field keeps |grad T| = 1 by putting slope into the heading coordinate (radial slope 0.63,
   |dT/dphi| 0.55 beyond 16 cells; ntrl-demo 0.93 and 0.15). Control `--zero-heading` (the network does not see phi directly): line of
   sight learned / true 1.00, 97 % within 10 % with no push (it plans 34 %: a control, not a recipe).
3. **Lift + push (the page's schedule)** fixes the far field at once (line-of-sight pairs within 10 %: 98-100 %) and never arrives
   through a wall; failures are stalls at walls with the goal behind them. Solved %, lift + push vs ntrl-demo (seeds pooled: 3 on plus 30 /
   plus 40 / Lakeville / Klickitat, 2 on Allensville / apartment 64, 1 on plus 22 / apartment 32): plus 22 68.2 vs 65.0, plus 30 64.2
   vs 56.8, plus 40 63.2 vs 59.2, apartment 32 75.4 vs 76.8, Allensville 99.6 vs 90.3, apartment 64 73.0 vs 66.3, Lakeville 76.5 vs
   69.1, Klickitat 85.5 vs 71.2; travel times within 10 %: 89 / 81 / 79 / 38 / 88 / 37 / 87 / 94 % vs 46 / 61 / 52 / 47 / 78 / 20 /
   74 / 77 %. On the thin-walled scenes (plus mazes, apartment) the detours are seen through at the end
   (53-100 % of detour pairs); on the houses mostly not (Allensville 0 %, Klickitat 20 %, Lakeville 55 %).
   On the 22-cell maze they exist at the end of the strong push (detour index 1.64, 87 % solved at epoch 2,000, planning seed 0) and
   are gone 1,000 epochs later (0.45; lift page: 1.66 -> 0.50). On the 40-cell maze they never form (0.22 at epoch 2,000).
4. **Why (four tests).**
   * Constant strong push (`lift_c10`): the index swings 1.02 / 0.67 / 0.54 / 1.17 / 1.01 / 0.33 between checkpoints, line-of-sight
     times 8-13 % long, 404 rollbacks. The quadratic cap's gradient vanishes at the constraint: a push buys a violation proportional
     to its strength, and the detours are made of it.
   * Exact cap (`--mu1 0.05`, `lift_x`) or stiff quadratic cap (`--mu 10`, `liftq`): detour index 0.00 / -0.02, 100 % seen through.
     The largest field that respects the cap at the trained points is the see-through one.
   * Inside pairs (embed_warp data variant 20261112T065945Z-91ae7981: one end of 20 % of the pairs inside obstacles) with the lift
     continued through the walls (`--depth 0.3`): S x slope 1.00-1.03 at every depth inside the walls, line of sight right without a
     push, detour index creeping to 0.45 (5,000 epochs) and 0.3-0.6 (12,500 epochs: 68.1 % solved, 84 % within 10 %, wall layer 0.98).
     Yet `inside.py`: the learned time between mirror configurations across a wall (0.8 cells apart) is 1.3 cells where the straight
     crossing costs 55. The wall is a symmetric bump in the lifted input, T ~ T0(x) + h; the truth needs a step in T0 across the wall's
     thickness at equal height, i.e. planar resolution. Detours saturate near 4 cells.
   * `jumps.py`: ntrl-demo holds a 13.5-cell step across a wall on the 22-cell maze (with its smeared wall layer, times 10 % long) and
     about 1 cell on the 40-cell maze.
5. **Planar resolution is the missing piece.** Octaves (embed_warp's whitened x2, x4, x8 rows; `--octaves cell` scales them with the
   box side): on the 22-cell maze under the strong push the detour index is stable (1.72-1.75 from epoch 600 to 2,000), 91 % solved at
   epoch 2,000, 83 % at 3,000 with 88 % of times within 10 %, then lost by 5,000 (68.1 %). Push kept at 1e-2 (`lift_c10_oct8`): 83.8 %
   solved, times 14 % long, 13.6 % wall hits. Push at 6e-3: 72.4 % (index 0.8-1.5 between checkpoints, 95 % solved at epoch 2,000).
   Controls: ntrl-demo + octaves 68.9 %, lift + octaves without push 58.1 %. 40-cell maze with scaled octaves: lift + push 67.5 %
   solved, 92 % within 10 %; strong push 71.3 % (ntrl-demo 59.2 % / 50 %, ntrl-demo + octaves 60.0 % / 72 %, lift + push without octaves
   63.6 % / 79 %). 30-cell maze: 76.2 % (74.3 / 77.9 / 76.4), 95 % within 10 % (ntrl-demo 56.8 %); 40-cell maze over
   three seeds 68.8 % (69.9 / 68.9 / 67.5).
   **Houses with scaled octaves: Klickitat 98.7 % solved (97.9 / 98.6 / 99.6), 98 % of times within 10 % (ntrl-demo 71.2 %,
   ntrl-demo + octaves 71.4 %, lift + push 85.5 %); Lakeville 97.9 % (97.7 / 97.9 / 98.2), 98 % (ntrl-demo 69.1 %, ntrl-demo + octaves
   65.6 %, lift + push 76.5 %); Allensville 99.9 % (two seeds; lift + push without octaves already 99.6 %; ntrl-demo 90.3 %).** Constant strong push with
   octaves: plus 22 89.4 % (95.1 / 83.8), plus 40 71.3 %, Klickitat 100.0 %, times 9-14 % long.
   The 64-cell apartment does not move (72.8 % with or without octaves, 73.6 % with the strong push; every detour seen through).
   `thick.py`: in the houses lift + push without octaves sees through obstacles thinner than about 1.5-2.5 cells on the straight line
   and respects thicker ones (Klickitat: 100 % seen through at 1-1.5 cells, 14 % at 2.5-4, 4 % at 4-8): wall thickness in cells, not
   scene size, decides where the page's recipe works; the octaves move that threshold below the wall thickness of the houses.
6. **Not adopted:** vector lift h n (62.5 % with the push; with inside pairs 58.6 % / 59.4 %), push scaled with the scene (plus 40:
   index 0.42 at epoch 2,000 with times 15 % long, 64.3 %), base frequencies scaled with the scene (64.9 %, still see-through), a denser
   segment check (the 48-point check misses 0.2-1.6 %), a larger bound (largest T* between free configurations 0.9-1.9 box lengths).
7. **Floor plans.** ntrl-demo is itself see-through on the apartment (65 % / 100 % of detour pairs at 32 / 64 cells) and fails by
   stalling; every lifted arm stays see-through there as well (octaves, one octave higher, strong push, inside pairs). The houses,
   with walls of real thickness and a closed outside, are where the recipe works best (item 3 and item 5).
8. **Open.** Why the apartment (a thin scan outline with free space outside) stays see-through under every arm, strong push and
   octaves included; a push schedule that keeps the octave fields' detours without the 14 % inflation on thin-walled mazes (the state
   at epoch 3,000 of `liftpush_oct8` has both, and is transient); the longest plus-maze queries (32-64 cells) are solved only by
   strong-push fields.

Pitfalls: the one-sided term must be faded in with the push (1,323 rollbacks in 100 epochs otherwise); `plus.install` is once per
process (one plus maze per process in figures and forensics); `report` / `prepare` module names exist in jeffrey/maze_scale, which
`scenes.py` puts first on `sys.path`; the registry-declared inside-pair variants of embed_warp only know its own scenes (no houses).
