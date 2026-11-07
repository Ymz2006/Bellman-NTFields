# Speed-shaped Fourier rows for ntrl-demo

Request (2026-11-04): test the hypothesis that Fourier embeddings work better if they capture the speed across the
environment. Building on the octaves of `jeffrey/grid_sync`: instead of plain sin/cos at set frequencies, learn the
frequencies relevant to the environment. For each (x, y), pick a direction and traverse a line; the value on the line
is a sum of sin/cos that is 0 where speed is high and 1 where it is low. Ground-truth or learned speed may be used.
(x, y) becomes n rows of such sums, one per direction, used as the feature. Try various numbers of terms, starting at
about 20-30 and working up.

Everything below is from registered runs (run ids in `report/tables.md` and on the findings page). Success = ntrl-demo's
MPPI over planning seeds 0/1/2, pooled over training seeds 87358/1/2 unless a single seed is stated.
Findings page: https://claude.ai/artifact/Lovr6s7aNpvLD1SJ5DJnHe (built by `page.py`; republish `report/speed_fourier.html` with `rows.png` and `rays.png`).

## Verdict

The rows capture speed, and capturing it does not help planning. Rows built on the integral of 1 / speed (part 2 below)
undo the damage but do not beat the octaves. Without the octaves, the integral embedding alone plans at the level of
ntrl-demo's random rows (part 3 below). Next to the octaves, ray hit counts do not help; an L1 norm on the integral helps in one
placement, landmark rows, which reach 52.1 % on 22×22 against 45.1 % for empty rows and 49.6 % for the octaves (part 4 below).

| 3 seeds | 22×22 | 13×13 |
|---|---|---|
| ntrl-demo | 23.4 % | 58.9 % |
| 128 octave rows (grid_sync) | 49.6 % | 94.8 % |
| 120 octaves + 16 empty rows (control) | 45.1 % | 92.2 % |
| + 16 speed rows, walls flipped (control) | 42.8 % | 92.0 % |
| + 16 speed rows, 30 terms | 39.1 % | 89.5 % |
| + exact speed, flipped | 41.8 % | 86.0 % (1 seed) |
| + exact speed | 37.1 % | 88.0 % (1 seed) |
| + 16 corridor-distance rows, flipped | 45.4 % | not run |
| + 16 corridor-distance rows | 53.4 % | 91.2 % |

1. **Fit.** R² against o = 1 − speed: 13×13 .53 / .76 / .92 / .95 / .96 at 5 / 10 / 20 / 30 / 60 terms; 22×22 .43 /
   .60 / .81 / .90 / .94. The 20-30 estimate was about right. As the fit improves the rows converge on one map:
   effective rank of 32 rows falls from 3.7 at 5 terms to 1.4 at 30 (13×13).
2. **Planning.** On 22×22, true rows are −6.0 points against empty (−9.0 / −4.2 / −4.9) and −3.7 against flipped
   (−8.7 / +1.3 / −3.6). Exact speed is −8.0 against empty (3 of 3 seeds lower) and −4.7 against flipped (3 of 3 lower).
   On 13×13 true rows are −2.7 against empty and −2.5 against flipped; seed noise there is several points.
3. **Terms** (`top line`, 3 seeds). More terms plan worse.

   | terms | 5 | 15 | 30 | 60 | 120 |
   |---|---|---|---|---|---|
   | 22×22 true | 43.6 | 43.3 | 39.1 | 35.6 | 36.2 |
   | 22×22 flipped | 43.0 | – | 42.8 | 41.0 | – |
   | 13×13 true | 93.2 | 92.8 | 89.5 | 87.2 | 86.8 |
   | 13×13 flipped | – | – | 92.0 | – | – |

4. **Capture does not predict success.** Speed readable after one random layer against success, rank correlation over
   all scored runs: 0.00 on 22×22 (72 runs), +0.23 on 13×13 (66 runs). On 22×22 octaves carry .01 and plan at 49.6 %;
   exact speed carries 1.00 and plans at 37.1 %.

## Mechanism

T(q, g) is a distance between two embeddings through one shared encoder, so two points are far apart in T only if f
separates them. Speed reads the same on both sides of a wall, so speed rows cannot separate them. On 22×22:

- Exact speed makes a ridge (rise 3.4 cells, leak 0) but the difference across walls is .24 of truth against .78 for
  empty rows. Distance rows hold it at .95.
- Freeze test (`ablate.py`: rows of both endpoints pinned to a constant in a trained network). Speed rows: rise 1.00 → 0,
  jump .39 → .36. Exact speed: rise 3.4 → 0, jump .24 → .24. Distance rows: jump .95 → .35, so they carry the
  separation; freezing them also disturbs the field (rise 1.2 → 8.0). Flipped distance rows: jump .78 → .81, and they
  plan like empty rows (45.4 % against 45.1 %).
- Row change across walls against corridor distance, rank correlation: distance rows .79, exact speed .15, speed rows
  −.10 (`wallprofile.py`).
- Separation (T / corridor distance across walls) against success: rank correlation .85 over 42 octave, speed, exact and
  distance runs; +.95 over the 15 term-sweep runs.
- Almost every failure is a collision. Exact speed has the tallest ridge and the most collisions (62.9 % of attempts
  against 54.9 % with empty rows), so a ridge alone does not keep the planner off walls.

Limits of this explanation:

- **Not explained:** why speed rows lower separation (.49 empty, .29 speed, .23 exact) rather than just not adding to
  it. The lower level survives freezing, so the octave rows themselves separate less. My reading is that the ridge
  removes loss pressure near walls. Not tested.
- **Separation is not sufficient:** ntrl-demo has the highest separation (.67) and plans at 23.4 %, with 1.62 bumps per path.
- **13×13 differs.** Jump is above 1 in every arm and separation barely moves with terms (.88 to .86), so it cannot
  explain the fall from 93.2 % to 86.8 %. That fall goes with wall leaks (.04 at 15 terms, .30 at 30; rank −.77) and
  bumps (−.85). Rise drops past 15 terms on both mazes (13×13 2.9 → .29, 22×22 2.4 → 1.00 → .25). Not explained.
- **Distance rows don't help 13×13** (91.2 % against 92.2 % empty and 94.8 % octave), consistent with no missing
  separation there. They use the exact maze graph, so they are a mechanism control, not a method.

## Part 2: rows built on the integral of 1 / speed

Request (2026-11-04): base the rows on the integral of 1 / speed, which is still a sum of sin/cos. Travel time is an integral
over the 30 pairs, so integrate each term. First make the integral work, then try adding the sin/cos terms rather than giving
the matrix of them on top of the integral. No cross table needed. Also try combining the speed rows with the integral.
The user stopped the runs on 2026-11-04 after the 22×22 arms; 19 queued 13×13 and share runs were not started.

**Verdict: the integral removes the damage of the speed rows and adds a little over empty rows. It does not beat the octaves.**

22×22, 3 seeds unless stated. Points against the control are per seed (87358 / 1 / 2).

| rows | success | per seed, of 720 | against its control | flipped (against true on the same seeds when fewer than 3) | jump / frozen | sep | bumps |
|---|---|---|---|---|---|---|---|
| 128 octave rows | 49.6 % | 410 / 327 / 334 | | | .77 | .49 | .32 |
| 120 octaves + 16 empty (control) | 45.1 % | 344 / 318 / 312 | | | .78 | .49 | .35 |
| + 16 speed rows, 1 − speed (part 1) | 39.1 % | 279 / 288 / 277 | −9.0 / −4.2 / −4.9 | 42.8 % | .39 / .36 | .29 | 1.33 |
| + 16 rows of 1 / speed, not integrated | 45.2 % | 353 / 320 / 303 | +1.2 / +0.3 / −1.3 | not run | .54 / .52 | .37 | 1.29 |
| + 16 integral rows, 30 terms (`integral`) | 47.4 % | 358 / 312 / 354 | +1.9 / −0.8 / +5.8 | 42.6 % | .84 / .86 | .54 | .68 |
| + 16 integral rows, exact (`integral`, 0 terms) | 48.1 % | 365 / 342 / 331 | +2.9 / +3.3 / +2.6 | 47.4 % against 50.7 % (1 seed) | .83 / .88 | .51 | .31 |
| + 16 integral rows / line total (`share`, seeds 1, 2) | 44.7 % | – / 331 / 313 | +1.8 / +0.1 | not run | .85 / 1.01 | .55 | .43 |
| + 8 integral + 8 speed rows (`both`) | 42.1 % | 302 / 286 / 322 | −5.8 / −4.4 / +1.4 | 45.4 % against 41.9 % (1 seed) | .45 / .44 | .33 | 1.05 |
| + 16 corridor-distance rows (part 1) | 53.4 % | | +10.1 / +7.8 / +6.9 | 45.4 % | .95 / .35 | .58 | .27 |
| 64 octaves + 128 empty (control) | 36.9 % | | | | .98 | .61 | .35 |
| 64 octaves + 128 per-term integrals (`iterms`) | 30.6 % | 206 / 230 / 224 | −8.8 / −2.8 / −7.5 | 31.1 % against 28.6 % (1 seed) | .25 / .20 | .26 | 6.43 |

1. **Against empty rows.** Exact integral +3.0 points, higher on 3 of 3 seeds. 30-term integral +2.3, higher on 2 of 3.
   The gain is small next to the spread between seeds (octave rows alone span 327 to 410).
2. **Against octaves.** Exact integral −1.5 points (−6.2 / +2.1 / −0.4); 30-term −2.2 (−7.2 / −2.1 / +2.8). The 16 slots
   are worth about as much filled with integrals as with the 8 octave rows they replace.
3. **The gain is the integral's, not the target's.** 1 / speed rows without the integral equal empty rows (45.2 %).
4. **Walls.** True integral rows beat flipped on 3 of 3 seeds on 22×22 (+2.6 / +7.2 / +4.6). On 13×13 (1 seed) flipped is
   ahead, 96.5 % against 93.8 %.
5. **Combining with speed rows** brings the speed rows' loss of separation back (jump .45). **Per-term channels** (the
   matrix) are below their own control, and T has 6.4 local minima per true path (bumps), where a descending planner stalls.
6. **13×13, seed 87358 only:** integral 93.8 %, exact 95.3 %, flipped integral 96.5 %, 1 / speed not integrated 96.7 %,
   both 90.5 %, per-term 76.2 %; octave 94.8 %, empty 90.2 % on that seed. Not separable at one seed.
7. **Fit.** A 30-term integral matches the exact integral with R² .997 on both mazes.

Mechanism. The integral runs along a straight line, and the line goes through walls, so two sides of a wall differ by
the time to cross it, not to go around. Speed is at its floor near every wall (36 % of nodes), so the row climbs as fast
along a wall as across one.

- Screen before training (`screen.py`, `report/screen.md`): row difference across a wall over difference across an open
  passage, adjacent cells. Integral 1.03 (22×22) and 1.05 (13×13), exact 1.04 / 1.06, share 1.16 / 1.34, speed rows
  .95 / 1.01, distance rows 4.27 / 4.40. Windowed and budget-limited integrals stayed at .90 to 1.12, so they were not trained.
- Freeze test: pinning integral rows leaves the jump at .84 → .86 (exact .83 → .88). The octave rows carry the separation.
  Only distance rows carry it themselves (.95 → .35).
- Not explained: why integral rows keep the octaves' separation (jump .84) when speed rows cut it (.39) and plain
  1 / speed rows sit between (.54).

Details: target `--target slow` is 1 / max(speed, 0.012), the label floor. `integral` integrates the fit from the line's
entry into the box, holds its exit value after it; outside the box the fit signal is the line's inside mean. `iterms`
gives a ramp channel plus one channel per term for 4 directions (needs `mix`). `both` puts 1 − speed line rows in the
second 8 slots. Run ids are in `report/tables.md`.

## Part 3: the integral embedding alone, no octaves

Request (2026-11-04): part 2 only added integral rows on top of the octaves; the idea was to get rid of the octaves entirely
and use this embedding alone. Form `pure`: 8 sin/cos rows of the heading alone + 240 table rows, so position reaches the
network only through the rows. `integral`: 240 line directions, one row each. `iterms`: 8 directions × (ramp + 29 terms).
`both`: 120 integral + 120 speed rows.

**Verdict: alone, the integral embedding plans at the level of ntrl-demo's random rows, about 40 points below the octaves.
The per-term matrix and the integral combined with speed rows collapse.**

Training seed 87358 only; successes of 600 (13×13) and 720 (22×22). Paired columns are 13×13 / 22×22.

| embedding | 13×13 | 22×22 | T / T* | jump | leak | direction | bumps | rank |
|---|---|---|---|---|---|---|---|---|
| 128 octave rows (reference) | 94.8 % (569) | 56.9 % (410) | 1.11 / 1.27 | 1.24 / 0.79 | 0.19 / 0.29 | 0.99 / 0.98 | 0.0 / 0.3 | 91 / 109 |
| ntrl-demo's 128 random rows (reference) | 57.8 % (347) | 21.1 % (152) | 1.31 / 1.24 | 1.70 / 0.68 | 0.22 / 0.45 | 0.99 / 0.95 | 0.1 / 1.9 | 14 / 14 |
| exact integral only | 54.7 % (328) | 16.1 % (116) | 1.04 / 0.60 | 1.10 / 0.14 | 0.08 / 0.09 | 0.98 / 0.82 | 1.0 / 16.6 | 6.2 / 5.5 |
| 30-term integral only | 42.8 % (257) | 18.1 % (130) | 1.04 / 0.57 | 1.08 / 0.11 | 0.13 / 0.10 | 0.98 / 0.78 | 1.0 / 17.7 | 6.0 / 5.4 |
| 30-term integral only, walls flipped (control) | 37.3 % (224) | 10.0 % (72) | 1.03 / 0.33 | 1.09 / 0.03 | 0.08 / 0.01 | 0.98 / 0.49 | 1.6 / 27.5 | 6.1 / 5.4 |
| length along the line only (control, `--target unit`) | 40.3 % (242) | 13.2 % (95) | 1.25 / 1.14 | 1.61 / 0.26 | 0.32 / 0.77 | 0.99 / 0.92 | 0.1 / 2.4 | 3.1 / 3.2 |
| 120 integral rows + 120 speed rows | 11.5 % (69) | 6.8 % (49) | 0.17 / 0.11 | 0.01 / 0.01 | 0.00 / 0.00 | 0.16 / 0.11 | 22.4 / 39.8 | 8.1 / 8.6 |
| one channel per sin/cos term, 8 directions × 30 | 6.0 % (36) | 3.3 % (24) | 0.10 / 0.09 | 0.00 / 0.00 | 0.00 / 0.00 | 0.02 / 0.01 | 34.5 / 49.9 | 169 / 187 |
| one channel per term, flipped (control) | 6.2 % (37) | 3.3 % (24) | 0.11 / 0.09 | 0.00 / 0.00 | 0.00 / 0.00 | 0.03 / 0.02 | 36.5 / 49.6 | 170 / 188 |
| speed rows only (part 1) | 9.3 % (56) | 5.0 % (36) | 0.10 / 0.07 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 27.4 / 44.1 | 2.4 / 3.8 |

1. **Against the references.** Exact integral −3.1 / −5.0 points against ntrl-demo and
   −40.1 / −40.8 against the octaves. 30-term integral −15.0 / −3.0 against ntrl-demo.
2. **The walls in the integral help.** Against length-only rows (no walls): +14.4 / +2.9 points.
   Against flipped walls (30 terms): +5.5 / +8.1 points. One seed; only the 13×13 gap over
   length-only is clearly outside seed noise.
3. **Per-term matrix.** 6.0 / 3.3 %, flipped 6.2 / 3.3 %: the failure is the channel form,
   not the maze.
4. **Integral + speed rows.** 11.5 / 6.8 %, near speed rows alone (9.3 / 5.0 %). The speed rows break
   what the integral rows alone achieve.

Mechanism.

- **Low rank.** Each integral row is a ramp with steps at the walls; 240 directions give nearly the same two coordinates.
  Effective rank 6.2 / 5.5 (octaves 91 / 109, ntrl-demo 14 / 14). Position is linearly readable (R² 0.94 / 0.97).
  It is a smooth position code with wall detail and no ladder of scales.
- **Length-only control, 22×22:** smooth and right in scale (T / T* 1.14, 2.4 bumps) but wall-blind (leak 0.77).
  **Real integral, 22×22:** walls appear (leak 0.09) but the field is compressed (T / T* 0.60), jump 0.14
  (octaves 0.79), 16.6 bumps. On 13×13 the exact-integral field is right in scale and separation
  (1.04, jump 1.10) and plans like ntrl-demo.
- **Collapse** (per-term, both, speed-only): speed fit stays 0.96 / 0.95 (per-term), T / T* about .1, jump 0, direction
  cosine about 0. Rank does not explain it (per-term 169 / 187, the highest here). Not explained. All three
  contain channels that follow the local speed pattern; the ramps next to them do not rescue it.

Scores: `mechanism.py`, `ablate.py` (jump, leak), `capture.py` (rank, position). Bumps = local minima of T along the true path
of a long query, per path. Run ids are in `report/tables.md`.

**Withdrawn idea (2026-11-04).** Grid-anchored rows, sin/cos waves that start and end on grid lines (`gridrows.py`, `--form plain
--rows-from anchor | axis | axiscell | lattice`), were implemented and visualized, then cancelled at the user's request before
any full run finished. No results. The code and the visualizer options remain. Octave row visualizer (`octaveview.py`,
`report/octave_rows.html`): https://claude.ai/artifact/8fdmsBYCKvA51jNKuyXA7n

## Part 4: ray hit counts, and a norm on the integral

Request (2026-11-05): instead of the integral, try a step function, in conjunction with the octave Fourier embeddings.
Test 1: shoot rays from a point with a direction; the step embedding is how many times the ray hits an obstacle in front,
where a hit is a passage from free space into an obstacle. Test 2: include the integral, but put an L1 or L∞ norm on it;
generate ideas on where to add the norm, and try them.

Every arm is form `top`: 120 octave rows + 16 new rows. Control: the same network with those 16 slots zero (`topzero`).

**Verdict: one placement of the norm works. Landmark rows (`--scope markint --norm l1`), the L1 norm over 64 directions of
the integral differences between a point and a landmark cell, raise 22×22 planning from 45.1 % to 52.1 %
(+7.2 / +9.3 / +4.4 points per seed). On 13×13, near its ceiling, they show no clear gain. The ray hit counts do not help. A norm added to the distance itself is rejected by
training and breaks planning when forced.**

22×22, training seeds 87358 / 1 / 2, successes of 720 per seed.

| rows next to 120 octaves | 22×22 | per seed | points against empty rows | 13×13 | lost when rows pinned | jump | bumps | screen |
|---|---|---|---|---|---|---|---|---|
| 120 octaves + 16 empty (control) | 45.1 % | 344 / 318 / 312 | – | 92.2 % (3 seeds) | – | 0.78 | 0.4 | – |
| 128 octave rows | 49.6 % | 410 / 327 / 334 | +9.2 / +1.3 / +3.1 | 94.8 % (3 seeds) | – | 0.77 | 0.3 | – |
| + 16 integral rows, no norm (part 2) | 48.1 % | 365 / 342 / 331 | +2.9 / +3.3 / +2.6 | 95.3 % (1 seed) | 52 % | 0.83 | 0.3 | 1.04 |
| + 16 ray hit counts (test 1) | 46.3 % | 355 / 337 / 309 | +1.5 / +2.6 / -0.4 | 90.3 % (1 seed) | 27 % | 0.91 | 0.3 | 1.05 |
| + 16 ray hit counts, walls flipped (control) | 48.4 % | 386 / 327 / 332 | +5.8 / +1.3 / +2.8 | – | 19 % | 0.85 | 0.3 | – |
| integral rows + L1 norm in the distance, trained weight | 45.6 % (1 seed) | 328 / – / – | -2.2 | – | – | 0.80 | 0.5 | – |
| integral rows + L∞ norm in the distance, trained weight | 50.7 % (1 seed) | 365 / – / – | +2.9 | – | – | 0.85 | 0.2 | – |
| integral rows + L1 norm in the distance, weight fixed at 0.2 | 20.6 % (1 seed) | 148 / – / – | -27.2 | – | – | 0.70 | 5.6 | – |
| integral rows + L∞ norm in the distance, weight fixed at 0.2 | 5.7 % (1 seed) | 41 / – / – | -42.1 | – | – | 0.92 | 17.0 | – |
| + 16 landmark rows: L1 norm of integral differences | 52.1 % | 396 / 385 / 344 | +7.2 / +9.3 / +4.4 | 93.3 % (3 seeds) | 64 % | 0.96 | 0.2 | 1.37 |
| same landmark rows without walls (control) | 46.2 % | 348 / 337 / 313 | +0.6 / +2.6 / +0.1 | – | 48 % | 0.79 | 0.3 | – |
| same landmark rows, walls flipped (control) | 45.3 % | 339 / 342 / 297 | -0.7 / +3.3 / -2.1 | – | 40 % | 0.87 | 0.2 | – |
| + 16 landmark rows: L∞ norm of integral differences | 46.9 % | 368 / 309 / 335 | +3.3 / -1.3 / +3.2 | – | 16 % | 0.80 | 0.4 | 1.09 |
| + 16 landmark rows: L1 norm of hit-count differences | 50.2 % | 348 / 383 / 353 | +0.6 / +9.0 / +5.7 | – | 44 % | 0.78 | 1.2 | 1.59 |
| same hit-count landmark rows, walls flipped (control) | 47.6 % | 351 / 351 / 326 | +1.0 / +4.6 / +1.9 | – | 33 % | 0.86 | 0.3 | – |
| + 16 integral rows divided by their L1 norm over directions | 46.2 % (1 seed) | 333 / – / – | -1.5 | – | 40 % | 0.70 | 0.4 | 0.99 |
| + 16 integral rows divided by their L∞ norm over directions | 50.3 % (1 seed) | 362 / – / – | +2.5 | – | 35 % | 0.64 | 0.5 | 1.04 |
| + 16 corridor-distance rows (uses the maze graph) | 53.4 % | 417 / 374 / 362 | +10.1 / +7.8 / +6.9 | 91.2 % (3 seeds) | 65 % | 0.95 | 0.3 | 4.27 |

Lost when rows pinned: share of successes that disappear when the 16 added rows of both endpoints are held constant in the
trained network (`frozenplan.py`). Jump, bumps: as in parts 2 and 3. Screen: row difference across a wall / across an open
passage before training (`rayscreen.py`, `report/rayscreen.md`); 1.0 = the rows do not tell the two sides apart.

### Test 1: ray hit counts (`rays.py`, `--scope hits`)

16 rays, 22.5° apart, counted to the edge of the maze, on the exact wall boxes (checked against `maze.boxes` on 400,000
points and against brute-force counts at every cell centre). 46.3 %, +1.5 / +2.6 / -0.4 points against empty rows.
Walls flipped: 48.4 %; true minus flipped -4.3 / +1.4 / -3.2. No effect of the wall placement. Screen
1.05: one cell of movement changes the counts about as much through an open passage as across a wall.
Not trained: counts within 1, 2 or 4 cells and the distance to the first hit (screen 0.98 to 1.01 on 22×22).

### Test 2: where the norm can go

1. **In the distance** (`--head l1 | linf`, `embed.install_head`): T = T_net + |a| · scale · N(Z(q) - Z(g)), Z the 16 exact
   integral rows, scale = half the median of T* / N over the queries.
   - Weight trained from 1 (seed 87358): ends at 0.0003 (L1) and 0.0000 (L∞). Planning
     328 and 365 of 720; integral rows alone 365.
     AdamW's weight decay (0.5, lr 5e-4, 25,000 steps) alone takes the weight to about 0.002, so this arm alone does not
     prove rejection; the fixed-weight arm settles it.
   - Weight fixed at 0.2 (`--head-weight 0.2`): 20.6 % (L1) and 5.7 % (L∞). Bumps
     5.6 and 17.0 per path against 0.3; plans reaching the goal
     25 % and 7 % against 69 %.
   - Why: T_net ≥ 0, so the term can only be corrected by shrinking a. At a = 1 it exceeds T* on 18 %
     (L1) and 27 % (L∞) of the queries (`headcheck.py`): a straight-line integral crosses walls the true
     path goes around. At 0.2 it exceeds T* on 0.4 % and 2 % and still breaks
     planning: the integral jumps when its line passes a wall end, which gives T local minima.
2. **Against landmarks** (`--scope markint --norm l1 | linf`): 16 landmark cells by farthest-point sampling in straight-line
   distance (no maze graph). Row m = norm over 64 directions of (integral at the point - integral at the landmark centre).
   - L1: 52.1 %, +7.2 / +9.3 / +4.4 against empty rows; pooled above the octaves (49.6 %; per seed
     -1.9 / +8.1 / +1.4) and near the corridor-distance rows (53.4 %), which need the maze graph.
   - Control, no walls (`--target unit`): 46.2 %; true minus control +6.7 / +6.7 / +4.3.
   - Control, walls flipped: 45.3 %; true minus flipped +7.9 / +6.0 / +6.5.
   - L∞: 46.9 %, +3.3 / -1.3 / +3.2 against empty rows.
   - From hit counts instead of integrals (`--scope markhits --norm l1`): 50.2 %, +0.6 / +9.0 / +5.7
     against empty rows; walls flipped 47.6 % (3 seeds), true minus flipped -0.4 / +4.4 / +3.7. A gain from the walls is not established for them. Rougher fields: 1.2 bumps per path against 0.2.
3. **Across directions at a point** (`--scope intnorm`): the 16 integral rows divided by their L1 or L∞ norm. One seed:
   46.2 % and 50.3 %, against 50.7 % for the plain integral rows on that seed.
   Inside seed noise.
4. **Along the ray.** The L∞ norm of 1 / speed along a ray is the largest slowness on it, the same for nearly every ray
   that meets a wall: it collapses to test 1. Not trained separately.
5. **On the network weights.** An L1 penalty on the first layer needs a change in ntrl-demo's training loop. Not run.

### Mechanism of the landmark rows

- **Before training** the rows already differ across walls: screen 1.37 against 1.04 for the 16
  integral rows, 1.09 for the L∞ version, 4.27 for corridor distance. The ordering L1 > L∞
  matches planning; the cause was not tested separately.
- **Trained field:** jump 0.96 (empty rows 0.78, no-walls control 0.79, flipped
  0.87, corridor distance 0.95); bumps 0.2.
- **The network reads the rows when planning:** pinning them removes 64 % of the successes (corridor distance
  65 %, no-walls control 48 %, flipped 40 %, hit counts 27 %, L∞ landmark rows 16 %,
  hit-count landmark rows 44 %).
- **13×13:** 93.3 % (3 seeds) against 92.2 % for empty rows (+4.8 / -2.8 / +1.2 per seed) and
  94.8 % for the octaves on the same seeds. No gain established. Hit counts 90.3 % (1 seed) against 90.2 %. Near the ceiling.

Figure of the rows: `report/rays.png` (`rayfigs.py`). Run ids: `report/tables.md`.

## Other forms (mostly 1 seed)

| form | 13×13 | 22×22 |
|---|---|---|
| 64 octaves + 128 empty (3 seeds) | 93.2 | 36.9 |
| 64 octaves + 128 line rows, true / flipped | 70.8 / 73.2 | 23.1 / 29.4 |
| same, 5 terms | 83.5 | – |
| 64 octaves + 128 ridge rows, true / flipped | 87.7 / 90.8 | 43.8 / 44.6 |
| 64 octaves + 128 single terms, true / flipped | 72.0 / 69.0 | – |
| 64 octaves + 128 copies of the map | 69.5 | – |
| plain sin/cos at the spectrum's 128 peaks | 66.3 | 29.6 |
| speed rows only | 9.3 | 5.0 |

Speed rows only: T/T* about .1 while speed fit stays .93-.97; position has linear R² .16 from these rows, .92 after a
random layer.

## Predictions that failed

- Speed rows would blur which cell a point is in: cell readout is .94 (22×22) and .97 (13×13), the same as empty rows.
- The rows would damage the direction of −∇T: median cosine along true paths moves from .98 to .97 (.94 with exact speed).
- Single-seed figures that did not hold over three seeds: `half` on 13×13 96.7 → 93.2 %; speed-row leak on 22×22 .09 → .21.

## Setup

Mazes, data, planner, scoring, network, losses, optimizer, 5,000 epochs and the network initialization are those of
`jeffrey/maze_scale` and `jeffrey/grid_sync`, unchanged. Prepared runs: 13×13 `20261101T192504Z-13cf1d04`, 22×22
`20261101T193403Z-cea05796`. 600 and 720 planning attempts per run. Only `NN.input_mapping` changes, still 256 inputs.

- **Row.** For each direction, lines across the maze; o = 1 − heading-mean speed along them; the M strongest peaks of
  that direction's spectrum; one least-squares sin/cos pair per peak per line (`linefit.py`). The rows are stored on a
  16-nodes-per-cell table and read through a cubic B-spline in pure torch, because the loss uses ∇T and needs
  double-differentiable features (`embed.py`).
- **Forms** (`--form`): `top` (120 octave rows, every 16th dropped, + 16 rows), `topzero` (those slots zero), `mix` (64
  octaves + 128 rows), `half` (`mix` zeroed), `oracle` (`top` with 16 copies of exact o(x, y, θ)), `pure` (8 heading-only
  rows + 240 rows), `plain` (`--rows-from spectrum`), `geo` (`top` with corridor distance to 16 farthest-point landmark cells).
- **Scopes** (`--scope`): `line`, `ridge`, `terms`, `map`, and for part 2 `integral`, `iterms`, `both`, `share` with `--target slow`. `--target unit` (part 3 control) sets 1 / speed to 1. Part 4 (`rays.py`): `hits` (ray hit counts, `--reach` cells or to the edge), `depth`, `markint` and `markhits` (landmark rows, `--norm l1 | linf`), `intnorm`; `--head l1 | linf` adds a norm of the row differences to T (`--head-weight` fixes its weight). **Controls:** `--walls flip` reads the table at (−x, −y).
- **Learned speed** (`--speed-from`) is implemented but not run: exact speed is the upper bound and does not help.

Scores: `capture.py` (speed and position readable from the features), `screen.py` (row separation before training), `mechanism.py`, `ablate.py`, `wallprofile.py`,
`report.py`, `figs.py`, `page.py`; part 4: `rayscreen.py` (screen before training), `frozenplan.py` (planning with the added rows pinned), `headcheck.py` (size of the head term), `rayfigs.py`. Run `mechanism.py`, `ablate.py` and `wallprofile.py` with `OMP_NUM_THREADS=4`: the
strict embedding digest changes with the thread count, and runs trained with 4.

```bash
~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/run.py --prepared results/registered_20261101T193403Z-cea05796 \
    --seed 87358 --form top --scope line --terms 30 --walls true
```

## Limits

Two mazes, one planner, one network. Line rows are heading-blind, and a heading-blind map holds at most .44 of the 3-D
speed variance; the exact-speed arm covers that gap. Flipped rows in the term sweep cover 5, 30 and 60 terms on 22×22
and 30 on 13×13. Flipped distance rows were not run on 13×13. Part 2 has 3 seeds on 22×22 only; its 13×13 arms have one seed,
the share arm two, and most flipped controls one. Part 3 is one training seed throughout. Part 4 has 3 seeds on 22×22 for the hit counts, the landmark rows and their controls; the head arms and the normalised integral have one seed. Results are modified-environment results on this workstation.
