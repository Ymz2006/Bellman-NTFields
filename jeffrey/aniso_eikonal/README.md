# Anisotropic eikonal on the warp (`jeffrey/aniso_eikonal`)

Request (2026-11-05, about 22:10 EST): "Here is the issue I have with the mazes example. I want a diffeomorphism on the current maze to
'compress' some features, check the plus maze study. However, I can plan on this warped space fine and get a path but it is no
longer the minimal travel time path since the diffeomorphism ruins the travel time. With this diffeomorphism, the position alone is
not enough to determine the speed, but the direction is required as well. Moving in certain directions should cost more. The solution
is an anisotropic formulation, in order to compress the space while keeping an accurate travel time. I think you can implement this
through an inner product on the Lie gradient of T instead of just the magnitude of it. Verify that this idea makes sense, then try
this on the plus mazes. The goal is for this idea to get similar results on just planning on the warped space. Use all GPUs, try for
12 hours." Follow-ups in the same session: start from the optimal constructed warp for the plus mazes and use only constructed warps
(the learned warps are not good yet); use ntrl-demo with no TD / normal loss as the base; 4,000 epochs only.

Redirect (2026-11-06, about 00:45 EST), after a first pass had found that a closed-form "deficit" term added to the stretched-space field
reproduces the travel time: "The idea I want is for this method to be used eventually in conjunction with a learned warp, and you will not
find success with the deficit term because the learned warp can be anisotropic, nonuniform and a lot messier. Treat this like a research
task. If it doesn't work with the biggest gap, try smaller gaps. However, the method you come up with must generalize. Stop the tests
with the deficit term. First get this idea to work, do not be stopped by the neural network's inability to represent large travel times.
Work with the smaller gaps first until you have a result similar to just planning on the stretched. Next, you can try to expand the
output of the neural network [the scalar on the distance head, a constant on the embeddings, anything that raises its maximum travel
time]."

Findings page: https://claude.ai/artifact/K9hXhpLNNXp37JXT4PkS6r (`report/aniso_eikonal.html`, from `page.py`).

## Summary

The request: a warp that compresses the maze makes planning easy, but a field trained on the stretched space no longer tells the maze’s travel time, because there the speed depends on the direction of travel. Fix it with an anisotropic eikonal equation, an inner product on the gradient of T in place of its length, and plan as well as on the stretched space. **The equation is right**: on the warp it is the maze’s own eikonal equation, to rounding on a network and to the grid’s accuracy on exact travel times. **Dropped into ntrl-demo’s loss it does not train**, at any gap, and not because of the network’s output range: things the stretched-space training gets for free are missing, and each has a general fix that uses only the warp’s Jacobian and the poses on the maze.

**With them one field comes close to the stretched space and tells the travel time.** Queries solved, against the stretched-space field trained on the same pairs: 38.5 against 41.6 % (22×22 quadrants) and 48.2 against 55.7 % (10×10) at gap ×4; 35.3 against 33.9 % and 47.8 against 53.7 % at gap ×12, where a distance head with a second scale supplies the output range (two or three training seeds each). On the 22×22 quadrants that is inside the spread between training seeds; on the 10×10 the single field is 7.5 and 5.9 points behind. Its travel time is within 10 % of the truth on 54 and 78 % of queries at gap ×4 and on 72 and 84 % at ×12; the stretched space’s on 2 and 4 %. The totals hide a trade at the widest gap: the single field solves more queries from the open band into a quadrant, and on the 10×10 quadrants far fewer from one quadrant to another (4.2 against 18.7 %). The cause of the second is measured below: for a far goal its travel time stops rising into the walls.

**A second route keeps the stretched-space field and learns the rest**: the head is trained with the plain length, a small unbounded network is added to it, and the inner product trains the sum. It plans 2.8 and 1.3 points below its own head at gap ×12 (33.8 and 47.3 %), with the travel time within 10 % on 77 and 84 % of queries. The correction can also be trained alone on a field that already exists: on the gap study’s full-loss fields at gap ×12 the sum solves 83.1 % of queries where the field itself solves 84.9 % (10×10 quadrants; 53.5 against 58.0 % on the 22×22), and its travel time is within 10 % of the truth on 83 and 73 % of queries.

* **The idea is correct**, and it is the same loss term as training on the maze through φ. The equation was never the obstacle.
* **Its training has blind spots the stretched space does not have**: no pair starts in the compressed region; the equation is satisfied by a field that ignores a compressed direction; and the pair weight drops everything the warp brought close. Open-start pairs, the straight-line lower bound and a compression-aware pair weight fix them, and none of them looks at how the warp was built.
* **With them, the warp’s time unit and a two-scale head, one field comes close to the stretched space and tells the maze’s travel time**: 38.5 against 41.6 % and 48.2 against 55.7 % at gap ×4, 35.3 against 33.9 % and 47.8 against 53.7 % at ×12; travel time within 10 % on 54 and 78 % of queries at ×4 and 72 and 84 % at ×12, where the stretched space has 2 and 4 %. On the 22×22 quadrants the difference is inside the spread between training seeds; on the 10×10 the single field is 7.5 and 5.9 points behind.
* **The output range is a separate problem**, and either a second scale inside the head or an unbounded far term solves it. A scalar on the whole head gives the range and costs the quadrants.
* **At wide gaps the single field loses the walls for far goals.** Its travel time rises into a quadrant’s wall for 55 % of near-wall poses when the goal is in another quadrant (reference 72 %), and quadrant-to-quadrant plans hit a wall in the quadrant they start in. It makes up the total on queries from the open band into a quadrant.
* **A two-part field keeps the walls**: the head trained as on the stretched space, plus a learned correction trained by the inner product on the sum. It plans 2.8 and 1.3 points below its own head at gap ×12 with the travel time within 10 % on 77 and 84 % of queries. Its weak spot is the opposite of the single field’s: diagonal moves through the band’s arms.
* **It attaches to fields that already exist.** The correction alone, trained on the gap study’s frozen full-loss fields, keeps their planning: 80–91 % of queries solved against 81–92 % on the 10×10 quadrants over gaps ×1, ×2, ×4, ×8, ×12 (never more than 1.7 points apart) and 53–61 against 58–60 % on the 22×22 (at most 4.5 points), and gives them the travel time across the band. It needs one care: where the field is already steeper than the equation asks, the correction must be asked for nothing rather than for a negative amount.
* **The base limits everything else here.** Without the temporal-difference and normal terms the stretched-space reference itself solves a third to a half of the queries where the gap study’s full loss had 58–85 %.
* **Not run.** The normal term of ntrl-demo’s full loss supervises exactly what the single field loses at wide gaps, the sign of the slope along a wall’s normal; the base asked for here leaves it out. The single field with that term, and either route on a learned warp, are the next tests.

## 1. The idea, and that it is right

Put a warp z = φ(x) in front of the network, so the travel time is T(x_0, x_1) = H(φ(x_0), φ(x_1)). On the maze the travel time obeys the eikonal equation S(x)·|∇_xT| = 1, with S the speed. The chain rule gives ∇_xT = Jᵀ∇_zH, with J = ∂φ/∂x, so the same equation written on the warp’s coordinates is

    S(x)² · ∇_zHᵀ (J Jᵀ) ∇_zH = 1   instead of   S(x)² · |∇_zH|² = 1

That is the proposal: the squared length of the gradient becomes an inner product with the matrix M = J Jᵀ, positive definite because φ is a diffeomorphism. With Lie gradients it reads the same, c_Hᵀ (J_φJ_φᵀ) c_H = 1 / S², with J_φ the differential of README_warp’s section 5. On the warp the robot’s speed along a unit direction u is S / |J⁻¹u|, so it depends on the direction wherever J is not a multiple of the identity, and the fastest way down the field is −M∇_zH, not −∇_zH. The plain length is the eikonal equation of another world, the stretched maze with the maze’s speeds, which is what the cross-warp study’s “trained on the warped maze” fields learn.

The warp is the gap study’s constructed one, fixed: (x, y, heading) → (F(x), F(y), m·heading), with F magnifying m = 6E/22.1 times over the quadrants and the band’s edge cells and once over the band’s interior (m = 23.9 at gap ×12). The code takes M from the warp’s autograd Jacobian as a full matrix and never uses that this one is diagonal.

(Figure: `report/indicatrix.png`.)

**Checked three ways.** (1) The matrix: J Jᵀ from autograd against central differences of the warp and against its analytic slopes (largest relative difference 1e-15 for the latter). (2) On a network: |∇_xT|² through the warp against ∇_zHᵀ M ∇_zH on the warp’s coordinates, 4,096 random pairs in float64, largest relative difference below 10⁻¹⁵ (logged by every training run). (3) On exact travel times of the gap ×12 maze (Dijkstra on a 1321×1321×13 grid, six goals, central differences at about two million nodes, table below): differentiated on the warp’s own node coordinates with the inner product, the true travel time gives S·‖∇T‖_M = 1.028 in the band’s centre and 1.050 in the quadrants, the same as on the maze’s coordinates (1.028 and 1.050; the 3–5 % above 1 is the grid’s own error). With the plain length it gives 1.05 in the quadrants, 14.2 in the arms (from 5.7 to 23.9 depending on the direction of travel) and 24.2 in the centre: the maze’s travel time does not solve the stretched maze’s equation.

**What the check also shows.** The inner product is, number for number, the eikonal loss taken on the maze through φ, which the gap study ran as its “constructed warp” rows (10 % and 18 % at gap ×12). So the equation was never the missing piece; what the training around it needs is.

| region | T*, on the maze | T*, inner product on the warp | T*, plain length on the warp | Tw, inner product | steepest descent on the warp vs fastest |
|---|---|---|---|---|---|
| quadrants | 1.050 | 1.050 | 1.05 (1.0–1.8) | 1.050 | 0° |
| band edge cells | 1.049 | 1.049 | 14.19 (1.0–23.9) | 0.981 | 37° |
| arms | 1.049 | 1.049 | 14.19 (5.7–23.9) | 0.981 | 55° |
| centre | 1.028 | 1.028 | 24.24 (24.2–24.6) | 0.225 | 9° |

S × gradient norm at random grid nodes of the gap ×12 maze (22×22 quadrants), by region; 1 = the equation holds. T*: the maze’s travel time; T_w: the stretched maze’s.

## 2. What the warp costs when it is ignored

How wrong is a planner that works on the stretched maze and ignores all this? Measured with exact fields, no network involved, on the 300 queries of the gap ×12 maze: read as the maze’s travel time, the stretched maze’s is exact inside a quadrant, 4.7× too short between adjacent quadrants and 15× too short across the open band. Its shortest path, timed on the maze, is 0.5 % slower than the maze’s shortest in the median, 4.0 % at the 90th percentile and 37 % for the worst query: the losses are in the open band, where a straight line on the warp is an L on the maze (figure).

(Figure: `report/paths.png`.)

## 3. Getting the inner product to train

Base, as asked: ntrl-demo’s network and loop with the eikonal term only (no temporal-difference term, no normal term), 4,000 epochs, on the warp’s coordinates. The reference is the same network trained with the plain length (the stretched space) and planned there; “similar” below means against that reference on the same training pairs. Smaller gaps first, as asked: at gap ×4 the largest travel time fits the network’s output comfortably, and the inner product still lost a quarter to a third of the reference’s queries. Five things were wrong, in the order found; each fix is general.

### 3.1 The unit of time

The field may learn c·T for any c; the loss divides the gradient by c. In a 22-cell unit (ntrl-demo’s own unit at gap ×1) the eikonal-only loss fails even at gap ×1, where range is irrelevant: 7.6 % and 9.8 % of queries solved, against 33.8 % and 45.4 % in the unit where the most magnified region’s equation is exactly the stretched space’s (c = the warp’s largest stretch; here one unit = 3.7 cells). At gap ×12, same-quadrant queries: 13.3 % and 22.8 % in the maze-side unit, 48.3 % and 76.1 % in the warp’s unit (stretched space: 38.9 % and 70.6 %). This is the control for the gap study’s falling “constructed warp” curve: same warp, same equation, and inside the quadrants the result follows the unit. Everything below uses the warp’s unit.

### 3.2 Pairs that start in the open

ntrl-demo’s sampler keeps a pair only if its first pose is within 0.175 cells of a wall. On the stretched space that is harmless: the band is small and smooth. For the maze’s travel time the band is where the field is steepest on the warp, and no pair starts inside it. At gap ×4 the field’s implied speed in the band’s centre was 0.73 and 0.86 of the true speed and open–open queries were solved 61.7 % and 63.3 % of the time. Adding 250,000 pairs whose first pose is anywhere in the open (to the 500,000 recipe pairs) makes the band obey the equation: implied speed 0.99 and 0.97, open–open 90.0 % and 91.7 %. The reference is retrained on the same pairs.

### 3.3 A lower bound the equation does not contain

The inner product weights the gradient along a direction by the square of the warp’s stretch there. Along a compressed direction that weight is 1/m² of the weight along a magnified one, so a field that simply does not vary along the compressed direction satisfies the equation almost everywhere. The loss cannot tell it from the travel time. That is what the network learned in the arms of the band (map below): at gap ×4 on the 10×10 quadrants, 15 % of random pairs of band poses had a travel time below 0.9 × their straight-line distance on the maze, which no travel time can be.

The missing information is exactly that bound. The speed is at most 1, so T(a, b) ≥ |x_a − x_b| for any two poses, with the distance taken on the maze. It needs no labels and no knowledge of the warp beyond the poses the pairs were drawn at. Added as a hinge on each batch’s own pairs and on its starts paired with shuffled ends (weight 10), it removes the violations (0 % and 0 %) and brings open–open to 100.0 % and 98.3 %. Alone, without the open-start pairs, it gives 30.2 % and 45.8 % of all queries; the two together 37.9 % and 44.9 %.

(Figure: `report/fields.png`.)

### 3.4 A pair weight that sees across the band

ntrl-demo weights each pair by exp(−T/2). On the stretched space two quadrants are near each other, so pairs between them count. With the maze’s travel time they are far, their weight is close to zero, and the field inside one quadrant for a goal in another is not trained: quadrant-to-quadrant queries were solved 2.5 % of the time on the 10×10 quadrants at gap ×4 against 30.3 % for the reference. The fix keeps ntrl-demo’s weight and shortens the time by how much the warp compressed the pair: exp(−T·ρ/2), ρ = |z_a − z_b| / (c·|x_a − x_b|). Inside a fully magnified region ρ = 1 and nothing changes; across a compressed one the pair weighs as it would on the stretched space. Quadrant-to-quadrant: 6.9 % (22×22: 5.0 → 8.3 %, reference 13.0 %). Weighting by the distance on the warp alone, without the time, destroys the quadrants (22.3 % and 26.7 % of all queries).

### 3.5 A distance head with more than one scale

The pair weight left quadrant-to-quadrant at 6.9 % against 30.3 %. The head is the likely cause (the section on walls below measures it). ntrl-demo’s distance is a sum over 16 groups of the largest feature difference in each group. For a goal across the band a group can be won by a feature that carries the band, which is constant inside the start’s quadrant, and the features that know the way out of the quadrant are then masked. Consistent with that, the field’s direction at the start is 54° off for adjacent-quadrant queries where the reference is 34° off, although for open–open queries it is 15° against 34°. Giving the head two scales, eight groups as they are and eight with their features multiplied by a factor, so that coarse groups can carry the band and fine ones the quadrants, helps where it was worst: quadrant-to-quadrant 17.9 % and 10.9 % at gap ×4 (reference 30.3 % and 13.0 %), all queries 48.2 % and 38.5 %.

**The same head lifts the output range.** At gap ×12 the true travel time reaches 34 units and the plain network’s largest output over the 300 queries is 17 (16 of 32 on the 10×10 quadrants): ntrl-demo’s head cannot exceed 25.6 and is about 10 between unrelated points. With the first four fixes and the head left alone, the travel time is within 10 % on 23 % of queries and 16 % of band pairs are still below the straight-line bound (22×22 quadrants). A scalar on the whole head (0.2 → 0.8, the first thing suggested) does give the range, travel time within 10 % on 78 % and 59 % of queries, but the quadrants pay for it: same quadrant 32.8 % and 45.0 %, all queries 27.8 % and 36.9 %. A factor inside the head, on the features of eight groups (×6), gives the range without that: travel time within 10 % on 72 % and 84 %, same quadrant 31.4 % and 58.9 %, all queries 35.3 % and 47.8 %. Other layouts are within a training seed’s noise of it on the 10×10 quadrants (twelve coarse groups 52.6 %, factor 3 50.0 %, a 1-2-4-8 ladder 46.3 %, factor 12 42.0 %); a soft, near-L1 head is worse (37.0 %).

| 22×22 quadrants, gap ×4 | all (%) | same quadrant | open–open | quadrant ↔ quadrant | T within 10 % (%) | speed, centre |
|---|---|---|---|---|---|---|
| inner product alone | 28.1 | 37.2 | 61.7 | 1.7 | 20 | 0.73 |
| + lower bound | 30.2 | 38.9 | 72.2 | 1.9 | 35 | 0.72 |
| + open-start pairs | 32.2 | 27.2 | 90.0 | 5.8 | 33 | 0.99 |
| + both | 37.9 | 31.7 | 100.0 | 5.0 | 55 | 0.95 |
| + pair weight | 37.0 | 32.8 | 98.3 | 8.3 | 63 | 0.96 |
| + two-scale head | 38.5 | 28.5 | 98.5 | 10.9 | 54 | 0.95 |
| reference: stretched space, same pairs | 41.6 | 31.1 | 100.0 | 13.0 | 2 | 4.41 |

| 10×10 quadrants, gap ×4 | all (%) | same quadrant | open–open | quadrant ↔ quadrant | T within 10 % (%) | speed, centre |
|---|---|---|---|---|---|---|
| inner product alone | 34.0 | 65.6 | 63.3 | 3.3 | 31 | 0.86 |
| + lower bound | 45.8 | 66.1 | 83.3 | 1.7 | 38 | 0.76 |
| + open-start pairs | 43.0 | 57.2 | 91.7 | 6.7 | 54 | 0.97 |
| + both | 44.9 | 54.4 | 98.3 | 2.5 | 76 | 0.93 |
| + pair weight | 42.2 | 51.1 | 98.3 | 6.9 | 78 | 0.96 |
| + two-scale head | 48.2 | 54.1 | 99.1 | 17.9 | 78 | 0.96 |
| reference: stretched space, same pairs | 55.7 | 67.0 | 99.4 | 30.3 | 4 | 5.17 |

Training seed 87358; the last two rows are pooled over the training seeds run. Each row adds one ingredient to the row above (the reference row does not use the inner product). Queries solved by MPPI on the warp; quadrant ↔ quadrant: adjacent and diagonal pooled; T within 10 %: share of the 300 queries whose field travel time is within 10 % of the true one; speed, centre: the speed the field implies on the maze over the true speed in the band’s centre.

## 4. Against planning on the stretched space

Queries solved, planner on the warp, over gaps ×1 to ×12: the single field with the full recipe 32–38 % (22×22 quadrants) and 41–58 % (10×10); the stretched-space field trained on the same pairs 34–42 % and 42–60 %; the inner product alone 23–34 % and 30–45 %. At gap ×4: 38.5 against 41.6 % and 48.2 against 55.7 % (three training seeds for the recipe, three training seeds for the reference). At gap ×12: 35.3 against 33.9 % and 47.8 against 53.7 % (two training seeds and three training seeds). On the 22×22 quadrants the two are within the spread between seeds at every gap (ranges over seeds at ×4: 37–40 and 37–45 %; the largest difference, 5.1 points at ×8, is between two single runs). On the 10×10 quadrants the single field is behind at ×4 and ×12, at the low end of the reference’s range (43–52 against 51–64 % and 47–48 against 48–59 %), and within three points either way at ×1, ×2 and ×8 (57.6 against 60.3, 54.9 against 51.3 and 41.0 against 42.0 %, one seed each).

**The totals are made differently.** At gap ×12, from the open band into a quadrant the single field solves 41.9 against 27.0 % and 75.3 against 64.3 %; from quadrant to quadrant 2.1 against 3.0 % and 4.2 against 18.7 %. Between the 8×8 quadrants of the 22×22 maze nobody solves much. Between the 5×5 quadrants of the 10×10 the reference does and the single field does not, and the next section finds the reason. At gap ×4 on the 10×10 quadrants it is also behind inside a quadrant (54.1 against 67.0 % of same-quadrant queries), which that reason does not cover.

**How much is noise.** The head inside a two-part run (below) is a stretched-space field trained from the same initial weights, on the same pairs, with the same loss as the reference run; no gradient from the correction reaches it, and only the order of the batches differs. So there are several trainings of the same thing. With seed 87358 they solve 54.3, 48.9 and 53.4 % at gap ×12 on the 10×10 quadrants, 31.6, 34.9 and 39.0 % on the 22×22; at gap ×4, 52.3, 47.4 and 54.8 % (10×10) and 37.3, 38.8 and 40.6 % (22×22). Single runs of one recipe differ by five points and more. The whiskers in the charts show the range over training seeds where three were run; a difference inside that range says nothing.

**Planned on the maze itself.** A field that tells the maze’s travel time can be descended on the maze with one step size and one temperature, which needs the warp but not its inverse. Gap ×4: 39.1 % and 50.4 % for the recipe, 40.0 % and 56.1 % for the stretched-space field. Gap ×12: 34.7 % and 51.9 % against 28.1 % and 56.6 % (training seed 87358).

All queries solved against the gap (300 certified queries per maze, 60 per region pair, planning seeds 0/1/2), MPPI on the warp’s coordinates, scored on the maze; pooled over training seeds, the whisker is the range over them. The recipe and its reference train on the same pairs (recipe pairs drawn on the warped maze plus open-start pairs); the inner product alone on the recipe pairs. Hover for the numbers.

## 5. Where the plans between quadrants fail: walls are not felt from afar

Between quadrants the single field falls behind as the band grows. The plans fail where they start: of the 114 failed plans between quadrants at gap ×12 on the 10×10 quadrants (planning seed 0), 97 touch a wall before they have left the start quadrant (66 of 95 for the reference).

MPPI compares the field’s values at the points of its rollouts and nothing else, so it keeps off a wall only where the travel time rises as the pose moves into the wall’s margin. The eikonal term sets the size of the gradient there. It says nothing about its sign along the wall’s normal. The charts count, over 18,000 poses inside the margin of the quadrants’ walls, how often the travel time rises into the wall. For a goal in the same quadrant every field does it for most poses at every gap (10×10 quadrants, gap ×12: 73 % for the recipe, 76 % for the reference). For a goal in another quadrant the stretched-space field keeps doing it (73 % at gap ×4, 72 % at ×12) and the inner product stops as the band grows: alone, 70 % at gap ×1 and 59 % at ×4 (51 % on the 22×22 quadrants; 50 % is a coin flip); with the recipe 72 % at ×4 and 55 % at ×12. The slope has the right size at those poses and either sign: at gap ×12 its quartiles along the wall’s normal are −0.52 and +0.71 for the recipe, +0.07 and +0.97 for the reference (+1 is the true slope out of the margin).

**Why.** ntrl-demo’s head adds, over 16 groups, the largest feature difference in each group. On the maze’s clock a goal across the band is far, and the groups are won by the features that carry the band. A wall can then only show as a ripple on the winner, and whether a ripple raises or lowers the travel time depends on which side the goal lies. On the stretched space the band is short, no feature wins everywhere, and the walls keep features of their own. The two-scale head of fix 5 is this remedy in part: at gap ×4 it brought the walls back (56 → 72 % on the 10×10 quadrants, 58 → 62 % on the 22×22), at ×12 it does not.

| field | ×1 | ×2 | ×4 | ×8 | ×12 |
|---|---|---|---|---|---|
| 22×22 quadrants · trained on the stretched space (plain length), same pairs · goal in same quadrant | 71 | 69 | 67 | 69 | 66 |
| 22×22 quadrants · trained on the stretched space (plain length), same pairs · goal in another quadrant | 67 | 64 | 63 | 65 | 63 |
| 22×22 quadrants · inner product, one field, full recipe · goal in same quadrant | 70 | 64 | 64 | 63 | 62 |
| 22×22 quadrants · inner product, one field, full recipe · goal in another quadrant | 68 | 65 | 62 | 59 | 58 |
| 22×22 quadrants · two-part field: stretched-space head + learned correction · goal in same quadrant | – | – | 67 | 67 | 67 |
| 22×22 quadrants · two-part field: stretched-space head + learned correction · goal in another quadrant | – | – | 63 | 62 | 65 |
| 22×22 quadrants · inner product alone · goal in same quadrant | 68 | 69 | 69 | – | 69 |
| 22×22 quadrants · inner product alone · goal in another quadrant | 60 | 59 | 51 | – | 43 |
| 10×10 quadrants · trained on the stretched space (plain length), same pairs · goal in same quadrant | 83 | 79 | 81 | 74 | 76 |
| 10×10 quadrants · trained on the stretched space (plain length), same pairs · goal in another quadrant | 79 | 75 | 73 | 69 | 72 |
| 10×10 quadrants · inner product, one field, full recipe · goal in same quadrant | 83 | 74 | 76 | 71 | 73 |
| 10×10 quadrants · inner product, one field, full recipe · goal in another quadrant | 79 | 69 | 72 | 62 | 55 |
| 10×10 quadrants · two-part field: stretched-space head + learned correction · goal in same quadrant | – | – | 78 | 78 | 75 |
| 10×10 quadrants · two-part field: stretched-space head + learned correction · goal in another quadrant | – | – | 74 | 68 | 72 |
| 10×10 quadrants · inner product alone · goal in same quadrant | 82 | 80 | 82 | – | 80 |
| 10×10 quadrants · inner product alone · goal in another quadrant | 70 | 67 | 59 | – | 43 |

Share of near-wall poses at which the field’s travel time rises as the pose moves towards the wall. Poses: inside the quadrant mazes, clearance 0.02 to 0.175 cells (the speed margin), 6,000 per clearance band; goal: a free pose in the same quadrant (left) or in another quadrant (right). Training seed 87358. At 50 % the field does not tell a wall from free space.

**Two more things tried on the single field at gap ×12; neither is kept.** (a) A far term: a small smooth network on the warp’s coordinates with no upper limit, T_F = scale·|e(z_a) − e(z_b)|, added to the head and trained with it. It takes the output range off the head: travel time within 10 % on 78 and 85 % of queries with the head as ntrl-demo has it. For adjacent goals the walls come back (67 % on the 10×10 quadrants), but the field’s direction at the start gets worse (55° off against 52°) and planning does not move: 33.0 and 45.9 % of all queries. (b) Two of the head’s 16 groups added up instead of entering the maximum, so that what a wall adds cannot be masked: adjacent-quadrant queries go from 5.3 to 14.4 % on the 10×10 quadrants and same-quadrant queries from 58.9 to 53.3 %; all queries 31.3 and 43.9 %.

## 6. The other route: the stretched space’s field plus a learned correction

If one bounded head cannot hold a long band and a fine maze at once, it need not. Write the travel time as T = H + F. **H** is ntrl-demo’s network and head, trained exactly as on the stretched space: plain length, ntrl-demo’s pair weight, and no gradient from anything else. **F** is the far term (a three-layer network, 128 wide, eight outputs, on the warp’s coordinates, no Fourier features, no upper limit), trained by the inner-product equation on the sum with the head’s gradient held fixed, and by the lower bound of fix 3. In the time unit of fix 1 the two equations are the same wherever the warp magnifies fully, so there F has nothing to add and the walls stay the head’s: with a goal in another quadrant the travel time rises into the wall for 65 and 72 % of near-wall poses at gap ×12, the reference’s level.

This has the shape of the first pass’s deficit term (the stretched maze’s time plus what the warp removed), which was set aside because its formula needs the constructed warp. Here there is no formula. F is a network; its scale is read from the smallest eigenvalue of J Jᵀ over the training poses; what it learns is whatever the inner-product equation asks for on top of H.

**Result.** Every run is planned twice, with the sum and with its head alone. The head alone is a stretched-space field trained on the same pairs from the same initial weights, so the pair is a controlled comparison. Over gaps ×4 to ×12 the sum solves 34–37 % of queries on the 22×22 quadrants where its head solves 37–41 %, and 46–54 % against 48–56 % on the 10×10. Gap ×12: 33.8 against 36.7 % and 47.3 against 48.6 % (two training seeds); gap ×4: 37.4 against 41.1 % and 54.3 against 56.2 %. The sum’s travel time is within 10 % of the truth on 77 and 84 % of queries at gap ×12 (65 and 78 % at ×4). Between quadrants at ×12 it solves 3.3 and 10.1 %, the single field 2.1 and 4.2 %. Planned on the maze instead of on the warp (training seed 87358): 27.2 and 47.3 % at gap ×12, 31.7 and 46.7 % at ×4.

**What it costs.** The sum plans a few points below its own head, and most of the loss is in one kind of query, from the open band into a quadrant: 29.7 against 38.6 % and 59.2 against 65.8 % at gap ×12. On those queries the single field solves 41.9 and 75.3 %. The likely reason is the form. In the band’s arms the warp compresses one direction only; a time along the arm and a time across it add up in H + F like the two legs of an L, where the maze’s travel time is the straight line (the diamonds in the last panel of the field maps above). Diagonal moves are over-priced there by up to 41 % (the field’s implied speed in the arms is 0.89 and 0.93 of the true one). Consistent with that, plans on the sum first reach the goal quadrant’s outline a median of 2.7 and 2.1 cells from where the true path enters it, plans on the head alone 1.7 and 1.6 cells. It is not the whole story: the single field arrives as far off (2.3 and 1.9 cells) and still solves the most. A correction that is a metric of its own cannot change the L.

**A second form, tried for that.** F = |u(z_a) − u(z_b)| − |z_a − z_b|, with u a learned map of the warp’s coordinates that starts as the identity: the straight distance in a learned embedding minus the straight distance on the warp. With u the warp undone this is exact for straight travel in the open, diagonals through the arms included, and it is again learned from the inner-product loss alone. It does repair the arms (implied speed 0.95 and 0.96 of the true one at gap ×12) and it can damage the quadrants: on the 10×10 quadrants the sum solves 38.4 % where its head solves 53.4 % at gap ×12 (same-quadrant queries 32.8 against 57.2 %), and on the 22×22 quadrants 31.2 against 40.6 % at gap ×4; the other two cases are close (36.7 against 39.0 % and 52.7 against 54.8 %). A difference of two straight-line distances keeps a slope inside a quadrant, where the correction should have none; the separate metric has none by construction. The first form is the one kept.

| maze, gap | seeds (two-part) | two-part field (%) | its head alone (%) | single field, recipe (%) | reference run (%) | T within 10 %: two-part | … single field | … reference |
|---|---|---|---|---|---|---|---|---|
| 22×22 quadrants ×4 | 2 | 37.4 | 41.1 | 38.5 | 41.6 | 65 | 54 | 2 |
| 22×22 quadrants ×8 | 1 | 35.3 | 37.7 | 32.3 | 37.4 | 75 | 62 | 1 |
| 22×22 quadrants ×12 | 2 | 33.8 | 36.7 | 35.3 | 33.9 | 77 | 72 | 0 |
| 10×10 quadrants ×4 | 2 | 54.3 | 56.2 | 48.2 | 55.7 | 78 | 78 | 4 |
| 10×10 quadrants ×8 | 1 | 46.0 | 47.6 | 41.0 | 42.0 | 81 | 82 | 3 |
| 10×10 quadrants ×12 | 2 | 47.3 | 48.6 | 47.8 | 53.7 | 84 | 84 | 5 |

Queries solved by MPPI on the warp, pooled over training seeds and planning seeds 0/1/2. “Its head alone”: the same runs planned with the correction switched off. Reference run: the stretched-space field of the main chart (training seed 87358 unless three seeds were run). T within 10 %: share of the 300 queries whose field travel time is within 10 % of the true one.

**On a field that already exists.** The head is not touched, so the correction can be trained alone on any stretched-space field. The gap study’s full-loss fields (temporal-difference and normal terms, so not this study’s base) were frozen and only F was trained, 4,000 epochs, 8 to 15 minutes each, with one change to its term. These fields run slow inside the quadrants by themselves (same-quadrant travel time 1.65× the truth on the 22×22 quadrants, 1.22× on the 10×10), so there the head alone is already steeper than the equation asks, and a correction that is a metric cannot take slope away. Where that is so, the sum’s target is the head’s own slope: the correction is asked to add nothing.

On the 10×10 quadrants the sum then solves 91.0, 85.6, 86.8, 80.3, 83.1 % of queries at gaps ×1, ×2, ×4, ×8, ×12, the fields themselves 91.8, 86.7, 84.2, 81.3, 84.9 %: never more than 1.7 points apart (three training seeds at ×12, one at the other gaps). On the 22×22 quadrants: 55.9, 58.3, 61.1, 57.4, 53.5 % against 57.8, 58.9, 60.1, 58.7, 58.0 %, at most 4.5 points apart. The travel time is within 10 % of the truth on 27, 53, 70, 77, 83 % of queries (10×10) and 3, 19, 46, 69, 73 % (22×22), where the stretched-space fields alone are at a few per cent once the gap is wide. The correction restores what the warp removed and nothing else: the fields’ own slowness inside the quadrants stays, and at small gaps most of a query’s time is spent there.

**Control: the plain two-sided term.** Without the change the sum lost up to 12.0 points on the 22×22 quadrants (46.3 against 58.3 % at gap ×8), on open–open queries too (73.9 %), which nothing else in this study fails. Those plans stall in the open band without touching a wall, on a flat spot of the summed field: 41 of the 120 plans into or inside the band at gap ×8 (planning seed 0), 11 with the changed term. Asked to cancel slope it cannot cancel, the correction folds.

| maze, gap | full-loss field + learned correction (%) | the full-loss field itself (%) | T within 10 % with the correction (%) | control: plain two-sided term (%) |
|---|---|---|---|---|
| 22×22 quadrants ×1 | 55.9 | 57.8 | 3 | 58.7 |
| 22×22 quadrants ×2 | 58.3 | 58.9 | 19 | 59.3 |
| 22×22 quadrants ×4 | 61.1 | 60.1 | 46 | 54.8 |
| 22×22 quadrants ×8 | 57.4 | 58.7 | 69 | 46.3 |
| 22×22 quadrants ×12 | 53.5 | 58.0 | 73 | 50.4 |
| 10×10 quadrants ×1 | 91.0 | 91.8 | 27 | 92.2 |
| 10×10 quadrants ×2 | 85.6 | 86.7 | 53 | 88.0 |
| 10×10 quadrants ×4 | 86.8 | 84.2 | 70 | 83.4 |
| 10×10 quadrants ×8 | 80.3 | 81.3 | 77 | 81.4 |
| 10×10 quadrants ×12 | 83.1 | 84.9 | 83 | 82.3 |

The gap study’s stretched-space fields trained with ntrl-demo’s full loss (dashed blue: planned as they are) and the same fields with the far term trained on top of them by the inner-product loss, the field frozen (solid). Dotted: the control, the correction trained with the plain two-sided term. Top: queries solved by MPPI on the warp. Bottom: share of queries whose travel time is within 10 % of the truth (the stretched-space field alone is at 1–6 % once the gap is wide). Training seed 87358; at gap ×12 the solid and dashed lines pool three training seeds.

## 7. Does the field tell the maze’s travel time?

Each point is one of the 300 queries at gap ×12: the field’s travel time from start to goal against the true one. The stretched-space field lies far below the diagonal for every query that crosses the band (median ratio 0.27 between adjacent quadrants, 0.05 open–open). The inner product alone bends over (0.40 between diagonal quadrants). With the recipe the cloud lies on the diagonal: 1.08 adjacent, 1.01 diagonal, 0.99 open–open, within 10 % of the truth on 72 % of all queries (84 % on the 10×10 quadrants). The two-part field’s cloud lies on the diagonal as well (77 and 84 % within 10 %). The charts under the scatter give that share at every gap.

What is left is inside the quadrants, where every field here is slow by the same margin (same-quadrant ratio 1.43 for the stretched space, 1.45 for the recipe): the eikonal-only base, not the warp.

**Direction.** At a query’s start, the angle between the field’s descent direction on the maze and the true one, gap ×4, median: open–open 15° and 13° for the recipe against 34° and 34° for the stretched space; same quadrant 31° and 31° against 28° and 26°. The stretched-space field points along the warp’s geometry; the recipe’s points along the maze’s.

## 8. Nothing here uses the constructed warp’s structure

The request’s end goal is a learned warp, which will be anisotropic, non-uniform and not separable. Every ingredient above is written for that case: the loss takes J Jᵀ from the warp’s autograd Jacobian as a full matrix; the time unit is the warp’s largest stretch; the lower bound and the pair weight need only each pose on the maze and on the warp; the open-start pairs and the head know nothing about the warp. A closed-form correction that exists for this constructed warp (the distance the warp removed, added back; it reproduces the travel time to 1 % on exact fields) was set aside for that reason: it needs the compressed region to be open and the warp to be separable.

**Test: a messy warp.** The constructed warp followed by a smooth random displacement of (x, y), eight plane waves, Jacobian up to 0.6 in norm: sheared, locally anisotropic everywhere, not separable (still fixed, not learned). Its Jacobian is full; the chain-rule identity holds to 10⁻¹⁵ on it. Gap ×4, same recipe, nothing retuned: 34.4 % and 41.1 % of queries solved (constructed warp, same training seed: 37.3 % and 49.2 %), travel time within 10 % on 61 % and 79 % of queries; the plain length on the same messy warp solves 39.4 % and 49.9 % with travel time within 10 % on 2 % and 4 %. Planned on the maze itself, which needs the warp but not its inverse: 39.1 % and 43.0 % for the recipe, 36.7 % and 40.9 % for the plain length. One training seed each.

**The two-part field on the same messy warp.** No formula for the correction exists here, and the warp magnifies fully almost nowhere, so the correction has work to do everywhere; the time unit is this warp’s own largest stretch, read from its Jacobian. Queries solved with the sum: 29.3 and 42.0 % on the warp, 33.9 and 42.6 % on the maze; with its head alone 41.3 and 46.0 %. Travel time within 10 % of the truth on 74 and 87 % of queries. The travel time carries over to the messy warp; the cost in planning is larger than on the constructed warp on the 22×22 quadrants.

| gap ×4, training seed 87358 | MPPI on the warp (%) | MPPI on the maze (%) | T within 10 % (%) | open–open T / T* | implied speed, arms | … centre |
|---|---|---|---|---|---|---|
| 22×22 quadrants · constructed warp · recipe | 38.5 | 39.1 | 54 | 0.99 | 0.92 | 0.95 |
| 22×22 quadrants · constructed warp · stretched space | 41.6 | 40.0 | 2 | 0.36 | 1.16 | 4.41 |
| 22×22 quadrants · messy warp · recipe | 34.4 | 39.1 | 61 | 1.00 | 0.92 | 0.96 |
| 22×22 quadrants · messy warp · stretched space | 39.4 | 36.7 | 2 | 0.36 | 1.19 | 4.34 |
| 22×22 quadrants · messy warp · two-part field | 29.3 | 33.9 | 74 | 0.97 | 0.97 | 1.03 |
| 10×10 quadrants · constructed warp · recipe | 48.2 | 50.4 | 78 | 0.99 | 0.96 | 0.96 |
| 10×10 quadrants · constructed warp · stretched space | 55.7 | 56.1 | 4 | 0.20 | 1.26 | 5.17 |
| 10×10 quadrants · messy warp · recipe | 41.1 | 43.0 | 79 | 0.98 | 0.96 | 0.97 |
| 10×10 quadrants · messy warp · stretched space | 49.9 | 40.9 | 4 | 0.21 | 1.29 | 4.93 |
| 10×10 quadrants · messy warp · two-part field | 42.0 | 42.6 | 87 | 0.96 | 0.99 | 1.03 |

## 9. What this says

* **The idea is correct**, and it is the same loss term as training on the maze through φ. The equation was never the obstacle.
* **Its training has blind spots the stretched space does not have**: no pair starts in the compressed region; the equation is satisfied by a field that ignores a compressed direction; and the pair weight drops everything the warp brought close. Open-start pairs, the straight-line lower bound and a compression-aware pair weight fix them, and none of them looks at how the warp was built.
* **With them, the warp’s time unit and a two-scale head, one field comes close to the stretched space and tells the maze’s travel time**: 38.5 against 41.6 % and 48.2 against 55.7 % at gap ×4, 35.3 against 33.9 % and 47.8 against 53.7 % at ×12; travel time within 10 % on 54 and 78 % of queries at ×4 and 72 and 84 % at ×12, where the stretched space has 2 and 4 %. On the 22×22 quadrants the difference is inside the spread between training seeds; on the 10×10 the single field is 7.5 and 5.9 points behind.
* **The output range is a separate problem**, and either a second scale inside the head or an unbounded far term solves it. A scalar on the whole head gives the range and costs the quadrants.
* **At wide gaps the single field loses the walls for far goals.** Its travel time rises into a quadrant’s wall for 55 % of near-wall poses when the goal is in another quadrant (reference 72 %), and quadrant-to-quadrant plans hit a wall in the quadrant they start in. It makes up the total on queries from the open band into a quadrant.
* **A two-part field keeps the walls**: the head trained as on the stretched space, plus a learned correction trained by the inner product on the sum. It plans 2.8 and 1.3 points below its own head at gap ×12 with the travel time within 10 % on 77 and 84 % of queries. Its weak spot is the opposite of the single field’s: diagonal moves through the band’s arms.
* **It attaches to fields that already exist.** The correction alone, trained on the gap study’s frozen full-loss fields, keeps their planning: 80–91 % of queries solved against 81–92 % on the 10×10 quadrants over gaps ×1, ×2, ×4, ×8, ×12 (never more than 1.7 points apart) and 53–61 against 58–60 % on the 22×22 (at most 4.5 points), and gives them the travel time across the band. It needs one care: where the field is already steeper than the equation asks, the correction must be asked for nothing rather than for a negative amount.
* **The base limits everything else here.** Without the temporal-difference and normal terms the stretched-space reference itself solves a third to a half of the queries where the gap study’s full loss had 58–85 %.
* **Not run.** The normal term of ntrl-demo’s full loss supervises exactly what the single field loses at wide gaps, the sign of the slope along a wall’s normal; the base asked for here leaves it out. The single field with that term, and either route on a learned warp, are the next tests.

## 10. How it was measured

* **Mazes and queries.** The gap study’s: the four 8×8 quadrants of the 22×22 cross maze, and the 10×10 maze cut into four 5×5 quadrants, pushed apart by an open band of 6, 12, 24, 48 and 72 cells. 300 certified queries per maze, 60 per region pair, planning seeds 0/1/2; a query is solved if MPPI arrives and the exact scorer finds no contact.
* **Warp.** The gap study’s constructed warp (`warp.constructed`), fixed; the messy warp of the generality test adds a smooth random displacement to it (`aniso.MessyWarp`).
* **Training.** ntrl-demo’s network and loop, eikonal term only, 4,000 epochs. Recipe pairs: 500,000 drawn on the warped maze (the gap study’s sets); open-start pairs: 250,000 more whose first pose is anywhere beyond the wall margin, drawn on the maze. Speeds are the maze’s.
* **The recipe (one field).** Inner product with J Jᵀ; time unit = the warp’s largest stretch; open-start pairs; lower-bound hinge, weight 10; pair weight exp(−Tρ/2); head with eight groups as they are and eight multiplied by 4 (gaps ×1 to ×4) or 6 (gaps ×8, ×12).
* **The two-part field.** Same pairs and time unit. Head: ntrl-demo’s, plain-length eikonal term, pair weight exp(−H/2). Correction: scale·|e(z_a) − e(z_b)|, e a 3-128-128-8 network with softplus units, scale = c / √(smallest eigenvalue of J Jᵀ) at the 1st percentile of the training poses, no weight decay; inner-product term on the sum with the head’s gradient detached, same pair weight, and the lower-bound hinge (weight 10) on the sum. On a frozen field (the gap study’s full-loss fields): only the correction is trained, without ntrl-demo’s rollback rule, and the sum’s target slope is the larger of the equation’s and the head’s own.
* **Planner.** maze_scale’s MPPI on the warp’s coordinates (step and tolerance × the quadrants’ magnification; path mapped back exactly and scored on the maze). Every field is planned in the maze’s time unit.
* **Truth.** T* by Dijkstra on maze_scale’s C-space grid (1/15 cell, 13 headings).
* **Not the pinned environment.** Host venv, RTX 3090 and RTX 5070; every run is registered (table below). One training seed unless a whisker is drawn.

## 11. Every field at every gap

| maze, gap, variant | seeds | planner on the warp | planner on the maze | same quadrant | adjacent quadrants | diagonal quadrants | quadrant – open | open – open | never arrive / arrive through a wall | T / T* (median) | within 10 % (%) | path time / T* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22×22 quadrants ×1 · zaniso-U22.1 | 1 | 7.6 | – | 6 | 0 | 0 | 0 | 32 | 797 / 35 | 0.76 | 23 | 1.122 |
| 22×22 quadrants ×1 · zaniso-Uquad | 1 | 33.8 | – | 36 | 7 | 0 | 35 | 92 | 486 / 110 | 1.29 | 5 | 1.137 |
| 22×22 quadrants ×1 · zaniso-Uquad-lb10 | 1 | 34.9 | – | 36 | 6 | 5 | 38 | 91 | 517 / 69 | 1.29 | 4 | 1.133 |
| 22×22 quadrants ×1 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 38.2 | – | 40 | 12 | 8 | 34 | 97 | 366 / 190 | 1.14 | 35 | 1.113 |
| 22×22 quadrants ×1 · ziso | 1 | 35.3 | 34.8 | 34 | 4 | 4 | 37 | 98 | 490 / 92 | 1.08 | 40 | 1.073 |
| 22×22 quadrants ×1 · ziso-open | 1 | 40.1 | – | 42 | 10 | 14 | 36 | 98 | 367 / 172 | 1.03 | 46 | 1.051 |
| 22×22 quadrants ×1 · zres-Uquad-open-far-lb10-full | 1 | 58.7 | – | 57 | 33 | 32 | 72 | 100 | 223 / 149 | 1.31 | 10 | 1.131 |
| 22×22 quadrants ×1 · zres-Uquad-open-far-lb10-full1s | 1 | 55.9 | – | 50 | 32 | 32 | 66 | 100 | 210 / 187 | 1.39 | 3 | 1.124 |
| 22×22 quadrants ×1 · zsplit-Uquad-posthoc | 1 | 35.1 | 34.9 | 34 | 3 | 4 | 39 | 94 | 502 / 82 | 1.23 | 7 | 1.119 |
| 22×22 quadrants ×2 · zaniso-U22.1 | 1 | 13.6 | – | 12 | 0 | 0 | 9 | 47 | 765 / 13 | 0.89 | 32 | 1.307 |
| 22×22 quadrants ×2 · zaniso-Uquad | 1 | 32.0 | – | 32 | 5 | 0 | 45 | 78 | 489 / 123 | 1.23 | 20 | 1.217 |
| 22×22 quadrants ×2 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 35.6 | – | 30 | 8 | 10 | 33 | 96 | 405 / 175 | 1.11 | 48 | 1.177 |
| 22×22 quadrants ×2 · ziso | 1 | 36.7 | 41.1 | 34 | 9 | 6 | 40 | 95 | 380 / 190 | 0.79 | 11 | 1.114 |
| 22×22 quadrants ×2 · ziso-open | 1 | 34.9 | – | 33 | 8 | 8 | 26 | 98 | 402 / 184 | 0.80 | 11 | 1.083 |
| 22×22 quadrants ×2 · zres-Uquad-open-far-lb10-full | 1 | 59.3 | – | 72 | 36 | 23 | 66 | 99 | 218 / 148 | 1.16 | 34 | 1.153 |
| 22×22 quadrants ×2 · zres-Uquad-open-far-lb10-full1s | 1 | 58.3 | – | 67 | 35 | 23 | 67 | 100 | 224 / 151 | 1.23 | 19 | 1.168 |
| 22×22 quadrants ×4 · zaniso-U22.1 | 1 | 4.0 | – | 4 | 0 | 0 | 0 | 16 | 799 / 65 | 1.15 | 2 | 1.818 |
| 22×22 quadrants ×4 · zaniso-Uquad | 1 | 28.1 | – | 37 | 3 | 0 | 38 | 62 | 574 / 73 | 1.15 | 20 | 1.156 |
| 22×22 quadrants ×4 · zaniso-Uquad-lb1 | 1 | 26.8 | – | 34 | 2 | 0 | 36 | 63 | 599 / 60 | 1.21 | 27 | 1.144 |
| 22×22 quadrants ×4 · zaniso-Uquad-lb10 | 1 | 30.2 | – | 39 | 4 | 0 | 36 | 72 | 569 / 59 | 1.21 | 35 | 1.134 |
| 22×22 quadrants ×4 · zaniso-Uquad-messy0.6-open-gs8x1+8x4-pwtauz-lb10 | 1 | 34.4 | 39.1 | 24 | 16 | 5 | 30 | 97 | 466 / 124 | 1.05 | 61 | 1.184 |
| 22×22 quadrants ×4 · zaniso-Uquad-open | 1 | 32.2 | – | 27 | 12 | 0 | 32 | 90 | 508 / 102 | 1.02 | 33 | 1.160 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-gs12x1+4x6-pwtauz-lb10 | 1 | 32.4 | – | 32 | 8 | 4 | 19 | 98 | 451 / 157 | 1.07 | 60 | 1.190 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 1 | 37.4 | – | 29 | 14 | 6 | 39 | 98 | 440 / 123 | 1.08 | 62 | 1.149 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 3 | 38.5 | 39.1 | 29 | 14 | 8 | 44 | 99 | 1303 / 358 | 1.09 | 54 | 1.178 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-lb10 | 1 | 37.9 | – | 32 | 8 | 2 | 48 | 100 | 455 / 104 | 1.04 | 55 | 1.151 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-pwtauz-lb10 | 1 | 37.0 | – | 33 | 12 | 5 | 37 | 98 | 479 / 88 | 1.06 | 63 | 1.140 |
| 22×22 quadrants ×4 · zaniso-Uquad-open-pwz-lb10 | 1 | 22.3 | – | 12 | 2 | 2 | 3 | 93 | 612 / 87 | 0.96 | 81 | 1.181 |
| 22×22 quadrants ×4 · ziso | 1 | 41.6 | 44.4 | 32 | 21 | 8 | 47 | 100 | 392 / 134 | 0.54 | 3 | 1.134 |
| 22×22 quadrants ×4 · ziso-messy0.6-open | 1 | 39.4 | 36.7 | 30 | 16 | 8 | 43 | 100 | 432 / 113 | 0.54 | 2 | 1.108 |
| 22×22 quadrants ×4 · ziso-open | 3 | 41.6 | 40.0 | 31 | 19 | 7 | 51 | 100 | 1186 / 392 | 0.55 | 2 | 1.089 |
| 22×22 quadrants ×4 · zres-Uquad-messy0.6-open-far-lb10 | 1 | 29.3 | 33.9 | 22 | 9 | 3 | 12 | 99 | 401 / 235 | 1.00 | 74 | 1.181 |
| 22×22 quadrants ×4 · zres-Uquad-open-far-lb10 | 2 | 37.4 | 31.7 | 32 | 16 | 7 | 36 | 96 | 810 / 317 | 1.03 | 65 | 1.153 |
| 22×22 quadrants ×4 · zres-Uquad-open-far-lb10-full | 1 | 54.8 | – | 60 | 44 | 24 | 62 | 83 | 280 / 127 | 1.09 | 50 | 1.142 |
| 22×22 quadrants ×4 · zres-Uquad-open-far-lb10-full1s | 1 | 61.1 | – | 61 | 47 | 31 | 69 | 98 | 223 / 127 | 1.11 | 46 | 1.130 |
| 22×22 quadrants ×4 · zres-Uquad-open-faru-lb10 | 1 | 31.2 | – | 34 | 16 | 12 | 29 | 66 | 522 / 97 | 1.04 | 69 | 1.224 |
| 22×22 quadrants ×4 · zsplit-Uquad-posthoc | 1 | 37.1 | 42.0 | 32 | 16 | 6 | 32 | 100 | 413 / 153 | 1.11 | 46 | 1.151 |
| 22×22 quadrants ×8 · zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 1 | 32.3 | – | 29 | 3 | 1 | 30 | 98 | 491 / 118 | 1.04 | 62 | 1.112 |
| 22×22 quadrants ×8 · ziso | 1 | 37.2 | 37.1 | 41 | 8 | 5 | 33 | 98 | 349 / 216 | 0.32 | 1 | 1.147 |
| 22×22 quadrants ×8 · ziso-open | 1 | 37.4 | – | 40 | 7 | 2 | 38 | 100 | 407 / 156 | 0.33 | 1 | 1.106 |
| 22×22 quadrants ×8 · zres-Uquad-open-far-lb10 | 1 | 35.3 | – | 38 | 3 | 5 | 32 | 99 | 468 / 114 | 1.00 | 75 | 1.118 |
| 22×22 quadrants ×8 · zres-Uquad-open-far-lb10-full | 1 | 46.3 | – | 73 | 25 | 18 | 42 | 74 | 362 / 121 | 1.02 | 72 | 1.147 |
| 22×22 quadrants ×8 · zres-Uquad-open-far-lb10-full1s | 1 | 57.4 | – | 69 | 42 | 21 | 68 | 88 | 234 / 149 | 1.04 | 69 | 1.092 |
| 22×22 quadrants ×12 · id | 1 | – | 25.3 | 13 | 2 | 0 | 12 | 100 | 0 / 672 | 1.30 | 6 | 1.006 |
| 22×22 quadrants ×12 · zaniso-Ubox | 1 | 9.7 | 10.4 | 13 | 0 | 0 | 0 | 35 | 795 / 18 | 0.47 | 2 | 1.081 |
| 22×22 quadrants ×12 · zaniso-Uquad | 1 | 22.9 | 22.9 | 48 | 1 | 0 | 16 | 49 | 608 / 86 | 0.67 | 6 | 1.091 |
| 22×22 quadrants ×12 · zaniso-Uquad-lb1 | 1 | 28.8 | 29.3 | 47 | 2 | 0 | 32 | 63 | 555 / 86 | 0.77 | 14 | 1.103 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-far-pwtauz-lb10 | 1 | 33.0 | – | 25 | 2 | 4 | 36 | 98 | 448 / 155 | 1.03 | 78 | 1.125 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-gs4x1+12x6-pwtauz-lb10 | 1 | 32.8 | – | 33 | 2 | 3 | 31 | 96 | 474 / 131 | 1.04 | 67 | 1.096 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 1 | 30.0 | – | 27 | 4 | 1 | 19 | 98 | 513 / 117 | 1.04 | 68 | 1.105 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 2 | 35.3 | 34.7 | 31 | 1 | 3 | 42 | 99 | 873 / 292 | 1.04 | 72 | 1.093 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x6-sum2-pwtauz-lb10 | 1 | 31.3 | – | 31 | 2 | 2 | 24 | 98 | 524 / 94 | 1.04 | 66 | 1.085 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-hs4-lb10 | 1 | 27.8 | – | 33 | 0 | 0 | 18 | 88 | 522 / 128 | 1.01 | 78 | 1.076 |
| 22×22 quadrants ×12 · zaniso-Uquad-open-lb10 | 1 | 29.6 | – | 43 | 0 | 0 | 22 | 83 | 554 / 80 | 0.85 | 23 | 1.066 |
| 22×22 quadrants ×12 · zboth-Uquad | 1 | 36.8 | 36.7 | 42 | 13 | 2 | 28 | 99 | 404 / 165 | 1.04 | 74 | 1.099 |
| 22×22 quadrants ×12 · ziso | 1 | 36.8 | 32.8 | 39 | 11 | 4 | 32 | 98 | 423 / 146 | 0.24 | 1 | 1.085 |
| 22×22 quadrants ×12 · ziso-full | 1 | 57.7 | 53.1 | 77 | 29 | 15 | 68 | 100 | 239 / 142 | 0.26 | 0 | 1.114 |
| 22×22 quadrants ×12 · ziso-lo0.05 | 1 | 37.6 | 8.7 | 43 | 7 | 4 | 34 | 100 | 465 / 97 | 0.21 | 0 | 1.230 |
| 22×22 quadrants ×12 · ziso-open | 3 | 33.9 | 28.1 | 36 | 5 | 1 | 27 | 100 | 1301 / 485 | 0.23 | 0 | 1.055 |
| 22×22 quadrants ×12 · ziso-sh | 1 | 34.6 | 8.6 | 42 | 8 | 0 | 25 | 98 | 448 / 141 | 0.22 | 0 | 1.067 |
| 22×22 quadrants ×12 · zres-Uquad-open-far-lb10 | 2 | 33.8 | 27.2 | 35 | 5 | 1 | 30 | 98 | 829 / 362 | 0.99 | 77 | 1.112 |
| 22×22 quadrants ×12 · zres-Uquad-open-far-lb10-full | 1 | 50.4 | – | 74 | 28 | 17 | 47 | 86 | 302 / 144 | 1.01 | 75 | 1.088 |
| 22×22 quadrants ×12 · zres-Uquad-open-far-lb10-full-aborted | 1 | 31.8 | – | 30 | 8 | 3 | 19 | 98 | 201 / 413 | 0.52 | 1 | 1.083 |
| 22×22 quadrants ×12 · zres-Uquad-open-far-lb10-full1s | 3 | 53.5 | – | 65 | 32 | 16 | 54 | 100 | 691 / 565 | 1.01 | 73 | 1.098 |
| 22×22 quadrants ×12 · zres-Uquad-open-faru-lb10 | 1 | 36.7 | – | 34 | 10 | 3 | 36 | 99 | 482 / 88 | 1.01 | 76 | 1.111 |
| 22×22 quadrants ×12 · zres-Uquad-open-faru-lb10-full | 1 | 49.3 | – | 64 | 19 | 13 | 53 | 97 | 345 / 111 | 0.99 | 78 | 1.133 |
| 22×22 quadrants ×12 · zsplit-Uquad | 1 | 26.9 | 27.8 | 32 | 0 | 0 | 28 | 74 | 572 / 86 | 1.42 | 1 | 1.059 |
| 22×22 quadrants ×12 · zsplit-Uquad-lo0.05 | 1 | 27.6 | 34.6 | 42 | 7 | 0 | 33 | 56 | 562 / 90 | 1.24 | 20 | 1.161 |
| 22×22 quadrants ×12 · zsplit-Uquad-lo0.05-posthoc | 1 | 28.9 | 37.7 | 43 | 7 | 4 | 26 | 64 | 535 / 105 | 1.05 | 73 | 1.270 |
| 22×22 quadrants ×12 · zsplit-Uquad-posthoc | 1 | 33.1 | 36.1 | 39 | 9 | 1 | 19 | 98 | 449 / 153 | 1.04 | 74 | 1.068 |
| 22×22 quadrants ×12 · zsplit-Uquad-posthoc-full | 1 | 57.2 | 57.2 | 77 | 29 | 16 | 66 | 99 | 237 / 148 | 1.07 | 66 | 1.101 |
| 22×22 quadrants ×12 · zsplit-Uquad-sh | 1 | 29.2 | 30.9 | 45 | 0 | 0 | 34 | 67 | 582 / 55 | 1.41 | 1 | 1.123 |
| 22×22 quadrants ×12 · zsplit-Uquad-sh-posthoc | 1 | 30.7 | 35.8 | 42 | 6 | 0 | 19 | 87 | 480 / 144 | 1.05 | 72 | 1.193 |
| 22×22 quadrants ×12 · zsplit-Uquad-tw0 | 1 | 15.3 | 14.7 | 15 | 0 | 0 | 2 | 59 | 760 / 2 | 0.84 | 21 | 1.042 |
| 10×10 quadrants ×1 · zaniso-U22.1 | 1 | 9.8 | – | 18 | 2 | 0 | 7 | 23 | 777 / 35 | 0.76 | 22 | 1.181 |
| 10×10 quadrants ×1 · zaniso-Uquad | 1 | 45.4 | 45.8 | 57 | 17 | 16 | 54 | 83 | 418 / 73 | 1.12 | 38 | 1.094 |
| 10×10 quadrants ×1 · zaniso-Uquad-lb10 | 1 | 49.1 | – | 64 | 18 | 11 | 66 | 86 | 380 / 78 | 1.12 | 42 | 1.097 |
| 10×10 quadrants ×1 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 57.6 | – | 75 | 32 | 36 | 51 | 95 | 295 / 87 | 1.06 | 64 | 1.104 |
| 10×10 quadrants ×1 · ziso | 1 | 55.1 | 57.6 | 72 | 29 | 36 | 56 | 83 | 325 / 79 | 0.91 | 37 | 1.048 |
| 10×10 quadrants ×1 · ziso-open | 1 | 60.3 | – | 72 | 32 | 51 | 54 | 94 | 279 / 78 | 0.91 | 35 | 1.046 |
| 10×10 quadrants ×1 · zres-Uquad-open-far-lb10-full | 1 | 92.2 | – | 99 | 83 | 83 | 97 | 98 | 38 / 32 | 1.09 | 48 | 1.072 |
| 10×10 quadrants ×1 · zres-Uquad-open-far-lb10-full1s | 1 | 91.0 | – | 100 | 86 | 77 | 94 | 98 | 30 / 51 | 1.15 | 27 | 1.088 |
| 10×10 quadrants ×1 · zsplit-Uquad-posthoc | 1 | 52.4 | 54.3 | 72 | 27 | 27 | 54 | 82 | 361 / 67 | 1.10 | 49 | 1.080 |
| 10×10 quadrants ×2 · zaniso-U22.1 | 1 | 12.6 | – | 18 | 3 | 0 | 5 | 36 | 745 / 42 | 0.81 | 30 | 1.108 |
| 10×10 quadrants ×2 · zaniso-Uquad | 1 | 43.7 | – | 52 | 7 | 6 | 64 | 89 | 447 / 60 | 1.11 | 45 | 1.115 |
| 10×10 quadrants ×2 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 54.9 | – | 59 | 25 | 25 | 69 | 97 | 275 / 131 | 1.06 | 62 | 1.120 |
| 10×10 quadrants ×2 · ziso | 1 | 49.9 | 48.9 | 53 | 15 | 15 | 69 | 97 | 406 / 45 | 0.66 | 6 | 1.056 |
| 10×10 quadrants ×2 · ziso-open | 1 | 51.3 | – | 51 | 9 | 29 | 71 | 97 | 352 / 86 | 0.66 | 9 | 1.054 |
| 10×10 quadrants ×2 · zres-Uquad-open-far-lb10-full | 1 | 88.0 | – | 94 | 74 | 77 | 96 | 99 | 66 / 42 | 1.06 | 58 | 1.100 |
| 10×10 quadrants ×2 · zres-Uquad-open-far-lb10-full1s | 1 | 85.6 | – | 94 | 70 | 72 | 92 | 100 | 60 / 70 | 1.08 | 53 | 1.096 |
| 10×10 quadrants ×4 · zaniso-U22.1 | 1 | 18.8 | – | 14 | 2 | 2 | 16 | 61 | 688 / 43 | 1.00 | 60 | 1.089 |
| 10×10 quadrants ×4 · zaniso-Uquad | 1 | 34.0 | – | 66 | 7 | 0 | 34 | 63 | 545 / 49 | 1.05 | 31 | 1.081 |
| 10×10 quadrants ×4 · zaniso-Uquad-lb1 | 1 | 39.0 | – | 64 | 3 | 2 | 51 | 75 | 481 / 68 | 1.10 | 49 | 1.069 |
| 10×10 quadrants ×4 · zaniso-Uquad-lb10 | 1 | 45.8 | – | 66 | 3 | 0 | 76 | 83 | 457 / 31 | 1.12 | 38 | 1.077 |
| 10×10 quadrants ×4 · zaniso-Uquad-messy0.6-open-gs8x1+8x4-pwtauz-lb10 | 1 | 41.1 | 43.0 | 43 | 16 | 14 | 34 | 98 | 395 / 135 | 1.04 | 79 | 1.098 |
| 10×10 quadrants ×4 · zaniso-Uquad-open | 1 | 43.0 | – | 57 | 12 | 1 | 53 | 92 | 440 / 73 | 1.00 | 54 | 1.081 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-gs12x1+4x6-pwtauz-lb10 | 1 | 44.4 | – | 44 | 8 | 14 | 57 | 98 | 417 / 83 | 1.05 | 79 | 1.102 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 1 | 48.9 | – | 54 | 22 | 31 | 39 | 98 | 342 / 118 | 1.04 | 79 | 1.101 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 3 | 48.2 | 50.4 | 54 | 18 | 17 | 52 | 99 | 1121 / 277 | 1.05 | 78 | 1.105 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-lb10 | 1 | 44.9 | – | 54 | 2 | 3 | 67 | 98 | 436 / 60 | 1.03 | 76 | 1.103 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-pwtauz-hs20-es0.1-lb10 | 1 | 37.7 | – | 36 | 8 | 18 | 28 | 98 | 458 / 103 | 1.05 | 78 | 1.119 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-pwtauz-lb10 | 1 | 42.2 | – | 51 | 7 | 7 | 48 | 98 | 424 / 96 | 1.04 | 78 | 1.098 |
| 10×10 quadrants ×4 · zaniso-Uquad-open-pwz-lb10 | 1 | 26.7 | – | 22 | 1 | 6 | 7 | 98 | 556 / 104 | 0.96 | 78 | 1.115 |
| 10×10 quadrants ×4 · ziso | 1 | 55.6 | 56.6 | 56 | 32 | 37 | 53 | 100 | 342 / 58 | 0.43 | 5 | 1.062 |
| 10×10 quadrants ×4 · ziso-messy0.6-open | 1 | 49.9 | 40.9 | 44 | 27 | 28 | 50 | 100 | 321 / 130 | 0.44 | 4 | 1.071 |
| 10×10 quadrants ×4 · ziso-open | 3 | 55.7 | 56.1 | 67 | 32 | 29 | 51 | 99 | 967 / 229 | 0.43 | 4 | 1.063 |
| 10×10 quadrants ×4 · zres-Uquad-messy0.6-open-far-lb10 | 1 | 42.0 | 42.6 | 46 | 18 | 26 | 21 | 99 | 343 / 179 | 0.97 | 87 | 1.126 |
| 10×10 quadrants ×4 · zres-Uquad-open-far-lb10 | 2 | 54.3 | 46.7 | 58 | 27 | 31 | 55 | 100 | 638 / 185 | 1.00 | 78 | 1.101 |
| 10×10 quadrants ×4 · zres-Uquad-open-far-lb10-full | 1 | 83.4 | – | 91 | 76 | 57 | 96 | 97 | 76 / 73 | 1.01 | 72 | 1.070 |
| 10×10 quadrants ×4 · zres-Uquad-open-far-lb10-full1s | 1 | 86.8 | – | 88 | 80 | 66 | 100 | 100 | 58 / 61 | 1.02 | 70 | 1.080 |
| 10×10 quadrants ×4 · zres-Uquad-open-faru-lb10 | 1 | 52.7 | – | 57 | 28 | 27 | 51 | 100 | 338 / 88 | 1.03 | 81 | 1.111 |
| 10×10 quadrants ×8 · zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 1 | 41.0 | – | 49 | 9 | 13 | 34 | 99 | 388 / 143 | 1.03 | 82 | 1.076 |
| 10×10 quadrants ×8 · ziso | 1 | 44.2 | 43.9 | 56 | 16 | 9 | 41 | 100 | 422 / 80 | 0.25 | 5 | 1.059 |
| 10×10 quadrants ×8 · ziso-open | 1 | 42.0 | – | 44 | 14 | 12 | 39 | 100 | 394 / 128 | 0.26 | 3 | 1.057 |
| 10×10 quadrants ×8 · zres-Uquad-open-far-lb10 | 1 | 46.0 | – | 54 | 16 | 12 | 48 | 100 | 361 / 125 | 0.98 | 81 | 1.069 |
| 10×10 quadrants ×8 · zres-Uquad-open-far-lb10-full | 1 | 81.4 | – | 85 | 73 | 56 | 94 | 99 | 103 / 64 | 0.99 | 79 | 1.066 |
| 10×10 quadrants ×8 · zres-Uquad-open-far-lb10-full1s | 1 | 80.3 | – | 85 | 72 | 57 | 88 | 100 | 89 / 88 | 1.00 | 77 | 1.072 |
| 10×10 quadrants ×12 · zaniso-U22.1 | 1 | 21.7 | 23.7 | 22 | 1 | 0 | 10 | 75 | 653 / 52 | 0.96 | 34 | 1.161 |
| 10×10 quadrants ×12 · zaniso-Ubox | 1 | 20.0 | 17.8 | 23 | 0 | 0 | 5 | 72 | 667 / 53 | 0.64 | 8 | 1.214 |
| 10×10 quadrants ×12 · zaniso-Uquad | 1 | 29.9 | 29.7 | 76 | 0 | 0 | 32 | 42 | 619 / 12 | 0.66 | 16 | 1.074 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-far-pwtauz-lb10 | 1 | 45.9 | – | 66 | 8 | 1 | 58 | 96 | 418 / 69 | 1.01 | 85 | 1.061 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs2x1+14x6-pwtauz-lb10 | 1 | 47.2 | – | 62 | 14 | 2 | 62 | 97 | 410 / 65 | 1.01 | 84 | 1.063 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs4x1+12x6-pwtauz-lb10 | 1 | 52.6 | – | 65 | 12 | 4 | 85 | 97 | 376 / 51 | 1.00 | 86 | 1.052 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs4x1+12x8-pwtauz-lb10 | 1 | 49.9 | – | 56 | 5 | 11 | 81 | 96 | 379 / 72 | 1.01 | 84 | 1.053 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 1 | 46.3 | – | 58 | 8 | 1 | 66 | 98 | 385 / 98 | 1.01 | 85 | 1.048 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs8x1+4x4+4x16-pwtauz-lb10 | 1 | 42.2 | – | 44 | 10 | 2 | 59 | 97 | 439 / 81 | 1.02 | 85 | 1.056 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x12-pwtauz-lb10 | 1 | 42.0 | – | 60 | 8 | 2 | 43 | 97 | 416 / 106 | 1.01 | 84 | 1.062 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x3-pwtauz-lb10 | 1 | 50.0 | – | 54 | 4 | 1 | 95 | 96 | 388 / 62 | 1.01 | 82 | 1.047 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 2 | 47.8 | 51.9 | 59 | 5 | 3 | 75 | 97 | 746 / 193 | 1.01 | 84 | 1.057 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-gs8x1+8x6-sum2-pwtauz-lb10 | 1 | 43.9 | – | 53 | 14 | 0 | 54 | 98 | 398 / 107 | 1.00 | 84 | 1.052 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-hs4-lb10 | 1 | 36.9 | – | 45 | 3 | 0 | 46 | 90 | 445 / 123 | 1.04 | 59 | 1.055 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-pwtauz-es6-lb10 | 1 | 40.9 | – | 36 | 12 | 2 | 58 | 97 | 455 / 77 | 1.04 | 82 | 1.052 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-pwtauz-hs4-lb10 | 1 | 37.4 | – | 38 | 6 | 0 | 44 | 98 | 419 / 144 | 1.03 | 83 | 1.058 |
| 10×10 quadrants ×12 · zaniso-Uquad-open-pwtauz-hs60-es0.1-lb10 | 1 | 37.0 | – | 41 | 3 | 0 | 45 | 96 | 460 / 107 | 1.02 | 81 | 1.046 |
| 10×10 quadrants ×12 · zboth-Uquad | 1 | 49.8 | 51.3 | 58 | 13 | 3 | 77 | 98 | 374 / 78 | 1.01 | 86 | 1.057 |
| 10×10 quadrants ×12 · ziso | 1 | 57.6 | 57.3 | 71 | 27 | 13 | 77 | 100 | 308 / 74 | 0.17 | 6 | 1.032 |
| 10×10 quadrants ×12 · ziso-full | 1 | 84.6 | 83.9 | 94 | 78 | 51 | 99 | 100 | 73 / 66 | 0.20 | 8 | 1.045 |
| 10×10 quadrants ×12 · ziso-lo0.05 | 1 | 53.7 | 11.8 | 58 | 28 | 8 | 74 | 100 | 344 / 73 | 0.14 | 7 | 1.114 |
| 10×10 quadrants ×12 · ziso-open | 3 | 53.7 | 56.6 | 68 | 24 | 14 | 64 | 99 | 970 / 279 | 0.17 | 5 | 1.039 |
| 10×10 quadrants ×12 · ziso-sh | 1 | 55.3 | 15.0 | 71 | 24 | 21 | 63 | 98 | 325 / 77 | 0.15 | 5 | 1.070 |
| 10×10 quadrants ×12 · zres-Uquad-open-far-lb10 | 2 | 47.3 | 47.3 | 60 | 13 | 7 | 59 | 97 | 773 / 175 | 0.98 | 84 | 1.061 |
| 10×10 quadrants ×12 · zres-Uquad-open-far-lb10-full | 1 | 82.3 | – | 95 | 76 | 50 | 95 | 96 | 90 / 69 | 0.98 | 85 | 1.047 |
| 10×10 quadrants ×12 · zres-Uquad-open-far-lb10-full-aborted | 1 | 53.4 | – | 62 | 30 | 19 | 56 | 100 | 115 / 304 | 0.37 | 1 | 1.062 |
| 10×10 quadrants ×12 · zres-Uquad-open-far-lb10-full1s | 3 | 83.1 | – | 91 | 70 | 60 | 96 | 99 | 203 / 253 | 0.98 | 83 | 1.048 |
| 10×10 quadrants ×12 · zres-Uquad-open-faru-lb10 | 1 | 38.4 | – | 33 | 9 | 5 | 49 | 96 | 498 / 56 | 1.00 | 85 | 1.093 |
| 10×10 quadrants ×12 · zsplit-Uquad | 1 | 47.1 | 47.9 | 73 | 4 | 0 | 73 | 85 | 439 / 37 | 1.33 | 5 | 1.040 |
| 10×10 quadrants ×12 · zsplit-Uquad-lo0.05 | 1 | 41.0 | 48.6 | 75 | 9 | 0 | 54 | 67 | 479 / 52 | 1.22 | 25 | 1.078 |
| 10×10 quadrants ×12 · zsplit-Uquad-lo0.05-posthoc | 1 | 48.8 | 53.2 | 58 | 27 | 7 | 74 | 78 | 392 / 69 | 1.02 | 87 | 1.071 |
| 10×10 quadrants ×12 · zsplit-Uquad-posthoc | 1 | 55.4 | 56.7 | 71 | 28 | 11 | 69 | 99 | 328 / 73 | 1.01 | 86 | 1.046 |
| 10×10 quadrants ×12 · zsplit-Uquad-posthoc-full | 1 | 84.7 | 84.2 | 94 | 78 | 51 | 100 | 100 | 73 / 65 | 1.03 | 85 | 1.044 |
| 10×10 quadrants ×12 · zsplit-Uquad-sh | 1 | 44.7 | 46.1 | 83 | 0 | 0 | 62 | 78 | 440 / 58 | 1.38 | 6 | 1.062 |
| 10×10 quadrants ×12 · zsplit-Uquad-sh-posthoc | 1 | 52.4 | 53.7 | 71 | 26 | 16 | 59 | 91 | 346 / 82 | 1.02 | 85 | 1.114 |
| 10×10 quadrants ×12 · zsplit-Uquad-tw0 | 1 | 21.3 | 21.7 | 22 | 0 | 0 | 8 | 77 | 696 / 12 | 0.88 | 36 | 1.080 |

Queries solved (%), pooled over training seeds and planning seeds 0/1/2. Variant names: `ziso` trained on the stretched space (plain length);
`zaniso-U<unit>` the inner product with that time unit (cells per unit; `quad` = the warp's largest stretch, 3.7 cells; `box` = the maze's side);
`-open` with the open-start pairs; `-lb10` lower-bound hinge, weight 10; `-pwtauz` pair weight exp(-T rho / 2) (`-pwz`: on the warp distance alone);
`-gs8x1+8x4` head with 8 groups as they are and 8 multiplied by 4 (`-hs4`: a scalar 4 on the whole head); `-messy0.6` the messy warp;
`zres-Uquad-open-far-lb10` the two-part field (head on the plain length, far term on the inner product of the sum; `-full`: the head taken, frozen, from the gap study's full-loss field);
`-far` a far term added to the single field; `-sum2` two head groups added up instead of entering the soft maximum; `id` no warp. First-pass variants (not pursued): `zsplit-...-posthoc` a stretched-space field plus the deficit term, `zsplit-...` / `zboth-...`
trained with it, `-sh` / `-lo0.05` other band densities, `-full` the gap study's full-loss fields, `-tw0` no pair weight.

## Appendix: the deficit term (first pass, set aside)

For this constructed warp the maze's travel time splits as T*(a, b) ≈ T_w(φ(a), φ(b)) / m + |u(a) − u(b)|, u(x) = x − φ(x) / m: the stretched maze's
travel time plus the straight-line distance in the coordinate the warp removed. On exact fields at gap ×12 the sum is within 5 % of T* on 97 % of
queries (run `verify`), and added to a stretched-space field without retraining it gave the same planning with true travel times (variants
`zsplit-Uquad-posthoc`, `...-posthoc-full` in the table above). It needs the compressed region to be open and the warp to be separable with a uniform
reference magnification, which a learned warp will not give, so it was dropped at the user's direction; the code (`aniso.split_class`, modes `zsplit`,
`zboth`) and the runs remain.

## Code

| file | what |
|---|---|
| `aniso.py` | `metric_full` (M = J Jᵀ of any warp, autograd), `eikonal_loss` (ntrl-demo's eikonal term with the inner product, a time unit and the pair weights), `residual_loss` (two-part field), `FarMetric` and `far_scale`, the field class (head scalar, embedding factor, group factors, additive groups; first-pass deficit term), `MessyWarp`, `load_field`, the checks |
| `atrain.py` | ntrl-demo's training loop with the eikonal term only; modes `id`, `ziso`, `zaniso`, `zres` (first pass: `zsplit`, `zboth`); the lower-bound hinge; `head_from` (train the correction on a frozen field) |
| `wallfeel.py` | post-hoc on registered checkpoints: where T rises into the quadrants' walls by goal class, and where failed plans first touch a wall (`report/wallfeel/<run id>.json`) |
| `pipeline.py` | one run: train (or `--checkpoint`), plan the gap study's queries on the maze and on the warp, travel-time accuracy, implied speed by region |
| `verify.py` | exact fields: the equations by region, the price of planning on the warped maze, the split identity |
| `run.py` | registers a run (stages `train`, `evalq`, `verify`); sources are frozen in `snap/<hash>/` |
| `report.py`, `figs.py`, `page.py`, `text_blocks.py`, `page_template.html`, `readme.py` | summary, figures, the findings page and this file |

Runs use the cross-warp study's prepared gap mazes, pairs and queries (`jeffrey/cross_warp`, stage `gapprepare --z-draw`) and its warp, planner and truth modules, copied into each snapshot.

## Reproduce

```bash
PY=~/.venvs/bellman-ntfield/bin/python
OPEN=results/registered_20261106T061244Z-8a3753d8     # gap x4, 22x22 quadrants: recipe pairs + open-start pairs (cross_warp gapprepare --z-draw --open-pairs 250000)
# the full recipe, and the stretched-space reference on the same pairs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4recipe \
    --extra '--mode zaniso --unit-cells quad --group-scales 8x1,8x4 --lb-weight 10 --pair-weight tauz'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4zisoopen --extra '--mode ziso'
# the two-part field (plans with the sum and with its head alone), and its correction alone on an existing field
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $PY jeffrey/aniso_eikonal/run.py --prepared $OPEN --seed 87358 --label c22k4zres \
    --extra '--mode zres --unit-cells quad --far auto --lb-weight 10 --planners z'
... --extra '--mode zres --unit-cells quad --far auto --lb-weight 10 --planners z --tag full --head-from results/registered_<field>/training/checkpoint.pt'
# the checks on exact fields
... --stage verify --prepared results/registered_20261105T070343Z-fda24e40 --label 'verify c22 k12'
$PY jeffrey/aniso_eikonal/report.py && $PY jeffrey/aniso_eikonal/wallfeel.py --all && $PY jeffrey/aniso_eikonal/figs.py && $PY jeffrey/aniso_eikonal/page.py && $PY jeffrey/aniso_eikonal/readme.py
```


## Registered runs

| registered run | maze | variant | training seed | planner on the warp (%) | planner on the maze (%) |
|---|---|---|---|---|---|
| 20261106T061107Z-4c5a4823 | 22×22 quadrants ×1 | zaniso-U22.1 | 87358 | 7.6 | – |
| 20261106T060132Z-f672b817 | 22×22 quadrants ×1 | zaniso-Uquad | 87358 | 33.8 | – |
| 20261106T063842Z-dc4835a0 | 22×22 quadrants ×1 | zaniso-Uquad-lb10 | 87358 | 34.9 | – |
| 20261106T084300Z-953984c3 | 22×22 quadrants ×1 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 38.2 | – |
| 20261106T053223Z-b416b234 | 22×22 quadrants ×1 | ziso | 87358 | 35.3 | 34.8 |
| 20261106T105131Z-aa65f0f4 | 22×22 quadrants ×1 | ziso-open | 87358 | 40.1 | – |
| 20261106T102447Z-720cd974 | 22×22 quadrants ×1 | zres-Uquad-open-far-lb10-full | 87358 | 58.7 | – |
| 20261106T130033Z-d3f19309 | 22×22 quadrants ×1 | zres-Uquad-open-far-lb10-full1s | 87358 | 55.9 | – |
| 20261106T054231Z-6da6208e | 22×22 quadrants ×1 | zsplit-Uquad-posthoc | 87358 | 35.1 | 34.9 |
| 20261106T061539Z-a0c23a0f | 22×22 quadrants ×2 | zaniso-U22.1 | 87358 | 13.6 | – |
| 20261106T060618Z-ffd2bb1d | 22×22 quadrants ×2 | zaniso-Uquad | 87358 | 32.0 | – |
| 20261106T084611Z-edd77aab | 22×22 quadrants ×2 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 35.6 | – |
| 20261106T054309Z-62682cba | 22×22 quadrants ×2 | ziso | 87358 | 36.7 | 41.1 |
| 20261106T105716Z-a9b7fa20 | 22×22 quadrants ×2 | ziso-open | 87358 | 34.9 | – |
| 20261106T103634Z-24318ade | 22×22 quadrants ×2 | zres-Uquad-open-far-lb10-full | 87358 | 59.3 | – |
| 20261106T124536Z-1088aef4 | 22×22 quadrants ×2 | zres-Uquad-open-far-lb10-full1s | 87358 | 58.3 | – |
| 20261106T061836Z-a731c7ca | 22×22 quadrants ×4 | zaniso-U22.1 | 87358 | 4.0 | – |
| 20261106T060135Z-9e785069 | 22×22 quadrants ×4 | zaniso-Uquad | 87358 | 28.1 | – |
| 20261106T063224Z-38b6e6a8 | 22×22 quadrants ×4 | zaniso-Uquad-lb1 | 87358 | 26.8 | – |
| 20261106T063127Z-bdd931cd | 22×22 quadrants ×4 | zaniso-Uquad-lb10 | 87358 | 30.2 | – |
| 20261106T075824Z-704f96ad | 22×22 quadrants ×4 | zaniso-Uquad-messy0.6-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 34.4 | 39.1 |
| 20261106T062259Z-8eec27e4 | 22×22 quadrants ×4 | zaniso-Uquad-open | 87358 | 32.2 | – |
| 20261106T073652Z-ff9cd9ac | 22×22 quadrants ×4 | zaniso-Uquad-open-gs12x1+4x6-pwtauz-lb10 | 87358 | 32.4 | – |
| 20261106T075049Z-5bfd03c5 | 22×22 quadrants ×4 | zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 87358 | 37.4 | – |
| 20261106T113553Z-cadfcb88 | 22×22 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 39.7 | – |
| 20261106T114246Z-ce60961c | 22×22 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 2 | 38.4 | – |
| 20261106T073259Z-8b410dc9 | 22×22 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 37.3 | – |
| 20261106T112337Z-5c607ae1 | 22×22 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | – | 39.1 |
| 20261106T063453Z-6607ed33 | 22×22 quadrants ×4 | zaniso-Uquad-open-lb10 | 87358 | 37.9 | – |
| 20261106T071320Z-56e1c2f7 | 22×22 quadrants ×4 | zaniso-Uquad-open-pwtauz-lb10 | 87358 | 37.0 | – |
| 20261106T071606Z-3b9549e1 | 22×22 quadrants ×4 | zaniso-Uquad-open-pwz-lb10 | 87358 | 22.3 | – |
| 20261106T053716Z-3a30a0b6 | 22×22 quadrants ×4 | ziso | 87358 | 41.6 | 44.4 |
| 20261106T080425Z-ef1e2f99 | 22×22 quadrants ×4 | ziso-messy0.6-open | 87358 | 39.4 | 36.7 |
| 20261106T114948Z-5e648f79 | 22×22 quadrants ×4 | ziso-open | 1 | 44.7 | – |
| 20261106T115730Z-1c9a3f01 | 22×22 quadrants ×4 | ziso-open | 2 | 42.7 | – |
| 20261106T062643Z-c0cda140 | 22×22 quadrants ×4 | ziso-open | 87358 | 37.3 | – |
| 20261106T111537Z-f7c43002 | 22×22 quadrants ×4 | ziso-open | 87358 | – | 40.0 |
| 20261106T112320Z-a06e78d4 | 22×22 quadrants ×4 | zres-Uquad-messy0.6-open-far-lb10 | 87358 | 29.3 | 33.9 |
| 20261106T111209Z-a809c273 | 22×22 quadrants ×4 | zres-Uquad-open-far-lb10 | 1 | 40.3 | – |
| 20261106T091629Z-aca75921 | 22×22 quadrants ×4 | zres-Uquad-open-far-lb10 | 87358 | 34.4 | – |
| 20261106T112605Z-29cba075 | 22×22 quadrants ×4 | zres-Uquad-open-far-lb10 | 87358 | – | 31.7 |
| 20261106T101013Z-26749f3d | 22×22 quadrants ×4 | zres-Uquad-open-far-lb10-full | 87358 | 54.8 | – |
| 20261106T124536Z-1be76eee | 22×22 quadrants ×4 | zres-Uquad-open-far-lb10-full1s | 87358 | 61.1 | – |
| 20261106T095403Z-2f5f62a3 | 22×22 quadrants ×4 | zres-Uquad-open-faru-lb10 | 87358 | 31.2 | – |
| 20261106T054843Z-f8510efc | 22×22 quadrants ×4 | zsplit-Uquad-posthoc | 87358 | 37.1 | 42.0 |
| 20261106T103923Z-0419031b | 22×22 quadrants ×8 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | 32.3 | – |
| 20261106T055017Z-f605ffa7 | 22×22 quadrants ×8 | ziso | 87358 | 37.2 | 37.1 |
| 20261106T104608Z-90f6e6e3 | 22×22 quadrants ×8 | ziso-open | 87358 | 37.4 | – |
| 20261106T103645Z-56cdbfc3 | 22×22 quadrants ×8 | zres-Uquad-open-far-lb10 | 87358 | 35.3 | – |
| 20261106T102033Z-d8a77fe6 | 22×22 quadrants ×8 | zres-Uquad-open-far-lb10-full | 87358 | 46.3 | – |
| 20261106T123004Z-dfa7270f | 22×22 quadrants ×8 | zres-Uquad-open-far-lb10-full1s | 87358 | 57.4 | – |
| 20261106T041027Z-a5474b25 | 22×22 quadrants ×12 | id | 87358 | – | 25.3 |
| 20261106T042811Z-d7adb58f | 22×22 quadrants ×12 | zaniso-Ubox | 87358 | 9.7 | 10.4 |
| 20261106T040348Z-1b0cb160 | 22×22 quadrants ×12 | zaniso-Uquad | 87358 | 22.9 | 22.9 |
| 20261106T055102Z-5e9056c9 | 22×22 quadrants ×12 | zaniso-Uquad-lb1 | 87358 | 28.8 | 29.3 |
| 20261106T085448Z-fcaea7fd | 22×22 quadrants ×12 | zaniso-Uquad-open-far-pwtauz-lb10 | 87358 | 33.0 | – |
| 20261106T082451Z-05b33827 | 22×22 quadrants ×12 | zaniso-Uquad-open-gs4x1+12x6-pwtauz-lb10 | 87358 | 32.8 | – |
| 20261106T074437Z-1b7c4cf6 | 22×22 quadrants ×12 | zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 87358 | 30.0 | – |
| 20261106T120627Z-101fd1d0 | 22×22 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 1 | 35.4 | – |
| 20261106T072301Z-68d6c6de | 22×22 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | 35.1 | – |
| 20261106T112108Z-5a558f56 | 22×22 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | – | 34.7 |
| 20261106T085959Z-e5763162 | 22×22 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-sum2-pwtauz-lb10 | 87358 | 31.3 | – |
| 20261106T065806Z-ee7c5021 | 22×22 quadrants ×12 | zaniso-Uquad-open-hs4-lb10 | 87358 | 27.8 | – |
| 20261106T070221Z-d4e99dd8 | 22×22 quadrants ×12 | zaniso-Uquad-open-lb10 | 87358 | 29.6 | – |
| 20261106T052020Z-8ed1ce87 | 22×22 quadrants ×12 | zboth-Uquad | 87358 | 36.8 | 36.7 |
| 20261106T045549Z-eeafe280 | 22×22 quadrants ×12 | ziso | 87358 | 36.8 | 32.8 |
| 20261106T045829Z-02da8fd6 | 22×22 quadrants ×12 | ziso-full | 87358 | 57.7 | 53.1 |
| 20261106T050943Z-a7e9f0a5 | 22×22 quadrants ×12 | ziso-lo0.05 | 87358 | 37.6 | 8.7 |
| 20261106T115305Z-d22eaf81 | 22×22 quadrants ×12 | ziso-open | 1 | 32.4 | – |
| 20261106T120052Z-98487fb0 | 22×22 quadrants ×12 | ziso-open | 2 | 37.6 | – |
| 20261106T092537Z-f1c9be8a | 22×22 quadrants ×12 | ziso-open | 87358 | 31.6 | – |
| 20261106T112843Z-7e5beab5 | 22×22 quadrants ×12 | ziso-open | 87358 | – | 28.1 |
| 20261106T050238Z-796b550d | 22×22 quadrants ×12 | ziso-sh | 87358 | 34.6 | 8.6 |
| 20261106T110248Z-4447c4a5 | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10 | 1 | 36.2 | – |
| 20261106T090715Z-909731f9 | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10 | 87358 | 31.4 | – |
| 20261106T093414Z-af5bf883 | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10 | 87358 | – | 27.2 |
| 20261106T093635Z-26316a69 | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10-full | 87358 | 50.4 | – |
| 20261106T093126Z-767ead7c | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10-full-aborted | 87358 | 31.8 | – |
| 20261106T131725Z-d222e289 | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 1 | 48.7 | – |
| 20261106T131725Z-c7a4ce0e | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 2 | 58.0 | – |
| 20261106T123004Z-9cf43e0f | 22×22 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 87358 | 53.8 | – |
| 20261106T094445Z-d5e9722d | 22×22 quadrants ×12 | zres-Uquad-open-faru-lb10 | 87358 | 36.7 | – |
| 20261106T095426Z-f5f8da28 | 22×22 quadrants ×12 | zres-Uquad-open-faru-lb10-full | 87358 | 49.3 | – |
| 20261106T042728Z-995627db | 22×22 quadrants ×12 | zsplit-Uquad | 87358 | 26.9 | 27.8 |
| 20261106T044320Z-121e68d7 | 22×22 quadrants ×12 | zsplit-Uquad-lo0.05 | 87358 | 27.6 | 34.6 |
| 20261106T051920Z-a9dee7bf | 22×22 quadrants ×12 | zsplit-Uquad-lo0.05-posthoc | 87358 | 28.9 | 37.7 |
| 20261106T050349Z-7f0362c5 | 22×22 quadrants ×12 | zsplit-Uquad-posthoc | 87358 | 33.1 | 36.1 |
| 20261106T045404Z-4f33e570 | 22×22 quadrants ×12 | zsplit-Uquad-posthoc-full | 87358 | 57.2 | 57.2 |
| 20261106T044154Z-4a9a4a12 | 22×22 quadrants ×12 | zsplit-Uquad-sh | 87358 | 29.2 | 30.9 |
| 20261106T051420Z-1a4792d4 | 22×22 quadrants ×12 | zsplit-Uquad-sh-posthoc | 87358 | 30.7 | 35.8 |
| 20261106T052448Z-a27fa9de | 22×22 quadrants ×12 | zsplit-Uquad-tw0 | 87358 | 15.3 | 14.7 |
| 20261106T061123Z-71913e68 | 10×10 quadrants ×1 | zaniso-U22.1 | 87358 | 9.8 | – |
| 20261106T055640Z-b87c1b4b | 10×10 quadrants ×1 | zaniso-Uquad | 87358 | 45.4 | 45.8 |
| 20261106T065052Z-725b816a | 10×10 quadrants ×1 | zaniso-Uquad-lb10 | 87358 | 49.1 | – |
| 20261106T083855Z-b8be9e6b | 10×10 quadrants ×1 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 57.6 | – |
| 20261106T053225Z-6ff8824d | 10×10 quadrants ×1 | ziso | 87358 | 55.1 | 57.6 |
| 20261106T105137Z-b6175d2e | 10×10 quadrants ×1 | ziso-open | 87358 | 60.3 | – |
| 20261106T102240Z-557e1437 | 10×10 quadrants ×1 | zres-Uquad-open-far-lb10-full | 87358 | 92.2 | – |
| 20261106T125336Z-7ae9217f | 10×10 quadrants ×1 | zres-Uquad-open-far-lb10-full1s | 87358 | 91.0 | – |
| 20261106T054056Z-5404e18a | 10×10 quadrants ×1 | zsplit-Uquad-posthoc | 87358 | 52.4 | 54.3 |
| 20261106T061654Z-8cdfc574 | 10×10 quadrants ×2 | zaniso-U22.1 | 87358 | 12.6 | – |
| 20261106T060841Z-a7d46c67 | 10×10 quadrants ×2 | zaniso-Uquad | 87358 | 43.7 | – |
| 20261106T084542Z-ca1fc7fe | 10×10 quadrants ×2 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 54.9 | – |
| 20261106T054344Z-2f9eab83 | 10×10 quadrants ×2 | ziso | 87358 | 49.9 | 48.9 |
| 20261106T105822Z-77647ce6 | 10×10 quadrants ×2 | ziso-open | 87358 | 51.3 | – |
| 20261106T102846Z-1da1a7ff | 10×10 quadrants ×2 | zres-Uquad-open-far-lb10-full | 87358 | 88.0 | – |
| 20261106T130134Z-68f0a37d | 10×10 quadrants ×2 | zres-Uquad-open-far-lb10-full1s | 87358 | 85.6 | – |
| 20261106T062144Z-ec65ff31 | 10×10 quadrants ×4 | zaniso-U22.1 | 87358 | 18.8 | – |
| 20261106T060423Z-28c8dab2 | 10×10 quadrants ×4 | zaniso-Uquad | 87358 | 34.0 | – |
| 20261106T064450Z-ee795f23 | 10×10 quadrants ×4 | zaniso-Uquad-lb1 | 87358 | 39.0 | – |
| 20261106T064227Z-455c323b | 10×10 quadrants ×4 | zaniso-Uquad-lb10 | 87358 | 45.8 | – |
| 20261106T075615Z-3e381282 | 10×10 quadrants ×4 | zaniso-Uquad-messy0.6-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 41.1 | 43.0 |
| 20261106T065309Z-a58f563e | 10×10 quadrants ×4 | zaniso-Uquad-open | 87358 | 43.0 | – |
| 20261106T073559Z-33d43237 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs12x1+4x6-pwtauz-lb10 | 87358 | 44.4 | – |
| 20261106T074710Z-c5ad0719 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 87358 | 48.9 | – |
| 20261106T113451Z-330f0106 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 1 | 52.2 | – |
| 20261106T113833Z-9a07cb06 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 2 | 43.2 | – |
| 20261106T073004Z-9ae989b5 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | 49.2 | – |
| 20261106T112112Z-6a176761 | 10×10 quadrants ×4 | zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10 | 87358 | – | 50.4 |
| 20261106T064653Z-2f9156e1 | 10×10 quadrants ×4 | zaniso-Uquad-open-lb10 | 87358 | 44.9 | – |
| 20261106T082312Z-30b6296e | 10×10 quadrants ×4 | zaniso-Uquad-open-pwtauz-hs20-es0.1-lb10 | 87358 | 37.7 | – |
| 20261106T070923Z-565bad83 | 10×10 quadrants ×4 | zaniso-Uquad-open-pwtauz-lb10 | 87358 | 42.2 | – |
| 20261106T070949Z-9fb203c5 | 10×10 quadrants ×4 | zaniso-Uquad-open-pwz-lb10 | 87358 | 26.7 | – |
| 20261106T054251Z-eff113fd | 10×10 quadrants ×4 | ziso | 87358 | 55.6 | 56.6 |
| 20261106T075802Z-f76c55a0 | 10×10 quadrants ×4 | ziso-messy0.6-open | 87358 | 49.9 | 40.9 |
| 20261106T114537Z-13f57c3a | 10×10 quadrants ×4 | ziso-open | 1 | 51.1 | – |
| 20261106T115523Z-b10be316 | 10×10 quadrants ×4 | ziso-open | 2 | 63.7 | – |
| 20261106T065656Z-fd7c268d | 10×10 quadrants ×4 | ziso-open | 87358 | 52.3 | – |
| 20261106T111803Z-3c86a338 | 10×10 quadrants ×4 | ziso-open | 87358 | – | 56.1 |
| 20261106T112258Z-b8620df6 | 10×10 quadrants ×4 | zres-Uquad-messy0.6-open-far-lb10 | 87358 | 42.0 | 42.6 |
| 20261106T110621Z-7548726a | 10×10 quadrants ×4 | zres-Uquad-open-far-lb10 | 1 | 64.8 | – |
| 20261106T091029Z-bf0fc7b0 | 10×10 quadrants ×4 | zres-Uquad-open-far-lb10 | 87358 | 43.8 | – |
| 20261106T093450Z-ea7850ea | 10×10 quadrants ×4 | zres-Uquad-open-far-lb10 | 87358 | – | 46.7 |
| 20261106T100838Z-a004a855 | 10×10 quadrants ×4 | zres-Uquad-open-far-lb10-full | 87358 | 83.4 | – |
| 20261106T124536Z-b9d340e9 | 10×10 quadrants ×4 | zres-Uquad-open-far-lb10-full1s | 87358 | 86.8 | – |
| 20261106T095223Z-1e6dad40 | 10×10 quadrants ×4 | zres-Uquad-open-faru-lb10 | 87358 | 52.7 | – |
| 20261106T092319Z-24c93e94 | 10×10 quadrants ×8 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | 41.0 | – |
| 20261106T055100Z-97b8b1be | 10×10 quadrants ×8 | ziso | 87358 | 44.2 | 43.9 |
| 20261106T105047Z-d654ed49 | 10×10 quadrants ×8 | ziso-open | 87358 | 42.0 | – |
| 20261106T100329Z-8fd34097 | 10×10 quadrants ×8 | zres-Uquad-open-far-lb10 | 87358 | 46.0 | – |
| 20261106T101230Z-dd548574 | 10×10 quadrants ×8 | zres-Uquad-open-far-lb10-full | 87358 | 81.4 | – |
| 20261106T130032Z-d6d85fbc | 10×10 quadrants ×8 | zres-Uquad-open-far-lb10-full1s | 87358 | 80.3 | – |
| 20261106T041540Z-5a557b12 | 10×10 quadrants ×12 | zaniso-U22.1 | 87358 | 21.7 | 23.7 |
| 20261106T041705Z-49c03664 | 10×10 quadrants ×12 | zaniso-Ubox | 87358 | 20.0 | 17.8 |
| 20261106T042156Z-9adf7cbc | 10×10 quadrants ×12 | zaniso-Uquad | 87358 | 29.9 | 29.7 |
| 20261106T085239Z-3e856aa1 | 10×10 quadrants ×12 | zaniso-Uquad-open-far-pwtauz-lb10 | 87358 | 45.9 | – |
| 20261106T083158Z-27847891 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs2x1+14x6-pwtauz-lb10 | 87358 | 47.2 | – |
| 20261106T081147Z-61e311c7 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs4x1+12x6-pwtauz-lb10 | 87358 | 52.6 | – |
| 20261106T083421Z-4efb59ca | 10×10 quadrants ×12 | zaniso-Uquad-open-gs4x1+12x8-pwtauz-lb10 | 87358 | 49.9 | – |
| 20261106T074353Z-7c37ca74 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10 | 87358 | 46.3 | – |
| 20261106T081057Z-fb6cd5df | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+4x4+4x16-pwtauz-lb10 | 87358 | 42.2 | – |
| 20261106T080924Z-b859d49c | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x12-pwtauz-lb10 | 87358 | 42.0 | – |
| 20261106T081755Z-27f6c11b | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x3-pwtauz-lb10 | 87358 | 50.0 | – |
| 20261106T120508Z-6048f257 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 1 | 47.4 | – |
| 20261106T072102Z-ac00a91d | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | 48.2 | – |
| 20261106T111953Z-c238a3d7 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10 | 87358 | – | 51.9 |
| 20261106T085801Z-aab0e919 | 10×10 quadrants ×12 | zaniso-Uquad-open-gs8x1+8x6-sum2-pwtauz-lb10 | 87358 | 43.9 | – |
| 20261106T070102Z-d4b0bbab | 10×10 quadrants ×12 | zaniso-Uquad-open-hs4-lb10 | 87358 | 36.9 | – |
| 20261106T083142Z-3e2023ac | 10×10 quadrants ×12 | zaniso-Uquad-open-pwtauz-es6-lb10 | 87358 | 40.9 | – |
| 20261106T072445Z-b70d3a3d | 10×10 quadrants ×12 | zaniso-Uquad-open-pwtauz-hs4-lb10 | 87358 | 37.4 | – |
| 20261106T082028Z-118de5f3 | 10×10 quadrants ×12 | zaniso-Uquad-open-pwtauz-hs60-es0.1-lb10 | 87358 | 37.0 | – |
| 20261106T052020Z-ecae1bd5 | 10×10 quadrants ×12 | zboth-Uquad | 87358 | 49.8 | 51.3 |
| 20261106T041508Z-847bdcc1 | 10×10 quadrants ×12 | ziso | 87358 | 57.6 | 57.3 |
| 20261106T045859Z-f57b8777 | 10×10 quadrants ×12 | ziso-full | 87358 | 84.6 | 83.9 |
| 20261106T050958Z-5030639c | 10×10 quadrants ×12 | ziso-lo0.05 | 87358 | 53.7 | 11.8 |
| 20261106T114950Z-d0c51159 | 10×10 quadrants ×12 | ziso-open | 1 | 59.3 | – |
| 20261106T120050Z-5d4d7a6f | 10×10 quadrants ×12 | ziso-open | 2 | 47.6 | – |
| 20261106T092609Z-4986c876 | 10×10 quadrants ×12 | ziso-open | 87358 | 54.3 | – |
| 20261106T113325Z-3be8db04 | 10×10 quadrants ×12 | ziso-open | 87358 | – | 56.6 |
| 20261106T050350Z-89024a58 | 10×10 quadrants ×12 | ziso-sh | 87358 | 55.3 | 15.0 |
| 20261106T105909Z-4aa8e95b | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10 | 1 | 48.0 | – |
| 20261106T090701Z-62ca6da7 | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10 | 87358 | 46.7 | – |
| 20261106T093404Z-b59badef | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10 | 87358 | – | 47.3 |
| 20261106T093646Z-675f87ea | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10-full | 87358 | 82.3 | – |
| 20261106T093300Z-3b35c6c8 | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10-full-aborted | 87358 | 53.4 | – |
| 20261106T131725Z-ec784676 | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 1 | 79.7 | – |
| 20261106T132530Z-331f6e52 | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 2 | 85.6 | – |
| 20261106T123004Z-ad5f335a | 10×10 quadrants ×12 | zres-Uquad-open-far-lb10-full1s | 87358 | 84.1 | – |
| 20261106T093819Z-3cabfa2c | 10×10 quadrants ×12 | zres-Uquad-open-faru-lb10 | 87358 | 38.4 | – |
| 20261106T042801Z-fe722587 | 10×10 quadrants ×12 | zsplit-Uquad | 87358 | 47.1 | 47.9 |
| 20261106T044926Z-09b4a816 | 10×10 quadrants ×12 | zsplit-Uquad-lo0.05 | 87358 | 41.0 | 48.6 |
| 20261106T052217Z-ce23aeab | 10×10 quadrants ×12 | zsplit-Uquad-lo0.05-posthoc | 87358 | 48.8 | 53.2 |
| 20261106T043931Z-00888948 | 10×10 quadrants ×12 | zsplit-Uquad-posthoc | 87358 | 55.4 | 56.7 |
| 20261106T045501Z-6482deed | 10×10 quadrants ×12 | zsplit-Uquad-posthoc-full | 87358 | 84.7 | 84.2 |
| 20261106T044216Z-fb879c51 | 10×10 quadrants ×12 | zsplit-Uquad-sh | 87358 | 44.7 | 46.1 |
| 20261106T051653Z-5fc77b5d | 10×10 quadrants ×12 | zsplit-Uquad-sh-posthoc | 87358 | 52.4 | 53.7 |
| 20261106T053204Z-eb817ec4 | 10×10 quadrants ×12 | zsplit-Uquad-tw0 | 87358 | 21.3 | 21.7 |
| 20261106T043929Z-07dc41a4 | 22×22 quadrants ×12 | ground-truth check | – | – | – |
