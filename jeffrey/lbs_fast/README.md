# LBS inference time: how fast can one path be generated on an LBS field

Study of 2026-11-23. Request:

> your task is to optimize the lbs ntfield inference time. Task 1) You will first try as much as possible to optimize for the inference
> time in the mppi. task 2) you can try to learn a distance field given a configuration so you do not have to query the distance field
> every time. task 3) you will try bidirectional mppi, where you plan, then start/end and the plan again. the goal is to get the shortest
> possible inference time. You don't need to verify the path, you jsut need to generate it

Starting point: `jeffrey/lbs_final` (LBS against ntrl-demo, RRT-Connect and LazyPRM). There LBS took 0.6-1.2 s per solved query on the 2-D
mazes, 4.4 ms per MPPI step on two CPU threads. Fields, scenes and queries here are those of that study, unchanged (`scenes.json`: scene key ->
its prepare run and LBS field run). Nothing is retrained except the small clearance models of Task 2.

## What is timed

Wall time of one call that turns a (start, goal) pair into a path, one query at a time, planning seed 0. The step budget is the earlier
studies' max(200, 3 x true path length / step). The path is not checked while planning. After the timing loop the exact scorer
(`maze_scale/plan.score`: reaches the goal and every configuration at 0.005 cells per coordinate is collision-free) scores every path, off
the clock, as the control: a faster planner that returns worse paths shows up there. One-off setup (torch.compile, building CUDA graphs,
compiling the C++ kernel) is not in the per-query time; it is recorded per variant (`setup_seconds`, about 10-20 s per compiled planner).

Machine: Ryzen 7 7800X3D (8 cores), RTX 3090 (GPU 1), torch 2.7.1. Timed runs go one at a time and only while no training runs
(`queue/runner.py`, lane `timed`).

## Code

| file | what |
|---|---|
| `fast.py` | `Field` (frozen field: normalized weights, Fourier matrix with the lift column, lift profile G, clearance function), `make_kernel` (one MPPI step for A agents as one tensor function), `Planner` (variants `goal`, `meet`, `both`; eager / torch.compile / CUDA graphs; K steps per call), `reference` (the lbs_final planner) |
| `native.cpp`, `native.py` | the whole MPPI loop of one query in C++ for the CPU (exact maze / 3-D house clearance, lift, network, metric), rows of a step split over OpenMP threads; loaded through ctypes, built into `~/.cache/lbs_fast` |
| `distnet.py`, `dist_train.py` | Task 2: learned clearance field (`grid` = learned multi-resolution grid, `mlp` = Fourier-feature MLP) |
| `bench.py` | the timing benchmark of one scene: a list of planner variants on one device, then the scorer |
| `run.py` | registry stages `bench` and `dist` (sources frozen into `snap/<hash>/`) |
| `queue/make_jobs.py`, `queue/runner.py` | job list and runner (`state.json`: job name -> run id) |
| `report.py`, `page.py`, `page_style.py`, `page_text.json` | `report/summary.json`, `report/tables.md`, `report/lbs_fast.html` |

## Task 1: what was changed in the MPPI step

The reference step (`plan.mppi` + `plan.field_cost`): 50 rollouts x 2 evaluated states = 100 (state, goal) pairs through `NN.out`.

1. **Goal encoded once.** `NN.out` encodes the 100 states and 100 copies of the goal. The field is a metric between two encodings, so the
   goal's encoding is computed once per query; a step encodes 100 rows instead of 200.
2. **Weights frozen, no autograd.** `NN.out` recomputes the Lipschitz-normalized weights of all 9 layers and builds a graph for gradients on
   every call.
3. **Lift for the states only.** The clearance (rectangle against the walls) and the lift height were computed for all 200 rows.
4. **One query, no bookkeeping.** The batched planner's index arithmetic for active queries is gone.
5. **The step as one compiled function** (`torch.compile`): sampling arithmetic, clearance, lift, network, metric, softmax update.
6. **GPU: CUDA graph replay** (`mode='reduce-overhead'`), **K steps per call** with one read-back of the positions per block (the arrival test
   is done on the block; at most K - 1 steps are wasted after arrival), and one random draw per block.
7. **CPU: native kernel.** The network is 9 products of a (rows x 256) matrix with a 256 x 256 matrix; a BLAS call on so small a matrix
   scales poorly over threads. The kernel gives each thread a share of the rows and runs the whole network on them: one parallel region per
   step.

Steps 1-5 with the reference's random numbers reproduce the reference's path (same stopping step; checked per query). Steps 6-7 draw
different random numbers, so their paths are statistically, not numerically, the reference's.

Knobs that change the planner (controls: success): step multiplier m (step x m, temperature / m, tolerance unchanged, budget / m) and
rollout samples.

## Task 2: learned clearance

The lift needs the clearance d(q) of every rollout state. `distnet.py` regresses clamp(d, -0.05, 0.25) cells (the lift only depends on d
inside [-0.0175, 0.175]) and the planner applies the closed-form lift profile to the model's output. Regressing d instead of the lift height
keeps the target Lipschitz. Samples are drawn on the fly (uniform in the C-space box, half redrawn near obstacles), L1 loss, 6,000 Adam steps
of 32,768 samples, 40-210 s per scene. Models: `grid` (8 levels, 1 to 27 nodes per cell, 2 features per node, dense up to 2^21 nodes and
hashed above, one hidden layer of 32) on all 34 scenes; `mlp128` (3 x 128) and `mlp512` (4 x 512) on three scenes for comparison. Control:
the exact clearance in the same planner.

## Task 3: bidirectional MPPI

- `meet`: two agents, one from the start and one from the goal, each one's target is the other's current position. Both agents' rollouts
  and the two current positions go through the encoder in one call. When the agents are within 3 steps + tolerance, the goal-side agent
  stops and the start-side agent finishes on it with the usual tolerance. Path = start agent's trace + the goal agent's trace reversed.
  Budget: half the reference's iterations (the same total number of agent steps).
- `both`: two independent agents with fixed targets (start -> goal, goal -> start); stops when either arrives or when one agent's position
  comes within the tolerance of any earlier position of the other.
- Control `goal` with `rev=1`: the one-way planner from the goal to the start, to separate "two chances" from "meeting half-way".

## Runs

Job name -> registered run id: `queue/state.json`. Job groups: `dist:<scene>:<model>`, `ladder:<scene>:<cpu1|cpu2|cpu4|cpu8|gpu>` (first 100
queries of maze10_1, maze22_1, gap4_1, Onaga3d), `knobs:<scene>:gpu` (300 queries), `distcmp:<scene>:<gpu|cpu2>`, `final:<scene>:<gpu|cpu>`
(all 34 scenes, first 250 queries), `direction:<scene>:gpu`.

## Results

`report.py` -> `report/summary.json`, `report/tables.md`; `page.py` -> `report/lbs_fast.html`.

## Status 2026-11-23 15:58: stopped by the user

The user stopped the study after the first results ("cuda graph 8 steps per call is good enough"). The queue (`queue/PAUSE` is set) had
finished the 40 clearance trainings and 14 of 106 timed jobs. No report page was published. `report/tables.md` holds what is in.

**Only the 10 x 10 maze ladder (jobs `ladder:maze10_1:*`, 14:03-14:12) was timed on an idle machine.** From about 14:37 `jeffrey/arm_placement`
trained on the same GPU and CPUs: the GPU timings of `ladder:maze22_1:gpu` and the 8-thread CPU timings of maze22_1 / gap4_1 are invalid
(noted on the runs); CPU timings up to 4 threads, success and step counts are unaffected. Run `20261123T203354Z-9183ebbd` was killed mid-run.

Maze 10 x 10, first 100 queries, all solved by every variant (ms per MPPI step / ms per query):

| planner | ms per step | ms per query |
|---|---|---|
| reference (lbs_final planner), CPU 2 threads | 4.14 | 604 |
| reference, CPU 8 threads | 3.02 | 440 |
| goal encoded once, frozen weights, no autograd, CPU 2 threads | 2.55 | 372 |
| + step compiled, CPU 2 threads | 0.89 | 130 |
| native C++ kernel, 1 / 2 / 4 / 8 threads | 1.32 / 0.65 / 0.35 / 0.195 | 192 / 95 / 52 / 28.4 |
| native, 8 threads, 25 / 12 samples | 0.107 / 0.070 | 15.7 / 10.5 |
| native, 8 threads, bidirectional / + 2 x step | 0.384 / 0.379 | 28.6 / 14.6 |
| reference on the GPU (RTX 3090) | 2.74 | 399 |
| GPU compiled | 0.434 | 63.3 |
| GPU CUDA graph, 1 step per call | 0.294 | 42.8 |
| GPU CUDA graph, 8 steps per call (reference random numbers / one draw per call) | 0.147 / 0.135 | 21.4 / 19.7 |
| GPU CUDA graph, 16 steps per call | 0.126 | 18.4 |

The cached and compiled variants stop at the reference's step on 100 of 100 queries (same planner, same path up to rounding).

Not registered (development tests on an idle GPU, 22 x 22 maze, first 100 queries): one-way 58 solved at 38 ms per solved query, learned grid
clearance 58 at 38 ms, bidirectional `meet` 69 at 21 ms, `meet` with 2 x step 67 at 11 ms, 3 x step 58 at 8.5 ms, `both` 72 at 37 ms.

Learned clearance (all 34 scenes trained, `dist:<scene>:grid`): mean error in the lift band 0.0005-0.0009 cells on mazes and gap mazes,
0.0013-0.0049 cells in the houses; `mlp128` 0.006-0.039, `mlp512` 0.001-0.009 on the three comparison scenes.

Not run: step-size / sample sweeps (`knobs`), learned-clearance planning comparison (`distcmp`), the direction control and the all-scene runs
(`final`). To resume: remove `queue/PAUSE` and start `queue/runner.py` on an idle machine; redo the maze22_1 and gap4_1 GPU ladders.

The chosen planner: `fast.Planner(field, settings, variant='goal', compile='reduce-overhead', block=8, noise='bulk')` (bench variant
`c=ro,k=8,n=bulk`).
