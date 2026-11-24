# LBS final results: LBS against ntrl-demo, RRT-Connect and LazyPRM on 2-D mazes, gap mazes and 3-D Gibson houses

Study of 2026-11-22. Request (16:00 EST, on the LBS benchmarking artifact P2NJd3SpZZHRZ9qLmnQCoB):

> expand the maze2d test, add 14x14 and 22x22. You will create a table with the 10, 14, 18, 22 mazes and you will create 3 of each with
> different seeds (this is table 1, 2d maze) Table 2: gap maze 2d [x1 / x2 / x4] from here, add x8 and for each size generate 3 of each maze
> and run everything. Table 3: gibson. For this one you will need to run a few more 3d enviornents, I want 10 in total. Pick the easy ones so
> the success rate is high. These tables and the full results you can store in a new artifact called LBS Final results. Also, I want you to
> report the time it took to generate the path, path len. These numbers will have standard deviation and mean since you will test randomly
> 500 pairs of start/end per env. For each table, there are 4 rows. ntlr-demo, lbs, rrt connect, and lprm. [...] Share the gpus with some
> other tests going on [...] SHARE, DON'T GIVE THE OTHER PROCESS YOUR COMPUTE AND DONT ACCEPT THE OTHER AGENT'S OFFER FOR COMPUTE.

## Scenes (3 instances each; `seeds.py`)

| table | scenes | instance i |
|---|---|---|
| 1, maze 2-D | maze_scale mazes 10, 14, 18, 22 cells a side | maze seed 1000 i + n (i = 1 is the layout of every earlier study) |
| 2, gap maze 2-D | cross_warp gap mazes on the 10 x 10 base, gap x1 / x2 / x4 / x8 (16 / 22 / 34 / 58 cells) | base maze seed 1000 i + 10 |
| 3, Gibson 3-D | point robot in R^3 (jeffrey/lbs scenes3.House3): Allensville, Mifflinburg, Noxapater, Markleeville, Lakeville (jeffrey/lbs) + Collierville, Benevolence, Onaga, Coffeen, Forkland | one per house |

The five new houses are the five smallest main floors (camera-position extent) of the gibson_tiny houses not used before: the "easy ones".
The SE(2) rectangle robot on the 2-D scenes, as in every local study.

## Queries (`prep.py`)

The scene's own prepare script (maze_scale/prepare.py, cross_warp/gapprepare.py --base maze10, jeffrey/lbs/prepare3.py via `prepare3r.py`)
runs unchanged except that its distance bins are replaced by one bin [0.5 cells, inf) with quota 500: 500 random (start, goal) pairs, starts
uniform over free space (clearance >= 0.1 cells), goals from the scene's 48 random goal poses, each pair certified by its truth path. The gap
mazes keep gapprepare's goal stratification (1/8 of the goals in each maze block, half in the open band). Training data (data seed 0),
truth and controls are the scripts' own.

## Planners (`bench.py`) — all on the CPU, 2 torch threads, one query at a time

| row | what |
|---|---|
| ntrl-demo | ntrl-demo's MPPI on the field trained by jeffrey/lbs with `--lift none --weight 0` (seed 87358, epoch 5,000) |
| LBS | the same MPPI on the LBS field (`--lift time --bell 1`, `--bell-long` 1 up to 24 cells, 8 above) |
| RRT-Connect | OMPL 2.0.1 (Python wheel, C++ planner), default range, `SimpleSetup.clear()` before every query, 30 s per query, then PathSimplifier.simplifyMax |
| LazyPRM | Python port of OMPL's LazyPRM.cpp (not in the 2.0.1 bindings): k = 5 within 0.2 x diagonal, lazy vertex/edge checks, fresh roadmap per query, 30 s, simplifyMax |

MPPI: planning seed 0, step budget max(200, 3 x truth path C-space length / step) as in every earlier study. Collision checks of the sampling
planners are the scorer's (exact clearance, <= 0.005 cells per coordinate). Every path is scored by maze_scale/plan.score. Time = wall time
to the path (MPPI until it reaches the goal or its budget; sampling planners solve + simplification). Path length = xy length in cells (SE(2)),
xyz length in cells for the 3-D houses (one cell = 0.25 m). Mean and std over the successful queries of the three instances (1,500 queries).

Why CPU: both GPUs train other studies' runs at the same time; one MPPI query on a shared GPU took 10-20 ms per step (latency bound), on
two CPU threads 3-4 ms. All four planners on the same CPU budget is also the like-for-like timing.

Fields reused from jeffrey/lbs (same scene, checked by bench.py): instance 1 of maze 10, maze 18, gap x1, x2, x4, and the five earlier 3-D houses.

Time limit: the first pass gave the sampling planners 10 s per query (`bench:<scene>:sampling`, kept in the registry). On 2026-11-23
01:30 the user asked for 30 s; `queue/make_jobs30.py` -> `jobs30.json` (one job per scene and planner, longest first), run by
`queue/runner30.py` next to the main runner (state30.json, runner30.log). It starts a job only while this study runs at most 4 timed
benchmarks during trainings (6 after), so the 8 physical cores are not oversubscribed. The report uses the 30 s run of an instance as soon
as it is in and marks the cells still on 10 s.

## Running

`queue/make_jobs.py` -> `queue/jobs.json`; `queue/runner.py` (nohup; one of this queue's runs per GPU next to whatever else runs there, three
CPU benchmark slots, RAM guards). `~/.venvs/ompl-bench` holds only `ompl` and reads the host venv's packages through a .pth file.

## Results

`report.py` -> `report/summary.json`, `report/tables.md`; `page.py` -> `report/lbs_final.html`, published as "2026-11-22 LBS Final Results": https://claude.ai/artifact/183QX9B49mSaMdvdrDHKee (republish the same file).

Status 2026-11-23 12:14: done. Main queue 170 / 170 jobs, 30 s pass 68 / 68, no failures. Report = 30 s runs for every sampling cell.

## Findings (1,500 queries per maze / gap cell, 500 per house; success %, time per solved query, path ÷ true shortest path)

| scene | ntrl-demo | LBS | RRT-Connect (30 s) | LazyPRM (30 s) |
|---|---|---|---|---|
| maze 10 / 14 / 18 / 22 | 90.4 / 47.9 / 21.6 / 5.3 | 100 / 97.0 / 96.3 / 69.1 | 100 / 99.9 / 98.4 / 84.9 | 100 / 100 / 99.9 / 99.1 |
| maze time (s) | 0.32-0.50 | 0.61 / 0.92 / 1.12 / 1.20 | 1.57 / 4.69 / 10.1 / 13.6 | 0.88 / 2.11 / 4.68 / 8.55 |
| maze path ratio | 1.02-1.05 | 1.03-1.04 | 1.13-1.21 | 1.18-1.35 |
| gap x1 / x2 / x4 / x8 | 75.9 / 69.1 / 56.3 / 56.4 | 100 / 99.8 / 99.5 / 95.4 | 100 / 99.1 / 96.8 / 85.8 | 100 / 100 / 99.9 / 97.3 |
| gap time (s) | 0.45-1.67 | 0.77-3.09 | 1.19-2.99 | 0.44-2.12 |
| houses (10) | 18.0-94.8 | 39.6-100 | 88.4-100 | 81.6-100 |
| house time (s) | 0.7-1.4 | 0.93-1.75 | 0.05-1.13 | 0.04-0.61 |

Per house (ntrl / LBS / RRT-Connect / LazyPRM): Collierville 89.4 / 99.6 / 100 / 100, Benevolence 94.8 / 99.8 / 100 / 100, Onaga 72.8 / 100 / 100 / 100,
Coffeen 50.2 / 75.0 / 94.2 / 88.0, Forkland 38.4 / 70.2 / 100 / 99.6, Allensville 40.2 / 56.4 / 97.6 / 95.0, Mifflinburg 47.8 / 87.8 / 100 / 100,
Noxapater 56.2 / 91.0 / 100 / 100, Markleeville 62.0 / 80.8 / 100 / 100, Lakeville 18.0 / 39.6 / 88.4 / 81.6.

1. **2-D mazes.** LBS keeps ntrl-demo's speed class (about 1 s) and adds 10-75 points of success; its paths are the shortest of all four
   (1.03-1.04 x). With 30 s the sampling planners overtake LBS on 18 x 18 and 22 x 22, at 4-11 x its time and 14-31 % longer paths.
   ntrl-demo fails by walls (1,216 of 1,421 misses at 22 x 22); LBS by stalls (447 of 463, 16 through a wall). Control by distance at
   22 x 22 (shorter / longer half of the queries by true path): LBS 79.6 / 58.7 %, ntrl 10.4 / 0.1, RRT-Connect 98.9 / 70.8, LazyPRM
   99.6 / 98.5: LBS's far field is the limit, not walls.
2. **Gap mazes.** LBS 95-100 % at every gap; ntrl-demo falls to 56 %. Mechanism, by goal position at gap x8 (goal in the open band /
   inside a maze block, 747 / 753 queries): LBS 100 / 90.8, LazyPRM 100 / 94.7, RRT-Connect 99.7 / 72.0, ntrl 98.8 / 14.3. RRT-Connect's
   misses are time-outs that mostly fail at both 10 s and 30 s (95 of 108 on gap8_1): uniform samples land mostly in the wide band, so the
   tree rarely enters the block's corridors (narrow-passage effect, grows with the band). gap x4: 99.9 / 93.5 for RRT-Connect.
3. **3-D houses.** The point robot in open rooms favours sampling (0.05-1.1 s, 82-100 %). LBS never plans through a wall (1 of 999
   misses); every miss is a stall. Control by detour (true path / straight line, pooled over the 10 houses, n = 3,637 / 1,017 / 346):
   < 1.1: LBS 93.2, RRT-Connect 99.9, LazyPRM 99.5, ntrl 72.4; 1.1-1.5: 51.8 / 95.8 / 92.4 / 17.3; >= 1.5: 24.6 / 85.3 / 75.7 / 11.0.
   The house differences follow the share of detour queries. The LBS field gets near-straight travel right and does not hold detours
   around walls in 3-D (same limit as jeffrey/lift_midpoint, where detours appear only where training traversals drive them).
4. **Time limit.** 10 s -> 30 s: RRT-Connect 22 x 22 31.4 -> 84.9, 18 x 18 55.0 -> 98.4, gap x4 86.5 -> 96.8, gap x8 78.5 -> 85.8;
   LazyPRM 22 x 22 68.0 -> 99.1. Houses and maze 10 moved <= 5 points. Share solved within 10 s on the 30 s runs is in the full tables.
5. **Timing.** LBS 4.4 ms per MPPI step on 2 CPU threads vs ntrl 2.75 ms (clearance 1.43 ms + lift table 0.12 ms + network 2.53 ms);
   the goal's lift is recomputed every step (caching it would save ~15 %, not applied). Timings were measured with up to 4 timed benches
   next to field trainings (one ~15 min window at 5, 06:30-06:50 Nov 23, after the main runner's learned benches overlapped the 30 s pass).

## LBS with the fast planner (2026-11-23 16:04-16:28; replaces the CPU LBS row)

User request: run LBS with jeffrey/lbs_fast's planner, `fast.Planner(fast.Field(net, scene), plan.settings(scene), variant='goal',
compile='reduce-overhead', block=8, noise='bulk')`, on the RTX 5070 alone. `queue/runner_fast.py` ran `jeffrey/lbs_fast/run.py bench --scene <k>
--device cuda --gpu 0 --variant 'fast|c=ro,k=8,n=bulk' --label lbsfinal` for all 34 scenes, one at a time, all 500 queries, waiting whenever
any non-display process was on GPU 0 (it never had to; the other sessions were asked to keep off it). Run ids: `queue/statefast.json`
(job `fast:<scene>`). Same prepared scenes and LBS fields (lbs_fast/scenes.json, checked against this study's runs), same step budget;
max |T - NN.out| <= 1e-6 on every scene. Every path scored by plan.score off the clock. report.py takes the LBS rows from these runs
(time = wall time of Planner.plan, start / goal on the GPU to the path on the CPU; one-off compile ~20 s per scene not counted; a converged
step past the budget would not count as success: 0 cases). House lengths recomputed as x-y-z length (plan.score's length is x-y only).
ntrl-demo, RRT-Connect and LazyPRM stay on 2 CPU threads, so the LBS times are GPU times next to CPU times. During these runs
jeffrey/arm_placement trained on the 3090 and its CPUs (load ~4 of 16 threads).

Fast GPU vs CPU LBS, same fields: success within 0.6 points in every cell (maze 100 / 97.1 / 96.3 / 69.1, gap 99.9 / 99.8 / 99.5 / 94.8),
path ratio unchanged, time per solved query 18-34 ms (maze), 21-76 ms (gap), 34-119 ms (houses) vs 0.6-1.75 s; 0.12-0.21 ms per MPPI step
vs 4.4 ms. Mechanism splits with the fast runs: maze22 short / long half 79.3 / 58.8 %; gap8 goal in band / in block 100 / 89.6 %; houses by
detour 93.2 / 51.8 / 24.3 %, 2 of 1,000 misses through a wall. Mean over all queries (failures run the full budget): 59 ms maze22, 86 ms gap8,
0.33 s Lakeville. The CPU LBS runs (bench:<scene>:learned, method lbs) stay in the registry.

## More 3-D houses, LBS only (2026-11-24 02:20, 10 hours)

User request: run LBS on more Gibson houses to find high-success ones, add them to the Gibson 3-D table; ntrl-demo only where LBS > 80 %;
both GPUs; tune the batch size for efficiency. `queue/runner_more.py` (statemore.json, runner_more.log, control.json): per house prep ->
LBS field (same recipe as the table's LBS fields) -> fast bench on the idle RTX 5070 (the table's LBS row) -> if the field's 3-seed check
is > 80 %: ntrl-demo field + CPU bench. No sampling planners on these houses (not asked). report.py adds every house with a fast bench
to the Gibson table.

Candidates: the 25 gibson_tiny houses not in the table, all larger than the ten (box side 55-136 cells vs 38-83). Preps smallest first;
trainings by the LBS success predicted from the prepared queries (detour mix weighted by the ten houses' pooled LBS success, minus 1 point
per cell of mean true path above 20; on the ten houses this ranks the 99-100 % ones first but overrates Lakeville 0.55 vs 0.40); not
trained below 0.6; preps pause while 3 prepared houses wait.

Efficiency probes (scratchpad, not registered; Collierville prep, 400 epochs, 3090):
- `--batch` is the midpoint-loss batch only (ntrl-demo's own minibatch is 2,000 pairs x 5 per epoch, fixed). 8192 -> 32768: identical
  ntrl phase (0.375 s/epoch), midpoint phase 0.45 -> 0.57 s/epoch (+27 %), no shorter training. Kept 8192 (= the table's recipe).
- Two trainings on one GPU: each 2.0x slower (77 s vs 38 s per 100 epochs, refresh 80 vs 41 s): the GPU is saturated by one, so one
  training per GPU. GPU memory: a house prep needs ~9 GB, so the 5070 takes one job at a time and houses >= 80 cells prep on the 3090 only
  (Wainscott, 86, ran out of memory on the 5070: run 20261124T072329Z-c613ebb8 failed).
