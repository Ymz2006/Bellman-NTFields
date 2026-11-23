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
| RRT-Connect | OMPL 2.0.1 (Python wheel, C++ planner), default range, 10 s per query, then PathSimplifier.simplifyMax |
| LazyPRM | Python port of OMPL's LazyPRM.cpp (not in the 2.0.1 bindings): k = 5 within 0.2 x diagonal, lazy vertex/edge checks, fresh roadmap per query, 10 s, simplifyMax |

MPPI: planning seed 0, step budget max(200, 3 x truth path C-space length / step) as in every earlier study. Collision checks of the sampling
planners are the scorer's (exact clearance, <= 0.005 cells per coordinate). Every path is scored by maze_scale/plan.score. Time = wall time
to the path (MPPI until it reaches the goal or its budget; sampling planners solve + simplification). Path length = xy length in cells (SE(2)),
xyz length in cells for the 3-D houses (one cell = 0.25 m). Mean and std over the successful queries of the three instances (1,500 queries).

Why CPU: both GPUs train other studies' runs at the same time; one MPPI query on a shared GPU took 10-20 ms per step (latency bound), on
two CPU threads 3-4 ms. All four planners on the same CPU budget is also the like-for-like timing.

Fields reused from jeffrey/lbs (same scene, checked by bench.py): instance 1 of maze 10, maze 18, gap x1, x2, x4, and the five earlier 3-D houses.

## Running

`queue/make_jobs.py` -> `queue/jobs.json`; `queue/runner.py` (nohup; one of this queue's runs per GPU next to whatever else runs there, three
CPU benchmark slots, RAM guards). `~/.venvs/ompl-bench` holds only `ompl` and reads the host venv's packages through a .pth file.

## Results

`report.py` -> `report/summary.json`, `report/tables.md`; `page.py` -> `report/lbs_final.html`, published as "2026-11-22 LBS Final Results": https://claude.ai/artifact/183QX9B49mSaMdvdrDHKee (republish the same file).
