# Reproduce Bellman NTFields from pose sampling

This is the clean **no-wall** UR5e method: offline speed + transition bank +
338177-parameter TravelCost, with Bellman/Eikonal/direction losses6/2/2.

The independent command generates its own training poses and physical labels,
trains speed from random initialization, samples the bank, trains TravelCost,
and evaluates the **same fixed600queries**. It does not load historical
`assets/speed.pt`, `assets/bank/`, checkpoints, original pose pools, or reference
trajectories. `assets/` and `training_inputs/` may be absent.

## 1. Get the code and the frozen problem

Requirements: Linux, NVIDIA CUDA-capable GPU, GPU-enabled Docker, host Python3.10+.
The recorded full validation uses RTX4090 and the pinned image; an Apple Silicon
Mac can run the GUI but not this NVIDIA training container. Use a Linux GPU server
for training. You need access to the private GitHub repository.

If the host `python3` is older than 3.10 (e.g. Ubuntu 20.04's 3.8 fails with
`'PosixPath' object has no attribute 'is_relative_to'`), replace `python3` with
`./docker-python` in the commands below. It runs the same host-side scripts in a
Python 3.11 container (`environment/host/Dockerfile`) that drives the host Docker
daemon; training still uses the pinned image. GUI dependencies are preinstalled,
so for step 4 skip the venv and run `./docker-python gui/viewer.py ...`. It also
adapts to Docker without the containerd image store (pinned ID resolved by its
verified config digest) and to NVIDIA drivers older than 550 (host libcuda used
instead of the image's CUDA 12.4 compat library, which fails on GeForce GPUs).

The pinned image's cuRobo kernels are sm_89 only (RTX 40-series/Ada). On other
GPUs, e.g. RTX 3090 (sm_86), opt in to a derived image that only recompiles
cuRobo (`environment/sm86/Dockerfile`, built on first use):
`BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py ...`.
That is a modified environment; each run records `environment_override.json`.

```bash
gh repo clone Rtlyc/bellman600-clean
cd bellman600-clean
mkdir -p dist
gh release download reproducibility-v2 --repo Rtlyc/bellman600-clean \
  --pattern 'bellman600-runtime.tar.gz*' --dir dist
(cd dist && sha256sum -c bellman600-runtime.tar.gz.sha256)
python3 tools/runtime_assets.py unpack dist/bellman600-runtime.tar.gz
```

The small runtime bundle contains the frozen robot/meshes and problem/runtime.
The six100-query suites, scene and normalization are **also checked into Git**.
Do not resample, repair, replace or filter evaluation queries. This is a fixed
benchmark, not a held-out statistical generalization study.

## 2. Restore the exact execution environment

The release supplies gzip-compressed Docker-image parts. No unpublished image
registry or pre-existing local tag is assumed.

```bash
gh release download reproducibility-v2 --repo Rtlyc/bellman600-clean \
  --pattern 'bellman600-environment.tar.gz.part-*' \
  --pattern 'environment-SHA256SUMS' --dir dist
(cd dist && sha256sum -c environment-SHA256SUMS)
cat dist/bellman600-environment.tar.gz.part-* > dist/bellman600-environment.tar.gz
docker --context default load -i dist/bellman600-environment.tar.gz
python3 run.py doctor-source
```

Pinned image ID:
`sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828`.
Docker load preserves this ID; the launcher uses the ID, not a mutable tag.
Allow disk space for the compressed download and about8.7GBof image layers.
GUI dependencies are separate: `environment/requirements-gui.txt`.

## 3. Run the entire method

```bash
python3 run.py from-source --smoke   # Small integration test; not a benchmark
python3 run.py from-source           # Full original sampling and full600evaluation
```

Only generate poses, train speed and sample the bank:

```bash
python3 run.py from-source-prepare
```

The registry prints a fresh output path, `results/registered_<run-id>/`.
Exact repeats are guarded; repeat for a concrete reason:

```bash
python3 run.py from-source --repeat-reason 'Verify reproducibility on colleague GPU'
```

All stages are recorded with commands, seeds, timings and hashes:

|Stage|Implementation|Output|
|---|---|---|
|FK task-pose sampling|`data_generation/source/sample_fk_ur5e.py`|cup/pour pools|
|Pair construction and physical labels|`data_generation/generate.py`, source converter|`sampled/inputs/original/`|
|Coverage augmentation|`data_generation/prepare_coverage.py`|`sampled/inputs/augmentation/`|
|Speed and bank|`experiments/from_source/prepare.py`, `bank.py`|`prepared/speed.pt`, `prepared/bank/`|
|TravelCost|`experiments/from_source/training.py`|`training/checkpoint.pt`|
|Strict scoring|`method/evaluate.py` and frozen scorer|`evaluation/evaluation.json`, trajectories|

Source samplers retain their original recipe; path adapters and provenance are
in `data_generation/PROVENANCE.json`. Preparation never reads evaluation data.
The full raw data-generation stage adds time beyond the previously reported
~10.9minutes that began with existing pose pools. Refer to measured evidence.

## 4. Inspect results and open the real Viser GUI

```bash
python3 -m venv .venv-gui
.venv-gui/bin/pip install -r environment/requirements-gui.txt
.venv-gui/bin/python gui/viewer.py \
  --evaluation-dir results/registered_<run-id>/evaluation --port 8901
```

Open http://127.0.0.1:8901. Real UR5e meshes, all6tasks, MPPI planning seeds0/1/2,
and RK4 recorded paths. The GUI does not require a GPU or recompute trajectories.
For existing-baseline playback without training, run `python gui/viewer.py` with
the runtime bundle installed: original recorded baseline paths are already in Git.
GUI defaults point to that original baseline; no bank/speed/checkpoint is needed
for recorded replay. The optional frozen asset bundle below is for historical
training/evaluation and numerical comparison.

## 5. Distinguish method reproduction from historical exact replay

**Independent method** (`from-source`): no old trained artifacts or source pools.
It generates its own data and reports its own measured result. Cross-GPU bitwise
identity is not promised; do not label any fresh result88.72%without measuring it.

**Historical exact replay** (`rebuild-reproduce`): uses original saved training
pose pools/labels to retrain speed and bank, then checks against frozen artifacts.
On the validated setup, checkpoint hash and all query success flags match the
historical run: MPPI1597/1800, RK4548/600. Obtain the optional packages:

```bash
gh release download reproducibility-v2 --repo Rtlyc/bellman600-clean \
  --pattern 'bellman600-clean-assets.tar.gz*' \
  --pattern 'no-wall-training-inputs.tar.gz*' --dir dist
(cd dist && sha256sum -c bellman600-clean-assets.tar.gz.sha256)
(cd dist && sha256sum -c no-wall-training-inputs.tar.gz.sha256)
python3 tools/assets.py unpack dist/bellman600-clean-assets.tar.gz
tar -xzf dist/no-wall-training-inputs.tar.gz
python3 run.py rebuild-reproduce
```

These packages are optional reference material for the independent command.
One training seed87358; MPPI seeds0/1/2are planning seeds. Networks do not accept
new obstacle locations. Structure/losses alone do not guarantee path safety;
independent scoring checks full-arm collision and20degree regional orientation
with at most0.001rad per-joint interpolation.
