# Rebuilt local environment (RTX 5070 + RTX 3090 workstation)

The pinned training image (`sha256:e7749bdb…`, CUDA 12.4, cuRobo kernels for sm_89) is not on this
machine and could not drive the RTX 5070 (sm_120 needs CUDA ≥ 12.8). This directory rebuilds the same
stack from public sources. **Every result produced with it is a modified-environment result**: each run
records `environment_override.json` next to its outputs.

|Piece|What|
|---|---|
|`Dockerfile`|Ubuntu 24.04 + CUDA 12.8.1 devel, Python 3.12, PyTorch 2.7.1+cu128 (pip wheels), the packages of `requirements.txt`, cuRobo v0.7.8 compiled from source for sm_86 and sm_120. Image tag `bellman600-env:local`.|
|`requirements.txt`|The Python packages, shared by the image and the host virtual environment.|
|Host venv|`~/.venvs/bellman-ntfield`: the same wheels installed with `uv`, plus the `curobo` package copied out of the image (identical Python 3.12.3 and torch wheels, so the compiled kernels load unchanged).|
|Host backend|`BELLMAN600_BACKEND=host` makes `launch.py` run the frozen code on the host inside a bubblewrap user namespace with the container's path layout (`/clean`, `/repo`, `/workspace`, `/opt/ur5e_description` read-only, `/outputs` writable), the same environment variables, `CUDA_DEVICE_ORDER=PCI_BUS_ID` (so `--gpu N` is the nvidia-smi index) and a `systemd-run --user` memory cap (`BELLMAN600_HOST_MEMORY`, default 20G).|

## Build the image

```bash
docker build --network=host -t bellman600-env:local -f environment/local/Dockerfile environment/local
```

About 10 minutes with cached wheels. `--network=host` is needed on this machine because the default
build network cannot reach the apt mirrors.

## Run inside Docker

Docker 29 reports `failed to discover GPU vendor from CDI` for `--gpus` until the NVIDIA Container Toolkit is
installed and a CDI spec exists. Installed on this workstation on 2026-10-25 (toolkit 1.20.1, CDI spec
`/etc/cdi/nvidia.yaml` naming GPUs 0 and 1, Docker runtime `nvidia` registered). One-time setup, root required:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
docker run --rm --gpus all bellman600-env:local nvidia-smi -L
```

Then the registered entry points work with the image as the opt-in derived environment (the docker shim
accepts it through its `org.bellman600.derived-from` label and records the change):

```bash
BELLMAN600_DERIVED_IMAGE=bellman600-env:local ./docker-python run.py cabinet-metric --gpu 1
```

`docker-python` first builds the small host-tools image `bellman600-host:py3.11`; on this machine that build
needs `BELLMAN600_BUILD_NETWORK=host` (the default build network has no apt/pip access, see above), e.g.

```bash
BELLMAN600_BUILD_NETWORK=host BELLMAN600_DERIVED_IMAGE=bellman600-env:local ./docker-python run.py face-speed --method bellman --cover faces --sampled registered_20261019T034213Z-f9c501f9 --gpu 1
```

`launch.py` caps each training container at `BELLMAN600_DOCKER_MEMORY` (default 20G, swap disabled) so a
runaway run cannot exhaust the 30 GB of host RAM. Verified 2026-10-25: the smoke pipelines
`registered_20261025T164148Z-e7c03212` (Bellman faces, GPU 1) and `registered_20261025T164201Z-42c4578b`
(metric faces, GPU 0) ran end to end inside the image; each run's `environment_override.json` records the
derived image and `docker.command.json` the container flags.

## Run on the host (what this workstation uses now)

```bash
~/.local/bin/uv venv --python /usr/bin/python3 ~/.venvs/bellman-ntfield
~/.local/bin/uv pip install --python ~/.venvs/bellman-ntfield/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
~/.local/bin/uv pip install --python ~/.venvs/bellman-ntfield/bin/python -r environment/local/requirements.txt pip
# cuRobo: copy the package compiled in the image (same Python and torch wheels)
cid=$(docker create bellman600-env:local) && docker cp $cid:/usr/local/lib/python3.12/dist-packages/curobo ~/.venvs/bellman-ntfield/lib/python3.12/site-packages/ \
  && docker cp $cid:/usr/local/lib/python3.12/dist-packages/. /tmp/dist-packages-list >/dev/null 2>&1; docker rm $cid >/dev/null
export BELLMAN600_BACKEND=host
~/.venvs/bellman-ntfield/bin/python run.py face-speed --method metric --cover faces --gpu 0
```

`python3` on this machine has neither torch nor numpy, so the host-side `run.py` is also run with the venv
interpreter; the registry then records the venv in the spec's `environment` (`backend: host-bwrap`), which
keeps host and pinned-image runs apart. GPU 0 is the RTX 5070 (12 GB, also drives the display), GPU 1 the
RTX 3090 (24 GB).

## Checks

`experiments/face_speed/tests/test_geometry.py` compares the pure-torch kinematics used by the face-speed
experiment with the cuRobo-made labels of a cabinet run (agreement to 1e-6 m); `tests/curobo_host_check.py`
(written when cuRobo is installed) compares the venv's cuRobo kinematics with the same labels and replays
the frozen witness certificates through the unchanged release scorer.
