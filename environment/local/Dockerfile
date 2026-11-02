# Rebuilt training/evaluation environment for this workstation (RTX 5070 = sm_120 and
# RTX 3090 = sm_86). NOT the pinned, frozen environment and NOT derived from it: the pinned
# image (CUDA 12.4, cuRobo kernels for sm_89 only) is not present on this machine and its
# CUDA 12.4 has no sm_120 support. This image rebuilds the same stack from public sources:
# Ubuntu 24.04 + CUDA 12.8 devel, Python 3.12, PyTorch 2.7.1+cu128 (pip wheels) and cuRobo
# compiled from source for sm_86 and sm_120. Results produced in it are modified-environment
# results; ./docker-python records environment_override.json for them.
#
# Build (about 20-30 minutes, mostly the cuRobo CUDA kernels):
#   docker build -t bellman600-env:local -f environment/local/Dockerfile environment/local
# Use with the registered entry points (needs the NVIDIA Container Toolkit, see README.md):
#   BELLMAN600_DERIVED_IMAGE=bellman600-env:local ./docker-python run.py cabinet --gpu 1
# The same wheels make the host virtual environment (environment/local/README.md), and the
# cuRobo package compiled here is copied into it, so both run the identical stack.
ARG BASE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE}
ARG CUROBO_REF=v0.7.8
ARG CUROBO_ARCH_LIST="8.6;12.0"
ARG MAX_JOBS=6
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_CUDA_ARCH_LIST=${CUROBO_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git ninja-build build-essential ca-certificates \
      libgl1 libglu1-mesa libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r /tmp/requirements.txt \
    && python3 -c "import torch, numpy; print(torch.__version__, torch.version.cuda, numpy.__version__)"

# cuRobo from source: the pinned image carried it at /opt/curobo and the sm86 derivation
# recompiled the same five CUDA extensions; here they are compiled for sm_86 and sm_120.
RUN git clone --depth 1 --branch ${CUROBO_REF} https://github.com/NVlabs/curobo /opt/curobo \
    && cd /opt/curobo \
    && TORCH_CUDA_ARCH_LIST="${CUROBO_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation . \
    && python3 -c "import torch, curobo, curobo.curobolib.kinematics_fused_cu, curobo.curobolib.geom_cu; print('curobo', curobo.__version__)" \
    && for so in /usr/local/lib/python3.12/dist-packages/curobo/curobolib/*_cu.*.so; do \
         cuobjdump --list-elf "$so" | grep -q sm_86 && cuobjdump --list-elf "$so" | grep -q sm_120 || exit 1; done \
    && rm -rf /opt/curobo/build

# The docker shim (environment/host/docker-image-alias.sh) accepts an opt-in image only when
# this label names the pinned image; the change label says what differs.
LABEL org.bellman600.derived-from="sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828" \
      org.bellman600.change="REBUILT from public sources, not derived: Ubuntu 24.04, CUDA 12.8.1, Python 3.12, torch 2.7.1+cu128, cuRobo ${CUROBO_REF} compiled for sm_86 and sm_120 (RTX 3090 and RTX 5070)"

WORKDIR /workspace
