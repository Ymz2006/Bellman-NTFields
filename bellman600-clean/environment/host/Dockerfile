# Host-side Python for run.py, tools/*.py, the experiment registry and the GUI,
# for machines whose system python3 is older than 3.10.
#
# This is NOT the training/evaluation environment. launch.py still starts the
# pinned image sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828
# through the host Docker daemon (socket mounted by ./docker-python).
#
# Build context is environment/ so that requirements-gui.txt is available:
#   docker build -t bellman600-host:py3.11 -f environment/host/Dockerfile environment
FROM python:3.11.9-slim-bookworm

# Static Docker CLI; talks to the host daemon via /var/run/docker.sock.
# `docker` is a wrapper that resolves the pinned image ID on classic-store daemons.
COPY --from=docker:24.0.9-cli /usr/local/bin/docker /usr/local/libexec/docker
COPY host/docker-image-alias.sh /usr/local/bin/docker

# The registry records `git rev-parse HEAD`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && chmod 755 /usr/local/bin/docker

COPY requirements-gui.txt /tmp/requirements-gui.txt
RUN pip install --no-cache-dir -r /tmp/requirements-gui.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
