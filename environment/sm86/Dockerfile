# Derived training/evaluation environment for Ampere GPUs such as RTX 3090 (sm_86).
#
# NOT the pinned, frozen environment. The pinned image's cuRobo CUDA extensions
# were built with TORCH_CUDA_ARCH_LIST=8.9 only (sm_89 code, no PTX), so they
# fail on sm_86 with "no kernel image is available". This image changes nothing
# else: it recompiles the same five extensions from the image's own /opt/curobo
# with the same setup.py flags, for 8.6 and 8.9, and replaces only those .so
# files. Runs that use it are modified-environment results, not exact
# reproductions; ./docker-python records an environment_override.json with them.
#
# Build with the classic builder (BuildKit cannot use a local image ID in FROM).
# BASE = pinned image; classic-store Docker names it by its config digest:
#   DOCKER_BUILDKIT=0 docker build -t bellman600-env:sm86 \
#     --build-arg BASE=sha256:fc21d92e72630699e9456a5cecc6f785ea8cb8f5eaacf9a26c815ea1a53dca32 \
#     environment/sm86
# Use:
#   BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py from-source --smoke
ARG BASE
FROM ${BASE}
ARG BASE
ARG CUROBO_ARCH_LIST="8.6;8.9"
LABEL org.bellman600.derived-from="${BASE}" \
      org.bellman600.change="cuRobo CUDA extensions rebuilt from /opt/curobo for TORCH_CUDA_ARCH_LIST=${CUROBO_ARCH_LIST}"

RUN set -eux; \
    src=/tmp/curobo-rebuild; \
    dst=/usr/local/lib/python3.10/dist-packages/curobo/curobolib; \
    cp -a /opt/curobo "$src"; \
    rm -rf "$src/build"; \
    cd "$src"; \
    TORCH_CUDA_ARCH_LIST="$CUROBO_ARCH_LIST" python3 setup.py build_ext; \
    built=$(ls "$src"/build/lib.*/curobo/curobolib/*_cu.*.so); \
    test "$(echo "$built" | wc -l)" -eq 5; \
    for so in $built; do \
      cuobjdump --list-elf "$so" | grep -q sm_86; \
      cuobjdump --list-elf "$so" | grep -q sm_89; \
      test -f "$dst/$(basename "$so")"; \
      cp "$so" "$dst/"; \
    done; \
    cd /; rm -rf "$src"
