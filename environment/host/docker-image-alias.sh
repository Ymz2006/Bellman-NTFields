#!/bin/bash
# Installed as `docker` in the host-tools image; forwards to the real CLI after
# host-compatibility adjustments that apply only to the pinned environment image.
# Every adjustment is logged to stderr, and for `docker run` recorded in
# <output>/environment_override.json next to launch.py's docker.command.json.
#
# 1. Image ID. launch.py pins the environment by its OCI image-index digest,
#    which is the image ID under Docker's containerd image store. Daemons using
#    the classic store (e.g. Docker 20.10) name the same loaded image by its
#    config digest. Verified from bellman600-environment.tar.gz (reproducibility-v2):
#      index e7749bdb... -> linux/amd64 manifest bd60d1ea... -> config fc21d92e...
#    Substituted only when the pinned ID is absent and that config is present.
#
# 2. Derived image (opt-in, BELLMAN600_DERIVED_IMAGE). Replaces the pinned image
#    with one built FROM it, e.g. environment/sm86 for RTX 3090. Accepted only if
#    its org.bellman600.derived-from label names the pinned image.
#
# 3. Driver libraries. The image's CUDA 12.4 ships forward-compat libcuda
#    550.54.15. With an older host driver, nvidia-container-toolkit mounts it,
#    and on GeForce GPUs CUDA then fails with error 804. ./docker-python passes
#    the host driver libraries in BELLMAN600_HOST_DRIVER_LIBS (soname=path,...);
#    they are mounted read-only into /usr/local/nvidia/lib64 under their soname
#    and unversioned name (Warp opens libcuda.so), which the image's
#    LD_LIBRARY_PATH already searches first (CUDA minor-version compatibility).
PINNED=sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828
CONFIG=sha256:fc21d92e72630699e9456a5cecc6f785ea8cb8f5eaacf9a26c815ea1a53dca32
REAL=/usr/local/libexec/docker

case " $* " in
  *" $PINNED "*) ;;
  *) exec "$REAL" "$@" ;;
esac

inspect() { "$REAL" image inspect --format "$2" "$1" 2>/dev/null; }

image=$PINNED
derived=
change=
if [ -n "${BELLMAN600_DERIVED_IMAGE:-}" ]; then
  derived=$BELLMAN600_DERIVED_IMAGE
  image=$(inspect "$derived" '{{.Id}}') || { echo "docker-python: derived image $derived not found" >&2; exit 1; }
  base=$(inspect "$derived" '{{index .Config.Labels "org.bellman600.derived-from"}}')
  if [ "$base" != "$PINNED" ] && [ "$base" != "$CONFIG" ]; then
    echo "docker-python: refusing $derived: derived-from label '$base' is not the pinned image" >&2
    exit 1
  fi
  change=$(inspect "$derived" '{{index .Config.Labels "org.bellman600.change"}}')
  echo "docker-python: DERIVED ENVIRONMENT $derived ($image) instead of pinned $PINNED: $change" >&2
elif ! inspect "$PINNED" '{{.Id}}' >/dev/null && inspect "$CONFIG" '{{.Id}}' >/dev/null; then
  image=$CONFIG
  echo "docker-python: classic image store; using config digest $CONFIG for pinned index $PINNED" >&2
fi

args=()
run=0
outputs=
libs_used=
for a in "$@"; do
  [ "$a" = run ] && run=1
  case "$a" in *:/outputs) outputs=${a%:/outputs} ;; esac
  if [ "$a" = "$PINNED" ]; then
    if [ "$run" = 1 ] && [ -n "${BELLMAN600_HOST_DRIVER_LIBS:-}" ]; then
      IFS=, read -ra libs <<< "$BELLMAN600_HOST_DRIVER_LIBS"
      for pair in "${libs[@]}"; do
        # Warp dlopens the unversioned name (libcuda.so), so cover both.
        so=${pair%%=*}
        args+=(-v "${pair#*=}:/usr/local/nvidia/lib64/$so:ro" -v "${pair#*=}:/usr/local/nvidia/lib64/${so%.so.*}.so:ro")
      done
      libs_used=$BELLMAN600_HOST_DRIVER_LIBS
      echo "docker-python: host driver older than image CUDA compat; using host libraries $libs_used" >&2
    fi
    a=$image
  fi
  args+=("$a")
done

if [ "$run" = 1 ] && [ -n "$outputs" ] && { [ "$image" != "$PINNED" ] || [ -n "$libs_used" ]; }; then
  python3 - "$outputs/environment_override.json" "$PINNED" "$image" "$derived" "$change" "$libs_used" <<'EOF'
import json, sys
path, pinned, used, derived, change, libs = sys.argv[1:]
with open(path, 'x') as f:
    json.dump({'pinned_image_id': pinned, 'image_id_used': used,
               'same_image_as_pinned': not derived,
               'derived_image': derived or None, 'derived_change': change or None,
               'host_driver_libraries': libs.split(',') if libs else [],
               'written_by': 'environment/host/docker-image-alias.sh'}, f, indent=2)
    f.write('\n')
EOF
fi
exec "$REAL" "${args[@]}"
