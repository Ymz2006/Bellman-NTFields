# Pinned runtime environment

Training and evaluation use an existing Docker image:

```text
sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828
```

On the original machine it is tagged `curobo-g1:physical-audit`, with a recorded
image size of 8,687,084,268 bytes. The same image must be kept to use the
validated PyTorch, CUDA, cuRobo, planning and scoring dependencies. There is
currently no validated Dockerfile, and the image is not assumed to be pullable
from a public registry.

To move to another machine, run this on a machine that already has the image:

```bash
docker --context default save curobo-g1:physical-audit -o bellman600-environment.tar
```

On the new machine:

```bash
docker --context default load -i bellman600-environment.tar
python3 run.py doctor
python3 run.py reproduce --smoke
```

The image archive needs extra disk space, so it was not exported automatically.
The Linux NVIDIA driver, Docker and GPU container support are provided by the
host; the original validation hardware was an RTX 4090 with driver 580.159.03.
Retraining is repeatable within the same environment; bitwise equality is not
guaranteed across GPUs or different CUDA implementations.

The GUI runs in the host Python environment, separate from the GPU training
environment; the GUI dependency versions on the original machine are recorded in
`gui-packages.json`. That version list is an environment snapshot, not a
guarantee that installation succeeds on other platforms.
