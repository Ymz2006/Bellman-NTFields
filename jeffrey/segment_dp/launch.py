"""Run one /clean script (argv after --) with launch.py's mounts, variables and backends; the registry command of the
run.py files in this folder. launch.py itself stays unedited (its builders are imported, not copied).

BELLMAN600_BACKEND=host     bubblewrap on the host, launch.py's host_command (BELLMAN600_HOST_MEMORY cap)
otherwise (Docker)          image BELLMAN600_DERIVED_IMAGE (default: launch.py's pinned image), --memory
                            BELLMAN600_DOCKER_MEMORY, launch.py's mounts read-only and /outputs writable
Both write environment_override.json (the image or interpreter actually used) and the exact command.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import launch as L


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--gpus', default='all')
    p.add_argument('tail', nargs=argparse.REMAINDER); a = p.parse_args()
    tail = [x for x in a.tail if x != '--']
    assert tail and tail[0].startswith('/clean/'), tail
    if os.environ.get('BELLMAN600_BACKEND', 'docker') == 'host':
        (a.output/'.cache/home').mkdir(parents=True, exist_ok=True)
        memory = os.environ.get('BELLMAN600_HOST_MEMORY', '20G'); cmd = L.host_command(tail, a.output, a.gpus, memory)
        (a.output/'host.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        override = dict(pinned_image_id=L.IMAGE, image_id_used=None, same_image_as_pinned=False, backend='host-bwrap', python=L.HOST_PYTHON, memory_cap=memory, gpus=a.gpus,
                        versions=L.host_probe(a.output, a.gpus), written_by='jeffrey/segment_dp/launch.py (BELLMAN600_BACKEND=host, launch.py builders)')
        (a.output/'environment_override.json').write_text(json.dumps(override, indent=2)+'\n')
        subprocess.run(cmd, check=True); return
    image = os.environ.get('BELLMAN600_DERIVED_IMAGE', L.IMAGE); memory = os.environ.get('BELLMAN600_DOCKER_MEMORY', '20G')
    used = subprocess.run(['docker', '--context', 'default', 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True).stdout.strip()
    cmd = ['docker', '--context', 'default', 'run', '--rm', '--memory', memory, '--memory-swap', memory, '--gpus', a.gpus, '--ipc=host', '--user', f'{os.getuid()}:{os.getgid()}', '-w', '/workspace']
    for v in L.VARIABLES: cmd += ['-e', v]
    cmd += ['-e', 'HOME=/outputs/.cache/home']
    for src, dst in L.MOUNTS: cmd += ['-v', str(src)+':'+dst+':ro']
    cmd += ['-v', str(a.output.resolve())+':/outputs', image, 'python3', '-u', *tail]
    (a.output/'.cache/home').mkdir(parents=True, exist_ok=True)
    (a.output/'docker.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    override = dict(pinned_image_id=L.IMAGE, image=image, image_id_used=used, same_image_as_pinned=used == L.IMAGE, backend='docker', memory_cap=memory, gpus=a.gpus,
                    written_by='jeffrey/segment_dp/launch.py (Docker backend, launch.py mounts and variables)')
    (a.output/'environment_override.json').write_text(json.dumps(override, indent=2)+'\n')
    subprocess.run(cmd, check=True)


if __name__ == '__main__': main()
