"""Run one /clean script (argv after --) in the same environment as launch.py; used by run_rounds.py.

Same as launch.py here but the script is an argument rather than pipeline.py: launch.py's mounts, variables and
host/Docker command builders unchanged, environment_override.json / host.command.json recorded.
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
                        versions=L.host_probe(a.output, a.gpus), written_by='jeffrey/gap_curriculum/stage_launch.py (BELLMAN600_BACKEND=host, launch.py builders)')
        (a.output/'environment_override.json').write_text(json.dumps(override, indent=2)+'\n')
        subprocess.run(cmd, check=True); return
    image = os.environ.get('BELLMAN600_DERIVED_IMAGE', L.IMAGE); memory = os.environ.get('BELLMAN600_DOCKER_MEMORY', '20G')
    cmd = ['docker', '--context', 'default', 'run', '--rm', '--memory', memory, '--memory-swap', memory, '--gpus', a.gpus, '--ipc=host', '--user', f'{os.getuid()}:{os.getgid()}', '-w', '/workspace']
    for v in L.VARIABLES: cmd += ['-e', v]
    for src, dst in L.MOUNTS: cmd += ['-v', str(src)+':'+dst+':ro']
    cmd += ['-v', str(a.output.resolve())+':/outputs', image, 'python3', '-u', *tail]
    (a.output/'docker.command.json').write_text(json.dumps(cmd, indent=2)+'\n'); subprocess.run(cmd, check=True)


if __name__ == '__main__': main()
