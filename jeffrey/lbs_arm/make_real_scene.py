"""Planning copy of a real-world cupboard scene (jeffrey/real_world/scenes/<name>.yml -> jeffrey/lbs_arm/scenes/<name>.yml).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_arm/make_real_scene.py real_world_6dof

The measured scene puts the pedestal's top at the arm's base (z = 0). The release scorer checks every collision sphere against every
cuboid, including the two fixed shoulder spheres (radius 0.105 m, the lower one centred 0.0745 m above the base), so with that pedestal
it rejects every configuration (the lower shoulder sphere is 30.5 mm inside it). The two-arm scene has one pedestal per arm, same rule. The training label ignores those two spheres; the
scorer does not. This copy lowers the pedestal's top by GAP = 0.04 m (9.5 mm below that sphere, more than the scorer's 5 mm), and
changes nothing else: the cupboard, the floor, the base pose and the grasp-point boxes are those of the measured scene.
"""
import sys
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
GAP = 0.04


def main():
    name = sys.argv[1]
    src = HERE.parent / 'real_world/scenes' / (name + '.yml')
    doc = yaml.safe_load(src.read_text())
    for c in doc['cuboids']:
        if c['name'].startswith('pedestal'):
            top = c['pose'][2] + c['dims'][2] / 2
            assert abs(top) < 1e-9, top
            c['dims'][2] = round(c['dims'][2] - GAP, 6)
            c['pose'][2] = round(c['pose'][2] - GAP / 2, 6)
    head = (f'# {name}, planning copy: jeffrey/real_world/scenes/{name}.yml with the pedestal top lowered by {GAP} m so that the release scorer,\n'
            '# which checks the fixed shoulder spheres too, accepts configurations (jeffrey/lbs_arm/make_real_scene.py). Nothing else differs.\n'
            '# Written by make_real_scene.py; do not edit by hand.\n\n')
    out = HERE / 'scenes' / (name + '.yml')
    out.write_text(head + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=200))
    print(out)


if __name__ == '__main__':
    main()
