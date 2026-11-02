"""Certified cabinet suites in ntrl-demo's joint domain: experiments/cabinet/make_tests.py, unchanged, with every
joint restricted to |q| <= the scene's `joint_limit` (ntrl-demo samples its arm data with joint limits +-pi,
dataprocessing/speed_sampling_arm_normal.py: torch_ik.setJointLimits(-pi, pi)).

The restriction is applied to the generator's Validator bounds, which bound the uniform endpoint pools, the RRT
targets and every witness configuration; strata, seeds, pair distances, RRT and the release scorer are unchanged.
The scorer still checks the release joint limits (+-2 pi), which the restricted witnesses satisfy.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
import yaml

spec=importlib.util.spec_from_file_location('cabinet_make_tests','/clean/experiments/cabinet/make_tests.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def main():
    scene=Path(sys.argv[sys.argv.index('--scene')+1]);out=Path(sys.argv[sys.argv.index('--output')+1])
    limit=float(yaml.safe_load(scene.read_text())['joint_limit'])
    base=M.Validator

    class Validator(base):
        def __init__(self,scene,dev):
            super().__init__(scene,dev);self.lo=torch.clamp(self.lo,min=-limit);self.hi=torch.clamp(self.hi,max=limit)

    M.Validator=Validator;M.main()
    path=out/'suite_manifest.json';manifest=json.loads(path.read_text())
    manifest.update(joint_limit=limit,restriction=dict(wrapper='experiments/ntrl_cabinet/make_tests_pi.py',
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rule='every joint |q| <= joint_limit for endpoints, RRT targets and witnesses'))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in manifest['suites']:
        z=__import__('numpy').load(out/'suites'/name['file'],allow_pickle=True)
        assert abs(z['starts_rad']).max()<=limit+1e-5 and abs(z['goals_rad']).max()<=limit+1e-5
        for w in z['witness_trajectories']:assert abs(w).max()<=limit+1e-5


if __name__=='__main__':main()
