"""One LBS field for a scene with several arms (real_world_12dof: 12 joints, lifted onto 13 dimensions): pipeline6.py, unchanged, with

    the scene        arm12.ArmSet in place of arm.CabinetArm (the module `arm` that pipeline6 imports is arm12 here)
    the scorer       core.score at 0.001 rad per joint (the release scorer's resolution; pipeline6's own count uses 0.002)

There is no release evaluator for two arms (the release scorer and its cuRobo robot are one arm), so this pipeline's count is the
result: the frozen queries planned by ntrl-demo's MPPI in the training coordinates (core.plan, planning seeds 0, 1, 2) and scored by
the rule that certified the suites (arm12: every arm by the release rule, and the arms 0.005 m apart). Arguments: pipeline6.py's.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import core as C  # noqa: E402

_score = C.score


def score(scene, paths, converged, step_rad=1e-3):
    return _score(scene, paths, converged, step_rad=step_rad)


C.score = score
import pipeline6  # noqa: E402

if __name__ == '__main__':
    pipeline6.main()
