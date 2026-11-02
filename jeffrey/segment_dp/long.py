"""Long-gap cabinet diagnostic (jeffrey/gap_curriculum/long.py, imported unchanged) for segment fields.

The same 400 uncertified long queries (gap 0.8-1.6, pairs of the frozen suites' endpoints, seed 20261027), native
RK4 and optionally the release MPPI; only the loader of the shared evaluator module is replaced by fields.load_field,
which reads segment-field checkpoints (and plain TravelCost checkpoints, unchanged). The planner's CUDA-Graph replay
of the cost (NTF_GRAPH_COST, bitwise-equal to eager) is off: the segment field's distance routing cannot be captured.
"""
import os, sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import importlib.util
from fields import load_field

if __name__ == '__main__':
    os.environ['NTF_GRAPH_COST'] = '0'
    spec = importlib.util.spec_from_file_location('gap_curriculum_long', '/clean/jeffrey/gap_curriculum/long.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.SHARED.load_scalar = lambda checkpoint, device: load_field(checkpoint, device)
    module.main()
