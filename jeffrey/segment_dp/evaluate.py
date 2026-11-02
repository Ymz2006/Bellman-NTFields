"""Frozen-benchmark evaluation of a segment field with the unchanged evaluators; only the field loader differs.

    evaluate.py nowall  --checkpoint F --output O [--smoke]                         method/evaluate.py
    evaluate.py cabinet --method bellman --checkpoint F --scene S --tests T --output O [--smoke]
                                                                                   experiments/cabinet/evaluate.py
Both evaluator modules are imported unchanged (native RK4 + release MPPI with its original settings, planning seeds
0/1/2, release scorer) and their module-level loader load_scalar is replaced by fields.load_field, which reads a
segment-field checkpoint (and, for a plain TravelCost checkpoint, is the original loader).
The planner's optional CUDA-Graph replay of the cost network (NTF_GRAPH_COST, a speed optimization whose replay is
checked bitwise against the eager forward) is switched off for fields of several segments: they route rows by
distance, a data-dependent shape that cannot be captured, so MPPI runs the same cost eagerly. A one-segment field is
its single model and keeps the replay.
"""
import os, sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import importlib.util
import torch
from fields import load_field

EVALUATORS = dict(nowall='/clean/method/evaluate.py', cabinet='/clean/experiments/cabinet/evaluate.py')


def main():
    scene, rest = sys.argv[1], sys.argv[2:]; path = EVALUATORS[scene]
    payload = torch.load(rest[rest.index('--checkpoint')+1], map_location='cpu', weights_only=False)
    if payload.get('kind') == 'segment_field' and len(payload['model_state_dicts']) > 1:
        os.environ['NTF_GRAPH_COST'] = '0'
    spec = importlib.util.spec_from_file_location('frozen_evaluate_'+scene, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.load_scalar = lambda checkpoint, device: load_field(checkpoint, device)
    sys.argv = [path, *rest]; module.main()


if __name__ == '__main__': main()
