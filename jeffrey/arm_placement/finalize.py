"""Copy what has to outlive results/ (git-ignored, and this machine is not persistent) into the study folder:

    models/<placement>/<field>_s<seed>.pt      weights of every finished field run (model_state_dict, B_state_dict, embed_config, epoch; no
                                               optimizer state), models/<placement>/scene.json + robot_model.npz (the scene the lift needs)
    models/best/                               the chosen placement's two models as full checkpoints (with optimizer state), its scene and
                                               its queries
    models/index.json                          placement -> files, registered run ids, solved shares
    report/demo.json, report/demo.npz          the demo run's results (picked configurations and paths), if --demo is given

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/finalize.py --best p_d100_s30_h60 [--demo results/registered_<demo run>]
"""
import argparse
import json
import shutil
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--best')
    p.add_argument('--demo', type=Path)
    a = p.parse_args()
    S = json.loads((HERE / 'report/summary.json').read_text())
    M = HERE / 'models'
    M.mkdir(exist_ok=True)
    index = {}
    for pl in S['placements']:
        if not pl['fields']:
            continue
        d = M / pl['name']
        d.mkdir(exist_ok=True)
        prep = ROOT / f'results/registered_{pl["prepare"]["run"]}'
        for name in ('scene.json', 'robot_model.npz', 'queries.npz'):
            shutil.copyfile(prep / name, d / name)
        row = dict(dx=pl['dx'], half_spacing=pl['s'], base_height=pl['h'], prepare_run=pl['prepare']['run'], fields=[])
        for fr in pl['fields']:
            src = ROOT / f'results/registered_{fr["run"]}/training/checkpoint.pt'
            out = d / f'{fr["label"]}_s{fr["seed"]}.pt'
            ck = torch.load(src, map_location='cpu', weights_only=False)
            torch.save({k: ck[k] for k in ('epoch', 'model_state_dict', 'B_state_dict', 'embed_config')}, out)
            bi = {q['steps']: q['success'] for q in fr.get('planners', []) if q['kind'] == 'bidir' and q['samples'] == 50}
            row['fields'].append(dict(file=str(out.relative_to(HERE)), field=fr['label'], seed=fr['seed'], run=fr['run'], args=fr['args'], solved_forward=fr['success'],
                                      solved_two_ended_600=bi.get(600), solved_two_ended_2400=bi.get(2400)))
            if a.best == pl['name']:
                (M / 'best').mkdir(exist_ok=True)
                shutil.copyfile(src, M / 'best' / f'{fr["label"]}_s{fr["seed"]}_full_checkpoint.pt')
        if a.best == pl['name']:
            for name in ('scene.json', 'robot_model.npz', 'queries.npz'):
                shutil.copyfile(prep / name, M / 'best' / name)
            shutil.copyfile(HERE / 'scenes' / (pl['name'] + '.yml'), M / 'best' / (pl['name'] + '.yml'))
        index[pl['name']] = row
    (M / 'index.json').write_text(json.dumps(dict(best=a.best, placements=index), indent=1))
    if a.demo:
        shutil.copyfile(a.demo / 'results.json', HERE / 'report/demo.json')
        shutil.copyfile(a.demo / 'demo.npz', HERE / 'report/demo.npz')
    size = sum(f.stat().st_size for f in M.rglob('*') if f.is_file()) / 1e6
    print(f'{len(index)} placements, {sum(len(r["fields"]) for r in index.values())} models, {size:.0f} MB in {M}')


if __name__ == '__main__':
    main()
