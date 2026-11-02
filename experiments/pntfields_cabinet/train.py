"""Train P-NTFields (baselines/P-NTFields, Ni & Qureshi, RSS 2023), unchanged, on data.py pairs of a cabinet scene.

Runs models/model_res_sigmoid.Model.train() as train/train_arm.py does: 10,000 epochs, AdamW lr 1e-3 (5e-4 after
epoch 8,000) weight decay 0.1, 6 minibatches of 10,000 per epoch, progressive speed alpha 0.5 -> 1.05, viscosity
gamma 0.001, epoch rollback at loss ratio 1.2, checkpoints and plots every 100 epochs. The final epoch is evaluated.
Differences from the released script (none touches the method):
  - Database: the original loader also reads a voxelized point cloud of the obstacle mesh that training never uses;
    this scene has no mesh, so the loader reads sampled_points.npy and speed.npy only (same _numpy2dataset).
  - pickle5 -> pickle (Python 3.8+), --epochs for smoke runs, seeds 87358 (the release script sets none).
  - Intermediate checkpoints are kept every 1,000 epochs (plus the final one).
"""
import argparse,hashlib,importlib.util,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pntfields_cabinet_field',HERE/'field.py');FIELD=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIELD)
PNT=FIELD.PNT


def provenance():
    files=['models/model_res_sigmoid.py','models/data_mlp.py','train/train_arm.py','test/arm_plan.py']
    head=(PNT/'.git/HEAD').read_text().strip()
    commit=(PNT/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') and (PNT/'.git'/head[5:]).exists() else head
    return dict(path=str(PNT),git_commit=commit,files={f:hashlib.sha256((PNT/f).read_bytes()).hexdigest() for f in files})


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10000);p.add_argument('--seed',type=int,default=87358);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);data,out=a.data.resolve(),a.output.resolve();started=time.time()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    md=FIELD.pnt_module()

    def database(path):
        points=np.load(f'{path}/sampled_points.npy');speed=np.load(f'{path}/speed.npy')
        print(points.shape,speed.shape,flush=True)
        return md.db._numpy2dataset(points,speed)
    md.db.Database=database
    (out/'model').mkdir()
    model=md.Model(str(out/'model'),str(data),6,device='cuda')
    epochs=2 if a.smoke else a.epochs;model.Params['Training']['Number of Epochs']=epochs
    json.dump(dict(params=model.Params,data=str(data),epochs=epochs,seed=a.seed,smoke=a.smoke,pntfields=provenance()),open(out/'config.json','w'),indent=2,default=str)
    model.train()
    folder=out/'model';final=sorted(folder.glob(f'Model_Epoch_{str(epochs).zfill(5)}_*.pt'))
    assert len(final)==1,f'no final-epoch checkpoint in {folder}'
    shutil.copyfile(final[0],out/'checkpoint.pt')
    payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);assert payload['epoch']==epochs
    for c in folder.glob('Model_Epoch_*.pt'):
        e=int(c.name.split('_')[2])
        if e%1000 and e!=epochs:c.unlink()
    json.dump(dict(complete=True,epochs=epochs,final_checkpoint=str(final[0].relative_to(out)),final_loss=float(payload['train_loss'][-1]),
        elapsed_seconds=time.time()-started,smoke=a.smoke),open(out/'progress.json','w'),indent=2)
    print('PNTFIELDS_TRAINING_COMPLETE',time.time()-started,flush=True)

if __name__=='__main__':main()
