"""Train metric NTFields (ntrl-demo, unmodified) on the cabinet pairs of metric/data.py.

Runs ntrl_demo/models/metric/model_train_metric.Model(...).train() with that repository's own settings:
5,000 epochs of 5 minibatches of 2,000 pairs, AdamW (lr 5e-4, weight decay 0.5), speed stretch alpha
1.025, TD and normal-alignment weights 1e-3, the metric pair decoder, epoch rollback when the loss
ratio leaves (0, 1.2), checkpoints every 100 epochs. The final epoch is evaluated (no selection).

Environment shims only, the vendored code is unchanged: `pickle5` is the standard-library pickle
(the backport is for Python < 3.8), and `igl`, used only to write the progress-plot meshes, gets a
numpy OBJ writer. ntrl-demo sets no seed; seeds 87358 are fixed here.
"""
import argparse,json,os,pickle,random,shutil,sys,time,types,warnings
from pathlib import Path
import numpy as np
import torch

VENDOR=Path(__file__).resolve().parent/'ntrl_demo'


def write_triangle_mesh(path,v,f):
    with open(path,'w') as out:
        out.writelines(f'v {a} {b} {c}\n' for a,b,c in np.asarray(v,float))
        out.writelines(f'f {a+1} {b+1} {c+1}\n' for a,b,c in np.asarray(f,int))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data,out=a.data.resolve(),a.output.resolve();started=time.time()
    # Repeated every epoch by torch 2.x for unchanged ntrl-demo calls; silenced to keep the log readable.
    warnings.filterwarnings('ignore',message="input's size at dim=0 does not match num_features")
    warnings.filterwarnings('ignore',message='You are using `torch.load` with `weights_only=False`')
    sys.modules['pickle5']=pickle
    igl=types.ModuleType('igl');igl.write_triangle_mesh=write_triangle_mesh;sys.modules['igl']=igl
    os.chdir(VENDOR);sys.path.insert(0,str(VENDOR))
    random.seed(87358);np.random.seed(87358);torch.manual_seed(87358);torch.cuda.manual_seed_all(87358)
    from models.metric import model_train_metric as md
    assert Path(md.__file__).resolve().parent==VENDOR/'models/metric','ntrl-demo models package shadowed'
    model=md.Model(str(out/'model'),str(data)+'/',6,[0.0]*6,device='cuda:0')
    epochs=2 if a.smoke else model.Params['Training']['Number of Epochs']
    model.Params['Training']['Number of Epochs']=epochs
    (out/'model').mkdir()
    json.dump(dict(params=model.Params,data=str(data),epochs=epochs,seed=87358,smoke=a.smoke,vendor=str(VENDOR)),open(out/'config.json','w'),indent=2,default=str)
    model.train()
    folder=Path(model.folder);final=sorted(folder.glob(f'Model_Epoch_{str(epochs).zfill(5)}_*.pt'))
    assert len(final)==1,f'no final-epoch checkpoint in {folder}'
    shutil.copyfile(final[0],out/'checkpoint.pt')
    payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);assert payload['epoch']==epochs
    json.dump(dict(complete=True,epochs=epochs,final_checkpoint=str(final[0].relative_to(out)),final_loss=float(payload['train_loss'][-1]),
        elapsed_seconds=time.time()-started,smoke=a.smoke),open(out/'progress.json','w'),indent=2)
    print('METRIC_TRAINING_COMPLETE',time.time()-started,flush=True)

if __name__=='__main__':main()
