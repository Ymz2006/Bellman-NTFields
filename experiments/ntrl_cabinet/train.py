"""Train metric NTFields with the user's ntrl-demo checkout (/clean/baselines/ntrl-demo), unchanged, on data.py pairs.

--variant metric      ntrl-demo models/metric (what experiments/cabinet/metric/train.py runs: B ~ N(0, 1),
                      speed stretch alpha 1.025, TD step 0.03 / s, output 0.2 x sum)
--variant metric_arm  ntrl-demo models/metric_arm, the package ntrl-demo's own train/train_arm.py trains the
                      UR5 arm with (B ~ 0.2 N(0, 1), alpha 1.0, TD step 0.02 / s, output 0.1 x sum)
--td-step / --normal-weight set the loss hyperparameters the paper gives for manipulators (arXiv 2505.05691,
Appendix C.2: TD step 0.005 and normal weight 2e-4 instead of the 3D values 0.02 / 1e-3 hard-coded in the demo):
the package's model_function_metric.py is loaded with exactly those constants substituted (file on disk unchanged).
--td-weight replaces the TD loss weight (1e-3 in the demo); --td-weight 0 --normal-weight 0 trains the bare eikonal
loss (with the causality weight exp(-0.5 T)) only.
--raw-speed trains on the paper's S* itself (Eq. 2-3): the demo trainer's stretch s^2 (2 - s)^2 (alpha s + 1 - alpha) is
removed by loading model_train_metric.py with that one line replaced. --weight-decay / --lr replace the demo's AdamW
weight decay 0.5 and fixed learning rate 5e-4 (optimizer parameters; not specified in the paper) the same way.
--metric-scale replaces the network's factor on its metric (metric_arm 0.1, metric 0.2; paper Eq. 10: none, i.e. 1):
with InstanceNorm latents this factor caps T and |grad T|, hence the slowest speed the field can represent.
Everything else is that package's Model.train(): AdamW lr 5e-4 weight decay 0.5, 5 minibatches of 2,000 per
epoch, epoch rollback, checkpoints every 100 epochs. --epochs overrides 'Number of Epochs' (default 5,000);
the final epoch is evaluated. Shims as in experiments/cabinet/metric/train.py: pickle5 -> pickle, igl ->
an OBJ writer for the progress plots. Seeds 87358.
"""
import argparse,hashlib,importlib.util,json,os,pickle,random,re,shutil,subprocess,sys,time,types,warnings
from pathlib import Path
import numpy as np
import torch

NTRL=Path('/clean/baselines/ntrl-demo')


def write_triangle_mesh(path,v,f):
    with open(path,'w') as out:
        out.writelines(f'v {a} {b} {c}\n' for a,b,c in np.asarray(v,float))
        out.writelines(f'f {a+1} {b+1} {c+1}\n' for a,b,c in np.asarray(f,int))
    return True


def provenance(variant):
    files=[f'models/{variant}/model_train_metric.py',f'models/{variant}/model_network_metric.py',f'models/{variant}/model_function_metric.py','models/data_mlp.py']
    head=(NTRL/'.git/HEAD').read_text().strip()
    commit=(NTRL/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') and (NTRL/'.git'/head[5:]).exists() else head
    return dict(path=str(NTRL),git_commit=commit,files={f:hashlib.sha256((NTRL/f).read_bytes()).hexdigest() for f in files})


def patch_function(variant,td_step,normal_weight,td_weight=None,log_eikonal=False):
    """Load models/<variant>/model_function_metric.py with the TD step and/or normal weight replaced."""
    path=NTRL/f'models/{variant}/model_function_metric.py';src=path.read_text();changes={}
    if td_step is not None:
        src,n=re.subn(r'length([01]) = \((0\.0[0-9]+)\)/\(Yobs',lambda m:f'length{m.group(1)} = ({td_step!r})/(Yobs',src);assert n==2,n;changes['td_step']=td_step
    if td_weight is not None:
        old='td_weight = 1e-3';assert src.count(old)==1;src=src.replace(old,f'td_weight = {td_weight!r}');changes['td_weight']=td_weight
    if normal_weight is not None:
        old='normal_weight = 1e-3';assert src.count(old)==1;src=src.replace(old,f'normal_weight = {normal_weight!r}');changes['normal_weight']=normal_weight
    if log_eikonal:   # the unbounded (log s'|grad T|)^2 of the original method instead of the bounded (sqrt(s'|grad T|) - 1)^2 (2026-11-01 representability test)
        for i in '01':
            old=f'loss{i} = loss_weight*(l{i}_2-1)**2';assert src.count(old)==1;src=src.replace(old,f'loss{i} = loss_weight*torch.log(l{i}+1e-8)**2')
        changes['log_eikonal']=True
    import models,importlib;package=importlib.import_module(f'models.{variant}')
    name=f'models.{variant}.model_function_metric';spec=importlib.util.spec_from_loader(name,loader=None,origin=str(path))
    module=importlib.util.module_from_spec(spec);module.__file__=str(path);exec(compile(src,str(path),'exec'),module.__dict__)
    sys.modules[name]=module;setattr(package,'model_function_metric',module)
    return dict(changes=changes,patched_source_sha256=hashlib.sha256(src.encode()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['metric','metric_arm'],default='metric');p.add_argument('--epochs',type=int,default=5000)
    p.add_argument('--seed',type=int,default=87358);p.add_argument('--td-step',type=float);p.add_argument('--raw-speed',action='store_true');p.add_argument('--metric-scale',type=float);p.add_argument('--weight-decay',type=float);p.add_argument('--lr',type=float);p.add_argument('--normal-weight',type=float);p.add_argument('--td-weight',type=float);p.add_argument('--log-eikonal',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data,out=a.data.resolve(),a.output.resolve();started=time.time()
    warnings.filterwarnings('ignore',message="input's size at dim=0 does not match num_features")
    warnings.filterwarnings('ignore',message='You are using `torch.load` with `weights_only=False`')
    sys.modules['pickle5']=pickle
    igl=types.ModuleType('igl');igl.write_triangle_mesh=write_triangle_mesh;sys.modules['igl']=igl
    os.chdir(NTRL);sys.path.insert(0,str(NTRL))
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    patch=patch_function(a.variant,a.td_step,a.normal_weight,a.td_weight,a.log_eikonal) if (a.td_step is not None or a.normal_weight is not None or a.td_weight is not None or a.log_eikonal) else None
    if a.metric_scale is not None:
        spec=importlib.util.spec_from_file_location('ntrl_cabinet_field',Path(__file__).resolve().parent/'field.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
        src=F.network_source(a.variant,a.metric_scale);path=NTRL/f'models/{a.variant}/model_network_metric.py';name=f'models.{a.variant}.model_network_metric'
        spec=importlib.util.spec_from_loader(name,loader=None,origin=str(path));module=importlib.util.module_from_spec(spec);module.__file__=str(path)
        exec(compile(src,str(path),'exec'),module.__dict__);sys.modules[name]=module
        import importlib as il;setattr(il.import_module(f'models.{a.variant}'),'model_network_metric',module)
        patch=dict(patch or {},metric_scale=a.metric_scale,patched_network_sha256=hashlib.sha256(src.encode()).hexdigest())
    subs=[]
    if a.raw_speed:subs+=[('speed = speed*speed*(2-speed)*(2-speed)','speed = speed  # paper Eq. 3: raw S*')]+([('alpha = 1.025#min','alpha = 1.0#min')] if a.variant=='metric' else [])
    if a.weight_decay is not None:subs+=[('weight_decay=0.5)',f'weight_decay={a.weight_decay!r})')]
    if a.lr is not None:subs+=[("self.optimizer.param_groups[0]['lr']  = 5e-4#",f"self.optimizer.param_groups[0]['lr']  = {a.lr!r}#")]
    if subs:
        path=NTRL/f'models/{a.variant}/model_train_metric.py';src=path.read_text()
        for old,new in subs:assert src.count(old)==1,old;src=src.replace(old,new)
        name=f'models.{a.variant}.model_train_metric';spec=importlib.util.spec_from_loader(name,loader=None,origin=str(path));module=importlib.util.module_from_spec(spec)
        module.__file__=str(path);exec(compile(src,str(path),'exec'),module.__dict__);sys.modules[name]=module
        import importlib as il;setattr(il.import_module(f'models.{a.variant}'),'model_train_metric',module)
        patch=dict(patch or {},raw_speed=a.raw_speed,weight_decay=a.weight_decay,lr=a.lr,train_substitutions=subs,patched_train_sha256=hashlib.sha256(src.encode()).hexdigest())
    if a.variant=='metric':from models.metric import model_train_metric as md
    else:from models.metric_arm import model_train_metric as md
    assert Path(md.__file__).resolve().parent==NTRL/'models'/a.variant,'ntrl-demo models package shadowed'
    if patch:assert md.model_function is sys.modules[f'models.{a.variant}.model_function_metric'],'patched loss not used'
    if a.metric_scale is not None:assert md.model_network is sys.modules[f'models.{a.variant}.model_network_metric'],'patched network not used'
    model=md.Model(str(out/'model'),str(data)+'/',6,[0.0]*6,device='cuda:0')
    epochs=2 if a.smoke else a.epochs;model.Params['Training']['Number of Epochs']=epochs
    (out/'model').mkdir()
    json.dump(dict(params=model.Params,data=str(data),epochs=epochs,seed=a.seed,smoke=a.smoke,variant=a.variant,ntrl_demo=provenance(a.variant),loss_patch=patch),
              open(out/'config.json','w'),indent=2,default=str)
    model.train()
    folder=Path(model.folder);final=sorted(folder.glob(f'Model_Epoch_{str(epochs).zfill(5)}_*.pt'))
    assert len(final)==1,f'no final-epoch checkpoint in {folder}'
    shutil.copyfile(final[0],out/'checkpoint.pt')
    payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);assert payload['epoch']==epochs
    # Intermediate checkpoints are kept only every 1,000 epochs (the plots and the final one stay).
    for c in folder.glob('Model_Epoch_*.pt'):
        e=int(c.name.split('_')[2])
        if e%1000 and e!=epochs:c.unlink()
    json.dump(dict(complete=True,epochs=epochs,variant=a.variant,final_checkpoint=str(final[0].relative_to(out)),final_loss=float(payload['train_loss'][-1]),
        elapsed_seconds=time.time()-started,smoke=a.smoke),open(out/'progress.json','w'),indent=2)
    print('METRIC_TRAINING_COMPLETE',time.time()-started,flush=True)

if __name__=='__main__':main()
