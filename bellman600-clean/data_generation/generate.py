"""Original FK pose sampling, physical labels and stratified coverage augmentation."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/clean/method')
from common import ROOT,RELEASE,write,sha
from labels import legacy_components,angles_from_signed_distance,tilt_speed,task_speeds


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
 p.add_argument('--scene',type=Path,default=RELEASE/'configs/scene.yml',help='Scene YAML for collision filtering and labels (default: frozen no-wall scene)');a=p.parse_args()
 out=a.output;out.mkdir(exist_ok=False);(out/'pools').mkdir();started=time.monotonic();timing={};gen=ROOT/'data_generation/source';scene=a.scene
 def step(name,script,args):
  command=[sys.executable,'-u',str(script),*map(str,args)];write(out/(name+'.command.json'),command)
  begin=time.monotonic()
  with (out/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  timing[name]=time.monotonic()-begin;print('SAMPLING_STAGE',name,timing[name],flush=True)
  write(out/'progress.json',dict(complete=False,timing=timing))
 pool_size=4096 if a.smoke else 2000000;pairs=4096 if a.smoke else 1000000
 common=['--n_points',pool_size,'--height','.70','--scene',scene,'--device','cuda:0','--seed','42']
 step('cup_pool',gen/'sample_fk_ur5e.py',common+['--task','cupping','--max_tilt_deg','20','--output',out/'pools/cup20.pkl'])
 step('pour_pool',gen/'sample_fk_ur5e.py',common+['--task','pouring','--x_tilt_max_deg','20','--max_pour_deg','20','--output',out/'pools/pour20.pkl'])
 original=out/'original_raw'
 step('original_pairs',gen/'pkl_to_training_data_multi_ur5e.py',[
  '--pkl_cupping',out/'pools/cup20.pkl','--pkl_pouring',out/'pools/pour20.pkl','--scene',scene,
  '--zone_mode','corner_table','--y_divide','0.0','--overlap','.10','--n_pairs',pairs,'--intra_ratio','.8',
  '--margin','.05','--offset','.005','--alpha','.65','--blend_mode','min','--self_margin','.02','--self_speed_margin','.1',
  '--source_aware','--signed_orient','--tilt_band','15','--cup_tilt_deg','15','--pour_tilt_deg','15','--corner_overlap_deg','45',
  '--seed','42','--device','cuda:0','--corner_cross_frac','.20','--out_dir',original])
 # Normalization is part of the fixed problem, not learned from evaluation data.
 expected=np.load(RELEASE/'configs/joint_normalization.npz');generated=np.load(original/'joint_normalization.npz')
 for key in ['q_min','q_max','q_center','q_range']:assert np.array_equal(expected[key],generated[key]),key
 config=json.loads((original/'pipeline_config.json').read_text())
 assert config['signed_orient'] and config['blend_mode']=='min'
 o=np.load(original/'d_obstacle.npy',mmap_mode='r');ss=np.load(original/'d_self.npy',mmap_mode='r')
 dc=np.load(original/'d_orient_cupping.npy',mmap_mode='r');dp=np.load(original/'d_orient_pouring.npy',mmap_mode='r')
 comp=legacy_components(o,ss,dc,dp,config['margin'],config['offset'],config['self_speed_scale'])
 for i,(task,dist) in enumerate([('cup',dc),('pour',dp)]):
  comp[...,2+i]=tilt_speed(angles_from_signed_distance(dist,config['margin'],config[task+'_tilt_deg'],config['tilt_band']),20,10,.1)
 xyz=np.load(original/'ee_positions.npy',mmap_mode='r');speeds=task_speeds(comp,xyz[:,[1,4]])[0]
 inputs=out/'inputs';(inputs/'original').mkdir(parents=True);shutil.copy2(original/'sampled_points.npy',inputs/'original/sampled_points.npy');np.save(inputs/'original/task_speed_physical.npy',speeds)
 coverage=out/'coverage'
 step('coverage',ROOT/'data_generation/prepare_coverage.py',['--source',original,'--output',coverage,'--scene',scene,
  '--per-bin','128' if a.smoke else '133334','--validation-per-bin','16' if a.smoke else '13334',
  '--pairs','4096' if a.smoke else '1000005','--generation-budget','1200'])
 (inputs/'augmentation').mkdir()
 for name in ['sampled_points.npy','task_speed_physical.npy']:shutil.copy2(coverage/'train'/name,inputs/'augmentation'/name)
 files={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*.npy')}
 write(inputs/'source_manifest.json',dict(files=files,source='newly sampled with historical FK and coverage recipes',evaluation_rows_read=False,smoke=a.smoke))
 write(out/'complete.json',dict(complete=True,smoke=a.smoke,scene=str(scene),scene_sha256=sha(scene),source_seeds=dict(fk=42,pair=42,coverage=20260912),
  pool_size_each=pool_size,original_pairs=pairs,augmentation_pairs=4096 if a.smoke else 1000005,
  timing=timing,elapsed_seconds=time.monotonic()-started,evaluation_rows_read=False,training_input_hashes=files))
if __name__=='__main__':main()
