"""Run clean baseline commands in the pinned, existing GPU environment."""
import argparse,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RELEASE=HERE/'runtime'
IMAGE='sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828'

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate','reproduce','wall-audit','wall-evaluate','wall-reproduce','rebuild-reproduce','from-source','from-source-prepare','transition-wall']);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
 p.add_argument('--scene',help='transition-wall scene name in experiments/transition_wall/scenes');p.add_argument('--gpus',default='all',help="Docker --gpus value, e.g. all or device=1");a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 cmd=['docker','--context','default','run','--rm','--gpus',a.gpus,'--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-w','/workspace']
 for v in ['PYTHONUTF8=1','PYTHONDONTWRITEBYTECODE=1','MPLCONFIGDIR=/outputs/.cache/matplotlib','XDG_CACHE_HOME=/outputs/.cache','CUDA_CACHE_PATH=/outputs/.cache/cuda','TORCH_EXTENSIONS_DIR=/outputs/.cache/torch_extensions','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','SELECTION_MODE=baseline','NTF_CACHE_GOAL_FK=1','NTF_GRAPH_COST=1','PYTHONPATH=/clean/method:/repo/method:/repo/vendor/ntrl:/workspace:/workspace/models:/workspace/train']:
  cmd+=['-e',v]
 for src,dst in [(HERE,'/clean:ro'),(RELEASE,'/repo:ro'),(RELEASE/'vendor/planner','/workspace:ro'),(RELEASE/'vendor/ntrl/models','/workspace/models:ro'),(RELEASE/'vendor/ntrl/train','/workspace/train:ro'),(RELEASE/'assets/runtime/robot','/opt/ur5e_description:ro'),(a.output.resolve(),'/outputs')]:cmd+=['-v',str(src)+':'+dst]
 if a.action in ('from-source','from-source-prepare'):tail=['/clean/experiments/from_source/pipeline.py','--output','/outputs']+(['--prepare-only'] if a.action=='from-source-prepare' else [])
 elif a.action=='transition-wall':tail=['/clean/experiments/transition_wall/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/transition_wall/scenes/{a.scene}.yml']
 elif a.action=='rebuild-reproduce':tail=['/clean/experiments/no_wall_rebuild/pipeline.py','--output','/outputs']
 elif a.action=='wall-reproduce':tail=['/clean/experiments/wall25_train/pipeline.py','--output','/outputs']
 elif a.action=='wall-audit':tail=['/clean/experiments/wall25/audit.py','--output','/outputs']
 elif a.action=='wall-evaluate':tail=['/clean/experiments/wall25/evaluate.py','--checkpoint','/clean/assets/checkpoint.pt','--output','/outputs/evaluation']
 elif a.action=='train':tail=['/clean/method/training.py','--output','/outputs/training']
 elif a.action=='evaluate':tail=['/clean/method/evaluate.py','--checkpoint','/clean/assets/checkpoint.pt','--output','/outputs/evaluation']
 else:tail=['/clean/tools/reproduce.py','--output','/outputs']
 if a.smoke:tail+=['--smoke']
 cmd += [IMAGE,'python3','-u',*tail]
 (a.output/'docker.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
