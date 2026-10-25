"""Run clean baseline commands in the pinned, existing GPU environment."""
import argparse,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RELEASE=HERE/'runtime'
IMAGE='sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828'

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate','reproduce','wall-audit','wall-evaluate','wall-reproduce','rebuild-reproduce','from-source','from-source-prepare','transition-wall','obstacle-scene','obstacle-tests','obstacle-evaluate','speed-test','cabinet-tests','cabinet','cabinet-metric','cabinet-evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
 p.add_argument('--scene',help='transition-wall, obstacle-scene or cabinet* scene name in experiments/{transition_wall,obstacle_scenes,cabinet}/scenes (cabinet*: default cabinet)');p.add_argument('--gpus',default='all',help="Docker --gpus value, e.g. all or device=1")
 p.add_argument('--run',help='obstacle-evaluate, speed-test, cabinet-evaluate: results/registered_<id> directory name')
 p.add_argument('--suite',choices=['scene','frozen600'],default='scene',help='obstacle-evaluate: scene test suites or the frozen 600 queries');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 cmd=['docker','--context','default','run','--rm','--gpus',a.gpus,'--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-w','/workspace']
 for v in ['PYTHONUTF8=1','PYTHONDONTWRITEBYTECODE=1','MPLCONFIGDIR=/outputs/.cache/matplotlib','XDG_CACHE_HOME=/outputs/.cache','CUDA_CACHE_PATH=/outputs/.cache/cuda','TORCH_EXTENSIONS_DIR=/outputs/.cache/torch_extensions','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','SELECTION_MODE=baseline','NTF_CACHE_GOAL_FK=1','NTF_GRAPH_COST=1','PYTHONPATH=/clean/method:/repo/method:/repo/vendor/ntrl:/workspace:/workspace/models:/workspace/train']:
  cmd+=['-e',v]
 for src,dst in [(HERE,'/clean:ro'),(RELEASE,'/repo:ro'),(RELEASE/'vendor/planner','/workspace:ro'),(RELEASE/'vendor/ntrl/models','/workspace/models:ro'),(RELEASE/'vendor/ntrl/train','/workspace/train:ro'),(RELEASE/'assets/runtime/robot','/opt/ur5e_description:ro'),(a.output.resolve(),'/outputs')]:cmd+=['-v',str(src)+':'+dst]
 if a.action in ('from-source','from-source-prepare'):tail=['/clean/experiments/from_source/pipeline.py','--output','/outputs']+(['--prepare-only'] if a.action=='from-source-prepare' else [])
 elif a.action=='transition-wall':tail=['/clean/experiments/transition_wall/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/transition_wall/scenes/{a.scene}.yml']
 elif a.action=='obstacle-scene':tail=['/clean/experiments/obstacle_scenes/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml']
 elif a.action=='obstacle-tests':tail=['/clean/experiments/obstacle_scenes/make_tests.py','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--output','/outputs/tests']
 elif a.action=='obstacle-evaluate' and a.suite=='frozen600':tail=['/clean/experiments/transition_wall/evaluate.py','--checkpoint',f'/clean/results/{a.run}/training/checkpoint.pt','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--endpoint-audit',f'/clean/results/{a.run}/audit/endpoint_audit.json','--output','/outputs/evaluation']
 elif a.action=='obstacle-evaluate':tail=['/clean/experiments/obstacle_scenes/evaluate_tests.py','--checkpoint',f'/clean/results/{a.run}/training/checkpoint.pt','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--tests',f'/clean/experiments/obstacle_scenes/tests/{a.scene}','--output','/outputs/evaluation']
 elif a.action.startswith('cabinet'):
  # Scene cabinet<suffix>: experiments/cabinet/scenes/cabinet<suffix>.yml, frozen suites experiments/cabinet/tests<suffix>.
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene;cabinet=['--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml'];tests=['--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):]]
  if a.action=='cabinet-tests':tail=['/clean/experiments/cabinet/make_tests.py',*cabinet,'--output','/outputs/tests']
  elif a.action in ('cabinet','cabinet-metric'):tail=['/clean/experiments/cabinet/pipeline.py','--output','/outputs',*cabinet,*tests,'--method','bellman' if a.action=='cabinet' else 'metric']
  else:tail=['/clean/experiments/cabinet/evaluate.py','--run',f'/clean/results/{a.run}',*cabinet,*tests,'--output','/outputs/evaluation']
 elif a.action=='speed-test':tail=['/clean/experiments/speed_test/test_speed.py','--run',a.run,'--output','/outputs/speed_test']
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
