"""Run clean baseline commands in the pinned, existing GPU environment.

Backends (BELLMAN600_BACKEND): `docker` (default) starts the pinned image through the host Docker daemon;
`host` runs the same command with the same mounts and variables in a bubblewrap user namespace on the host
(environment/local/README.md: Python from BELLMAN600_HOST_PYTHON, the rebuilt venv). The host backend is a
modified environment; every run records environment_override.json and host.command.json.
Both backends cap RAM (BELLMAN600_DOCKER_MEMORY / BELLMAN600_HOST_MEMORY, default 20G).
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
RELEASE=HERE/'runtime'
IMAGE='sha256:e7749bdb59fb8ef51a462ab56c5febfd55aec43ec199eb99cec1e013e1dc0828'
ACTIONS=['train','evaluate','reproduce','wall-audit','wall-evaluate','wall-reproduce','rebuild-reproduce','from-source','from-source-prepare','transition-wall','obstacle-scene','obstacle-tests','obstacle-evaluate','speed-test','cabinet-tests','cabinet','cabinet-metric','cabinet-evaluate','face-speed','face-speed-evaluate','face-speed-accuracy','face-speed-guard','point-field','point-field-bellman','ntrl-cabinet','pntfields-cabinet']
VARIABLES=['PYTHONUTF8=1','PYTHONDONTWRITEBYTECODE=1','MPLCONFIGDIR=/outputs/.cache/matplotlib','XDG_CACHE_HOME=/outputs/.cache','CUDA_CACHE_PATH=/outputs/.cache/cuda','TORCH_EXTENSIONS_DIR=/outputs/.cache/torch_extensions','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','SELECTION_MODE=baseline','NTF_CACHE_GOAL_FK=1','NTF_GRAPH_COST=1','PYTHONPATH=/clean/method:/repo/method:/repo/vendor/ntrl:/workspace:/workspace/models:/workspace/train']
MOUNTS=[(HERE,'/clean'),(RELEASE,'/repo'),(RELEASE/'vendor/planner','/workspace'),(RELEASE/'vendor/ntrl/models','/workspace/models'),(RELEASE/'vendor/ntrl/train','/workspace/train'),(RELEASE/'assets/runtime/robot','/opt/ur5e_description')]
HOST_PYTHON=os.environ.get('BELLMAN600_HOST_PYTHON',str(Path.home()/'.venvs/bellman-ntfield/bin/python'))


def host_command(tail,output,gpus,memory):
 """bubblewrap argv reproducing the container layout: /clean, /repo, /workspace, /opt/ur5e_description read-only, /outputs writable."""
 python=Path(HOST_PYTHON).absolute()   # not resolve(): the venv's python symlink must stay inside the venv
 cmd=['bwrap','--unshare-pid','--die-with-parent','--clearenv','--chdir','/workspace','--tmpfs','/tmp','--proc','/proc','--dev-bind','/dev','/dev','--ro-bind','/sys','/sys']
 for d in ['/usr','/lib','/lib64','/bin','/sbin','/etc']:
  if Path(d).exists():cmd+=['--ro-bind',d,d]
 venv=python.parents[1];cmd+=['--ro-bind',str(venv),str(venv)]
 for src,dst in MOUNTS:cmd+=['--ro-bind',str(src),dst]
 cmd+=['--bind',str(output.resolve()),'/outputs']
 env=dict(v.split('=',1) for v in VARIABLES);env.update(HOME='/outputs/.cache/home',PATH=f'{venv}/bin:/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8',CUDA_DEVICE_ORDER='PCI_BUS_ID',BELLMAN600_BACKEND='host')
 if gpus!='all':
  assert gpus.startswith('device='),gpus;env['CUDA_VISIBLE_DEVICES']=gpus[len('device='):]
 for k,v in env.items():cmd+=['--setenv',k,v]
 cmd+=[str(python),'-u',*tail]
 if memory and shutil.which('systemd-run'):cmd=['systemd-run','--user','--scope','-q','-p',f'MemoryMax={memory}',*cmd]
 return cmd


def host_probe(output,gpus):
 """Versions of the host environment, recorded as the modified-environment evidence."""
 probe=("import json,platform,sys,torch\n"
        "row=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),"
        "gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],platform=platform.platform())\n"
        "try:\n import curobo;row['curobo']=curobo.__version__\nexcept Exception as e:row['curobo']='unavailable: '+str(e)[:80]\n"
        "try:\n import numpy;row['numpy']=numpy.__version__\nexcept Exception:pass\n"
        "print(json.dumps(row))")
 out=subprocess.run(host_command(['-c',probe],output,gpus,None),capture_output=True,text=True)
 try:return json.loads(out.stdout.strip().splitlines()[-1])
 except Exception:return dict(error=out.stderr[-500:])


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=ACTIONS);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
 p.add_argument('--scene',help='transition-wall, obstacle-scene, cabinet* or face-speed* scene name in experiments/{transition_wall,obstacle_scenes,cabinet}/scenes (cabinet*: default cabinet)');p.add_argument('--gpus',default='all',help="Docker --gpus value, e.g. all or device=1")
 p.add_argument('--run',help='obstacle-evaluate, speed-test, cabinet-evaluate, face-speed-evaluate: results/registered_<id> directory name')
 p.add_argument('--suite',choices=['scene','frozen600'],default='scene',help='obstacle-evaluate: scene test suites or the frozen 600 queries')
 p.add_argument('--method',choices=['bellman','metric'],help='face-speed: field trained with the per-piece speed');p.add_argument('--cover',choices=['faces','spheres','links','split','environment','release'],help='face-speed: pieces of the speed')
 p.add_argument('--speed-model',choices=['shared','separate'],default='shared',help='face-speed bellman: one trunk with an output per piece, or one network per piece')
 p.add_argument('--data',help='face-speed metric: results/registered_<id>/data to reuse');p.add_argument('--sampled',help='face-speed bellman: results/registered_<id>/sampled to reuse')
 p.add_argument('--speed-steps',type=int,default=20000,help='face-speed bellman: speed network steps');p.add_argument('--runs',nargs='*',help='face-speed-accuracy: speed networks (results/registered_<id>) to test')
 p.add_argument('--filter',choices=['cover','release'],default='cover',help='face-speed metric: clearance of the x0 near-obstacle filter');p.add_argument('--rows',choices=['pieces','body','none'],default='pieces',help='face-speed-guard: one constraint per piece, one for the body speed, or no filter')
 p.add_argument('--margin',type=float,help='face-speed metric: speed label margin (m)');p.add_argument('--offset',type=float,help='face-speed metric: speed label offset (m)');p.add_argument('--pairs',type=int,help='face-speed metric: training pairs')
 p.add_argument('--alpha',type=float,default=.5,help='face-speed-guard: barrier rate');p.add_argument('--floor',type=float,default=.15,help='face-speed-guard: speed floor of the filter')
 p.add_argument('--speed',choices=['learned','exact'],help='point-field-bellman: learned point field or exact distance');p.add_argument('--reference',help='point-field-bellman: baseline run to match')
 p.add_argument('--self-net',help='point-field: split run whose self network is used by the learned-self controls')
 p.add_argument('--ntrl-args',default='',help='ntrl-cabinet: options for experiments/ntrl_cabinet/pipeline.py')
 p.add_argument('--pnt-args',default='',help='pntfields-cabinet: options for experiments/pntfields_cabinet/pipeline.py')
 a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 if a.action in ('from-source','from-source-prepare'):tail=['/clean/experiments/from_source/pipeline.py','--output','/outputs']+(['--prepare-only'] if a.action=='from-source-prepare' else [])
 elif a.action=='transition-wall':tail=['/clean/experiments/transition_wall/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/transition_wall/scenes/{a.scene}.yml']
 elif a.action=='obstacle-scene':tail=['/clean/experiments/obstacle_scenes/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml']
 elif a.action=='obstacle-tests':tail=['/clean/experiments/obstacle_scenes/make_tests.py','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--output','/outputs/tests']
 elif a.action=='obstacle-evaluate' and a.suite=='frozen600':tail=['/clean/experiments/transition_wall/evaluate.py','--checkpoint',f'/clean/results/{a.run}/training/checkpoint.pt','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--endpoint-audit',f'/clean/results/{a.run}/audit/endpoint_audit.json','--output','/outputs/evaluation']
 elif a.action=='obstacle-evaluate':tail=['/clean/experiments/obstacle_scenes/evaluate_tests.py','--checkpoint',f'/clean/results/{a.run}/training/checkpoint.pt','--scene',f'/clean/experiments/obstacle_scenes/scenes/{a.scene}.yml','--tests',f'/clean/experiments/obstacle_scenes/tests/{a.scene}','--output','/outputs/evaluation']
 elif a.action=='ntrl-cabinet':
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene;import shlex
  tail=['/clean/experiments/ntrl_cabinet/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml','--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):],
   *shlex.split(a.ntrl_args)]+(['--data',f'/clean/results/{a.data}/data'] if a.data else [])
 elif a.action=='pntfields-cabinet':
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene;import shlex
  tail=['/clean/experiments/pntfields_cabinet/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml','--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):],
   *shlex.split(a.pnt_args)]+(['--data',f'/clean/results/{a.data}/data'] if a.data else [])
 elif a.action=='point-field':
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene
  tail=['/clean/experiments/point_field/pipeline.py','--output','/outputs','--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml','--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):],
   '--sampled',f'/clean/results/{a.sampled}/sampled','--steps',str(a.speed_steps)]+(['--runs',*[f'/clean/results/{r}' for r in a.runs]] if a.runs else [])+(['--self-net',f'/clean/results/{a.self_net}'] if a.self_net else [])
 elif a.action=='point-field-bellman':
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene
  tail=['/clean/experiments/point_field/bellman_pipeline.py','--output','/outputs','--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml','--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):],
   '--sampled',f'/clean/results/{a.sampled}/sampled','--cover',a.speed]+(['--field',f'/clean/results/{a.run}/fields/pool/field.pt'] if a.speed=='learned' else [])+(['--reference',f'/clean/results/{a.reference}/prepared'] if a.reference and not a.smoke else [])
 elif a.action.startswith('cabinet') or a.action.startswith('face-speed'):
  # Scene cabinet<suffix>: experiments/cabinet/scenes/cabinet<suffix>.yml, frozen suites experiments/cabinet/tests<suffix>.
  scene=a.scene or 'cabinet';assert scene.startswith('cabinet'),scene;cabinet=['--scene',f'/clean/experiments/cabinet/scenes/{scene}.yml'];tests=['--tests','/clean/experiments/cabinet/tests'+scene[len('cabinet'):]]
  if a.action=='cabinet-tests':tail=['/clean/experiments/cabinet/make_tests.py',*cabinet,'--output','/outputs/tests']
  elif a.action in ('cabinet','cabinet-metric'):tail=['/clean/experiments/cabinet/pipeline.py','--output','/outputs',*cabinet,*tests,'--method','bellman' if a.action=='cabinet' else 'metric']
  elif a.action in ('cabinet-evaluate','face-speed-evaluate'):tail=['/clean/experiments/cabinet/evaluate.py','--run',f'/clean/results/{a.run}',*cabinet,*tests,'--output','/outputs/evaluation']
  elif a.action=='face-speed-accuracy':tail=['/clean/experiments/face_speed/speed_accuracy.py',*cabinet,*tests,'--output','/outputs/accuracy','--runs',*[f'/clean/results/{r}' for r in (a.runs or [])]]
  elif a.action=='face-speed-guard':tail=['/clean/experiments/face_speed/guard_evaluate.py','--run',f'/clean/results/{a.run}',*cabinet,*tests,'--output','/outputs/evaluation','--cover',a.cover or 'faces','--rows',a.rows,'--alpha',str(a.alpha),'--floor',str(a.floor)]
  else:
   tail=['/clean/experiments/face_speed/pipeline.py','--output','/outputs',*cabinet,*tests,'--method',a.method,'--cover',a.cover,'--speed-steps',str(a.speed_steps),'--filter',a.filter,'--speed-model',a.speed_model]+sum(([f'--{k}',str(getattr(a,k))] for k in ('margin','offset','pairs') if getattr(a,k) is not None),[])
   if a.data:tail+=['--data',f'/clean/results/{a.data}/data']
   if a.sampled:tail+=['--sampled',f'/clean/results/{a.sampled}/sampled']
 elif a.action=='speed-test':tail=['/clean/experiments/speed_test/test_speed.py','--run',a.run,'--output','/outputs/speed_test']
 elif a.action=='rebuild-reproduce':tail=['/clean/experiments/no_wall_rebuild/pipeline.py','--output','/outputs']
 elif a.action=='wall-reproduce':tail=['/clean/experiments/wall25_train/pipeline.py','--output','/outputs']
 elif a.action=='wall-audit':tail=['/clean/experiments/wall25/audit.py','--output','/outputs']
 elif a.action=='wall-evaluate':tail=['/clean/experiments/wall25/evaluate.py','--checkpoint','/clean/assets/checkpoint.pt','--output','/outputs/evaluation']
 elif a.action=='train':tail=['/clean/method/training.py','--output','/outputs/training']
 elif a.action=='evaluate':tail=['/clean/method/evaluate.py','--checkpoint','/clean/assets/checkpoint.pt','--output','/outputs/evaluation']
 else:tail=['/clean/tools/reproduce.py','--output','/outputs']
 if a.smoke:tail+=['--smoke']
 if os.environ.get('BELLMAN600_BACKEND','docker')=='host':
  (a.output/'.cache/home').mkdir(parents=True,exist_ok=True)
  memory=os.environ.get('BELLMAN600_HOST_MEMORY','20G')
  cmd=host_command(tail,a.output,a.gpus,memory)
  (a.output/'host.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
  override=dict(pinned_image_id=IMAGE,image_id_used=None,same_image_as_pinned=False,backend='host-bwrap',python=HOST_PYTHON,memory_cap=memory,gpus=a.gpus,
   versions=host_probe(a.output,a.gpus),written_by='launch.py (BELLMAN600_BACKEND=host)')
  (a.output/'environment_override.json').write_text(json.dumps(override,indent=2)+'\n')
  print('launch.py: HOST BACKEND (bwrap, not the pinned image):',json.dumps(override['versions']),flush=True)
  subprocess.run(cmd,check=True);return
 memory=os.environ.get('BELLMAN600_DOCKER_MEMORY','20G')
 cmd=['docker','--context','default','run','--rm','--memory',memory,'--memory-swap',memory,'--gpus',a.gpus,'--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-w','/workspace']
 for v in VARIABLES:cmd+=['-e',v]
 for src,dst in MOUNTS:cmd+=['-v',str(src)+':'+dst+':ro']
 cmd+=['-v',str(a.output.resolve())+':/outputs']
 cmd += [IMAGE,'python3','-u',*tail]
 (a.output/'docker.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
