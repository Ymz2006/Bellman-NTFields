"""Replay the certificates of the frozen cabinet test suites with the release scorer (tilt gate off).

Checks the manifest hashes, the suite fields (100 queries, joint-space distance 0.02-4.0 rad, exact
normalization, endpoints in their recorded regions, no repeated query) and that every stored witness
path starts and ends at its query and passes score_cross_suite.py in the scene. Runs in the pinned
environment; read-only on the suites.
"""
import argparse,json,runpy,sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE,sha
sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import scene_boxes
import numpy as np


def within(pos,box):return bool(((pos>=box[:,0]-1e-6)&(pos<=box[:,1]+1e-6)).all())


def main():
    p=argparse.ArgumentParser();p.add_argument('--tests',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.tests/'suite_manifest.json').read_text());assert manifest['scene_sha256']==sha(a.scene)
    box,cupboard,compartments=scene_boxes(a.scene)
    norm=np.load(RELEASE/'configs/joint_normalization.npz');center,scale=np.asarray(norm['q_center'],np.float32),np.asarray(norm['q_range'],np.float32)
    seen=set();report=dict(scene=manifest['scene'],suites={})
    for item in manifest['suites']:
        path=a.tests/'suites'/item['file'];assert sha(path)==item['sha256'],item['file']
        z=np.load(path,allow_pickle=True);s,g=z['starts_rad'],z['goals_rad'];n=len(s)
        assert n==item['n']==100 and np.array_equal((s-center)/scale,z['starts_norm']) and np.array_equal((g-center)/scale,z['goals_norm'])
        d=np.linalg.norm(g-s,axis=1);assert (d>=.02).all() and (d<=4.).all()
        for k,(xyz,regions) in enumerate([(np.stack([z['ee_x_start'],z['ee_y_start'],z['ee_z_start']],1),z['start_region']),
                                          (np.stack([z['ee_x_goal'],z['ee_y_goal'],z['ee_z_goal']],1),z['goal_region'])]):
            for pos,region in zip(xyz,regions):
                assert within(pos,box),'endpoint outside the sampling box'
                if region in compartments:assert within(pos,compartments[region]),f'not in compartment {region}'
                elif region=='free':assert not within(pos,cupboard),'free endpoint inside the cupboard'
        for q in np.concatenate([s,g],1):
            key=q.tobytes();assert key not in seen,'repeated query';seen.add(key)
        w=z['witness_trajectories'];assert all(np.array_equal(w[i][0],s[i]) and np.array_equal(w[i][-1],g[i]) for i in range(n))
        traj=a.output/(item['task']+'.witness.npz');arr=np.empty(n,dtype=object)
        for i in range(n):arr[i]=w[i]
        np.savez(traj,trajectories=arr,query_ids=np.arange(n));out=a.output/(item['task']+'.score.json')
        sys.argv=['/workspace/score_cross_suite.py','--suite',str(path),'--scene',str(a.scene),'--norm_file',str(RELEASE/'configs/joint_normalization.npz'),
                  '--trajectories',str(traj),'--labels',item['task'],'--json_out',str(out),'--tilt_deg','180']
        runpy.run_path('/workspace/score_cross_suite.py',run_name='__main__')
        row=json.loads(out.read_text())[item['task']]
        report['suites'][item['task']]=dict(n=n,witnesses_accepted=row['success'],direct_line=int(z['direct_line_strict_feasible'].sum()),
            cspace_l2_median=float(np.median(d)),witness_cspace_length_median=float(np.median(z['witness_cspace_length'])),
            start_regions={str(k):int(v) for k,v in zip(*np.unique(z['start_region'],return_counts=True))},
            goal_regions={str(k):int(v) for k,v in zip(*np.unique(z['goal_region'],return_counts=True))})
        assert row['success']==n,f"{item['task']}: {row['success']}/{n} witnesses accepted"
    report['all_accepted']=True;(a.output/'verify.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=1))


if __name__=='__main__':main()
