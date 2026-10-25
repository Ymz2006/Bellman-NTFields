"""Audit a completed standalone reproduction against frozen retraining evidence."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    status=read(run/'status.json');evaluation=read(run/'evaluation/evaluation.json')
    assert status['complete'] and not status['smoke'] and evaluation['complete']
    progress=read(run/'training/progress.json')
    assert progress['round']==5 and progress['optimizer_steps']==15620
    current_hash=hashlib.sha256((run/'training/checkpoint.pt').read_bytes()).hexdigest()
    reference_hash=hashlib.sha256((ROOT/'assets/retrained_checkpoint.pt').read_bytes()).hexdigest()
    comparison={}
    for name,pattern,count in [('mppi','bellman_mppi_*.score.json',1800),('rk4','bellman_rk4_*.score.json',600)]:
        n=0;differences=[]
        for path in sorted((run/'evaluation').glob(pattern)):
            label=path.name.removesuffix('.score.json')
            ours=read(path)[label]['per_query'];theirs=read(ROOT/'evidence/retraining/evaluation'/path.name)[label]['per_query']
            assert ours.keys()==theirs.keys()
            for q in ours:
                n+=1
                if ours[q]['success']!=theirs[q]['success']:differences.append(dict(label=label,query_id=q))
        assert n==count
        comparison[name]=dict(n=n,differences=differences)
    result=dict(complete=True,checkpoint_sha256=current_hash,reference_checkpoint_sha256=reference_hash,
        checkpoint_identical=current_hash==reference_hash,mppi_successes=evaluation['mppi_successes'],
        native_success=evaluation['native_success'],training=progress,paired_with_retraining=comparison)
    (run/'migration_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if current_hash!=reference_hash or any(v['differences'] for v in comparison.values()):
        raise SystemExit('Run completed but differs from historical retraining; inspect migration_validation.json.')
if __name__=='__main__':main()
