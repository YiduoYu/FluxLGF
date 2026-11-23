"""Sequential subprocess runner; JSONL logs are append-only and preserve failures."""
import json,random,subprocess,sys,time,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/opt/anaconda3/envs/hypre_env/bin/python'
def batch(jobs,out,seed=1612,timeout=360):
    jobs=list(jobs);random.Random(seed).shuffle(jobs)
    with (HERE/out).open('a') as f:
        for i,job in enumerate(jobs):
            spec={k:v for k,v in job.items() if k not in ['tag','rep']}
            t=time.perf_counter()
            try:
                proc=subprocess.run([PY,str(HERE/'solver.py'),json.dumps(spec)],capture_output=True,text=True,timeout=timeout)
                row=json.loads([s for s in proc.stdout.splitlines() if s.startswith('{')][-1])
                row.update(exitcode=proc.returncode,stderr=proc.stderr[-2000:])
            except Exception as e:row=dict(**spec,ok=False,exception=repr(e))
            row.update(job,process_s=time.perf_counter()-t)
            f.write(json.dumps(row)+'\n');f.flush()
            print(i+1,len(jobs),job.get('tag'),job.get('fam'),job.get('n'),job.get('chi'),
                  'ok',row['ok'],'it',row.get('iterations'),'s',round(row.get('total_s',0),3),
                  'Ritz',row.get('ritz_steps'),'trigger',row.get('triggered'),flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='profile':
        cases=[dict(fam=f,n=511,chi=c,p=[.5,.5,.5],rhs=1) for f in ['circ','disk','two_vortices','ellipse','band','rot'] for c in [1e3,1e5]]
        jobs=[dict(**c,method='adaptive',gate='none',tag='adaptive_m1',rep=0) for c in cases]
        batch(jobs,'profile.jsonl')
    elif stage=='gates':
        cases=[dict(fam=f,n=511,chi=c,p=[.5,.5,.5],rhs=1) for f in ['circ','disk','two_vortices','ellipse','band','rot'] for c in [1e3,1e5]]
        jobs=[dict(**c,method='amg',tag='p8',rep=0) for c in cases]
        batch(jobs,'profile_baselines.jsonl')
    elif stage=='topology':
        specs=[dict(m=2,split=False,tag='two_coords'),dict(m=1,split=True,tag='split')]
        jobs=[dict(fam=f,n=511,chi=1e5,p=[.5,.5,.5],rhs=1,method='adaptive',gate='none',rep=0,**s)
              for f in ['circ','disk','two_vortices','ellipse'] for s in specs]
        batch(jobs,'topology_dev.jsonl')
    else:
        lock=json.loads((HERE/'LOCK.json').read_text())
        jobs=[dict(**c,**s,tag=tag,rep=rep) for c in lock['cases'][stage]
              for tag,s in lock['methods'].items() for rep in range(lock['repeats'][stage])]
        batch(jobs,stage+'.jsonl',seed=lock['order_seed'])
