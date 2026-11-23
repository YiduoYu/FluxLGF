"""Sequential isolated measurements; raw results retained, including failures/timeouts."""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PY='/opt/anaconda3/envs/hypre_env/bin/python'

def batch(jobs,out,seed=61,timeout=240):
    random.Random(seed).shuffle(jobs)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMPI_MCA_btl='self')
    with open(HERE/out,'a') as f:
        for i,job in enumerate(jobs):
            cmd=[PY,str(HERE/'experiment.py')]
            for k,v in job.items():
                if k in ('tag','rep'): continue
                cmd+=['--'+k]+([str(x) for x in v] if isinstance(v,list) else [str(v)])
            t=time.perf_counter()
            try:
                p=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=timeout)
                lines=[x for x in p.stdout.splitlines() if x.startswith('{')]
                row=json.loads(lines[-1]) if lines else dict(ok=False,stdout=p.stdout,stderr=p.stderr)
                row.update(exitcode=p.returncode,stderr=p.stderr[-2000:])
            except subprocess.TimeoutExpired:
                row=dict(ok=False,exception='timeout',timeout_s=timeout)
            row.update(job,elapsed_process_s=time.perf_counter()-t)
            f.write(json.dumps(row)+'\n'); f.flush()
            print(f'{i+1}/{len(jobs)} {job} ok={row["ok"]} it={row.get("iterations")} total={row.get("total_s",0):.3f}',flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='baseline':
        jobs=[dict(fam=fam,n=255,cfg=cfg) for fam in ['circ','disk','rot']
              for cfg in ['base','theta50','theta70','agg','p8','fal','schwarz','schwarz2','cheb']]
        batch(jobs,'baseline_screen.jsonl')
    elif stage=='prototype':
        specs=[dict(method='amg')]
        specs += [dict(method='linear',m=m) for m in (4,16,24)]
        specs += [dict(method='lift',m=m,bins=b,smooth=s) for m,b,s in [(1,24,0),(1,48,0),(1,96,0),(1,48,2),(2,48,0),(2,48,2)]]
        specs += [dict(method='local',m=m,width=w,smooth=s) for m,w,s in [(2,32,0),(2,32,2),(4,64,0),(4,64,2),(4,32,2)]]
        jobs=[dict(fam=fam,n=255,**spec) for fam in ['circ','disk'] for spec in specs]
        batch(jobs,'prototype_screen.jsonl')
    elif stage=='scale_dev':
        specs=[dict(method='amg',cfg=c) for c in ['base','fal','p8','cheb','theta50']]
        specs += [dict(method='lift',m=m,bins=b,smooth=s,steps=k,div=d)
                  for m,b,s,k,d in [(1,48,0,32,4),(1,96,0,32,4),(1,192,0,32,4),
                                   (1,96,2,32,4),(2,96,0,32,4),(1,96,0,64,4),
                                   (1,96,0,32,8)]]
        specs += [dict(method='linear',m=24),dict(method='local',m=2,width=32,smooth=0)]
        batch([dict(fam=f,n=511,**s) for f in ['circ','disk'] for s in specs], 'scale_dev.jsonl')
    elif stage=='ablate_dev':
        specs=[dict(method='lift',m=m,bins=b,basis='weighted') for m,b in [(1,48),(1,96),(1,192),(2,96)]]
        specs += [dict(method='lift',m=1,bins=96,source='operator',basis=k) for k in ['hats','weighted']]
        specs += [dict(method='lift',m=1,bins=96,cfg='p8'),dict(method='amg',cfg='p8')]
        batch([dict(fam=f,n=511,**s) for f in ['circ','disk'] for s in specs], 'ablate_dev.jsonl')
    elif stage=='gate_dev':
        specs=[dict(method='amg'),dict(method='lift',basis='weighted',m=1,bins=96,pilot=12)]
        batch([dict(fam=f,n=511,**s) for f in ['circ','disk','rot','poisson','jump'] for s in specs], 'gate_dev.jsonl')
    elif stage.startswith('validate'):
        lock=json.loads((HERE/'LOCK.json').read_text())
        cases=lock['cases'][stage]
        jobs=[]
        for rep in range(lock['repeats'][stage]):
            for case in cases:
                for tag,spec in lock['methods'].items():
                    job=dict(case,**spec,tag=tag,rep=rep)
                    if job['method']=='lift': job['bins']=max(48,3*(job['n']+1)//16)
                    jobs.append(job)
        batch(jobs,stage+'.jsonl',seed=lock['order_seed'],timeout=360)
