"""Predeclared topology stress tests; same locked method, no tuning on these cases."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

if '--worker' in sys.argv:
    import argparse
    import experiment as ex
    import numpy as np
    from ops_aniso import tensor,assemble
    original=ex.problem
    def topology_problem(fam,n,chi,p,rhs=0):
        if fam not in ('ellipse','two_vortices'): return original(fam,n,chi,p,rhs)
        t=np.linspace(0,1,n+2); X,Y=np.meshgrid(t,t)
        if fam=='ellipse':
            vx=-(Y-.47)/(.65**2); vy=(X-.53)/(1.25**2)
        else:
            # Tangent to streamfunction sin(2*pi*x) sin(pi*y): two closed-cell families.
            vx=-np.pi*np.sin(2*np.pi*X)*np.cos(np.pi*Y)
            vy=2*np.pi*np.cos(2*np.pi*X)*np.sin(np.pi*Y)
        theta=np.arctan2(vy,vx); one=np.ones_like(X)
        k11,k12,k22=tensor(theta,chi*one,one)
        x,y=X[1:-1,1:-1],Y[1:-1,1:-1]
        q=1+.2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
        A=assemble(k11,k12,k22,q,1/(n+1))
        u=np.sin(np.pi*x)*np.sin(np.pi*y)+.23*np.sin(3*np.pi*x)*np.sin(2*np.pi*y)+.17*np.sin(7*np.pi*x)*np.sin(5*np.pi*y)
        return A,A@u.ravel(),u.ravel()
    ex.problem=topology_problem
    d=dict(n=511,chi=1e5,p=[.5,.5,.5],rhs=1,cfg='base',method='amg',m=1,steps=32,div=4,width=64,
           bins=96,smooth=0,maxiter=1500,pilot=0,gate=.1,source='ritz',basis='weighted')
    d.update(json.loads(sys.argv[2]))
    try: print(json.dumps(ex.run(argparse.Namespace(**d))),flush=True)
    except Exception as e: print(json.dumps(dict(**d,ok=False,exception=repr(e))),flush=True)
else:
    lock=json.loads((HERE/'LOCK.json').read_text())
    jobs=[dict(fam=f,**spec,tag=tag,rep=r) for f in ['ellipse','two_vortices']
          for tag,spec in lock['methods'].items() for r in range(2)]
    random.Random(1116).shuffle(jobs)
    (HERE/'STRESS_LOCK.json').write_text(json.dumps(jobs,indent=2)+'\n')
    with open(HERE/'topology.jsonl','a') as out:
        for j,job in enumerate(jobs):
            tag=job['tag']; rep=job['rep']; cfg={k:v for k,v in job.items() if k not in ('tag','rep')}
            try:
                p=subprocess.run([sys.executable,__file__,'--worker',json.dumps(cfg)],capture_output=True,text=True,timeout=240)
                row=json.loads([s for s in p.stdout.splitlines() if s.startswith('{')][-1])
                row.update(stderr=p.stderr[-2000:],exitcode=p.returncode)
            except Exception as e: row=dict(**cfg,ok=False,exception=repr(e))
            row.update(tag=tag,rep=rep); out.write(json.dumps(row)+'\n'); out.flush()
            print(j+1,len(jobs),row['fam'],tag,row.get('iterations'),row.get('total_s'),row['ok'],flush=True)
