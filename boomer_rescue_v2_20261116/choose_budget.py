"""Final development comparison; all geometries were used in round one."""
import json
from pathlib import Path
from bench import batch
HERE=Path(__file__).resolve().parent
lock=json.loads((HERE.parent/'boomer_rescue_20261116/LOCK.json').read_text())
p=lock['cases']['validate2047'][0]['p']
jobs=[]
for f in ['circ','disk','two_vortices']:
    for m,method in [(2,'adaptive'),(2,'fixed')]:
        jobs.append(dict(fam=f,n=1023,chi=1e5,p=p if f!='two_vortices' else [.5]*3,rhs=1,
                         m=m,method=method,gate='both',gate_steps=40.,tag=f'{method}_m{m}',rep=0))
jobs.append(dict(fam='two_vortices',n=1023,chi=1e5,p=[.5]*3,rhs=1,m=1,
                 method='adaptive',gate='both',gate_steps=40.,tag='adaptive_m1',rep=0))
batch(jobs,'budget_dev.jsonl',seed=1614)
