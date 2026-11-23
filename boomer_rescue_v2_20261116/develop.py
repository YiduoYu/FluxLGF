"""More development on previously seen geometries; no independent validation here."""
import json
from pathlib import Path
from bench import batch
HERE=Path(__file__).resolve().parent
lock=json.loads((HERE.parent/'boomer_rescue_20261116/LOCK.json').read_text())
pa=lock['cases']['validate511'][0]['p'];pb=lock['cases']['validate2047'][0]['p']
cases=[dict(fam=f,n=511,chi=c,p=pa,rhs=r) for f in ['circ','disk','band','rot'] for c,r in [(1e3,1),(1e5,1),(1e3,3)]]
cases+=[dict(fam=f,n=1023,chi=1e5,p=pb,rhs=1) for f in ['circ','disk']]
specs=[dict(method='amg',tag='p8'),dict(method='adaptive',gate='both',gate_steps=25.,gate_lambda=.04,tag='gate_adaptive')]
jobs=[dict(**case,**s,rep=0) for case in cases for s in specs]
batch(jobs,'gate_dev.jsonl',seed=1613)
