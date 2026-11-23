"""Predeclare new validation geometries and forcing seeds before running them."""
import json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
rng=random.Random(271116)
ps=[[rng.uniform(.2,.8) for _ in range(3)] for _ in range(2)]
small=[dict(fam=f,n=511,chi=chi,p=p,rhs=rhs,rhs_seed=8713+pi*101)
       for pi,p in enumerate(ps) for f in ['circ','disk'] for chi in [1e3,1e5] for rhs in [3,4]]
small+=[dict(fam=f,n=511,chi=1e5,p=ps[1],rhs=4,rhs_seed=8927)
        for f in ['band','rot','poisson','jump','ellipse','two_vortices']]
medium=[dict(fam=f,n=1023,chi=chi,p=ps[0],rhs=rhs,rhs_seed=8713)
        for f in ['circ','disk'] for chi,rhs in [(1e3,4),(1e5,4),(1e5,3)]]
large=[dict(fam=f,n=2047,chi=1e5,p=ps[1],rhs=4,rhs_seed=8814) for f in ['circ','disk']]
out=dict(geometry_seed=271116,p=ps,cases=dict(validate511=small,validate1023=medium,validate2047=large),
         note='New geometries, independently crossed contrasts at 511, new smooth and random-forcing RHS. No validation solves have been inspected when this file is written.')
target=HERE/'VALIDATION_CASES.json'
if target.exists():assert json.loads(target.read_text())==out
else:target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(p=ps,counts={k:len(v) for k,v in out['cases'].items()}),indent=2))
