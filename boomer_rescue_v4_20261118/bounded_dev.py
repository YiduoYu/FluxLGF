from pilot import trial
import json,random
from pathlib import Path
h=Path(__file__).resolve().parent
jobs=[(f,v,p,k) for f in ['circ','disk','two_vortices'] for v,p,k in [('coarse0',0,0),('bounded',0,2),('bounded',0,6),('bounded',1,2),('jacobi',0,1),('jacobi',0,2)]]
random.Random(12960).shuffle(jobs)
with (h/'bounded_dev.jsonl').open('x') as out:
    for f,v,p,k in jobs:
        r=trial((511,f,1e5,[.21,.73,.35],4,17119),v,polish=p,basis_steps=k)
        out.write(json.dumps(r)+'\n');out.flush()
        print(f,v,p,k,r['it'],round(r['total'],3),r.get('row_cap'),flush=True)
