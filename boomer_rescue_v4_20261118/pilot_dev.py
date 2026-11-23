from pilot import trial
from pathlib import Path
import json,random
h=Path(__file__).resolve().parent
jobs=[(f,v,c,p) for f in ['circ','disk','two_vortices','ellipse'] for v,c,p in [('old',2,0),('coarse0',2,0),('coarse1',2,1),('pilot_hats',2,0),('pilot_hats',2,1),('pilot_hats',2,3),('pilot_direct',8,1)]]
random.Random(2205).shuffle(jobs)
with (h/'pilot_dev.jsonl').open('x') as out:
    for f,v,c,p in jobs:
        row=trial((255,f,1e5,[.21,.73,.35],4,17119),v,count=c,polish=p)
        out.write(json.dumps(row)+'\n');out.flush()
        print(f,v,c,p,row['activated'],row['it'],round(row['total'],3),flush=True)
