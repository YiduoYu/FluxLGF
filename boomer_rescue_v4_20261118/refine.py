from method import measure
from pathlib import Path
import json,random
h=Path(__file__).resolve().parent
jobs=[(f,m,sw,st) for f in ['disk','circ','two_vortices'] for m,sw,st in [('legacy',0,32),('legacy',1,32),('legacy',2,32),('legacy',1,16),('direct16',1,32)]]
random.Random(18113).shuffle(jobs)
with (h/'refine.jsonl').open('x') as out:
    for f,m,sw,st in jobs:
        r=measure((511,f,1e5,[.21,.73,.35],4,17119),m,1,sw,st)
        out.write(json.dumps(r)+'\n');out.flush()
        print(f,m,sw,st,r['it'],round(r['total'],3),flush=True)
