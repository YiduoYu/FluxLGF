from method import measure
from pathlib import Path
import json,random
h=Path(__file__).resolve().parent
cases=[(255,f,1e5,[.21,.73,.35],4,17119) for f in ['circ','disk','two_vortices','ellipse']]
variants=[('legacy',1,0),('uniform',4,0),('greedy',4,0),('uniform',4,1),('greedy',4,1),('legacy',1,1),('uniform',4,2),('uniform',2,1)]
jobs=[(c,v) for c in cases for v in variants]
random.Random(1811).shuffle(jobs)
with (h/'dev.jsonl').open('x') as out:
    for c,v in jobs:
        row=measure(c,*v)
        out.write(json.dumps(row)+'\n');out.flush()
        print(c[1],v,row['it'],round(row['total'],3),flush=True)
