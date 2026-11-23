from shared import s,np,problem,probe
import json
from pathlib import Path
h=Path(__file__).resolve().parent
with (h/'gate_dev.jsonl').open('x') as out:
    for n in [255,511,1023]:
        for f,c in [('disk',1e5),('circ',1e5),('two_vortices',1e5),('ellipse',1e5),('disk',1e3),('circ',1e3),('circ',1e4),('band',1e5),('rot',1e5),('jump',1e5),('poisson',1)]:
            A,_,_=problem((n+1)//4-1,f,c,[.21,.73,.35],0,11)
            B=s.v1.Boomer(A,'p8')
            _,meta=probe(A,B,0)
            B.close()
            row=dict(n=n,f=f,chi=c,early=meta['early'])
            out.write(json.dumps(row)+'\n');out.flush()
            print(n,f,c,round(meta['early'],4),flush=True)
