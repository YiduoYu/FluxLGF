"""Clean replacements for flagged timing records and extra short-control repeats."""
import json
import subprocess
import sys
from pathlib import Path
from batch import batch

HERE=Path(__file__).resolve().parent
keys={'fam','n','chi','p','rhs','cfg','method','m','steps','div','width','bins','smooth','maxiter',
      'source','basis','pilot','gate','tag','rep'}
rows=[json.loads(s) for s in (HERE/'validate1023.jsonl').read_text().splitlines()][:5]
jobs=[{k:v for k,v in r.items() if k in keys} for r in rows]
(HERE/'REMEASURE_LOCK.json').write_text(json.dumps(dict(reason='Small symmetry diagnostic overlapped early 1023 runs; conservatively replace first five timings, preserving originals.',jobs=jobs),indent=2)+'\n')
batch(jobs,'remeasure1023.jsonl',seed=1116,timeout=360)
lock=json.loads((HERE/'LOCK.json').read_text())
cases=[c for c in lock['cases']['validate511'] if c['fam'] in ['poisson','jump','rot']]
jobs=[]
for rep in [2,3,4]:
    for case in cases:
        for tag,spec in lock['methods'].items():
            j=dict(case,**spec,tag=tag,rep=rep)
            if j['method']=='lift': j['bins']=96
            jobs.append(j)
batch(jobs,'controls_repeats.jsonl',seed=1216)
subprocess.run([sys.executable,str(HERE/'stress.py')],check=True)
subprocess.run([sys.executable,str(HERE/'diagnose_scale.py')],check=True)
