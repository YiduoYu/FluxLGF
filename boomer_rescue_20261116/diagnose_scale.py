"""Post-validation exploratory mechanism check, NEVER merged into locked validation."""
import json
from pathlib import Path
from batch import batch

HERE=Path(__file__).resolve().parent
lock=json.loads((HERE/'LOCK.json').read_text())
jobs=[]
for rep in range(2):
    for case in lock['cases']['validate2047']:
        j=dict(case,**lock['methods']['rescue'],tag='rescue96_posthoc',rep=rep,bins=384)
        j['steps']=96
        jobs.append(j)
(HERE/'DIAGNOSTIC_LOCK.json').write_text(json.dumps(dict(
    purpose='Exploratory check after observing 2047 scaling deterioration. Change only Ritz budget 32 to 96; no inclusion in locked validation or headline speedup.',
    jobs=jobs),indent=2)+'\n')
batch(jobs,'diagnose_scale.jsonl',seed=1117,timeout=360)
