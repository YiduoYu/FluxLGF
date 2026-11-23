import json
from pathlib import Path
from bench import batch
HERE=Path(__file__).resolve().parent
lock=json.loads((HERE/'LOCK.json').read_text())
cases=[c for cc in lock['cases'].values() for c in cc if c['fam'] in ['circ','disk'] and c['chi']==1e3]
jobs=[dict(**c,**s,tag=tag,rep=2) for c in cases for tag,s in lock['methods'].items()]
target=HERE/'TIMING_EXTRA_JOBS.json'
if not target.exists():target.write_text(json.dumps(jobs,indent=2)+'\n')
batch(jobs,'controls_moderate_extra.jsonl',seed=191116)
