"""Check every secondary job and the locked gates, print comparative summaries."""
import json,hashlib
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
lock=json.loads((root/'cluster_lock.json').read_text())
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==h for k,h in lock['hashes'].items())
rows=list(map(json.loads,(root/'cluster.jsonl').read_text().splitlines()))
assert [[r['index'],r['rep'],r['method']] for r in rows]==lock['jobs']
assert len(rows)==40 and all(r['ok'] and r['residual']<=1e-8 for r in rows)
cases=[]
for i,case in enumerate(lock['cases']):
    stats={}
    for m in ['first2','random','all','early','anchor']:
        group=[r for r in rows if r['index']==i and r['method']==m]
        assert len(group)==2 and group[0]['it']==group[1]['it']
        stats[m]=dict(it=group[0]['it'],seconds=median(r['total'] for r in group),
                      dim=group[0]['dim'],max_nnz=group[0]['row_cap'])
    anchor=next(r for r in rows if r['index']==i and r['method']=='anchor')
    cases.append(dict(case=case,stats=stats,cluster=anchor['cluster'],condition=anchor['condition']))
ratios={m:[c['stats']['anchor']['seconds']/c['stats'][m]['seconds'] for c in cases]
        for m in ['first2','random','all','early']}
ratios={m:dict(median=median(vals),wins=sum(v<1 for v in vals),individual=vals) for m,vals in ratios.items()}
wins=sum(c['stats']['anchor']['it']<c['stats']['first2']['it'] for c in cases)
print(json.dumps(dict(cases=cases,ratios=ratios,iterations_wins=wins,
                      passed=wins>=3 and ratios['first2']['median']<.9,
                      max_residual=max(r['residual'] for r in rows)),indent=2))
