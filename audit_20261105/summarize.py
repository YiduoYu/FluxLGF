import json,csv,statistics,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
r=[json.loads(l) for l in (HERE/'evidence/amg_scan.jsonl').read_text().splitlines()]
assert len(r)==78, len(r)
groups=collections.defaultdict(list)
for x in r:
 assert 'total_s' in x,x
 groups[(x['m'],x['n'],x['coarsen'],x['theta'],x['agg'],x['pmax'])].append(x)
summary=[]
for key,v in sorted(groups.items()):
 summary.append(dict(m=key[0],n=key[1],coarsen=key[2],theta=key[3],agg=key[4],pmax=key[5],repeats=len(v),median_s=statistics.median(x['total_s'] for x in v),min_s=min(x['total_s'] for x in v),max_s=max(x['total_s'] for x in v),inner_iterations=[sum(h[0]for h in x['hist'])for x in v],all_success=all(x['success']for x in v)))
(HERE/'evidence/amg_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table=[]
for shape in [(32,1),(32,10),(64,1)]:
 rows=[x for x in summary if (x['m'],x['n'])==shape]
 baseline=next(x for x in rows if (x['coarsen'],x['theta'],x['agg'],x['pmax'])==('HMIS',.5,1,4))
 best=min((x for x in rows if x['coarsen']!='bt-fd'),key=lambda x:x['median_s'])
 bt=next(x for x in rows if x['coarsen']=='bt-fd')
 table.append(dict(m=shape[0],n=shape[1],baseline_s=baseline['median_s'],best_amg_s=best['median_s'],best_setting=f"{best['coarsen']}, theta={best['theta']}, agg={best['agg']}, Pmax={best['pmax']}",btfd_s=bt['median_s'],btfd_over_baseline=bt['median_s']/baseline['median_s'],btfd_over_best=bt['median_s']/best['median_s']))
(HERE/'evidence/amg_comparison.json').write_text(json.dumps(table,indent=2)+'\n')
print(json.dumps(table,indent=2));print('success',sum(x['success']for x in r),'of',len(r))
