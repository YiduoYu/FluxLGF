"""Summarize independent problem cases, keeping failed jobs visible."""
from pathlib import Path
from statistics import median
import json
from validate import hashes
H=Path(__file__).resolve().parent
lock=json.loads((H/'LOCK.json').read_text())
assert hashes()==lock['hashes']
rows=[json.loads(x) for x in (H/'validation.jsonl').read_text().splitlines()]
identity=lambda r:(r['index'],r['variant'],r['rep'])
assert len(rows)==len(lock['jobs'])==198
assert set(map(identity,rows))==set(map(identity,lock['jobs']))
assert all(r['case']==j['case'] for r,j in zip(rows,lock['jobs']))
failures=[r for r in rows if not r['ok']]
if failures:
    print(json.dumps(failures,indent=2))
    raise RuntimeError('Review failed methods before ranking')
cases=[]
for i in range(9):
    matches=[r for r in rows if r['index']==i]
    variants={}
    for v in dict.fromkeys(r['variant'] for r in matches):
        group=[r for r in matches if r['variant']==v]
        assert len(group)==3 and max(r['residual'] for r in group)<=1e-8
        assert len({r['it'] for r in group})==1
        variants[v]=dict(seconds=median(r['total'] for r in group),
                         times=[r['total'] for r in sorted(group,key=lambda x:x['rep'])],
                         cpu=median(r['cpu'] for r in group),it=group[0]['it'],
                         triggered=group[0].get('activated'),dim=group[0].get('dim'))
    rec=dict(case=matches[0]['case'],variants=variants)
    point=variants['new']['seconds']
    rec['plain_ratio']=point/min(variants['p4']['seconds'],variants['p8']['seconds'])
    for other in ['raw','old','fine_pilot_aligned','fixed96','direct16']:
        if other in variants:rec[other+'_ratio']=point/variants[other]['seconds']
    cases.append(rec)
hard=cases[:6]
def metrics(key,group=hard):
    ratios=[c[key] for c in group]
    return dict(median=median(ratios),wins=sum(r<1 for r in ratios),worst=max(ratios))
out=dict(cases=cases,records=len(rows),failures=len(failures),max_residual=max(r['residual'] for r in rows),
         max_coordinate_ritz_residual=max(max(r['coord_residual']) for r in rows if 'coord_residual' in r),
         max_entries_per_row=max(r['row_cap'] for r in rows if r['variant']=='new' and r['activated']),
         hard={k:metrics(k) for k in ['plain_ratio','old_ratio','raw_ratio','fine_pilot_aligned_ratio','fixed96_ratio','direct16_ratio']},
         control=metrics('plain_ratio',cases[6:]))
out['hard']['cpu_old_ratio']=median(c['variants']['new']['cpu']/c['variants']['old']['cpu'] for c in hard)
active=[c for c in hard if c['variants']['new']['triggered']]
out['active_fine_alignment']=metrics('raw_ratio',active)
out['decision']=dict(main_time=out['hard']['old_ratio']['median']<=.90,main_wins=out['hard']['old_ratio']['wins']>=4,control=out['control']['worst']<=1.10)
(H/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# 全部正式计时','', '每格为三次总时间的中位数（秒），括号内为PCG迭代数。原始精确数据见 validation.jsonl。','',
       '| 案例 | P4 | P8 | 上轮门控 | 原门控+校正 | 粗探测 | 粗探测+校正 | 固定96步帽函数 | 直接16模态 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in cases:
    n,f,chi,_,rhs,_=c['case']
    cells=[]
    for tag in ['p4','p8','old','fine_pilot_aligned','raw','new','fixed96','direct16']:
        v=c['variants'].get(tag)
        cells.append(f"{v['seconds']:.3f} ({v['it']})" if v else '—')
    lines.append('| '+f'{f} {n}² χ={chi:g} RHS{rhs}'+' | '+' | '.join(cells)+' |')
(H/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
