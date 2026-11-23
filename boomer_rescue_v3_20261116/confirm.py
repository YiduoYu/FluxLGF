import json,subprocess,random,statistics
from pathlib import Path
from run import hashes,PY
H=Path(__file__).resolve().parent
l=json.loads((H/'LOCK.json').read_text());assert hashes()==l['hashes']
a=json.loads((H/'aggregate.json').read_text());assert a['records']==200
anchors=json.loads((H/'ANCHOR_SUMMARY.json').read_text())
trigger=any(not .9<=r['total_s_ratio']<=1.1 for r in anchors if 'total_s_ratio' in r)
if not trigger:
    (H/'CONFIRMATION_SUMMARY.json').write_text(json.dumps(dict(triggered=False,reason='All available anchor wall ratios within 10%'),indent=2)+'\n')
    print('No confirmation triggered');raise SystemExit(0)
selected={r['case']:r['best_direct'] for r in a['per_case']}
rng=random.Random(17118)
blocks=[(c,r) for c in l['cases'] for r in range(2)];rng.shuffle(blocks)
jobs=[]
for c,rep in blocks:
    tags=['weighted',selected[c['case']]];rng.shuffle(tags)
    for tag in tags:jobs.append(dict(c,**l['methods'][tag],tag=tag,rep=rep))
(H/'CONFIRMATION_PLAN.json').write_text(json.dumps(dict(selected=selected,jobs=jobs,source_hashes=hashes()),indent=2)+'\n')
p=H/'confirmation.jsonl';old=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
seen={(r['case'],r['tag'],r['rep']) for r in old}
with p.open('a') as f:
    for j in jobs:
        if (j['case'],j['tag'],j['rep']) in seen:continue
        spec={k:v for k,v in j.items() if k not in ['case','tag','rep']}
        proc=subprocess.run([PY,str(H/'study.py'),json.dumps(spec)],capture_output=True,text=True,timeout=240)
        row=json.loads([x for x in proc.stdout.splitlines() if x.startswith('{')][-1]);row.update(j,exitcode=proc.returncode,stderr=proc.stderr)
        f.write(json.dumps(row)+'\n');f.flush()
        print(j['case'],j['tag'],j['rep'],row['ok'],row.get('total_s'),flush=True)
rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==32 and all(r['ok'] for r in rows)
out=[]
for c in l['cases']:
    wr=[r for r in rows if r['case']==c['case'] and r['tag']=='weighted']
    dr=[r for r in rows if r['case']==c['case'] and r['tag']==selected[c['case']]]
    ratio=statistics.median(r['total_s'] for r in wr)/statistics.median(r['total_s'] for r in dr)
    cpu=statistics.median(r['cpu_s'] for r in wr)/statistics.median(r['cpu_s'] for r in dr)
    pair=[next(r['total_s'] for r in wr if r['rep']==i)/next(r['total_s'] for r in dr if r['rep']==i) for i in range(2)]
    out.append(dict(case=c['case'],direct=selected[c['case']],ratio=ratio,cpu_ratio=cpu,pair_ratios=pair,
                    weighted_s=statistics.median(r['total_s'] for r in wr),direct_s=statistics.median(r['total_s'] for r in dr)))
summary=dict(triggered=True,records=32,per_case=out,median_ratio=statistics.median(r['ratio'] for r in out),
             median_cpu_ratio=statistics.median(r['cpu_ratio'] for r in out),wins=sum(r['ratio']<1 for r in out),
             max_residual=max(r['residual'] for r in rows))
(H/'CONFIRMATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
