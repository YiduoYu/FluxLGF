import json,subprocess,random,time,statistics
from pathlib import Path
from run import hashes,PY
H=Path(__file__).resolve().parent
l=json.loads((H/'LOCK.json').read_text());assert hashes()==l['hashes']
jobs=[dict(c,rep=r,variant='p8') for c in l['cases'] for r in range(2)]
random.Random(17117).shuffle(jobs)
p=H/'anchors.jsonl';old=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
seen={(r['case'],r['rep']) for r in old}
with p.open('a') as f:
    for j in jobs:
        if (j['case'],j['rep']) in seen:continue
        spec={k:v for k,v in j.items() if k not in ['case','rep']}
        proc=subprocess.run([PY,str(H/'study.py'),json.dumps(spec)],capture_output=True,text=True,timeout=240)
        row=json.loads([x for x in proc.stdout.splitlines() if x.startswith('{')][-1]);row.update(j,exitcode=proc.returncode,stderr=proc.stderr)
        f.write(json.dumps(row)+'\n');f.flush()
        print(j['case'],j['rep'],row['ok'],row.get('total_s'),flush=True)
rows=[json.loads(x) for x in p.read_text().splitlines()]
assert len(rows)==16 and all(r['ok'] for r in rows)
original=[json.loads(x) for x in (H/'results.jsonl').read_text().splitlines()]
note=json.loads((H/'RESUME_NOTE.json').read_text());first=original[:note['completed_rows_before_resume']]
out=[]
for c in l['cases']:
    before=[r for r in first if r['case']==c['case'] and r['tag']=='p8' and r['ok']]
    now=[r for r in rows if r['case']==c['case']]
    d=dict(case=c['case'],original_session_samples=len(before),new_samples=len(now))
    if before:
        for field in ['total_s','cpu_s']:
            d[field+'_ratio']=statistics.median(r[field] for r in now)/statistics.median(r[field] for r in before)
        d['iteration_identity']=all(r['iterations']==before[0]['iterations'] for r in now+before)
    out.append(d)
(H/'ANCHOR_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
