import json,sys,platform,statistics
from pathlib import Path
from run import hashes
HERE=Path(__file__).resolve().parent
lock=json.loads((HERE/'LOCK.json').read_text())
assert lock['hashes']==hashes()
rows=[json.loads(x) for x in (HERE/'results.jsonl').read_text().splitlines()]
expected={(c['case'],t,r) for c in lock['cases'] for t in lock['methods'] for r in range(3 if c['n']==511 else 2)}
assert len(rows)==len(expected)==200
assert {(r['case'],r['tag'],r['rep']) for r in rows}==expected
for r in rows:
    c=next(c for c in lock['cases'] if c['case']==r['case'])
    assert all(r[k]==v for k,v in c.items())
    assert all(r[k]==v for k,v in lock['methods'][r['tag']].items())
    if r['ok']:assert r['residual']<=1e-8 and r.get('exitcode')==0
    if r['variant'] in ['weighted','hats','local','direct'] and r['ok']:
        assert r['ritz_steps']==96
    if r['variant']=='weighted' and r['ok']:assert r['max_row_nnz']<=4
    if r['variant']=='adaptive' and r['ok']:
        assert r['m']==2 and r['gate_steps']==40 and r['gate_lambda']==.04 and r['pilot']==16
nonasa=[r for r in rows if r['variant']!='asa']
assert all(r['ok'] for r in nonasa),'Non-alpha-SA failure: review before claiming viability'
diagnostics=[json.loads(x) for x in (HERE/'diagnostics.jsonl').read_text().splitlines()]
assert len(diagnostics)==4
for d in diagnostics:
    assert all(r['ok'] for r in d['comparisons']+d['rotations'])
    e0=next(r['projection_error'] for r in d['comparisons'] if r['tag']=='direct2')
    e1=next(r['projection_error'] for r in d['comparisons'] if r['tag']=='weighted')
    assert all(b<=a+1e-7 for a,b in zip(e0,e1))
    assert max(r['direct_invariance_error'] for r in d['rotations'])<1e-8
    assert d['rotations'][0]['iterations']==d['rotations'][-1]['iterations'],'90-degree swap/sign invariance failed'
    assert d['reproduction']['linear']<1e-10 and d['reproduction']['quadratic']<1e-10
    for j in d['joint_rotations']:
        if j['ok']:
            assert j['residual']<=1e-8 and j['max_row_nnz']<=3
            assert j['coordinate_reproduction_error']<1e-9
            assert all(b<=a+1e-7 for a,b in zip(e0,j['projection_error']))
iteration_spreads={}
for c,t,_ in expected:
    rr=[r for r in rows if r['case']==c and r['tag']==t and r['ok']]
    if rr:iteration_spreads[f'{c}/{t}']=max(r['iterations'] for r in rr)-min(r['iterations'] for r in rr)
out=dict(status='passed',records=len(rows),source_hashes='unchanged',non_asa_successes=len(nonasa),
         asa_successes=sum(r['ok'] for r in rows if r['variant']=='asa'),
         failures=[{k:r.get(k) for k in ['case','tag','rep','message','exception','timeout','budget_skipped','residual']} for r in rows if not r['ok']],
         max_successful_residual=max(r['residual'] for r in rows if r['ok']),
         max_iteration_spread=max(iteration_spreads.values()),
         diagnostic_cases=len(diagnostics),diagnostic_solves=sum(len(d['comparisons'])+len(d['rotations']) for d in diagnostics),
         exploratory_joint_successes=sum(j['ok'] for d in diagnostics for j in d['joint_rotations']),
         exploratory_joint_failures=sum(not j['ok'] for d in diagnostics for j in d['joint_rotations']),
         stderr_records=sum(bool(r.get('stderr')) for r in rows))
anchors=[json.loads(x) for x in (HERE/'anchors.jsonl').read_text().splitlines()]
assert len(anchors)==16 and all(r['ok'] and r['residual']<=1e-8 and r['exitcode']==0 for r in anchors)
assert len({(r['case'],r['rep']) for r in anchors})==16
anchor_summary=json.loads((HERE/'ANCHOR_SUMMARY.json').read_text())
trigger=any(not .9<=r['total_s_ratio']<=1.1 for r in anchor_summary if 'total_s_ratio' in r)
confirm=json.loads((HERE/'CONFIRMATION_SUMMARY.json').read_text())
assert confirm['triggered']==trigger
if trigger:
    rr=[json.loads(x) for x in (HERE/'confirmation.jsonl').read_text().splitlines()]
    assert len(rr)==32 and all(r['ok'] and r['residual']<=1e-8 and r['exitcode']==0 for r in rr)
    assert len({(r['case'],r['tag'],r['rep']) for r in rr})==32
out.update(anchor_successes=16,confirmation_triggered=trigger,confirmation_successes=32 if trigger else 0)
(HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
