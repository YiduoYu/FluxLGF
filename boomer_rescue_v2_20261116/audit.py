"""Non-oracle validation of numerical records, fixed parameters, and sparse structure."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
from summarize import collect,key
HERE=Path(__file__).resolve().parent
rows=collect(); lock=json.loads((HERE/'LOCK.json').read_text())
assert len(rows)==304 and len({key(r)+(r['tag'],r['rep']) for r in rows})==304
assert len({key(r) for r in rows})==30
assert all(hashlib.sha256((HERE/f).read_bytes()).hexdigest()==v for f,v in lock['source_hashes'].items())
for r in rows:
    for k,v in lock['methods'][r['tag']].items():assert r[k]==v
    assert r.get('audit')=='double-double' and r.get('residual',1)<1e-8 and r['ok']
    assert abs(r['total_s']-r['setup_s']-r['solve_s'])<1e-7
    if r['tag'] in ['adaptive','fixed96']:
        assert r['triggered']==(r['min_ritz']<.04 and r['predicted_remaining']>40)
        if r['triggered']:
            assert r['nnz']<=4*r['n']**2
            assert r['dim']<=2*(max(48,3*(r['n']+1)//16)+1)
            assert r['quality_met']==(max(r['ritz_residual'])<=.1)
    if r['rhs']==3:assert r['error'] is None
g=defaultdict(list)
for r in rows:g[key(r)+(r['tag'],)].append(r)
spread=max(max(x['iterations'] for x in rr)-min(x['iterations'] for x in rr) for rr in g.values())
skip=0
for r in rows:
    if r['tag']=='adaptive' and not r['triggered']:
        p8=g[key(r)+('p8',)]
        assert all(x['iterations']==r['iterations'] for x in p8)
        skip+=1
coords=defaultdict(list)
for r in rows:
    if r['tag']=='adaptive' and r['triggered']:
        coords[(r['fam'],r['n'],r['chi'],tuple(r['p']))].append(r)
rhs_checks=0
for rr in coords.values():
    for r in rr:
        assert r['ritz_steps']==rr[0]['ritz_steps']
        assert max(abs(a-b) for a,b in zip(r['mu'],rr[0]['mu']))<1e-12
    if len({r['rhs'] for r in rr})>1:rhs_checks+=1
out=dict(status='passed',measurements=304,cases=30,locked_hashes_passed=True,
         max_residual=max(r['residual'] for r in rows),max_repeat_iteration_spread=spread,
         skipped_adaptive_records_identical_iterations_to_p8=skip,
         operator_groups_with_multiple_RHS_and_identical_ritz_summaries=rhs_checks,
         activated_records_checked_sparse_nnz_bound=sum(r.get('triggered',False) for r in rows),
         adaptive_quality_failures=sum(not r.get('quality_met',True) for r in rows if r['tag']=='adaptive'),
         stderr_records=sum(bool(r.get('stderr','').strip()) for r in rows))
(HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
