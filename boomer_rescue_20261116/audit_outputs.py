"""Check completeness, independent residuals, parameter lock and timing identities."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from summarize import collect, identity, casekey, read

HERE=Path(__file__).resolve().parent
lock=json.loads((HERE/'LOCK.json').read_text())
hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in lock['source_hashes']}
assert hashes==lock['source_hashes'], 'Locked source was changed'
raw_counts={name:len(read(name)) for name in ['baseline_screen.jsonl','prototype_screen.jsonl',
    'scale_dev.jsonl','ablate_dev.jsonl','gate_dev.jsonl','validate511.jsonl','validate1023.jsonl',
    'validate2047.jsonl','remeasure1023.jsonl','controls_repeats.jsonl','topology.jsonl','diagnose_scale.jsonl']}
assert [raw_counts[n] for n in ['validate511.jsonl','validate1023.jsonl','validate2047.jsonl',
    'remeasure1023.jsonl','controls_repeats.jsonl','topology.jsonl','diagnose_scale.jsonl']]==[72,24,12,5,27,12,4]
rows=collect()
assert len(rows)==147 and len({identity(r) for r in rows})==147
assert len({casekey(r) for r in rows})==20
diagnostics=read('diagnose_scale.jsonl')
assert len({identity(r) for r in diagnostics})==4
for r in diagnostics:
    assert r['ok'] and r['residual']<=1e-8 and r['audit']=='double-double'
    assert r['steps']==96 and r['n']==2047 and r['bins']==384
    for key,value in lock['methods']['rescue'].items():
        if key!='steps': assert r[key]==value
for r in rows:
    assert r['ok'] and r['residual']<=1e-8 and r['audit']=='double-double'
    assert abs(r['setup_s']+r['solve_s']-r['total_s'])<1e-7
    for key,value in lock['methods'][r['tag']].items(): assert r[key]==value,(r,key)
    if r['tag']=='rescue': assert r['bins']==max(48,3*(r['n']+1)//16)
g=defaultdict(list)
for r in rows:g[casekey(r)+(r['tag'],)].append(r)
iteration_spreads={str(k):max(r['iterations'] for r in rr)-min(r['iterations'] for r in rr)
                   for k,rr in g.items()}
dependencies=['generalize/snn/cases_param.py','anisotropic/ops_aniso.py','solver_challenge/ir_fp128.py']
dep_hashes={p:hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest() for p in dependencies}
out=dict(status='passed',locked_hashes=hashes,dependency_hashes_at_audit=dep_hashes,
         raw_counts=raw_counts,retained_validation_measurements=len(rows),unique_cases=20,
         largest_repeat_iteration_difference=max(iteration_spreads.values()),
         max_double_double_residual=max(r['residual'] for r in rows),
         max_posthoc_residual=max(r['residual'] for r in diagnostics),
         stderr_records=sum(bool(r.get('stderr','').strip()) for r in rows),
         note='Dependencies were hashed at audit time; these were not locked before experiments. Post-hoc diagnostics are excluded from retained validation.')
(HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
