import json,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
cases=json.loads((HERE/'VALIDATION_CASES.json').read_text())
common=dict(cfg='p8',m=2,pilot=16,gate='both',gate_steps=40.,gate_lambda=.04,
            rtol=.1,min_steps=24,max_steps=128,split=False,div=4,bins=0)
methods=dict(base=dict(method='amg',cfg='base'),p8=dict(method='amg',cfg='p8'),
             adaptive=dict(method='adaptive',**common),fixed96=dict(method='fixed',fixed_steps=96,**common))
files=['solver.py','bench.py','PROTOCOL.md','VALIDATION_CASES.json',
       '../boomer_rescue_20261116/experiment.py','../generalize/snn/cases_param.py',
       '../anisotropic/ops_aniso.py','../solver_challenge/ir_fp128.py']
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         cases=cases['cases'],methods=methods,repeats={key:2 for key in cases['cases']},order_seed=171116,
         control_extra_repetition=2,
         definitions=dict(hard='circ/disk, chi=1e5, all declared grids and RHS (14 cases)',
                          controls='circ/disk chi=1e3 plus band/rot/poisson/jump (14 cases)',
                          topology='ellipse and two_vortices (2 cases)',
                          baseline='faster median of base/p8 for each case',
                          success='all solves audited; hard median ratio <=0.75; every control median ratio <=1.10; both largest cases ratio <=0.75; report topology separately'),
         caveats=['Gate is an empirical remaining-iteration proxy, NOT a calibrated setup-cost predictor.',
                  'Two coordinates selected from development; no geometry-family routing.',
                  'Adaptive/fixed96 share gates, fine Boomer, coordinate count and sparse basis; only Ritz stopping differs.',
                  'Support splitting rejected on development cost; it is not in either locked method.',
                  'All first-round problems are development; this file fixes fresh validation cases.'],
         source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in files})
target=HERE/'LOCK.json'
assert not target.exists(),'Do not overwrite a validation lock'
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cases={k:len(v) for k,v in out['cases'].items()},methods=methods),indent=2))
