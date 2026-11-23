import hashlib,json,random,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/opt/anaconda3/envs/hypre_env/bin/python'
def hashes():
    names=['study.py','run.py','PROTOCOL.md','../boomer_rescue_v2_20261116/solver.py',
           '../boomer_rescue_20261116/experiment.py','../generalize/snn/cases_param.py',
           '../anisotropic/ops_aniso.py','../solver_challenge/ir_fp128.py']
    return {p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in names}
def lock():
    C=[.21,.73,.35];D=[.47,.19,.81]
    cases=[dict(case=f'{f}_511',fam=f,n=511,chi=1e5,p=C,rhs=4,rhs_seed=17119)
           for f in ['circ','disk','ellipse','two_vortices']]
    cases += [dict(case=f'{f}_1023',fam=f,n=1023,chi=1e5,p=D,rhs=3,rhs_seed=17123)
              for f in ['circ','disk','two_vortices']]
    cases += [dict(case='circ_mid_1023',fam='circ',n=1023,chi=1e4,p=D,rhs=4,rhs_seed=17123)]
    methods={v:dict(variant=v) for v in ['base','p8','adaptive','weighted','hats','local','asa']}
    methods.update({f'direct{m}':dict(variant='direct',m=m) for m in [2,16,32]})
    data=dict(cases=cases,methods=methods,hashes=hashes(),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=HERE/'LOCK.json'
    if path.exists():raise RuntimeError('Refuse to overwrite lock')
    path.write_text(json.dumps(data,indent=2)+'\n')
def run():
    locked=json.loads((HERE/'LOCK.json').read_text())
    assert hashes()==locked['hashes'],'Locked source changed'
    jobs=[dict(c,**method,tag=tag,rep=r) for c in locked['cases']
          for tag,method in locked['methods'].items() for r in range(3 if c['n']==511 else 2)]
    random.Random(17116).shuffle(jobs)
    out=HERE/'results.jsonl';old=[json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    key=lambda x:(x['case'],x['tag'],x['rep'])
    seen={key(x) for x in old};timeouts={(x['case'],x['tag']) for x in old if x.get('timeout')}
    with out.open('a') as f:
        for i,job in enumerate(jobs):
            if key(job) in seen:continue
            spec={k:v for k,v in job.items() if k not in ['case','tag','rep']}
            started=time.perf_counter()
            if (job['case'],job['tag']) in timeouts:
                row=dict(ok=False,budget_skipped=True,message='Same case/method previously timed out at 240s')
            else:
                try:
                    p=subprocess.run([PY,str(HERE/'study.py'),json.dumps(spec)],capture_output=True,text=True,timeout=240)
                    row=json.loads([x for x in p.stdout.splitlines() if x.startswith('{')][-1])
                    row.update(exitcode=p.returncode,stderr=p.stderr[-3000:])
                except subprocess.TimeoutExpired:
                    row=dict(ok=False,timeout=True,message='240 second process time budget')
                    timeouts.add((job['case'],job['tag']))
                except Exception as exc:row=dict(ok=False,exception=repr(exc))
            row.update(job,process_s=time.perf_counter()-started)
            f.write(json.dumps(row)+'\n');f.flush()
            print(i+1,len(jobs),job['case'],job['tag'],job['rep'],'ok',row['ok'],
                  'it',row.get('iterations'),'s',round(row.get('total_s',0),3),
                  row.get('exception',row.get('message','')),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='lock':lock()
    else:run()
