from pathlib import Path
import sys,json,random,hashlib,time,subprocess
H=Path(__file__).resolve().parent
hard=[
(511,'circ',1e5,[.41,.83,.17],3,19371),
(511,'disk',1e5,[.88,.66,.32],4,19373),
(511,'two_vortices',1e5,[.16,.24,.79],3,19375),
(511,'four_cells',1e5,[.57,.91,.25],4,19377),
(511,'ellipse',3e4,[.08,.38,.86],4,19379),
(767,'disk',1e5,[.29,.87,.54],3,19381)]
controls=[
(511,'band',1e5,[.32,.84,.71],4,19383),
(511,'rot',1e5,[.67,.31,.52],3,19385),
(511,'disk',1e3,[.76,.15,.23],4,19387)]
common=['p4','p8','old','raw','new','fine_pilot_aligned']
references=['fixed96','direct16']
def hashes():
    paths=['validate.py','shared.py','pilot.py','method.py','PROTOCOL.md',
           '../boomer_rescue_v3_20261116/study.py','../boomer_rescue_v2_20261116/solver.py',
           '../boomer_rescue_20261116/experiment.py','../generalize/snn/cases_param.py',
           '../anisotropic/ops_aniso.py','../solver_challenge/ir_fp128.py']
    return {name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in paths}
def jobs():
    rng=random.Random(91261)
    blocks=[(i,rep) for i in range(9) for rep in range(3)]
    rng.shuffle(blocks)
    schedule=[]
    for i,rep in blocks:
        variants=common.copy()
        if i<6: variants+=references
        rng.shuffle(variants)
        schedule.extend(dict(case=(hard+controls)[i],index=i,rep=rep,variant=v) for v in variants)
    return schedule

def execute(job):
    from pilot import trial
    from shared import solve
    from method import measure
    case,variant=job['case'],job['variant']
    if variant in ('new','raw'):
        return solve(case,int(variant=='new'),.08)
    if variant=='fine_pilot_aligned':
        return trial(case,'coarse1',polish=1)
    if variant in ('fixed96','direct16'):
        import method
        # Substitute only the new problem generator; all legacy cases use their existing path.
        if case[1]=='four_cells':
            from pilot import four_cells
            old=method.s.problem
            def wrap(fam,n,chi,p,rhs=1,rhs_seed=3781):
                return four_cells(n,chi,p,rhs,rhs_seed) if fam=='four_cells' else old(fam,n,chi,p,rhs,rhs_seed)
            method.s.problem=wrap
        return measure(case,'legacy' if variant=='fixed96' else 'direct16',steps=96)
    return trial(case,variant)
if __name__=='__main__':
    if sys.argv[1]=='one':
        job=json.loads(sys.argv[2]);print(json.dumps(execute(job)))
    elif sys.argv[1]=='lock':
        with (H/'LOCK.json').open('x') as f:
            json.dump(dict(hashes=hashes(),jobs=jobs(),unix_time=time.time()),f,indent=2)
    else:
        data=json.loads((H/'LOCK.json').read_text())
        assert data['hashes']==hashes()
        p=H/'validation.jsonl'
        existing=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        completed={(r['index'],r['variant'],r['rep']) for r in existing}
        with p.open('a') as out:
            for idx,job in enumerate(data['jobs']):
                if (job['index'],job['variant'],job['rep']) in completed:continue
                try:
                    proc=subprocess.run([sys.executable,__file__,'one',json.dumps(job)],capture_output=True,text=True,timeout=240)
                    if proc.returncode:raise RuntimeError(proc.stderr[-1800:])
                    row=json.loads(proc.stdout.strip().splitlines()[-1])
                except Exception as exc:
                    row=dict(ok=False,exception=repr(exc))
                row.update(job)
                out.write(json.dumps(row)+'\n');out.flush()
                print(idx+1,job['index'],job['variant'],row['ok'],row.get('it'),round(row.get('total',0),2),flush=True)
