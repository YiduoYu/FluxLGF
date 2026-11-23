"""Post-discovery, same-column-budget test on fresh four-cell fields."""
import json,sys,time,random,subprocess,hashlib
from pathlib import Path
from method import s,np,la
from pilot import problem
from shared import solve

root=Path(__file__).resolve().parent
cases=[
 (511,'four_cells',1e5,[.24,.69,.08],3,25343),
 (511,'four_cells',1e5,[.82,.37,.74],4,25349),
 (383,'four_cells',8e4,[.39,.13,.61],3,25351),
 (511,'four_cells',6e4,[.63,.77,.42],4,25357)]
methods=['first2','all','anchor','random','early']
def schedule():
    rng=random.Random(79371)
    blocks=[(c,r) for c in range(len(cases)) for r in range(2)]
    rng.shuffle(blocks)
    jobs=[]
    for c,r in blocks:
        order=methods.copy()
        rng.shuffle(order)
        jobs.extend((c,r,v) for v in order)
    return jobs
def encode(W):
    _,_,pivot=la.qr(W.T,mode='economic',pivoting=True)
    anchors=pivot[:W.shape[1]]
    block=W[anchors]
    phase=2*np.pi*np.arange(len(anchors))/len(anchors)
    target=np.column_stack((np.cos(phase),np.sin(phase)))
    target[abs(target)<1e-14]=0
    V=W@la.solve(block,target)
    return V,anchors,float(np.linalg.cond(block))
def run(case,kind):
    if kind=='early':
        return solve(case,1,.08)
    n,f,chi,p,rhs,seed=case
    A,b,_=problem(*case)
    since=time.perf_counter()
    B=s.v1.Boomer(A,'p8')
    nc=(n+1)//4-1
    C,_,_=problem(nc,f,chi,p,0,seed)
    Bc=s.v1.Boomer(C,'p8')
    W,info=s.coordinates(C,Bc,5,fixed_steps=96)
    Bc.close()
    mu=np.asarray(info['mu'])
    m=min(4,int(np.sum(mu<=1.2*mu[0])))
    W=W[:,:max(2,m)]
    extra={}
    if kind=='all':
        V=W
    elif m<3 or kind=='first2':
        V=W[:,:2]
    elif kind=='anchor':
        V,J,cond=encode(W)
        extra.update(anchors=J.tolist(),condition=cond)
    elif kind=='random':
        # Two generic mixtures in the same slow subspace. The node-space random
        # probes make the resulting span invariant to eigenvector sign choices.
        G=np.random.default_rng(87651).normal(size=(W.shape[0],2))
        mixing,_=la.qr(W.T@(C@G),mode='economic')
        V=W@mixing
    else:
        raise ValueError(kind)
    V=s.v1.prolong(nc,n)@V
    for col in range(V.shape[1]):
        V[:,col]-=B(A@V[:,col])
    total_bins=2*max(48,3*(n+1)//16)
    bins=total_bins//V.shape[1]
    Z=s.v1.smooth_basis(A,s.v1.lift_basis(V,bins,'hats'),0)
    M=s.v1.Correction(A,B,Z)
    setup=time.perf_counter()-since
    x,it,ok,_,_,msg=s.cg(A,M,b)
    total=time.perf_counter()-since
    residual=float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
    B.close()
    return dict(total=total,setup=setup,it=it,ok=bool(ok and residual<=1e-8),
                residual=residual,message=msg,dim=M.dim,row_cap=int(np.diff(Z.indptr).max()),
                mu=mu.tolist(),ritz_residual=info['ritz_residual'],cluster=m,**extra)
def manifest():
    paths=['cluster_probe.py','CLUSTER_PROTOCOL.md','shared.py','method.py','pilot.py']
    return dict(cases=cases,jobs=schedule(),hashes={k:hashlib.sha256((root/k).read_bytes()).hexdigest() for k in paths})
if __name__=='__main__':
    if sys.argv[1]=='one':
        i,rep,method=json.loads(sys.argv[2])
        print(json.dumps(run(cases[i],method)))
    elif sys.argv[1]=='lock':
        with (root/'cluster_lock.json').open('x') as h:json.dump(manifest(),h,indent=2)
    else:
        lock=json.loads((root/'cluster_lock.json').read_text())
        assert json.loads(json.dumps(manifest()))==lock
        with (root/'cluster.jsonl').open('x') as out:
            for i,rep,method in lock['jobs']:
                job=(i,rep,method)
                proc=subprocess.run([sys.executable,__file__,'one',json.dumps(job)],capture_output=True,text=True,timeout=180)
                row=json.loads(proc.stdout.splitlines()[-1]) if proc.returncode==0 else dict(ok=False,error=proc.stderr[-1600:])
                row.update(index=i,rep=rep,method=method)
                print(json.dumps(row),file=out,flush=True)
                print(i,rep,method,row.get('it'),row.get('total'),flush=True)
