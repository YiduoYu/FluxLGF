"""A single coarse spectral run supplies both a difficulty probe and the spline coordinates."""
import time
from pilot import s,np,la,problem
from study import DenseCorrection


def probe(A,B,cutoff,probe_steps=12):
    budget=128
    V=np.empty((A.shape[0],budget+1),order='F')
    AV=np.empty_like(V)
    v=np.random.default_rng(91).standard_normal(A.shape[0])
    V[:,0]=v/np.sqrt(v@(A@v))
    AV[:,0]=A@V[:,0]
    H=np.zeros((budget+1,budget))
    early=None
    for j in range(budget):
        w=B(AV[:,j])
        for _ in range(2):
            h=AV[:,:j+1].T@w
            w-=V[:,:j+1]@h
            H[:j+1,j]+=h
        aw=A@w
        beta=np.sqrt(max(w@aw,0))
        H[j+1,j]=beta
        k=j+1
        if k in (probe_steps,24,32,48,64,96,128) or beta<1e-13:
            mu,Y=la.eigh((H[:k,:k]+H[:k,:k].T)/2,subset_by_index=(0,min(2,k)-1))
            eta=beta*abs(Y[-1])/np.maximum(abs(mu),1e-30)
            if early is None:
                early=float(mu[0])
                if early>=cutoff:
                    return None,dict(early=early,second=float(mu[-1]),steps=k)
            if (k>=24 and max(eta)<=.1) or k==budget or beta<1e-13:
                W=V[:,:k]@Y
                checks=[]
                for v,eig in zip(W.T,mu):
                    defect=B(A@v)-eig*v
                    checks.append(float(np.sqrt(max(0,defect@(A@defect)))/max(abs(eig),1e-30)))
                return W,dict(early=early,steps=k,coord_residual=checks)
        V[:,k]=w/beta
        AV[:,k]=aw/beta
    raise AssertionError('No return at maximum budget')


def solve(case,polish,cutoff,probe_steps=12,direct=False,transfer='hats'):
    n,family,chi,p,rhs,seed=case
    A,b,u=problem(*case)
    started=time.perf_counter()
    cpu=time.process_time()
    B=s.v1.Boomer(A,'p8')
    nc=(n+1)//4-1
    Ac,_,_=problem(nc,family,chi,p,0,seed)
    Bc=s.v1.Boomer(Ac,'p8')
    W,info=probe(Ac,Bc,cutoff,probe_steps)
    Bc.close()
    M=B
    if W is not None:
        V=s.v1.prolong(nc,n)@W
        for j in range(polish):
            for col in range(V.shape[1]):
                V[:,col]-=B(A@V[:,col])
        if direct:
            M=DenseCorrection(A,B,V)
        else:
            Z=s.v1.smooth_basis(A,s.v1.lift_basis(V,max(48,3*(n+1)//16),transfer),0)
            M=s.v1.Correction(A,B,Z)
            info['row_cap']=int(np.diff(Z.indptr).max())
        info.update(dim=M.dim)
    setup=time.perf_counter()-started
    x,it,ok,_,_,msg=s.cg(A,M,b)
    elapsed=time.perf_counter()-started
    cpu_s=time.process_time()-cpu
    rr=np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b)
    B.close()
    return dict(info,case=case,total=elapsed,cpu=cpu_s,setup=setup,it=it,activated=W is not None,
                residual=float(rr),ok=bool(ok and rr<=1e-8),message=msg)
