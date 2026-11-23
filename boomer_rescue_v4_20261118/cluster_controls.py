"""Untimed, post hoc controls for the necessity of QR anchoring."""
import json
from cluster_probe import cases,encode,s,np,la,problem
for i,case in enumerate(cases):
    n,f,chi,p,rhs,seed=case
    A,b,_=problem(*case)
    nc=(n+1)//4-1
    C,_,_=problem(nc,f,chi,p,0,seed)
    B=s.v1.Boomer(A,'p8')
    Bc=s.v1.Boomer(C,'p8')
    W,meta=s.coordinates(C,Bc,5,fixed_steps=96)
    Bc.close()
    m=min(4,int(np.sum(np.asarray(meta['mu'])<=1.2*meta['mu'][0])))
    W=W[:,:m]
    encoded,J,_=encode(W)
    O,_=la.qr(np.random.default_rng(76319).normal(size=(m,m)))
    rotated,J2,_=encode(W@O)
    coordinate_difference=float(la.norm(rotated-encoded)/la.norm(encoded))
    phase=2*np.pi*np.arange(m)/m
    code=np.column_stack((np.cos(phase),np.sin(phase)))
    code[abs(code)<1e-14]=0
    U=W/W[np.argmax(abs(W),axis=0),np.arange(m)]
    cardinal=la.solve(W[J,:].T,W.T).T
    result={}
    for name,coarse in [('modal_peak_code',U@code),('all_cardinal',cardinal)]:
        V=s.v1.prolong(nc,n)@coarse
        for j in range(V.shape[1]):
            V[:,j]-=B(A@V[:,j])
        bins=2*max(48,3*(n+1)//16)//V.shape[1]
        Z=s.v1.smooth_basis(A,s.v1.lift_basis(V,bins,'hats'),0)
        M=s.v1.Correction(A,B,Z)
        x,it,ok,_,_,_=s.cg(A,M,b)
        rel=float(la.norm(s.v1.audit_residual(A,b,x))/la.norm(b))
        assert ok and rel<=1e-8
        result[name]=dict(it=it,residual=rel,max_nnz=int(np.diff(Z.indptr).max()))
    B.close()
    print(json.dumps(dict(index=i,same_anchors=bool(np.all(J==J2)),rotation_error=coordinate_difference,tests=result)),flush=True)
