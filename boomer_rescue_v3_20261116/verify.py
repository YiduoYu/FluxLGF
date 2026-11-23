import json
import study as e
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

A,b,u=e.s.problem('circ',15,1e3,[.5]*3)
B=e.s.v1.Boomer(A,'p8')
V,ri=e.s.coordinates(A,B,m=2,fixed_steps=32)
Md=e.DenseCorrection(A,B,V)
Ms=e.s.v1.Correction(A,B,e.s.v1.smooth_basis(A,sp.csr_matrix(V),0))
r=np.random.default_rng(7).normal(size=len(b))
assert np.linalg.norm(Md(r)-Ms(r))/np.linalg.norm(Md(r))<1e-10
for kind in ['direct','weighted','hats','local']:
    Z=e.basis(A,V,15,kind,8)
    M=e.DenseCorrection(A,B,Z) if kind=='direct' else e.s.v1.Correction(A,B,Z)
    MM=np.column_stack([M(v) for v in np.eye(len(b))])
    sym=np.linalg.norm(MM-MM.T)/np.linalg.norm(MM)
    mineig=np.linalg.eigvalsh((MM+MM.T)/2)[0]
    x,it,ok,_,_,msg=e.s.cg(A,M,b,tol=1e-10)
    assert sym<1e-9 and mineig>0 and ok and np.linalg.norm(x-u)/np.linalg.norm(u)<1e-7
    if kind in ['weighted','local']:
        dense=Z.toarray();coef=la.lstsq(dense,V)[0]
        reprerr=np.linalg.norm(dense@coef-V)/np.linalg.norm(V)
        assert reprerr<1e-9
    else:reprerr=None
    print(json.dumps(dict(kind=kind,symmetry=sym,min_eigenvalue=mineig,
                         representation_error=reprerr,iterations=it,passed=True)))
B.close()
asa=e.run(dict(n=31,fam='disk',chi=1e3,variant='asa'))
assert asa['ok']
print(json.dumps(dict(check='actual_pyamg_asa',passed=True,**asa)))
