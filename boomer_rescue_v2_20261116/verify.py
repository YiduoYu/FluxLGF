import json
import solver as s
import numpy as np
from scipy.sparse.linalg import spsolve
A,b,u=s.problem('circ',15,1e3,[.5]*3)
B=s.v1.Boomer(A,'p8')
x,k,ok,_,_,_=s.cg(A,B,b)
xold,kold,okold,_,_=s.v1.solve(A,B,b)
assert ok and okold and k==kold and np.linalg.norm(x-xold)<1e-13
xp,kp,_,state,hh,_=s.cg(A,B,b,maxiter=3)
xc,kc,okc,_,_,_=s.cg(A,B,b,x0=xp,state=state)
assert okc and kp+kc==k and np.linalg.norm(xc-x)<1e-13
W,info=s.coordinates(A,B,2,rtol=.1,min_steps=8,max_steps=64)
assert np.allclose(info['ritz_residual'],info['ritz_trace'][-1]['relative_ritz_residual'],rtol=1e-4,atol=1e-8)
for split in [False,True]:
    Z=s.v1.lift_basis(W,8,'weighted')
    if split:Z=s.split_supports(Z,15,min_component=2)
    Z=s.v1.smooth_basis(A,Z,0);M=s.v1.Correction(A,B,Z)
    dense=np.column_stack([M(v) for v in np.eye(len(b))])
    sym=np.linalg.norm(dense-dense.T)/np.linalg.norm(dense)
    mineig=np.linalg.eigvalsh((dense+dense.T)/2)[0]
    xx,_,yes,_,_,_=s.cg(A,M,b,tol=1e-10)
    assert sym<1e-10 and mineig>0 and yes and np.linalg.norm(xx-spsolve(A,b))/np.linalg.norm(u)<1e-7
    print(json.dumps(dict(split=split,symmetry=sym,min_eigenvalue=mineig)))
B.close()
print(json.dumps(dict(check='CG_continuation_and_Ritz_residual',passed=True,ritz=info)))
