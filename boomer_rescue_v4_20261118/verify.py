"""Independent dense reference for the constrained update and pilot Ritz extraction."""
from pilot import s,np,pcg_pilot,ritz_coefficients
from relax import bank
from scipy.linalg import eigh
import json
A,b,u=s.problem('disk',15,1e3,[.5]*3,4,149)
B=s.v1.Boomer(A,'p8')
denseA=A.toarray()
denseB=np.column_stack([B(e) for e in np.eye(len(b))])
x,state,h,Q=pcg_pilot(A,B,b,8)
assert state is not None
ref_x,_,_,_,_,_=s.cg(A,B,b,maxiter=8)
assert np.linalg.norm(x-ref_x)/np.linalg.norm(x)<1e-12
mu,Y=ritz_coefficients(state,2)
q=Q.astype(float)
orth_err=np.max(abs(q.T@np.linalg.solve(denseB,q)-np.eye(8)))
mu_ref=eigh(q.T@denseA@q,q.T@np.linalg.solve(denseB,q),eigvals_only=True)[:2]
assert np.allclose(mu,mu_ref,rtol=1e-5) and orth_err<1e-5
v=np.random.default_rng(19).uniform(-1,1,len(b))
v/=max(abs(v))
bins=8
t=(v-v.min())/(v.max()-v.min())*bins
j=np.minimum(t.astype(int),bins-1)
z0=round(-v.min()/(v.max()-v.min())*bins)
f=t-j
I=np.eye(bins+1)
I[:,z0]=0
left=I[j]
difference=I[j+1]-left
bound=abs(denseA).sum(axis=1)*(difference**2).sum(axis=1)
energies=[]
for step in range(7):
    z=left+difference*f[:,None]
    az=denseA@z
    energies.append(.5*float((z*az).sum()))
    observed=bank(A,v[:,None],bins,step).toarray()
    expected=z[:,np.any(abs(z)>1e-14,axis=0)]
    assert np.allclose(observed,expected,rtol=1e-11,atol=1e-12)
    f=np.clip(f-(az*difference).sum(axis=1)/bound,0,1)
assert np.max(np.diff(energies))<0
B.close()
print(json.dumps(dict(dense_reference=True,energy=energies,pilot_orthogonality=orth_err,ritz_relative_error=(abs(mu/mu_ref-1)).tolist())))
