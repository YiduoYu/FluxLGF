"""Post-validation explanation of the 32/96-step discrepancy on four_cells.

Align two A-orthonormal frames with their polar Procrustes map. Repeat PCG with
unchanged value bins. These runs use the same validation matrix and are exploratory.
"""
from pathlib import Path
import json
from method import s,np
from pilot import problem
from validate import hard

n,family,chi,p,rhs,seed=hard[3]
A,b,_=problem(*hard[3])
nc=(n+1)//4-1
C,_,_=problem(nc,family,chi,p,0,seed)
Bc=s.v1.Boomer(C,'p8')
B=s.v1.Boomer(A,'p8')
P=s.v1.prolong(nc,n)
frames={}
ritz={}
for k in (32,96,128):
    W,meta=s.coordinates(C,Bc,6,fixed_steps=k)
    frames[k]=W
    ritz[k]={'eigenvalues':meta['mu'],'residuals':meta['ritz_residual']}
Bc.close()
def compare(source,target):
    cross=source.T@(C@target)
    u,sigma,vt=np.linalg.svd(cross)
    return u@vt, {'overlap':cross.tolist(),'singular_values':sigma.tolist(),
                  'angles_degrees':np.rad2deg(np.arccos(np.clip(sigma,0,1))).tolist()}
def pcg(W):
    Z=s.v1.smooth_basis(A,s.v1.lift_basis(P@W,3*(n+1)//16,'hats'),0)
    M=s.v1.Correction(A,B,Z)
    x,it,ok,_,_,msg=s.cg(A,M,b)
    rel=float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
    assert ok and rel<=1e-8, msg
    return dict(it=it,residual=rel,dim=M.dim)
short=frames[32][:,:2]
long=frames[96][:,:2]
rotation,pair=compare(long,short)
_,stability=compare(frames[128][:,:2],long)
def rotate2(theta):
    t=np.deg2rad(theta)
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
runs={'short':pcg(short),'long':pcg(long),'long_rotated_to_short':pcg(long@rotation),
      'short_rotated_to_long':pcg(short@rotation.T)}
for degrees in [15,30,45,60,75]:
    runs['long_rotated_'+str(degrees)]=pcg(long@rotate2(degrees))
B.close()
report=dict(case=hard[3],ritz=ritz,comparison_32_96=pair,comparison_96_128=stability,
            long_six_by_short_two=(frames[96].T@(C@short)).tolist(),runs=runs)
print(json.dumps(report,indent=2))
