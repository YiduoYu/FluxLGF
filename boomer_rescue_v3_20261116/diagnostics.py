"""Untimed mechanism checks; never run concurrently with performance measurements."""
import study as e
import numpy as np
import scipy.linalg as la
import json,time
from pathlib import Path
from joint import joint_basis
HERE=Path(__file__).resolve().parent

def project(M,A,W):
    AW=A@W
    if isinstance(M,e.DenseCorrection):C=M.coarse(M.Z.T@AW)
    else:C=M.lu.solve(M.ZT@AW)
    R=W-M.Z@C
    return np.sqrt(np.maximum(np.sum(R*(A@R),axis=0),0)/np.sum(W*AW,axis=0))

def coarse_condition(M):
    C=M.C if isinstance(M,e.DenseCorrection) else M.C.toarray()
    scale=np.sqrt(np.diag(C));Cn=C/scale[:,None]/scale[None,:]
    ev=la.eigvalsh((Cn+Cn.T)*.5)
    return dict(min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),
                diagonal_scaled_condition=float(ev[-1]/ev[0]) if ev[0]>0 else None)

def one(fam):
    n=255;p=[.21,.73,.35];A,b,u=e.s.problem(fam,n,1e5,p,4,17119)
    B=e.s.v1.Boomer(A,'p8');nc=63;Ac,_,_=e.s.problem(fam,nc,1e5,p)
    Bc=e.s.v1.Boomer(Ac,'p8')
    W,ci=e.s.coordinates(Ac,Bc,m=32,fixed_steps=96)
    V=e.s.v1.prolong(nc,n)@W;Bc.close()
    # Fine-grid probes are recomputed on a distinct operator, not chosen by RHS.
    T,ti=e.s.coordinates(A,B,m=16,fixed_steps=128)
    rows=[]
    base2=e.DenseCorrection(A,B,V[:,:2]);r=np.random.default_rng(71).normal(size=n*n)
    for tag,m in [('direct2',2),('direct16',16),('direct32',32),('local',2),('weighted',2),('hats',2)]:
        Z=e.basis(A,V[:,:m],n,'direct' if tag.startswith('direct') else tag,48)
        M=e.DenseCorrection(A,B,Z) if tag.startswith('direct') else e.s.v1.Correction(A,B,Z)
        err=project(M,A,T)
        x,it,ok,_,_,_=e.s.cg(A,M,b)
        rr=float(np.linalg.norm(e.s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        rows.append(dict(tag=tag,dim=M.dim,projection_error=err.tolist(),iterations=it,
                         coarse_condition=coarse_condition(M),ok=ok and rr<=1e-8,residual=rr))
    rotations=[]
    joint_rotations=[]
    for angle in [0.,np.pi/8,np.pi/4,np.pi/2]:
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        VR=V[:,:2]@R;D=e.DenseCorrection(A,B,VR)
        inv=float(np.linalg.norm(D(r)-base2(r))/np.linalg.norm(base2(r)))
        assert inv<1e-8
        Z=e.basis(A,VR,n,'weighted',48);M=e.s.v1.Correction(A,B,Z)
        x,it,ok,_,_,_=e.s.cg(A,M,b)
        rr=float(np.linalg.norm(e.s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        rotations.append(dict(angle_degrees=float(np.degrees(angle)),direct_invariance_error=inv,
                              dim=M.dim,projection_error=project(M,A,T).tolist(),
                              coarse_condition=coarse_condition(M),
                              iterations=it,ok=ok and rr<=1e-8,residual=rr))
        try:
            J,ji=joint_basis(VR,W[:,:2]@R,98)
            MJ=e.s.v1.Correction(A,B,J)
            jz=MJ(r)
            if angle==0:joint_z0=jz
            difference=float(np.linalg.norm(jz-joint_z0)/np.linalg.norm(joint_z0))
            x,jit,jok,_,_,_=e.s.cg(A,MJ,b)
            jrr=float(np.linalg.norm(e.s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
            joint_rotations.append(dict(ji,angle_degrees=float(np.degrees(angle)),
                                        iterations=jit,ok=jok and jrr<=1e-8,residual=jrr,
                                        relative_preconditioner_change=difference,
                                        projection_error=project(MJ,A,T).tolist(),
                                        coarse_condition=coarse_condition(MJ)))
        except Exception as exc:
            joint_rotations.append(dict(angle_degrees=float(np.degrees(angle)),ok=False,exception=repr(exc)))
    # Check reproduction with the actual entry-dropping/empty-column removal.
    vv=V[:,0]/max(abs(V[:,0]));lo=min(0.,vv.min());hi=max(0.,vv.max())
    knots=np.linspace(lo,hi,49);t=(vv-lo)/(hi-lo)*48;j=np.minimum(np.floor(t).astype(int),47);f=t-j
    cc=np.column_stack((j,j+1)).ravel()
    values=np.column_stack((vv*(1-f),vv*f)).ravel()
    active=np.unique(cc[abs(values)>1e-14])
    raw=e.s.v1.lift_basis(V[:,:1],48,'weighted')
    assert raw.shape[1]==len(active)
    lin=raw@np.ones(len(active));square=raw@knots[active]
    reproduction=dict(linear=float(np.linalg.norm(lin-vv)),quadratic=float(np.linalg.norm(square-vv**2)))
    assert reproduction['linear']<1e-10 and reproduction['quadratic']<1e-10
    B.close()
    return dict(fam=fam,n=n,coarse_ritz=ci,fine_probe_ritz=ti,comparisons=rows,
                rotations=rotations,joint_rotations=joint_rotations,reproduction=reproduction)

if __name__=='__main__':
    with (HERE/'diagnostics.jsonl').open('w') as f:
        for fam in ['circ','disk','ellipse','two_vortices']:
            row=one(fam);f.write(json.dumps(row)+'\n');f.flush()
            print(fam,[(x['tag'],x['iterations']) for x in row['comparisons']],
                  'rotations',[(x['angle_degrees'],x['iterations']) for x in row['rotations']],
                  'joint',[(x['angle_degrees'],x.get('iterations'),x.get('dim'),x['ok']) for x in row['joint_rotations']],flush=True)
