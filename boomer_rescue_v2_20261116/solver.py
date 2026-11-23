"""Adaptive spectral-coordinate correction. First-round code is imported unchanged."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ['OMPI_MCA_btl']='self'
import json
import sys
import time
import resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'boomer_rescue_20261116'))
import experiment as v1
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.ndimage import label
from ops_aniso import tensor,assemble

DEFAULTS=dict(fam='circ',n=511,chi=1e5,p=[.5,.5,.5],rhs=1,rhs_seed=3781,
              method='adaptive',cfg='p8',pilot=16,gate='none',gate_steps=25.,
              gate_lambda=.04,m=1,rtol=.1,min_steps=24,max_steps=128,fixed_steps=96,
              div=4,bins=0,split=False,min_component=8,max_components=4096,maxiter=1500)

def problem(fam,n,chi,p,rhs=1,rhs_seed=3781):
    if fam in ['ellipse','two_vortices']:
        t=np.linspace(0,1,n+2); X,Y=np.meshgrid(t,t)
        if fam=='ellipse':
            cx=.53+.12*(p[0]-.5); cy=.47+.12*(p[1]-.5)
            aa=1.25+.4*(p[2]-.5); bb=.65-.2*(p[2]-.5)
            vx=-(Y-cy)/bb**2; vy=(X-cx)/aa**2
        else:
            ax=.6*(p[0]-.5); ay=.6*(p[1]-.5)
            gx=X+ax*np.sin(2*np.pi*X)/(2*np.pi)
            gy=Y+ay*np.sin(2*np.pi*Y)/(2*np.pi)
            vx=-np.pi*np.sin(2*np.pi*gx)*np.cos(np.pi*gy)*(1+ay*np.cos(2*np.pi*Y))
            vy=2*np.pi*np.cos(2*np.pi*gx)*np.sin(np.pi*gy)*(1+ax*np.cos(2*np.pi*X))
        one=np.ones_like(X); k11,k12,k22=tensor(np.arctan2(vy,vx),chi*one,one)
        xx,yy=X[1:-1,1:-1],Y[1:-1,1:-1]
        q=1+.2*np.cos(2*np.pi*xx)*np.cos(2*np.pi*yy)
        A=assemble(k11,k12,k22,q,1/(n+1))
    else:
        A,_,_=v1.problem(fam,n,chi,p,0)
        xx,yy=np.meshgrid(np.arange(1,n+1)/(n+1),np.arange(1,n+1)/(n+1))
    if rhs==3:
        # Actual random FORCING, not A times a random manufactured solution.
        return A,np.random.default_rng(rhs_seed).standard_normal(n*n),None
    u=np.sin(np.pi*xx)*np.sin(np.pi*yy)
    if rhs==1: u+=.23*np.sin(3*np.pi*xx)*np.sin(2*np.pi*yy)+.17*np.sin(7*np.pi*xx)*np.sin(5*np.pi*yy)
    if rhs==2: u=np.random.default_rng(rhs_seed).standard_normal((n,n))
    if rhs==4:
        rng=np.random.default_rng(rhs_seed)
        for i,j in [(2,5),(4,3),(6,7),(9,2)]:
            u+=rng.uniform(-.3,.3)*np.sin(i*np.pi*xx)*np.sin(j*np.pi*yy)
    u=u.ravel(); return A,A@u,u

def cg(A,B,b,maxiter=1500,x0=None,state=None,tol=1e-8):
    x=np.zeros_like(b) if x0 is None else x0.copy(); normb=np.linalg.norm(b)
    if state is None:
        r=b-A@x; z=B(r); p=z.copy(); rz=r@z
    else:r,p,rz=state['r'].copy(),state['p'].copy(),state['rz']
    al=[]; be=[]; hist=[]; energy=[]; restarts=0
    for k in range(1,maxiter+1):
        ap=A@p; pap=p@ap
        if pap<=0 or rz<=0 or not np.isfinite(rz):
            return x,k,False,None,hist,'breakdown'
        alpha=rz/pap; al.append(alpha); x+=alpha*p; r-=alpha*ap
        rel=float(np.linalg.norm(r)/normb); hist.append(rel)
        if rel<tol*.8:
            r=b-A@x
            if np.linalg.norm(r)/normb<tol:return x,k,True,None,hist,''
            restarts+=1; z=B(r); p=z.copy(); rz=r@z
            if restarts>8:return x,k,False,None,hist,'residual floor'
            al=[]; be=[]; energy=[]
            continue
        z=B(r); rz2=r@z; beta=rz2/rz; be.append(beta)
        energy.append(float(np.sqrt(max(rz2,0))))
        p=z+beta*p; rz=rz2
    return x,maxiter,False,dict(r=r,p=p,rz=rz,al=al,be=be,energy=energy),hist,'iteration limit'

def prediction(state,hist):
    if state is None:return dict(min_ritz=1.,rate=0.,predicted_remaining=0.)
    lam=v1.probe_min(state)
    h=np.maximum(np.asarray(hist[-8:]),1e-300)
    slope=float(-np.polyfit(np.arange(len(h)),np.log(h),1)[0])
    # Use the latest residual and observed contraction, without a solution oracle.
    remaining=float(np.log(max(hist[-1],1e-8)/1e-8)/max(slope,1e-3))
    return dict(min_ritz=lam,rate=slope,predicted_remaining=remaining)

def coordinates(A,B,m=1,rtol=.1,min_steps=24,max_steps=128,fixed_steps=0):
    """A-inner-product Arnoldi; stopping uses relative Ritz residual, never RHS."""
    budget=fixed_steps or max_steps
    v=np.random.default_rng(91).standard_normal(A.shape[0]); av=A@v
    v/=np.sqrt(v@av)
    V=np.empty((A.shape[0],budget+1),order='F'); AV=np.empty_like(V)
    V[:,0]=v; AV[:,0]=A@v; H=np.zeros((budget+1,budget)); trace=[]
    checkpoints={16,24,32,48,64,96,128,budget,min_steps}
    for j in range(budget):
        w=B(AV[:,j])
        for _ in range(2):
            hh=AV[:,:j+1].T@w; w-=V[:,:j+1]@hh; H[:j+1,j]+=hh
        aw=A@w; beta=float(np.sqrt(max(w@aw,0)))
        H[j+1,j]=beta; k=j+1
        if beta>1e-13:V[:,j+1]=w/beta; AV[:,j+1]=aw/beta
        if k in checkpoints or beta<=1e-13:
            mu,Y=la.eigh((H[:k,:k]+H[:k,:k].T)/2,subset_by_index=[0,min(m,k)-1])
            eta=beta*abs(Y[-1,:])/np.maximum(abs(mu),1e-30)
            trace.append(dict(steps=k,mu=mu.tolist(),relative_ritz_residual=eta.tolist()))
            if (not fixed_steps and k>=min_steps and np.max(eta)<=rtol) or k==budget or beta<=1e-13:
                W=V[:,:k]@Y
                # One independent A-norm residual check for each returned coordinate.
                actual=[]
                for col,eig in zip(W.T,mu):
                    rr=B(A@col)-eig*col
                    actual.append(float(np.sqrt(max(rr@(A@rr),0))/max(abs(eig),1e-30)))
                return W,dict(ritz_steps=k,ritz_trace=trace,mu=mu.tolist(),ritz_residual=actual,
                              quality_met=bool(max(actual)<=rtol))

def split_supports(Z,n,min_component=8,max_components=4096):
    """Split disconnected 4-neighbor support components; preserve all entries.

    Tiny components share their original residual column; no entries are dropped.
    This geometrical splitting is exploratory and not claimed matrix-only.
    """
    Z=Z.tocsc(); rows=[]; cols=[]; vals=[]; col=0
    for j in range(Z.shape[1]):
        ids=Z.indices[Z.indptr[j]:Z.indptr[j+1]]; vv=Z.data[Z.indptr[j]:Z.indptr[j+1]]
        mask=np.zeros(n*n,dtype=bool); mask[ids]=True
        lab,count=label(mask.reshape(n,n)); groups=lab.ravel()[ids]
        sizes=np.bincount(groups,minlength=count+1)
        large=np.flatnonzero(sizes>=min_component); large=large[large>0]
        small=~np.isin(groups,large)
        for g in large:
            keep=groups==g; rows.append(ids[keep]); vals.append(vv[keep]); cols.append(np.full(keep.sum(),col)); col+=1
        if small.any():rows.append(ids[small]); vals.append(vv[small]); cols.append(np.full(small.sum(),col)); col+=1
        if col>max_components:raise RuntimeError('support splitting exceeded coarse-space budget')
    return sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,col))

def run(spec):
    args=dict(DEFAULTS,**spec); n=args['n']; m=args['m']
    A,b,u=problem(args['fam'],n,args['chi'],args['p'],args['rhs'],args['rhs_seed'])
    t=time.perf_counter(); cpu=time.process_time(); B=v1.Boomer(A,args['cfg'])
    base_setup=time.perf_counter()-t; pilot_s=0.; pilot_it=0; start=state=None; done=False; hist0=[]; info={}
    use=args['method']!='amg'
    if use and args['pilot']:
        tp=time.perf_counter(); start,pilot_it,done,state,hist0,msg=cg(A,B,b,maxiter=args['pilot'])
        pilot_s=time.perf_counter()-tp
        if not done and state is None:raise RuntimeError(msg)
        pred=prediction(state,hist0); info.update(pred)
        info.update(predicted_remaining_s=pred['predicted_remaining']*pilot_s/max(pilot_it,1),
                    activation_budget_s=args['gate_steps']*pilot_s/max(pilot_it,1))
        use=not done
        if args['gate']=='time':use=use and pred['predicted_remaining']>args['gate_steps']
        elif args['gate']=='ritz':use=use and pred['min_ritz']<args['gate_lambda']
        elif args['gate']=='both':use=use and pred['predicted_remaining']>args['gate_steps'] and pred['min_ritz']<args['gate_lambda']
    info['triggered']=use; solver=B
    if use:
        tc=time.perf_counter(); nc=max(15,(n+1)//args['div']-1)
        Ac,_,_=problem(args['fam'],nc,args['chi'],args['p'],0,args['rhs_seed'])
        Bc=v1.Boomer(Ac,args['cfg'])
        W,ri=coordinates(Ac,Bc,m,args['rtol'],args['min_steps'],args['max_steps'],
                         args['fixed_steps'] if args['method']=='fixed' else 0)
        Bc.close(); V=v1.prolong(nc,n)@W
        info.update(ri,coordinate_s=time.perf_counter()-tc)
        bins=args['bins'] or max(48,3*(n+1)//16)
        Z=v1.lift_basis(V,bins,'weighted'); info['unsplit_dim']=Z.shape[1]
        if args['split']:Z=split_supports(Z,n,args['min_component'],args['max_components'])
        Z=v1.smooth_basis(A,Z,0); solver=v1.Correction(A,B,Z)
        info.update(dim=solver.dim,nnz=solver.nnz,coarse_nnz=solver.cnnz)
    setup=time.perf_counter()-t-pilot_s
    if done:x,it,ok,hist,msg=start,pilot_it,True,hist0,''
    else:
        x,it,ok,_,hist,msg=cg(A,solver,b,args['maxiter']-pilot_it,x0=start,state=None if use else state)
        it+=pilot_it; hist=hist0+hist
    total=time.perf_counter()-t; cpusec=time.process_time()-cpu
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rr=v1.audit_residual(A,b,x); residual=float(np.linalg.norm(rr)/np.linalg.norm(b))
    error=None if u is None else float(np.linalg.norm(x-u)/np.linalg.norm(u))
    result=dict(**args,**info,ok=bool(ok and residual<=1e-8),iterations=it,residual=residual,error=error,
                base_setup_s=base_setup,setup_s=setup,solve_s=total-setup,total_s=total,cpu_s=cpusec,
                pilot_s=pilot_s,pilot_iterations=pilot_it,peak_rss_bytes=peak,message=msg,
                history=hist,audit='double-double',petsc_version=v1.PETSc.Sys.getVersion())
    B.close(); return result

if __name__=='__main__':
    spec=json.loads(sys.argv[1])
    try: print(json.dumps(run(spec)),flush=True)
    except Exception as e:
        import traceback
        print(json.dumps(dict(**spec,ok=False,exception=repr(e),traceback=traceback.format_exc())),flush=True)
        sys.exit(1)
