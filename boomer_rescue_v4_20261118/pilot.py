"""Extract Ritz coordinates from the PCG pilot already spent on the target RHS."""
from method import s,np,la,DenseCorrection
import time


def pcg_pilot(A,B,b,length):
    # The stored B*r vectors are B^{-1}-orthonormal after signed normalization.
    # Alternating signs make the Lanczos subdiagonal positive.
    norm_b=np.linalg.norm(b)
    x=np.zeros_like(b)
    r=b.copy()
    z=B(r)
    p=z.copy()
    product=r@z
    vectors=np.empty((b.size,length),dtype=np.float32,order='F')
    alpha,beta,history=[],[],[]
    for j in range(length):
        vectors[:,j]=((-1)**j/np.sqrt(product))*z
        ap=A@p
        step=product/(p@ap)
        alpha.append(step)
        x+=step*p
        r-=step*ap
        history.append(float(np.linalg.norm(r)/norm_b))
        if history[-1]<.8e-8 and np.linalg.norm(b-A@x)/norm_b<1e-8:
            return x,None,history,vectors[:,:j+1]
        z=B(r)
        new_product=r@z
        beta.append(new_product/product)
        p=z+beta[-1]*p
        product=new_product
    state=dict(r=r,p=p,rz=product,al=alpha,be=beta,energy=[])
    return x,state,history,vectors


def ritz_coefficients(state,count):
    a=np.asarray(state['al'])
    b=np.asarray(state['be'])[:-1]
    diagonal=1/a
    diagonal[1:]+=b/a[:-1]
    return la.eigh_tridiagonal(diagonal,np.sqrt(b)/a[:-1],select='i',select_range=(0,count-1))


def four_cells(n,chi,p,rhs,seed):
    # Tangent to sin(2pi gx)sin(2pi gy) contours on a smoothly warped square.
    from ops_aniso import tensor,assemble
    axis=np.linspace(0,1,n+2)
    X,Y=np.meshgrid(axis,axis)
    ax,ay=.7*(p[0]-.5),.7*(p[1]-.5)
    gx=X+ax*np.sin(2*np.pi*X)/(2*np.pi)
    gy=Y+ay*np.sin(2*np.pi*Y)/(2*np.pi)
    vx=-np.sin(2*np.pi*gx)*np.cos(2*np.pi*gy)*(1+ay*np.cos(2*np.pi*Y))
    vy=np.cos(2*np.pi*gx)*np.sin(2*np.pi*gy)*(1+ax*np.cos(2*np.pi*X))
    one=np.ones_like(X)
    k=tensor(np.arctan2(vy,vx),chi*one,one)
    xx,yy=X[1:-1,1:-1],Y[1:-1,1:-1]
    A=assemble(*k,1+.2*np.cos(2*np.pi*xx)*np.cos(2*np.pi*yy),1/(n+1))
    if rhs==3:
        return A,np.random.default_rng(seed).standard_normal(n*n),None
    u=np.sin(np.pi*xx)*np.sin(np.pi*yy)+.21*np.sin(3*np.pi*xx)*np.sin(7*np.pi*yy)
    return A,A@u.ravel(),u.ravel()


def problem(n,family,chi,p,rhs,seed):
    if family=='four_cells':
        return four_cells(n,chi,p,rhs,seed)
    return s.problem(family,n,chi,p,rhs,seed)


def trial(case, variant, length=16, count=2, polish=0, basis_steps=0):
    n,family,chi,p,rhs,seed=case
    A,b,u=problem(*case)
    started=time.perf_counter()
    cpu=time.process_time()
    B=s.v1.Boomer(A,'base' if variant=='p4' else 'p8')
    info=dict(activated=False)
    if variant in ('p4','p8'):
        x,k,ok,_,hist,msg=s.cg(A,B,b)
    else:
        if variant in ('old','coarse0','coarse1','bounded','jacobi'):
            x,k,done,state,hist,_=s.cg(A,B,b,maxiter=length)
            vectors=None
        else:
            x,state,hist,vectors=pcg_pilot(A,B,b,length)
            k=len(hist)
            done=state is None
        pred=s.prediction(state,hist)
        active=not done and pred['min_ritz']<.04 and pred['predicted_remaining']>40
        info.update(activated=bool(active),**pred)
        ok=done
        msg=''
        if not done:
            M=B
            if active:
                if vectors is None:
                    nc=(n+1)//4-1
                    Ac,_,_=problem(nc,family,chi,p,0,seed)
                    Bc=s.v1.Boomer(Ac,'p8')
                    W,ri=s.coordinates(Ac,Bc,2)
                    Bc.close()
                    V=s.v1.prolong(nc,n)@W
                    info['coarse_steps']=ri['ritz_steps']
                else:
                    mu,Y=ritz_coefficients(state,count)
                    # Blocked conversion avoids a full float64 copy of the pilot.
                    V=np.empty((len(b),count))
                    for lo in range(0,len(b),32768):
                        sl=slice(lo,lo+32768)
                        V[sl]=vectors[sl].astype(float)@Y
                    info['mu']=mu.tolist()
                    del vectors
                for j in range(polish):
                    for axis in range(V.shape[1]):
                        V[:,axis]-=B(A@V[:,axis])
                if variant=='pilot_direct':
                    M=DenseCorrection(A,B,V)
                else:
                    bins=max(48,3*(n+1)//16)
                    kind='weighted' if variant=='old' else 'hats'
                    if variant=='bounded':
                        from relax import bank
                        raw=bank(A,V,bins,basis_steps)
                    else:
                        raw=s.v1.lift_basis(V,bins,kind)
                    Z=s.v1.smooth_basis(A,raw,basis_steps if variant=='jacobi' else 0)
                    M=s.v1.Correction(A,B,Z)
                    info['row_cap']=int(np.diff(Z.indptr).max())
                info['dim']=M.dim
            x,k2,ok,_,_,msg=s.cg(A,M,b,maxiter=1500-k,x0=x,state=None if active else state)
            k+=k2
    elapsed=time.perf_counter()-started
    cpu_s=time.process_time()-cpu
    rr=np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b)
    B.close()
    return dict(info,case=case,variant=variant,length=length,count=count,polish=polish,it=k,basis_steps=basis_steps,
                total=elapsed,cpu=cpu_s,ok=bool(ok and rr<=1e-8),residual=float(rr),message=msg)
