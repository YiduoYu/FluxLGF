"""Untimed boundary check. Displace the separatrices using an asymmetric warp."""
import json
import cluster_probe as study
from method import np,s
from ops_aniso import tensor,assemble

def field(n,family,chi,p,rhs,seed):
    ax,ay,_=p
    axis=np.linspace(0,1,n+2)
    X,Y=np.meshgrid(axis,axis)
    gx=X+ax*np.sin(np.pi*X)/np.pi
    gy=Y+ay*np.sin(np.pi*Y)/np.pi
    vx=-np.sin(2*np.pi*gx)*np.cos(2*np.pi*gy)*(1+ay*np.cos(np.pi*Y))
    vy=np.cos(2*np.pi*gx)*np.sin(2*np.pi*gy)*(1+ax*np.cos(np.pi*X))
    one=np.ones_like(X)
    k=tensor(np.arctan2(vy,vx),chi*one,one)
    xx,yy=X[1:-1,1:-1],Y[1:-1,1:-1]
    A=assemble(*k,1+.2*np.cos(2*np.pi*xx)*np.cos(2*np.pi*yy),1/(n+1))
    b=np.random.default_rng(seed).normal(size=n*n)
    return A,b,None

study.problem=field
for alpha in [.04,.22,.45]:
    case=(383,'asymmetric_four',1e5,[alpha,-.7*alpha,0],3,77681)
    A,b,_=field(*case)
    B=s.v1.Boomer(A,'p8')
    x,it,ok,_,_,_=s.cg(A,B,b)
    residual=float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
    assert ok and residual<=1.e-8
    B.close()
    results={'p8':dict(it=it,residual=residual)}
    for kind in ('first2','all','anchor','random'):
        row=study.run(case,kind)
        assert row['ok']
        results[kind]={k:row.get(k) for k in ('cluster','dim','row_cap','it','residual','mu')}
    print(json.dumps(dict(alpha=alpha,case=case,results=results)),flush=True)
