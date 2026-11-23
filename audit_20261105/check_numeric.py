"""Read-only numerical audit of production functions; run with hypre_env Python."""
import sys, json
from pathlib import Path
from fractions import Fraction as Fr
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'regions'), str(ROOT/'solver_challenge')]
from ir_fp128 import DDResidual, two_sum, two_prod
from rsolve import ScalarFDLGF, FDMBox, DeflatedFDLGF
from btfd import BTFD
from bt_elim import schur_ion
from fdlgf_real import background_matrix
from gen_gen import generate

out = {}
rng = np.random.default_rng(20261105)
# Independent rational reference for stored float64 data, including tiny residuals.
n=400
ah=rng.uniform(1e15,1e17,n); xh=rng.uniform(1e-4,1e-3,n)
xl=rng.uniform(-1,1,n)*np.spacing(xh)*0.4
data=np.column_stack([ah,-ah,np.full(n,512.)]).ravel()
indices=np.arange(3*n); indptr=np.arange(0,3*n+1,3)
A=sp.csr_matrix((data,indices,indptr),shape=(n,3*n))
hi=np.repeat(xh,3); lo=np.repeat(xl,3)
b=512*xh
got=DDResidual(A,b)(hi,lo)
exact=np.array([float(Fr(float(b[i]))-sum((Fr(float(data[3*i+j]))*(Fr(float(hi[3*i+j]))+Fr(float(lo[3*i+j]))) for j in range(3)),Fr(0))) for i in range(n)])
out['dd_cancellation']={'relative_error':float(np.linalg.norm(got-exact)/np.linalg.norm(exact)), 'max_abs_error':float(np.max(abs(got-exact)))}
# Use actual generated stencil rows and high/low solution vectors.
A,b=generate(12,1,2,1,0,.02,1)
h=rng.uniform(.0001,.0005,len(b)); l=rng.uniform(-.4,.4,len(b))*np.spacing(h)
rr=DDResidual(A,b)(h,l); ids=rng.choice(len(b),200,replace=False)
ref=[]
for i in ids:
    ref.append(float(Fr(float(b[i]))-sum((Fr(float(v))*(Fr(float(h[j]))+Fr(float(l[j]))) for j,v in zip(A.indices[A.indptr[i]:A.indptr[i+1]],A.data[A.indptr[i]:A.indptr[i+1]])),Fr(0))))
out['dd_generated_rows']={'relative_error':float(np.linalg.norm(rr[ids]-ref)/np.linalg.norm(ref))}
# Scalar balanced identity with moderate conditioning and an actual feature LU.
m=5; M,_=background_matrix(m,2.,3.,.2); M=M+sp.diags(np.where(np.arange(m**3)==62,7.,0.))
P=ScalarFDLGF(M,m,ring=1)
r=rng.standard_normal(m**3)
def Q(v):
    z=np.zeros_like(v); z[P.F]=spsolve(M[P.F][:,P.F],v[P.F]); return z
q=Q(r); z=P.G(r-M@q); expected=q+z-Q(M@z)
out['scalar_balanced_identity']=float(np.linalg.norm(P(r)-expected)/np.linalg.norm(expected))
# Block triangular algebra: replace approximate scalar inverses by exact reference solves.
A,b=generate(4,1,2,1,0,0,1); N=len(b)//3
P=BTFD.__new__(BTFD); P.N=N; P.A12=A[:N,N:2*N]
P.d22=A[N:2*N,N:2*N].diagonal(); P.a23=A[N:2*N,2*N:].diagonal(); P.a32=A[2*N:,N:2*N].diagonal()
S=schur_ion(A[2*N:,2*N:],P.a23,P.a32,P.d22)
P.p3=lambda v:spsolve(S,v); P.p1=lambda v:spsolve(A[:N,:N],v)
B=sp.bmat([[A[:N,:N],P.A12,None],[None,sp.diags(P.d22),sp.diags(P.a23)],[None,sp.diags(P.a32),A[2*N:,2*N:]]]).tocsr()
# Strong scaling here makes residual sensitive; report relative difference to direct solution too.
r=rng.standard_normal(3*N); z=P(r); ref=spsolve(B,r)
out['block_triangular_identity']={'solution_relative_difference':float(np.linalg.norm(z-ref)/np.linalg.norm(ref)), 'scaled_residual':float(np.linalg.norm((r-B@z)/B.diagonal())/np.linalg.norm(r/B.diagonal()))}
# Repeat the algebraic identity on a well-conditioned block matrix so the direct reference is reliable.
N=64; D=np.full(N,5.);c=np.full(N,-1.3);M1,_=background_matrix(4,2.,3.,.1);M3,_=background_matrix(4,4.,2.,0.)
P.N=N;P.A12=-sp.eye(N)*.7;P.d22=D;P.a23=c;P.a32=c
S=schur_ion(M3,c,c,D);P.p3=lambda v:spsolve(S,v);P.p1=lambda v:spsolve(M1,v)
B=sp.bmat([[M1,P.A12,None],[None,sp.diags(D),sp.diags(c)],[None,sp.diags(c),M3]]).tocsr()
r=rng.standard_normal(3*N);ref=spsolve(B,r)
out['block_triangular_moderate']=float(np.linalg.norm(P(r)-ref)/np.linalg.norm(ref))
# H55 coarse-space identity, independently form W^T M W and use a sparse direct reference.
from gen3t import diffusion
m=12;kappa=np.ones((m,m,m))*2.;kappa[3:9,3:9,3:9]=20.
M=diffusion(m,kappa.ravel())+sp.eye(m**3)*3.
DeflatedFDLGF.MIN_REGION=8
DP=DeflatedFDLGF(M,m,ring=0)
F=DP.F;C=DP.C;K=DP.K
Wf=sp.csr_matrix((np.ones(len(F)),(F,np.arange(len(F)))),shape=(m**3,len(F)))
Wc=sp.lil_matrix((m**3,K));Wc[C,:]=DP.VC.T
W=sp.hstack([Wf,Wc.tocsr()]).tocsr();coarse=(W.T@M@W).tocsc()
def QD(v):return W@spsolve(coarse,W.T@v)
r=rng.standard_normal(m**3);q=QD(r);z=DP.G(r-M@q);expected=q+z-QD(M@z)
out['deflated_balanced_identity']={'K':K,'relative_difference':float(np.linalg.norm(DP(r)-expected)/np.linalg.norm(expected))}
# One-cell interior box must include six Dirichlet faces.
mask=np.zeros((5,5,5),bool); mask[2,2,2]=True
G=FDMBox(mask,10.,1.,0.,5,dirichlet=True)
r=mask.ravel().astype(float)
out['singleton_box']={'actual':float(G(r)[62]),'expected':1/61,'ratio':float(G(r)[62]*61)}
# Verify options overrides, directly against production factory.
from petsc4py import PETSc
from run_ir import inner
key='ir_amg_pc_hypre_boomeramg_strong_threshold'
PETSc.Options()[key]=.7
ksp,_=inner(A,1e-6,'amg-3d')
out['amg_option_overwrite']={'requested':.7,'effective_database_value':PETSc.Options().getReal(key)}
ksp.view()
print(json.dumps(out,indent=2))
(Path(__file__).parent/'evidence/numeric.json').write_text(json.dumps(out,indent=2)+'\n')
