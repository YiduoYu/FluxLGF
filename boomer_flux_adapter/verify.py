"""Small numerical contracts for basis construction, the wrapper and stencil fit."""
from bench import CASES, inputs, flux, np, s
from adapter import FluxAdapter, flux_basis
from recover import recover_flux
from scipy import sparse


def main():
    n=79
    _,family,chi,p,rhs,seed=CASES[3]
    A,b,_=inputs(n,family,chi,p,rhs,seed)
    psi=flux(n,family,p)
    bank,regions=flux_basis(psi,24)
    assert regions==4 and bank.shape==(n*n,24) and np.diff(bank.indptr).max()<=4
    assert np.linalg.eigvalsh((bank.T@bank).toarray()).min()>0
    assert flux_basis(np.ones((9,9)))[0].shape==(49,0)
    small=np.zeros((5,5))
    small[2,2]=1
    tiny,k=flux_basis(small,budget=7)
    assert k==1 and tiny.shape==(9,1)
    B=s.v1.Boomer(A,"p8")
    try:
        M=FluxAdapter(A,B,psi,budget=24)
        # Changing units and origin of flux leaves the correction invariant.
        M2=FluxAdapter(A,B,3*psi+1.5,budget=24)
        rng=np.random.default_rng(66)
        r,t=rng.normal(size=(2,n*n))
        mr,mt=M(r),M(t)
        assert r@mr>0 and t@mt>0
        assert abs(t@mr-r@mt)<2e-10*np.sqrt((t@mt)*(r@mr))
        assert np.linalg.norm(M2(r)-mr)<2e-10*np.linalg.norm(mr)
        assert np.array_equal(FluxAdapter(A,B)(r),B(r))
        x,it,ok,_,_,_=s.cg(A,M,b)
        assert ok and np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b)<1e-8
    finally:
        B.close()
    # A scalar reaction changes the diagonal but cannot rotate the recovered axis.
    v,info=recover_flux(A,n)
    w,_=recover_flux(A+sparse.eye(n*n,format="csr"),n)
    assert v is not None and np.allclose(v,w) and info["defect"]<=.025
    iso=sparse.eye(n*n,format="csr")
    assert recover_flux(iso,n)[0] is None
    wrong=sparse.csr_matrix(([1.],([n+1],[n+4])),shape=A.shape)
    try:
        recover_flux(A+wrong,n)
    except ValueError:
        pass
    else:
        raise AssertionError("sampled nonlocal coupling accepted")
    print("passed: region count, rank, row cap, plateau, flux units, symmetry, positivity, "
          "fallback, audited PCG, reaction invariance, stencil checks")


if __name__=="__main__":
    main()
