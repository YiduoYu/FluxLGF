"""Untimed fine-grid Ritz projection checks. Run after timing campaigns."""
import json
from bench import inputs, flux, np, s
from adapter import flux_basis
from recover import recover_flux
from diagnostics import project


def check(family, params):
    n=127
    A,_,_=inputs(n,family,1e5,params,3,51851)
    B=s.v1.Boomer(A,"p8")
    try:
        modes,info=s.coordinates(A,B,m=8,fixed_steps=128)
        psi=flux(n,family,params)
        fitted,meta=recover_flux(A,n)
        rows={}
        for method,coord,degree,separate in (
                ("pooled",psi,3,False),("linear",psi,1,True),
                ("flux",psi,3,True),("recover",fitted,3,True)):
            if coord is None:
                rows[method]={"projection_error":[1.]*modes.shape[1],"columns":0}
                continue
            Z,_=flux_basis(coord,budget=32,degree=degree,separate=separate)
            M=s.v1.Correction(A,B,Z)
            rows[method]={"projection_error":project(M,A,modes).tolist(),"columns":Z.shape[1]}
        return dict(family=family,n=n,params=params,ritz_residual=info["ritz_residual"],
                    eigenvalues=info["mu"],fit=meta,methods=rows)
    finally:
        B.close()


if __name__=="__main__":
    print(json.dumps([check("four_cells",[.61,.33,.27]),check("six_cells",[.22,-.31,0])],indent=2))
