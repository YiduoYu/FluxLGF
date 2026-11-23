"""Exploratory joint-coordinate P1 interpolation; NOT part of the locked benchmark.

Select centers on coarse spectral coordinates, include their convex hull and origin,
triangulate, interpolate fine spectral coordinates, then omit the origin basis.
Two-mode span is retained; homogeneous boundary coordinate 0 maps to zero.
"""
import study as e
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull,Delaunay

def joint_basis(V,W,target_dim):
    assert V.shape[1]==W.shape[1]==2
    pool=np.vstack((np.zeros((1,2)),W))
    _,indices=np.unique(pool,axis=0,return_index=True)
    pool=pool[np.sort(indices)] # preserve original node ordering, even after rotations
    scale=float(np.max(np.linalg.norm(pool,axis=1)))
    pool=pool/scale; points=V/scale
    hull=sorted(ConvexHull(pool).vertices.tolist())
    chosen=[0]+[i for i in hull if i!=0]
    target=max(target_dim+1,len(chosen)) # origin basis is later omitted
    distances=np.full(len(pool),np.inf)
    for i in chosen:distances=np.minimum(distances,np.sum((pool-pool[i])**2,axis=1))
    while len(chosen)<min(target,len(pool)):
        i=int(np.argmax(distances))
        if distances[i]<1e-28:break
        chosen.append(i);distances=np.minimum(distances,np.sum((pool-pool[i])**2,axis=1))
    centers=pool[chosen]
    tri=Delaunay(centers)
    triangles=centers[tri.simplices];edges=triangles[:,1:]-triangles[:,:1]
    area2=abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])
    maxedge2=np.maximum(np.max(np.sum(edges**2,axis=2),axis=1),np.sum((triangles[:,2]-triangles[:,1])**2,axis=1))
    quality=area2/np.maximum(maxedge2,1e-300)
    simplex=tri.find_simplex(points,tol=1e-10)
    outside=int(np.sum(simplex<0))
    if outside:raise RuntimeError(f'{outside} fine coordinates outside selected coarse convex hull')
    tf=tri.transform[simplex]
    first=np.einsum('nij,nj->ni',tf[:,:2,:],points-tf[:,2,:])
    weights=np.column_stack((first,1-first.sum(axis=1)))
    if weights.min()<-1e-8:raise RuntimeError('Unexpected barycentric extrapolation')
    weights=np.maximum(weights,0);weights/=weights.sum(axis=1)[:,None]
    cc=tri.simplices[simplex].ravel();rr=np.repeat(np.arange(len(V)),3);ww=weights.ravel()
    keep=(cc!=0)&(abs(ww)>1e-14)
    Z=sp.csr_matrix((ww[keep],(rr[keep],cc[keep]-1)),shape=(len(V),len(centers)-1))
    # Verify affine coordinate reproduction before dropping empty columns/scaling.
    reproduction=float(np.linalg.norm(Z@centers[1:]-points)/np.linalg.norm(points))
    Z=e.s.v1.smooth_basis(None,Z,0)
    return Z,dict(target_dim=target_dim,selected_centers=len(centers),hull_vertices=len(hull),
                  dim=Z.shape[1],max_row_nnz=int(np.diff(Z.indptr).max()),
                  coordinate_reproduction_error=reproduction,selected_pool_indices=chosen,
                  minimum_triangle_quality=float(quality.min()),
                  percentile1_triangle_quality=float(np.percentile(quality,1)),
                  outside_hull=outside)

if __name__=='__main__':
    import json
    # Numerical implementation check, not performance evidence.
    A,b,u=e.s.problem('circ',15,1e3,[.5]*3)
    B=e.s.v1.Boomer(A,'p8');W,_=e.s.coordinates(A,B,m=2,fixed_steps=32)
    Z,info=joint_basis(W,W,24);M=e.s.v1.Correction(A,B,Z)
    dense=np.column_stack([M(v) for v in np.eye(len(b))])
    symmetry=np.linalg.norm(dense-dense.T)/np.linalg.norm(dense)
    mineig=np.linalg.eigvalsh((dense+dense.T)/2)[0]
    x,it,ok,_,_,_=e.s.cg(A,M,b,tol=1e-10)
    assert info['max_row_nnz']<=3 and info['coordinate_reproduction_error']<1e-9
    assert symmetry<1e-9 and mineig>0 and ok and np.linalg.norm(x-u)/np.linalg.norm(u)<1e-7
    print(json.dumps(dict(info,passed=True,symmetry=symmetry,min_eigenvalue=mineig,iterations=it)))
    B.close()
