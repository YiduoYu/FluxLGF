from method import s,np,sp
from pathlib import Path
import ctypes
from numpy.ctypeslib import ndpointer
LIB=ctypes.CDLL(str(Path(__file__).resolve().parent/'libknot.so'))
integer=ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
real=ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
LIB.relax_weights.argtypes=[ctypes.c_int,integer,integer,real,integer,ctypes.c_int,ctypes.c_int,real]
LIB.relax_weights.restype=ctypes.c_int

def bank(A,V,bins,steps):
    blocks=[]
    for v in V.T:
        v=v/max(abs(v))
        lo,hi=min(0.,v.min()),max(0.,v.max())
        t=(v-lo)/(hi-lo)*bins
        j=np.minimum(np.floor(t).astype(np.int32),bins-1)
        f=np.ascontiguousarray(t-j,dtype=np.float64)
        zero=round(-lo/(hi-lo)*bins)
        if steps and LIB.relax_weights(A.shape[0],A.indptr,A.indices,A.data,j,zero,steps,f):
            raise MemoryError('Failed to allocate weight buffers')
        cc=np.column_stack((j,j+1)).ravel()
        ww=np.column_stack((1-f,f)).ravel()
        keep=(cc!=zero)&(ww>1e-14)
        block=sp.csr_matrix((ww[keep],(np.repeat(np.arange(len(v)),2)[keep],cc[keep])),shape=(len(v),bins+1))
        blocks.append(block[:,np.asarray(block.getnnz(axis=0)).ravel()>0])
    return sp.hstack(blocks,format='csr')
