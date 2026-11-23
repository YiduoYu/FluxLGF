/* Why SysPFMG is not in compare_structured.py: stand-alone check, no Python. Two Poisson fields on n^3, solved to 1e-8.
   usage: syspfmg_check n couple cross     cross: 0 no cross-field stencil entries, 1 one point entry, 2 a 7-point cross stencil
   hypre 3.2.0 (conda-forge, hypre_env): cross 0 converges in 33-34 iterations; cross 1 stops at the limit of 50 with residual 0,
   also with couple 0; cross 2 stops the same way or crashes. A usage error here is not ruled out.
   build: clang -O2 syspfmg_check.c -I$ENV/include -L$ENV/lib -lHYPRE -lmpi -Wl,-rpath,$ENV/lib -o syspfmg_check */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "HYPRE_utilities.h"
#include "HYPRE_sstruct_ls.h"
int main(int argc,char**argv){
 MPI_Init(&argc,&argv); HYPRE_Initialize();
 int n=atoi(argv[1]); double cpl=atof(argv[2]); int cross=atoi(argv[3]); int k=2;
 long N=(long)n*n*n; int size=cross==2?14:(cross?8:7);
 HYPRE_Int lo[3]={0,0,0},hi[3]={n-1,n-1,n-1};
 HYPRE_Int off[7][3]={{0,0,0},{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
 HYPRE_SStructGrid g; HYPRE_SStructVariable t[2]={HYPRE_SSTRUCT_VARIABLE_CELL,HYPRE_SSTRUCT_VARIABLE_CELL};
 HYPRE_SStructGridCreate(MPI_COMM_WORLD,3,1,&g); HYPRE_SStructGridSetExtents(g,0,lo,hi); HYPRE_SStructGridSetVariables(g,0,k,t); HYPRE_SStructGridAssemble(g);
 HYPRE_SStructGraph gr; HYPRE_SStructGraphCreate(MPI_COMM_WORLD,g,&gr); HYPRE_SStructGraphSetObjectType(gr,HYPRE_SSTRUCT);
 HYPRE_SStructStencil st[2];
 for(int v=0;v<k;v++){HYPRE_SStructStencilCreate(3,size,&st[v]); for(int e=0;e<7;e++)HYPRE_SStructStencilSetEntry(st[v],e,off[e],v); if(cross==1)HYPRE_SStructStencilSetEntry(st[v],7,off[0],1-v); if(cross==2)for(int e=0;e<7;e++)HYPRE_SStructStencilSetEntry(st[v],7+e,off[e],1-v); HYPRE_SStructGraphSetStencil(gr,0,v,st[v]);}
 HYPRE_SStructGraphAssemble(gr);
 HYPRE_SStructMatrix A; HYPRE_SStructMatrixCreate(MPI_COMM_WORLD,gr,&A); HYPRE_SStructMatrixSetObjectType(A,HYPRE_SSTRUCT); HYPRE_SStructMatrixInitialize(A);
 double *val=malloc(sizeof(double)*N*size); HYPRE_Int ent[14]={0,1,2,3,4,5,6,7,8,9,10,11,12,13};
 long c=0; for(int z=0;z<n;z++)for(int y=0;y<n;y++)for(int x=0;x<n;x++,c++){ double *w=val+c*size; int m=0;
   w[1]=x>0?-1:0; w[2]=x<n-1?-1:0; w[3]=y>0?-1:0; w[4]=y<n-1?-1:0; w[5]=z>0?-1:0; w[6]=z<n-1?-1:0;
   for(int e=1;e<7;e++)m+=w[e]!=0; w[0]=m+0.1+fabs(cpl); if(cross)w[7]=-fabs(cpl); if(cross==2)for(int e=8;e<14;e++)w[e]=0;}
 for(int v=0;v<k;v++)HYPRE_SStructMatrixSetBoxValues(A,0,lo,hi,v,size,ent,val);
 HYPRE_SStructMatrixAssemble(A);
 HYPRE_SStructVector b,x; HYPRE_SStructVectorCreate(MPI_COMM_WORLD,g,&b); HYPRE_SStructVectorCreate(MPI_COMM_WORLD,g,&x);
 HYPRE_SStructVectorSetObjectType(b,HYPRE_SSTRUCT); HYPRE_SStructVectorSetObjectType(x,HYPRE_SSTRUCT);
 HYPRE_SStructVectorInitialize(b); HYPRE_SStructVectorInitialize(x);
 double *r=malloc(sizeof(double)*N); for(long i=0;i<N;i++)r[i]=1.0+((i*37)%11)/11.0;
 double *zz=calloc(N,sizeof(double));
 for(int v=0;v<k;v++){HYPRE_SStructVectorSetBoxValues(b,0,lo,hi,v,r);HYPRE_SStructVectorSetBoxValues(x,0,lo,hi,v,zz);}
 HYPRE_SStructVectorAssemble(b); HYPRE_SStructVectorAssemble(x);
 HYPRE_SStructSolver s; HYPRE_SStructSysPFMGCreate(MPI_COMM_WORLD,&s); HYPRE_SStructSysPFMGSetMaxIter(s,50); HYPRE_SStructSysPFMGSetTol(s,1e-8); HYPRE_SStructSysPFMGSetLogging(s,1);
 HYPRE_SStructSysPFMGSetup(s,A,b,x); HYPRE_SStructSysPFMGSolve(s,A,b,x);
 HYPRE_Int its; double res; HYPRE_SStructSysPFMGGetNumIterations(s,&its); HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(s,&res);
 printf("n %d couple %g cross %d: its %d res %.2e\n",n,cpl,cross,(int)its,res);
 MPI_Finalize(); return 0;}
