/* Rowwise constrained minimization of half trace(Z^T A Z), one coordinate.
   left[i] is a fixed interval. The omitted zero knot never has a column.
   Return 0 on success. CSR inputs have been checked by the Python wrapper. */
#include <math.h>
#include <stdlib.h>
int relax_weights(int n, const int *row, const int *col, const double *a,
                  const int *left, int zero, int steps, double *weight) {
    double *next = malloc(sizeof(double)*n);
    double *bound = malloc(sizeof(double)*n);
    if (!next || !bound) { free(next); free(bound); return 1; }
    for (int i=0;i<n;i++) {
        double sum=0;
        for(int k=row[i];k<row[i+1];k++) sum+=fabs(a[k]);
        const int l=left[i], r=l+1;
        bound[i]=sum*((l!=zero)+(r!=zero));
    }
    double *current=weight, *other=next;
    for(int step=0;step<steps;step++) {
        for(int i=0;i<n;i++) {
            const int l=left[i], r=l+1;
            double gradient=0;
            for(int k=row[i];k<row[i+1];k++) {
                const int j=col[k], lj=left[j], rj=lj+1;
                const double f=current[j];
                const int dl=(r!=zero && r==lj)-(l!=zero && l==lj);
                const int dr=(r!=zero && r==rj)-(l!=zero && l==rj);
                gradient+=a[k]*(dl*(1-f)+dr*f);
            }
            double proposed=current[i]-gradient/bound[i];
            other[i]=fmax(0.,fmin(1.,proposed));
        }
        double *tmp=current; current=other; other=tmp;
    }
    if(current!=weight) for(int i=0;i<n;i++) weight[i]=current[i];
    free(next);free(bound);return 0;
}
