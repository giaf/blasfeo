#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"


// Collection of macros  and functions inteded to be used to compute compare and check matrices

#if defined(LA_HIGH_PERFORMANCE)
// Panel major element extraction macro
#define MATEL_LIBSTR(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(PS-1)))*(sA)->cn+(aj)*PS+((ai)&(PS-1))])
#define MATEL_LIB(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#elif defined(LA_BLAS) | defined(LA_REFERENCE)
#define MATEL_LIBSTR(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#else
#error : wrong LA choice
#endif

// Column major element extraction macro
//
#define VECEL_LIBSTR(sa,ai) ((sa)->pa[ai])
#define VECEL_LIB(sa,ai) ((sa)->pa[ai])



int GECMP_LIBSTR(int n, int m, struct STRMAT *sB, struct STRMAT_REF *rB, struct STRMAT *sA, struct STRMAT_REF *rA, int debug);
