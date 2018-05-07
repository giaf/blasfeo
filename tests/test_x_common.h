#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../include/blasfeo_common.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%-16s= %s\n", #x, STR(x));

#ifndef LA
	#error LA undefined
#endif

#ifndef TARGET
	#error TARGET undefined
#endif

#ifndef PRECISION
	#error PRECISION undefined
#endif

#ifndef MIN_KERNEL_SIZE
	#error MIN_KERNEL_SIZE undefined
#endif

#ifndef ROUTINE
	#error ROUTINE undefined
#endif

#define concatenate(var, post) var ## post
#define string(var) STR(var)

#define REF(fun) concatenate(fun, _ref)

#ifndef VERBOSE
#define VERBOSE 0
#endif


#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


// Collection of macros  and functions inteded to be used to compute compare and check matrices

#if defined(LA_HIGH_PERFORMANCE)
// Panel major element extraction macro
#define MATEL_LIBSTR(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(PS-1)))*(sA)->cn+(aj)*PS+((ai)&(PS-1))])
#define MATEL_LIB(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#elif defined(LA_BLAS_WRAPPER) | defined(LA_REFERENCE)
#define MATEL_LIBSTR(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#else
#error : wrong LA choice
#endif

// Column major element extraction macro
//
#define VECEL_LIBSTR(sa,ai) ((sa)->pa[ai])
#define VECEL_LIB(sa,ai) ((sa)->pa[ai])



int GECMP_LIBSTR(int n, int m, int bi, int bj, struct STRMAT *sB, struct STRMAT_REF *rB, struct STRMAT *sA, struct STRMAT_REF *rA, int* err_i, int* err_j, int debug);
