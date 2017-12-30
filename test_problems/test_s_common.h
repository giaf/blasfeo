#define PRECISION Single
#define GECMP_LIBSTR sgecmp_libstr
#define REAL float
#define XMAT blasfeo_smat
#define XVEC blasfeo_svec
#define PS S_PS
#define PRINT_XMAT s_print_strmat

#define XMAT_REF blasfeo_smat_ref
#define XVEC_REF blasfeo_svec_ref
#define PRINT_XMAT_REF blasfeo_s_print_strmat_ref

#include "test_x_common.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_aux_test.h"
