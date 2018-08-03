// BLASFEO routines
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_blas.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"

// BLASFEO External dependencies
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_timing.h"

// BLASFEO LA:REFERENCE routines
#include "../include/blasfeo_s_blas3_ref.h"
#include "../include/blasfeo_s_aux_ref.h"
#include "../include/blasfeo_s_aux_ext_dep_ref.h"

#include "../include/blasfeo_s_aux_test.h"

#define PRECISION Single
#define GECMP_LIBSTR sgecmp_libstr
#define REAL float

#define ZEROS s_zeros
#define FREE s_free

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec
#define ALLOCATE_STRMAT blasfeo_allocate_smat
#define PACK_STRMAT blasfeo_pack_smat
#define FREE_STRMAT blasfeo_free_smat

#define PS S_PS
#define PRINT_STRMAT blasfeo_print_smat

#define STRMAT_REF blasfeo_smat_ref
#define STRVEC_REF blasfeo_svec_ref
#define PRINT_STRMAT_REF blasfeo_print_smat_ref
#define ALLOCATE_STRMAT_REF blasfeo_allocate_smat_ref
#define PACK_STRMAT_REF blasfeo_pack_smat_ref
#define FREE_STRMAT_REF blasfeo_free_smat_ref
