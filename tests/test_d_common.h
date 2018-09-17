// BLASFEO routines
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_blas.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"

// BLASFEO External dependencies
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_timing.h"

// BLASFEO LA:REFERENCE routines
#include "../include/blasfeo_d_aux_ref.h"
#include "../include/blasfeo_d_aux_ext_dep_ref.h"
#include "../include/blasfeo_d_blas3_ref.h"

#include "../include/blasfeo_d_aux_test.h"


#define PRECISION Double
#define GECMP_LIBSTR dgecmp_libstr
#define REAL double

#define ZEROS d_zeros
#define FREE d_free

#define STRMAT blasfeo_dmat
#define STRVEC blasfeo_dvec

#define ALLOCATE_STRMAT blasfeo_allocate_dmat
#define PACK_STRMAT blasfeo_pack_dmat
#define PRINT_STRMAT blasfeo_print_dmat
#define FREE_STRMAT blasfeo_free_dmat

#define PS D_PS


#define STRMAT_REF blasfeo_dmat_ref
#define STRVEC_REF blasfeo_dvec_ref

#define ALLOCATE_STRMAT_REF blasfeo_allocate_dmat_ref
#define PACK_STRMAT_REF blasfeo_pack_dmat_ref
#define PRINT_STRMAT_REF blasfeo_print_dmat_ref
#define FREE_STRMAT_REF blasfeo_free_dmat_ref


#define PRINT_STRMAT_REF blasfeo_print_dmat_ref

