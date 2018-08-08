/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* BLASFEO is free software; you can redistribute it and/or                                        *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* BLASFEO is distributed in the hope that it will be useful,                                      *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with BLASFEO; if not, write to the Free Software                                  *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

/*
 * blas3 functions for LA:REFERENCE (column major)
 *
 * blas/s_blas_lib*.c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_aux.h"


#define REAL float 
#define STRMAT blasfeo_smat_ref
#define STRVEC blasfeo_svec_ref

#define GEMM_NN_LIBSTR    blasfeo_sgemm_nn_ref
#define GEMM_NT_LIBSTR    blasfeo_sgemm_nt_ref

#define SYRK_LN_LIBSTR    blasfeo_ssyrk_ln_ref
#define SYRK_LN_MN_LIBSTR blasfeo_ssyrk_ln_mn_ref

#define TRSM_LLNU_LIBSTR  blasfeo_strsm_llnu_ref
#define TRSM_LUNN_LIBSTR  blasfeo_strsm_lunn_ref
#define TRSM_RLTU_LIBSTR  blasfeo_strsm_rltu_ref
#define TRSM_RLTN_LIBSTR  blasfeo_strsm_rltn_ref
#define TRSM_RUTN_LIBSTR  blasfeo_strsm_rutn_ref
#define TRMM_RUTN_LIBSTR  blasfeo_strmm_rutn_ref
#define TRMM_RLNN_LIBSTR  blasfeo_strmm_rlnn_ref

#define COPY scopy_
#define GEMM sgemm_
#define SYRK ssyrk_
#define TRMM strmm_
#define TRSM strsm_


// TESTING_MODE
#include "x_blas3_lib.c"
