/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#if defined(LA_BLAS)
#if defined(REF_BLAS_BLIS)
#include "s_blas_64.h"
#elif defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT blasfeo_smat

#define GEMM_NN_LIBSTR blasfeo_sgemm_nn
#define GEMM_NT_LIBSTR blasfeo_sgemm_nt
#define SYRK_LN_LIBSTR blasfeo_ssyrk_ln
#define SYRK_LN_MN_LIBSTR blasfeo_ssyrk_ln_mn
#define TRMM_RLNN_LIBSTR blasfeo_strmm_rlnn
#define TRMM_RUTN_LIBSTR blasfeo_strmm_rutn
#define TRSM_LLNU_LIBSTR blasfeo_strsm_llnu
#define TRSM_LUNN_LIBSTR blasfeo_strsm_lunn
#define TRSM_RLTN_LIBSTR blasfeo_strsm_rltn
#define TRSM_RLTU_LIBSTR blasfeo_strsm_rltu
#define TRSM_RUTN_LIBSTR blasfeo_strsm_rutn

#define COPY scopy_
#define GEMM sgemm_
#define SYRK ssyrk_
#define TRMM strmm_
#define TRSM strsm_



#include "x_blas3_lib.c"

