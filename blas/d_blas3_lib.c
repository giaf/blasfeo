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

#if defined(LA_BLAS_WRAPPER)
#if defined(REF_BLAS_BLIS)
#include "d_blas_64.h"
#elif defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "d_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define STRMAT blasfeo_dmat

#define GEMM_NN_LIBSTR blasfeo_dgemm_nn
#define GEMM_NT_LIBSTR blasfeo_dgemm_nt
#define SYRK_LN_LIBSTR blasfeo_dsyrk_ln
#define SYRK_LN_MN_LIBSTR blasfeo_dsyrk_ln_mn
#define TRMM_RLNN_LIBSTR blasfeo_dtrmm_rlnn
#define TRMM_RUTN_LIBSTR blasfeo_dtrmm_rutn
#define TRSM_LLNU_LIBSTR blasfeo_dtrsm_llnu
#define TRSM_LUNN_LIBSTR blasfeo_dtrsm_lunn
#define TRSM_RLTN_LIBSTR blasfeo_dtrsm_rltn
#define TRSM_RLTU_LIBSTR blasfeo_dtrsm_rltu
#define TRSM_RUTN_LIBSTR blasfeo_dtrsm_rutn

#define COPY dcopy_
#define GEMM dgemm_
#define SYRK dsyrk_
#define TRMM dtrmm_
#define TRSM dtrsm_



#include "x_blas3_lib.c"
