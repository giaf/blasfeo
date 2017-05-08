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
#include "d_blas_64.h"
#else
#include "d_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define STRMAT d_strmat

#define GEMM_NN_LIBSTR dgemm_nn_libstr
#define GEMM_NT_LIBSTR dgemm_nt_libstr
#define SYRK_LN_LIBSTR dsyrk_ln_libstr
#define SYRK_LN_MN_LIBSTR dsyrk_ln_mn_libstr
#define TRMM_RLNN_LIBSTR dtrmm_rlnn_libstr
#define TRMM_RUTN_LIBSTR dtrmm_rutn_libstr
#define TRSM_LLNU_LIBSTR dtrsm_llnu_libstr
#define TRSM_LUNN_LIBSTR dtrsm_lunn_libstr
#define TRSM_RLTN_LIBSTR dtrsm_rltn_libstr
#define TRSM_RLTU_LIBSTR dtrsm_rltu_libstr
#define TRSM_RUTN_LIBSTR dtrsm_rutn_libstr

#define COPY dcopy_
#define GEMM dgemm_
#define SYRK dsyrk_
#define TRMM dtrmm_
#define TRSM dtrsm_



#include "x_blas3_lib.c"
