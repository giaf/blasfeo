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
#include "s_blas.h"
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT s_strmat



#define GEMM_NN_LIBSTR sgemm_nn_libstr
#define GEMM_NT_LIBSTR sgemm_nt_libstr
#define SYRK_LN_LIBSTR ssyrk_ln_libstr
#define TRMM_RLNN_LIBSTR strmm_rlnn_libstr
#define TRMM_RUTN_LIBSTR strmm_rutn_libstr
#define TRSM_LLNU_LIBSTR strsm_llnu_libstr
#define TRSM_LUNN_LIBSTR strsm_lunn_libstr
#define TRSM_RLTN_LIBSTR strsm_rltn_libstr
#define TRSM_RLTU_LIBSTR strsm_rltu_libstr
#define TRSM_RUTN_LIBSTR strsm_rutn_libstr



#define COPY scopy_
#define GEMM sgemm_
#define SYRK ssyrk_
#define TRMM strmm_
#define TRSM strsm_



#include "x_blas3_lib.c"

