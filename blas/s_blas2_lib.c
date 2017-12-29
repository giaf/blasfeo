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
#if defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT s_strmat
#define STRVEC s_strvec

#define GEMV_N_LIBSTR sgemv_n_libstr
#define GEMV_NT_LIBSTR sgemv_nt_libstr
#define GEMV_T_LIBSTR sgemv_t_libstr
#define SYMV_L_LIBSTR ssymv_l_libstr
#define TRMV_LNN_LIBSTR strmv_lnn_libstr
#define TRMV_LTN_LIBSTR strmv_ltn_libstr
#define TRMV_UNN_LIBSTR strmv_unn_libstr
#define TRMV_UTN_LIBSTR strmv_utn_libstr
#define TRSV_LNN_LIBSTR strsv_lnn_libstr
#define TRSV_LNN_MN_LIBSTR strsv_lnn_mn_libstr
#define TRSV_LNU_LIBSTR strsv_lnu_libstr
#define TRSV_LTN_LIBSTR strsv_ltn_libstr
#define TRSV_LTN_MN_LIBSTR strsv_ltn_mn_libstr
#define TRSV_LTU_LIBSTR strsv_ltu_libstr
#define TRSV_UNN_LIBSTR strsv_unn_libstr
#define TRSV_UTN_LIBSTR strsv_utn_libstr

#define COPY scopy_
#define GEMV sgemv_
#define SYMV ssymv_
#define TRMV strmv_
#define TRSV strsv_



#include "x_blas2_lib.c"

