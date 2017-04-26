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
#include <math.h>

#if defined(LA_BLAS)
#if defined(REF_BLAS_BLIS)
#include "s_blas_64.h"
#else
#include "s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT s_strmat
#define STRVEC s_strvec

#define GELQF_LIBSTR sgelqf_libstr
#define GELQF_WORK_SIZE_LIBSTR sgelqf_work_size_libstr
#define GEQRF_LIBSTR sgeqrf_libstr
#define GEQRF_WORK_SIZE_LIBSTR sgeqrf_work_size_libstr
#define GETF2_NOPIVOT sgetf2_nopivot
#define GETRF_NOPIVOT_LIBSTR sgetrf_nopivot_libstr
#define GETRF_LIBSTR sgetrf_libstr
#define POTRF_L_LIBSTR spotrf_l_libstr
#define POTRF_L_MN_LIBSTR spotrf_l_mn_libstr
#define SYRK_POTRF_LN_LIBSTR ssyrk_spotrf_ln_libstr

#define COPY scopy_
#define GELQF sgelqf_
#define GEMM sgemm_
#define GER sger_
#define GEQRF sgeqrf_
#define GEQR2 sgeqr2_
#define GETRF sgetrf_
#define POTRF spotrf_
#define SCAL sscal_
#define SYRK ssyrk_
#define TRSM strsm_


#include "x_lapack_lib.c"

