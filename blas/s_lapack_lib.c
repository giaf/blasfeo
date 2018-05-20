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

#if defined(LA_BLAS_WRAPPER)
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
#define STRVEC blasfeo_svec

#define GELQF_PD_DA_LIBSTR blasfeo_sgelqf_pd_da
#define GELQF_PD_LA_LIBSTR blasfeo_sgelqf_pd_la
#define GELQF_PD_LIBSTR blasfeo_sgelqf_pd
#define GELQF_LIBSTR blasfeo_sgelqf
#define GELQF_WORK_SIZE_LIBSTR blasfeo_sgelqf_worksize
#define GEQRF_LIBSTR blasfeo_sgeqrf
#define GEQRF_WORK_SIZE_LIBSTR blasfeo_sgeqrf_worksize
#define GETF2_NOPIVOT sgetf2_nopivot
#define GETRF_NOPIVOT_LIBSTR blasfeo_sgetrf_nopivot
#define GETRF_LIBSTR blasfeo_sgetrf_rowpivot
#define POTRF_L_LIBSTR blasfeo_spotrf_l
#define POTRF_L_MN_LIBSTR blasfeo_spotrf_l_mn
#define PSTRF_L_LIBSTR spstrf_l_libstr
#define SYRK_POTRF_LN_LIBSTR blasfeo_ssyrk_spotrf_ln

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

