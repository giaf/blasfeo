/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(LA_EXTERNAL_BLAS_WRAPPER)
#if defined(EXTERNAL_BLAS_BLIS)
#include <blis.h>
#elif defined(EXTERNAL_BLAS_MKL)
//#include <mkl.h>
#else
#include "../include/s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec

#define GELQF_PD_DA_LIBSTR blasfeo_sgelqf_pd_da
#define GELQF_PD_LA_LIBSTR blasfeo_sgelqf_pd_la
#define GELQF_PD_LLA_LIBSTR blasfeo_sgelqf_pd_lla
#define GELQF_PD_LIBSTR blasfeo_sgelqf_pd
#define GELQF_LIBSTR blasfeo_sgelqf
#define GELQF_WORK_SIZE_LIBSTR blasfeo_sgelqf_worksize
#define GEQRF_LIBSTR blasfeo_sgeqrf
#define GEQRF_WORK_SIZE_LIBSTR blasfeo_sgeqrf_worksize
#define GETF2_NOPIVOT sgetf2_nopivot
#define GETRF_NOPIVOT_LIBSTR blasfeo_sgetrf_np
#define GETRF_LIBSTR blasfeo_sgetrf_rp
#define POTRF_L_LIBSTR blasfeo_spotrf_l
#define POTRF_L_MN_LIBSTR blasfeo_spotrf_l_mn
#define PSTRF_L_LIBSTR spstrf_l_libstr
#define SYRK_POTRF_LN_LIBSTR blasfeo_ssyrk_spotrf_ln
#define SYRK_POTRF_LN_MN_LIBSTR blasfeo_ssyrk_spotrf_ln_mn

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

