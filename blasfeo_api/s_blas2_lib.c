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

#if defined(LA_EXTERNAL_BLAS_WRAPPER)
#if defined(EXTERNAL_BLAS_BLIS)
#include "blis.h"
#elif defined(EXTERNAL_BLAS_MKL)
#include "mkl.h"
#else
#include "../include/s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"



#define REAL float

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec

#define GEMV_N_LIBSTR blasfeo_sgemv_n
#define GEMV_NT_LIBSTR blasfeo_sgemv_nt
#define GEMV_T_LIBSTR blasfeo_sgemv_t
#define SYMV_L_LIBSTR blasfeo_ssymv_l
#define TRMV_LNN_LIBSTR blasfeo_strmv_lnn
#define TRMV_LTN_LIBSTR blasfeo_strmv_ltn
#define TRMV_UNN_LIBSTR blasfeo_strmv_unn
#define TRMV_UTN_LIBSTR blasfeo_strmv_utn
#define TRSV_LNN_LIBSTR blasfeo_strsv_lnn
#define TRSV_LNN_MN_LIBSTR blasfeo_strsv_lnn_mn
#define TRSV_LNU_LIBSTR blasfeo_strsv_lnu
#define TRSV_LTN_LIBSTR blasfeo_strsv_ltn
#define TRSV_LTN_MN_LIBSTR blasfeo_strsv_ltn_mn
#define TRSV_LTU_LIBSTR blasfeo_strsv_ltu
#define TRSV_UNN_LIBSTR blasfeo_strsv_unn
#define TRSV_UTN_LIBSTR blasfeo_strsv_utn

#define COPY scopy_
#define GEMV sgemv_
#define SYMV ssymv_
#define TRMV strmv_
#define TRSV strsv_



#include "x_blas2_lib.c"

