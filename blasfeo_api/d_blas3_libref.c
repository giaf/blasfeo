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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#if defined(LA_REFERENCE)
	#define XMATEL_A(X, Y) pA[(X)+lda*(Y)]
	#define XMATEL_B(X, Y) pB[(X)+ldb*(Y)]
	#define XMATEL_C(X, Y) pC[(X)+ldc*(Y)]
	#define XMATEL_D(X, Y) pD[(X)+ldd*(Y)]
#else
	#define XMATEL_A(X, Y) XMATEL(sA, X, Y)
	#define XMATEL_B(X, Y) XMATEL(sB, X, Y)
	#define XMATEL_C(X, Y) XMATEL(sC, X, Y)
	#define XMATEL_D(X, Y) XMATEL(sD, X, Y)
#endif



#define REAL double
#define XMAT blasfeo_dmat_ref
#define XMATEL MATEL_REF
#define XVEC blasfeo_dvec_ref
#define XVECEL VECEL_REF



#define GEMM_NN    blasfeo_dgemm_nn_ref
#define GEMM_NT    blasfeo_dgemm_nt_ref
#define GEMM_TN    blasfeo_dgemm_tn_ref
#define GEMM_TT    blasfeo_dgemm_tt_ref
#define SYRK_LN    blasfeo_dsyrk_ln_ref
#define SYRK_LN_MN blasfeo_dsyrk_ln_mn_ref
#define SYRK_LT    blasfeo_dsyrk_lt_ref
#define SYRK_UN    blasfeo_dsyrk_un_ref
#define SYRK_UT    blasfeo_dsyrk_ut_ref
#define TRMM_RLNN  blasfeo_dtrmm_rlnn_ref
#define TRMM_RUTN  blasfeo_dtrmm_rutn_ref
#define TRSM_LUNU  blasfeo_dtrsm_lunu_ref
#define TRSM_LUNN  blasfeo_dtrsm_lunn_ref
#define TRSM_LUTU  blasfeo_dtrsm_lutu_ref
#define TRSM_LUTN  blasfeo_dtrsm_lutn_ref
#define TRSM_LLNU  blasfeo_dtrsm_llnu_ref
#define TRSM_LLNN  blasfeo_dtrsm_llnn_ref
#define TRSM_LLTU  blasfeo_dtrsm_lltu_ref
#define TRSM_LLTN  blasfeo_dtrsm_lltn_ref
#define TRSM_RUNU  blasfeo_dtrsm_runu_ref
#define TRSM_RUNN  blasfeo_dtrsm_runn_ref
#define TRSM_RUTU  blasfeo_dtrsm_rutu_ref
#define TRSM_RUTN  blasfeo_dtrsm_rutn_ref
#define TRSM_RLNU  blasfeo_dtrsm_rlnu_ref
#define TRSM_RLNN  blasfeo_dtrsm_rlnn_ref
#define TRSM_RLTU  blasfeo_dtrsm_rltu_ref
#define TRSM_RLTN  blasfeo_dtrsm_rltn_ref



// TESTING_MODE
#include "x_blas3_lib.c"
