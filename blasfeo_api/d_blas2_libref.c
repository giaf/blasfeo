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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define STRMAT blasfeo_dmat_ref
#define STRVEC blasfeo_dvec_ref

#define GEMV_N_LIBSTR blasfeo_dgemv_n_ref
#define GEMV_NT_LIBSTR blasfeo_dgemv_nt_ref
#define GEMV_T_LIBSTR blasfeo_dgemv_t_ref
#define SYMV_L_LIBSTR blasfeo_dsymv_l_ref
#define TRMV_LNN_LIBSTR blasfeo_dtrmv_lnn_ref
#define TRMV_LTN_LIBSTR blasfeo_dtrmv_ltn_ref
#define TRMV_UNN_LIBSTR blasfeo_dtrmv_unn_ref
#define TRMV_UTN_LIBSTR blasfeo_dtrmv_utn_ref
#define TRSV_LNN_LIBSTR blasfeo_dtrsv_lnn_ref
#define TRSV_LNN_MN_LIBSTR blasfeo_dtrsv_lnn_mn_ref
#define TRSV_LNU_LIBSTR blasfeo_dtrsv_lnu_ref
#define TRSV_LTN_LIBSTR blasfeo_dtrsv_ltn
#define TRSV_LTN_MN_LIBSTR blasfeo_dtrsv_ltn_mn_ref
#define TRSV_LTU_LIBSTR blasfeo_dtrsv_ltu_ref
#define TRSV_UNN_LIBSTR blasfeo_dtrsv_unn_ref
#define TRSV_UTN_LIBSTR blasfeo_dtrsv_utn_ref


#include "x_blas2_lib.c"
