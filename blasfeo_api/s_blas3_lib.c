/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
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
#define XMAT blasfeo_smat
#define XVEC blasfeo_svec

#define GEMM_NN blasfeo_sgemm_nn
#define GEMM_NT blasfeo_sgemm_nt
#define GEMM_TN blasfeo_sgemm_tn
#define GEMM_TT blasfeo_sgemm_tt
#define SYRK_LN blasfeo_ssyrk_ln
#define SYRK_LN_MN blasfeo_ssyrk_ln_mn
#define SYRK_LT blasfeo_ssyrk_lt
#define SYRK_UN blasfeo_ssyrk_un
#define SYRK_UT blasfeo_ssyrk_ut
#define TRMM_RLNN blasfeo_strmm_rlnn
#define TRMM_RUTN blasfeo_strmm_rutn
#define TRSM_LLNN blasfeo_strsm_llnn
#define TRSM_LLNU blasfeo_strsm_llnu
#define TRSM_LLTN blasfeo_strsm_lltn
#define TRSM_LLTU blasfeo_strsm_lltu
#define TRSM_LUNN blasfeo_strsm_lunn
#define TRSM_LUNU blasfeo_strsm_lunu
#define TRSM_LUTN blasfeo_strsm_lutn
#define TRSM_LUTU blasfeo_strsm_lutu
#define TRSM_RLNN blasfeo_strsm_rlnn
#define TRSM_RLNU blasfeo_strsm_rlnu
#define TRSM_RLTN blasfeo_strsm_rltn
#define TRSM_RLTU blasfeo_strsm_rltu
#define TRSM_RUNN blasfeo_strsm_runn
#define TRSM_RUNU blasfeo_strsm_runu
#define TRSM_RUTN blasfeo_strsm_rutn
#define TRSM_RUTU blasfeo_strsm_rutu

#define COPY scopy_
#define GEMM sgemm_
#define SYRK ssyrk_
#define TRMM strmm_
#define TRSM strsm_



#include "x_blas3_lib.c"

