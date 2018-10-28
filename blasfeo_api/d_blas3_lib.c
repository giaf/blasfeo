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

#if defined(LA_BLAS_WRAPPER)
#if defined(REF_BLAS_BLIS)
#include "blis.h"
#elif defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "../include/d_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define XMAT blasfeo_dmat

#define GEMM_NN blasfeo_dgemm_nn
#define GEMM_NT blasfeo_dgemm_nt
#define GEMM_TN blasfeo_dgemm_tn
#define GEMM_TT blasfeo_dgemm_tt
#define SYRK_LN blasfeo_dsyrk_ln
#define SYRK_LN_MN blasfeo_dsyrk_ln_mn
#define SYRK_LT blasfeo_dsyrk_lt
#define SYRK_UN blasfeo_dsyrk_un
#define SYRK_UT blasfeo_dsyrk_ut
#define TRMM_RLNN blasfeo_dtrmm_rlnn
#define TRMM_RUTN blasfeo_dtrmm_rutn
#define TRSM_LLNU blasfeo_dtrsm_llnu
#define TRSM_LUNN blasfeo_dtrsm_lunn
#define TRSM_RLTN blasfeo_dtrsm_rltn
#define TRSM_RLTU blasfeo_dtrsm_rltu
#define TRSM_RUTN blasfeo_dtrsm_rutn

#define COPY dcopy_
#define GEMM dgemm_
#define SYRK dsyrk_
#define TRMM dtrmm_
#define TRSM dtrsm_



#include "x_blas3_lib.c"
