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
#include <math.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"


#define REAL double
#define XMAT blasfeo_dmat_ref
#define XVEC blasfeo_dvec_ref

#define GEMM_NN    blasfeo_dgemm_nn_ref
#define GEMM_NT    blasfeo_dgemm_nt_ref
#define GEMM_TN    blasfeo_dgemm_tn_ref
#define GEMM_TT    blasfeo_dgemm_tt_ref

#define SYRK_LN    blasfeo_dsyrk_ln_ref
#define SYRK_LN_MN blasfeo_dsyrk_ln_mn_ref
#define SYRK_LT    blasfeo_dsyrk_lt_ref
#define SYRK_UN    blasfeo_dsyrk_un_ref
#define SYRK_UT    blasfeo_dsyrk_ut_ref

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
