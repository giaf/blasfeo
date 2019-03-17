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

