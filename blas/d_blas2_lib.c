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

#if defined(LA_BLAS)
#if defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "d_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define STRMAT blasfeo_dmat
#define STRVEC blasfeo_dvec

#define GEMV_N_LIBSTR dgemv_n_libstr
#define GEMV_NT_LIBSTR dgemv_nt_libstr
#define GEMV_T_LIBSTR dgemv_t_libstr
#define SYMV_L_LIBSTR dsymv_l_libstr
#define TRMV_LNN_LIBSTR dtrmv_lnn_libstr
#define TRMV_LTN_LIBSTR dtrmv_ltn_libstr
#define TRMV_UNN_LIBSTR dtrmv_unn_libstr
#define TRMV_UTN_LIBSTR dtrmv_utn_libstr
#define TRSV_LNN_LIBSTR dtrsv_lnn_libstr
#define TRSV_LNN_MN_LIBSTR dtrsv_lnn_mn_libstr
#define TRSV_LNU_LIBSTR dtrsv_lnu_libstr
#define TRSV_LTN_LIBSTR dtrsv_ltn_libstr
#define TRSV_LTN_MN_LIBSTR dtrsv_ltn_mn_libstr
#define TRSV_LTU_LIBSTR dtrsv_ltu_libstr
#define TRSV_UNN_LIBSTR dtrsv_unn_libstr
#define TRSV_UTN_LIBSTR dtrsv_utn_libstr

#define COPY dcopy_
#define GEMV dgemv_
#define SYMV dsymv_
#define TRMV dtrmv_
#define TRSV dtrsv_



#include "x_blas2_lib.c"
