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

#if defined(LA_BLAS)
#include "d_blas.h"
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double

#define STRMAT d_strmat
#define STRVEC d_strvec

#define GETF2_NOPIVOT dgetf2_nopivot
#define GETRF_NOPIVOT_LIBSTR dgetrf_nopivot_libstr
#define GETRF_LIBSTR dgetrf_libstr
#define POTRF_L_LIBSTR dpotrf_l_libstr
#define POTRF_L_MN_LIBSTR dpotrf_l_mn_libstr
#define SYRK_POTRF_LN_LIBSTR dsyrk_dpotrf_ln_libstr

#define COPY dcopy_
#define GER dger_
#define SCAL dscal_
#define GEMM dgemm_
#define SYRK dsyrk_
#define TRSM dtrsm_
#define GETRF dgetrf_
#define POTRF dpotrf_


#include "x_lapack_lib.c"
