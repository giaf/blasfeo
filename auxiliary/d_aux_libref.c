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

/*
 * auxiliary functions for LA:REFERENCE (column major)
 *
 * auxiliary/d_aux_lib*.c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../include/blasfeo_common.h"


#define REAL double
#define STRMAT d_strmat_ref
#define STRVEC d_strvec_ref




#define SIZE_STRMAT blasfeo_d_memsize_strmat_ref
#define SIZE_DIAG_STRMAT blasfeo_d_memsize_diag_strmat_ref
#define SIZE_STRVEC blasfeo_d_memsize_strvec_ref

#define CREATE_STRMAT blasfeo_d_create_strmat_ref
#define CREATE_STRVEC blasfeo_d_create_strvec_ref

#define CVT_MAT2STRMAT blasfeo_d_cvt_mat2strmat_ref
#define CVT_TRAN_MAT2STRMAT blasfeo_d_cvt_tran_mat2strmat_ref
#define CVT_VEC2STRVEC blasfeo_d_cvt_vec2strvec_ref
#define CVT_STRMAT2MAT blasfeo_d_cvt_strmat2mat_ref
#define CVT_TRAN_STRMAT2MAT blasfeo_d_cvt_tran_strmat2mat_ref
#define CVT_STRVEC2VEC blasfeo_d_cvt_strvec2vec_ref
#define CAST_MAT2STRMAT blasfeo_d_cast_mat2strmat_ref
#define CAST_DIAG_MAT2STRMAT blasfeo_d_cast_diag_mat2strmat_ref
#define CAST_VEC2VECMAT blasfeo_d_cast_vec2vecmat_ref

#define GECP_LIBSTR blasfeo_dgecp_ref
#define GESC_LIBSTR blasfeo_dgesc_ref
#define GECPSC_LIBSTR blasfeo_dgecpsc_ref



// TESTING
#include "x_aux_lib.c"
