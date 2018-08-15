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
#define STRMAT blasfeo_dmat_ref
#define STRVEC blasfeo_dvec_ref




#define SIZE_STRMAT blasfeo_memsize_dmat_ref
#define SIZE_DIAG_STRMAT blasfeo_memsize_diag_dmat_ref
#define SIZE_STRVEC blasfeo_memsize_dvec_ref

#define CREATE_STRMAT blasfeo_create_dmat_ref
#define CREATE_STRVEC blasfeo_create_dvec_ref

#define CVT_MAT2STRMAT blasfeo_pack_dmat_ref
#define CVT_TRAN_MAT2STRMAT blasfeo_pack_tran_dmat_ref
#define CVT_VEC2STRVEC blasfeo_pack_dvec_ref
#define CVT_STRMAT2MAT blasfeo_unpack_dmat_ref
#define CVT_TRAN_STRMAT2MAT blasfeo_unpack_tran_dmat_ref
#define CVT_STRVEC2VEC blasfeo_unpack_dvec_ref
#define CAST_MAT2STRMAT blasfeo_d_cast_mat2strmat_ref
#define CAST_DIAG_MAT2STRMAT blasfeo_d_cast_diag_mat2strmat_ref
#define CAST_VEC2VECMAT blasfeo_d_cast_vec2vecmat_ref

#define GEAD_LIBSTR blasfeo_dgead_ref
#define GECP_LIBSTR blasfeo_dgecp_ref
#define GESC_LIBSTR blasfeo_dgesc_ref
#define GECPSC_LIBSTR blasfeo_dgecpsc_ref



// TESTING_MODE
#include "x_aux_lib.c"
