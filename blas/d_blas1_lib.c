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

#define FABS fabs
#define SQRT sqrt

#if defined(LA_BLAS)
#include "d_blas.h"
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



#define REAL double

#define STRMAT d_strmat
#define STRVEC d_strvec

#define AXPY_LIBSTR daxpy_libstr
#define VECMULDOT_LIBSTR dvecmuldot_libstr
#define DOT_LIBSTR ddot_libstr
#define ROTG_LIBSTR drotg_libstr
#define COLROT_LIBSTR dcolrot_libstr
#define ROWROT_LIBSTR drowrot_libstr

#define AXPY daxpy_
#define COPY dcopy_
#define DSCAL dscal_
#define ROT drot_
#define ROTG drotg_


#include "x_blas1_lib.c"
