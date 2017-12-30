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

#define FABS fabsf
#define SQRT sqrtf

#if defined(LA_BLAS)
#if defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"



#define REAL float

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec

#define AXPY_LIBSTR saxpy_libstr
#define AXPBY_LIBSTR saxpby_libstr
#define VECMULACC_LIBSTR svecmulacc_libstr
#define VECMULDOT_LIBSTR svecmuldot_libstr
#define DOT_LIBSTR sdot_libstr
#define ROTG_LIBSTR srotg_libstr
#define COLROT_LIBSTR scolrot_libstr
#define ROWROT_LIBSTR srowrot_libstr

#define AXPY saxpy_
#define COPY scopy_
#define SCAL sscal_
#define ROT srot_
#define ROTG srotg_


#include "x_blas1_lib.c"

