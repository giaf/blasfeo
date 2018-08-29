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

#define FABS fabsf
#define SQRT sqrtf

#if defined(LA_BLAS_WRAPPER)
#if defined(REF_BLAS_BLIS)
#include "../include/s_blas_64.h"
#elif defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "../include/s_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"



#define REAL float

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec

#define AXPY_LIBSTR blasfeo_saxpy
#define AXPBY_LIBSTR blasfeo_saxpby
#define VECMUL_LIBSTR blasfeo_svecmul
#define VECMULACC_LIBSTR blasfeo_svecmulacc
#define VECMULDOT_LIBSTR blasfeo_svecmuldot
#define DOT_LIBSTR blasfeo_sdot
#define ROTG_LIBSTR blasfeo_srotg
#define COLROT_LIBSTR blasfeo_scolrot
#define ROWROT_LIBSTR blasfeo_srowrot

#define AXPY saxpy_
#define COPY scopy_
#define SCAL sscal_
#define ROT srot_
#define ROTG srotg_


#include "x_blas1_lib.c"

