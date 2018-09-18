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

#define FABS fabs
#define SQRT sqrt

#if defined(LA_BLAS_WRAPPER)
#if defined(REF_BLAS_BLIS)
#include "../include/d_blas_64.h"
#elif defined(REF_BLAS_MKL)
#include "mkl.h"
#else
#include "../include/d_blas.h"
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



#define REAL double

#define STRMAT blasfeo_dmat
#define STRVEC blasfeo_dvec

#define AXPY_LIBSTR blasfeo_daxpy
#define AXPBY_LIBSTR blasfeo_daxpby
#define VECMUL_LIBSTR blasfeo_dvecmul
#define VECMULACC_LIBSTR blasfeo_dvecmulacc
#define VECMULDOT_LIBSTR blasfeo_dvecmuldot
#define DOT_LIBSTR blasfeo_ddot
#define ROTG_LIBSTR blasfeo_drotg
#define COLROT_LIBSTR blasfeo_dcolrot
#define ROWROT_LIBSTR blasfeo_drowrot

#define AXPY daxpy_
#define COPY dcopy_
#define SCAL dscal_
#define ROT drot_
#define ROTG drotg_


#include "x_blas1_lib.c"
