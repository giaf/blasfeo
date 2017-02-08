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
#if defined(REF_BLAS_OPENBLAS)
#include <f77blas.h>
#elif defined(REF_BLAS_BLIS)
#elif defined(REF_BLAS_NETLIB)
#include "s_blas.h"
#elif defined(REF_BLAS_MKL)
#include <mkl_blas.h>
#endif
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"



#if defined(LA_REFERENCE)



void saxpy_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi)
	{
	int ii;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	for(ii=0; ii<m; ii++)
		y[ii] += alpha * x[ii];
	return;
	}



void saxpy_bkp_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{
	int ii;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
	for(ii=0; ii<m; ii++)
		{
		z[ii] = y[ii];
		y[ii] += alpha * x[ii];
		}
	return;
	}



#elif defined(LA_BLAS)



void saxpy_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi)
	{
	int i1 = 1;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
#if defined(REF_BLAS_MKL)
	saxpy(&m, &alpha, x, &i1, y, &i1);
#else
	saxpy_(&m, &alpha, x, &i1, y, &i1);
#endif
	return;
	}



void saxpy_bkp_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{
	int i1 = 1;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
#if defined(REF_BLAS_MKL)
	scopy(&m, y, &i1, z, &i1);
	saxpy(&m, &alpha, x, &i1, y, &i1);
#else
	scopy_(&m, y, &i1, z, &i1);
	saxpy_(&m, &alpha, x, &i1, y, &i1);
#endif
	return;
	}



#else

#error : wrong LA choice

#endif


