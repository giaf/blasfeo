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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



// y = y + alpha*x, with increments equal to 1
void saxpy_lib(int kmax, float alpha, float *x, float *y)
	{

	int ii;

	ii = 0;
	for( ; ii<kmax-3; ii+=4)
		{
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		y[ii+1] = y[ii+1] + alpha*x[ii+1];
		y[ii+2] = y[ii+2] + alpha*x[ii+2];
		y[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
	for( ; ii<kmax; ii++)
		{
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		}

	return;

	}



// z = y, y = y + alpha*x, with increments equal to 1
void saxpy_bkp_lib(int kmax, float alpha, float *x, float *y, float *z)
	{

	int ii;

	ii = 0;
	for( ; ii<kmax-3; ii+=4)
		{
		z[ii+0] = y[ii+0];
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		z[ii+1] = y[ii+1];
		y[ii+1] = y[ii+1] + alpha*x[ii+1];
		z[ii+2] = y[ii+2];
		y[ii+2] = y[ii+2] + alpha*x[ii+2];
		z[ii+3] = y[ii+3];
		y[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
	for( ; ii<kmax; ii++)
		{
		z[ii+0] = y[ii+0];
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		}

	return;

	}



#if defined(LA_HIGH_PERFORMANCE)



void saxpy_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi)
	{
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	saxpy_lib(m, alpha, x, y);
	return;
	}



void saxpy_bkp_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
	saxpy_bkp_lib(m, alpha, x, y, z);
	return;
	}



#else

#error : wrong LA choice

#endif

