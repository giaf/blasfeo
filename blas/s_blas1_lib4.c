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
#include "../include/blasfeo_s_kernel.h"



#if defined(LA_HIGH_PERFORMANCE)



// z = y + alpha*x, with increments equal to 1
void saxpy_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
	int ii;
	ii = 0;
	for( ; ii<m-3; ii+=4)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		z[ii+1] = y[ii+1] + alpha*x[ii+1];
		z[ii+2] = y[ii+2] + alpha*x[ii+2];
		z[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
	for( ; ii<m; ii++)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		}
	return;
	}



void saxpy_bkp_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
	int ii;
	ii = 0;
	for( ; ii<m-3; ii+=4)
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
	for( ; ii<m; ii++)
		{
		z[ii+0] = y[ii+0];
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		}
	return;
	}



// multiply two vectors and compute dot product
float svecmuldot_libstr(int m, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(m<=0)
		return 0.0;

	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;
	int ii;
	float dot = 0.0;

	ii = 0;

	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
	return dot;
	}



#else

#error : wrong LA choice

#endif

