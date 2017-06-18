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



#if defined(LA_REFERENCE)



void AXPY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int ii;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		z[ii+1] = y[ii+1] + alpha*x[ii+1];
		z[ii+2] = y[ii+2] + alpha*x[ii+2];
		z[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
	for(; ii<m; ii++)
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
	return;
	}



// multiply two vectors and compute dot product
REAL VECMULDOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		z[ii+1] = x[ii+1] * y[ii+1];
		z[ii+2] = x[ii+2] * y[ii+2];
		z[ii+3] = x[ii+3] * y[ii+3];
		dot += z[ii+0] + z[ii+1] + z[ii+2] + z[ii+3];
		}
	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
	return dot;
	}



// compute dot product of two vectors
REAL DOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		dot += x[ii+0] * y[ii+0];
		dot += x[ii+1] * y[ii+1];
		dot += x[ii+2] * y[ii+2];
		dot += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		dot += x[ii+0] * y[ii+0];
		}
	return dot;
	}



#elif defined(LA_BLAS)



void AXPY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int i1 = 1;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	if(y!=z)
		COPY(&m, y, &i1, z, &i1);
	AXPY(&m, &alpha, x, &i1, z, &i1);
	return;
	}



// multiply two vectors and compute dot product
REAL VECMULDOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
	return dot;
	}



// compute dot product of two vectors
REAL DOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		dot += x[ii+0] * y[ii+0];
		dot += x[ii+1] * y[ii+1];
		dot += x[ii+2] * y[ii+2];
		dot += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		dot += x[ii+0] * y[ii+0];
		}
	return dot;
	}



#else

#error : wrong LA choice

#endif


