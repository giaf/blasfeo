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
#include "../include/blasfeo_s_aux.h"



#if defined(LA_HIGH_PERFORMANCE)



void sgemv_n_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(m<0)
		return;

	const int bs = 8;

	int i;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;

	i = 0;
	// clean up at the beginning
	if(ai%bs!=0)
		{
		kernel_sgemv_n_8_gen_lib8(n, &alpha, pA, x, &beta, y-ai%bs, z-ai%bs, ai%bs, m+ai%bs);
		pA += bs*sda;
		y += 8 - ai%bs;
		z += 8 - ai%bs;
		m -= 8 - ai%bs;
		}
	// main loop
	for( ; i<m-7; i+=8)
		{
		kernel_sgemv_n_8_lib8(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i]);
		}
	if(i<m)
		{
		kernel_sgemv_n_8_vs_lib8(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i], m-i);
		}
		
	return;

	}



void sgemv_t_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(n<=0)
		return;
	
	const int bs = 8;

	int i;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;

	if(ai%bs==0)
		{
		i = 0;
		for( ; i<n-7; i+=8)
			{
			kernel_sgemv_t_8_lib8(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
			}
		if(i<n)
			{
			kernel_sgemv_t_8_vs_lib8(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i], n-i);
			}
		}
	else
		{
		i = 0;
		for( ; i<n; i+=8)
			{
			kernel_sgemv_t_8_gen_lib8(m, &alpha, ai%bs, &pA[i*bs], sda, x, &beta, &y[i], &z[i], n-i);
			}
		}
	
	return;

	}



#else

#error : wrong LA choice

#endif
