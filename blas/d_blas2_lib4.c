/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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

#include "../include/d_kernel.h"



void dgemv_n_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z)
	{

	if(m<=0)
		return;
	
	const int bs = 4;

	int i;

	i = 0;
	for( ; i<m-3; i+=4)
		{
		kernel_dgemv_n_4_lib4(n, &pA[i*sda], x, alg, &y[i], &z[i]);
		}
	if(i<m)
		{
		kernel_dgemv_n_4_vs_lib4(n, &pA[i*sda], x, alg, &y[i], &z[i], m-i);
		}
	
	}



void dgemv_t_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z)
	{

	if(n<=0)
		return;
	
	const int bs = 4;

	int j;

	j = 0;
	for( ; j<n-3; j+=4)
		{
		kernel_dgemv_t_4_lib4(m, &pA[j*bs], sda, alg, x, &y[j], &z[j]);
		}
	if(j<n)
		{
		kernel_dgemv_t_4_vs_lib4(m, &pA[j*bs], sda, alg, x, &y[j], &z[j], n-j);
		}
	
	}

