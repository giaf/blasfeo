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

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_aux.h"



void sgemm_nt_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	int i, j, l;

	i = 0;

	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nt_8x8_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
//			kernel_sgemm_nt_8x4_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
//			kernel_sgemm_nt_8x4_lib8(k, &alpha, &pA[i*sda], &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], &pD[(j+4)*bs+i*sdd]);
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nt_8x4_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
				if(j<n-4)
					{
					kernel_sgemm_nt_8x4_gen_lib8(k, &alpha, &pA[i*sda], &pB[4+j*sdb], &beta, 0, &pC[(j+4)*bs+i*sdc], sdc, 0, &pD[(j+4)*bs+i*sdd], sdd, 0, 8, 0, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nt_8x4_gen_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, 0, &pC[(j+0)*bs+i*sdc], sdc, 0, &pD[(j+0)*bs+i*sdd], sdd, 0, 8, 0, n-j);
				}
			}
		}
	if(m>i)
		{
		goto left_8;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_8:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_8x4_gen_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, 0, &pC[(j+0)*bs+i*sdc], sdc, 0, &pD[(j+0)*bs+i*sdd], sdd, 0, m-i, 0, 4);
		kernel_sgemm_nt_8x4_gen_lib8(k, &alpha, &pA[i*sda], &pB[4+j*sdb], &beta, 0, &pC[(j+4)*bs+i*sdc], sdc, 0, &pD[(j+4)*bs+i*sdd], sdd, 0, m-i, 0, n-(j+4));
		}
	if(j<n)
		{
		kernel_sgemm_nt_8x4_gen_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, 0, &pC[(j+0)*bs+i*sdc], sdc, 0, &pD[(j+0)*bs+i*sdd], sdd, 0, m-i, 0, n-j);
		}
	return;

	}




