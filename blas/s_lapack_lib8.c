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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"



void spotrf_l_libstr(int m, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ci>0 | di>0)
		{
		printf("\nspotrf_l_libstr: feature not implemented yet: ci>0, di>0\n");
		exit(1);
		}

	const int bs = 8;

	int i, j;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	i = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<i; j+=8)
			{
			kernel_strsm_nt_rl_inv_24x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0]);
			kernel_strsm_nt_rl_inv_24x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4]);
			}
		kernel_spotrf_nt_l_24x4_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0]);
		kernel_spotrf_nt_l_20x4_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_16x4_lib8((j+8), &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8]);
		kernel_spotrf_nt_l_12x4_lib8((j+12), &pD[(i+8)*sdd], sdd, &pD[4+(j+8)*sdd], &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, &dD[j+12]);
		kernel_spotrf_nt_l_8x8_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else if(m-i<=12)
			{
			goto left_12;
			}
		else if(m-i<=16)
			{
			goto left_16;
			}
		else
			{
			goto left_24;
			}
		}
#else
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<i; j+=8)
			{
			kernel_strsm_nt_rl_inv_16x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0]);
			kernel_strsm_nt_rl_inv_16x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4]);
			}
		kernel_spotrf_nt_l_16x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0]);
		kernel_spotrf_nt_l_12x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_8x8_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8]);
		}
	if(m>i)
		{
		if(m-i<=8)
			{
			goto left_8;
			}
		else
			{
			goto left_16;
			}
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_X64_INTEL_HASWELL)
	left_24: // 17 <= m <= 23
	j = 0;
	for(; j<i & j<m-7; j+=8)
		{
		kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, m-(j+0));
		kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, m-(j+4));
		}
	kernel_spotrf_nt_l_24x4_vs_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0], m-(i+0), m-(j+0));
	kernel_spotrf_nt_l_20x4_vs_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4], m-(i+0), m-(j+4));
	kernel_spotrf_nt_l_16x4_vs_lib8((j+8), &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8], m-(i+8), m-(j+8));
	kernel_spotrf_nt_l_12x4_vs_lib8((j+12), &pD[(i+8)*sdd], sdd, &pD[4+(j+8)*sdd], &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, &dD[j+12], m-(i+8), m-(j+12));
	if(j<m-20) // 21 - 23
		{
		kernel_spotrf_nt_l_8x8_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), m-(j+16));
		}
	else // 17 18 19 20
		{
		kernel_spotrf_nt_l_8x4_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), m-(j+16));
		}
	return;
#endif

	left_16: // 9 <= m <= 16
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, m-(j+0));
		kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, m-(j+4));
		}
	kernel_spotrf_nt_l_16x4_vs_lib8(j+0, &pD[(i+0)*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+j*sdc], sdc, &pD[(j+0)*bs+j*sdd], sdd, &dD[j+0], m-(i+0), m-(j+0));
	kernel_spotrf_nt_l_12x4_vs_lib8(j+4, &pD[(i+0)*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+j*sdc], sdc, &pD[(j+4)*bs+j*sdd], sdd, &dD[j+4], m-(i+0), m-(j+4));
	if(j<m-12) // 13 - 16
		{
		kernel_spotrf_nt_l_8x8_vs_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), m-(j+8));
		}
	else // 9 - 12
		{
		kernel_spotrf_nt_l_8x4_vs_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), m-(j+8));
		}
	return;

#if defined(TARGET_X64_INTEL_HASWELL)
	left_12: // 9 <= m <= 12
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_strsm_nt_rl_inv_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		kernel_strsm_nt_rl_inv_4x8_vs_lib8(j, &pD[(i+8)*sdd], &pD[j*sdd], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], &pD[j*bs+j*sdd], &dD[j], m-(i+8), m-j);
		}
	kernel_spotrf_nt_l_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
	kernel_strsm_nt_rl_inv_4x8_vs_lib8(j, &pD[(i+8)*sdd], &pD[j*sdd], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], &pD[j*bs+j*sdd], &dD[j], m-(i+8), m-j);
	if(j<m-8) // 9 - 12
		{
		kernel_spotrf_nt_l_8x4_vs_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[(j+8)], m-(i+8), m-(j+8));
		}
	return;
#endif

	left_8: // 1 <= m <= 8
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_strsm_nt_rl_inv_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		}
	if(j<m-4) // 5 - 8
		{
		kernel_spotrf_nt_l_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		}
	else // 1 - 4
		{
		kernel_spotrf_nt_l_8x4_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		}
	return;

#if defined(TARGET_X64_INTEL_HASWELL)
	left_4: // 1 <= m <= 4
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_strsm_nt_rl_inv_4x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		}
	kernel_spotrf_nt_l_8x4_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
	return;
#endif

	}



void spotrf_l_mn_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0 | n<=0)
		return;

	if(ci>0 | di>0)
		{
		printf("\nspotrf_l_mn_libstr: feature not implemented yet: ci>0, di>0\n");
		exit(1);
		}

	const int bs = 8;

	int i, j;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	i = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<i & j<n-7; j+=8)
			{
			kernel_strsm_nt_rl_inv_24x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0]);
			kernel_strsm_nt_rl_inv_24x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4]);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
				if(j<n-4) // 5 6 7
					{
					kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[(j+4)*bs+(j+4)*sdd], &dD[j+4], m-i, n-(j+4));
					}
				}
			else // dpotrf
				{
				if(j<n-23)
					{
					kernel_spotrf_nt_l_24x4_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0]);
					kernel_spotrf_nt_l_20x4_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4]);
					kernel_spotrf_nt_l_16x4_lib8((j+8), &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8]);
					kernel_spotrf_nt_l_12x4_lib8((j+12), &pD[(i+8)*sdd], sdd, &pD[4+(j+8)*sdd], &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, &dD[j+12]);
					kernel_spotrf_nt_l_8x8_lib8((j+16), &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16]);
					}
				else
					{
					if(j<n-4) // 5 - 23
						{
						kernel_spotrf_nt_l_24x4_vs_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0], m-(i+0), n-(j+0));
						kernel_spotrf_nt_l_20x4_vs_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4], m-(i+0), n-(j+4));
						if(j==n-8)
							return;
						if(j<n-12) // 13 - 23
							{
							kernel_spotrf_nt_l_16x4_vs_lib8((j+8), &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8], m-(i+8), n-(j+8));
							kernel_spotrf_nt_l_12x4_vs_lib8((j+12), &pD[(i+8)*sdd], sdd, &pD[4+(j+8)*sdd], &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, &dD[j+12], m-(i+8), n-(j+12));
							if(j==n-16)
								return;
							if(j<n-20) // 21 - 23
								{
								kernel_spotrf_nt_l_8x8_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), n-(j+16));
								}
							else // 17 18 19 20
								{
								kernel_spotrf_nt_l_8x4_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), n-(j+16));
								}
							}
						else // 9 10 11 12
							{
							kernel_spotrf_nt_l_16x4_vs_lib8(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8], m-(i+8), n-(j+8));
							}
						}
					else // 1 2 3 4
						{
						kernel_spotrf_nt_l_24x4_vs_lib8(j, &pD[(i+0)*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &dD[j], m-(i+0), n-j);
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=8)
			{
			goto left_8;
			}
		else if(m-i<=16)
			{
			goto left_16;
			}
		else
			{
			goto left_24;
			}
		}
#else
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<i & j<n-7; j+=8)
			{
			kernel_strsm_nt_rl_inv_16x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0]);
			kernel_strsm_nt_rl_inv_16x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4]);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
				if(j<n-4) // 5 6 7
					{
					kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[(j+4)*bs+(j+4)*sdd], &dD[j+4], m-i, n-(j+4));
					}
				}
			else // dpotrf
				{
				if(j<n-15)
					{
					kernel_spotrf_nt_l_16x4_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0]);
					kernel_spotrf_nt_l_12x4_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4]);
					kernel_spotrf_nt_l_8x8_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8]);
					}
				else
					{
					if(j<n-4) // 5 - 15
						{
						kernel_spotrf_nt_l_16x4_vs_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0], m-(i+0), n-(j+0));
						kernel_spotrf_nt_l_12x4_vs_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4], m-(i+0), n-(j+4));
						if(j==n-8) // 8
							return;
						if(j<n-12) // 13 - 15
							{
							kernel_spotrf_nt_l_8x8_vs_lib8(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), n-(j+8));
							}
						else // 9 10 11 12
							{
							kernel_spotrf_nt_l_8x4_vs_lib8(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), n-(j+8));
							}
						}
					else // 1 2 3 4
						{
						kernel_spotrf_nt_l_16x4_vs_lib8(j, &pD[(i+0)*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &dD[j], m-(i+0), n-j);
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=8)
			{
			goto left_8;
			}
		else
			{
			goto left_16;
			}
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_X64_INTEL_HASWELL)
	left_24:
	j = 0;
	for(; j<i & j<n-7; j+=8)
		{
		kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
		kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, n-(j+4));
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
			if(j<n-4) // 5 6 7
				{
				kernel_strsm_nt_rl_inv_24x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, n-(j+4));
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 - 23
				{
				kernel_spotrf_nt_l_24x4_vs_lib8((j+0), &pD[(i+0)*sdd], sdd, &pD[(j+0)*sdd], &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, &dD[j+0], m-(i+0), n-(j+0));
				kernel_spotrf_nt_l_20x4_vs_lib8((j+4), &pD[(i+0)*sdd], sdd, &pD[4+(j+0)*sdd], &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, &dD[j+4], m-(i+0), n-(j+4));
				if(j>=n-8)
					return;
				if(j<n-12) // 13 - 23
					{
					kernel_spotrf_nt_l_16x4_vs_lib8((j+8), &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8], m-(i+8), n-(j+8));
					kernel_spotrf_nt_l_12x4_vs_lib8((j+12), &pD[(i+8)*sdd], sdd, &pD[4+(j+8)*sdd], &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, &dD[j+12], m-(i+8), n-(j+12));
					if(j>=n-16)
						return;
					if(j<n-20) // 21 - 23
						{
						kernel_spotrf_nt_l_8x8_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), n-(j+16));
						}
					else // 17 18 19 20
						{
						kernel_spotrf_nt_l_8x4_vs_lib8(j+16, &pD[(i+16)*sdd], &pD[(j+16)*sdd], &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], &dD[j+16], m-(i+16), n-(j+16));
						}
					}
				else // 9 10 11 12
					{
					kernel_spotrf_nt_l_16x4_vs_lib8(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, &dD[j+8], m-(i+8), n-(j+8));
					}
				}
			else // 1 2 3 4
				{
				kernel_spotrf_nt_l_24x4_vs_lib8(j, &pD[(i+0)*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &dD[j], m-(i+0), n-j);
				}
			}
		}
	return;
#endif

	left_16:
	j = 0;
	for(; j<i & j<n-7; j+=8)
		{
		kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[0+(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
		kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, n-(j+4));
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+0, &pD[i*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, &pD[(j+0)*bs+(j+0)*sdd], &dD[j+0], m-i, n-(j+0));
			if(j<n-4) // 5 6 7
				{
				kernel_strsm_nt_rl_inv_16x4_vs_lib8(j+4, &pD[i*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, &pD[4+(j+4)*bs+(j+0)*sdd], &dD[j+4], m-i, n-(j+4));
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 - 15
				{
				kernel_spotrf_nt_l_16x4_vs_lib8(j+0, &pD[(i+0)*sdd], sdd, &pD[0+j*sdd], &pC[(j+0)*bs+j*sdc], sdc, &pD[(j+0)*bs+j*sdd], sdd, &dD[j+0], m-(i+0), n-(j+0));
				kernel_spotrf_nt_l_12x4_vs_lib8(j+4, &pD[(i+0)*sdd], sdd, &pD[4+j*sdd], &pC[(j+4)*bs+j*sdc], sdc, &pD[(j+4)*bs+j*sdd], sdd, &dD[j+4], m-(i+0), n-(j+4));
				if(j>=n-8)
					return;
				if(j<n-12) // 13 - 15
					{
					kernel_spotrf_nt_l_8x8_vs_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), n-(j+8));
					}
				else // 9 - 12
					{
					kernel_spotrf_nt_l_8x4_vs_lib8((j+8), &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], &dD[j+8], m-(i+8), n-(j+8));
					}
				}
			else // 1 2 3 4
				{
				kernel_spotrf_nt_l_16x4_vs_lib8(j, &pD[(i+0)*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &dD[j], m-(i+0), n-j);
				}
			}
		}
	return;

	left_8:
	j = 0;
	for(; j<i & j<n-7; j+=8)
		{
		kernel_strsm_nt_rl_inv_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			if(j<n-4) // 5 6 7
				{
				kernel_strsm_nt_rl_inv_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
				}
			else // 1 2 3 4
				{
				kernel_strsm_nt_rl_inv_8x4_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 6 7
				{
				kernel_spotrf_nt_l_8x8_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
				}
			else // 1 2 3 4
				{
				kernel_spotrf_nt_l_8x4_vs_lib8(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
				}
			}
		}
	return;

	}



int sgeqrf_work_size_libstr(int m, int n)
	{
	printf("\nsgeqrf_work_size_libstr: feature not implemented yet\n");
	exit(1);
	return 0;
	}



void sgeqrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
	printf("\nsgeqrf_libstr: feature not implemented yet\n");
	exit(1);
	return;
	}




