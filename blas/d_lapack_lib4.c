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

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"



/****************************
* old interface
****************************/

void dpotrf_nt_l_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dpptrf
				{
				if(n<j-11)
					{
					kernel_dpotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j]);
					kernel_dpotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4]);
					kernel_dpotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8]);
					}
				else
					{
					kernel_dpotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_dpotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
						if(j<n-8)
							{
							kernel_dpotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8], m-i-8, n-j-8);
							}
						}
					}
				}
			}
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
		else
			{
			goto left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dpotrf
				{
//				if(j<n-7)
				if(0)
					{
					kernel_dpotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j]);
					kernel_dpotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4]);
					}
				else
					{
					kernel_dpotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_dpotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4], m-i-4, n-j-4);
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-3)
					{
					kernel_dpotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j]);
					}
				else
					{
					kernel_dpotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
					}
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_X64_INTEL_HASWELL)
	left_12:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dpotrf
			{
			kernel_dpotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
			if(j<n-4)
				{
				kernel_dpotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
				if(j<n-8)
					{
					kernel_dpotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8], m-i-8, n-j-8);
					}
				}
			}
		}
	return;
#endif

#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
	left_8:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dpotrf
			{
			kernel_dpotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
			if(j<n-4)
				{
				kernel_dpotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4], m-i-4, n-j-4);
				}
			}
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dpotrf
			{
			kernel_dpotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		}
	return;

	}



void dsyrk_dpotrf_nt_l_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;

	int alg = 1; // XXX

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(j<i) // dgemm
				{
				kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-11)
					{
					kernel_dsyrk_dpotrf_nt_l_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j]);
					kernel_dsyrk_dpotrf_nt_l_8x4_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4]);
					kernel_dsyrk_dpotrf_nt_l_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8]);
					}
				else
					{
					kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
						if(j<n-8)
							{
							kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8], m-i-8, n-j-8);
							}
						}
					}
				}
			}
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
		else
			{
			goto left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(j<i) // dgemm
				{
				kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-7)
//				if(0)
					{
					kernel_dsyrk_dpotrf_nt_l_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j]);
					kernel_dsyrk_dpotrf_nt_l_4x4_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4]);
					}
				else
					{
					kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4], m-i-4, n-j-4);
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(i<j) // dgemm
				{
				kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-3)
					{
					kernel_dsyrk_dpotrf_nt_l_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j]);
					}
				else
					{
					kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
					}
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_X64_INTEL_HASWELL)
	left_12:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
			if(j<n-4)
				{
				kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
				if(j<n-8)
					{
					kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8], m-i-8, n-j-8);
					}
				}
			}
		}
	return;
#endif

#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
	left_8:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
			if(j<n-4)
				{
				kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &inv_diag_D[j+4], m-i-4, n-j-4);
				}
			}
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		}
	return;

	}



void dgetrf_nn_nopivot_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;

	int ii, jj, ie;

	// main loop
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for( ; ii<m-11; ii+=12)
		{
		jj = 0;
		// solve lower
		ie = n<ii ? n : ii; // ie is multiple of 4
		for( ; jj<ie-3; jj+=4)
			{
			kernel_dtrsm_nn_ru_inv_12x4_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj]);
			}
		if(jj<ie)
			{
			kernel_dtrsm_nn_ru_inv_12x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
			jj+=4;
			}
		// factorize
		if(jj<n-3)
			{
			kernel_dgetrf_nn_l_12x4_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dgetrf_nn_l_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		if(jj<n-3)
			{
			kernel_dgetrf_nn_m_12x4_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dgetrf_nn_m_12x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		if(jj<n-3)
			{
			kernel_dgetrf_nn_r_12x4_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dgetrf_nn_r_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		// solve upper 
		for( ; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nn_ll_one_12x4_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd);
			}
		if(jj<n)
			{
			kernel_dtrsm_nn_ll_one_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, m-ii, n-jj);
			}
		}
	if(m>ii)
		{
		if(m-ii<=4)
			{
			goto left_4;
			}
		else if(m-ii<=8)
			{
			goto left_8;
			}
		else 
			{
			goto left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for( ; ii<m-7; ii+=8)
		{
		jj = 0;
		// solve lower
		ie = n<ii ? n : ii; // ie is multiple of 4
		for( ; jj<ie-3; jj+=4)
			{
			kernel_dtrsm_nn_ru_inv_8x4_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj]);
			}
		if(jj<ie)
			{
			kernel_dtrsm_nn_ru_inv_8x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
			jj+=4;
			}
		// factorize
		if(jj<n-3)
			{
			kernel_dgetrf_nn_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj]);
			kernel_dtrsm_nn_ru_inv_4x4_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
			kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-(ii+4), n-jj);
			jj+=4;
			}
		if(jj<n-3)
			{
			kernel_dtrsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd]);
			kernel_dgetrf_nn_4x4_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
			kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &inv_diag_D[jj], m-(ii+4), n-jj);
			jj+=4;
			}
		// solve upper 
		for( ; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nn_ll_one_8x4_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd],sdd,  &pD[ii*bs+ii*sdd], sdd);
			}
		if(jj<n)
			{
			kernel_dtrsm_nn_ll_one_8x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, m-ii, n-jj);
			}
		}
	if(m>ii)
		{
		if(m-ii<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for( ; ii<m-3; ii+=4)
		{
		jj = 0;
		// solve lower
		ie = n<ii ? n : ii; // ie is multiple of 4
		for( ; jj<ie-3; jj+=4)
			{
			kernel_dtrsm_nn_ru_inv_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj]);
			}
		if(jj<ie)
			{
			kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
			jj+=4;
			}
		// factorize
		if(jj<n-3)
			{
			kernel_dgetrf_nn_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		// solve upper 
		for( ; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd]);
			}
		if(jj<n)
			{
			kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
			}
		}
	if(m>ii)
		{
		goto left_4;
		}

#endif

	// common return if i==m
	return;

#if defined(TARGET_X64_INTEL_HASWELL)
	left_12:
	jj = 0;
	// solve lower
	ie = n<ii ? n : ii; // ie is multiple of 4
	for( ; jj<ie; jj+=4)
		{
		kernel_dtrsm_nn_ru_inv_12x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
		}
	// factorize
	if(jj<n)
		{
		kernel_dgetrf_nn_l_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	if(jj<n)
		{
		kernel_dgetrf_nn_l_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	if(jj<n)
		{
		kernel_dgetrf_nn_r_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	// solve upper 
	for( ; jj<n; jj+=4)
		{
		kernel_dtrsm_nn_ll_one_12x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, m-ii, n-jj);
		}
	return;

#endif

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	left_8:
	jj = 0;
	// solve lower
	ie = n<ii ? n : ii; // ie is multiple of 4
	for( ; jj<ie; jj+=4)
		{
		kernel_dtrsm_nn_ru_inv_8x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
		}
	// factorize
	if(jj<n)
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dgetrf_nn_l_8x4_vs_lib4(jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &inv_diag_D[jj], m-ii, n-jj);
#else
		kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
		kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-(ii+4), n-jj);
#endif
		jj+=4;
		}
	if(jj<n)
		{
		kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
		kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[(ii+4)*sdd], &pD[jj*bs], sdd, &pC[jj*bs+(ii+4)*sdc], &pD[jj*bs+(ii+4)*sdd], &inv_diag_D[jj], m-(ii+4), n-jj);
		jj+=4;
		}
	// solve upper 
	for( ; jj<n; jj+=4)
		{
		kernel_dtrsm_nn_ll_one_8x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], sdc, &pD[jj*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, m-ii, n-jj);
		}
	return;

#endif

	left_4:
	jj = 0;
	// solve lower
	ie = n<ii ? n : ii; // ie is multiple of 4
	for( ; jj<ie; jj+=4)
		{
		kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
		}
	// factorize
	if(jj<n)
		{
		kernel_dgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	// solve upper 
	for( ; jj<n; jj+=4)
		{
		kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
		}
	return;

	}



void dgetrf_nn_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D, int *ipiv)
	{

	if(m<=0)
		return;
	
	const int bs = 4;

	int ii, jj, i0, i1, j0, ll, p;

	double d1 = 1.0;
	double dm1 = -1.0;

	// needs to perform row-excanges on the yet-to-be-factorized matrix too
	if(pC!=pD)
		dgecp_lib(m, n, 1.0, 0, pC, sdc, 0, pD, sdd);

	// minimum matrix size
	p = n<m ? n : m; // XXX

	// main loop
#if defined(TARGET_X64_INTEL_HASWELL)
	// 12 columns at a time
	jj = 0;
	for(; jj<p-11; jj+=12)
		{
		// pivot & factorize & solve lower
		// left block-column
		ii = jj;
		i0 = ii;
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
				}
			}
		kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
			drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
			drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
			drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
			}
		// middle block-column
		ii = i0;
		kernel_dtrsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd]);
		ii += 4;
		i1 = ii;
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, 4);
				}
			}
		kernel_dgetrf_pivot_4_lib4(m-i1, &pD[(jj+4)*bs+i1*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i1]);
		ipiv[i1+0] += i1;
		if(ipiv[i1+0]!=i1+0)
			{
			drowsw_lib(jj+4, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
			drowsw_lib(n-jj-8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+8)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+8)*bs);
			}
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+4, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+8)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+8)*bs);
			}
		ipiv[i1+2] += i1;
		if(ipiv[i1+2]!=i1+2)
			{
			drowsw_lib(jj+4, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
			drowsw_lib(n-jj-8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+8)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+8)*bs);
			}
		ipiv[i1+3] += i1;
		if(ipiv[i1+3]!=i1+3)
			{
			drowsw_lib(jj+4, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
			drowsw_lib(n-jj-8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+8)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+8)*bs);
			}
		// right block-column
		ii = i0;
		kernel_dtrsm_nn_ll_one_8x4_lib4(ii, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd);
		ii += 8;
		i1 = ii;
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4((jj+8), &dm1, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], &pD[(jj+8)*bs+ii*sdd], m-ii, 4);
				}
			}
		kernel_dgetrf_pivot_4_lib4(m-i1, &pD[(jj+8)*bs+i1*sdd], sdd, &inv_diag_D[(jj+8)], &ipiv[i1]);
		ipiv[i1+0] += i1;
		if(ipiv[i1+0]!=i1+0)
			{
			drowsw_lib(jj+8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
			drowsw_lib(n-jj-12, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+12)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+12)*bs);
			}
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-12, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+12)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+12)*bs);
			}
		ipiv[i1+2] += i1;
		if(ipiv[i1+2]!=i1+2)
			{
			drowsw_lib(jj+8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
			drowsw_lib(n-jj-12, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+12)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+12)*bs);
			}
		ipiv[i1+3] += i1;
		if(ipiv[i1+3]!=i1+3)
			{
			drowsw_lib(jj+8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
			drowsw_lib(n-jj-12, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+12)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+12)*bs);
			}

		// solve upper 
//		i0 -= 8; // 4 ???
		ll = jj+12;
		for( ; ll<n-3; ll+=4)
			{
			kernel_dtrsm_nn_ll_one_12x4_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd);
			}
		if(ll<n)
			{
			kernel_dtrsm_nn_ll_one_12x4_vs_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd, 12, n-ll);
			}
		}
	if(m>=n)
		{
		if(n-jj>0)
			{
			if(n-jj<=4)
				goto left_n_4;
			else if(n-jj<=8)
				goto left_n_8;
			else
				goto left_n_12;
			}
		}
	else // n>m
		{
		if(m-jj>0)
			{
			if(m-jj<=4)
				goto left_m_4;
			else if(m-jj<=8)
				goto left_m_8;
			else
				goto left_m_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	// 8 columns at a time
	jj = 0;
	for(; jj<p-7; jj+=8)
		{
		// pivot & factorize & solve lower
		// left block-column
		ii = jj;
		i0 = ii;
#if defined(TARGET_X64_INTEL_HASWELL) // XXX
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
				}
			}
#else // SANDY_BRIDGE
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgemm_nn_8x4_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
				}
			}
#endif
		kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
			drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
			drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
			drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
			}
		// right block-column
		ii = i0;
		kernel_dtrsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd]);
		ii += 4;
		i0 = ii;
#if defined(TARGET_X64_INTEL_HASWELL) // XXX
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, 4);
				}
			}
#else // SANDY_BRIDGE
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgemm_nn_8x4_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, 4);
				}
			}
#endif
		kernel_dgetrf_pivot_4_lib4(m-i0, &pD[(jj+4)*bs+i0*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			drowsw_lib(jj+4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
			drowsw_lib(n-jj-8, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+8)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+8)*bs);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj+4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-8, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+8)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+8)*bs);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			drowsw_lib(jj+4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
			drowsw_lib(n-jj-8, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+8)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+8)*bs);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			drowsw_lib(jj+4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
			drowsw_lib(n-jj-8, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+8)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+8)*bs);
			}

		// solve upper 
		i0 -= 4;
		ll = jj+8;
		for( ; ll<n-3; ll+=4)
			{
			kernel_dtrsm_nn_ll_one_8x4_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd);
			}
		if(ll<n)
			{
			kernel_dtrsm_nn_ll_one_8x4_vs_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd, 8, n-ll);
			}
		}
	if(m>=n)
		{
		if(n-jj>0)
			{
			if(n-jj<=4) // (m>=1 && n==1) || (m>=2 && n==2) || m>=3 && n==3
				{
				goto left_n_4;
				}
			else // (m>=5 && n==5) || (m>=6 && n==6) || (m>=7 && n==7)
				goto left_n_8;
			}
		}
	else // n>m
		{
		if(m-jj>0)
			{
			if(m-jj<=4) // (m==1 && n>=2) || (m==2 && n>=3) || (m==3 && n>=4) || (m==4 && n>=5)
				goto left_m_4;
			else // (m==5 && n>=6) || (m==6 && n>=7) || (m==7 && n>=8)
				{
				goto left_m_8;
				}
			}
		}
#else
	// 4 columns at a time
	jj = 0;
	for(; jj<p-3; jj+=4) // XXX
		{
		// pivot & factorize & solve lower
		ii = jj;
		i0 = ii;
#if defined(TARGET_X64_INTEL_HASWELL) // XXX
		for( ; ii<m-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>8)
				{
				kernel_dgemm_nn_12x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
				}
			}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) // XXX
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgemm_nn_8x4_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd);
			}
		if(m-ii>0)
			{
			if(m-ii>4)
				{
				kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
				}
			else
				{
				kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
				}
			}
#else
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nn_4x4_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd]);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
			}
#endif
		kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
			drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
			drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
			drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
			}

		// solve upper
		ll = jj+4;
		for( ; ll<n-3; ll+=4)
			{
			kernel_dtrsm_nn_ll_one_4x4_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd]);
			}
		if(n-ll>0)
			{
			kernel_dtrsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], 4, n-ll);
			}
		}
	if(m>=n)
		{
		if(n-jj>0)
			{
			goto left_n_4;
			}
		}
	else
		{
		if(m-jj>0)
			{
			goto left_m_4;
			}
		}
#endif

	// common return if jj==n
	return;


	// clean up
#if defined(TARGET_X64_INTEL_HASWELL)
	left_n_12:
	// 9-12 columns at a time
	// pivot & factorize & solve lower
	// left block-column
	ii = jj;
	i0 = ii;
	for( ; ii<m-8; ii+=12)
		{
		kernel_dgemm_nn_12x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
		}
	if(m-ii>4)
		{
		kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
//		ii+=8;
		}
	else if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
//		ii+=4;
		}
	kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	ipiv[i0+1] += i0;
	if(ipiv[i0+1]!=i0+1)
		{
		drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
		drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
		}
	ipiv[i0+2] += i0;
	if(ipiv[i0+2]!=i0+2)
		{
		drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
		drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
		}
	ipiv[i0+3] += i0;
	if(ipiv[i0+3]!=i0+3)
		{
		drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
		drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
		}
	// middle block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd], 4, n-jj-4);
	ii += 4;
	i1 = ii;
	for( ; ii<m-8; ii+=12)
		{
		kernel_dgemm_nn_12x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, n-jj-4);
		}
	if(m-ii>4)
		{
		kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, n-jj-4);
		}
	else if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, n-jj-4);
		}
	kernel_dgetrf_pivot_4_vs_lib4(m-i1, n-jj-4, &pD[(jj+4)*bs+i1*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i1]);
	ipiv[i1+0] += i1;
	if(ipiv[i1+0]!=i1+0)
		{
		drowsw_lib(jj+4, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
		drowsw_lib(n-jj-8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+8)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+8)*bs);
		}
	if(n-jj-4>1)
		{
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+4, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+8)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+8)*bs);
			}
		if(n-jj-4>2)
			{
			ipiv[i1+2] += i1;
			if(ipiv[i1+2]!=i1+2)
				{
				drowsw_lib(jj+4, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
				drowsw_lib(n-jj-8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+8)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+8)*bs);
				}
			if(n-jj-4>3)
				{
				ipiv[i1+3] += i1;
				if(ipiv[i1+3]!=i1+3)
					{
					drowsw_lib(jj+4, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
					drowsw_lib(n-jj-8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+8)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+8)*bs);
					}
				}
			}
		}
	// right block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_8x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, 8, n-jj-8);
	ii += 8;
	i1 = ii;
	for( ; ii<m-8; ii+=12)
		{
		kernel_dgemm_nn_12x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, m-ii, n-jj-8);
		}
	if(m-ii>4)
		{
		kernel_dgemm_nn_8x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, m-ii, n-jj-8);
		}
	else if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], &pD[(jj+8)*bs+ii*sdd], m-ii, n-jj-8);
		}
	kernel_dgetrf_pivot_4_vs_lib4(m-i1, n-jj-8, &pD[(jj+8)*bs+i1*sdd], sdd, &inv_diag_D[(jj+8)], &ipiv[i1]);
	ipiv[i1+0] += i1;
	if(ipiv[i1+0]!=i1+0)
		{
		drowsw_lib(jj+8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
		drowsw_lib(n-jj-12, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+12)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+12)*bs);
		}
	if(n-jj-8>1)
		{
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-12, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+12)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+12)*bs);
			}
		if(n-jj-8>2)
			{
			ipiv[i1+2] += i1;
			if(ipiv[i1+2]!=i1+2)
				{
				drowsw_lib(jj+8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
				drowsw_lib(n-jj-12, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+12)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+12)*bs);
				}
			if(n-jj-8>3)
				{
				ipiv[i1+3] += i1;
				if(ipiv[i1+3]!=i1+3)
					{
					drowsw_lib(jj+8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
					drowsw_lib(n-jj-12, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+12)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+12)*bs);
					}
				}
			}
		}

	// solve upper 
	// there is no upper
	return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL)
	left_m_12:
	// 9-12 rows at a time
	// pivot & factorize & solve lower
	// left block-column
	ii = jj;
	i0 = ii;
	kernel_dgemm_nn_12x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
	kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	ipiv[i0+1] += i0;
	if(ipiv[i0+1]!=i0+1)
		{
		drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
		drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
		}
	ipiv[i0+2] += i0;
	if(ipiv[i0+2]!=i0+2)
		{
		drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
		drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
		}
	ipiv[i0+3] += i0;
	if(ipiv[i0+3]!=i0+3)
		{
		drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
		drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
		}
	// middle block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd], 4, n-jj-4);
	ii += 4;
	i1 = ii;
	kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, n-jj-4);
	kernel_dgetrf_pivot_4_vs_lib4(m-i1, n-jj-4, &pD[(jj+4)*bs+i1*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i1]);
	ipiv[i1+0] += i1;
	if(ipiv[i1+0]!=i1+0)
		{
		drowsw_lib(jj+4, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
		drowsw_lib(n-jj-8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+8)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+8)*bs);
		}
	if(m-jj-4>1)
		{
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+4, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+8)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+8)*bs);
			}
		if(m-jj-4>2)
			{
			ipiv[i1+2] += i1;
			if(ipiv[i1+2]!=i1+2)
				{
				drowsw_lib(jj+4, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
				drowsw_lib(n-jj-8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+8)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+8)*bs);
				}
			if(m-jj-4>3)
				{
				ipiv[i1+3] += i1;
				if(ipiv[i1+3]!=i1+3)
					{
					drowsw_lib(jj+4, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
					drowsw_lib(n-jj-8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+8)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+8)*bs);
					}
				}
			}
		}
	// right block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_8x4_vs_lib4(ii, &pD[ii*sdd], sdd, &pD[(jj+8)*bs], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[(jj+8)*bs+ii*sdd], sdd, &pD[ii*bs+ii*sdd], sdd, 8, n-jj-8);
	ii += 8;
	i1 = ii;
	kernel_dgemm_nn_4x4_vs_lib4((jj+8), &dm1, &pD[ii*sdd], &pD[(jj+8)*bs], sdd, &d1, &pD[(jj+8)*bs+ii*sdd], &pD[(jj+8)*bs+ii*sdd], m-ii, n-jj-8);
	kernel_dgetrf_pivot_4_vs_lib4(m-i1, n-jj-8, &pD[(jj+8)*bs+i1*sdd], sdd, &inv_diag_D[(jj+8)], &ipiv[i1]);
	ipiv[i1+0] += i1;
	if(ipiv[i1+0]!=i1+0)
		{
		drowsw_lib(jj+8, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs);
		drowsw_lib(n-jj-12, pD+(i1+0)/bs*bs*sdd+(i1+0)%bs+(jj+12)*bs, pD+(ipiv[i1+0])/bs*bs*sdd+(ipiv[i1+0])%bs+(jj+12)*bs);
		}
	if(m-jj-8>1)
		{
		ipiv[i1+1] += i1;
		if(ipiv[i1+1]!=i1+1)
			{
			drowsw_lib(jj+8, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs);
			drowsw_lib(n-jj-12, pD+(i1+1)/bs*bs*sdd+(i1+1)%bs+(jj+12)*bs, pD+(ipiv[i1+1])/bs*bs*sdd+(ipiv[i1+1])%bs+(jj+12)*bs);
			}
		if(m-jj-8>2)
			{
			ipiv[i1+2] += i1;
			if(ipiv[i1+2]!=i1+2)
				{
				drowsw_lib(jj+8, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs);
				drowsw_lib(n-jj-12, pD+(i1+2)/bs*bs*sdd+(i1+2)%bs+(jj+12)*bs, pD+(ipiv[i1+2])/bs*bs*sdd+(ipiv[i1+2])%bs+(jj+12)*bs);
				}
			if(m-jj-8>3)
				{
				ipiv[i1+3] += i1;
				if(ipiv[i1+3]!=i1+3)
					{
					drowsw_lib(jj+8, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs);
					drowsw_lib(n-jj-12, pD+(i1+3)/bs*bs*sdd+(i1+3)%bs+(jj+12)*bs, pD+(ipiv[i1+3])/bs*bs*sdd+(ipiv[i1+3])%bs+(jj+12)*bs);
					}
				}
			}
		}

	// solve upper 
//	i0 -= 8;
	ll = jj+12;
	for( ; ll<n; ll+=4)
		{
		kernel_dtrsm_nn_ll_one_12x4_vs_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd, m-i0, n-ll);
		}
	return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	left_n_8:
	// 5-8 columns at a time
	// pivot & factorize & solve lower
	// left block-column
	ii = jj;
	i0 = ii;
	for( ; ii<m-4; ii+=8)
		{
		kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
		}
	if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
//		ii+=4;
		}
	kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	ipiv[i0+1] += i0;
	if(ipiv[i0+1]!=i0+1)
		{
		drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
		drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
		}
	ipiv[i0+2] += i0;
	if(ipiv[i0+2]!=i0+2)
		{
		drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
		drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
		}
	ipiv[i0+3] += i0;
	if(ipiv[i0+3]!=i0+3)
		{
		drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
		drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
		}
	// right block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd], 4, n-jj-4);
	ii += 4;
	i0 = ii;
	for( ; ii<m-4; ii+=8)
		{
		kernel_dgemm_nn_8x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], sdd, &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], sdd, &pD[(jj+4)*bs+ii*sdd], sdd, m-ii, n-jj-4);
		}
	if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, n-jj-4);
		}
	kernel_dgetrf_pivot_4_vs_lib4(m-i0, n-jj-4, &pD[(jj+4)*bs+i0*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj+4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-8, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+8)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+8)*bs);
		}
	if(n-jj-4>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj+4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-8, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+8)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+8)*bs);
			}
		if(n-jj-4>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				drowsw_lib(jj+4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				drowsw_lib(n-jj-8, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+8)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+8)*bs);
				}
			if(n-jj-4>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					drowsw_lib(jj+4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					drowsw_lib(n-jj-8, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+8)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+8)*bs);
					}
				}
			}
		}

	// solve upper 
	// there is no upper
	return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	left_m_8:
	// 5-8 rows at a time
	// pivot & factorize & solve lower
	// left block-column
	ii = jj;
	i0 = ii;
	kernel_dgemm_nn_8x4_vs_lib4(jj, &dm1, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, m-ii, 4);
	kernel_dgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	ipiv[i0+1] += i0;
	if(ipiv[i0+1]!=i0+1)
		{
		drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
		drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
		}
	ipiv[i0+2] += i0;
	if(ipiv[i0+2]!=i0+2)
		{
		drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
		drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
		}
	ipiv[i0+3] += i0;
	if(ipiv[i0+3]!=i0+3)
		{
		drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
		drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
		}
	// right block-column
	ii = i0;
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], &pD[ii*bs+ii*sdd], 4, n-jj-4);
	ii += 4;
	i0 = ii;
	kernel_dgemm_nn_4x4_vs_lib4((jj+4), &dm1, &pD[ii*sdd], &pD[(jj+4)*bs], sdd, &d1, &pD[(jj+4)*bs+ii*sdd], &pD[(jj+4)*bs+ii*sdd], m-ii, n-jj-4);
	kernel_dgetrf_pivot_4_vs_lib4(m-i0, n-jj-4, &pD[(jj+4)*bs+i0*sdd], sdd, &inv_diag_D[(jj+4)], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj+4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-8, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+8)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+8)*bs);
		}
	if(m-jj-4>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj+4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-8, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+8)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+8)*bs);
			}
		if(m-jj-4>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				drowsw_lib(jj+4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				drowsw_lib(n-jj-8, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+8)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+8)*bs);
				}
			if(m-jj-4>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					drowsw_lib(jj+4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					drowsw_lib(n-jj-8, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+8)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+8)*bs);
					}
				}
			}
		}

	// solve upper 
	i0 -= 4;
	ll = jj+8;
	for( ; ll<n; ll+=4)
		{
		kernel_dtrsm_nn_ll_one_8x4_vs_lib4(i0, &pD[i0*sdd], sdd, &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[ll*bs+i0*sdd], sdd, &pD[i0*bs+i0*sdd], sdd, m-i0, n-ll);
		}
	return;
#endif


	left_n_4:
	// 1-4 columns at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
#if 0//defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX)
	for( ; ii<m-4; ii+=8)
		{
		kernel_dgemm_nn_8x4_vs_lib4(m-ii, n-jj, jj, &pD[ii*sdd], sdd, &pD[jj*bs], sdd, -1, &pD[jj*bs+ii*sdd], sdd, &pD[jj*bs+ii*sdd], sdd, 0, 0);
		}
	if(m-ii>0)
		{
		kernel_dgemm_nn_4x4_vs_lib4(m-ii, n-jj, jj, &pD[ii*sdd], &pD[jj*bs], sdd, -1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], 0, 0);
//		ii+=4;
		}
#else
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, n-jj);
		}
#endif
	kernel_dgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	if(n-jj>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		if(n-jj>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
				}
			if(n-jj>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
					}
				}
			}
		}

	// solve upper
	if(0) // there is no upper
		{
		ll = jj+4;
		for( ; ll<n; ll+=4)
			{
			kernel_dtrsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], m-i0, n-ll);
			}
		}
	return;


	left_m_4:
	// 1-4 rows at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
	kernel_dgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, n-jj);
	kernel_dgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		drowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		drowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	if(m-i0>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			drowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			drowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		if(m-i0>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				drowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				drowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
				}
			if(m-i0>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					drowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					drowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
					}
				}
			}
		}

	// solve upper
	ll = jj+4;
	for( ; ll<n; ll+=4)
		{
		kernel_dtrsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], m-i0, n-ll);
		}
	return;

	}


# if 0
void dlauum_dpotrf_blk_nt_l_lib(int m, int n, int nv, int *rv, int *cv, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;
	
	// TODO remove
	int k = cv[nv-1];

	const int bs = 4;

	int i, j, l;
	int ii, iii, jj, kii, kiii, kjj, k0, k1;

	i = 0;
	ii = 0;
	iii = 0;

#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-7; i+=8)
		{

		while(ii<nv && rv[ii]<i+8)
			ii++;
		if(ii<nv)
			kii = cv[ii];
		else
			kii = cv[ii-1];

		j = 0;
		jj = 0;
		for(; j<i && j<n-3; j+=4)
			{

			while(jj<nv && rv[jj]<j+4)
				jj++;
			if(jj<nv)
				kjj = cv[jj];
			else
				kjj = cv[jj-1];
			k0 = kii<kjj ? kii : kjj;

			kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{

			while(jj<nv && rv[jj]<j+4)
				jj++;
			if(jj<nv)
				kjj = cv[jj];
			else
				kjj = cv[jj-1];
			k0 = kii<kjj ? kii : kjj;

			if(j<i) // dgemm
				{
				kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], 8, n-j);
				}
			else // dsyrk
				{
				kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], 8, n-j);
				if(j<n-4)
					{
					kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], alg, &pC[(j+4)*bs+(j+4)*sdc], &pD[(j+4)*bs+(j+4)*sdd], &inv_diag_D[j+4], 4, n-j-4); // TODO
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for(; i<m-3; i+=4)
		{

		while(ii<nv && rv[ii]<i+4)
			ii++;
		if(ii<nv)
			kii = cv[ii];
		else
			kii = cv[ii-1];

		j = 0;
		jj = 0;
		for(; j<i && j<n-3; j+=4)
			{

			while(jj<nv && rv[jj]<j+4)
				jj++;
			if(jj<nv)
				kjj = cv[jj];
			else
				kjj = cv[jj-1];
			k0 = kii<kjj ? kii : kjj;

			kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{

			while(jj<nv && rv[jj]<j+4)
				jj++;
			if(jj<nv)
				kjj = cv[jj];
			else
				kjj = cv[jj-1];
			k0 = kii<kjj ? kii : kjj;

			if(i<j) // dgemm
				{
				kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], 4, n-j);
				}
			else // dsyrk
				{
				kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], 4, n-j);
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
	left_8:

	kii = cv[nv-1];

	j = 0;
	jj = 0;
	for(; j<i && j<n-3; j+=4)
		{

		while(jj<nv && rv[jj]<j+4)
			jj++;
		if(jj<nv)
			kjj = cv[jj];
		else
			kjj = cv[jj-1];
		k0 = kii<kjj ? kii : kjj;

		kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{

		while(jj<nv && rv[jj]<j+4)
			jj++;
		if(jj<nv)
			kjj = cv[jj];
		else
			kjj = cv[jj-1];
		k0 = kii<kjj ? kii : kjj;

		if(j<i) // dgemm
			{
			kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k0, &pA[i*sda], sda, &pB[j*sdb], j, &pD[i*sdd], sdd, &pD[j*sdd], alg, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, &inv_diag_D[j], m-i, n-j);
			if(j<n-4)
				{
				kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], alg, &pC[(j+4)*bs+(j+4)*sdc], &pD[(j+4)*bs+(j+4)*sdd], &inv_diag_D[j+4], m-i-4, n-j-4); // TODO
				}
			}
		}
	return;
#endif

	left_4:

	kii = cv[nv-1];

	j = 0;
	jj = 0;
	for(; j<i && j<n-3; j+=4)
		{

		while(jj<nv && rv[jj]<j+4)
			jj++;
		if(jj<nv)
			kjj = cv[jj];
		else
			kjj = cv[jj-1];
		k0 = kii<kjj ? kii : kjj;

		kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{

		while(jj<nv && rv[jj]<j+4)
			jj++;
		if(jj<nv)
			kjj = cv[jj];
		else
			kjj = cv[jj-1];
		k0 = kii<kjj ? kii : kjj;

		if(j<i) // dgemm
			{
			kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k0, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], alg, &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		}
	return;

	}
#endif




/****************************
* new interface
****************************/



#if defined(LA_HIGH_PERFORMANCE)



// dpotrf
void dpotrf_l_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
		printf("\ndpotrf_l_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}
	const int bs = D_BS;
	int sdc = sC->cn;
	int sdd = sD->cn;
	double *pC = sC->pA + cj*bs;
	double *pD = sD->pA + dj*bs;
	double *dD = sD->dA; // XXX what to do if di and dj are not zero
	dpotrf_nt_l_lib(m, n, pC, sdc, pD, sdd, dD);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}



// dsyrk dpotrf
void dsyrk_dpotrf_ln_libstr(int m, int n, int k, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\ndsyrk_dpotrf_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}
	const int bs = D_BS;
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	double *pA = sA->pA + aj*bs;
	double *pB = sB->pA + bj*bs;
	double *pC = sC->pA + cj*bs;
	double *pD = sD->pA + dj*bs;
	double *dD = sD->dA; // XXX what to do if di and dj are not zero
	dsyrk_dpotrf_nt_l_lib(m, n, k, pA, sda, pB, sdb, pC, sdc, pD, sdd, dD);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}



// dgetrf without pivoting
void dgetrf_nopivot_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
		printf("\ndgetf_nopivot_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}
	const int bs = D_BS;
	int sdc = sC->cn;
	int sdd = sD->cn;
	double *pC = sC->pA + cj*bs;
	double *pD = sD->pA + dj*bs;
	double *dD = sD->dA; // XXX what to do if di and dj are not zero
	dgetrf_nn_nopivot_lib(m, n, pC, sdc, pD, sdd, dD);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}




// dgetrf pivoting
void dgetrf_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj, int *ipiv)
	{
	if(ci!=0 | di!=0)
		{
		printf("\ndgetrf_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}
	const int bs = D_BS;
	int sdc = sC->cn;
	int sdd = sD->cn;
	double *pC = sC->pA + cj*bs;
	double *pD = sD->pA + dj*bs;
	double *dD = sD->dA; // XXX what to do if di and dj are not zero
	dgetrf_nn_lib(m, n, pC, sdc, pD, sdd, dD, ipiv);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}



#else

#error : wrong LA choice

#endif


