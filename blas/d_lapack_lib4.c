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

#if defined(LA_BLAS)
#if defined(LA_BLAS_OPENBLAS)
#include <f77blas.h>
#elif defined(LA_BLAS_BLIS)
#elif defined(LA_BLAS_NETLIB)
#include "d_blas.h"
#elif defined(LA_BLAS_MKL)
#include <mkl_blas.h>
#include <mkl_lapack.h>
#endif
#endif

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
				kernel_dpotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
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
//				if(j<n-7)
				if(0)
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
				kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
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
			if(j<n-8)
				{
				kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
				kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &inv_diag_D[j+8], m-i-8, n-j-8);
				}
			if(j<n-4)
				{
				kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &inv_diag_D[j+4], m-i-4, n-j-4);
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



#if defined(LA_BLASFEO)



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




#elif defined(LA_BLAS)



// dpotrf
void dpotrf_l_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	int i1 = 1;
	int mmn = m-n;
	int info;
	double d1 = 1.0;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
#if defined(LA_BLAS_MKL)
			dcopy(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#else
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#endif
		}
#if defined(LA_BLAS_MKL)
	dpotrf(&cl, &n, pD, &(sD->m), &info);
	dtrsm(&cr, &cl, &ct, &cn, &mmn, &n, &d1, pD, &(sD->m), pD+n, &(sD->m));
#else
	dpotrf_(&cl, &n, pD, &(sD->m), &info);
	dtrsm_(&cr, &cl, &ct, &cn, &mmn, &n, &d1, pD, &(sD->m), pD+n, &(sD->m));
#endif
	return;
	}



// dgetrf without pivoting
void dgetf2_nopivot(int m, int n, double *A, int lda)
	{
	if(m<=0 | n<=0)
		return;
	int i, j, itmp0, itmp1;
	int jmax = m<n ? m : n;
	int i1 = 1;
	double dtmp;
	double dm1 = -1.0;

	for(j=0; j<jmax; j++)
		{
		itmp0 = m-j-1;
		dtmp = 1.0/A[j+lda*j];
#if defined(LA_BLAS_MKL)
		dscal(&itmp0, &dtmp, &A[(j+1)+lda*j], &i1);
#else
		dscal_(&itmp0, &dtmp, &A[(j+1)+lda*j], &i1);
#endif
		itmp1 = n-j-1;
#if defined(LA_BLAS_MKL)
		dger(&itmp0, &itmp1, &dm1, &A[(j+1)+lda*j], &i1, &A[j+lda*(j+1)], &lda, &A[(j+1)+lda*(j+1)], &lda);
#else
		dger_(&itmp0, &itmp1, &dm1, &A[(j+1)+lda*j], &i1, &A[j+lda*(j+1)], &lda, &A[(j+1)+lda*(j+1)], &lda);
#endif
		}
	
	return;

	}

// dsyrk dpotrf
void dsyrk_dpotrf_ln_libstr(int m, int n, int k, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	int mmn = m-n;
	int info;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
#if defined(LA_BLAS_MKL)
			dcopy(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#else
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#endif
		}
#if defined(LA_BLAS_MKL)
	dgemm(&cn, &ct, &m, &n, &k, &d1, pA, &(sA->m), pB, &(sB->m), &d1, pD, &(sD->m));
	dpotrf(&cl, &n, pD, &(sD->m), &info);
	dtrsm(&cr, &cl, &ct, &cn, &mmn, &n, &d1, pD, &(sD->m), pD+n, &(sD->m));
#else
	dgemm_(&cn, &ct, &m, &n, &k, &d1, pA, &(sA->m), pB, &(sB->m), &d1, pD, &(sD->m));
	dpotrf_(&cl, &n, pD, &(sD->m), &info);
	dtrsm_(&cr, &cl, &ct, &cn, &mmn, &n, &d1, pD, &(sD->m), pD+n, &(sD->m));
#endif
	return;
	}



// dgetrf without pivoting
void dgetrf_nopivot_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	// TODO with custom level 2 LAPACK + level 3 BLAS
//	printf("\nfeature not implemented yet\n\n");
//	exit(1);
	int jj;
	int i1 = 1;
	double d1 = 1.0;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
#if defined(LA_BLAS_MKL)
			dcopy(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#else
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#endif
		}
	dgetf2_nopivot(m, n, pD, sD->m);
	return;
	}



// dgetrf pivoting
void dgetrf_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj, int *ipiv)
	{
	// TODO with custom level 2 LAPACK + level 3 BLAS
//	printf("\nfeature not implemented yet\n\n");
//	exit(1);
	int jj;
	int i1 = 1;
	double d1 = 1.0;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
#if defined(LA_BLAS_MKL)
			dcopy(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#else
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
#endif
		}
	int info;
#if defined(LA_BLAS_MKL)
	dgetrf(&m, &n, pD, &(sD->m), ipiv, &info);
#else
	dgetrf_(&m, &n, pD, &(sD->m), ipiv, &info);
#endif
	int tmp = m<n ? m : n;
	for(jj=0; jj<tmp; jj++)
		ipiv[jj] -= 1;
	return;
	}



#elif defined(LA_REFERENCE)



// dpotrf
void dpotrf_l_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int ii, jj, kk;
	double
		f_00_inv, 
		f_10, f_11_inv,
		c_00, c_01,
		c_10, c_11;
	int ldc = sC->m;
	int ldd = sD->m;
	double *pC = sC->pA + ci + cj*ldc;
	double *pD = sD->pA + di + dj*ldd;
	double *dD = sD->dA;
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		// factorize diagonal
		c_00 = pC[jj+0+ldc*(jj+0)];;
		c_10 = pC[jj+1+ldc*(jj+0)];;
		c_11 = pC[jj+1+ldc*(jj+1)];;
		for(kk=0; kk<jj; kk++)
			{
			c_00 -= pD[jj+0+ldd*kk] * pD[jj+0+ldd*kk];
			c_10 -= pD[jj+1+ldd*kk] * pD[jj+0+ldd*kk];
			c_11 -= pD[jj+1+ldd*kk] * pD[jj+1+ldd*kk];
			}
		if(c_00>0)
			{
			f_00_inv = 1.0/sqrt(c_00);
			}
		else
			{
			f_00_inv = 0.0;
			}
		dD[jj+0] = f_00_inv;
		pD[jj+0+ldd*(jj+0)] = c_00 * f_00_inv;
		f_10 = c_10 * f_00_inv;
		pD[jj+1+ldd*(jj+0)] = f_10;
		c_11 -= f_10 * f_10;
		if(c_11>0)
			{
			f_11_inv = 1.0/sqrt(c_11);
			}
		else
			{
			f_11_inv = 0.0;
			}
		dD[jj+1] = f_11_inv;
		pD[jj+1+ldd*(jj+1)] = c_11 * f_11_inv;
		// solve lower
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			c_00 = pC[ii+0+ldc*(jj+0)];
			c_10 = pC[ii+1+ldc*(jj+0)];
			c_01 = pC[ii+0+ldc*(jj+1)];
			c_11 = pC[ii+1+ldc*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pD[jj+0+ldd*kk];
				c_10 -= pD[ii+1+ldd*kk] * pD[jj+0+ldd*kk];
				c_01 -= pD[ii+0+ldd*kk] * pD[jj+1+ldd*kk];
				c_11 -= pD[ii+1+ldd*kk] * pD[jj+1+ldd*kk];
				}
			c_00 *= f_00_inv;
			c_10 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			pD[ii+1+ldd*(jj+0)] = c_10;
			c_01 -= c_00 * f_10;
			c_11 -= c_10 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01 * f_11_inv;
			pD[ii+1+ldd*(jj+1)] = c_11 * f_11_inv;
			}
		for(; ii<m; ii++)
			{
			c_00 = pC[ii+0+ldc*(jj+0)];
			c_01 = pC[ii+0+ldc*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pD[jj+0+ldd*kk];
				c_01 -= pD[ii+0+ldd*kk] * pD[jj+1+ldd*kk];
				}
			c_00 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			c_01 -= c_00 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01 * f_11_inv;
			}
		}
	for(; jj<n; jj++)
		{
		// factorize diagonal
		c_00 = pC[jj+ldc*jj];;
		for(kk=0; kk<jj; kk++)
			{
			c_00 -= pD[jj+ldd*kk] * pD[jj+ldd*kk];
			}
		if(c_00>0)
			{
			f_00_inv = 1.0/sqrt(c_00);
			}
		else
			{
			f_00_inv = 0.0;
			}
		dD[jj] = f_00_inv;
		pD[jj+ldd*jj] = c_00 * f_00_inv;
		// solve lower
		for(ii=jj+1; ii<m; ii++)
			{
			c_00 = pC[ii+ldc*jj];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+ldd*kk] * pD[jj+ldd*kk];
				}
			pD[ii+ldd*jj] = c_00 * f_00_inv;
			}
		}
	return;
	}



// dsyrk dpotrf
void dsyrk_dpotrf_ln_libstr(int m, int n, int k, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int ii, jj, kk;
	double
		f_00_inv, 
		f_10, f_11_inv,
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *pA = sA->pA + ai + aj*lda;
	double *pB = sB->pA + bi + bj*ldb;
	double *pC = sC->pA + ci + cj*ldc;
	double *pD = sD->pA + di + dj*ldd;
	double *dD = sD->dA;
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		// factorize diagonal
		c_00 = pC[jj+0+ldc*(jj+0)];;
		c_10 = pC[jj+1+ldc*(jj+0)];;
		c_11 = pC[jj+1+ldc*(jj+1)];;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+0+lda*kk] * pB[jj+0+ldb*kk];
			c_10 += pA[jj+1+lda*kk] * pB[jj+0+ldb*kk];
			c_11 += pA[jj+1+lda*kk] * pB[jj+1+ldb*kk];
			}
		for(kk=0; kk<jj; kk++)
			{
			c_00 -= pD[jj+0+ldd*kk] * pD[jj+0+ldd*kk];
			c_10 -= pD[jj+1+ldd*kk] * pD[jj+0+ldd*kk];
			c_11 -= pD[jj+1+ldd*kk] * pD[jj+1+ldd*kk];
			}
		if(c_00>0)
			{
			f_00_inv = 1.0/sqrt(c_00);
			}
		else
			{
			f_00_inv = 0.0;
			}
		dD[jj+0] = f_00_inv;
		pD[jj+0+ldd*(jj+0)] = c_00 * f_00_inv;
		f_10 = c_10 * f_00_inv;
		pD[jj+1+ldd*(jj+0)] = f_10;
		c_11 -= f_10 * f_10;
		if(c_11>0)
			{
			f_11_inv = 1.0/sqrt(c_11);
			}
		else
			{
			f_11_inv = 0.0;
			}
		dD[jj+1] = f_11_inv;
		pD[jj+1+ldd*(jj+1)] = c_11 * f_11_inv;
		// solve lower
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			c_00 = pC[ii+0+ldc*(jj+0)];
			c_10 = pC[ii+1+ldc*(jj+0)];
			c_01 = pC[ii+0+ldc*(jj+1)];
			c_11 = pC[ii+1+ldc*(jj+1)];
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_10 += pA[ii+1+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				c_11 += pA[ii+1+lda*kk] * pB[jj+1+ldb*kk];
				}
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pD[jj+0+ldd*kk];
				c_10 -= pD[ii+1+ldd*kk] * pD[jj+0+ldd*kk];
				c_01 -= pD[ii+0+ldd*kk] * pD[jj+1+ldd*kk];
				c_11 -= pD[ii+1+ldd*kk] * pD[jj+1+ldd*kk];
				}
			c_00 *= f_00_inv;
			c_10 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			pD[ii+1+ldd*(jj+0)] = c_10;
			c_01 -= c_00 * f_10;
			c_11 -= c_10 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01 * f_11_inv;
			pD[ii+1+ldd*(jj+1)] = c_11 * f_11_inv;
			}
		for(; ii<m; ii++)
			{
			c_00 = pC[ii+0+ldc*(jj+0)];
			c_01 = pC[ii+0+ldc*(jj+1)];
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				}
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pD[jj+0+ldd*kk];
				c_01 -= pD[ii+0+ldd*kk] * pD[jj+1+ldd*kk];
				}
			c_00 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			c_01 -= c_00 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01 * f_11_inv;
			}
		}
	for(; jj<n; jj++)
		{
		// factorize diagonal
		c_00 = pC[jj+ldc*jj];;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+lda*kk] * pB[jj+ldb*kk];
			}
		for(kk=0; kk<jj; kk++)
			{
			c_00 -= pD[jj+ldd*kk] * pD[jj+ldd*kk];
			}
		if(c_00>0)
			{
			f_00_inv = 1.0/sqrt(c_00);
			}
		else
			{
			f_00_inv = 0.0;
			}
		dD[jj] = f_00_inv;
		pD[jj+ldd*jj] = c_00 * f_00_inv;
		// solve lower
		for(ii=jj+1; ii<m; ii++)
			{
			c_00 = pC[ii+ldc*jj];
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+lda*kk] * pB[jj+ldb*kk];
				}
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+ldd*kk] * pD[jj+ldd*kk];
				}
			pD[ii+ldd*jj] = c_00 * f_00_inv;
			}
		}
	return;
	}



// dgetrf without pivoting
void dgetf2_nopivot(int m, int n, double *A, int lda, double *dA)
	{
	int ii, jj, kk, itmp0, itmp1;
	int iimax = m<n ? m : n;
	int i1 = 1;
	double dtmp;
	double dm1 = -1.0;

	for(ii=0; ii<iimax; ii++)
		{
		itmp0 = m-ii-1;
		dtmp = 1.0/A[ii+lda*ii];
		dA[ii] = dtmp;
		for(jj=0; jj<itmp0; jj++)
			{
			A[ii+1+jj+lda*ii] *= dtmp;
			}
		itmp1 = n-ii-1;
		for(jj=0; jj<itmp1; jj++)
			{
			for(kk=0; kk<itmp0; kk++)
				{
				A[(ii+1+kk)+lda*(ii+1+jj)] -= A[(ii+1+kk)+lda*ii] * A[ii+lda*(ii+1+jj)];
				}
			}
		}
	return;
	}



// dgetrf without pivoting
void dgetrf_nopivot_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int ii, jj, kk;
//	int i1 = 1;
//	double d1 = 1.0;
	double
		d_00_inv, d_11_inv,
		d_00, d_01,
		d_10, d_11;
	int ldc = sC->m;
	int ldd = sD->m;
	double *pC = sC->pA + ci + cj*ldc;
	double *pD = sD->pA + di + dj*ldd;
	double *dD = sD->dA;
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
#if 1
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		// upper
		ii = 0;
		for(; ii<jj-1; ii+=2)
			{
			// correct upper
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_10 = pC[(ii+1)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			d_11 = pC[(ii+1)+ldc*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve upper
			d_10 -= pD[(ii+1)+ldd*kk] * d_00;
			d_11 -= pD[(ii+1)+ldd*kk] * d_01;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			}
		for(; ii<jj; ii++)
			{
			// correct upper
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve upper
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			}
		// diagonal
		ii = jj;
		if(ii<m-1)
			{
			// correct diagonal
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_10 = pC[(ii+1)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			d_11 = pC[(ii+1)+ldc*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// factorize diagonal
			d_00_inv = 1.0/d_00;
			d_10 *= d_00_inv;
			d_11 -= d_10 * d_01;
			d_11_inv = 1.0/d_11;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			dD[ii+0] = d_00_inv;
			dD[ii+1] = d_11_inv;
			ii += 2;
			}
		else if(ii<m)
			{
			// correct diagonal
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// factorize diagonal
			d_00_inv = 1.0/d_00;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			dD[ii+0] = d_00_inv;
			ii += 1;
			}
		// lower
		for(; ii<m-1; ii+=2)
			{
			// correct lower
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_10 = pC[(ii+1)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			d_11 = pC[(ii+1)+ldc*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve lower
			d_00 *= d_00_inv;
			d_10 *= d_00_inv;
			d_01 -= d_00 * pD[kk+ldd*(jj+1)];
			d_11 -= d_10 * pD[kk+ldd*(jj+1)];
			d_01 *= d_11_inv;
			d_11 *= d_11_inv;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			// correct lower
			d_00 = pC[(ii+0)+ldc*(jj+0)];
			d_01 = pC[(ii+0)+ldc*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve lower
			d_00 *= d_00_inv;
			d_01 -= d_00 * pD[kk+ldd*(jj+1)];
			d_01 *= d_11_inv;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			}
		}
	for(; jj<n; jj++)
		{
		// upper
		ii = 0;
		for(; ii<jj-1; ii+=2)
			{
			// correct upper
			d_00 = pC[(ii+0)+ldc*jj];
			d_10 = pC[(ii+1)+ldc*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve upper
			d_10 -= pD[(ii+1)+ldd*kk] * d_00;
			pD[(ii+0)+ldd*jj] = d_00;
			pD[(ii+1)+ldd*jj] = d_10;
			}
		for(; ii<jj; ii++)
			{
			// correct upper
			d_00 = pC[(ii+0)+ldc*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve upper
			pD[(ii+0)+ldd*jj] = d_00;
			}
		// diagonal
		ii = jj;
		if(ii<m-1)
			{
			// correct diagonal
			d_00 = pC[(ii+0)+ldc*jj];
			d_10 = pC[(ii+1)+ldc*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*jj];
				}
			// factorize diagonal
			d_00_inv = 1.0/d_00;
			d_10 *= d_00_inv;
			pD[(ii+0)+ldd*jj] = d_00;
			pD[(ii+1)+ldd*jj] = d_10;
			dD[ii+0] = d_00_inv;
			ii += 2;
			}
		else if(ii<m)
			{
			// correct diagonal
			d_00 = pC[(ii+0)+ldc*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				}
			// factorize diagonal
			d_00_inv = 1.0/d_00;
			pD[(ii+0)+ldd*jj] = d_00;
			dD[ii+0] = d_00_inv;
			ii += 1;
			}
		// lower
		for(; ii<m-1; ii+=2)
			{
			// correct lower
			d_00 = pC[(ii+0)+ldc*jj];
			d_10 = pC[(ii+1)+ldc*jj];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve lower
			d_00 *= d_00_inv;
			d_10 *= d_00_inv;
			pD[(ii+0)+ldd*jj] = d_00;
			pD[(ii+1)+ldd*jj] = d_10;
			}
		for(; ii<m; ii++)
			{
			// correct lower
			d_00 = pC[(ii+0)+ldc*jj];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve lower
			d_00 *= d_00_inv;
			pD[(ii+0)+ldd*jj] = d_00;
			}
		}
#else
	if(pC!=pD)
		{
		for(jj=0; jj<n; jj++)
			{
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*jj] = pC[ii+ldc*jj];
				}
			}
		}
	dgetf2_nopivot(m, n, pD, ldd, dD);
#endif
	return;
	}



// dgetrf pivoting
void dgetrf_libstr(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj, int *ipiv)
	{
	int ii, i0, jj, kk, ip, itmp0, itmp1;
	double dtmp, dmax;
	double
		d_00_inv, d_11_inv,
		d_00, d_01,
		d_10, d_11;
	int i1 = 1;
	double d1 = 1.0;
	int ldc = sC->m;
	int ldd = sD->m;
	double *pC = sC->pA+ci+cj*ldc;
	double *pD = sD->pA+di+dj*ldd;
	double *dD = sD->dA;
	if(di==0 & dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	// copy if needed
	if(pC!=pD)
		{
		for(jj=0; jj<n; jj++)
			{
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*jj] = pC[ii+ldc*jj];
				}
			}
		}
	// factorize
#if 1
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<jj-1; ii+=2)
			{
			// correct upper
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_10 = pD[(ii+1)+ldd*(jj+0)];
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			d_11 = pD[(ii+1)+ldd*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve upper
			d_10 -= pD[(ii+1)+ldd*kk] * d_00;
			d_11 -= pD[(ii+1)+ldd*kk] * d_01;
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			}
		for(; ii<jj; ii++)
			{
			// correct upper
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			// solve upper
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			}
		// correct diagonal and lower and look for pivot
		// correct
		ii = jj;
		i0 = ii;
		for(; ii<m-1; ii+=2)
			{
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_10 = pD[(ii+1)+ldd*(jj+0)];
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			d_11 = pD[(ii+1)+ldd*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			}
		// look for pivot & solve
		// left column
		ii = i0;
		dmax = 0;
		ip = ii;
		for(; ii<m-1; ii+=2)
			{
			d_00 = pD[(ii+0)+ldd*jj];
			d_10 = pD[(ii+1)+ldd*jj];
			dtmp = d_00>0 ? d_00 : -d_00;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			dtmp = d_10>0 ? d_10 : -d_10;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+1;
				}
			}
		for(; ii<m; ii++)
			{
			d_00 = pD[(ii+0)+ldd*jj];
			dtmp = d_00>0 ? d_00 : -d_00;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			}
		// row swap
		ii = i0;
		ipiv[ii] = ip;
		if(ip!=ii)
			{
			for(kk=0; kk<n; kk++)
				{
				dtmp = pD[ii+ldd*kk];
				pD[ii+ldd*kk] = pD[ip+ldd*kk];
				pD[ip+ldd*kk] = dtmp;
				}
			}
		// factorize diagonal
		d_00 = pD[(ii+0)+ldd*(jj+0)];
		d_00_inv = 1.0/d_00;
		pD[(ii+0)+ldd*(jj+0)] = d_00;
		dD[ii] = d_00_inv;
		ii += 1;
		// solve & compute next pivot
		dmax = 0;
		ip = ii;
		for(; ii<m-1; ii+=2)
			{
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_10 = pD[(ii+1)+ldd*(jj+0)];
			d_00 *= d_00_inv;
			d_10 *= d_00_inv;
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			d_11 = pD[(ii+1)+ldd*(jj+1)];
			d_01 -= d_00 * pD[jj+ldd*(jj+1)];
			d_11 -= d_10 * pD[jj+ldd*(jj+1)];
			dtmp = d_01>0 ? d_01 : -d_01;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			dtmp = d_11>0 ? d_11 : -d_11;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+1;
				}
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+1)+ldd*(jj+0)] = d_10;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			pD[(ii+1)+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			d_00 = pD[(ii+0)+ldd*(jj+0)];
			d_00 *= d_00_inv;
			d_01 = pD[(ii+0)+ldd*(jj+1)];
			d_01 -= d_00 * pD[jj+ldd*(jj+1)];
			dtmp = d_01>0 ? d_01 : -d_01;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			pD[(ii+0)+ldd*(jj+0)] = d_00;
			pD[(ii+0)+ldd*(jj+1)] = d_01;
			}
		// row swap
		ii = i0+1;
		ipiv[ii] = ip;
		if(ip!=ii)
			{
			for(kk=0; kk<n; kk++)
				{
				dtmp = pD[ii+ldd*kk];
				pD[ii+ldd*kk] = pD[ip+ldd*kk];
				pD[ip+ldd*kk] = dtmp;
				}
			}
		// factorize diagonal
		d_00 = pD[ii+ldd*(jj+1)];
		d_00_inv = 1.0/d_00;
		pD[ii+ldd*(jj+1)] = d_00;
		dD[ii] = d_00_inv;
		ii += 1;
		// solve lower
		for(; ii<m; ii++)
			{
			d_00 = pD[ii+ldd*(jj+1)];
			d_00 *= d_00_inv;
			pD[ii+ldd*(jj+1)] = d_00;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<jj-1; ii+=2)
			{
			// correct upper
			d_00 = pD[(ii+0)+ldd*jj];
			d_10 = pD[(ii+1)+ldd*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve upper
			d_10 -= pD[(ii+1)+ldd*kk] * d_00;
			pD[(ii+0)+ldd*jj] = d_00;
			pD[(ii+1)+ldd*jj] = d_10;
			}
		for(; ii<jj; ii++)
			{
			// correct upper
			d_00 = pD[ii+ldd*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pD[ii+ldd*kk] * pD[kk+ldd*jj];
				}
			// solve upper
			pD[ii+ldd*jj] = d_00;
			}
		i0 = ii;
		ii = jj;
		// correct diagonal and lower and look for pivot
		dmax = 0;
		ip = ii;
		for(; ii<m-1; ii+=2)
			{
			d_00 = pD[(ii+0)+ldd*jj];
			d_10 = pD[(ii+1)+ldd*jj];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				d_10 -= pD[(ii+1)+ldd*kk] * pD[kk+ldd*jj];
				}
			dtmp = d_00>0 ? d_00 : -d_00;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			dtmp = d_10>0 ? d_10 : -d_10;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+1;
				}
			pD[(ii+0)+ldd*jj] = d_00;
			pD[(ii+1)+ldd*jj] = d_10;
			}
		for(; ii<m; ii++)
			{
			d_00 = pD[(ii+0)+ldd*jj];
			for(kk=0; kk<jj; kk++)
				{
				d_00 -= pD[(ii+0)+ldd*kk] * pD[kk+ldd*jj];
				}
			dtmp = d_00>0 ? d_00 : -d_00;
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+0;
				}
			pD[(ii+0)+ldd*jj] = d_00;
			}
		// row swap
		ii = i0;
		ipiv[ii] = ip;
		if(ip!=ii)
			{
			for(kk=0; kk<n; kk++)
				{
				dtmp = pD[ii+ldd*kk];
				pD[ii+ldd*kk] = pD[ip+ldd*kk];
				pD[ip+ldd*kk] = dtmp;
				}
			}
		// factorize diagonal
		d_00 = pD[ii+ldd*jj];
		d_00_inv = 1.0/d_00;
		pD[ii+ldd*jj] = d_00;
		dD[ii] = d_00_inv;
		ii += 1;
		for(; ii<m; ii++)
			{
			// correct lower
			d_00 = pD[ii+ldd*jj];
			// solve lower
			d_00 *= d_00_inv;
			pD[ii+ldd*jj] = d_00;
			}
		}
#else
	int iimax = m<n ? m : n;
	for(ii=0; ii<iimax; ii++)
		{
		dmax = (pD[ii+ldd*ii]>0 ? pD[ii+ldd*ii] : -pD[ii+ldd*ii]);
		ip = ii;
		for(jj=1; jj<m-ii; jj++)
			{
			dtmp = pD[ii+jj+ldd*ii]>0 ? pD[ii+jj+ldd*ii] : -pD[ii+jj+ldd*ii];
			if(dtmp>dmax)
				{
				dmax = dtmp;
				ip = ii+jj;
				}
			}
		ipiv[ii] = ip;
		if(ip!=ii)
			{
			for(jj=0; jj<n; jj++)
				{
				dtmp = pD[ii+jj*ldd];
				pD[ii+jj*ldd] = pD[ip+jj*ldd];
				pD[ip+jj*ldd] = dtmp;
				}
			}
		itmp0 = m-ii-1;
		dtmp = 1.0/pD[ii+ldd*ii];
		dD[ii] = dtmp;
		for(jj=0; jj<itmp0; jj++)
			{
			pD[ii+1+jj+ldd*ii] *= dtmp;
			}
		itmp1 = n-ii-1;
		for(jj=0; jj<itmp1; jj++)
			{
			for(kk=0; kk<itmp0; kk++)
				{
				pD[(ii+1+kk)+ldd*(ii+1+jj)] -= pD[(ii+1+kk)+ldd*ii] * pD[ii+ldd*(ii+1+jj)];
				}
			}
		}
#endif
	return;	
	}






#else

#error : wrong LA choice

#endif


