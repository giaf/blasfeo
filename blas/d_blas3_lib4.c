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



void dgemm_ntnn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_ntnn_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
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
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_ntnn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
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

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	left_8:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void dgemm_ntnt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_ntnt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd);
			}
		if(j<n)
			{
			kernel_dgemm_ntnt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
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
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_ntnt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_ntnt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], 4, n-j);
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

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	left_8:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_ntnt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_ntnt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], m-i, n-j);
		}
	return;

	}



void dgemm_nttn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_nttn_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_dgemm_nttn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
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
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_nttn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_nttn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], 4, n-j);
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

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	left_8:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_nttn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
		}
#endif

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_nttn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void dgemm_nttt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_nttt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd);
			}
		if(j<n)
			{
			kernel_dgemm_nttt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
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
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_nttt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_nttt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], 4, n-j);
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

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	left_8:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_nttt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_dgemm_nttt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], m-i, n-j);
		}
	return;

	}



void dsyrk_ntnn_l_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_ntnn_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(j<i) // dgemm
				{
				kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, 8, n-j);
				}
			else // dsyrk
				{
				kernel_dsyrk_ntnn_l_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, 8, n-j);
				if(j<n-4)
					{
					kernel_dsyrk_ntnn_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], alg, &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], 4, n-j-4);
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
			kernel_dgemm_ntnn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			if(i<j) // dgemm
				{
				kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
				}
			else // dsyrk
				{
				kernel_dsyrk_ntnn_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
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

#if defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	left_8:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_ntnn_l_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
			if(j<n-4)
				{
				kernel_dsyrk_ntnn_l_4x4_vs_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], alg, &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], m-i-4, n-j-4);
				}
			}
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_dsyrk_ntnn_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	return;

	}



void dtrmm_ntnn_lu_lib(int m, int n, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
#if 0 //defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_ntnn_lu_8x4_lib4(n-j, &pA[j*bs+i*sda], sda, &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n) // TODO specialized edge routine
			{
			kernel_dtrmm_ntnn_lu_8x4_vs_lib4(n-j, &pA[j*bs+i*sda], sda, &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
			}
		}
	if(i<m)
		{
		if(m-i<5)
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
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_ntnn_lu_4x4_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n) // TODO specialized edge routine
			{
			kernel_dtrmm_ntnn_lu_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	if(i<m)
		{
		goto left_4;
		}
#endif
	
	// common return
	return;

#if 0 //defined(TARGET_X64_SANDY_BRIDGE) || defined(TARGET_X64_HASWELL)
	// clean up
	left_8:
	j = 0;
	for(; j<n-3; j+=4)
		{
		kernel_dtrmm_ntnn_lu_8x4_vs_lib4(n-j, &pA[j*bs+i*sda], sda, &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
		}
	if(j<n) // TODO specialized edge routine
		{
		kernel_dtrmm_ntnn_lu_8x4_vs_lib4(n-j, &pA[j*bs+i*sda], sda, &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<n-3; j+=4)
		{
		kernel_dtrmm_ntnn_lu_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	if(j<n) // TODO specialized edge routine
		{
		kernel_dtrmm_ntnn_lu_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}




