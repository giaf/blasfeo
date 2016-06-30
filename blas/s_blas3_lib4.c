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

#include "../include/blasfeo_s_kernel.h"



void sgemm_ntnn_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_ntnn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void sgemm_ntnt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_ntnt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_ntnt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], 4, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_ntnt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], m-i, n-j);
		}
	return;

	}



void sgemm_nttn_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nttn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_nttn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], 4, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nttn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void sgemm_nttt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nttt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_nttt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], 4, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nttt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], m-i, n-j);
		}
	return;

	}



void ssyrk_ntnn_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_sgemm_ntnn_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			if(i<j) // dgemm
				{
				kernel_sgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
				}
			else // dsyrk
				{
				kernel_ssyrk_ntnn_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], 4, n-j);
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_sgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_sgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_ssyrk_ntnn_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	return;

	}



void strmm_ntnn_ru_lib(int m, int n, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_strmm_ntnn_ru_4x4_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n) // TODO specialized edge routine
			{
			kernel_strmm_ntnn_ru_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	if(i<m)
		{
		goto left_4;
		}
	
	// common return
	return;

	left_4:
	j = 0;
//	for(; j<n-3; j+=4)
	for(; j<n; j+=4)
		{
		kernel_strmm_ntnn_ru_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
//	if(j<n) // TODO specialized edge routine
//		{
//		kernel_strmm_ntnn_ru_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
//		}
	return;

	}


