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



void dgemm_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, int tc, double *pC, int sdc, int td, double *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, l;

	if(tc==0)
		{
		if(td==0) // tc==0, td==0
			{

			i = 0;

#if defined(TARGET_X64_SANDY_BRIDGE)
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
					goto left_00_4;
					}
				else
					{
					goto left_00_8;
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
				goto left_00_4;
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions

#if defined(TARGET_X64_SANDY_BRIDGE)
			left_00_8:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_ntnn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
				}
			return;
#endif

			left_00_4:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_ntnn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
				}
			return;



			}
		else // tc==0, td==1
			{

			i = 0;
#if defined(TARGET_X64_SANDY_BRIDGE)
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
					goto left_01_4;
					}
				else
					{
					goto left_01_8;
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
				goto left_01_4;
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions

#if defined(TARGET_X64_SANDY_BRIDGE)
			left_01_8:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_ntnt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
				}
			return;
#endif

			left_01_4:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_ntnt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], m-i, n-j);
				}
			return;



			}
		}
	else // tc==1
		{
		if(td==0) // tc==1, td==0
			{

			i = 0;
#if defined(TARGET_X64_SANDY_BRIDGE)
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
					goto left_10_4;
					}
				else
					{
					goto left_10_8;
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
				goto left_10_4;
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions

#if defined(TARGET_X64_SANDY_BRIDGE)
			left_10_8:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_nttn_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, m-i, n-j);
				}
#endif

			left_10_4:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_nttn_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], m-i, n-j);
				}
			return;



			}
		else // tc==1, td==1
			{

			i = 0;
#if defined(TARGET_X64_SANDY_BRIDGE)
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
					goto left_11_4;
					}
				else
					{
					goto left_11_8;
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
				goto left_11_4;
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions

#if defined(TARGET_X64_SANDY_BRIDGE)
			left_11_8:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_nttt_8x4_vs_lib4(k, &pA[i*sda], sda, &pB[j*sdb], alg, &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, m-i, n-j);
				}
			return;
#endif

			left_11_4:
			j = 0;
			for(; j<n; j+=4)
				{
				kernel_dgemm_nttt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], m-i, n-j);
				}
			return;

			}
		}
	}




