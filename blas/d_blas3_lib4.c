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

			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 0, &pD[j*bs+i*sdd]);
					}
				if(j<n)
					{
//					if(n-j==3)
//						{
						kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 0, &pD[j*bs+i*sdd], 4, n-j);
//						}
//					else
//						{
//						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
//						}
					}
				}
			if(m>i)
				{
//				if(m-i==3)
//					{
					goto left_00_4;
//					}
//				else // m-i==2 || m-i==1
//					{
//					goto left_00_2;
//					}
				}

			// common return if i==m
			return;

			// clean up loops definitions
			left_00_4:
			j = 0;
//			for(; j<n-2; j+=4)
			for(; j<n; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 0, &pD[j*bs+i*sdd], m-i, n-j);
				}
//			if(j<n)
//				{
//				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
//				}
			return;



			}
		else // tc==0, td==1
			{



			i = 0;
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 1, &pD[i*bs+j*sdd]);
					}
				if(j<n)
					{
//					if(n-j==3)
//						{
						kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 1, &pD[i*bs+j*sdd], 4, n-j);
//						}
//					else
//						{
//						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
//						}
					}
				}
			if(m>i)
				{
//				if(m-i==3)
//					{
					goto left_01_4;
//					}
//				else // m-i==2 || m-i==1
//					{
//					goto left_01_2;
//					}
				}

			// common return if i==m
			return;

			// clean up loops definitions
			left_01_4:
			j = 0;
//			for(; j<n-2; j+=4)
			for(; j<n; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 0, &pC[j*bs+i*sdc], 1, &pD[i*bs+j*sdd], m-i, n-j);
				}
//			if(j<n)
//				{
//				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
//				}
			return;



			}
		}
	else // tc==1
		{
		if(td==0) // tc==1, td==0
			{



			i = 0;
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 0, &pD[j*bs+i*sdd]);
					}
				if(j<n)
					{
///					if(n-j==3)
///						{
						kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 0, &pD[j*bs+i*sdd], 4, n-j);
///						}
///					else
///						{
///						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
///						}
					}
				}
			if(m>i)
				{
//				if(m-i==3)
//					{
					goto left_10_4;
//					}
//				else // m-i==2 || m-i==1
//					{
//					goto left_10_2;
//					}
				}

			// common return if i==m
			return;

			// clean up loops definitions
			left_10_4:
			j = 0;
//			for(; j<n-2; j+=4)
			for(; j<n; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 0, &pD[j*bs+i*sdd], m-i, n-j);
				}
//			if(j<n)
//				{
//				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
//				}
			return;



			}
		else // tc==1, td==1
			{



			i = 0;
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 1, &pD[i*bs+j*sdd]);
					}
				if(j<n)
					{
//					if(n-j==3)
//						{
						kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 1, &pD[i*bs+j*sdd], 4, n-j);
//						}
//					else
//						{
//						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
//						}
					}
				}
			if(m>i)
				{
//				if(m-i==3)
//					{
					goto left_11_4;
//					}
//				else // m-i==2 || m-i==1
//					{
//					goto left_11_2;
//					}
				}

			// common return if i==m
			return;

			// clean up loops definitions
			left_11_4:
			j = 0;
//			for(; j<n-2; j+=4)
			for(; j<n; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], alg, 1, &pC[i*bs+j*sdc], 1, &pD[i*bs+j*sdd], m-i, n-j);
				}
//			if(j<n)
//				{
//				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
//				}
			return;



			}
		}
	}




