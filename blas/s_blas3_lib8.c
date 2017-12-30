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
#if defined(DIM_CHECK)
#include <stdio.h>
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_aux.h"



void sgemm_nt_libstr(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m==0 | n==0)
		return;
	
#if defined(DIM_CHECK)
	// TODO check that sA=!sD or that if sA==sD then they do not overlap (same for sB)
	// non-negative size
	if(m<0) printf("\n****** sgemm_nt_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgemm_nt_libstr : n<0 : %d<0 *****\n", n);
	if(k<0) printf("\n****** sgemm_nt_libstr : k<0 : %d<0 *****\n", k);
	// non-negative offset
	if(ai<0) printf("\n****** sgemm_nt_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgemm_nt_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgemm_nt_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgemm_nt_libstr : bj<0 : %d<0 *****\n", bj);
	if(ci<0) printf("\n****** sgemm_nt_libstr : ci<0 : %d<0 *****\n", ci);
	if(cj<0) printf("\n****** sgemm_nt_libstr : cj<0 : %d<0 *****\n", cj);
	if(di<0) printf("\n****** sgemm_nt_libstr : di<0 : %d<0 *****\n", di);
	if(dj<0) printf("\n****** sgemm_nt_libstr : dj<0 : %d<0 *****\n", dj);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** sgemm_nt_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+k > sA->n) printf("\n***** sgemm_nt_libstr : aj+k > col(A) : %d+%d > %d *****\n", aj, k, sA->n);
	// B: n x k
	if(bi+n > sB->m) printf("\n***** sgemm_nt_libstr : bi+n > row(B) : %d+%d > %d *****\n", bi, n, sB->m);
	if(bj+k > sB->n) printf("\n***** sgemm_nt_libstr : bj+k > col(B) : %d+%d > %d *****\n", bj, k, sB->n);
	// C: m x n
	if(ci+m > sC->m) printf("\n***** sgemm_nt_libstr : ci+m > row(C) : %d+%d > %d *****\n", ci, n, sC->m);
	if(cj+n > sC->n) printf("\n***** sgemm_nt_libstr : cj+n > col(C) : %d+%d > %d *****\n", cj, k, sC->n);
	// D: m x n
	if(di+m > sD->m) printf("\n***** sgemm_nt_libstr : di+m > row(D) : %d+%d > %d *****\n", di, n, sD->m);
	if(dj+n > sD->n) printf("\n***** sgemm_nt_libstr : dj+n > col(D) : %d+%d > %d *****\n", dj, k, sD->n);
#endif

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

#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
				if(j<n-4)
					{
					kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, 8, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, 8, n-j);
				}
			}
		}
	if(m-i>0)
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
//		else if(m-i<=20)
//			{
//			goto left_20;
//			}
		else
			{
			goto left_24;
			}
		}
#else
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
				if(j<n-4)
					{
					kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, 8, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, 8, n-j);
				}
			}
		}
	if(m-i>0)
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
		else
			{
			goto left_16;
			}
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

	left_24:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, 4);
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		}
	return;

#if defined(TARGET_X64_INTEL_HASWELL)
	left_20:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, 4);
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(i+16)*sdc], &pD[(j+0)*bs+(i+16)*sdd], m-(i+16), n-j);
		}
	if(j<n)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(i+16)*sdc], &pD[(j+0)*bs+(i+16)*sdd], m-(i+16), n-j);
		}
	return;
#endif

	left_16:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, 4);
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		}
	return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
left_12:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(i+8)*sdc], &pD[(j+0)*bs+(i+8)*sdd], m-(i+8), n-j);
		}
	if(j<n)
		{
		kernel_sgemm_nt_8x4_vs_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(i+8)*sdc], &pD[(j+0)*bs+(i+8)*sdd], m-(i+8), n-j);
		}
	return;
#endif

	left_8:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		kernel_sgemm_nt_8x4_vs_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		}
	return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	left_4:
	j = 0;
	for(; j<n; j+=8)
		{
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		}
	return;
#endif

	}



void sgemm_nn_libstr(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m==0 | n==0)
		return;
	
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgemm_nt_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgemm_nt_libstr : n<0 : %d<0 *****\n", n);
	if(k<0) printf("\n****** sgemm_nt_libstr : k<0 : %d<0 *****\n", k);
	// non-negative offset
	if(ai<0) printf("\n****** sgemm_nt_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgemm_nt_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgemm_nt_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgemm_nt_libstr : bj<0 : %d<0 *****\n", bj);
	if(ci<0) printf("\n****** sgemm_nt_libstr : ci<0 : %d<0 *****\n", ci);
	if(cj<0) printf("\n****** sgemm_nt_libstr : cj<0 : %d<0 *****\n", cj);
	if(di<0) printf("\n****** sgemm_nt_libstr : di<0 : %d<0 *****\n", di);
	if(dj<0) printf("\n****** sgemm_nt_libstr : dj<0 : %d<0 *****\n", dj);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** sgemm_nn_libstr : ai+m > row(A) : %d+%d > %d *****\n\n", ai, m, sA->m);
	if(aj+k > sA->n) printf("\n***** sgemm_nn_libstr : aj+k > col(A) : %d+%d > %d *****\n\n", aj, k, sA->n);
	// B: k x n
	if(bi+k > sB->m) printf("\n***** sgemm_nn_libstr : bi+k > row(B) : %d+%d > %d *****\n\n", bi, k, sB->m);
	if(bj+n > sB->n) printf("\n***** sgemm_nn_libstr : bj+n > col(B) : %d+%d > %d *****\n\n", bj, n, sB->n);
	// C: m x n
	if(ci+m > sC->m) printf("\n***** sgemm_nn_libstr : ci+m > row(C) : %d+%d > %d *****\n\n", ci, n, sC->m);
	if(cj+n > sC->n) printf("\n***** sgemm_nn_libstr : cj+n > col(C) : %d+%d > %d *****\n\n", cj, k, sC->n);
	// D: m x n
	if(di+m > sD->m) printf("\n***** sgemm_nn_libstr : di+m > row(D) : %d+%d > %d *****\n\n", di, n, sD->m);
	if(dj+n > sD->n) printf("\n***** sgemm_nn_libstr : dj+n > col(D) : %d+%d > %d *****\n\n", dj, k, sD->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs + bi/bs*bs*sdb;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	int offsetB = bi%bs;

	int i, j, l;

	i = 0;

#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nn_24x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nn_24x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nn_24x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
				if(j<n-4)
					{
					kernel_sgemm_nn_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, 16, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nn_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, 16, n-j);
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
#if 1
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nn_16x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nn_16x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nn_16x4_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
				if(j<n-4)
					{
					kernel_sgemm_nn_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, 16, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nn_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, 16, n-j);
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
#else
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
#if 1
			kernel_sgemm_nn_8x8_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
#else
			kernel_sgemm_nn_8x4_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
			kernel_sgemm_nn_8x4_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], &pD[(j+4)*bs+i*sdd]);
#endif
			}
		if(j<n)
			{
			if(j<n-3)
				{
				kernel_sgemm_nn_8x4_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd]);
				if(j<n-4)
					{
					kernel_sgemm_nn_8x4_gen_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+4)*bs], sdb, &beta, 0, &pC[(j+4)*bs+i*sdc], sdc, 0, &pD[(j+4)*bs+i*sdd], sdd, 0, 8, 0, n-(j+4));
					}
				}
			else
				{
				kernel_sgemm_nn_8x4_gen_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, 0, &pC[(j+0)*bs+i*sdc], sdc, 0, &pD[(j+0)*bs+i*sdd], sdd, 0, 8, 0, n-j);
				}
			}
		}
	if(m>i)
		{
		goto left_8;
		}
#endif
#endif

	// common return if i==m
	return;

#if defined(TARGET_X64_INTEL_HASWELL)
	left_24:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nn_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		kernel_sgemm_nn_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		kernel_sgemm_nn_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		}
	return;
#endif

	left_16:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nn_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		kernel_sgemm_nn_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+4)*bs], sdb, &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		kernel_sgemm_nn_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-j);
		}
	return;

	left_8:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_sgemm_nn_8x8_vs_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		kernel_sgemm_nn_8x4_vs_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[(j+0)*bs], sdb, &beta, &pC[(j+0)*bs+i*sdc], &pD[(j+0)*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void ssyrk_ln_libstr(int m, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ci>0 | di>0)
		{
		printf("\nssyrk_ln_libstr: feature not implemented yet: ci>0, di>0\n");
		exit(1);
		}

	const int bs = 8;

	int i, j;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	i = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<i; j+=8)
			{
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}

		kernel_ssyrk_nt_l_24x4_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd);
		kernel_ssyrk_nt_l_20x4_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd);
		kernel_ssyrk_nt_l_16x4_lib8(k, &alpha, &pA[(j+8)*sda], sda, &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd);
		kernel_ssyrk_nt_l_12x4_lib8(k, &alpha, &pA[(j+8)*sda], sda, &pB[4+(j+8)*sdb], &beta, &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd);
		kernel_ssyrk_nt_l_8x8_lib8(k, &alpha, &pA[(j+16)*sda], &pB[0+(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd]);
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
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		kernel_ssyrk_nt_l_16x4_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd);
		kernel_ssyrk_nt_l_12x4_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd);
		kernel_ssyrk_nt_l_8x8_lib8(k, &alpha, &pA[(j+8)*sda], &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd]);
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
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, m-(j+0));
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, m-(j+4));
		}
	kernel_ssyrk_nt_l_24x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, m-(i+0), m-(j+0));
	kernel_ssyrk_nt_l_20x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, m-(i+0), m-(j+4));
	kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(j+8)*sda], sda, &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, m-(i+8), m-(j+8));
	kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(j+8)*sda], sda, &pB[4+(j+8)*sdb], &beta, &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, m-(i+8), m-(j+12));
	if(j<m-20) // 21 - 23
		{
		kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(j+16)*sda], &pB[0+(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), m-(j+16));
		}
	else // 17 18 19 20
		{
		kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(j+16)*sda], &pB[0+(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), m-(j+16));
		}
	return;
#endif

	left_16: // 13 <= m <= 16
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, m-(j+0));
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, m-(j+4));
		}
	kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, m-(i+0), m-(j+0));
	kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, m-(i+0), m-(j+4));
	if(j<m-12) // 13 - 16
		{
		kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(j+8)*sda], &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), m-(j+8));
		}
	else // 9 - 12
		{
		kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(j+8)*sda], &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), m-(j+8));
		}
	return;

	left_12: // 9 <= m <= 12
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[(i+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(i+0)*sdc], &pD[(j+0)*bs+(i+0)*sdd], m-(i+0), m-(j+0));
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(i+8)*sdc], &pD[(j+0)*bs+(i+8)*sdd], m-(i+0), m-(j+0));
		}
	kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(j+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], &pD[(j+0)*bs+(j+0)*sdd], m-(i+0), m-(j+0));
	kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(j+8)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+8)*sdc], &pD[(j+0)*bs+(j+8)*sdd], m-(i+8), m-(j+0));
	if(j<m-8) // 9 - 12
		{
		kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(j+8)*sda], &pB[0+(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), m-(j+8));
		}
	return;

	left_8: // 5 <= m <= 8
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[(i+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(i+0)*sdc], &pD[(j+0)*bs+(i+0)*sdd], m-(i+0), m-(j+0));
		}
	if(j<m-4) // 5 - 8
		{
		kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(j+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], &pD[(j+0)*bs+(j+0)*sdd], m-(i+0), m-(j+0));
		}
	else // 1 - 4
		{
		kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], &pD[(j+0)*bs+(j+0)*sdd], m-(i+0), m-(j+0));
		}
	return;

	left_4: // 1 <= m <= 4
	j = 0;
	for(; j<i; j+=8)
		{
		kernel_sgemm_nt_4x8_vs_lib8(k, &alpha, &pA[(i+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(i+0)*sdc], &pD[(j+0)*bs+(i+0)*sdd], m-(i+0), m-(j+0));
		}
	kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(j+0)*sda], &pB[0+(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], &pD[(j+0)*bs+(j+0)*sdd], m-(i+0), m-(j+0));
	return;

	}



void ssyrk_ln_mn_libstr(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ci>0 | di>0)
		{
		printf("\nssyrk_ln_mn_libstr: feature not implemented yet: ci>0, di>0\n");
		exit(1);
		}

	const int bs = 8;

	int i, j;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	i = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; i<m-23; i+=24)
		{
		j = 0;
		for(; j<i & j<n-7; j+=8)
			{
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_24x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
				if(j<n-4) // 5 6 7
					{
					kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
					}
				}
			else // dpotrf
				{
				if(j<n-23)
					{
					kernel_ssyrk_nt_l_24x4_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd);
					kernel_ssyrk_nt_l_20x4_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd);
					kernel_ssyrk_nt_l_16x4_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd);
					kernel_ssyrk_nt_l_12x4_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[4+(j+8)*sdb], &beta, &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd);
					kernel_ssyrk_nt_l_8x8_lib8(k, &alpha, &pA[(i+16)*sda], &pB[(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd]);
					}
				else
					{
					if(j<n-4) // 5 - 23
						{
						kernel_ssyrk_nt_l_24x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+0));
						kernel_ssyrk_nt_l_20x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+4));
						if(j==n-8)
							return;
						if(j<n-12) // 13 - 23
							{
							kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+8));
							kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[4+(j+8)*sdb], &beta, &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+12));
							if(j==n-16)
								return;
							if(j<n-20) // 21 - 23
								{
								kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), n-(j+16));
								}
							else // 17 18 19 20
								{
								kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), n-(j+16));
								}
							}
						else // 9 10 11 12
							{
							kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+8));
							}
						}
					else // 1 2 3 4
						{
						kernel_ssyrk_nt_l_24x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, m-(i+0), n-j);
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
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd);
			kernel_sgemm_nt_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
				if(j<n-4) // 5 6 7
					{
					kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
					}
				}
			else // dpotrf
				{
				if(j<n-15)
					{
					kernel_ssyrk_nt_l_16x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd);
					kernel_ssyrk_nt_l_12x4_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd);
					kernel_ssyrk_nt_l_8x8_lib8(k, &alpha, &pA[(i+8)*sda], &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd]);
					}
				else
					{
					if(j<n-4) // 5 - 15
						{
						kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+0));
						kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+4));
						if(j==n-8) // 8
							return;
						if(j<n-12) // 13 - 15
							{
							kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), n-(j+8));
							}
						else // 9 10 11 12
							{
							kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), n-(j+8));
							}
						}
					else // 1 2 3 4
						{
						kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, m-(i+0), n-j);
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
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
		kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
			if(j<n-4) // 5 6 7
				{
				kernel_sgemm_nt_24x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 - 23
				{
				kernel_ssyrk_nt_l_24x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[(j+0)*sdb], &beta, &pC[(j+0)*bs+(j+0)*sdc], sdc, &pD[(j+0)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+0));
				kernel_ssyrk_nt_l_20x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[4+(j+0)*sdb], &beta, &pC[(j+4)*bs+(j+0)*sdc], sdc, &pD[(j+4)*bs+(j+0)*sdd], sdd, m-(i+0), n-(j+4));
				if(j>=n-8)
					return;
				if(j<n-12) // 13 - 23
					{
					kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+8));
					kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[4+(j+8)*sdb], &beta, &pC[(j+12)*bs+(j+8)*sdc], sdc, &pD[(j+12)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+12));
					if(j>=n-16)
						return;
					if(j<n-20) // 21 - 23
						{
						kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), n-(j+16));
						}
					else // 17 18 19 20
						{
						kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(i+16)*sda], &pB[(j+16)*sdb], &beta, &pC[(j+16)*bs+(j+16)*sdc], &pD[(j+16)*bs+(j+16)*sdd], m-(i+16), n-(j+16));
						}
					}
				else // 9 10 11 12
					{
					kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], sda, &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], sdc, &pD[(j+8)*bs+(j+8)*sdd], sdd, m-(i+8), n-(j+8));
					}
				}
			else // 1 2 3 4
				{
				kernel_ssyrk_nt_l_24x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, m-(i+0), n-j);
				}
			}
		}
	return;
#endif

	left_16:
	j = 0;
	for(; j<i & j<n-7; j+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
		kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+i*sdc], sdc, &pD[(j+0)*bs+i*sdd], sdd, m-i, n-(j+0));
			if(j<n-4) // 5 6 7
				{
				kernel_sgemm_nt_16x4_vs_lib8(k, &alpha, &pA[i*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+i*sdc], sdc, &pD[(j+4)*bs+i*sdd], sdd, m-i, n-(j+4));
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 - 15
				{
				kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[0+j*sdb], &beta, &pC[(j+0)*bs+j*sdc], sdc, &pD[(j+0)*bs+j*sdd], sdd, m-(i+0), n-(j+0));
				kernel_ssyrk_nt_l_12x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[4+j*sdb], &beta, &pC[(j+4)*bs+j*sdc], sdc, &pD[(j+4)*bs+j*sdd], sdd, m-(i+0), n-(j+4));
				if(j>=n-8)
					return;
				if(j<n-12) // 13 - 15
					{
					kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), n-(j+8));
					}
				else // 9 - 12
					{
					kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[(i+8)*sda], &pB[(j+8)*sdb], &beta, &pC[(j+8)*bs+(j+8)*sdc], &pD[(j+8)*bs+(j+8)*sdd], m-(i+8), n-(j+8));
					}
				}
			else // 1 2 3 4
				{
				kernel_ssyrk_nt_l_16x4_vs_lib8(k, &alpha, &pA[(i+0)*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+j*sdc], sdc, &pD[j*bs+j*sdd], sdd, m-(i+0), n-j);
				}
			}
		}
	return;

	left_8:
	j = 0;
	for(; j<i & j<n-7; j+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			if(j<n-4) // 5 6 7
				{
				kernel_sgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
				}
			else // 1 2 3 4
				{
				kernel_sgemm_nt_8x4_vs_lib8(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
				}
			}
		else // dpotrf
			{
			if(j<n-4) // 5 6 7
				{
				kernel_ssyrk_nt_l_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], m-i, n-j);
				}
			else // 1 2 3 4
				{
				kernel_ssyrk_nt_l_8x4_vs_lib8(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], m-i, n-j);
				}
			}
		}
	return;

	}



// dtrmm_right_lower_nottransposed_notunit (B, i.e. the first matrix, is triangular !!!)
void strmm_rlnn_libstr(int m, int n, float alpha, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sD, int di, int dj)
	{

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;

	pA += ai/bs*bs*sda;
	pB += bi/bs*bs*sdb;
	int offsetB = bi%bs;
	int di0 = di-ai%bs;
	int offsetD;
	if(di0>=0)
		{
		pD += di0/bs*bs*sdd;
		offsetD = di0%bs;
		}
	else
		{
		pD += -8*sdd;
		offsetD = bs+di0;
		}
	
	int ii, jj;

	int offsetB4;

	if(offsetB<4)
		{
		offsetB4 = offsetB+4;
		ii = 0;
		if(ai%bs!=0)
			{
			jj = 0;
			for(; jj<n-4; jj+=8)
				{
				kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, ai%bs, m-ii, 0, n-jj);
				kernel_strmm_nn_rl_8x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, ai%bs, m-ii, 0, n-jj-4);
				}
			m -= bs-ai%bs;
			pA += bs*sda;
			pD += bs*sdd;
			}
		if(offsetD==0)
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			// XXX create left_24 once the _gen_ kernel exist !!!
			for(; ii<m-23; ii+=24)
				{
				jj = 0;
				for(; jj<n-7; jj+=8)
					{
					kernel_strmm_nn_rl_24x4_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd);
					kernel_strmm_nn_rl_24x4_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd);
					}
				if(n-jj>0)
					{
					kernel_strmm_nn_rl_24x4_vs_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd, 24, n-jj);
					if(n-jj>4)
						{
						kernel_strmm_nn_rl_24x4_vs_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd, 24, n-jj-4);
						}
					}
				}
#endif
			for(; ii<m-15; ii+=16)
				{
				jj = 0;
				for(; jj<n-7; jj+=8)
					{
					kernel_strmm_nn_rl_16x4_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd);
					kernel_strmm_nn_rl_16x4_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd);
					}
				if(n-jj>0)
					{
					kernel_strmm_nn_rl_16x4_vs_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd, 16, n-jj);
					if(n-jj>4)
						{
						kernel_strmm_nn_rl_16x4_vs_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd, 16, n-jj-4);
						}
					}
				}
			if(m-ii>0)
				{
				if(m-ii<=8)
					goto left_8;
				else
					goto left_16;
				}
			}
		else
			{
			for(; ii<m-8; ii+=16)
				{
				jj = 0;
				for(; jj<n-4; jj+=8)
					{
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
					}
				if(n-jj>0)
					{
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
					}
				}
			if(m-ii>0)
				goto left_8;
			}
		}
	else
		{
		offsetB4 = offsetB-4;
		ii = 0;
		if(ai%bs!=0)
			{
			jj = 0;
			for(; jj<n-4; jj+=8)
				{
				kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, ai%bs, m-ii, 0, n-jj);
				kernel_strmm_nn_rl_8x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, ai%bs, m-ii, 0, n-jj-4);
				}
			m -= bs-ai%bs;
			pA += bs*sda;
			pD += bs*sdd;
			}
		if(offsetD==0)
			{
			for(; ii<m-15; ii+=16)
				{
				jj = 0;
				for(; jj<n-7; jj+=8)
					{
					kernel_strmm_nn_rl_16x4_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd);
					kernel_strmm_nn_rl_16x4_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd);
					}
				if(n-jj>0)
					{
					kernel_strmm_nn_rl_16x4_vs_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs], sdd, 8, n-jj);
					if(n-jj>4)
						{
						kernel_strmm_nn_rl_16x4_vs_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, &pD[ii*sdd+(jj+4)*bs], sdd, 8, n-jj-4);
						}
					}
				}
			if(m-ii>0)
				{
				if(m-ii<=8)
					goto left_8;
				else
					goto left_16;
				}
			}
		else
			{
			for(; ii<m-8; ii+=16)
				{
				jj = 0;
				for(; jj<n-4; jj+=8)
					{
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
					}
				if(n-jj>0)
					{
					kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
					}
				}
			if(m-ii>0)
				goto left_8;
			}
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_16:
	if(offsetB<4)
		{
		jj = 0;
		for(; jj<n-4; jj+=8)
			{
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
			}
		if(n-jj>0)
			{
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			}
		}
	else
		{
		jj = 0;
		for(; jj<n-4; jj+=8)
			{
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], sda, offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
			}
		if(n-jj>0)
			{
			kernel_strmm_nn_rl_16x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], sda, offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			}
		}
	return;

	left_8:
	if(offsetB<4)
		{
		jj = 0;
		for(; jj<n-4; jj+=8)
			{
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], offsetB4, &pB[jj*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
			}
		if(n-jj>0)
			{
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			}
		}
	else
		{
		jj = 0;
		for(; jj<n-4; jj+=8)
			{
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj-4, &alpha, &pA[ii*sda+(jj+4)*bs], offsetB4, &pB[(jj+8)*sdb+(jj+4)*bs], sdb, offsetD, &pD[ii*sdd+(jj+4)*bs], sdd, 0, m-ii, 0, n-jj-4);
			}
		if(n-jj>0)
			{
			kernel_strmm_nn_rl_8x4_gen_lib8(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
			}
		}
	return;

	}



// dtrsm_right_lower_transposed_notunit
void strsm_rltn_libstr(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_rltn_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	const int bs = 8;

	// TODO alpha

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	float *dA = sA->dA;

	int i, j;
	
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(n, 1.0, ai, pA, sda, dA);
			for(i=0; i<n; i++)
				dA[i] = 1.0 / dA[i];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(n, 1.0, ai, pA, sda, dA);
		for(i=0; i<n; i++)
			dA[i] = 1.0 / dA[i];
		sA->use_dA = 0;
		}

	if(m<=0 || n<=0)
		return;
	
	i = 0;

	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_strsm_nt_rl_inv_8x4_lib8(j+0, &pD[i*sdd], &pA[0+j*sda], &pB[(j+0)*bs+i*sdb], &pD[(j+0)*bs+i*sdd], &pA[0+(j+0)*bs+j*sda], &dA[j+0]);
			kernel_strsm_nt_rl_inv_8x4_lib8(j+4, &pD[i*sdd], &pA[4+j*sda], &pB[(j+4)*bs+i*sdb], &pD[(j+4)*bs+i*sdd], &pA[4+(j+4)*bs+j*sda], &dA[j+0]);
			}
		if(n-j>0)
			{
			kernel_strsm_nt_rl_inv_8x4_vs_lib8(j+0, &pD[i*sdd], &pA[0+j*sda], &pB[(j+0)*bs+i*sdb], &pD[(j+0)*bs+i*sdd], &pA[0+(j+0)*bs+j*sda], &dA[j+0], m-i, n-j-0);
			if(n-j>4)
				{
				kernel_strsm_nt_rl_inv_8x4_vs_lib8(j+4, &pD[i*sdd], &pA[4+j*sda], &pB[(j+4)*bs+i*sdb], &pD[(j+4)*bs+i*sdd], &pA[4+(j+4)*bs+j*sda], &dA[j+4], m-i, n-j-4);
				}
			}
		}
	if(m>i)
		{
		goto left_8;
		}

	// common return if i==m
	return;

	left_8:
	j = 0;
	for(; j<n-4; j+=8)
		{
		kernel_strsm_nt_rl_inv_8x4_vs_lib8(j+0, &pD[i*sdd], &pA[0+j*sda], &pB[(j+0)*bs+i*sdb], &pD[(j+0)*bs+i*sdd], &pA[0+(j+0)*bs+j*sda], &dA[j+0], m-i, n-j-0);
		kernel_strsm_nt_rl_inv_8x4_vs_lib8(j+4, &pD[i*sdd], &pA[4+j*sda], &pB[(j+4)*bs+i*sdb], &pD[(j+4)*bs+i*sdd], &pA[4+(j+4)*bs+j*sda], &dA[j+4], m-i, n-j-4);
		}
	if(n-j>0)
		{
		kernel_strsm_nt_rl_inv_8x4_vs_lib8(j+0, &pD[i*sdd], &pA[0+j*sda], &pB[(j+0)*bs+i*sdb], &pD[(j+0)*bs+i*sdd], &pA[0+(j+0)*bs+j*sda], &dA[j+0], m-i, n-j-0);
		}
	return;

	}




