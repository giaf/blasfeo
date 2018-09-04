/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_target.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"



#define KMAX 256



void blasfeo_dgemm(char *ta, char *tb, int *pm, int *pn, int *pk, double *alpha, double *A, int *plda, double *B, int *pldb, double *beta, double *C, int *pldc)
	{

	int m = *pm;
	int n = *pn;
	int k = *pk;
	int lda = *plda;
	int ldb = *pldb;
	int ldc = *pldc;

	int ii, jj;

	int bs = 4;

// TODO visual studio alignment
#if defined(TARGET_X64_INTEL_HASWELL)
	double pU[3*4*KMAX] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	double pU[2*4*KMAX] __attribute__ ((aligned (64)));
#elif defined(TARGET_GENERIC)
	double pU[1*4*KMAX];
#else
	double pU[1*4*KMAX] __attribute__ ((aligned (64)));
#endif
	int sdu = (m+3)/4*4;
	sdu = sdu<KMAX ? sdu : KMAX;

	struct blasfeo_dmat sA, sB;
	int sda, sdb;
	int sA_size, sB_size;
	void *smat_mem, *smat_mem_align;

	if(*ta=='n' | *ta=='N')
		{
		if(*tb=='n' | *tb=='N')
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			if(n>=256 | k>=256 | k>KMAX)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			if(n>=56 | k>=56 | k>KMAX)
#elif defined(TARGET_X64_INTEL_CORE)
			if(n>=8 | k>=8 | k>KMAX)
#else
			if(n>=12 | k>=12 | k>KMAX)
#endif
				{
				goto nn_1;
				}
			else
				{
				goto nn_0;
				}
			}
		else if(*tb=='t' | *tb=='T')
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			if(n>=96 | k>=96 | k>KMAX)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			if(n>=56 | k>=56 | k>KMAX)
#elif defined(TARGET_X64_INTEL_CORE)
			if(n>=8 | k>=8 | k>KMAX)
#else
			if(n>=12 | k>=12 | k>KMAX)
#endif
				{
				goto nt_1;
				}
			else
				{
				goto nt_0;
				}
			}
		else
			{
			printf("\nBLASFEO: dpotrf: wrong value for tb\n");
			return;
			}
		}
	else if(*ta=='t' | *ta=='T')
		{
		if(*tb=='n' | *tb=='N')
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			if(n>=256 | k>=256 | k>KMAX)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			if(n>=56 | k>=56 | k>KMAX)
#elif defined(TARGET_X64_INTEL_CORE)
			if(n>=8 | k>=8 | k>KMAX)
#else
			if(n>=12 | k>=12 | k>KMAX)
#endif
				{
				goto tn_1;
				}
			else
				{
				goto tn_0;
				}
			}
		else if(*tb=='t' | *tb=='T')
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			if(n>=96 | k>=96 | k>KMAX)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			if(n>=56 | k>=56 | k>KMAX)
#elif defined(TARGET_X64_INTEL_CORE)
			if(n>=8 | k>=8 | k>KMAX)
#else
			if(n>=12 | k>=12 | k>KMAX)
#endif
				{
				goto tt_1;
				}
			else
				{
				goto tt_0;
				}
			}
		else
			{
			printf("\nBLASFEO: dpotrf: wrong value for tb\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dpotrf: wrong value for ta\n");
		return;
		}

	return;
	


nn_0:
	
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_0_left_4;
			}
		if(m-ii<=8)
			{
			goto nn_0_left_8;
			}
		else
			{
			goto nn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_0_left_4;
			}
		else
			{
			goto nn_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_0_left_4;
		}
#endif
	goto nn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nn_0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nn_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_0_return;
#endif

nn_0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_0_return;

nn_0_return:
	return;



nn_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(n, k);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(n, k, &sB, smat_mem_align+sA_size);

	blasfeo_pack_tran_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_1_left_4;
			}
		if(m-ii<=8)
			{
			goto nn_1_left_8;
			}
		else
			{
			goto nn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_1_left_4;
			}
		else
			{
			goto nn_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_1_left_4;
		}
#endif
	goto nn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nn_1_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, sA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nn_1_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, sA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_1_return;
#endif

nn_1_left_4:
	kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nn_1_return;

nn_1_return:
	free(smat_mem);
	return;



nt_0:

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_0_left_4;
			}
		if(m-ii<=8)
			{
			goto nt_0_left_8;
			}
		else
			{
			goto nt_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_0_left_4;
			}
		else
			{
			goto nt_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_0_left_4;
		}
#endif
	goto nt_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii+0, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nt_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii+0, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_0_return;
#endif

nt_0_left_4:
	kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_0_return;

nt_0_return:
	return;



nt_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(k, n);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(k, n, &sB, smat_mem_align+sA_size);

	blasfeo_pack_dmat(k, n, B, ldb, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_1_left_4;
			}
		if(m-ii<=8)
			{
			goto nt_1_left_8;
			}
		else
			{
			goto nt_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_1_left_4;
			}
		else
			{
			goto nt_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_1_left_4;
		}
#endif
	goto nt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_1_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, sA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nt_1_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, sA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_1_return;
#endif

nt_1_left_4:
	kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto nt_1_return;

nt_1_return:
	free(smat_mem);
	return;



tn_0:

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_0_left_4;
			}
		if(m-ii<=8)
			{
			goto tn_0_left_8;
			}
		else
			{
			goto tn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_0_left_4;
			}
		else
			{
			goto tn_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tn_0_left_4;
		}
#endif
	goto tn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tn_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tn_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_0_return;
#endif

tn_0_left_4:
	kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_0_return;

tn_0_return:
	return;



tn_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(n, k);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(n, k, &sB, smat_mem_align+sA_size);

	blasfeo_pack_tran_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_1_left_4;
			}
		if(m-ii<=8)
			{
			goto tn_1_left_8;
			}
		else
			{
			goto tn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_1_left_4;
			}
		else
			{
			goto tn_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
		if(ii<m)
		{
		goto tn_1_left_4;
		}
#endif
	goto tn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tn_1_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tn_1_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_1_return;
#endif

tn_1_left_4:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tn_1_return;

tn_1_return:
free(smat_mem);
	return;



tt_0:

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_0_left_4;
			}
		if(m-ii<=8)
			{
			goto tt_0_left_8;
			}
		else
			{
			goto tt_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_0_left_4;
			}
		else
			{
			goto tt_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_0_left_4;
		}
#endif
	goto tt_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tt_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_0_return;
#endif

tt_0_left_4:
	kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_0_return;

tt_0_return:
	return;



tt_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(k, n);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(k, n, &sB, smat_mem_align+sA_size);

	blasfeo_pack_dmat(k, n, B, ldb, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_1_left_4;
			}
		if(m-ii<=8)
			{
			goto tt_1_left_8;
			}
		else
			{
			goto tt_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_1_left_4;
			}
		else
			{
			goto tt_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_1_left_4;
		}
#endif
	goto tt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_1_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tt_1_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_1_return;
#endif

tt_1_left_4:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}
	goto tt_1_return;

tt_1_return:
	free(smat_mem);
	return;

	}

