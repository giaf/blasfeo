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



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dtrmm dtrmm_
#endif



void blasfeo_dtrmm(char *side, char *uplo, char *transa, char *diag, int *pm, int *pn, double *alpha, double *A, int *plda, double *B, int *pldb)
	{

	int m = *pm;
	int n = *pn;
	int lda = *plda;
	int ldb = *pldb;

	double d_0 = 0.0;

	int ii, jj;

	int ps = 4;

	if(m<=0 | n<=0)
		return;

// TODO visual studio alignment
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	double pU0[3*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	double pU0[2*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#elif defined(TARGET_GENERIC)
	double pU0[1*4*K_MAX_STACK];
#else
	double pU0[1*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#endif

	int k0;
	// TODO update if necessary !!!!!
	if(*side=='l' | *side=='L')
		k0 = m;
	else
		k0 = n;

	int sdu0 = (k0+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	struct blasfeo_dmat sA, sB;
	double *pU, *pB, *dA, *dB;
	int sda, sdb, sdu;
	int sA_size, sB_size;
	void *mem, *mem_align;

	if(*side=='l' | *side=='L')
		{
		printf("\nBLASFEO: dtrmm: not implemented yet\n");
		return;
		}
	else if(*side=='r' | *side=='R')
		{
		if(*uplo=='l' | *uplo=='L')
			{
			if(*transa=='n' | *transa=='N')
				{
				if(*diag=='n' | *diag=='N')
					{
#if defined(TARGET_X64_INTEL_HASWELL)
					if(m>300 | n>300 | m>K_MAX_STACK) // XXX cond on m !!!!!
#else
					if(m>=12 | n>=12 | m>K_MAX_STACK) // XXX cond on m !!!!!
#endif
						{
						goto rlnn_1;
						}
					else
						{
						goto rlnn_0;
						}
					}
				else if(*diag=='u' | *diag=='U')
					{
					printf("\nBLASFEO: dtrmm: not implemented yet\n");
					return;
					}
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C')
				{
				printf("\nBLASFEO: dtrmm: not implemented yet\n");
				return;
				}
			else
				{
				printf("\nBLASFEO: dtrmm: wrong value for transa\n");
				return;
				}
			}
		else if(*uplo=='u' | *uplo=='U')
			{
			printf("\nBLASFEO: dtrmm: not implemented yet\n");
			return;
			}
		else
			{
			printf("\nBLASFEO: dtrmm: wrong value for uplo\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dtrmm: wrong value for side\n");
		return;
		}



rlnn_0:
	pU = pU0;
	sdu = sdu0;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(n, B+ii, ldb, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nn_rl_12x4_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nn_rl_12x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rlnn_0_left_4;
			}
		if(m-ii<=8)
			{
			goto rlnn_0_left_8;
			}
		else
			{
			goto rlnn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(n, B+ii, ldb, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nn_rl_8x4_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nn_rl_8x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rlnn_0_left_4;
			}
		else
			{
			goto rlnn_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(n, B+ii, ldb, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nn_rl_4x4_lib4cc(n-jj, alpha, pU+jj*ps, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nn_rl_4x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto rlnn_0_left_4;
		}
#endif
	goto rlnn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rlnn_0_left_12:
	kernel_dpack_nn_12_lib4(n, B+ii, ldb, pU, sdu);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nn_rl_12x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rlnn_0_left_8:
	kernel_dpack_nn_8_vs_lib4(n, B+ii, ldb, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nn_rl_8x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, sdu, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_0_return;
#endif

rlnn_0_left_4:
	kernel_dpack_nn_4_vs_lib4(n, B+ii, ldb, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nn_rl_4x4_vs_lib4cc(n-jj, alpha, pU+jj*ps, A+jj+jj*lda, lda, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_0_return;

rlnn_0_return:
	return;



rlnn_1:
	sA_size = blasfeo_memsize_dmat(12, n);
	sB_size = blasfeo_memsize_dmat(n, n);
	mem = malloc(sA_size+sB_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(12, n, &sA, mem_align);
	blasfeo_create_dmat(n, n, &sB, mem_align+sA_size);

	pU = sA.pA;
	sdu = sA.cn;
	pB = sB.pA;
	sdb = sB.cn;

	for(ii=0; ii<n-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(n-ii, A+ii+ii*lda, lda, pB+ii*ps+ii*sdb);
		}
	if(ii<n)
		{
		kernel_dpack_tn_4_vs_lib4(n-ii, A+ii+ii*lda, lda, pB+ii*ps+ii*sdb, n-ii);
		}

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(n, B+ii, ldb, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nt_ru_12x4_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nt_ru_12x4_vs_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rlnn_1_left_4;
			}
		if(m-ii<=8)
			{
			goto rlnn_1_left_8;
			}
		else
			{
			goto rlnn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(n, B+ii, ldb, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nt_ru_8x4_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nt_ru_8x4_vs_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rlnn_1_left_4;
			}
		else
			{
			goto rlnn_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(n, B+ii, ldb, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrmm_nt_ru_4x4_lib44c(n-jj, alpha, pU+jj*ps, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb);
			}
		if(jj<n)
			{
			kernel_dtrmm_nt_ru_4x4_vs_lib44c(n-jj, alpha, pU+jj*ps, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto rlnn_1_left_4;
		}
#endif
goto rlnn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rlnn_1_left_12:
	kernel_dpack_nn_12_vs_lib4(n, B+ii, ldb, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nt_ru_12x4_vs_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rlnn_1_left_8:
	kernel_dpack_nn_8_vs_lib4(n, B+ii, ldb, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nt_ru_8x4_vs_lib44c(n-jj, alpha, pU+jj*ps, sdu, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_1_return;
#endif

rlnn_1_left_4:
	kernel_dpack_nn_4_vs_lib4(n, B+ii, ldb, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrmm_nt_ru_4x4_vs_lib44c(n-jj, alpha, pU+jj*ps, pB+jj*ps+jj*sdb, &d_0, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, m-ii, n-jj);
		}
goto rlnn_1_return;

rlnn_1_return:
	free(mem);
	return;

	}

