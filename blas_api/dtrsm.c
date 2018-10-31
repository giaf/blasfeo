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
#define blasfeo_dtrsm dtrsm_
#endif



void blasfeo_dtrsm(char *side, char *uplo, char *transa, char *diag, int *pm, int *pn, double *alpha, double *A, int *plda, double *B, int *pldb)
	{

	int m = *pm;
	int n = *pn;
	int lda = *plda;
	int ldb = *pldb;

	double d_m10 = -1.0;
	double *d_m1 = &d_m10;

	int ii, jj;

	int ps = 4;

	if(m<=0 | n<=0)
		return;

// TODO visual studio alignment
#if defined(TARGET_GENERIC)
	double pd0[K_MAX_STACK];
#else
	double pd0[K_MAX_STACK] __attribute__ ((aligned (64)));
#endif

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
		printf("\nBLASFEO: dtrsm: not implemented yet\n");
		return;
		}
	else if(*side=='r' | *side=='R')
		{
		if(*uplo=='l' | *uplo=='L')
			{
			if(*transa=='n' | *transa=='N')
				{
				printf("\nBLASFEO: dtrsm: not implemented yet\n");
				return;
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C')
				{
				if(*diag=='n' | *diag=='N')
					{
#if defined(TARGET_X64_INTEL_HASWELL)
					if(m>=120 | n>=120 | m>K_MAX_STACK) // XXX cond on m !!!!!
#else
					if(m>=12 | n>=12 | m>K_MAX_STACK) // XXX cond on m !!!!!
#endif
						{
						goto rltn_1;
						}
					else
						{
						goto rltn_0;
						}
					}
				else if(*diag=='u' | *diag=='U')
					{
					printf("\nBLASFEO: dtrsm: not implemented yet\n");
					return;
					}
				}
			else
				{
				printf("\nBLASFEO: dtrsm: wrong value for transa\n");
				return;
				}
			}
		else if(*uplo=='u' | *uplo=='U')
			{
			printf("\nBLASFEO: dtrsm: not implemented yet\n");
			return;
			}
		else
			{
			printf("\nBLASFEO: dtrsm: wrong value for uplo\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dtrsm: wrong value for side\n");
		return;
		}



rltn_0:
	pU = pU0;
	sdu = sdu0;
	dA = pd0;

	for(ii=0; ii<n; ii++)
		dA[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_12_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_0_left_4;
			}
		if(m-ii<=8)
			{
			goto rltn_0_left_8;
			}
		else
			{
			goto rltn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_8_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_0_left_4;
			}
		else
			{
			goto rltn_0_left_8;
			}
		}
#else
	for(ii=0; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_4_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		goto rltn_0_left_4;
		}
#endif
	goto rltn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rltn_0_left_12:
		for(jj=0; jj<n; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
			kernel_dpack_nn_12_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
			}
	goto rltn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rltn_0_left_8:
		for(jj=0; jj<n; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
			kernel_dpack_nn_8_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
			}
	goto rltn_0_return;
#endif

rltn_0_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
		kernel_dpack_nn_4_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, m-ii);
		}
goto rltn_0_return;

rltn_0_return:
	return;



rltn_1:
	sA_size = blasfeo_memsize_dmat(12, n);
	sB_size = blasfeo_memsize_dmat(n, n);
	mem = malloc(sA_size+sB_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(12, n, &sA, mem_align);
	blasfeo_create_dmat(n, n, &sB, mem_align+sA_size);

	// TODO pack triangle
//	blasfeo_pack_dmat(n, n, A, lda, &sB, 0, 0);
	blasfeo_pack_l_dmat(n, n, A, lda, &sB, 0, 0);
//	blasfeo_print_dmat(n, n, &sB, 0, 0);

	pU = sA.pA;
	sdu = sA.cn;
	pB = sB.pA;
	sdb = sB.cn;
	dB = sB.dA;

	for(ii=0; ii<n; ii++)
		dB[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_12_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_1_left_4;
			}
		if(m-ii<=8)
			{
			goto rltn_1_left_8;
			}
		else
			{
			goto rltn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_8_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_1_left_4;
			}
		else
			{
			goto rltn_1_left_8;
			}
		}
#else
	for(ii=0; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_4_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		goto rltn_1_left_4;
		}
#endif
	goto rltn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rltn_1_left_12:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_12_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
		}
goto rltn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rltn_1_left_8:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_8_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
		}
goto rltn_1_return;
#endif

rltn_1_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_4_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, m-ii);
		}
goto rltn_1_return;

rltn_1_return:
	free(mem);
	return;

	}
