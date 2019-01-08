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
#include "../include/blasfeo_d_blas.h"



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dsyrk dsyrk_
#endif



void blasfeo_dsyrk(char *uplo, char *ta, int *pm, int *pk, double *alpha, double *A, int *plda, double *beta, double *C, int *pldc)
	{

	int m = *pm;
	int k = *pk;
	int lda = *plda;
	int ldc = *pldc;

	if(m<=0)
		return;

	int ii, jj;

	int bs = 4;


	void *mem, *mem_align;
	double *pU;
	int sdu;
	struct blasfeo_dmat sA;
	int sA_size;
	int m1, k1;


	// stack memory allocation
// TODO visual studio alignment
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	double pU0[3*4*K_MAX_STACK] __attribute__ ((aligned (64)));
//	double pD0[4*16] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	double pU0[2*4*K_MAX_STACK] __attribute__ ((aligned (64)));
//	double pD0[2*16] __attribute__ ((aligned (64)));
#elif defined(TARGET_GENERIC)
	double pU0[1*4*K_MAX_STACK];
//	double pD0[1*16];
#else
	double pU0[1*4*K_MAX_STACK] __attribute__ ((aligned (64)));
//	double pD0[1*16] __attribute__ ((aligned (64)));
#endif
	int sdu0 = (k+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;


	// select algorithm
	if(*uplo=='l' | *uplo=='L')
		{
		if(*ta=='n' | *ta=='N')
			{
			goto ln;
			}
		else if(*ta=='t' | *ta=='T' | *ta=='c' | *ta=='C')
			{
			goto lt;
			}
		else
			{
			printf("\nBLASFEO: dsyrk: wrong value for ta\n");
			return;
			}
		}
	else if(*uplo=='u' | *uplo=='U')
		{
		if(*ta=='n' | *ta=='N')
			{
			goto un;
			}
		else if(*ta=='t' | *ta=='T' | *ta=='c' | *ta=='C')
			{
			goto ut;
			}
		else
			{
			printf("\nBLASFEO: dsyrk: wrong value for ta\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dsyrk: wrong value for uplo\n");
		return;
		}



/************************************************
* ln
************************************************/
ln:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>=100 | k>=100 | k>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | k>=64 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if(m>=32 | k>=32 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if(m>16 | k>16 | k>K_MAX_STACK)
#else
	if(m>=12 | k>=12 | k>K_MAX_STACK)
#endif
		{
		goto lx_1;
		}
	else
		{
		goto ln_0;
		}

ln_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, pU, sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_12x4_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		kernel_dsyrk_nt_l_8x8_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
//		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
//		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ln_0_left_4;
			}
		if(m-ii<=8)
			{
			goto ln_0_left_8;
			}
		else
			{
			goto ln_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, pU, sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+4*sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ln_0_left_4;
			}
		else
			{
			goto ln_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		}
	if(ii<m)
		{
		goto ln_0_left_4;
		}
#endif
	goto ln_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
ln_0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc,  m-(ii+8), m-(jj+8));
	goto ln_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ln_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+4*sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
#endif
	goto ln_0_return;
#endif

ln_0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
	goto ln_0_return;

ln_0_return:
	return;



/************************************************
* lt
************************************************/
lt:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>300 | k>300 | k>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | k>=64 | k>K_MAX_STACK)
#elif  defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if(m>=32 | k>=32 | k>K_MAX_STACK)
#elif  defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if(m>16 | k>16 | k>K_MAX_STACK)
#else
	if(m>=12 | k>=12 | k>K_MAX_STACK)
#endif
		{
		goto lx_1;
		}
	else
		{
		goto lt_0;
		}

lt_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_12x4_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		kernel_dsyrk_nt_l_8x8_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
//		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
//		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto lt_0_left_4;
			}
		if(m-ii<=8)
			{
			goto lt_0_left_8;
			}
		else
			{
			goto lt_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+4*sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ln_0_left_4;
			}
		else
			{
			goto ln_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		}
	if(ii<m)
		{
		goto lt_0_left_4;
		}
#endif
	goto lt_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
lt_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-(ii+8));
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc,  m-(ii+8), m-(jj+8));
	goto ln_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lt_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-(ii+4));
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+4*sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
#endif
	goto ln_0_return;
#endif

lt_0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
	goto lt_0_return;

lt_0_return:
	return;


/************************************************
* un
************************************************/
un:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>=108 | k>=108 | k>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | k>=64 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if(m>=32 | k>=32 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if(m>16 | k>16 | k>K_MAX_STACK)
#else
	if(m>=12 | k>=12 | k>K_MAX_STACK)
#endif
		{
		goto ux_1;
		}
	else
		{
		goto un_0;
		}

un_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, pU, sdu);
		kernel_dsyrk_nt_u_8x8_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
//		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
//		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
		kernel_dsyrk_nt_u_12x4_lib44c(k, alpha, pU, sdu, pU+8*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto un_0_left_4;
			}
		if(m-ii<=8)
			{
			goto un_0_left_8;
			}
		else
			{
			goto un_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, pU, sdu);
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto un_0_left_4;
			}
		else
			{
			goto un_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_4x4_vs_lib4cc(k, alpha, pU, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		goto un_0_left_4;
		}
#endif
	goto un_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
un_0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
//	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
//	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
	kernel_dsyrk_nt_u_12x4_vs_lib44c(k, alpha, pU, sdu, pU+8*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc, m-ii, m-(ii+8));
	goto un_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
un_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
#endif
	goto un_0_return;
#endif

un_0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	goto un_0_return;

un_0_return:
	return;



/************************************************
* ut
************************************************/
ut:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>300 | k>300 | k>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | k>=64 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if(m>=32 | k>=32 | k>K_MAX_STACK)
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if(m>16 | k>16 | k>K_MAX_STACK)
#else
	if(m>=12 | k>=12 | k>K_MAX_STACK)
#endif
		{
		goto ux_1;
		}
	else
		{
		goto ut_0;
		}

ut_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
//		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
//		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
		kernel_dsyrk_nt_u_8x8_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		kernel_dsyrk_nt_u_12x4_lib44c(k, alpha, pU, sdu, pU+8*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_12x4_vs_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ut_0_left_4;
			}
		if(m-ii<=8)
			{
			goto ut_0_left_8;
			}
		else
			{
			goto ut_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_8x4_vs_lib4cc(k, alpha, pU, sdu, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ut_0_left_4;
			}
		else
			{
			goto ut_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_4x4_vs_lib4cc(k, alpha, pU, A+jj*lda, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		goto ut_0_left_4;
		}
#endif
	goto ut_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
ut_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-(ii+8));
//	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
//	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	kernel_dsyrk_nt_u_12x4_vs_lib44c(k, alpha, pU, sdu, pU+8*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc, m-ii, m-(ii+8));
	goto ut_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ut_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-(ii+4));
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU, sdu, pU, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU, sdu, pU+4*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
#endif
	goto ut_0_return;
#endif

ut_0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU, pU, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	goto ut_0_return;

ut_0_return:
	return;



lx_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	sA_size = blasfeo_memsize_dmat(m1, k1);
	mem = malloc(sA_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(m, k, &sA, mem_align);

	if(*ta=='n' | *ta=='N')
		blasfeo_pack_dmat(m, k, A, lda, &sA, 0, 0);
	else
		blasfeo_pack_tran_dmat(k, m, A, lda, &sA, 0, 0);
	pU = sA.pA;
	sdu = sA.cn;
//	blasfeo_print_dmat(m, k, &sA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_12x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_l_8x8_lib44c(k, alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
#else
		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc);
#endif
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto lx_1_left_4;
			}
		if(m-ii<=8)
			{
			goto lx_1_left_8;
			}
		else
			{
			goto lx_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto lx_1_left_4;
			}
		else
			{
			goto lx_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
		}
	if(ii<m)
		{
		goto lx_1_left_4;
		}
#endif
	goto lx_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lx_1_left_12:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc,  m-(ii+8), m-(jj+8));
#endif
	goto lx_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lx_1_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
#endif
	goto lx_1_return;
#endif

lx_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
	goto lx_1_return;

lx_1_return:
	free(mem);
	return;


ux_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	sA_size = blasfeo_memsize_dmat(m1, k1);
	mem = malloc(sA_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(m, k, &sA, mem_align);

	if(*ta=='n' | *ta=='N')
		blasfeo_pack_dmat(m, k, A, lda, &sA, 0, 0);
	else
		blasfeo_pack_tran_dmat(k, m, A, lda, &sA, 0, 0);
	pU = sA.pA;
	sdu = sA.cn;
//	blasfeo_print_dmat(m, k, &sA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_u_8x8_lib44c(k, alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
#else
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
#endif
		kernel_dsyrk_nt_u_12x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_12x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ux_1_left_4;
			}
		if(m-ii<=8)
			{
			goto ux_1_left_8;
			}
		else
			{
			goto ux_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		kernel_dsyrk_nt_u_8x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_8x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto ux_1_left_4;
			}
		else
			{
			goto ux_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dsyrk_nt_u_4x4_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		goto ux_1_left_4;
		}
#endif
	goto ux_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ux_1_left_12:
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
#endif
	kernel_dsyrk_nt_u_12x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, beta, C+ii+(ii+8)*ldc, ldc, C+ii+(ii+8)*ldc, ldc, m-ii, m-(ii+8));
	goto ux_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ux_1_left_8:
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44c(k, alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, beta, C+ii+(ii+4)*ldc, ldc, C+ii+(ii+4)*ldc, ldc, m-ii, m-(ii+4));
#endif
	goto ux_1_return;
#endif

ux_1_left_4:
	kernel_dsyrk_nt_u_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+ii*sdu, beta, C+ii+ii*ldc, ldc, C+ii+ii*ldc, ldc, m-ii, m-ii);
	goto ux_1_return;

ux_1_return:
	free(mem);
	return;

	}


