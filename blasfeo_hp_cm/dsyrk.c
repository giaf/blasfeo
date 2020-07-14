/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include <blasfeo_target.h>
#include <blasfeo_block_size.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>



// TODO move to a header file to reuse across routines
#define EL_SIZE 8 // double precision

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
#define M_KERNEL 12 // max kernel: 12x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes

#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
#define M_KERNEL 8 // max kernel: 8x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes

#else // assume generic target
#define M_KERNEL 4 // max kernel: 4x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes // TODO 32-bytes for cortex A9
#endif



void blasfeo_hp_dsyrk3_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dsyrk3_ln (cm) %d %d %f %p %d %d %f %p %d %d %p %d %d\n", m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d\n", A, lda, C, ldc, D, ldd);

	int ii, jj;

	const int ps = 4; //D_PS;

#if defined(TARGET_GENERIC)
	double pU0[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( double pU0[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu0 = (k+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	double *pU;
	int sdu;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
//	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;



//	goto ln_0;
//	goto lx_1;
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>=200 | k>=200 | k>K_MAX_STACK)
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

	// never to get here
	return;



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
			kernel_dgemm_nt_12x4_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_12x4_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		kernel_dsyrk_nt_l_8x8_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
//		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
//		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+8*sdu, pU+8*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd);
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
			kernel_dgemm_nt_8x4_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+4*sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
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
			kernel_dgemm_nt_4x4_lib4ccc(k, &alpha, pU, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
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
		kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
//	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+8*sdu, pU+8*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd,  m-(ii+8), m-(jj+8));
	goto ln_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ln_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+4*sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#endif
	goto ln_0_return;
#endif

ln_0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
	goto ln_0_return;

ln_0_return:
	return;



lx_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m1, k1);
	mem = malloc(tA_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m, k, &tA, (void *) mem_align);

	pU = tA.pA;
	sdu = tA.cn;

//	if(ta=='n' | ta=='N')
//		blasfeo_pack_dmat(m, k, A, lda, &tA, 0, 0);
//	else
//		blasfeo_pack_tran_dmat(k, m, A, lda, &tA, 0, 0);
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_dpack_tt_4_lib4(m, A+ii*lda, lda, pU+ii*ps, sdu);
		}
	if(ii<k)
		{
		kernel_dpack_tt_4_vs_lib4(m, A+ii*lda, lda, pU+ii*ps, sdu, k-ii);
		}

//	blasfeo_print_dmat(m, k, &tA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_l_8x8_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
#else
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd);
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
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
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
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
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
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd,  m-(ii+8), m-(jj+8));
#endif
	goto lx_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lx_1_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#endif
	goto lx_1_return;
#endif

lx_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
	goto lx_1_return;

lx_1_return:
	free(mem);
	return;



	// never to get here
	return;

	}



void blasfeo_hp_dsyrk3_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dsyrk3_lt (cm) %d %d %f %p %d %d %f %p %d %d %p %d %d\n", m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d\n", A, lda, C, ldc, D, ldd);

	int ii, jj;

	const int ps = 4; //D_PS;

#if defined(TARGET_GENERIC)
	double pU0[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( double pU0[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu0 = (k+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	double *pU;
	int sdu;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
//	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;



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

	// never to get here
	return;



lt_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_12x4_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_l_8x8_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
#else
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+8*sdu, pU+8*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd);
#endif
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+4*sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto lt_0_left_4;
			}
		else
			{
			goto lt_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4ccc(k, &alpha, pU, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		}
	if(ii<m)
		{
		goto lt_0_left_4;
		}
#endif
	goto lt_0_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lt_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-(ii+8));
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+4*sdu, sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+8*sdu, pU+8*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd,  m-(ii+8), m-(jj+8));
#endif
	goto lt_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lt_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-(ii+4));
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+4*sdu, pU+4*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#endif
	goto lt_0_return;
#endif

lt_0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
	goto lt_0_return;

lt_0_return:
	return;



lx_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m1, k1);
	mem = malloc(tA_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m, k, &tA, (void *) mem_align);

	pU = tA.pA;
	sdu = tA.cn;

//	if(ta=='n' | ta=='N')
//		blasfeo_pack_dmat(m, k, A, lda, &tA, 0, 0);
//	else
//		blasfeo_pack_tran_dmat(k, m, A, lda, &tA, 0, 0);

	for(ii=0; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU+ii*sdu);
		}
	if(ii<m)
		{
		kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU+ii*sdu, m-ii);
		}


//	blasfeo_print_dmat(m, k, &tA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_l_8x8_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
#else
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd);
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
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd);
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
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		kernel_dsyrk_nt_l_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
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
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+(ii+8)*sdu, pU+(jj+8)*sdu, &beta, C+(ii+8)+(jj+8)*ldc, ldc, D+(ii+8)+(jj+8)*ldd, ldd,  m-(ii+8), m-(jj+8));
#endif
	goto lx_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
lx_1_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_l_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
#else
	kernel_dsyrk_nt_l_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+(ii+4)*sdu, pU+(jj+4)*sdu, &beta, C+(ii+4)+(jj+4)*ldc, ldc, D+(ii+4)+(jj+4)*ldd, ldd,  m-(ii+4), m-(jj+4));
#endif
	goto lx_1_return;
#endif

lx_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
	goto lx_1_return;

lx_1_return:
	free(mem);
	return;



	// never to get here
	return;

	}



void blasfeo_hp_dsyrk3_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dsyrk3_un (cm) %d %d %f %p %d %d %f %p %d %d %p %d %d\n", m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d\n", A, lda, C, ldc, D, ldd);

	int ii, jj;

	const int ps = 4; //D_PS;

#if defined(TARGET_GENERIC)
	double pU0[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( double pU0[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu0 = (k+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	double *pU;
	int sdu;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
//	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;



#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>=200 | k>=200 | k>K_MAX_STACK)
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

	// never to get here
	return;



un_0:
	pU = pU0;
	sdu = sdu0;
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, pU, sdu);
		kernel_dsyrk_nt_u_8x8_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
//		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
//		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
		kernel_dsyrk_nt_u_12x4_lib44cc(k, &alpha, pU, sdu, pU+8*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4ccc(k, &alpha, pU, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, A+jj, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
//	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
//	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
	kernel_dsyrk_nt_u_12x4_vs_lib44cc(k, &alpha, pU, sdu, pU+8*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd, m-ii, m-(ii+8));
	goto un_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
un_0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	goto un_0_return;
#endif

un_0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	goto un_0_return;

un_0_return:
	return;



ux_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m1, k1);
	mem = malloc(tA_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m, k, &tA, (void *) mem_align);

	pU = tA.pA;
	sdu = tA.cn;

//	if(ta=='n' | ta=='N')
//		blasfeo_pack_dmat(m, k, A, lda, &tA, 0, 0);
//	else
//		blasfeo_pack_tran_dmat(k, m, A, lda, &tA, 0, 0);
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_dpack_tt_4_lib4(m, A+ii*lda, lda, pU+ii*ps, sdu);
		}
	if(ii<k)
		{
		kernel_dpack_tt_4_vs_lib4(m, A+ii*lda, lda, pU+ii*ps, sdu, k-ii);
		}

//	blasfeo_print_dmat(m, k, &tA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_u_8x8_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
#else
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
#endif
		kernel_dsyrk_nt_u_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	kernel_dsyrk_nt_u_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd, m-ii, m-(ii+8));
	goto ux_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ux_1_left_8:
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	goto ux_1_return;
#endif

ux_1_left_4:
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	goto ux_1_return;

ux_1_return:
	free(mem);
	return;

	// never to get here
	return;

	}



void blasfeo_hp_dsyrk3_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dsyrk3_ut (cm) %d %d %f %p %d %d %f %p %d %d %p %d %d\n", m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d\n", A, lda, C, ldc, D, ldd);

	int ii, jj;

	const int ps = 4; //D_PS;

#if defined(TARGET_GENERIC)
	double pU0[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( double pU0[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu0 = (k+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	double *pU;
	int sdu;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
//	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;



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
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_u_8x8_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
#else
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
#endif
		kernel_dsyrk_nt_u_12x4_lib44cc(k, &alpha, pU, sdu, pU+8*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4ccc(k, &alpha, pU, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, A+jj*lda, lda, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
			}
		}
	if(ii<m)
		{
		goto ut_0_left_4;
		}
#endif
	goto ut_0_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ut_0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-(ii+8));
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	kernel_dsyrk_nt_u_12x4_vs_lib44cc(k, &alpha, pU, sdu, pU+8*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd, m-ii, m-(ii+8));
	goto ut_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ut_0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU+0*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-(ii+4));
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU, sdu, pU, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU, sdu, pU+4*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	goto ut_0_return;
#endif

ut_0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU, pU, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	goto ut_0_return;

ut_0_return:
	return;



ux_1:
	k1 = (k+128-1)/128*128;
	m1 = (m+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m1, k1);
	mem = malloc(tA_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m, k, &tA, (void *) mem_align);

	pU = tA.pA;
	sdu = tA.cn;

//	if(ta=='n' | ta=='N')
//		blasfeo_pack_dmat(m, k, A, lda, &tA, 0, 0);
//	else
//		blasfeo_pack_tran_dmat(k, m, A, lda, &tA, 0, 0);

	for(ii=0; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU+ii*sdu);
		}
	if(ii<m)
		{
		kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU+ii*sdu, m-ii);
		}

//	blasfeo_print_dmat(m, k, &tA, 0, 0);

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dsyrk_nt_u_8x8_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
#else
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
#endif
		kernel_dsyrk_nt_u_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd);
		for(jj=ii+12; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		kernel_dsyrk_nt_u_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd);
		for(jj=ii+8; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
		kernel_dsyrk_nt_u_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd);
		for(jj=ii+4; jj<m-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<m)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+jj*sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, m-jj);
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
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	kernel_dsyrk_nt_u_12x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+8)*sdu, &beta, C+ii+(ii+8)*ldc, ldc, D+ii+(ii+8)*ldd, ldd, m-ii, m-(ii+8));
	goto ux_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
ux_1_left_8:
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dsyrk_nt_u_8x8_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+ii*sdu, sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
#else
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	kernel_dsyrk_nt_u_8x4_vs_lib44cc(k, &alpha, pU+ii*sdu, sdu, pU+(ii+4)*sdu, &beta, C+ii+(ii+4)*ldc, ldc, D+ii+(ii+4)*ldd, ldd, m-ii, m-(ii+4));
#endif
	goto ux_1_return;
#endif

ux_1_left_4:
	kernel_dsyrk_nt_u_4x4_vs_lib44cc(k, &alpha, pU+ii*sdu, pU+ii*sdu, &beta, C+ii+ii*ldc, ldc, D+ii+ii*ldd, ldd, m-ii, m-ii);
	goto ux_1_return;

ux_1_return:
	free(mem);
	return;

	// never to get here
	return;

	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_dsyrk3_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk3_ln(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk3_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk3_lt(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk3_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk3_un(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk3_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk3_ut(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj);
	}



#endif

