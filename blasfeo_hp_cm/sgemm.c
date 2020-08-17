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
#include <blasfeo_s_aux.h>
#include <blasfeo_s_kernel.h>



#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )
#define blasfeo_dmat blasfeo_cm_dmat
#define blasfeo_hp_sgemm_nn blasfeo_hp_cm_sgemm_nn
#define blasfeo_hp_sgemm_nt blasfeo_hp_cm_sgemm_nt
#define blasfeo_hp_sgemm_tn blasfeo_hp_cm_sgemm_tn
#define blasfeo_hp_sgemm_tt blasfeo_hp_cm_sgemm_tt
#define blasfeo_sgemm_nn blasfeo_cm_sgemm_nn
#define blasfeo_sgemm_nt blasfeo_cm_sgemm_nt
#define blasfeo_sgemm_tn blasfeo_cm_sgemm_tn
#define blasfeo_sgemm_tt blasfeo_cm_sgemm_tt
#endif



// TODO move to a header file to reuse across routines
#define EL_SIZE 4 // single precision

#if defined(TARGET_X64_INTEL_HASWELL)
#define M_KERNEL 24 // max kernel: 24x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes

#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#define M_KERNEL 16 // max kernel: 16x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes

#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7)
#define M_KERNEL 8 // max kernel: 8x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes

#else // assume generic target
#define M_KERNEL 4 // max kernel: 4x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes // TODO 32-bytes for cortex A9
#endif



void blasfeo_hp_sgemm_nn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_sgemm_nn (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	float *A = sA->pA + ai + aj*lda;
	float *B = sB->pA + bi + bj*ldb;
	float *C = sC->pA + ci + cj*ldc;
	float *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj;

// no global bs, to be able to mix them in different algorithms !!!
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	const int bs = 8;
//#else
//	const int bs = 4;
//#endif
	const int ps = S_PS;
//	const int ps_4 = 4;
//	const int ps_8 = 8;

#if defined(TARGET_GENERIC)
	float pU[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( float pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_smat tA, tB;
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
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

//	goto nn_2; // no pack
//	goto nn_m0; // pack A
//	goto nn_n0; // pack B
//	goto nn_1; // pack A and B
	if( k<=K_MAX_STACK )
		{
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		goto nn_1; // XXX all matrices for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!
//		if( m<=48 & n<=48 )
//		if( (m<=12 & n<=12) | (m_min*k_cache + n_cache*k_cache <= l1_cache_el) )
		if( (m<=m_kernel & n<=m_kernel) | (m_kernel_cache*k_cache + n_cache*k_cache <= l1_cache_el) | (m<m_kernel & (m_cache*k_cache + m_kernel_cache*k_cache <= l1_cache_el) ) )
			{
			goto nn_2; // small matrix: no pack
//			goto nn_m0; // small matrix: pack A
			}
#else
		if( m<=8 & n<=8 )
			{
			goto nn_2; // small matrix: no pack
			}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
		if( m<=2*m_kernel | n<=2*m_kernel | k<448 )
#else
		if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
			{
			if( m<=n*4 )
				{
				goto nn_m0; // long matrix: pack A
				}
			else
				{
				goto nn_n0; // tall matrix: pack B
				}
			}
		}
	goto nn_1; // big matrix: pack A and B

	// never to get here
	return;


nn_m0:

	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		kernel_spack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nn_4x4_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_m0_left_4;
		}
#endif
	goto nn_m0_return;

nn_m0_left_4:
	kernel_spack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m0_return;

nn_m0_return:
	return;



nn_n0:

	jj = 0;
#if 1
	for(; jj<n-3; jj+=4)
		{
		kernel_spack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_sgemm_nt_4x4_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_sgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto nn_n0_left_4;
		}
#endif
	goto nn_n0_return;

nn_n0_left_4:
	kernel_spack_tn_4_vs_lib4(k, B+(jj+0)*ldb, ldb, pU+0*sdu, n-jj-0);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_sgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_n0_return;

nn_n0_return:
	return;



nn_1:

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_smat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_smat(ps, n1, k1);
	mem = malloc(tA_size+tB_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_smat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_smat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

	pack_B = 1;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-23; ii+=24)
		{
		kernel_spack_nn_24_lib8(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto nn_1_left_8;
			}
		else if(m-ii<=16)
			{
			goto nn_1_left_16;
			}
		else
			{
			goto nn_1_left_24;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-15; ii+=16)
		{
		kernel_spack_nn_16_lib8(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto nn_1_left_8;
			}
		else
			{
			goto nn_1_left_16;
			}
		}
#elif 0//defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_nn_8_lib8(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_8x8_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//				kernel_sgemm_nt_8x8_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//				kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				t
			else
				{
				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		goto nn_1_left_8;
		}
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_nn_8_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_spack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
//			kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
			}
		pack_B = 0;
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
		kernel_spack_nn_4_lib4(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_spack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		goto nn_1_left_4;
		}
#endif
	goto nn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nn_1_left_24:
	kernel_spack_nn_24_vs_lib8(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nn_1_left_16:
	kernel_spack_nn_16_vs_lib8(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nn_1_left_8:
	kernel_spack_nn_8_vs_lib8(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nn_1_return;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_1_left_8:
	kernel_spack_nn_8_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
		}
	goto nn_1_return;
#endif

nn_1_left_4:
	kernel_spack_nn_4_vs_lib4(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_1_return;

nn_1_return:
	free(mem);
	return;



nn_2:

	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nn_4x4_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_2_left_4;
		}
#endif
	goto nn_2_return;

nn_2_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_2_return;

nn_2_return:
	return;



	// never to get here
	return;

	}



void blasfeo_hp_sgemm_nt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_sgemm_nt (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	float *A = sA->pA + ai + aj*lda;
	float *B = sB->pA + bi + bj*ldb;
	float *C = sC->pA + ci + cj*ldc;
	float *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj;

// no global bs, to be able to mix them in different algorithms !!!
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	const int bs = 8;
//#else
//	const int bs = 4;
//#endif
	const int ps = S_PS;
//	const int ps_4 = 4;
//	const int ps_8 = 8;

#if defined(TARGET_GENERIC)
	float pU[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( float pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_smat tA, tB;
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
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

//	goto nt_2; // no pack
//	goto nt_m0; // pack A
//	goto nt_n0; // pack B
//	goto nt_1; // pack A and B
	if( k<=K_MAX_STACK )
		{
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		goto nt_1; // XXX all matrices for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!
//		if( m<=48 & n<=48 )
//		if( (m<=12 & n<=12) | (m_min*k_cache + n_cache*k_cache <= l1_cache_el) )
		if( (m<=m_kernel & n<=m_kernel) | (m_kernel_cache*k_cache + n_cache*k_cache <= l1_cache_el) | (m<m_kernel & (m_cache*k_cache + m_kernel_cache*k_cache <= l1_cache_el) ) )
			{
			goto nt_2; // small matrix: no pack
//			goto nt_m0; // small matrix: pack A
			}
#else
		if( m<=8 & n<=8 )
			{
			goto nt_2; // small matrix: no pack
			}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
		if( m<=2*m_kernel | n<=2*m_kernel | k<200 )
#else
		if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
			{
			if( m<=n )
				{
				goto nt_m0; // long matrix: pack A
				}
			else
				{
				goto nt_n0; // tall matrix: pack B
				}
			}
		}
	goto nt_1; // big matrix: pack A and B

	// never to get here
	return;


nt_m0:

	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		kernel_spack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_4x4_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_m0_left_4;
		}
#endif
	goto nt_m0_return;

nt_m0_left_4:
	kernel_spack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_m0_return;

nt_m0_return:
	return;



nt_n0:

	jj = 0;
#if 1
for(; jj<n-3; jj+=4)
		{
		kernel_spack_nn_4_lib4(k, B+jj, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_sgemm_nt_4x4_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_sgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto nt_n0_left_4;
		}
#endif
	goto nt_n0_return;

nt_n0_left_4:
	kernel_spack_nn_4_vs_lib4(k, B+jj, ldb, pU, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_sgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_n0_return;

nt_n0_return:
	return;



nt_1:

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_smat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_smat(ps, n1, k1);
	mem = malloc(tA_size+tB_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_smat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_smat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<k-7; ii+=8)
		{
		kernel_spack_tt_8_lib8(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb);
		}
	if(ii<k)
		{
		kernel_spack_tt_8_vs_lib8(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb, k-ii);
		}
#else
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_spack_tt_4_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb);
		}
	if(ii<k)
		{
		kernel_spack_tt_4_vs_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb, k-ii);
		}
#endif

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-23; ii+=24)
		{
		kernel_spack_nn_24_lib8(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto nt_1_left_8;
			}
		else if(m-ii<=16)
			{
			goto nt_1_left_16;
			}
		else
			{
			goto nt_1_left_24;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-15; ii+=16)
		{
		kernel_spack_nn_16_lib8(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto nt_1_left_8;
			}
		else
			{
			goto nt_1_left_16;
			}
		}
#elif 0//defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_nn_8_lib8(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_8x8_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//				kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		goto nt_1_left_8;
		}
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_nn_8_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
//			kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
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
		kernel_spack_nn_4_lib4(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_1_left_4;
		}
#endif
	goto nt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_1_left_24:
	kernel_spack_nn_24_vs_lib8(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nt_1_left_16:
	kernel_spack_nn_16_vs_lib8(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nt_1_left_8:
	kernel_spack_nn_8_vs_lib8(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto nt_1_return;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nt_1_left_8:
	kernel_spack_nn_8_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
//		kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
		}
	goto nt_1_return;
#endif

nt_1_left_4:
	kernel_spack_nn_4_vs_lib4(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_1_return;

nt_1_return:
	free(mem);
	return;



nt_2:
	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_4x4_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_2_left_4;
		}
#endif
	goto nt_2_return;

nt_2_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_2_return;

nt_2_return:
	return;



	// never to get here
	return;

	}



void blasfeo_hp_sgemm_tn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_sgemm_tn (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	float *A = sA->pA + ai + aj*lda;
	float *B = sB->pA + bi + bj*ldb;
	float *C = sC->pA + ci + cj*ldc;
	float *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj;

// no global bs, to be able to mix them in different algorithms !!!
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	const int bs = 8;
//#else
//	const int bs = 4;
//#endif
	const int ps = S_PS;
//	const int ps_4 = 4;
//	const int ps_8 = 8;

#if defined(TARGET_GENERIC)
	float pU[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( float pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_smat tA, tB;
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
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

//	goto tn_2; // no pack
//	goto tn_m0; // pack A
//	goto tn_n0; // pack B
//	goto tn_1; // pack A and B
	if( k<=K_MAX_STACK )
		{
		// no algorithm for small matrix
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//		if( m<=48 & n<=48 )
//		if( (m<=12 & n<=12) | (m_min*k_cache + n_cache*k_cache <= l1_cache_el) )
//		if( (m<=m_kernel & n<=m_kernel) | (m_kernel_cache*k_cache + n_cache*k_cache <= l1_cache_el) | (m<m_kernel & (m_cache*k_cache + m_kernel_cache*k_cache <= l1_cache_el) ) )
//			{
//			goto tn_2; // small matrix: no pack
//			goto tn_m0; // small matrix: pack A
//			}
#else
//		if( m<=8 & n<=8 )
//			{
//			goto tn_2; // small matrix: no pack
//			}
#endif
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		goto tn_1; // XXX all matrices for now !!!!!
		if( m<=2*m_kernel | n<=2*m_kernel | k<448 )
#else
		if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
			{
			if( m<=n )
				{
				goto tn_m0; // long matrix: pack A
				}
			else
				{
				goto tn_n0; // tall matrix: pack B
				}
			}
		}
	goto tn_1; // big matrix: pack A and B

	// never to get here
	return;


tn_m0:

	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		kernel_spack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nn_4x4_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tn_m0_left_4;
		}
#endif
	goto tn_m0_return;

tn_m0_left_4:
	kernel_spack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_m0_return;

tn_m0_return:
	return;



tn_n0:

	jj = 0;
#if 1
	for(; jj<n-3; jj+=4)
		{
		kernel_spack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_sgemm_tt_4x4_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_sgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tn_n0_left_4;
		}
#endif
	goto tn_n0_return;

tn_n0_left_4:
	kernel_spack_tn_4_vs_lib4(k, B+(jj+0)*ldb, ldb, pU+0*sdu, n-jj-0);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_sgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_n0_return;

tn_n0_return:
	return;



tn_1:

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_smat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_smat(ps, n1, k1);
	mem = malloc(tA_size+tB_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_smat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_smat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

	pack_B = 1;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-23; ii+=24)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
		kernel_spack_tn_8_lib8(k, A+(ii+16)*lda, lda, tA.pA+2*sda*ps);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto tn_1_left_8;
			}
		else if(m-ii<=16)
			{
			goto tn_1_left_16;
			}
		else
			{
			goto tn_1_left_24;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-15; ii+=16)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto tn_1_left_8;
			}
		else
			{
			goto tn_1_left_16;
			}
		}
#elif 0//defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		for(jj=0; jj<n-7; jj+=8)
			{
			if(pack_B)
				kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_8x8_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			if(n-jj>4)
				{
				kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//				kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		goto tn_1_left_8;
		}
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_spack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_spack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
//			kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
			}
		pack_B = 0;
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
		kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_spack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_sgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
		}
		if(ii<m)
		{
		goto tn_1_left_4;
		}
#endif
	goto tn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tn_1_left_24:
	kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
	kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
	kernel_spack_tn_8_vs_lib8(k, A+(ii+16)*lda, lda, tA.pA+2*sda*ps, m-ii-16);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tn_1_left_16:
	kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
	kernel_spack_tn_8_vs_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps, m-ii-8);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tn_1_left_8:
	kernel_spack_tn_8_vs_lib8(k, A+ii*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		if(pack_B)
			kernel_spack_tn_8_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
		kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		if(pack_B)
			kernel_spack_tn_8_vs_lib8(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tn_1_return;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_1_left_8:
	kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_spack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
		}
	goto tn_1_return;
#endif

tn_1_left_4:
	kernel_spack_tn_4_vs_lib4(k, A+(ii+0)*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_spack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_1_return;

tn_1_return:
free(mem);
	return;



	// never to get here
	return;

	}



void blasfeo_hp_sgemm_tt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_sgemm_tt (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	float *A = sA->pA + ai + aj*lda;
	float *B = sB->pA + bi + bj*ldb;
	float *C = sC->pA + ci + cj*ldc;
	float *D = sD->pA + di + dj*ldd;

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj;

// no global bs, to be able to mix them in different algorithms !!!
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	const int bs = 8;
//#else
//	const int bs = 4;
//#endif
	const int ps = S_PS;
//	const int ps_4 = 4;
//	const int ps_8 = 8;

#if defined(TARGET_GENERIC)
	float pU[M_KERNEL*K_MAX_STACK];
#else
	ALIGNED( float pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_smat tA, tB;
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
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

//	goto tt_2; // no pack
//	goto tt_m0; // pack A
//	goto tt_n0; // pack B
//	goto tt_1; // pack A and B
	if( k<=K_MAX_STACK )
		{
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		goto tt_1; // XXX all matrices for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!
//		if( m<=48 & n<=48 )
//		if( (m<=12 & n<=12) | (m_min*k_cache + n_cache*k_cache <= l1_cache_el) )
		if( (m<=m_kernel & n<=m_kernel) | (m_kernel_cache*k_cache + n_cache*k_cache <= l1_cache_el) | (m<m_kernel & (m_cache*k_cache + m_kernel_cache*k_cache <= l1_cache_el) ) )
			{
			goto tt_2; // small matrix: no pack
//			goto tt_m0; // small matrix: pack A
			}
#else
		if( m<=8 & n<=8 )
			{
			goto tt_2; // small matrix: no pack
			}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
		if( m<=2*m_kernel | n<=2*m_kernel | k<448 )
#else
		if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
			{
			if( m*4<=n | k<=4 ) // XXX k too !!!
				{
				goto tt_m0; // long matrix: pack A
				}
			else
				{
				goto tt_n0; // tall matrix: pack B
				}
			}
		}
	goto tt_1; // big matrix: pack A and B

	// never to get here
	return;


tt_m0:

	ii = 0;
#if 1
	for(; ii<m-3; ii+=4)
		{
		kernel_spack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_4x4_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_m0_left_4;
		}
#endif
	goto tt_m0_return;

tt_m0_left_4:
	kernel_spack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_m0_return;

tt_m0_return:
	return;



tt_n0:

	jj = 0;
#if 1
	for(; jj<n-3; jj+=4)
		{
		kernel_spack_nn_4_lib4(k, B+jj, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_sgemm_tt_4x4_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_sgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tt_n0_left_4;
		}
#endif
	goto tt_n0_return;

tt_n0_left_4:
	kernel_spack_nn_4_vs_lib4(k, B+jj, ldb, pU, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_sgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_n0_return;

tt_n0_return:
	return;



tt_1:

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_smat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_smat(ps, n1, k1);
	mem = malloc(tA_size+tB_size+64);
	blasfeo_align_64_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_smat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_smat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<k-7; ii+=8)
		{
		kernel_spack_tt_8_lib8(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb);
		}
	if(ii<k)
		{
		kernel_spack_tt_8_vs_lib8(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb, k-ii);
		}
#else
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_spack_tt_4_lib4(n, B+ii*ldb, ldb, tB.pA+ii*4, sdb);
		}
	if(ii<k)
		{
		kernel_spack_tt_4_vs_lib4(n, B+ii*ldb, ldb, tB.pA+ii*4, sdb, k-ii);
		}
#endif

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-23; ii+=24)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
		kernel_spack_tn_8_lib8(k, A+(ii+16)*lda, lda, tA.pA+2*sda*ps);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_24x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto tt_1_left_8;
			}
		else if(m-ii<=16)
			{
			goto tt_1_left_16;
			}
		else
			{
			goto tt_1_left_24;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-15; ii+=16)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
			kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_16x4_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		if(m-ii<=8)
			{
			goto tt_1_left_8;
			}
		else
			{
			goto tt_1_left_16;
			}
		}
#elif 0//defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
		for(jj=0; jj<n-7; jj+=8)
			{
			kernel_sgemm_nt_8x8_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//			kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(jj<n)
			{
			if(n-jj>4)
				{
				kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//				kernel_sgemm_nt_8x4_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
				}
			else
				{
				kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-jj);
				}
			}
		}
	if(ii<m)
		{
		goto tt_1_left_8;
		}
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_spack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
//			kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
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
		kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_sgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_1_left_4;
		}
#endif
	goto tt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_1_left_24:
	kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
	kernel_spack_tn_8_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps);
	kernel_spack_tn_8_vs_lib8(k, A+(ii+16)*lda, lda, tA.pA+2*sda*ps, m-ii-16);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_24x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tt_1_left_16:
	kernel_spack_tn_8_lib8(k, A+ii*lda, lda, tA.pA);
	kernel_spack_tn_8_vs_lib8(k, A+(ii+8)*lda, lda, tA.pA+sda*ps, m-ii-8);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_16x4_vs_lib88cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tt_1_left_8:
	kernel_spack_tn_8_vs_lib8(k, A+ii*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_sgemm_nt_8x8_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+4, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
		}
	if(jj<n)
		{
		kernel_sgemm_nt_8x4_vs_lib88cc(k, &alpha, tA.pA, tB.pA+jj*sdb+0, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
		}
	goto tt_1_return;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tt_1_left_8:
	kernel_spack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_spack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
//		kernel_sgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA+4*sda, tB.pA+jj*sdb, &beta, C+(ii+4)+jj*ldc, ldc, D+(ii+4)+jj*ldd, ldd, m-(ii+4), n-jj);
		}
	goto tt_1_return;
#endif

tt_1_left_4:
	kernel_spack_tn_4_vs_lib4(k, A+(ii+0)*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_1_return;

tt_1_return:
	free(mem);
	return;



tt_2:

	jj = 0;
#if 1
	for(; jj<n-3; jj+=4)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_sgemm_tt_4x4_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_sgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tt_2_left_4;
		}
#endif
	goto tt_2_return;

tt_2_left_4:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_sgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_2_return;

tt_2_return:
	return;



	// never to get here
	return;

	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_sgemm_nn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgemm_nt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgemm_tn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgemm_tt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



#endif

