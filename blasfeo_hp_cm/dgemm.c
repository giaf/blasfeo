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

//#include <xmmintrin.h>

#include <blasfeo_target.h>
#include <blasfeo_block_size.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>

#include <blasfeo_timing.h>



#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )
#define blasfeo_dmat blasfeo_cm_dmat
#define blasfeo_hp_dgemm_nn blasfeo_hp_cm_dgemm_nn
#define blasfeo_hp_dgemm_nt blasfeo_hp_cm_dgemm_nt
#define blasfeo_hp_dgemm_tn blasfeo_hp_cm_dgemm_tn
#define blasfeo_hp_dgemm_tt blasfeo_hp_cm_dgemm_tt
#define blasfeo_dgemm_nn blasfeo_cm_dgemm_nn
#define blasfeo_dgemm_nt blasfeo_cm_dgemm_nt
#define blasfeo_dgemm_tn blasfeo_cm_dgemm_tn
#define blasfeo_dgemm_tt blasfeo_cm_dgemm_tt
#endif



// TODO move to a header file to reuse across routines
#define EL_SIZE 8 // double precision

// TODO detect LLC cache size !!!!!!!!!
#if defined(TARGET_X64_INTEL_HASWELL)
#define M_KERNEL 12 // max kernel: 12x4
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define L2_CACHE_EL (256*1024/EL_SIZE) // L2 data cache size: 256 kB ; DTLB1 64*4kB = 256 kB
#define LLC_CACHE_EL (6*1024*1024/EL_SIZE) // LLC cache size: 6 MB
#define KC 256 // 192
#define NC 512 //72 // 120
#define MC 3000 //1500

#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#define M_KERNEL 8 // max kernel: 8x4
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define L2_CACHE_EL (256*1024/EL_SIZE) // L2 data cache size: 256 kB ; DTLB1 64*4kB = 256 kB
#define LLC_CACHE_EL (6*1024*1024/EL_SIZE) // LLC cache size: 4 MB
#define KC 320 //256 //320
#define NC 64 //72 //60 // 120
#define MC 800 //800

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A73)
#define M_KERNEL 8 // max kernel: 8x4
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define LLC_CACHE_EL (1*1024*1024/EL_SIZE) // LLC cache size: 1 MB
#define KC 320
#define NC 256
#define MC 3000

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
#define M_KERNEL 8 // max kernel: 8x4
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define LLC_CACHE_EL (1*1024*1024/EL_SIZE) // LLC cache size: 1 MB // 2 MB ???
#define KC 192
#define NC 48
#define MC 600

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
#define M_KERNEL 12 // max kernel: 12x4
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define LLC_CACHE_EL (256*1024/EL_SIZE) // LLC cache size: 256 kB
#define KC 160
#define NC 128
#define MC 3000

#else // assume generic target
#define M_KERNEL 4 // max kernel: 4x4
#define L1_CACHE_EL (32*1024/EL_SIZE) // L1 data cache size: 32 kB
#define CACHE_LINE_EL (64/EL_SIZE) // data cache size: 64 bytes // TODO 32-bytes for cortex A9
#define KC 512


#endif



// TODO implement packing here ???
static void blasfeo_hp_dgemm_nt_m1(int m, int n, int k, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *C, int ldc, double *D, int ldd)
	{

	int ii, jj;

	double *pA_p;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		jj = 0;
		if(n>0)
			{
			pA_p = m-ii<=12 ? pA : pA+(ii+12)*sda;
			kernel_dgemm_nt_12xn_p0_lib44cc(n, k, &alpha, pA+ii*sda, sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, pA_p);
			jj += n;
			}
#else
		for(jj=0; jj<n-3; jj+=4)
			{
#if defined(TARGET_X64_INTEL_HASWELL)
			kernel_dgemm_nt_12x4_p0_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#else
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#endif
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
#endif
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_m1_left_4;
			}
		if(m-ii<=8)
			{
			goto nn_m1_left_8;
			}
		else
			{
			goto nn_m1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)
			if(n-jj>4)
				{
				kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
				}
			else
				{
				pA_p = m-ii<=8 ? pA : pA+(ii+8)*sda;
				kernel_dgemm_nt_8x4_p_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, pA_p);
				}
#else
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
#endif
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_m1_left_4;
			}
		else
			{
			goto nn_m1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_m1_left_4;
		}
#endif
	goto nn_m1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_m1_left_12:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_m1_left_8:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, pA+ii*sda, sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m1_return;
#endif

nn_m1_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m1_return;

nn_m1_return:
	return;

	return;

	}



static void blasfeo_hp_dgemm_nt_n1(int m, int n, int k, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *C, int ldc, double *D, int ldd)
	{

	int ii, jj;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL) //| defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-11; jj+=12)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x12_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x12_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nn_n1_left_4;
			}
		if(n-jj<=8)
			{
			goto nn_n1_left_8;
			}
		else
			{
			goto nn_n1_left_12;
			}
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) //| defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; jj<n-7; jj+=8)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x8_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
//			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pA+ii*sda, pB+(jj+0)*sdb, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd);
//			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pA+ii*sda, pB+(jj+4)*sdb, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x8_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nn_n1_left_4;
			}
		else
			{
			goto nn_n1_left_8;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto nn_n1_left_4;
		}
#endif
	goto nn_n1_return;

#if defined(TARGET_X64_INTEL_HASWELL) //| defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_n1_left_12:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_n1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) //| defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_n1_left_8:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x8_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_n1_return;
#endif

nn_n1_left_4:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, pA+ii*sda, pB+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_n1_return;

nn_n1_return:
	return;

	return;

	}



static void blasfeo_hp_dgemm_nn_m0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_m0_left_4;
			}
		if(m-ii<=8)
			{
			goto nn_m0_left_8;
			}
		else
			{
			goto nn_m0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_m0_left_4;
			}
		else
			{
			goto nn_m0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_m0_left_4;
		}
#endif
	goto nn_m0_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_m0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_m0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_m0_return;
#endif

nn_m0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nn_4x12_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nn_4x8_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nn_4x8_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nn_m0_return;

nn_m0_return:
	return;

	}



static void blasfeo_hp_dgemm_nn_n0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, B+(jj+8)*ldb, ldb, pU+8*sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x12_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x12_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nn_n0_left_4;
			}
		else if(n-jj<=8)
			{
			goto nn_n0_left_8;
			}
		else
			{
			goto nn_n0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-7; jj+=8)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x8_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			kernel_dgemm_nt_4x8_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nn_n0_left_4;
			}
		else
			{
			goto nn_n0_left_8;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto nn_n0_left_4;
		}
#endif
	goto nn_n0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nn_n0_left_12:
	kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
	kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+8)*ldb, ldb, pU+8*sdu, n-jj-8);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_n0_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_n0_left_8:
	kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu, n-jj-4);
	for(ii=0; ii<m; ii+=4)
		{
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		kernel_dgemm_nt_4x8_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
		}
	goto nn_n0_return;
#endif

nn_n0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+0)*ldb, ldb, pU+0*sdu, n-jj-0);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-8; ii+=12)
		{
		kernel_dgemm_nt_12x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m-4)
		{
		kernel_dgemm_nt_8x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(ii<m)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(ii=0; ii<m-4; ii+=8)
		{
		kernel_dgemm_nt_8x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nn_n0_return;

nn_n0_return:
	return;


	}



static void blasfeo_hp_dgemm_nt_m0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_m0_left_4;
			}
		if(m-ii<=8)
			{
			goto nt_m0_left_8;
			}
		else
			{
			goto nt_m0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_m0_left_4;
			}
		else
			{
			goto nt_m0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_m0_left_4;
		}
#endif
	goto nt_m0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_m0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii+0, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_m0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nt_m0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii+0, lda, pU, sdu, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_m0_return;
#endif

nt_m0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, pU, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nt_4x12_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nt_4x8_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nt_4x8_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nt_m0_return;

nt_m0_return:
	return;

	}



static void blasfeo_hp_dgemm_nt_n0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{
		kernel_dpack_nn_12_lib4(k, B+jj, ldb, pU, sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x12_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x12_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nt_n0_left_4;
			}
		else if(n-jj<=8)
			{
			goto nt_n0_left_8;
			}
		else
			{
			goto nt_n0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-7; jj+=8)
		{
		kernel_dpack_nn_8_lib4(k, B+jj, ldb, pU, sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x8_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			kernel_dgemm_nt_4x8_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto nt_n0_left_4;
			}
		else
			{
			goto nt_n0_left_8;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{
		kernel_dpack_nn_4_lib4(k, B+jj, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto nt_n0_left_4;
		}
#endif
	goto nt_n0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_n0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, B+jj, ldb, pU, sdu, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_n0_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nt_n0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, B+jj, ldb, pU, sdu, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		kernel_dgemm_nt_4x8_vs_libc4cc(k, &alpha, A+ii, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
		}
	goto nt_n0_return;
#endif

nt_n0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, B+jj, ldb, pU, n-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-8; ii+=12)
		{
		kernel_dgemm_nt_12x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m-4)
		{
		kernel_dgemm_nt_8x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(ii<m)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(ii=0; ii<m-4; ii+=8)
		{
		kernel_dgemm_nt_8x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4cc(k, &alpha, A+ii, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nt_n0_return;

nt_n0_return:
	return;

	}



static void blasfeo_hp_dgemm_tn_m0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_m0_left_4;
			}
		if(m-ii<=8)
			{
			goto tn_m0_left_8;
			}
		else
			{
			goto tn_m0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tn_m0_left_4;
			}
		else
			{
			goto tn_m0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tn_m0_left_4;
		}
#endif
	goto tn_m0_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_m0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_m0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_m0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_m0_return;
#endif

tn_m0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nn_4x12_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nn_4x8_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nn_4x8_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_lib4ccc(k, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto tn_m0_return;

tn_m0_return:
	return;

	}



static void blasfeo_hp_dgemm_tn_n0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, B+(jj+8)*ldb, ldb, pU+8*sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x12_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x12_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tn_n0_left_4;
			}
		else if(n-jj<=8)
			{
			goto tn_n0_left_8;
			}
		else
			{
			goto tn_n0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-7; jj+=8)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x8_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			kernel_dgemm_tt_4x8_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tn_n0_left_4;
			}
		else
			{
			goto tn_n0_left_8;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{
		kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x4_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tn_n0_left_4;
		}
#endif
	goto tn_n0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tn_n0_left_12:
	kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
	kernel_dpack_tn_4_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+8)*ldb, ldb, pU+8*sdu, n-jj-8);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x12_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_n0_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_n0_left_8:
	kernel_dpack_tn_4_lib4(k, B+(jj+0)*ldb, ldb, pU);
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+4)*ldb, ldb, pU+4*sdu, n-jj-4);
	for(ii=0; ii<m; ii+=4)
		{
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		kernel_dgemm_tt_4x8_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
		}
	goto tn_n0_return;
#endif

tn_n0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, B+(jj+0)*ldb, ldb, pU+0*sdu, n-jj-0);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-8; ii+=12)
		{
		kernel_dgemm_tt_12x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m-4)
		{
		kernel_dgemm_tt_8x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(ii=0; ii<m-4; ii+=8)
		{
		kernel_dgemm_tt_8x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto tn_n0_return;

tn_n0_return:
	return;

	}



static void blasfeo_hp_dgemm_tt_m0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_m0_left_4;
			}
		if(m-ii<=8)
			{
			goto tt_m0_left_8;
			}
		else
			{
			goto tt_m0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto tt_m0_left_4;
			}
		else
			{
			goto tt_m0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_m0_left_4;
		}
#endif
	goto tt_m0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_m0_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_m0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tt_m0_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4ccc(k, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_m0_return;
#endif

tt_m0_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+ii*lda, lda, pU, m-ii);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nt_4x12_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nt_4x8_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nt_4x8_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4ccc(k, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto tt_m0_return;

tt_m0_return:
	return;

	}



static void blasfeo_hp_dgemm_tt_n0(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, double *D, int ldd, double *pU, int sdu)
	{

	int ii, jj;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{
		kernel_dpack_nn_12_lib4(k, B+jj, ldb, pU, sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x12_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x12_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tt_n0_left_4;
			}
		else if(n-jj<=8)
			{
			goto tt_n0_left_8;
			}
		else
			{
			goto tt_n0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-7; jj+=8)
		{
		kernel_dpack_nn_8_lib4(k, B+jj, ldb, pU, sdu);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x8_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
			kernel_dgemm_tt_4x8_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tt_n0_left_4;
			}
		else
			{
			goto tt_n0_left_8;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{
		kernel_dpack_nn_4_lib4(k, B+jj, ldb, pU);
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x4_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tt_n0_left_4;
		}
#endif
	goto tt_n0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_n0_left_12:
	kernel_dpack_nn_12_vs_lib4(k, B+jj, ldb, pU, sdu, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x12_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_n0_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tt_n0_left_8:
	kernel_dpack_nn_8_vs_lib4(k, B+jj, ldb, pU, sdu, n-jj);
	for(ii=0; ii<m; ii+=4)
		{
//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		kernel_dgemm_tt_4x8_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, sdu, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
//#else
//		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+(jj+0)*ldc, ldc, D+ii+(jj+0)*ldd, ldd, m-ii, n-(jj+0));
//		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU+4*sdu, &beta, C+ii+(jj+4)*ldc, ldc, D+ii+(jj+4)*ldd, ldd, m-ii, n-(jj+4));
//#endif
		}
	goto tt_n0_return;
#endif

tt_n0_left_4:
	kernel_dpack_nn_4_vs_lib4(k, B+jj, ldb, pU, n-jj);
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-8; ii+=12)
		{
		kernel_dgemm_tt_12x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m-4)
		{
		kernel_dgemm_tt_8x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(ii=0; ii<m-4; ii+=8)
		{
		kernel_dgemm_tt_8x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x4_vs_libc4cc(k, &alpha, A+ii*lda, lda, pU, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto tt_n0_return;

tt_n0_return:
	return;

	}




#ifdef HP_BLAS

static void blas_hp_dgemm_nn(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
	{

#if defined(PRINT_NAME)
	printf("\nblas_hp_dgemm_nn %d %d %d %f %p %d %p %d %f %p %d\n", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

	if(m<=0 | n<=0)
		return;

	int ldd = ldc;
	double *D = C;

#else

void blasfeo_hp_dgemm_nn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dgemm_nn (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *B = sB->pA + bi + bj*ldb;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

#endif

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj, ll;
	int iii;
	int mc, nc, kc;
	int mleft, nleft, kleft;
	int mc0, nc0, kc0;
	int ldc1;
	double beta1;
	double *pA, *pB, *C1;
	int pack_A;

	const int ps = 4; //D_PS;

#if defined(TARGET_GENERIC)
	double pU[M_KERNEL*K_MAX_STACK];
#else
//	ALIGNED( double pU[M_KERNEL*K_MAX_STACK], 64 );
	ALIGNED( double pU[M_KERNEL*K_MAX_STACK], 4096 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
#if defined(TARGET_X64_INTEL_HASWELL)
	const int l2_cache_el = L2_CACHE_EL;
#endif
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const int llc_cache_el = LLC_CACHE_EL;
#endif
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

	int m_a = m==lda ? m : m_cache;
	int m_a_kernel = m<=m_kernel ? m_a : m_kernel_cache;
	int k_b = k==ldb ? k : k_cache;
	int m_c = m==ldc ? m : m_cache;
	int m_d = m==ldd ? m : m_cache;

#if defined(PACKING_ALG_2)
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	goto nn_m0; // pack A
#else
	goto nn_2; // no pack
#endif
#endif
#if defined(PACKING_ALG_M0)
	goto nn_m0; // pack A
#endif
#if defined(PACKING_ALG_N0)
	goto nn_n0; // pack B
#endif
#if defined(PACKING_ALG_1)
	goto nn_1; // pack A and B
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( (m<=m_kernel & n<=m_kernel) | (m_a_kernel*k + k_b*n <= l1_cache_el) )
		{
//		printf("\nalg 2\n");
		goto nn_2; // small matrix: no pack
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=48 & n<=48 & k<=K_MAX_STACK )
		{
		goto nn_m0; // small matrix: pack A
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=m_kernel & n<=m_kernel & k<160) )
		{
//		printf("\nalg 2\n");
		goto nn_2; // small matrix: no pack
		}
#else
	if( m<=8 & n<=8 )
		{
		goto nn_2; // small matrix: no pack
		}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
	if( m<n*4 )
		{
//		if( m<=2*m_kernel | m_a*k + k_b*n <= llc_cache_el )
		if( m<=2*m_kernel | m_a*k + k_b*n + m_c*n + m_d*n <= llc_cache_el )
			{
//			printf("\nalg m0\n");
			goto nn_m0; // long matrix: pack A
			}
		}
	else
		{
		if( n<=2*m_kernel | m_a*k <= l2_cache_el )
			{
//			printf("\nalg n0\n");
			goto nn_n0; // tall matrix: pack B
			}
		}
#else
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=2*m_kernel | n<=2*m_kernel | k<56 )
#elif defined(TARGET_X64_INTEL_CORE)
	if( m<=1*m_kernel | n<=1*m_kernel | k<8 )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( m<=2*m_kernel | n<=2*m_kernel | m_a*k + k_b*n <= llc_cache_el )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=2*m_kernel | n<=2*m_kernel) & k<160 )
#else
	if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
		{
		if( m<=n*4 )
			{
//			printf("\nalg m0\n");
			goto nn_m0; // long matrix: pack A
			}
		else
			{
//			printf("\nalg n0\n");
			goto nn_n0; // tall matrix: pack B
			}
		}
#endif
//	printf("\nalg 1\n");
	goto nn_1; // big matrix: pack A and B
	
	// never to get here
	return;


nn_m0:
	
	if(K_MAX_STACK<=0)
		goto nn_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_nn_m0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
#if 0
			if(k-ll<2*kc)
				{
				if(k-ll<=kc) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}
#else
			kleft = k-ll<kc ? k-ll : kc;
#endif

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_nn_m0(m, n, kleft, alpha, A+ll*lda, lda, B+ll, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}

		}

	return;



nn_n0:

	if(K_MAX_STACK<=0)
		goto nn_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_nn_n0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_nn_n0(m, n, kleft, alpha, A+ll*lda, lda, B+ll, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;



nn_1:

//#if 0
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)

	// cache blocking alg

	mc0 = MC;
	nc0 = NC;
	kc0 = KC;

//	mc0 = 12;
//	nc0 = 8;
//	kc0 = 4;

	mc = m<mc0 ? m : mc0;
	nc = n<nc0 ? n : nc0;
	kc = k<kc0 ? k : kc0;

//	tA_size = blasfeo_pm_memsize_dmat(ps, mc, kc);
//	tB_size = blasfeo_pm_memsize_dmat(ps, nc, kc);
	tA_size = blasfeo_pm_memsize_dmat(ps, mc0, kc0);
	tB_size = blasfeo_pm_memsize_dmat(ps, nc0, kc0);
//	printf("\nsize %d %d page %d %d\n", tA_size, tB_size, tA_size%4096, tB_size%4096);
	tA_size = (tA_size + 4096 - 1) / 4096 * 4096;
	tB_size = (tB_size + 4096 - 1) / 4096 * 4096;
//	printf("\nsize %d %d page %d %d\n", tA_size, tB_size, tA_size%4096, tB_size%4096);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+2*4096);
//	mem = malloc(2*tA_size+2*tB_size+2*4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);

//	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, mc0, kc0, &tA, (void *) mem_align);
	mem_align += tA_size;

	mem_align += 4096-4*128;
//	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, nc0, kc0, &tB, (void *) mem_align);
	mem_align += tB_size;

//	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) (mem_align+0));
//	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align+tB_size);

//	double time_pack_A = 0.0;
//	double time_pack_B = 0.0;
//	double time_kernel = 0.0;

//	blasfeo_timer timer;

	pA = tA.pA;
	pB = tB.pA;

	for(ii=0; ii<m; ii+=mleft)
		{

		mleft = m-ii<mc ? m-ii : mc;

		for(ll=0; ll<k; ll+=kleft)
			{

#if 1
			if(k-ll<2*kc0)
				{
				if(k-ll<=kc0) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}
#else
			kleft = k-ll<kc ? k-ll : kc;
#endif

			sda = (kleft+4-1)/4*4;
			sdb = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

//	for(ii=0; ii<m; ii+=mleft)
//		{
//
//		mleft = m-ii<mc ? m-ii : mc;

			// pack A
	//		blasfeo_tic(&timer);
#if 1
			for(iii=0; iii<kleft-3; iii+=4)
				{
				kernel_dpack_tt_4_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda);
				}
			if(iii<kleft)
				{
				kernel_dpack_tt_4_vs_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda, kleft-iii);
				}
#endif
	//		time_pack_A += blasfeo_toc(&timer);

			for(jj=0; jj<n; jj+=nleft)
				{

				nleft = n-jj<nc ? n-jj : nc;

				// pack and tran B
	//			blasfeo_tic(&timer);
#if 1
//				for(iii=0; iii<nleft-3; iii+=4)
				for(iii=0; iii<nleft-4; iii+=4)
					{
					kernel_dpack_tn_4_p0_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb);
//					d_print_mat(4, kleft, pB+iii*sdb, 4);
//					exit(1);
					}
				if(iii<nleft)
					{
					kernel_dpack_tn_4_vs_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb, nleft-iii);
					}
#endif
	//			time_pack_B += blasfeo_toc(&timer);

				// prefetch A
//				for(iii=0; iii<kleft; iii+=2)
//					{
//					__builtin_prefetch(pA+4*iii);
//					__builtin_prefetch(pA+4*sda+4*iii);
//					}
				// prefetch B
//				for(iii=0; iii<nleft*kleft; iii+=8)
//					{
//					__builtin_prefetch(pB+iii, 0, 2);
//					__builtin_prefetch(pB+iii);
//					_mm_prefetch(pB+iii, _MM_HINT_T1);
//					}

	//			blasfeo_tic(&timer);
				blasfeo_hp_dgemm_nt_m1(mleft, nleft, kleft, alpha, pA, sda, pB, sdb, beta1, C1+ii+jj*ldc1, ldc1, D+ii+jj*ldd, ldd);
	//			time_kernel += blasfeo_toc(&timer);

				}
			
			}

		}

//	printf("\ntime: pack_A %e, pack_B %e, kernel %e, kernel2 %e, kernel3 %e\n", time_pack_A, time_pack_B, time_kernel, time_kernel2, time_kernel3); 
	free(mem);

	return;

#elif 0 //defined(TARGET_ARMV8A_ARM_CORTEX_A57)

	// cache blocking alg

	mc0 = NC; //MC;
	nc0 = MC; //NC;
	kc0 = KC;

#if 0
	mc0 = 8;//12;
	nc0 = 8;
	kc0 = 4;
#endif

	mc = m<mc0 ? m : mc0;
	nc = n<nc0 ? n : nc0;
	kc = k<kc0 ? k : kc0;

	tA_size = blasfeo_pm_memsize_dmat(ps, mc, kc);
	tB_size = blasfeo_pm_memsize_dmat(ps, nc, kc);
	tA_size = (tA_size + 4096 - 1) / 4096 * 4096;
	tB_size = (tB_size + 4096 - 1) / 4096 * 4096;
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);

	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align);
	mem_align += tA_size;

	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) mem_align);
	mem_align += tB_size;

//	double time_pack_A = 0.0;
//	double time_pack_B = 0.0;
//	double time_kernel = 0.0;

//	blasfeo_timer timer;

	pA = tA.pA;
	pB = tB.pA;

	for(jj=0; jj<n; jj+=nleft)
		{

		nleft = n-jj<nc ? n-jj : nc;

		for(ll=0; ll<k; ll+=kleft)
			{

#if 1
			if(k-ll<2*kc0)
				{
				if(k-ll<=kc0) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}
#else
			kleft = k-ll<kc ? k-ll : kc;
#endif

			sda = (kleft+4-1)/4*4;
			sdb = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			// pack and tran B
//			blasfeo_tic(&timer);
#if 1
			for(iii=0; iii<nleft-3; iii+=4)
				{
				kernel_dpack_tn_4_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb);
				}
			if(iii<nleft)
				{
				kernel_dpack_tn_4_vs_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb, nleft-iii);
				}
#endif
//			time_pack_B += blasfeo_toc(&timer);

			for(ii=0; ii<m; ii+=mleft)
				{

				mleft = m-ii<mc ? m-ii : mc;

				// pack A
		//		blasfeo_tic(&timer);
#if 1
				for(iii=0; iii<kleft-3; iii+=4)
					{
					kernel_dpack_tt_4_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda);
					}
				if(iii<kleft)
					{
					kernel_dpack_tt_4_vs_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda, kleft-iii);
					}
#endif
		//		time_pack_A += blasfeo_toc(&timer);

	//			blasfeo_tic(&timer);
				blasfeo_hp_dgemm_nt_n1(mleft, nleft, kleft, alpha, pA, sda, pB, sdb, beta1, C1+ii+jj*ldc1, ldc1, D+ii+jj*ldd, ldd);
//				dgemm_kernel(mleft, nleft, kleft, alpha, pA, pB, D+ii+jj*ldd, ldd);
//				dgemm_kernel(nleft, mleft, kleft, alpha, pB, pA, D+jj+ii*ldd, ldd);
//				d_print_mat(mleft, nleft, D+ii+jj*ldd, ldd);
//				exit(1);
	//			time_kernel += blasfeo_toc(&timer);

				}
			
			}

		}

//	printf("\ntime: pack_A %e, pack_B %e, kernel %e, kernel2 %e, kernel3 %e\n", time_pack_A, time_pack_B, time_kernel, time_kernel2, time_kernel3); 
	free(mem);

	return;

#else

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_dmat(ps, n1, k1);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

//	blasfeo_pack_tran_dmat(k, n, B, ldb, &tB, 0, 0);
	pack_B = 1;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
		kernel_dpack_nn_4_lib4(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
		}
	if(ii<m)
		{
		goto nn_1_left_4;
		}
#endif
	goto nn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_1_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nn_1_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_1_return;
#endif

nn_1_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_1_return;

nn_1_return:
	free(mem);
	return;

#endif


nn_2:
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_12x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_2_left_4;
			}
		if(m-ii<=8)
			{
			goto nn_2_left_8;
			}
		else
			{
			goto nn_2_left_12;
			}
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_8x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nn_2_left_4;
			}
		else
			{
			goto nn_2_left_8;
			}
		}
#elif ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nn_2_left_4;
		}
#endif
	goto nn_2_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nn_2_left_12:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_12x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_2_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
nn_2_left_8:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_8x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nn_2_return;
#endif

#if ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nn_2_left_4:
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nn_4x12_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nn_4x8_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nn_4x8_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nn_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nn_2_return;
#endif

nn_2_return:
	return;


	// never to get here
	return;

	}



#ifdef HP_BLAS

static void blas_hp_dgemm_nt(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
	{

#if defined(PRINT_NAME)
	printf("\nblas_hp_dgemm_nt %d %d %d %f %p %d %p %d %f %p %d\n", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

	if(m<=0 | n<=0)
		return;

	int ldd = ldc;
	double *D = C;

#else

void blasfeo_hp_dgemm_nt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dgemm_nt (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *B = sB->pA + bi + bj*ldb;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

#endif

	int ii, jj, ll;
	int iii;
	int mc, nc, kc;
	int mleft, nleft, kleft;
	int mc0, nc0, kc0;
	int ldc1;
	double beta1;
	double *pA, *pB, *C1;

#if defined(TARGET_GENERIC)
	double pU[4*K_MAX_STACK];
#else
	ALIGNED( double pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int ps = D_PS;
	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
#if defined(TARGET_X64_INTEL_HASWELL)
	const int l2_cache_el = L2_CACHE_EL;
#endif
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const int llc_cache_el = LLC_CACHE_EL;
#endif
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

	int m_a = m==lda ? m : m_cache;
	int m_a_kernel = m<=m_kernel ? m_a : m_kernel_cache;
	int n_b = n==ldb ? n : n_cache;

#if defined(PACKING_ALG_2)
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	goto nt_m0; // pack A
#else
	goto nt_2; // no pack
#endif
#endif
#if defined(PACKING_ALG_M0)
	goto nt_m0; // pack A
#endif
#if defined(PACKING_ALG_N0)
	goto nt_n0; // pack B
#endif
#if defined(PACKING_ALG_1)
	goto nt_1; // pack A and B
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) 
	if( (m<=m_kernel & n<=m_kernel) | (m_a_kernel*k + n_b*k <= l1_cache_el) )
		{
//		printf("\nalg 2\n");
		goto nt_2; // small matrix: no pack
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=48 & n<=48 & k<=K_MAX_STACK )
		{
		goto nt_m0; // small matrix: pack A
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=m_kernel & n<=m_kernel & k<160) )
		{
//		printf("\nalg 2\n");
		goto nt_2; // small matrix: no pack
		}
#else
	if( m<=8 & n<=8 )
		{
		goto nt_2; // small matrix: no pack
		}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
	if( m<=n )
		{
		if( m<=2*m_kernel | n_b*k <= l2_cache_el )
			{
//			printf("\nalg m0\n");
			goto nt_m0; // long matrix: pack A
			}
		}
	else
		{
		if( n<=2*m_kernel | m_a*k <= l2_cache_el )
			{
//			printf("\nalg n0\n");
			goto nt_n0; // tall matrix: pack B
			}
		}
#else
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=2*m_kernel | n<=2*m_kernel | k<56 )
#elif defined(TARGET_X64_INTEL_CORE)
	if( m<=1*m_kernel | n<=1*m_kernel | k<8 )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( m<=2*m_kernel | n<=2*m_kernel | m_a*k + n_b*k <= llc_cache_el )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=2*m_kernel | n<=2*m_kernel) & k<160 )
#else
	if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
		{
		if( m<=n )
			{
//			printf("\nalg m0\n");
			goto nt_m0; // long matrix: pack A
			}
		else
			{
//			printf("\nalg n0\n");
			goto nt_n0; // tall matrix: pack B
			}
		}
#endif
//	printf("\nalg 1\n");
	goto nt_1; // big matrix: pack A and B



nt_m0:

	if(K_MAX_STACK<=0)
		goto nt_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_nt_m0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_nt_m0(m, n, kleft, alpha, A+ll*lda, lda, B+ll*ldb, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;



nt_n0:

	if(K_MAX_STACK<=0)
		goto nt_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_nt_n0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_nt_n0(m, n, kleft, alpha, A+ll*lda, lda, B+ll*ldb, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;




nt_1:

//#if 0
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)

	// cache blocking alg

	mc0 = MC;
	nc0 = NC;
	kc0 = KC;

//	mc0 = 12;
//	nc0 = 8;
//	kc0 = 4;

	mc = m<mc0 ? m : mc0;
	nc = n<nc0 ? n : nc0;
	kc = k<kc0 ? k : kc0;

	tA_size = blasfeo_pm_memsize_dmat(ps, mc, kc);
	tB_size = blasfeo_pm_memsize_dmat(ps, nc, kc);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) (mem_align+tA_size));

	pA = tA.pA;
	pB = tB.pA;

	for(ii=0; ii<m; ii+=mleft)
		{

		mleft = m-ii<mc ? m-ii : mc;

		for(ll=0; ll<k; ll+=kleft)
			{

			if(k-ll<2*kc0)
				{
				if(k-ll<=kc0) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}

			sda = (kleft+4-1)/4*4;
			sdb = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			// pack A
			for(iii=0; iii<kleft-3; iii+=4)
				{
				kernel_dpack_tt_4_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda);
				}
			if(iii<kleft)
				{
				kernel_dpack_tt_4_vs_lib4(mleft, A+ii+(ll+iii)*lda, lda, pA+iii*ps, sda, kleft-iii);
				}

			for(jj=0; jj<n; jj+=nleft)
				{

				nleft = n-jj<nc ? n-jj : nc;

				// pack B
	//			printf("\nhere\n");
	//			d_print_mat(nleft, kleft, B+jj+ll*ldb, ldb);
				for(iii=0; iii<kleft-3; iii+=4)
					{
	//				d_print_mat(nleft, 4, B+jj+(ll+iii)*ldb, ldb);
					kernel_dpack_tt_4_lib4(nleft, B+jj+(ll+iii)*ldb, ldb, pB+iii*ps, sdb);
					}
				if(iii<kleft)
					{
					kernel_dpack_tt_4_vs_lib4(nleft, B+jj+(ll+iii)*ldb, ldb, pB+iii*ps, sdb, kleft-iii);
					}
	//			d_print_mat(4, kleft, pB, 4);
	//			d_print_mat(4, kleft, pB+4*sdb, 4);

				blasfeo_hp_dgemm_nt_m1(mleft, nleft, kleft, alpha, pA, sda, pB, sdb, beta1, C1+ii+jj*ldc1, ldc1, D+ii+jj*ldd, ldd);
	//			d_print_mat(m, nleft, D+jj*ldd, ldd);

				}

			}
		
		}

	free(mem);

	return;

#else

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_dmat(ps, n1, k1);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

//	blasfeo_pack_dmat(n, k, B, ldb, &tB, 0, 0);
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_dpack_tt_4_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb);
		}
	if(ii<k)
		{
		kernel_dpack_tt_4_vs_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb, k-ii);
		}

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, tA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
		kernel_dpack_nn_4_lib4(k, A+ii, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_1_left_4;
		}
#endif
	goto nt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nt_1_left_12:
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
nt_1_left_8:
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, tA.pA, sda, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_1_return;
#endif

nt_1_left_4:
	kernel_dpack_nn_4_vs_lib4(k, A+ii, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_1_return;

nt_1_return:
	free(mem);
	return;

#endif



nt_2:
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_2_left_4;
			}
		if(m-ii<=8)
			{
			goto nt_2_left_8;
			}
		else
			{
			goto nt_2_left_12;
			}
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto nt_2_left_4;
			}
		else
			{
			goto nt_2_left_8;
			}
		}
#elif ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto nt_2_left_4;
		}
#endif
	goto nt_2_return;

#if defined(TARGET_X64_INTEL_HASWELL)
nt_2_left_12:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_2_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
nt_2_left_8:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto nt_2_return;
#endif

#if ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
nt_2_left_4:
#if defined(TARGET_X64_INTEL_HASWELL)
	for(jj=0; jj<n-8; jj+=12)
		{
		kernel_dgemm_nt_4x12_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n-4)
		{
		kernel_dgemm_nt_4x8_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(jj=0; jj<n-4; jj+=8)
		{
		kernel_dgemm_nt_4x8_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(jj<n)
		{
		kernel_dgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_libcccc(k, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto nt_2_return;
#endif

nt_2_return:
	return;


	// never to get here
	return;

	}



#ifdef HP_BLAS

static void blas_hp_dgemm_tn(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
	{

#if defined(PRINT_NAME)
	printf("\nblas_hp_dgemm_tn %d %d %d %f %p %d %p %d %f %p %d\n", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

	if(m<=0 | n<=0)
		return;

	int ldd = ldc;
	double *D = C;

#else

void blasfeo_hp_dgemm_tn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dgemm_tn (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *B = sB->pA + bi + bj*ldb;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

#endif

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj, ll;
	int iii;
	int mc, nc, kc;
	int mleft, nleft, kleft;
	int mc0, nc0, kc0;
	int ldc1;
	double beta1;
	double *pA, *pB, *C1;

#if defined(TARGET_GENERIC)
	double pU[4*K_MAX_STACK];
#else
	ALIGNED( double pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int ps = D_PS;
	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const int llc_cache_el = LLC_CACHE_EL;
#endif
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

	int k_a = k==lda ? k : k_cache;
	int k_b = k==ldb ? k : k_cache;
	int m_c = m==ldc ? m : m_cache;
	int m_d = m==ldd ? m : m_cache;

#if defined(PACKING_ALG_M0)
	goto tn_m0; // pack A
#endif
#if defined(PACKING_ALG_N0)
	goto tn_n0; // pack B
#endif
#if defined(PACKING_ALG_1)
	goto tn_1; // pack A and B
#endif

	// no algorithm for small matrix
#if defined(TARGET_X64_INTEL_HASWELL)
	if( m<=2*m_kernel | n<=2*m_kernel | k_a*m + k_b*n + m_c*n + m_d*n <= llc_cache_el )
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=2*m_kernel | n<=2*m_kernel | k<56 )
#elif defined(TARGET_X64_INTEL_CORE)
	if( m<=1*m_kernel | n<=1*m_kernel | k<8 )
#elif defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( m<=2*m_kernel | n<=2*m_kernel | k_a*m + k_b*n <= llc_cache_el )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=2*m_kernel | n<=2*m_kernel) & k<160 )
#else
	if( m<=1*m_kernel | n<=1*m_kernel | k<12 )
#endif
		{
		if( m<=n )
			{
//			printf("\nalg m0\n");
			goto tn_m0; // long matrix: pack A
			}
		else
			{
//			printf("\nalg n0\n");
			goto tn_n0; // tall matrix: pack B
			}
		}
//	printf("\nalg 1\n");
	goto tn_1; // big matrix: pack A and B

	// never to get here
	return;


tn_m0:

	if(K_MAX_STACK<=0)
		goto tn_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_tn_m0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_tn_m0(m, n, kleft, alpha, A+ll, lda, B+ll, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}

		}

	return;



tn_n0:

	if(K_MAX_STACK<=0)
		goto tn_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_tn_n0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_tn_n0(m, n, kleft, alpha, A+ll, lda, B+ll, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;



tn_1:

//#if 0
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)

	// cache blocking alg

	mc0 = MC;
	nc0 = NC;
	kc0 = KC;

//	mc0 = 12;
//	nc0 = 8;
//	kc0 = 4;

	mc = m<mc0 ? m : mc0;
	nc = n<nc0 ? n : nc0;
	kc = k<kc0 ? k : kc0;

	tA_size = blasfeo_pm_memsize_dmat(ps, mc, kc);
	tB_size = blasfeo_pm_memsize_dmat(ps, nc, kc);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) (mem_align+tA_size));

	pA = tA.pA;
	pB = tB.pA;

	for(ii=0; ii<m; ii+=mleft)
		{

		mleft = m-ii<mc ? m-ii : mc;

		for(ll=0; ll<k; ll+=kleft)
			{

			if(k-ll<2*kc0)
				{
				if(k-ll<=kc0) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}

			sda = (kleft+4-1)/4*4;
			sdb = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			// pack A
			for(iii=0; iii<mleft-3; iii+=4)
				{
				kernel_dpack_tn_4_lib4(kleft, A+ll+(ii+iii)*lda, lda, pA+iii*sda);
				}
			if(iii<mleft)
				{
				kernel_dpack_tn_4_vs_lib4(kleft, A+ll+(ii+iii)*lda, lda, pA+iii*sda, mleft-iii);
				}

			for(jj=0; jj<n; jj+=nleft)
				{

				nleft = n-jj<nc ? n-jj : nc;

				// pack B
				for(iii=0; iii<nleft-3; iii+=4)
					{
					kernel_dpack_tn_4_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb);
					}
				if(iii<nleft)
					{
					kernel_dpack_tn_4_vs_lib4(kleft, B+ll+(jj+iii)*ldb, ldb, pB+iii*sdb, nleft-iii);
					}

				blasfeo_hp_dgemm_nt_m1(mleft, nleft, kleft, alpha, pA, sda, pB, sdb, beta1, C1+ii+jj*ldc1, ldc1, D+ii+jj*ldd, ldd);

				}

			}

		}

	free(mem);

	return;

#else

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_dmat(ps, n1, k1);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

//	blasfeo_pack_tran_dmat(k, n, B, ldb, &tB, 0, 0);
	pack_B = 1;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, tA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			if(pack_B)
				kernel_dpack_tn_4_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb);
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			if(pack_B)
				kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		pack_B = 0;
		}
		if(ii<m)
		{
		goto tn_1_left_4;
		}
#endif
	goto tn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_1_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, tA.pA+8*sda, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tn_1_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_1_return;
#endif

tn_1_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+0)*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		if(pack_B)
			kernel_dpack_tn_4_vs_lib4(k, B+jj*ldb, ldb, tB.pA+jj*sdb, n-jj);
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tn_1_return;

tn_1_return:
free(mem);
	return;

#endif


	// never to get here
	return;

	}



#ifdef HP_BLAS

static void blas_hp_dgemm_tt(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
	{

#if defined(PRINT_NAME)
	printf("\nblas_hp_dgemm_tt %d %d %d %f %p %d %p %d %f %p %d\n", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

	if(m<=0 | n<=0)
		return;

	int ldd = ldc;
	double *D = C;

#else

void blasfeo_hp_dgemm_tt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{

#if defined(PRINT_NAME)
	printf("\nblasfeo_hp_dgemm_tt (cm) %d %d %d %f %p %d %d %p %d %d %f %p %d %d %p %d %d\n", m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#endif

	if(m<=0 | n<=0)
		return;

	// extract pointer to column-major matrices from structures
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	double *A = sA->pA + ai + aj*lda;
	double *B = sB->pA + bi + bj*ldb;
	double *C = sC->pA + ci + cj*ldc;
	double *D = sD->pA + di + dj*ldd;

#endif

//	printf("\n%p %d %p %d %p %d %p %d\n", A, lda, B, ldb, C, ldc, D, ldd);

	int ii, jj, ll;
	int iii;
	int mc, nc, kc;
	int mleft, nleft, kleft;
	int mc0, nc0, kc0;
	int ldc1;
	double beta1;
	double *pA, *pB, *C1;

#if defined(TARGET_GENERIC)
	double pU[4*K_MAX_STACK];
#else
	ALIGNED( double pU[M_KERNEL*K_MAX_STACK], 64 );
#endif
	int sdu = (k+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;

	struct blasfeo_pm_dmat tA, tB;
	int sda, sdb;
	int tA_size, tB_size;
	void *mem;
	char *mem_align;
	int m1, n1, k1;
	int pack_B;

	const int ps = D_PS;
	const int m_kernel = M_KERNEL;
	const int l1_cache_el = L1_CACHE_EL;
#if defined(TARGET_X64_INTEL_HASWELL)
	const int l2_cache_el = L2_CACHE_EL;
#endif
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const int llc_cache_el = LLC_CACHE_EL;
#endif
	const int reals_per_cache_line = CACHE_LINE_EL;

	const int m_cache = (m+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int n_cache = (n+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int k_cache = (k+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	const int m_kernel_cache = (m_kernel+reals_per_cache_line-1)/reals_per_cache_line*reals_per_cache_line;
	int m_min = m_cache<m_kernel_cache ? m_cache : m_kernel_cache;
//	int n_min = n_cache<m_kernel_cache ? n_cache : m_kernel_cache;

	int k_a = k==lda ? k : k_cache;
	int n_b = n==ldb ? n : n_cache;
	int n_b_kernel = n<=m_kernel ? n_b : m_kernel_cache;
	int m_c = m==ldc ? m : m_cache;
	int m_d = m==ldd ? m : m_cache;

#if defined(PACKING_ALG_2)
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	goto tt_m0; // pack A
#else
	goto tt_2; // no pack
#endif
#endif
#if defined(PACKING_ALG_M0)
	goto tt_m0; // pack A
#endif
#if defined(PACKING_ALG_N0)
	goto tt_n0; // pack B
#endif
#if defined(PACKING_ALG_1)
	goto tt_1; // pack A and B
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( (m<=m_kernel & n<=m_kernel) | (k_a*m + n_b_kernel*k <= l1_cache_el) )
		{
		goto tt_2; // small matrix: no pack
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=48 & n<=48 & k<=K_MAX_STACK )
		{
		goto tt_m0; // small matrix: pack A
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=m_kernel & n<=m_kernel & k<160) )
		{
//		printf("\nalg 2\n");
		goto tt_2; // small matrix: no pack
		}
#else
	if( m<=8 & n<=8 )
		{
		goto tt_2; // small matrix: no pack
		}
#endif
#if defined(TARGET_X64_INTEL_HASWELL)
	if( m*4<=n | k<=4 ) // XXX k too !!!
		{
		if( m<=2*m_kernel | n_b*k <= l2_cache_el )
			{
//				printf("\nalg m0\n");
			goto tt_m0; // long matrix: pack A
			}
		}
	else
		{
		if( n<=2*m_kernel | k_a*m + n_b*k + m_c*n + m_d*n <= llc_cache_el )
			{
//				printf("\nalg n0\n");
			goto tt_n0; // tall matrix: pack B
			}
		}
#else
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if( m<=2*m_kernel | n<=2*m_kernel | k<56 )
#elif defined(TARGET_X64_INTEL_CORE)
	if( m<=1*m_kernel | n<=1*m_kernel | k<8 )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	if( m<=2*m_kernel | n<=2*m_kernel | k_a*m + n_b*k <= llc_cache_el )
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	if( (m<=2*m_kernel | n<=2*m_kernel) & k<160 )
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
#endif
	goto tt_1; // big matrix: pack A and B

	// never to get here
	return;


tt_m0:

	if(K_MAX_STACK<=0)
		goto tt_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_tt_m0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_tt_m0(m, n, kleft, alpha, A+ll, lda, B+ll*ldb, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;



tt_n0:

	if(K_MAX_STACK<=0)
		goto tt_1;

	// k-blocking alg

	kc = K_MAX_STACK<KC ? K_MAX_STACK : KC;
//	kc = 4;

	if(k<kc)
		{
		blasfeo_hp_dgemm_tt_n0(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, pU, sdu);
		}
	else
		{
		for(ll=0; ll<k; ll+=kleft)
			{
			kleft = k-ll<kc ? k-ll : kc;

			sdu = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			blasfeo_hp_dgemm_tt_n0(m, n, kleft, alpha, A+ll, lda, B+ll*ldb, ldb, beta1, C1, ldc1, D, ldd, pU, sdu);
			}
		}

	return;



tt_1:

//#if 0
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)

	// cache blocking alg

	mc0 = MC;
	nc0 = NC;
	kc0 = KC;

//	mc0 = 12;
//	nc0 = 8;
//	kc0 = 4;

	mc = m<mc0 ? m : mc0;
	nc = n<nc0 ? n : nc0;
	kc = k<kc0 ? k : kc0;

	tA_size = blasfeo_pm_memsize_dmat(ps, mc, kc);
	tB_size = blasfeo_pm_memsize_dmat(ps, nc, kc);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, mc, kc, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, nc, kc, &tB, (void *) (mem_align+tA_size));

	pA = tA.pA;
	pB = tB.pA;

	for(ii=0; ii<m; ii+=mleft)
		{

		mleft = m-ii<mc ? m-ii : mc;

		for(ll=0; ll<k; ll+=kleft)
			{

			if(k-ll<2*kc0)
				{
				if(k-ll<=kc0) // last
					{
					kleft = k-ll;
					}
				else // second last
					{
					kleft = (k-ll+1)/2;
					kleft = (kleft+4-1)/4*4;
					}
				}
			else
				{
				kleft = kc;
				}

			sda = (kleft+4-1)/4*4;
			sdb = (kleft+4-1)/4*4;

			beta1 = ll==0 ? beta : 1.0;
			C1 = ll==0 ? C : D;
			ldc1 = ll==0 ? ldc : ldd;

			// pack A
			for(iii=0; iii<mleft-3; iii+=4)
				{
				kernel_dpack_tn_4_lib4(kleft, A+ll+(ii+iii)*lda, lda, pA+iii*sda);
				}
			if(iii<mleft)
				{
				kernel_dpack_tn_4_vs_lib4(kleft, A+ll+(ii+iii)*lda, lda, pA+iii*sda, mleft-iii);
				}

			for(jj=0; jj<n; jj+=nc)
				{

				nleft = n-jj<nc ? n-jj : nc;

				// pack B
	//			printf("\nhere\n");
	//			d_print_mat(nleft, kleft, B+jj+ll*ldb, ldb);
				for(iii=0; iii<kleft-3; iii+=4)
					{
	//				d_print_mat(nleft, 4, B+jj+(ll+iii)*ldb, ldb);
					kernel_dpack_tt_4_lib4(nleft, B+jj+(ll+iii)*ldb, ldb, pB+iii*ps, sdb);
					}
				if(iii<kleft)
					{
					kernel_dpack_tt_4_vs_lib4(nleft, B+jj+(ll+iii)*ldb, ldb, pB+iii*ps, sdb, kleft-iii);
					}
	//			d_print_mat(4, kleft, pB, 4);
	//			d_print_mat(4, kleft, pB+4*sdb, 4);

				blasfeo_hp_dgemm_nt_m1(mleft, nleft, kleft, alpha, pA, sda, pB, sdb, beta1, C1+ii+jj*ldc1, ldc1, D+ii+jj*ldd, ldd);
	//			d_print_mat(m, nleft, D+jj*ldd, ldd);

				}

			}

		}

	free(mem);

	return;

#else

	k1 = (k+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	tA_size = blasfeo_pm_memsize_dmat(ps, m_kernel, k1);
	tB_size = blasfeo_pm_memsize_dmat(ps, n1, k1);
//	mem = malloc(tA_size+tB_size+64);
//	blasfeo_align_64_byte(mem, (void **) &mem_align);
	mem = malloc(tA_size+tB_size+4096);
	blasfeo_align_4096_byte(mem, (void **) &mem_align);
	blasfeo_pm_create_dmat(ps, m_kernel, k, &tA, (void *) mem_align);
	blasfeo_pm_create_dmat(ps, n, k, &tB, (void *) (mem_align+tA_size));

	sda = tA.cn;
	sdb = tB.cn;

//	blasfeo_pack_dmat(n, k, B, ldb, &tB, 0, 0);
	for(ii=0; ii<k-3; ii+=4)
		{
		kernel_dpack_tt_4_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb);
		}
	if(ii<k)
		{
		kernel_dpack_tt_4_vs_lib4(n, B+ii*ldb, ldb, tB.pA+ii*ps, sdb, k-ii);
		}

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, tA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
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
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(jj<n)
			{
			kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(ii<m)
		{
		goto tt_1_left_4;
		}
#endif
	goto tt_1_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tt_1_left_12:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+8)*lda, lda, tA.pA+8*sda, m-ii-8);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
tt_1_left_8:
	kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, tA.pA);
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+4)*lda, lda, tA.pA+4*sda, m-ii-4);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib44cc(k, &alpha, tA.pA, sda, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_1_return;
#endif

tt_1_left_4:
	kernel_dpack_tn_4_vs_lib4(k, A+(ii+0)*lda, lda, tA.pA, m-ii);
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44cc(k, &alpha, tA.pA, tB.pA+jj*sdb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_1_return;

tt_1_return:
	free(mem);
	return;

#endif



tt_2:

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x12_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x12_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tt_2_left_4;
			}
		else if(n-jj<=8)
			{
			goto tt_2_left_8;
			}
		else
			{
			goto tt_2_left_12;
			}
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; jj<n-7; jj+=8)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x8_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x8_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto tt_2_left_4;
			}
		else
			{
			goto tt_2_left_8;
			}
		}
#elif ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; jj<n-3; jj+=4)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			kernel_dgemm_tt_4x4_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd);
			}
		if(ii<m)
			{
			kernel_dgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
			}
		}
	if(jj<n)
		{
		goto tt_2_left_4;
		}
#endif
	goto tt_2_return;

#if defined(TARGET_X64_INTEL_HASWELL)
tt_2_left_12:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x12_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_2_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
tt_2_left_8:
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x8_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	goto tt_2_return;
#endif

#if ! defined(TARGET_X64_INTEL_SANDY_BRIDGE)
tt_2_left_4:
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-8; ii+=12)
		{
		kernel_dgemm_tt_12x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m-4)
		{
		kernel_dgemm_tt_8x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	else if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(ii=0; ii<m-4; ii+=8)
		{
		kernel_dgemm_tt_8x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
	if(ii<m)
		{
		kernel_dgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#else
	for(ii=0; ii<m; ii+=4)
		{
		kernel_dgemm_tt_4x4_vs_libcccc(k, &alpha, A+ii*lda, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, D+ii+jj*ldd, ldd, m-ii, n-jj);
		}
#endif
	goto tt_2_return;
#endif

tt_2_return:
	return;

	// never to get here
	return;

	}



#if defined(LA_HIGH_PERFORMANCE)
#ifndef HP_BLAS



void blasfeo_dgemm_nn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_nt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_tn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_tt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



#endif
#endif

