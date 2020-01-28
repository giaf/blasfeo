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


#include "../include/blasfeo_target.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"



#if defined(FORTRAN_BLAS_API)
#define blas_dgetrf_np dgetrf_np_
#endif



void blas_dgetrf_np(int *pm, int *pn, double *C, int *pldc, int *info)
	{

#if defined(PRINT_NAME)
	printf("\nblas_dgetrf_np %d %d %p %d %d\n", *pm, *pn, C, *pldc, *info);
#endif

//#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//#else
//	printf("\nblas_dgetrf_np: not implemented yet\n");
//	exit(1);
//#endif

	int m = *pm;
	int n = *pn;
	int ldc = *pldc;

//	d_print_mat(m, n, C, ldc);
//	printf("\nm %d n %d ldc %d\n", m, n, ldc);

	*info = 0;

	if(m<=0 | n<=0)
		return;

	int ps = 4;

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
	int sdu0 = (m+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;


	struct blasfeo_dmat sC;
	int sdu, sdc;
	double *pU, *pC, *pd;
	int sC_size, stot_size;
	void *smat_mem, *smat_mem_align;
	int m1, n1;

//	int n4 = n<4 ? n : 4;

	int p = m<n ? m : n;

	int m_max;


	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;

	int ii, jj;

	// TODO
#if defined(TARGET_X64_INTEL_HASWELL)
///	if(m>300 | n>300 | m>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
///	if(m>240 | n>240 | m>K_MAX_STACK)
#else
///	if(m>=12 | n>=12 | m>K_MAX_STACK)
#endif
///		{
///		goto alg1;
///		}
///	else
///		{
		goto alg0;
///		}



alg0:

	pU = pU0;
	sdu = sdu0;
	pd = pd0;

	jj = 0;
#if 0
// TODO other targets
#else
	for(; jj<n-3; jj+=4)
		{

		m_max = m<jj ? m : jj;

		ii = 0;

		// pack
		kernel_dpack_tn_4_lib4(m_max, C+jj*ldc, ldc, pU);

		// solve upper
		for( ; ii<m_max-3; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_4x4_lib4c44c(ii, pU, C+ii, ldc, &d1, pU+ii*ps, pU+ii*ps, C+ii+ii*ldc, ldc);
			}
		if(ii<m_max)
			{
			kernel_dtrsm_nt_rl_one_4x4_vs_lib4c44c(ii, pU, C+ii, ldc, &d1, pU+ii*ps, pU+ii*ps, C+ii+ii*ldc, ldc, 4, m_max-ii);
			ii = m_max;
			}

		// unpack
		kernel_dunpack_nt_4_lib4(m_max, pU, C+jj*ldc, ldc);

		// factorize
		if(ii<m-3)
			{
			kernel_dgetrf_nt_4x4_libc4cc(jj, C+ii, ldc, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
			ii += 4;
			}
		else if(ii<m)
			{
			kernel_dgetrf_nt_4x4_vs_libc4cc(jj, C+ii, ldc, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj, m-ii, n-jj);
			ii = m;
			}

		// solve lower
		for( ; ii<m-3; ii+=4)
			{
			kernel_dtrsm_nt_run_inv_4x4_libc4ccc(jj, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			}
		if(ii<m)
			{
			kernel_dtrsm_nt_run_inv_4x4_vs_libc4ccc(jj, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj, m-ii, n-jj);
			}

		}
	if(jj<n)
		{
		goto left_4_0;
		}
#endif
	goto end_0;



left_4_0:
	m_max = m<jj ? m : jj;

	ii = 0;

	// pack
	kernel_dpack_tn_4_lib4(m_max, C+jj*ldc, ldc, pU);

	// solve upper
	for( ; ii<m_max; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_4x4_vs_lib4c44c(ii, pU, C+ii, ldc, &d1, pU+ii*ps, pU+ii*ps, C+ii+ii*ldc, ldc, n-jj, m_max-ii);
		}

	// unpack
	kernel_dunpack_nt_4_lib4(m_max, pU, C+jj*ldc, ldc);

	// factorize
	if(ii<m)
		{
		kernel_dgetrf_nt_4x4_vs_libc4cc(jj, C+ii, ldc, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj, m-ii, n-jj);
		ii += 4;
		}

	// solve lower
	for( ; ii<m; ii+=4)
		{
		kernel_dtrsm_nt_run_inv_4x4_vs_libc4ccc(jj, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj, m-ii, n-jj);
		}

	goto end_0;



end_0:
	return;

	}
