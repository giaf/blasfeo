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
#define blasfeo_dgetrf dgetrf_
#define blasfeo_dlaswp dlaswp_
#endif



void blasfeo_dgetrf(int *pm, int *pn, double *C, int *pldc, int *ipiv, int *info)
	{

	int m = *pm;
	int n = *pn;
	int ldc = *pldc;

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

	int p = m<n ? m : n;


	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;

	int ii, jj;

	int arg0, arg1, arg2;

	double *dummy = NULL;

	int ipiv_tmp[4] = {};


	// TODO
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>200 | n>200 | m>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>240 | n>240 | m>K_MAX_STACK)
#else
	if(m>=12 | n>=12 | m>K_MAX_STACK)
#endif
		{
		goto alg1;
		}
	else
		{
		goto alg0;
		}



alg0:

	pU = pU0;
	sdu = sdu0;
	pd = pd0;

	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; jj<n-11; jj+=12)
		{

		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
		kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);
		kernel_dpack_tn_4_lib4(jj, C+(jj+8)*ldc, ldc, pU+8*sdu);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_12x4_lib4c4c(ii, pU, sdu, C+ii, ldc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, C+ii+ii*ldc, ldc);
			}

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x12_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x12_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_12_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<12; ii++)
			{
			ipiv[jj+ii] += jj;
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc);

		// pivot
		for(ii=0; ii<12; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				kernel_drowsw_lib(jj, C+jj+ii, ldc, C+ipiv[jj+ii], ldc);
				kernel_drowsw_lib(n-jj-12, C+jj+ii+(jj+12)*ldc, ldc, C+ipiv[jj+ii]+(jj+12)*ldc, ldc);
				}
			}

		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto left_4_0;
			}
		if(n-jj<=8)
			{
			goto left_8_0;
			}
		else
			{
			goto left_12_0;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; jj<n-7; jj+=8)
		{

		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
		kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_8x4_lib4c4c(ii, pU, sdu, C+ii, ldc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, C+ii+ii*ldc, ldc);
			}

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x8_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x8_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_8_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<8; ii++)
			{
			ipiv[jj+ii] += jj;
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);

		// pivot
		for(ii=0; ii<8; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				kernel_drowsw_lib(jj, C+jj+ii, ldc, C+ipiv[jj+ii], ldc);
				kernel_drowsw_lib(n-jj-8, C+jj+ii+(jj+8)*ldc, ldc, C+ipiv[jj+ii]+(jj+8)*ldc, ldc);
				}
			}

		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto left_4_0;
			}
		else
			{
			goto left_8_0;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{

		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_4x4_lib4c4c(ii, pU, C+ii, ldc, &d1, pU+ii*ps, pU+ii*ps, C+ii+ii*ldc, ldc);
			}

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x4_vs_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_4_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<4; ii++)
			{
			ipiv[jj+ii] += jj;
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);

		// apply pivot
		for(ii=0; ii<4; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				kernel_drowsw_lib(jj, C+jj+ii, ldc, C+ipiv[jj+ii], ldc);
				kernel_drowsw_lib(n-jj-4, C+jj+ii+(jj+4)*ldc, ldc, C+ipiv[jj+ii]+(jj+4)*ldc, ldc);
				}
			}

		}
	if(jj<n)
		{
		goto left_4_0;
		}
#endif
	goto end_0;



left_12_0:

	// pack
	kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
	kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(jj, C+(jj+8)*ldc, ldc, pU+8*sdu, n-jj-8);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_12x4_lib4c4c(ii, pU, sdu, C+ii, ldc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, C+ii+ii*ldc, ldc);
		}

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
#if 0
	kernel_dgetrf_pivot_12_vs_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
#else
	kernel_dgetrf_pivot_8_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);

	// apply pivot to right column
	for(ii=0; ii<8; ii++)
		{
		if(ipiv[jj+ii]+jj!=jj+ii)
			{
			kernel_drowsw_lib(n-jj-8, C+jj+ii+(jj+8)*ldc, ldc, C+ipiv[jj+ii]+jj+(jj+8)*ldc, ldc);
			}
		}

	// pack
	kernel_dpack_tn_4_vs_lib4(8, C+jj+(jj+8)*ldc, ldc, pU+8*sdu+jj*ps, n-jj-8);

	// solve top right block
	kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(0, pU+8*sdu+jj*ps, C+jj+jj*ldc, ldc, &d1, pU+8*sdu+jj*ps, pU+8*sdu+jj*ps, C+jj+jj*ldc, ldc, n-jj-8, m-jj);
	kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(4, pU+8*sdu+jj*ps, C+jj+4+jj*ldc, ldc, &d1, pU+8*sdu+(jj+4)*ps, pU+8*sdu+(jj+4)*ps, C+(jj+4)+(jj+4)*ldc, ldc, n-jj-8, m-jj-4);

	// unpack
	kernel_dunpack_nt_4_vs_lib4(8, pU+8*sdu+jj*ps, C+jj+(jj+8)*ldc, ldc, n-jj-8);

	// correct rigth block
	ii = 8;
	// TODO larger kernels ???
	for(; ii<m-jj; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4c(8, &dm1, C+jj+ii+jj*ldc, ldc, pU+8*sdu+jj*ps, &d1, C+jj+ii+(jj+8)*ldc, ldc, C+jj+ii+(jj+8)*ldc, ldc, m-ii-jj, n-jj-8);
		}

	// fact right column
	kernel_dgetrf_pivot_4_vs_lib(m-(jj+8), C+(jj+8)+(jj+8)*ldc, ldc, pd+(jj+8), ipiv+(jj+8), n-jj-8);
	for(ii=0; ii<n-jj-8; ii++)
		ipiv[jj+8+ii] += 8;

	// apply pivot to left column
	// TODO fix too !!!
	for(ii=8; ii<n-jj; ii++)
		{
		if(ipiv[jj+ii]+jj!=jj+ii)
			{
			kernel_drowsw_lib(8, C+jj+ii+jj*ldc, ldc, C+ipiv[jj+ii]+jj+jj*ldc, ldc);
			}
		}
#endif

	for(ii=0; ii<n-jj; ii++)
		{
		ipiv[jj+ii] += jj;
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc, n-jj-8);

	// pivot
	for(ii=0; ii<n-jj; ii++)
		{
		if(ipiv[jj+ii]!=jj+ii)
			{
			kernel_drowsw_lib(jj, C+jj+ii, ldc, C+ipiv[jj+ii], ldc);
//			kernel_drowsw_lib(n-jj-12, C+jj+ii+(jj+12)*ldc, ldc, C+ipiv[jj+ii]+(jj+12)*ldc, ldc);
			}
		}

	goto end_0;



#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
left_8_0:

	// pack
	kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
	kernel_dpack_tn_4_vs_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu, n-jj-4);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_8x4_lib4c4c(ii, pU, sdu, C+ii, ldc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, C+ii+ii*ldc, ldc);
		}

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x8_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
#if 0
	kernel_dgetrf_pivot_8_vs_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj, n-jj);
#else
	// TODO if haswell and n-jj==8 ...

	kernel_dgetrf_pivot_4_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);

	// apply pivot to right column
	for(ii=0; ii<4; ii++)
		{
		if(ipiv[jj+ii]+jj!=jj+ii)
			{
			kernel_drowsw_lib(n-jj-4, C+jj+ii+(jj+4)*ldc, ldc, C+ipiv[jj+ii]+jj+(jj+4)*ldc, ldc);
			}
		}

	// pack
	kernel_dpack_tn_4_vs_lib4(4, C+jj+(jj+4)*ldc, ldc, pU+4*sdu+jj*ps, n-jj-4);

	// solve top right block
	kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(0, pU+4*sdu+jj*ps, C+jj, ldc, &d1, pU+4*sdu+jj*ps, pU+4*sdu+jj*ps, C+jj+jj*ldc, ldc, n-jj-4, m-jj);

	// unpack
	kernel_dunpack_nt_4_vs_lib4(4, pU+4*sdu+jj*ps, C+jj+(jj+4)*ldc, ldc, n-jj-4);

	// correct rigth block
	ii = 4;
	// TODO larger kernels ???
	for(; ii<m-jj; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4c(4, &dm1, C+jj+ii+jj*ldc, ldc, pU+4*sdu+jj*ps, &d1, C+jj+ii+(jj+4)*ldc, ldc, C+jj+ii+(jj+4)*ldc, ldc, m-ii-jj, n-jj-4);
		}

	// fact right column
	kernel_dgetrf_pivot_4_vs_lib(m-(jj+4), C+(jj+4)+(jj+4)*ldc, ldc, pd+(jj+4), ipiv+(jj+4), n-jj-4);
//	for(ii=4; ii<n-jj; ii++)
	for(ii=4; ii<8; ii++)
		ipiv[jj+ii] += 4;

	// apply pivot to left column
//	for(ii=4; ii<n-jj; ii++)
	for(ii=4; ii<8; ii++)
		{
		if(ipiv[jj+ii]+jj!=jj+ii)
			{
			kernel_drowsw_lib(4, C+jj+ii+jj*ldc, ldc, C+ipiv[jj+ii]+jj+jj*ldc, ldc);
			}
		}
#endif

	for(ii=jj; ii<n; ii++)
		{
		ipiv[ii] += jj;
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc, n-jj-4);

	// pivot
	for(ii=0; ii<n-jj; ii++)
		{
		if(ipiv[jj+ii]!=jj+ii)
			{
			kernel_drowsw_lib(jj, C+jj+ii, ldc, C+ipiv[jj+ii], ldc);
//			kernel_drowsw_lib(n-jj-8, C+jj+ii+(jj+8)*ldc, ldc, C+ipiv[jj+ii]+(jj+8)*ldc, ldc);
			}
		}


	goto end_0;
#endif



left_4_0:

	// pack
	kernel_dpack_tn_4_vs_lib4(jj, C+jj*ldc, ldc, pU, n-jj);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_4x4_lib4c4c(ii, pU, C+ii, ldc, &d1, pU+ii*ps, pU+ii*ps, C+ii+ii*ldc, ldc);
		}

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
	kernel_dgetrf_pivot_4_vs_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj, n-jj);
	for(ii=jj; ii<n; ii++)
		{
		ipiv[ii] += jj;
		}

	// unpack
	kernel_dunpack_nt_4_vs_lib4(jj, pU, C+jj*ldc, ldc, n-jj);

	// pivot
	arg0 = n-jj-4;
	arg1 = n-1;
	blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
//	blasfeo_dlaswp(&arg0, C+(jj+4)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_0;



end_0:
	// from 0-index to 1-index
	for(ii=0; ii<p; ii++)
		ipiv[ii] += 1;
	return;



alg1:

	m1 = (m+128-1)/128*128;
	n1 = (n+128-1)/128*128;
	sC_size = blasfeo_memsize_dmat(m1, n1) + 12*m1*sizeof(double);
//	sC_size = blasfeo_memsize_dmat(m, m);
	stot_size = sC_size;
	smat_mem = malloc(stot_size+64);
	blasfeo_align_64_byte(smat_mem, &smat_mem_align);
	pU = smat_mem_align;
	sdu = (m+3)/4*4;
	blasfeo_create_dmat(m, n, &sC, smat_mem_align+12*m1*sizeof(double));
	pC = sC.pA;
	sdc = sC.cn;
	pd = sC.dA;



	jj = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; jj<n-11; jj+=12)
		{
		
		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
		kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);
		kernel_dpack_tn_4_lib4(jj, C+(jj+8)*ldc, ldc, pU+8*sdu);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_12x4_lib4(ii, pU, sdu, pC+ii*sdc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, pC+ii*sdc+ii*ps);
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc);

		// pack
		kernel_dpack_tt_12_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc);

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x12_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x12_vs_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
			}

		// pivot & factorize & solve
#if defined(TARGET_X64_INTEL_HASWELL)
		kernel_dgetrf_pivot_12_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);
#else

		// fact left column
//		kernel_dgetrf_pivot_8_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);
		// fact left column
		kernel_dgetrf_pivot_4_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);

		// apply pivot to right column
		for(ii=jj; ii<jj+4; ii++)
			{
			if(ipiv[ii]!=ii)
				{
				blasfeo_drowsw(4, &sC, ii, jj+4, &sC, ipiv[ii]+jj, jj+4);
				}
			}

		// solve top right block
		kernel_dtrsm_nn_ll_one_4x4_vs_lib4(0, dummy, dummy, 0, &d1, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+jj*ps, m-jj, n-jj-4);

		// correct rigth block
		ii = 4;
		for(; ii<m-jj-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc);
			}
		if(ii<m-jj)
			{
			if(m-jj-ii<=4)
				{
				kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, pC+(jj+ii)*sdc+(jj+4)*ps, m-jj-ii, n-jj);
				}
			else if(m-jj-ii<=8)
				{
				kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
				}
			else //if(m-jj-ii<=12)
				{
				kernel_dgemm_nn_12x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
				}
			}

		// fact right column
		kernel_dgetrf_pivot_4_vs_lib4(m-jj-4, pC+(jj+4)*sdc+(jj+4)*ps, sdc, pd+jj+4, ipiv+jj+4, n-jj-4);

//		for(ii=jj+4; ii<n; ii++)
		for(ii=jj+4; ii<jj+8; ii++)
			ipiv[ii] += 4;

		// apply pivot to left column
//		for(ii=jj+4; ii<n; ii++)
		for(ii=jj+4; ii<jj+8; ii++)
			{
			if(ipiv[ii]!=ii)
				{
				blasfeo_drowsw(4, &sC, ii, jj, &sC, ipiv[ii]+jj, jj);
				}
			}

		// apply pivot to right column
		for(ii=jj; ii<jj+8; ii++)
			{
			if(ipiv[ii]!=ii)
				{
				blasfeo_drowsw(4, &sC, ii, jj+8, &sC, ipiv[ii]+jj, jj+8);
				}
			}

		// solve top right block
		kernel_dtrsm_nn_ll_one_8x4_vs_lib4(0, dummy, 0, dummy, 0, &d1, pC+jj*sdc+(jj+8)*ps, sdc, pC+jj*sdc+(jj+8)*ps, sdc, pC+jj*sdc+jj*ps, sdc, m-jj, n-jj-8);

		// correct rigth block
		ii = 8;
		for(; ii<m-jj-11; ii+=12)
			{
			kernel_dgemm_nn_12x4_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc);
			}
		if(ii<m-jj)
			{
			if(m-jj-ii<=4)
				{
				kernel_dgemm_nn_4x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, pC+(jj+ii)*sdc+(jj+8)*ps, m-jj-ii, n-jj);
				}
			else if(m-jj-ii<=8)
				{
				kernel_dgemm_nn_8x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, m-jj-ii, n-jj);
				}
			else //if(m-jj-ii<=12)
				{
				kernel_dgemm_nn_12x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, m-jj-ii, n-jj);
				}
			}

		// fact right column
		kernel_dgetrf_pivot_4_vs_lib4(m-jj-8, pC+(jj+8)*sdc+(jj+8)*ps, sdc, pd+jj+8, ipiv+jj+8, n-jj-8);

//		for(ii=jj+8; ii<n; ii++)
		for(ii=jj+8; ii<jj+12; ii++)
			ipiv[ii] += 8;

		// apply pivot to left column
//		for(ii=jj+8; ii<n; ii++)
		for(ii=jj+8; ii<jj+12; ii++)
			{
			if(ipiv[ii]!=ii)
				{
				blasfeo_drowsw(8, &sC, ii, jj, &sC, ipiv[ii]+jj, jj);
				}
			}

#endif

		for(ii=0; ii<12; ii++)
			{
			ipiv[jj+ii] += jj;
			}
		// unpack
		kernel_dunpack_nn_12_lib4(12, pC+jj*sdc+jj*ps, sdc, C+jj+jj*ldc, ldc);

		// pivot
		for(ii=0; ii<12; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				blasfeo_drowsw(jj, &sC, jj+ii, 0, &sC, ipiv[jj+ii], 0);
				}
			}
		arg0 = n-jj-12;
		arg1 = jj+11;
		blasfeo_dlaswp(&arg0, C+(jj+12)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto left_4_1;
			}
		else if(n-jj<=8)
			{
			goto left_8_1;
			}
		else
			{
			goto left_12_1;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; jj<n-7; jj+=8)
		{
		
		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
		kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_8x4_lib4(ii, pU, sdu, pC+ii*sdc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, pC+ii*sdc+ii*ps);
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);

		// pack
		kernel_dpack_tt_8_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc);

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x8_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x8_vs_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
			}

		// pivot & factorize & solve
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		kernel_dgetrf_pivot_8_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);
#else

		// fact left column
		kernel_dgetrf_pivot_4_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);

		// apply pivot to right column
		for(ii=0; ii<4; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				blasfeo_drowsw(4, &sC, jj+ii, jj+4, &sC, ipiv[jj+ii]+jj, jj+4);
				}
			}

		// solve top right block
		kernel_dtrsm_nn_ll_one_4x4_vs_lib4(0, dummy, dummy, 0, &d1, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+jj*ps, m-jj, n-jj-4);

		// correct rigth block
		ii = 4;
		for(; ii<m-jj-7; ii+=8)
			{
			kernel_dgemm_nn_8x4_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc);
			}
		if(ii<m-jj)
			{
			if(m-jj-ii<=4)
				{
				kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, pC+(jj+ii)*sdc+(jj+4)*ps, m-jj-ii, n-jj);
				}
			else //if(m-jj-ii<=8)
				{
				kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
				}
			}

		// fact right column
		kernel_dgetrf_pivot_4_lib4(m-jj-4, pC+(jj+4)*sdc+(jj+4)*ps, sdc, pd+jj+4, ipiv+jj+4);

//		for(ii=4; ii<n-jj; ii++)
		for(ii=4; ii<8; ii++)
			ipiv[jj+ii] += 4;

		// apply pivot to left column
//		for(ii=4; ii<n-jj; ii++)
		for(ii=4; ii<8; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				blasfeo_drowsw(4, &sC, jj+ii, jj, &sC, ipiv[jj+ii]+jj, jj);
				}
			}
#endif

		for(ii=0; ii<8; ii++)
			{
			ipiv[jj+ii] += jj;
			}
		// unpack
		kernel_dunpack_nn_8_lib4(8, pC+jj*sdc+jj*ps, sdc, C+jj+jj*ldc, ldc);


		// pivot
		for(ii=0; ii<8; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				blasfeo_drowsw(jj, &sC, jj+ii, 0, &sC, ipiv[jj+ii], 0);
				}
			}
		arg0 = n-jj-8;
		arg1 = jj+7;
		blasfeo_dlaswp(&arg0, C+(jj+8)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

		}
	if(jj<n)
		{
		if(n-jj<=4)
			{
			goto left_4_1;
			}
		else if(n-jj<=8)
			{
			goto left_8_1;
			}
		}
#else
	for(; jj<n-3; jj+=4)
		{

		// pack
		kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);

		// solve upper
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrsm_nt_rl_one_4x4_lib4(ii, pU, pC+ii*sdc, &d1, pU+ii*ps, pU+ii*ps, pC+ii*sdc+ii*ps);
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);

		// pack
		kernel_dpack_tt_4_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc);

		// correct
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_lib4(jj, &dm1, pC+ii*sdc, pU, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x4_vs_lib4(jj, &dm1, pC+ii*sdc, pU, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_4_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);
		for(ii=0; ii<4; ii++)
			{
			ipiv[jj+ii] += jj;
			}
		// unpack
		kernel_dunpack_nn_4_lib4(4, pC+jj*sdc+jj*ps, C+jj+jj*ldc, ldc);

		// pivot
		for(ii=0; ii<4; ii++)
			{
			if(ipiv[jj+ii]!=jj+ii)
				{
				blasfeo_drowsw(jj, &sC, jj+ii, 0, &sC, ipiv[jj+ii], 0);
				}
			}
		arg0 = n-jj-4;
		arg1 = jj+3;
		blasfeo_dlaswp(&arg0, C+(jj+4)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

		}
	if(jj<n)
		{
		goto left_4_1;
		}
#endif
	goto end_1;



#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
left_12_1:

	// pack
	kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
	kernel_dpack_tn_4_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(jj, C+(jj+8)*ldc, ldc, pU+8*sdu, n-jj);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_12x4_lib4(ii, pU, sdu, pC+ii*sdc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, pC+ii*sdc+ii*ps);
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc, n-jj);

	// pack
	kernel_dpack_tt_8_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc);
	kernel_dpack_tt_4_vs_lib4(m-jj, C+jj+(jj+8)*ldc, ldc, pC+jj*sdc+(jj+8)*ps, sdc, n-jj-8);

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
		}

	// pivot & factorize & solve
#if 0
	kernel_dgetrf_pivot_12_vs_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj, n-jj);
#else
	// fact left column
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgetrf_pivot_8_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);
#else

	// fact left column
	kernel_dgetrf_pivot_4_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);

	// apply pivot to right column
	for(ii=jj; ii<jj+4; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(4, &sC, ii, jj+4, &sC, ipiv[ii]+jj, jj+4);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(0, dummy, dummy, 0, &d1, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+jj*ps, m-jj, n-jj-4);

	// correct rigth block
	ii = 4;
	for(; ii<m-jj-11; ii+=12)
		{
		kernel_dgemm_nn_12x4_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc);
		}
	if(ii<m-jj)
		{
		if(m-jj-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, pC+(jj+ii)*sdc+(jj+4)*ps, m-jj-ii, n-jj);
			}
		else if(m-jj-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
			}
		else //if(m-jj-ii<=12)
			{
			kernel_dgemm_nn_12x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
			}
		}

	// fact right column
	kernel_dgetrf_pivot_4_vs_lib4(m-jj-4, pC+(jj+4)*sdc+(jj+4)*ps, sdc, pd+jj+4, ipiv+jj+4, n-jj-4);

//	for(ii=jj+4; ii<n; ii++)
	for(ii=jj+4; ii<jj+8; ii++)
		ipiv[ii] += 4;

	// apply pivot to left column
//	for(ii=jj+4; ii<n; ii++)
	for(ii=jj+4; ii<jj+8; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(4, &sC, ii, jj, &sC, ipiv[ii]+jj, jj);
			}
		}

#endif

	// apply pivot to right column
	for(ii=jj; ii<jj+8; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(4, &sC, ii, jj+8, &sC, ipiv[ii]+jj, jj+8);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_8x4_vs_lib4(0, dummy, 0, dummy, 0, &d1, pC+jj*sdc+(jj+8)*ps, sdc, pC+jj*sdc+(jj+8)*ps, sdc, pC+jj*sdc+jj*ps, sdc, m-jj, n-jj-8);

	// correct rigth block
	ii = 8;
	for(; ii<m-jj-11; ii+=12)
		{
		kernel_dgemm_nn_12x4_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc);
		}
	if(ii<m-jj)
		{
		if(m-jj-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, pC+(jj+ii)*sdc+(jj+8)*ps, m-jj-ii, n-jj);
			}
		else if(m-jj-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, m-jj-ii, n-jj);
			}
		else //if(m-jj-ii<=12)
			{
			kernel_dgemm_nn_12x4_vs_lib4(8, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+8)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, pC+(jj+ii)*sdc+(jj+8)*ps, sdc, m-jj-ii, n-jj);
			}
		}

	// fact right column
	kernel_dgetrf_pivot_4_vs_lib4(m-jj-8, pC+(jj+8)*sdc+(jj+8)*ps, sdc, pd+jj+8, ipiv+jj+8, n-jj-8);

	for(ii=jj+8; ii<n; ii++)
		ipiv[ii] += 8;

	// apply pivot to left column
	for(ii=jj+8; ii<n; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(8, &sC, ii, jj, &sC, ipiv[ii]+jj, jj);
			}
		}
#endif

	for(ii=0; ii<n-jj; ii++)
		{
		ipiv[jj+ii] += jj;
		}
	// unpack
	kernel_dunpack_nn_12_vs_lib4(n-jj, pC+jj*sdc+jj*ps, sdc, C+jj+jj*ldc, ldc, m-jj);

	// pivot
	for(ii=jj; ii<n; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(jj, &sC, ii, 0, &sC, ipiv[ii], 0);
			}
		}
//	arg0 = n-jj-12;
//	arg1 = jj+11;
//	blasfeo_dlaswp(&arg0, C+(jj+12)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_1;
#endif



#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
left_8_1:

	// pack
	kernel_dpack_tn_4_lib4(jj, C+jj*ldc, ldc, pU);
	kernel_dpack_tn_4_vs_lib4(jj, C+(jj+4)*ldc, ldc, pU+4*sdu, n-jj);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_8x4_lib4(ii, pU, sdu, pC+ii*sdc, &d1, pU+ii*ps, sdu, pU+ii*ps, sdu, pC+ii*sdc+ii*ps);
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc, n-jj);

	// pack
	kernel_dpack_tt_4_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc);
	kernel_dpack_tt_4_vs_lib4(m-jj, C+jj+(jj+4)*ldc, ldc, pC+jj*sdc+(jj+4)*ps, sdc, n-jj-4);

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x8_vs_lib4(jj, &dm1, pC+ii*sdc, pU, sdu, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
		}

	// pivot & factorize & solve
#if 0
	kernel_dgetrf_pivot_8_vs_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj, n-jj);
#else
	// TODO if haswell and n-jj==8 ...

	// fact left column
	kernel_dgetrf_pivot_4_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj);

	// apply pivot to right column
	for(ii=jj; ii<jj+4; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(4, &sC, ii, jj+4, &sC, ipiv[ii]+jj, jj+4);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(0, dummy, dummy, 0, &d1, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+(jj+4)*ps, pC+jj*sdc+jj*ps, m-jj, n-jj-4);

	// correct rigth block
	ii = 4;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-jj-11; ii+=12)
		{
		kernel_dgemm_nn_12x4_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc);
		}
	if(ii<m-jj)
		{
		if(m-jj-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, pC+(jj+ii)*sdc+(jj+4)*ps, m-jj-ii, n-jj);
			}
		else if(m-jj-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
			}
		else //if(m-jj-ii<=12)
			{
			kernel_dgemm_nn_12x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
			}
		}
#else
	for(; ii<m-jj-7; ii+=8)
		{
		kernel_dgemm_nn_8x4_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc);
		}
	if(ii<m-jj)
		{
		if(m-jj-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, pC+(jj+ii)*sdc+(jj+4)*ps, m-jj-ii, n-jj);
			}
		else //if(m-jj-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+(jj+ii)*sdc+jj*ps, sdc, 0, pC+jj*sdc+(jj+4)*ps, sdc, &d1, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, pC+(jj+ii)*sdc+(jj+4)*ps, sdc, m-jj-ii, n-jj);
			}
		}
#endif

	// fact right column
	kernel_dgetrf_pivot_4_vs_lib4(m-jj-4, pC+(jj+4)*sdc+(jj+4)*ps, sdc, pd+jj+4, ipiv+jj+4, n-jj-4);

	for(ii=jj+4; ii<n; ii++)
		ipiv[ii] += 4;

	// apply pivot to left column
	for(ii=jj+4; ii<n; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(4, &sC, ii, jj, &sC, ipiv[ii]+jj, jj);
			}
		}
#endif

	for(ii=0; ii<n-jj; ii++)
		{
		ipiv[jj+ii] += jj;
		}

	// unpack
	kernel_dunpack_nn_8_vs_lib4(n-jj, pC+jj*sdc+jj*ps, sdc, C+jj+jj*ldc, ldc, m-jj);

	// pivot
	for(ii=jj; ii<n; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(jj, &sC, ii, 0, &sC, ipiv[ii], 0);
			}
		}
//	arg0 = n-jj-8;
//	arg1 = n-1;
//	blasfeo_dlaswp(&arg0, C+(jj+8)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_1;
#endif



left_4_1:

	// pack
	kernel_dpack_tn_4_vs_lib4(jj, C+jj*ldc, ldc, pU, n-jj);

	// solve upper
	for(ii=0; ii<jj; ii+=4)
		{
		kernel_dtrsm_nt_rl_one_4x4_lib4(ii, pU, pC+ii*sdc, &d1, pU+ii*ps, pU+ii*ps, pC+ii*sdc+ii*ps);
		}

	// unpack
	kernel_dunpack_nt_4_vs_lib4(jj, pU, C+jj*ldc, ldc, n-jj);

	// pack
	kernel_dpack_tt_4_vs_lib4(m-jj, C+jj+jj*ldc, ldc, pC+jj*sdc+jj*ps, sdc, n-jj);

	// correct
	ii = jj;
	for( ; ii<m; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib4(jj, &dm1, pC+ii*sdc, pU, &d1, pC+jj*ps+ii*sdc, pC+jj*ps+ii*sdc, m-ii, n-jj);
		}

	// pivot & factorize & solve
	kernel_dgetrf_pivot_4_vs_lib4(m-jj, pC+jj*sdc+jj*ps, sdc, pd+jj, ipiv+jj, n-jj);
	for(ii=jj; ii<n; ii++)
		{
		ipiv[ii] += jj;
		}
	// unpack
	kernel_dunpack_nn_4_vs_lib4(n-jj, pC+jj*sdc+jj*ps, C+jj+jj*ldc, ldc, m-jj);

	// pivot
	for(ii=jj; ii<n; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			blasfeo_drowsw(jj, &sC, ii, 0, &sC, ipiv[ii], 0);
			}
		}
//	arg0 = n-jj-4;
//	arg1 = n-1;
//	blasfeo_dlaswp(&arg0, C+(jj+4)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_1;



end_1:
	// unpack
	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
#if 1
	for(; ii<m-11 & ii<n-11; ii+=12)
		{
		kernel_dunpack_nn_12_lib4(ii, pC+ii*sdc, sdc, C+ii, ldc);
		}
	for(; ii<m-11; ii+=12)
		{
		kernel_dunpack_nn_12_lib4(n, pC+ii*sdc, sdc, C+ii, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			kernel_dunpack_nn_4_vs_lib4(n, pC+ii*sdc, C+ii, ldc, m-ii);
			}
		else if(m-ii<=8)
			{
			kernel_dunpack_nn_8_vs_lib4(n, pC+ii*sdc, sdc, C+ii, ldc, m-ii);
			}
		else //if(m-ii<=12)
			{
			kernel_dunpack_nn_12_vs_lib4(n, pC+ii*sdc, sdc, C+ii, ldc, m-ii);
			}
		}
#else
	for(; ii<n-11; ii+=12)
		{
		kernel_dunpack_tt_4_lib4(m-ii-12, pC+(ii+12)*sdc+ii*ps, sdc, C+(ii+12)+ii*ldc, ldc);
		kernel_dunpack_tt_4_lib4(m-ii-12, pC+(ii+12)*sdc+(ii+4)*ps, sdc, C+(ii+12)+(ii+4)*ldc, ldc);
		kernel_dunpack_tt_4_lib4(m-ii-12, pC+(ii+12)*sdc+(ii+8)*ps, sdc, C+(ii+12)+(ii+8)*ldc, ldc);
		}
	// TODO
#endif
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	for(; ii<m-7 & ii<n-7; ii+=8)
		{
		kernel_dunpack_nn_8_lib4(ii, pC+ii*sdc, sdc, C+ii, ldc);
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dunpack_nn_8_lib4(n, pC+ii*sdc, sdc, C+ii, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			kernel_dunpack_nn_4_vs_lib4(n, pC+ii*sdc, C+ii, ldc, m-ii);
			}
		else //if(m-ii<=8)
			{
			kernel_dunpack_nn_8_vs_lib4(n, pC+ii*sdc, sdc, C+ii, ldc, m-ii);
			}
		}
#else
	for(; ii<m-3, ii<n-3; ii+=4)
		{
		kernel_dunpack_nn_4_lib4(ii, pC+ii*sdc, C+ii, ldc);
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dunpack_nn_4_lib4(n, pC+ii*sdc, C+ii, ldc);
		}
	if(ii<m)
		{
		kernel_dunpack_nn_4_vs_lib4(n, pC+ii*sdc, C+ii, ldc, m-ii);
		}
#endif
	// TODO clean loops

	free(smat_mem);
	// from 0-index to 1-index
	for(ii=0; ii<p; ii++)
		ipiv[ii] += 1;
	return;

	}


