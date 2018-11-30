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
	double pd[K_MAX_STACK];
#else
	double pd[K_MAX_STACK] __attribute__ ((aligned (64)));
#endif

#if defined(TARGET_X64_INTEL_HASWELL)
	double pU[3*4*K_MAX_STACK] __attribute__ ((aligned (64)));
	double pD[4*16] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	double pU[2*4*K_MAX_STACK] __attribute__ ((aligned (64)));
	double pD[2*16] __attribute__ ((aligned (64)));
#elif defined(TARGET_GENERIC)
	double pU[1*4*K_MAX_STACK];
	double pD[1*16];
#else
	double pU[1*4*K_MAX_STACK] __attribute__ ((aligned (64)));
	double pD[1*16] __attribute__ ((aligned (64)));
#endif
	int sdu = (m+3)/4*4;
	sdu = sdu<K_MAX_STACK ? sdu : K_MAX_STACK;


	struct blasfeo_dmat sC;
	int sdc;
	double *pc;
	int sC_size, stot_size;
	void *smat_mem, *smat_mem_align;
	int m1;


	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;

	int ii, jj;

	int arg0, arg1;



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
			kernel_dgemm_nt_4x12_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, 4);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_12_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<12; ii++)
			{
			ipiv[jj+ii] += jj; // TODO +1 !!!
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc);

		// pivot
		arg0 = n-jj-12;
		arg1 = jj+11;
		blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
		blasfeo_dlaswp(&arg0, C+(jj+12)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

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
#elif 0//defined(TARGET_X64_INTEL_SANDY_BRIDGE)
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
			kernel_dgemm_nt_4x8_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, 4);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_8_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<8; ii++)
			{
			ipiv[jj+ii] += jj; // TODO +1 !!!
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
		kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);

		// pivot
		arg0 = n-jj-8;
		arg1 = jj+7;
		blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
		blasfeo_dlaswp(&arg0, C+(jj+8)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

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
		// TODO
		ii = jj;
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgemm_nt_4x4_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		if(m-ii>0)
			{
			kernel_dgemm_nt_4x4_vs_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, 4);
			}

		// pivot & factorize & solve
		kernel_dgetrf_pivot_4_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
		for(ii=0; ii<4; ii++)
			{
			ipiv[jj+ii] += jj; // TODO +1 !!!
			}

		// unpack
		kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);

		// pivot
		arg0 = n-jj-4;
		arg1 = jj+3;
		blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
		blasfeo_dlaswp(&arg0, C+(jj+4)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

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
	for( ; ii<m-3; ii+=4)
		{
		kernel_dgemm_nt_4x12_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
	// TODO vs
	kernel_dgetrf_pivot_12_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
	for(ii=0; ii<12; ii++)
		{
		ipiv[jj+ii] += jj; // TODO +1 !!!
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+8*sdu, C+(jj+8)*ldc, ldc, n-jj-8);

	// pivot
	arg0 = n-jj-12;
	arg1 = jj+11;
	blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
	blasfeo_dlaswp(&arg0, C+(jj+12)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_0;



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
	for( ; ii<m-3; ii+=4)
		{
		kernel_dgemm_nt_4x8_vs_libc4c(jj, &dm1, C+ii, ldc, pU, sdu, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
	// TODO vs
	kernel_dgetrf_pivot_8_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
	for(ii=0; ii<8; ii++)
		{
		ipiv[jj+ii] += jj; // TODO +1 !!!
		}

	// unpack
	kernel_dunpack_nt_4_lib4(jj, pU, C+jj*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(jj, pU+4*sdu, C+(jj+4)*ldc, ldc, n-jj-4);

	// pivot
	arg0 = n-jj-8;
	arg1 = jj+7;
	blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
	blasfeo_dlaswp(&arg0, C+(jj+8)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_0;



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
	for( ; ii<m-3; ii+=4)
		{
		kernel_dgemm_nt_4x4_vs_libc4c(jj, &dm1, C+ii, ldc, pU, &d1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, n-jj);
		}

	// pivot & factorize & solve
	// TODO vs
	kernel_dgetrf_pivot_4_lib(m-jj, C+jj+jj*ldc, ldc, pd+jj, ipiv+jj);
	for(ii=0; ii<4; ii++)
		{
		ipiv[jj+ii] += jj; // TODO +1 !!!
		}

	// unpack
	kernel_dunpack_nt_4_vs_lib4(jj, pU, C+jj*ldc, ldc, n-jj);

	// pivot
	arg0 = n-jj-4;
	arg1 = jj+3;
	blasfeo_dlaswp(&jj, C, &ldc, &jj, &arg1, ipiv, &i1);
	blasfeo_dlaswp(&arg0, C+(jj+4)*ldc, &ldc, &jj, &arg1, ipiv, &i1);

	goto end_0;



end_0:
	return;

	}


