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
#define blasfeo_dpotrf dpotrf_
#endif



void blasfeo_dpotrf(char *uplo, int *pm, double *C, int *pldc, int *info)
	{

	int m = *pm;
	int ldc = *pldc;

	if(m<=0)
		return;

	*info = 0;

	int ii, jj;

	int bs = 4;

	double d_1 = 1.0;

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


	if(*uplo=='l' | *uplo=='L')
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		if(m>=128 | m>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		if(m>=64 | m>K_MAX_STACK)
#else
		if(m>=12 | m>K_MAX_STACK)
#endif
			{
			goto l_1;
			}
		else
			{
			goto l_0;
			}
		}
	else if(*uplo=='u' | *uplo=='U')
		{
#if defined(TARGET_X64_INTEL_HASWELL)
		if(m>=256 | m>K_MAX_STACK)
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		if(m>=64 | m>K_MAX_STACK)
#else
		if(m>=12 | m>K_MAX_STACK)
#endif
			{
			goto u_1;
			}
		else
			{
			goto u_0;
			}
		}
	else
		{
		printf("\nBLASFEO: dpotrf: wrong value for uplo\n");
		return;
		}


l_0:

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4ccc(jj, pU, sdu, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_12_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_12x4_lib44c(jj, pU, sdu, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_8_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs, sdu);
		kernel_dpotrf_nt_l_8x8_lib44c(jj+4, pU+4*sdu, sdu, pU+4*sdu, sdu, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto l_0_left_4;
			}
		if(m-ii<=8)
			{
			goto l_0_left_8;
			}
		else
			{
			goto l_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4ccc(jj, pU, sdu, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_8_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_8x4_lib44c(jj, pU, sdu, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_4_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs);
		kernel_dpotrf_nt_l_4x4_lib44c(jj+4, pU+4*sdu, pU+4*sdu, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto l_0_left_4;
			}
		else
			{
			goto l_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(jj, pU, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_4_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs);
			}
		kernel_dpotrf_nt_l_4x4_lib44c(jj, pU, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		}
	if(ii<m)
		{
		goto l_0_left_4;
		}
#endif
	goto l_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
l_0_left_12:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(jj, pU, sdu, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj, m-ii, ii-jj);
		kernel_dpack_nn_12_vs_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu, m-ii);
		}
	kernel_dpotrf_nt_l_12x4_vs_lib44c(jj, pU, sdu, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj, m-ii, m-jj);
	kernel_dpack_nn_8_vs_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs, sdu, m-ii-4);
	kernel_dpotrf_nt_l_8x8_vs_lib44c(jj+4, pU+4*sdu, sdu, pU+4*sdu, sdu, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4, m-ii-4, m-ii-4);
	goto l_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
l_0_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(jj, pU, sdu, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj, m-ii, ii-jj);
		kernel_dpack_nn_8_vs_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu, m-ii);
		}
	kernel_dpotrf_nt_l_8x4_vs_lib44c(jj, pU, sdu, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj, m-ii, m-jj);
	kernel_dpack_nn_4_vs_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs, m-ii-4);
	kernel_dpotrf_nt_l_4x4_vs_lib44c(jj+4, pU+4*sdu, pU+4*sdu, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4, m-ii-4, m-jj-4);
	goto l_0_return;
#endif

l_0_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(jj, pU, C+jj, ldc, &d_1, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj, m-ii, ii-jj);
		kernel_dpack_nn_4_vs_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, m-ii);
		}
	kernel_dpotrf_nt_l_4x4_vs_lib44c(jj, pU, pU, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj, m-ii, m-jj);
	goto l_0_return;

l_0_return:
	for(ii=0; ii<m; ii++)
		{
		if(pd[ii]==0.0)
			{
			*info = ii+1;
			return;
			}
		}
	return;



l_1:
	
	m1 = (m+128-1)/128*128;
	sC_size = blasfeo_memsize_dmat(m1, m1);
//	sC_size = blasfeo_memsize_dmat(m, m);
	stot_size = sC_size;
	smat_mem = malloc(stot_size+63);
	blasfeo_align_64_byte(smat_mem, &smat_mem_align);
	blasfeo_create_dmat(m, m, &sC, smat_mem_align);
	sdc = sC.cn;
	pc = sC.dA;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib44cc(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_12_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc);
			}
		kernel_dpotrf_nt_l_12x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		kernel_dpack_nn_8_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs, sdc);
		kernel_dpotrf_nt_l_8x8_lib44c(jj+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(jj+4)*sdc, sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4);
		kernel_dpack_nn_4_lib4(4, C+ii+8+(jj+4)*ldc, ldc, sC.pA+(ii+8)*sdc+(jj+4)*bs);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto l_1_left_4;
			}
		if(m-ii<=8)
			{
			goto l_1_left_8;
			}
		else
			{
			goto l_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib44cc(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_8_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc);
			}
		kernel_dpotrf_nt_l_8x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		kernel_dpack_nn_4_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs);
		kernel_dpotrf_nt_l_4x4_lib44c(jj+4, sC.pA+(ii+4)*sdc, sC.pA+(jj+4)*sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto l_1_left_4;
			}
		else
			{
			goto l_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib44cc(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_4_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs);
			}
		kernel_dpotrf_nt_l_4x4_lib44c(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		}
	if(ii<m)
		{
		goto l_1_left_4;
		}
#endif
	goto l_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
l_1_left_12:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib44cc(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj, m-ii, ii-jj);
		kernel_dpack_nn_12_vs_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc, m-ii);
		}
	kernel_dpotrf_nt_l_12x4_vs_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj, m-ii, m-jj);
	kernel_dpack_nn_8_vs_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs, sdc, m-ii-4);
	kernel_dpotrf_nt_l_8x8_vs_lib44c(jj+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(jj+4)*sdc, sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4, m-ii-4, m-jj-4);
	goto l_1_return;
#endif


#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
l_1_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib44cc(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj, m-ii, ii-jj);
		kernel_dpack_nn_8_vs_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc, m-ii);
		}
	kernel_dpotrf_nt_l_8x4_vs_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj, m-ii, m-jj);
	kernel_dpack_nn_4_vs_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs, m-ii-4);
	kernel_dpotrf_nt_l_4x4_vs_lib44c(jj+4, sC.pA+(ii+4)*sdc, sC.pA+(jj+4)*sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4, m-ii-4, m-jj-4);
	goto l_1_return;
#endif

l_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj, m-ii, ii-jj);
		kernel_dpack_nn_4_vs_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, m-ii);
		}
	kernel_dpotrf_nt_l_4x4_vs_lib44c(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj, m-ii, m-jj);
	goto l_1_return;

l_1_return:
	for(ii=0; ii<m; ii++)
		{
		if(pc[ii]==0.0)
			{
			*info = ii+1;
			free(smat_mem);
			return;
			}
		}
	free(smat_mem);
	return;



u_0:

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, pU+jj*bs+0*sdu);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, pU+jj*bs+4*sdu);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+8)*ldc, ldc, pU+jj*bs+8*sdu);

			kernel_dtrsm_nn_ru_inv_12x4_lib4c4c(jj, pU, sdu, C+jj*ldc, ldc, pU+jj*bs, sdu, pU+jj*bs, sdu, C+jj+jj*ldc, ldc, pd+jj);

			kernel_dunpack_nt_4_lib4(4, pU+jj*bs+0*sdu, C+jj+(ii+0)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, pU+jj*bs+4*sdu, C+jj+(ii+4)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, pU+jj*bs+8*sdu, C+jj+(ii+8)*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, pD+0*4);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pD+4*4);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+8)*ldc, ldc, pD+8*4);
		kernel_dpotrf_nt_l_12x4_lib4(ii, pU, sdu, pU, pD, bs, pD, bs, pd+ii);
		kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+4*4, C+ii+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+8*4, C+ii+(ii+8)*ldc, ldc);

		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pU+4*sdu+ii*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+8)*ldc, ldc, pU+8*sdu+ii*bs);

#if 1
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD+0*4);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pD+8*4);
		kernel_dpack_tn_4_lib4(4, C+ii+8+(ii+8)*ldc, ldc, pD+12*4);
		kernel_dpotrf_nt_l_8x8_lib4(ii+4, pU+4*sdu, sdu, pU+4*sdu, sdu, pD, 8, pD, 8, pd+ii+4);
		kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+4+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+8*4, C+ii+4+(ii+8)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+12*4, C+ii+8+(ii+8)*ldc, ldc);
#else
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD+0*4);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pD+4*4);
		kernel_dpotrf_nt_l_8x4_lib4(ii+4, pU+4*sdu, sdu, pU+4*sdu, pD, bs, pD, bs, pd+ii+4);
		kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+4+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+4*4, C+ii+4+(ii+8)*ldc, ldc);

		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pU+8*sdu+(ii+4)*bs);

		kernel_dpack_tn_4_lib4(4, C+ii+8+(ii+8)*ldc, ldc, pD);
		kernel_dpotrf_nt_l_4x4_lib4(ii+8, pU+8*sdu, pU+8*sdu, pD, pD, pd+ii+8);
		kernel_dunpack_nt_4_lib4(4, pD, C+ii+8+(ii+8)*ldc, ldc);
#endif
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto u_0_left_4;
			}
		if(m-ii<=8)
			{
			goto u_0_left_8;
			}
		else
			{
			goto u_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, pU+jj*bs+0*sdu);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, pU+jj*bs+4*sdu);
			kernel_dtrsm_nn_ru_inv_8x4_lib4c4c(jj, pU, sdu, C+jj*ldc, ldc, pU+jj*bs, sdu, pU+jj*bs, sdu, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dunpack_nt_4_lib4(4, pU+jj*bs+0*sdu, C+jj+(ii+0)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, pU+jj*bs+4*sdu, C+jj+(ii+4)*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, pD+0*4);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pD+4*4);
		kernel_dpotrf_nt_l_8x4_lib4(ii, pU, sdu, pU, pD, bs, pD, bs, pd+ii);
		kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pD+4*4, C+ii+(ii+4)*ldc, ldc);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pU+4*sdu+ii*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD);
		kernel_dpotrf_nt_l_4x4_lib4(ii+4, pU+4*sdu, pU+4*sdu, pD, pD, pd+ii+4);
		kernel_dunpack_nt_4_lib4(4, pD, C+ii+4+(ii+4)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto u_0_left_4;
			}
		else
			{
			goto u_0_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+ii*ldc, ldc, pU+jj*bs);
			kernel_dtrsm_nn_ru_inv_4x4_lib4c4c(jj, pU, C+jj*ldc, ldc, pU+jj*bs, pU+jj*bs, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dunpack_nt_4_lib4(4, pU+jj*bs, C+jj+ii*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+ii*ldc, ldc, pD);
		kernel_dpotrf_nt_l_4x4_lib4(ii, pU, pU, pD, pD, pd+ii);
		kernel_dunpack_nt_4_lib4(4, pD, C+ii+ii*ldc, ldc);
		}
	if(ii<m)
		{
		goto u_0_left_4;
		}
#endif
	goto u_0_return;

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
u_0_left_12:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, pU+jj*bs+0*sdu);
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, pU+jj*bs+4*sdu);
		kernel_dpack_tn_4_vs_lib4(4, C+jj+(ii+8)*ldc, ldc, pU+jj*bs+8*sdu, m-ii-8);

		kernel_dtrsm_nn_ru_inv_12x4_lib4c4c(jj, pU, sdu, C+jj*ldc, ldc, pU+jj*bs, sdu, pU+jj*bs, sdu, C+jj+jj*ldc, ldc, pd+jj); // TODO vs

		kernel_dunpack_nt_4_lib4(4, pU+jj*bs+0*sdu, C+jj+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, pU+jj*bs+4*sdu, C+jj+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_vs_lib4(4, pU+jj*bs+8*sdu, C+jj+(ii+8)*ldc, ldc, m-ii-8);
		}
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, pD+0*4);
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pD+4*4);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+(ii+8)*ldc, ldc, pD+8*4, m-ii-8);
	kernel_dpotrf_nt_l_12x4_vs_lib4(ii, pU, sdu, pU, pD, bs, pD, bs, pd+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+(ii+0)*ldc, ldc);
	kernel_dunpack_nt_4_lib4(4, pD+4*4, C+ii+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, pD+8*4, C+ii+(ii+8)*ldc, ldc, m-ii-8);

	kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pU+4*sdu+ii*bs);
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+8)*ldc, ldc, pU+8*sdu+ii*bs);

#if 1
	kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD+0*4);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pD+8*4, m-ii-8);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+8+(ii+8)*ldc, ldc, pD+12*4, m-ii-8);
	kernel_dpotrf_nt_l_8x8_vs_lib4(ii+4, pU+4*sdu, sdu, pU+4*sdu, sdu, pD, 8, pD, 8, pd+ii+4, m-ii-4, m-ii-4);
	kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+4+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, pD+8*4, C+ii+4+(ii+8)*ldc, ldc, m-ii-8);
	kernel_dunpack_nt_4_vs_lib4(4, pD+12*4, C+ii+8+(ii+8)*ldc, ldc, m-ii-8);
#else
	kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD+0*4);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pD+4*4, m-ii-8);
	kernel_dpotrf_nt_l_8x4_vs_lib4(ii+4, pU+4*sdu, sdu, pU+4*sdu, pD, bs, pD, bs, pd+ii+4, m-ii-4, m-ii-4);
	kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+4+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, pD+4*4, C+ii+4+(ii+8)*ldc, ldc, m-ii-8);

	kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+8)*ldc, ldc, pU+8*sdu+(ii+4)*bs);

	kernel_dpack_tn_4_vs_lib4(4, C+ii+8+(ii+8)*ldc, ldc, pD, m-ii-8);
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii+8, pU+8*sdu, pU+8*sdu, pD, pD, pd+ii+8, m-ii-8, m-ii-8);
	kernel_dunpack_nt_4_vs_lib4(4, pD, C+ii+8+(ii+8)*ldc, ldc, m-ii-8);
#endif
	goto u_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
u_0_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, pU+jj*bs+0*sdu);
		kernel_dpack_tn_4_vs_lib4(4, C+jj+(ii+4)*ldc, ldc, pU+jj*bs+4*sdu, m-ii-4);

		kernel_dtrsm_nn_ru_inv_8x4_lib4c4c(jj, pU, sdu, C+jj*ldc, ldc, pU+jj*bs, sdu, pU+jj*bs, sdu, C+jj+jj*ldc, ldc, pd+jj); // TODO vs

		kernel_dunpack_nt_4_lib4(4, pU+jj*bs+0*sdu, C+jj+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_vs_lib4(4, pU+jj*bs+4*sdu, C+jj+(ii+4)*ldc, ldc, m-ii-4);
		}
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, pD+0*4);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+(ii+4)*ldc, ldc, pD+4*4, m-ii-4);
	kernel_dpotrf_nt_l_8x4_vs_lib4(ii, pU, sdu, pU, pD, bs, pD, bs, pd+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_lib4(4, pD+0*4, C+ii+(ii+0)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, pD+4*4, C+ii+(ii+4)*ldc, ldc, m-ii-4);

	kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, pU+4*sdu+ii*bs);

	kernel_dpack_tn_4_vs_lib4(4, C+ii+4+(ii+4)*ldc, ldc, pD, m-ii-4); // TODO pack vs with m and n, or triangle
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii+4, pU+4*sdu, pU+4*sdu, pD, pD, pd+ii+4, m-ii-4, m-ii-4);
	kernel_dunpack_nt_4_vs_lib4(4, pD, C+ii+4+(ii+4)*ldc, ldc, m-ii-4); // TODO pack vs with m and n, or triangle
	goto u_0_return;
#endif

u_0_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_vs_lib4(4, C+jj+ii*ldc, ldc, pU+jj*bs, m-ii);
		kernel_dtrsm_nn_ru_inv_4x4_lib4c4c(jj, pU, C+jj*ldc, ldc, pU+jj*bs, pU+jj*bs, C+jj+jj*ldc, ldc, pd+jj); // TODO vs
		kernel_dunpack_nt_4_vs_lib4(4, pU+jj*bs, C+jj+ii*ldc, ldc, m-ii); // TODO vs
		}
	kernel_dpack_tn_4_vs_lib4(4, C+ii+ii*ldc, ldc, pD, m-ii); // TODO pack vs with m and n, or triangle
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii, pU, pU, pD, pD, pd+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_vs_lib4(4, pD, C+ii+ii*ldc, ldc, m-ii); // TODO pack vs with m and n, or triangle
	goto u_0_return;

u_0_return:
	for(ii=0; ii<m; ii++)
		{
		if(pd[ii]==0.0)
			{
			*info = ii+1;
			return;
			}
		}
	return;
	


u_1:

	m1 = (m+128-1)/128*128;
	sC_size = blasfeo_memsize_dmat(m1, m1);
//	sC_size = blasfeo_memsize_dmat(m, m);
	stot_size = sC_size;
	smat_mem = malloc(stot_size+63);
	blasfeo_align_64_byte(smat_mem, &smat_mem_align);
	blasfeo_create_dmat(m, m, &sC, smat_mem_align);
	sdc = sC.cn;
	pc = sC.dA;

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<m-11; ii+=12)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+jj*bs);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+jj*bs);
			kernel_dtrsm_nt_rl_inv_12x4_lib4(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+jj*sdc+jj*bs, pc+jj);
			kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+jj*bs, C+jj+(ii+0)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+jj*bs, C+jj+(ii+4)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+jj*bs, C+jj+(ii+8)*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+ii*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+ii*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+ii*bs);
		kernel_dpotrf_nt_l_12x4_lib4(ii, sC.pA+ii*sdc, sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sdc, sC.pA+ii*sdc+ii*bs, sdc, pc+ii);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+ii*bs, C+ii+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+ii*bs, C+ii+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+ii*bs, C+ii+(ii+8)*ldc, ldc);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+(ii+4)*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+4)*bs);
#if 1
		kernel_dpack_tn_4_lib4(4, C+ii+8+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+8)*bs);
		kernel_dpotrf_nt_l_8x8_lib4(ii+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, pc+ii+4);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+(ii+4)*bs, C+ii+4+(ii+8)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+(ii+8)*bs, C+ii+8+(ii+8)*ldc, ldc);
#else
		kernel_dpotrf_nt_l_8x4_lib4(ii+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, pc+ii+4);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+(ii+4)*bs, C+ii+4+(ii+8)*ldc, ldc);
		kernel_dpack_tn_4_lib4(4, C+ii+8+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+8)*bs);
		kernel_dpotrf_nt_l_4x4_lib4(ii+8, sC.pA+(ii+8)*sdc, sC.pA+(ii+8)*sdc, sC.pA+(ii+8)*sdc+(ii+8)*bs, sC.pA+(ii+8)*sdc+(ii+8)*bs, pc+ii+8);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+8)*sdc+(ii+8)*bs, C+ii+8+(ii+8)*ldc, ldc);
#endif
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto u_1_left_4;
			}
		if(m-ii<=8)
			{
			goto u_1_left_8;
			}
		else
			{
			goto u_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-7; ii+=8)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+jj*bs);
			kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs);
			kernel_dtrsm_nt_rl_inv_8x4_lib4(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+jj*sdc+jj*bs, pc+jj);
			kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+jj*bs, C+jj+(ii+0)*ldc, ldc);
			kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+jj*bs, C+jj+(ii+4)*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+ii*bs);
		kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+ii*bs);
		kernel_dpotrf_nt_l_8x4_lib4(ii, sC.pA+ii*sdc, sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sdc, sC.pA+ii*sdc+ii*bs, sdc, pc+ii);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+ii*bs, C+ii+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+ii*bs, C+ii+(ii+4)*ldc, ldc);
		kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+(ii+4)*bs);
		kernel_dpotrf_nt_l_4x4_lib4(ii+4, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sC.pA+(ii+4)*sdc+(ii+4)*bs, pc+ii+4);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto u_1_left_4;
			}
		else
			{
			goto u_1_left_8;
			}
		}
#else
	for(; ii<m-3; ii+=4)
		{
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dpack_tn_4_lib4(4, C+jj+ii*ldc, ldc, sC.pA+ii*sdc+jj*bs);
			kernel_dtrsm_nt_rl_inv_4x4_lib4(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs, sC.pA+jj*sdc+jj*bs, pc+jj);
			kernel_dunpack_nt_4_lib4(4, sC.pA+ii*sdc+jj*bs, C+jj+ii*ldc, ldc);
			}
		kernel_dpack_tn_4_lib4(4, C+ii+ii*ldc, ldc, sC.pA+ii*sdc+ii*bs);
		kernel_dpotrf_nt_l_4x4_lib4(ii, sC.pA+ii*sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sC.pA+ii*sdc+ii*bs, pc+ii);
		kernel_dunpack_nt_4_lib4(4, sC.pA+ii*sdc+ii*bs, C+ii+ii*ldc, ldc);
		}
	if(ii<m)
		{
		goto u_1_left_4;
		}
#endif
	goto u_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
u_1_left_12:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+jj*bs);
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs);
		kernel_dpack_tn_4_vs_lib4(4, C+jj+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+jj*bs, m-ii-8);
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+jj*sdc+jj*bs, pc+jj, m-ii, m-jj);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+jj*bs, C+jj+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+jj*bs, C+jj+(ii+4)*ldc, ldc);
		kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+jj*bs, C+jj+(ii+8)*ldc, ldc, m-ii-8);
		}
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+ii*bs);
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+ii*bs);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+ii*bs, m-ii-8);
	kernel_dpotrf_nt_l_12x4_vs_lib4(ii, sC.pA+ii*sdc, sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sdc, sC.pA+ii*sdc+ii*bs, sdc, pc+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+ii*bs, C+ii+(ii+0)*ldc, ldc);
	kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+ii*bs, C+ii+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+ii*bs, C+ii+(ii+8)*ldc, ldc, m-ii-8);
	kernel_dpack_tn_4_lib4(4, C+ii+4+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+(ii+4)*bs);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+4+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+4)*bs, m-ii-8);
#if 1
	kernel_dpack_tn_4_vs_lib4(4, C+ii+8+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+8)*bs, m-ii-8); // TODO triangle
	kernel_dpotrf_nt_l_8x8_lib4(ii+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, pc+ii+4);
	kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+(ii+4)*bs, C+ii+4+(ii+8)*ldc, ldc, m-ii-8);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+(ii+8)*bs, C+ii+8+(ii+8)*ldc, ldc, m-ii-8); // TODO triangle
#else
	kernel_dpotrf_nt_l_8x4_vs_lib4(ii+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sdc, pc+ii+4, m-ii-4, m-ii-4);
	kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+(ii+4)*bs, C+ii+4+(ii+8)*ldc, ldc, m-ii-8);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+8+(ii+8)*ldc, ldc, sC.pA+(ii+8)*sdc+(ii+8)*bs, m-ii-8); // TODO triangle
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii+8, sC.pA+(ii+8)*sdc, sC.pA+(ii+8)*sdc, sC.pA+(ii+8)*sdc+(ii+8)*bs, sC.pA+(ii+8)*sdc+(ii+8)*bs, pc+ii+8, m-ii-8, m-ii-8);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+8)*sdc+(ii+8)*bs, C+ii+8+(ii+8)*ldc, ldc, m-ii-8); // TODO triangle
#endif
	goto u_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
u_1_left_8:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_lib4(4, C+jj+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+jj*bs);
		kernel_dpack_tn_4_vs_lib4(4, C+jj+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs, m-ii-4);
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+jj*sdc+jj*bs, pc+jj, m-ii, m-jj);
		kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+jj*bs, C+jj+(ii+0)*ldc, ldc);
		kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+4)*sdc+jj*bs, C+jj+(ii+4)*ldc, ldc, m-ii-4);
		}
	kernel_dpack_tn_4_lib4(4, C+ii+(ii+0)*ldc, ldc, sC.pA+(ii+0)*sdc+ii*bs);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+ii*bs, m-ii-4);
	kernel_dpotrf_nt_l_8x4_vs_lib4(ii, sC.pA+ii*sdc, sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sdc, sC.pA+ii*sdc+ii*bs, sdc, pc+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_lib4(4, sC.pA+(ii+0)*sdc+ii*bs, C+ii+(ii+0)*ldc, ldc);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+4)*sdc+ii*bs, C+ii+(ii+4)*ldc, ldc, m-ii-4);
	kernel_dpack_tn_4_vs_lib4(4, C+ii+4+(ii+4)*ldc, ldc, sC.pA+(ii+4)*sdc+(ii+4)*bs, m-ii-4); // TODO triangle
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii+4, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc, sC.pA+(ii+4)*sdc+(ii+4)*bs, sC.pA+(ii+4)*sdc+(ii+4)*bs, pc+ii+4, m-ii-4, m-ii-4);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+(ii+4)*sdc+(ii+4)*bs, C+ii+4+(ii+4)*ldc, ldc, m-ii-4); // TODO triangle
	goto u_1_return;
#endif

u_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dpack_tn_4_vs_lib4(4, C+jj+ii*ldc, ldc, sC.pA+ii*sdc+jj*bs, m-ii);
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, &d_1, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs, sC.pA+jj*sdc+jj*bs, pc+jj, m-ii, m-jj);
		kernel_dunpack_nt_4_vs_lib4(4, sC.pA+ii*sdc+jj*bs, C+jj+ii*ldc, ldc, m-ii);
		}
	kernel_dpack_tn_4_vs_lib4(4, C+ii+ii*ldc, ldc, sC.pA+ii*sdc+ii*bs, m-ii); // TODO triangle
	kernel_dpotrf_nt_l_4x4_vs_lib4(ii, sC.pA+ii*sdc, sC.pA+ii*sdc, sC.pA+ii*sdc+ii*bs, sC.pA+ii*sdc+ii*bs, pc+ii, m-ii, m-ii);
	kernel_dunpack_nt_4_vs_lib4(4, sC.pA+ii*sdc+ii*bs, C+ii+ii*ldc, ldc, m-ii); // TODO triangle
	goto u_1_return;

u_1_return:
	for(ii=0; ii<m; ii++)
		{
		if(pc[ii]==0.0)
			{
			*info = ii+1;
			free(smat_mem);
			return;
			}
		}
	free(smat_mem);
	return;

	}
