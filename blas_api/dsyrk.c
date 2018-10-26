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

	int ii, jj;

	int bs = 4;


	void *mem, *mem_align;
	double *pU;
	int sdu;
	struct blasfeo_dmat sA;
	int sA_size;


	// stack memory allocation
// TODO visual studio alignment
#if defined(TARGET_X64_INTEL_HASWELL)
	double pU0[3*4*K_MAX_STACK] __attribute__ ((aligned (64)));
//	double pD0[4*16] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
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
//			if(m>=0 | k>=0 | k>K_MAX_STACK)
			if(m>=256 | k>=256 | k>K_MAX_STACK)
				{
				goto ln_1;
				}
			else
				{
				goto ln_0;
				}
			}
		else if(*ta=='t' | *ta=='T')
			{
			printf("\nBLASFEO: dpotrf: not implemente yet\n");
			return;
			}
		else
			{
			printf("\nBLASFEO: dpotrf: wrong value for ta\n");
			return;
			}
		}
	else if(*uplo=='u' | *uplo=='U')
		{
		if(*ta=='n' | *ta=='N')
			{
			printf("\nBLASFEO: dpotrf: not implemente yet\n");
			return;
			}
		else if(*ta=='t' | *ta=='T')
			{
			printf("\nBLASFEO: dpotrf: not implemente yet\n");
			return;
			}
		else
			{
			printf("\nBLASFEO: dpotrf: wrong value for ta\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dpotrf: wrong value for uplo\n");
		return;
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
//		kernel_dsyrk_nt_l_8x8_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		kernel_dsyrk_nt_l_8x4_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc);
		kernel_dsyrk_nt_l_4x4_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc);
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
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
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

ln_0_left_12:
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dpack_nn_12_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_12x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_12x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
//	kernel_dsyrk_nt_l_8x8_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU+4*sdu, sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+8*sdu, pU+8*sdu, beta, C+(ii+8)+(jj+8)*ldc, ldc, C+(ii+8)+(jj+8)*ldc, ldc,  m-(ii+8), m-(jj+8));
#endif
	goto ln_0_return;

ln_0_left_8:
#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	kernel_dpack_nn_8_vs_lib4(k, A+ii, lda, pU, sdu, m-ii);
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_8x4_vs_lib4cc(k, alpha, pU, sdu, A+jj, lda, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	// TODO haswell 8x8
	kernel_dsyrk_nt_l_8x4_vs_lib44c(k, alpha, pU, sdu, pU, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc,  m-ii, m-jj);
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+4*sdu, pU+4*sdu, beta, C+(ii+4)+(jj+4)*ldc, ldc, C+(ii+4)+(jj+4)*ldc, ldc,  m-(ii+4), m-(jj+4));
#endif
	goto ln_0_return;

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


ln_1:
	sA_size = blasfeo_memsize_dmat(m, k);
	mem = malloc(sA_size+63);
	mem_align = (void *) ( ( ( (unsigned long long) mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(m, k, &sA, mem_align);

	blasfeo_pack_dmat(m, k, A, lda, &sA, 0, 0);
	pU = sA.pA;
	sdu = sA.cn;

	ii = 0;
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
		goto ln_1_left_4;
		}
	goto ln_1_return;

ln_1_left_4:
	for(jj=0; jj<ii; jj+=4)
		{
		kernel_dgemm_nt_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
		}
	kernel_dsyrk_nt_l_4x4_vs_lib44c(k, alpha, pU+ii*sdu, pU+jj*sdu, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, m-ii, m-jj);
	goto ln_1_return;

ln_1_return:
	free(mem);
	return;

	}


