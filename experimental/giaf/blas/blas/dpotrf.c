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

#include "../include/blasfeo_d_kernel.h"

#include "../../../../include/blasfeo_target.h"
#include "../../../../include/blasfeo_common.h"
#include "../../../../include/blasfeo_d_aux.h"
#include "../../../../include/blasfeo_d_kernel.h"



void blasfeo_dpotrf(char *uplo, int *pm, double *C, int *pldc) // TODO int *info
	{

	if(*uplo!='l')
		{
		printf("\nUplo u not implemented yet\n");
		exit(1);
		}

	int m = *pm;
	int ldc = *pldc;

	int ii, jj;

	int bs = 4;

	double pd[256] __attribute__ ((aligned (64)));

	double pU[12*256] __attribute__ ((aligned (64)));
	int sdu = 256;

	struct blasfeo_dmat sC;
	int sdc;
	double *pc;
	int sC_size, stot_size;
	void *smat_mem, *smat_mem_align;


//	if(*uplo=='l')
//		{
		if(m>=96)
			{
			goto l_1;
			}
		else
			{
			goto l_0;
			}
//		}


l_0:

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_12_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_12x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_8_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs, sdu);
		kernel_dpotrf_nt_l_8x4_lib4cc(jj+4, pU+4*sdu, sdu, C+jj+4, ldc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
		kernel_dpack_nn_4_lib4(4, C+ii+8+(jj+4)*ldc, ldc, pU+8*sdu+(jj+4)*bs);
		kernel_dpotrf_nt_l_4x4_lib4cc(jj+8, pU+8*sdu, C+jj+8, ldc, C+ii+8+(jj+8)*ldc, ldc, C+ii+8+(jj+8)*ldc, ldc, pd+jj+8);
		}
	for(; ii<m-7; ii+=8)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_8_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_8x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_4_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs);
		kernel_dpotrf_nt_l_4x4_lib4cc(jj+4, pU+4*sdu, C+jj+4, ldc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
		}
	for(; ii<m-3; ii+=4)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4cc(jj, pU, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_4_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs);
			}
		kernel_dpotrf_nt_l_4x4_lib4cc(jj, pU, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
		}

	return;


l_1:
	
	sC_size = blasfeo_memsize_dmat(m, m);
	stot_size = sC_size + m*sizeof(double);
	smat_mem = malloc(stot_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(m, m, &sC, smat_mem_align);
	sdc = sC.cn;
	pc = smat_mem_align + sC_size;

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_12_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc);
			}
		kernel_dpotrf_nt_l_12x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		kernel_dpack_nn_8_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs, sdc);
		kernel_dpotrf_nt_l_8x4_lib44c(jj+4, sC.pA+(ii+4)*sdc, sdc, sC.pA+(jj+4)*sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4);
		kernel_dpack_nn_4_lib4(4, C+ii+8+(jj+4)*ldc, ldc, sC.pA+(ii+8)*sdc+(jj+4)*bs);
		kernel_dpotrf_nt_l_4x4_lib44c(jj+8, sC.pA+(ii+8)*sdc, sC.pA+(jj+8)*sdc, C+ii+8+(jj+8)*ldc, ldc, C+ii+8+(jj+8)*ldc, ldc, pc+jj+8);
		}
	for(; ii<m-7; ii+=8)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_8_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs, sdc);
			}
		kernel_dpotrf_nt_l_8x4_lib44c(jj, sC.pA+ii*sdc, sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		kernel_dpack_nn_4_lib4(4, C+ii+4+jj*ldc, ldc, sC.pA+(ii+4)*sdc+jj*bs);
		kernel_dpotrf_nt_l_4x4_lib44c(jj+4, sC.pA+(ii+4)*sdc, sC.pA+(jj+4)*sdc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pc+jj+4);
		}
	for(; ii<m-3; ii+=4)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib44c(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pc+jj);
			kernel_dpack_nn_4_lib4(4, C+ii+jj*ldc, ldc, sC.pA+ii*sdc+jj*bs);
			}
		kernel_dpotrf_nt_l_4x4_lib44c(jj, sC.pA+ii*sdc, sC.pA+jj*sdc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pc+jj);
		}

	free(smat_mem);

	return;

	}
