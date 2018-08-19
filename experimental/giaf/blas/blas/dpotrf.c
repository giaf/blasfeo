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

#include "../../../../include/blasfeo_target.h"
#include "../../../../include/blasfeo_common.h"
#include "../../../../include/blasfeo_d_aux.h"
#include "../../../../include/blasfeo_d_kernel.h"



void blasfeo_dpotrf(char *uplo, int *pm, double *C, int *pldc)
	{

	int m = *pm;
	int ldc = *pldc;

	int ii, jj;

	int bs = 4;

	double pd[300] __attribute__ ((aligned (64))); // TODO change to 256 !!!

	double pU[12*300] __attribute__ ((aligned (64))); // TODO change to 256 !!!
	int sdu = 256;

	ii = 0;
#if 1
	for(; ii<m-11; ii+=12)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_12_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_12x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
//		kernel_dpotrf_nt_l_4x4_lib4cc(jj, pU, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
//		kernel_dtrsm_nt_rl_inv_8x4_lib4cc(jj, pU+4*sdu, sdu, C+jj, ldc, C+ii+4+jj*ldc, ldc, C+ii+4+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_8_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs, sdu);
		kernel_dpotrf_nt_l_8x4_lib4cc(jj+4, pU+4*sdu, sdu, C+jj+4, ldc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
//		kernel_dpotrf_nt_l_4x4_lib4cc(jj+4, pU+4*sdu, C+jj+4, ldc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
//		kernel_dtrsm_nt_rl_inv_4x4_lib4cc(jj+4, pU+8*sdu, C+jj+4, ldc, C+ii+8+(jj+4)*ldc, ldc, C+ii+8+(jj+4)*ldc, ldc, C+jj+4+(jj+4)*ldc, ldc, pd+jj+4);
		kernel_dpack_nn_4_lib4(4, C+ii+8+(jj+4)*ldc, ldc, pU+8*sdu+(jj+4)*bs);
		kernel_dpotrf_nt_l_4x4_lib4cc(jj+8, pU+8*sdu, C+jj+8, ldc, C+ii+8+(jj+8)*ldc, ldc, C+ii+8+(jj+8)*ldc, ldc, pd+jj+8);
		}
#endif
#if 1
	for(; ii<m-7; ii+=8)
		{
		jj = 0;
		for(; jj<ii; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
			kernel_dpack_nn_8_lib4(4, C+ii+jj*ldc, ldc, pU+jj*bs, sdu);
			}
		kernel_dpotrf_nt_l_8x4_lib4cc(jj, pU, sdu, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
//		kernel_dpotrf_nt_l_4x4_lib4cc(jj, pU, C+jj, ldc, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc, pd+jj);
//		kernel_dtrsm_nt_rl_inv_4x4_lib4cc(jj, pU+4*sdu, C+jj, ldc, C+ii+4+jj*ldc, ldc, C+ii+4+jj*ldc, ldc, C+jj+jj*ldc, ldc, pd+jj);
		kernel_dpack_nn_4_lib4(4, C+ii+4+jj*ldc, ldc, pU+4*sdu+jj*bs);
		kernel_dpotrf_nt_l_4x4_lib4cc(jj+4, pU+4*sdu, C+jj+4, ldc, C+ii+4+(jj+4)*ldc, ldc, C+ii+4+(jj+4)*ldc, ldc, pd+jj+4);
		}
#endif
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

	}
