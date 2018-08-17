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



// XXX implementation for x64 intel haswell only
void blasfeo_dgemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc)
	{

	if(*m>256 | *n>256 | *k>256)
		{
		printf("\nSizes m | n | k >256 not implemented yet\n");
		exit(1);
		}

	if(*m%4!=0 | !n%4!=0)
		{
		printf("\nSizes m | n %4!=0 not implemented yet\n");
		exit(1);
		}

	int ii, jj;

	int bs = 4;

	double pU[3072] __attribute__ ((aligned (64)));
	int sdu = 256;

	if(*ta=='n')
		{

		if(*tb=='n')
			{

			ii = 0;
			for(; ii<*m-11; ii+=12)
				{
				kernel_dpack_nn_12_lib4(*k, A+ii+0, *lda, pU, sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_12x4_lib4x(*k, alpha, pU, sdu, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-7; ii+=8)
				{
				kernel_dpack_nn_8_lib4(*k, A+ii+0, *lda, pU, sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_8x4_lib4x(*k, alpha, pU, sdu, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-3; ii+=4)
				{
				kernel_dpack_nn_4_lib4(*k, A+ii, *lda, pU);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_4x4_lib4x(*k, alpha, pU, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}

			}
		else // tb==t
			{

			ii = 0;
			for(; ii<*m-11; ii+=12)
				{
				kernel_dpack_nn_12_lib4(*k, A+ii+0, *lda, pU, sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_12x4_lib4x(*k, alpha, pU, sdu, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-7; ii+=8)
				{
				kernel_dpack_nn_8_lib4(*k, A+ii+0, *lda, pU, sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_8x4_lib4x(*k, alpha, pU, sdu, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-3; ii+=4)
				{
				kernel_dpack_nn_4_lib4(*k, A+ii, *lda, pU);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_4x4_lib4x(*k, alpha, pU, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}

			}

		}
	else // ta==t
		{

		if(*tb=='n')
			{

			ii = 0;
			for(; ii<*m-11; ii+=12)
				{
				kernel_dpack_tn_4_lib4(*k, A+(ii+0)**lda, *lda, pU);
				kernel_dpack_tn_4_lib4(*k, A+(ii+4)**lda, *lda, pU+4*sdu);
				kernel_dpack_tn_4_lib4(*k, A+(ii+8)**lda, *lda, pU+8*sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_12x4_lib4x(*k, alpha, pU, sdu, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-7; ii+=8)
				{
				kernel_dpack_tn_4_lib4(*k, A+(ii+0)**lda, *lda, pU);
				kernel_dpack_tn_4_lib4(*k, A+(ii+4)**lda, *lda, pU+4*sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_8x4_lib4x(*k, alpha, pU, sdu, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-3; ii+=4)
				{
				kernel_dpack_tn_4_lib4(*k, A+ii**lda, *lda, pU);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nn_4x4_lib4x(*k, alpha, pU, B+jj**ldb, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}

			}
		else // tb==t
			{

			ii = 0;
			for(; ii<*m-11; ii+=12)
				{
				kernel_dpack_tn_4_lib4(*k, A+(ii+0)**lda, *lda, pU);
				kernel_dpack_tn_4_lib4(*k, A+(ii+4)**lda, *lda, pU+4*sdu);
				kernel_dpack_tn_4_lib4(*k, A+(ii+8)**lda, *lda, pU+8*sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_12x4_lib4x(*k, alpha, pU, sdu, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-7; ii+=8)
				{
				kernel_dpack_tn_4_lib4(*k, A+(ii+0)**lda, *lda, pU);
				kernel_dpack_tn_4_lib4(*k, A+(ii+4)**lda, *lda, pU+4*sdu);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_8x4_lib4x(*k, alpha, pU, sdu, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}
			for(; ii<*m-3; ii+=4)
				{
				kernel_dpack_tn_4_lib4(*k, A+ii**lda, *lda, pU);
				for(jj=0; jj<*n-3; jj+=4)
					{
					kernel_dgemm_nt_4x4_lib4x(*k, alpha, pU, B+jj, *ldb, beta, C+ii+jj**ldc, *ldc, C+ii+jj**ldc, *ldc);
					}
				}

			}

		}
	
	return;

	}

