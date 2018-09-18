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



// XXX implementation for x64 intel haswell only
void blasfeo_dgemm(char *ta, char *tb, int *pm, int *pn, int *pk, double *alpha, double *A, int *plda, double *B, int *pldb, double *beta, double *C, int *pldc)
	{

	int m = *pm;
	int n = *pn;
	int k = *pk;
	int lda = *plda;
	int ldb = *pldb;
	int ldc = *pldc;

	if(m%4!=0 | n%4!=0)
		{
		printf("\nSizes m or n not multiple of 4: not implemented yet\n");
		exit(1);
		}

	int ii, jj;

	int bs = 4;

	double pU[3072] __attribute__ ((aligned (64)));
	int sdu = 256;

	struct blasfeo_dmat sA, sB;
	int sda, sdb;
	int sA_size, sB_size;
	void *smat_mem, *smat_mem_align;

	if(*ta=='n')
		{
		if(*tb=='n')
			{
			if(m>=256 | n>=256 | k>=256)
				{
				goto nn_1;
				}
			else
				{
				goto nn_0;
				}
			}
		else // tb==t
			{
			if(m>=96 | n>=96 | k>=96)
				{
				goto nt_1;
				}
			else
				{
				goto nt_0;
				}
			}
		}
	else // ta==t
		{
		if(*tb=='n')
			{
			if(m>=256 | n>=256 | k>=256)
				{
				goto tn_1;
				}
			else
				{
				goto tn_0;
				}
			}
		else // tb==t
			{
			if(m>=96 | n>=96 | k>=96)
				{
				goto tt_1;
				}
			else
				{
				goto tt_0;
				}
			}
		}

	return;
	


nn_0:
	
	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	
	return;



nn_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(n, k);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(n, k, &sB, smat_mem_align+sA_size);

	blasfeo_pack_tran_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}

	free(smat_mem);

	return;



nt_0:

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii+0, lda, pU, sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	
	return;



nt_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(k, n);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(k, n, &sB, smat_mem_align+sA_size);

	blasfeo_pack_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_nn_12_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_nn_8_lib4(k, A+ii, lda, sA.pA, sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_nn_4_lib4(k, A+ii, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}

	free(smat_mem);

	return;



tn_0:

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_12x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_8x4_lib4cc(k, alpha, pU, sdu, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nn_4x4_lib4cc(k, alpha, pU, B+jj*ldb, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	
	return;



tn_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(n, k);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(n, k, &sB, smat_mem_align+sA_size);

	blasfeo_pack_tran_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}

	free(smat_mem);

	return;



tt_0:

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, pU+8*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, pU);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, pU+4*sdu);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib4cc(k, alpha, pU, sdu, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+ii*lda, lda, pU);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib4cc(k, alpha, pU, B+jj, ldb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	
	return;



tt_1:

	sA_size = blasfeo_memsize_dmat(12, k);
	sB_size = blasfeo_memsize_dmat(k, n);
	smat_mem = malloc(sA_size+sB_size+63);
	smat_mem_align = (void *) ( ( ( (unsigned long long) smat_mem ) + 63) / 64 * 64 );
	blasfeo_create_dmat(12, k, &sA, smat_mem_align);
	blasfeo_create_dmat(k, n, &sB, smat_mem_align+sA_size);

	blasfeo_pack_dmat(k, n, B, k, &sB, 0, 0);

	sda = sA.cn;
	sdb = sB.cn;

	ii = 0;
	for(; ii<m-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		kernel_dpack_tn_4_lib4(k, A+(ii+8)*lda, lda, sA.pA+8*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_12x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		kernel_dpack_tn_4_lib4(k, A+(ii+4)*lda, lda, sA.pA+4*sda);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_8x4_lib44c(k, alpha, sA.pA, sda, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}
	for(; ii<m-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(k, A+(ii+0)*lda, lda, sA.pA);
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dgemm_nt_4x4_lib44c(k, alpha, sA.pA, sB.pA+jj*sdb, beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
			}
		}

	free(smat_mem);

	return;

	}

