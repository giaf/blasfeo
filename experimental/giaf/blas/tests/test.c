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
#include "../../../../include/blasfeo_timing.h"
#include "../../../../include/blasfeo_d_aux.h"
#include "../../../../include/blasfeo_d_aux_ext_dep.h"
#include "../../../../include/blasfeo_d_kernel.h"


int main()
	{

	const double GHz_max = 3.3;
	const double flops_max = 16;

	printf("A = [%f %f];\n", GHz_max, flops_max);
	printf("\n");
	printf("B = [\n");

	int ii, jj, ll;

	int rep, rep_in;
	int nrep_in = 10; // number of benchmark batches

	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};

//	for(ll=0; ll<1; ll++)
//	for(ll=0; ll<4; ll++)
//	for(ll=0; ll<24; ll++)
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
//		int n = 12;
		int nrep = nnrep[ll]/nrep_in;
		nrep = nrep>1 ? nrep : 1;
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<12 ? 12 : n;
//		n = n<8 ? 8 : n;


		double *A = malloc(n*n*sizeof(double));
		for(ii=0; ii<n*n; ii++)
			A[ii] = ii;
		int lda = n;
//		d_print_mat(n, n, A, n);

		double *B = malloc(n*n*sizeof(double));
		for(ii=0; ii<n*n; ii++)
			B[ii] = 0;
		for(ii=0; ii<n; ii++)
			B[ii*(n+1)] = 1.0;
		int ldb = n;
//		d_print_mat(n, n, B, ldb);

		double *C = malloc(n*n*sizeof(double));
		for(ii=0; ii<n*n; ii++)
			C[ii] = -1;
		int ldc = n;
//		d_print_mat(n, n, C, ldc);


		int bs = 4;

		struct blasfeo_dmat sA; blasfeo_allocate_dmat(n, n, &sA);
		blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
		int sda = sA.cn;
		struct blasfeo_dmat sB; blasfeo_allocate_dmat(n, n, &sB);
		blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
		int sdb = sB.cn;
		struct blasfeo_dmat sC; blasfeo_allocate_dmat(n, n, &sC);
		blasfeo_pack_dmat(n, n, C, n, &sC, 0, 0);
		int sdc = sC.cn;


		/* timing */
		blasfeo_timer timer;

		double time_blasfeo   = 1e15;
		double time_blas      = 1e15;
		double time_blas_pack = 1e15;
		double tmp_time_blasfeo;
		double tmp_time_blas;
		double tmp_time_blas_pack;

		/* benchmarks */

		double alpha = 1.0;
		double beta = 0.0;





		/* call blas */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLAS
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

#if 1
				ii = 0;
				for(; ii<n-11; ii+=12)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_12x4_lib(n, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//						kernel_dgemm_nt_12x4_lib(n, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
						}
					}
				for(; ii<n-7; ii+=8)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_8x4_lib(n, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//						kernel_dgemm_nt_8x4_lib(n, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
						}
					}
				for(; ii<n-3; ii+=4)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_4x4_lib(n, &alpha, A+ii, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//						kernel_dgemm_nt_4x4_lib(n, &alpha, A+ii, lda, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
						}
					}
#else
				ii = 0;
				for(; ii<n-3; ii+=4)
					{
					for(jj=0; jj<n-2; jj+=3)
						{
						kernel_dgemm_tn_4x3_lib(n, &alpha, A+ii*lda, lda, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
						}
					}
#endif

				}

			tmp_time_blas = blasfeo_toc(&timer) / nrep;
			time_blas = tmp_time_blas<time_blas ? tmp_time_blas : time_blas;
			// BENCHMARK_BLAS

			}

//		d_print_mat(n, n, C, ldc);





		/* call blas with packing */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

//				blasfeo_pack_tran_dmat(n, n, B, n, &sB, 0, 0);
//				blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);

				double pU[3072] __attribute__ ((aligned (64)));
				int sdu = 256;
				ii = 0;
				if(n<=256)
					{
					for(; ii<n-11; ii+=12)
						{
//						blasfeo_pack_dmat(12, n, A+ii, n, &sA, ii, 0);
//						blasfeo_pack_tran_dmat(n, 12, A+ii*lda, n, &sA, ii, 0);
						kernel_dpack_nn_12_lib4(n, A+ii+0, lda, pU, sdu);
						for(jj=0; jj<n-3; jj+=4)
							{
//							kernel_dgemm_nt_12x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*bs, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
							kernel_dgemm_nn_12x4_lib4x(n, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//							kernel_dgemm_nt_12x4_lib4x(n, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
							}
						}
					for(; ii<n-7; ii+=8)
						{
//						blasfeo_pack_dmat(8, n, A+ii, n, &sA, ii, 0);
//						blasfeo_pack_tran_dmat(n, 8, A+ii*lda, n, &sA, ii, 0);
						kernel_dpack_nn_8_lib4(n, A+ii+0, lda, pU, sdu);
						for(jj=0; jj<n-3; jj+=4)
							{
//							kernel_dgemm_nt_8x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*bs, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
							kernel_dgemm_nn_8x4_lib4x(n, &alpha, pU, sdu, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//							kernel_dgemm_nt_8x4_lib4x(n, &alpha, pU, sdu, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
							}
						}
					for(; ii<n-3; ii+=4)
						{
//						blasfeo_pack_dmat(4, n, A+ii, n, pU, 0, 0);
//						blasfeo_pack_tran_dmat(n, 4, A+ii*lda, n, &sA, ii, 0);
						kernel_dpack_nn_4_lib4(n, A+ii, lda, pU);
						for(jj=0; jj<n-3; jj+=4)
							{
//							kernel_dgemm_nt_4x4_lib4x(n, &alpha, sA.pA+ii*sda, sB.pA+jj*bs, &beta, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs);
							kernel_dgemm_nn_4x4_lib4x(n, &alpha, pU, B+jj*ldb, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
//							kernel_dgemm_nt_4x4_lib4x(n, &alpha, pU, B+jj, ldb, &beta, C+ii+jj*ldc, ldc, C+ii+jj*ldc, ldc);
							}
						}
					}

//				blasfeo_unpack_dmat(n, n, &sC, 0, 0, C, n);

				}

			tmp_time_blas_pack = blasfeo_toc(&timer) / nrep;
			time_blas_pack = tmp_time_blas_pack<time_blas_pack ? tmp_time_blas_pack : time_blas_pack;
			// BENCHMARK_BLASFEO_END

			}

//		d_print_mat(n, n, C, ldc);





		/* call blasfeo */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

#if 1
				ii = 0;
				for(; ii<n-11; ii+=12)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_12x4_lib4(n, &alpha, sA.pA+ii*sda, sda, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
//						kernel_dgemm_nt_12x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
						}
					}
				for(; ii<n-7; ii+=8)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_8x4_lib4(n, &alpha, sA.pA+ii*sda, sda, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
//						kernel_dgemm_nt_8x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
						}
					}
				for(; ii<n-3; ii+=4)
					{
					for(jj=0; jj<n-3; jj+=4)
						{
						kernel_dgemm_nn_4x4_lib4(n, &alpha, sA.pA+ii*sda, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs);
//						kernel_dgemm_nt_4x4_lib4(n, &alpha, sA.pA+ii*sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs);
						}
					}
#else
				double pT[3072] __attribute__ ((aligned (64)));
				int sdt = 256;
				ii = 0;
				if(n<=256)
					{
					for(; ii<n-11; ii+=12)
						{
						kernel_dpatr_tn_4_lib4(n, sA.pA+(ii+0)*bs, sda, pT+0*sdt);
						kernel_dpatr_tn_4_lib4(n, sA.pA+(ii+4)*bs, sda, pT+4*sdt);
						kernel_dpatr_tn_4_lib4(n, sA.pA+(ii+8)*bs, sda, pT+8*sdt);
						for(jj=0; jj<n-3; jj+=4)
							{
							kernel_dgemm_nn_12x4_lib4(n, &alpha, pT, sdt, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
	//						kernel_dgemm_nt_12x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
							}
						}
					for(; ii<n-7; ii+=8)
						{
						kernel_dpatr_tn_4_lib4(n, sA.pA+(ii+0)*bs, sda, pT+0*sdt);
						kernel_dpatr_tn_4_lib4(n, sA.pA+(ii+4)*bs, sda, pT+4*sdt);
						for(jj=0; jj<n-3; jj+=4)
							{
							kernel_dgemm_nn_8x4_lib4(n, &alpha, pT, sdt, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
	//						kernel_dgemm_nt_8x4_lib4(n, &alpha, sA.pA+ii*sda, sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sdc, sC.pA+ii*sdc+jj*bs, sdc);
							}
						}
					for(; ii<n-3; ii+=4)
						{
						kernel_dpatr_tn_4_lib4(n, sA.pA+ii*bs, sda, pT);
						for(jj=0; jj<n-3; jj+=4)
							{
							kernel_dgemm_nn_4x4_lib4(n, &alpha, pT, 0, sB.pA+jj*bs, sdb, &beta, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs);
	//						kernel_dgemm_nt_4x4_lib4(n, &alpha, sA.pA+ii*sda, sB.pA+jj*sdb, &beta, sC.pA+ii*sdc+jj*bs, sC.pA+ii*sdc+jj*bs);
							}
						}
					}
#endif

				}

			tmp_time_blasfeo = blasfeo_toc(&timer) / nrep;
			time_blasfeo = tmp_time_blasfeo<time_blasfeo ? tmp_time_blasfeo : time_blasfeo;
			// BENCHMARK_BLASFEO_END

			}

//		d_print_mat(n, n, C, ldc);
//		blasfeo_print_dmat(n, n, &sC, 0, 0);



		double Gflops_max = flops_max * GHz_max;

		double flop_operation = 2.0*n*n*n;

		double Gflops_blas      = 1e-9*flop_operation/time_blas;
		double Gflops_blas_pack = 1e-9*flop_operation/time_blas_pack;
		double Gflops_blasfeo   = 1e-9*flop_operation/time_blasfeo;

		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n",
			n,
			Gflops_blas, 100.0*Gflops_blas/Gflops_max,
			Gflops_blas_pack, 100.0*Gflops_blas_pack/Gflops_max,
			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max);

		free(A);
		free(B);
		free(C);
		blasfeo_free_dmat(&sA);
		blasfeo_free_dmat(&sB);
		blasfeo_free_dmat(&sC);

		}

	printf("];\n");

	return 0;

	}
