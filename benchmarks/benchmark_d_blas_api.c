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



#include "../include/blasfeo.h"
#include "benchmark_x_common.h"



#if defined(EXTERNAL_BLAS_NETLIB)
//#include "cblas.h"
//#include "lapacke.h"
#include "../include/d_blas.h"
#endif

#if defined(EXTERNAL_BLAS_OPENBLAS)
void openblas_set_num_threads(int num_threads);
//#include "cblas.h"
//#include "lapacke.h"
#include "../include/d_blas.h"
#endif

#if defined(EXTERNAL_BLAS_BLIS)
//void omp_set_num_threads(int num_threads);
#include "blis.h"
#endif

#if defined(EXTERNAL_BLAS_MKL)
#include "mkl.h"
#endif



int main()
	{

#if defined(EXTERNAL_BLAS_OPENBLAS)
openblas_set_num_threads(1);
#endif

#if !defined(BENCHMARKS_MODE)
	printf("\n\n Recompile BLASFEO with BENCHMARKS_MODE=1 to run this benchmark.\n");
	printf("On CMake use -DBLASFEO_BENCHMARKS=ON .\n\n");
	return 0;
#endif
#if !defined(BLAS_API)
	printf("\nRecompile with BLAS_API=1 to run this benchmark!\n\n");
	return 0;
#endif

	printf("\n");
	printf("\n");
	printf("\n");

	printf("BLASFEO performance test - BLAS API - double precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit benchmarks/cpu_freq.h to modify this value).\n", GHz_max);
	printf("\n");

	// maximum flops per cycle, double precision
#if defined(TARGET_X64_INTEL_HASWELL)
	const float flops_max = 16;
	printf("Testing BLAS version for AVX2 and FMA instruction sets, 64 bit (optimized for Intel Haswell): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const float flops_max = 8;
	printf("Testing BLAS version for AVX instruction set, 64 bit (optimized for Intel Sandy Bridge): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_CORE)
	const float flops_max = 4;
	printf("Testing BLAS version for SSE3 instruction set, 64 bit (optimized for Intel Core): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_AMD_BULLDOZER)
	const float flops_max = 8;
	printf("Testing BLAS version for SSE3 and FMA instruction set, 64 bit (optimized for AMD Bulldozer): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X86_AMD_JAGUAR)
	const float flops_max = 2;
	printf("Testing BLAS version for AVX instruction set, 32 bit (optimized for AMD Jaguar): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X86_AMD_BARCELONA)
	const float flops_max = 4; // 2 on jaguar
	printf("Testing BLAS version for SSE3 instruction set, 32 bit (optimized for AMD Barcelona): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const float flops_max = 4;
	printf("Testing BLAS version for NEONv2 instruction set, 64 bit (optimized for ARM Cortex A57): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	const float flops_max = 4;
	printf("Testing BLAS version for NEONv2 instruction set, 64 bit (optimized for ARM Cortex A53): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const float flops_max = 2;
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A15): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	const float flops_max = 0.5;
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A7): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_GENERIC)
	const float flops_max = 2;
	printf("Testing BLAS version for generic scalar instruction set: theoretical peak %5.1f Gflops ???\n", flops_max*GHz_max);
#endif



	FILE *f;
	f = fopen("./build/benchmark_one.m", "w"); // a

//	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
//	fprintf(f, "\n");
//	fprintf(f, "B = [\n");

	printf("\nn\t Gflops\t    %%\t Gflops\n\n");


	int ii, jj, ll;

	int rep, rep_in;
	int nrep_in = 4; // number of benchmark batches

	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};

//	for(ll=0; ll<1; ll++)
//	for(ll=0; ll<2; ll++) // up to 8
//	for(ll=0; ll<3; ll++) // up to 12
//	for(ll=0; ll<4; ll++) // up to 16
//	for(ll=0; ll<24; ll++)
//	for(ll=0; ll<63; ll++) // up to 256
	for(ll=0; ll<75; ll++) // up to 300
//	for(ll=0; ll<115; ll++) // up to 460
//	for(ll=0; ll<120; ll++) // up to 700

		{

		int n = nn[ll];
//		int n = 12;
		int nrep = nnrep[ll];
		nrep = nrep>1 ? nrep : 1;
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<12 ? 12 : n;
//		n = n<8 ? 8 : n;
//		nrep = 1;


		double *A; d_zeros_align(&A, n, n);
		for(ii=0; ii<n*n; ii++)
			A[ii] = ii;
		int lda = n;
//		d_print_mat(n, n, A, n);

		double *B; d_zeros_align(&B, n, n);
		for(ii=0; ii<n*n; ii++)
			B[ii] = 0;
		for(ii=0; ii<n; ii++)
			B[ii*(n+1)] = 1.0;
		int ldb = n;
//		d_print_mat(n, n, B, ldb);

		double *C; d_zeros_align(&C, n, n);
		for(ii=0; ii<n*n; ii++)
			C[ii] = -1;
		int ldc = n;
//		d_print_mat(n, n, C, ldc);

		double *D; d_zeros_align(&D, n, n);
		for(ii=0; ii<n*n; ii++)
			D[ii] = -1;
		int ldd = n;
//		d_print_mat(n, n, C, ldc);

		int *ipiv = malloc(n*sizeof(int));

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
		struct blasfeo_dmat sD; blasfeo_allocate_dmat(n, n, &sD);
		blasfeo_pack_dmat(n, n, D, n, &sD, 0, 0);
		int sdd = sD.cn;


		/* timing */
		blasfeo_timer timer;

		double time_blasfeo   = 1e15;
		double time_blas      = 1e15;
		double time_blas_api  = 1e15;
		double tmp_time_blasfeo;
		double tmp_time_blas;
		double tmp_time_blas_api;

		/* benchmarks */

		char ta = 'n';
		char tb = 't';
		char uplo = 'u';
		int info = 0;

		double alpha = 1.0;
		double beta = 0.0;

		char c_l = 'l';
		char c_n = 'n';
		char c_r = 'r';
		char c_t = 't';
		char c_u = 'u';





#if 1
		/* call blas */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

//			for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
//			dgemm_(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);

			// BENCHMARK_BLAS
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

//				dtrsm_(&c_r, &c_l, &c_t, &c_n, &n, &n, &alpha, B, &n, C, &n);

//				dtrmm_(&c_r, &c_l, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);

//				for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
//				dgemm_(&ta, &tb, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				for(ii=0; ii<n*n; ii++) D[ii] = C[ii];
//				dpotrf_(&uplo, &n, D, &n, &info);
//				dpotrf_(&uplo, &n, B, &n, &info);

				}

			tmp_time_blas = blasfeo_toc(&timer) / nrep;
			time_blas = tmp_time_blas<time_blas ? tmp_time_blas : time_blas;
			// BENCHMARK_BLAS

			}
#endif

//		d_print_mat(n, n, C, ldc);
//		d_print_mat(n, n, D, ldd);





#if 1
		/* call blas with packing */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

//			for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
//			blasfeo_dgemm(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

				blasfeo_dgemm(&c_n, &c_n, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				blasfeo_dgemm(&c_n, &c_t, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				blasfeo_dgemm(&c_t, &c_n, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				blasfeo_dgemm(&c_t, &c_t, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);

//				blasfeo_dsyrk(&c_l, &c_n, &n, &n, &alpha, A, &n, &beta, C, &n);
//				blasfeo_dsyrk(&c_l, &c_t, &n, &n, &alpha, A, &n, &beta, C, &n);
//				blasfeo_dsyrk(&c_u, &c_n, &n, &n, &alpha, A, &n, &beta, C, &n);
//				blasfeo_dsyrk(&c_u, &c_t, &n, &n, &alpha, A, &n, &beta, C, &n);

//				blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &n, &n, &alpha, B, &n, C, &n);
//				blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &n, &n, &alpha, B, &n, C, &n);

//				blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &n, &n, &alpha, B, &n, C, &n);

//				blasfeo_dpotrf(&c_l, &n, B, &n, &info);
//				blasfeo_dpotrf(&c_u, &n, B, &n, &info);

//				blasfeo_dgetrf(&n, &n, B, &n, ipiv, &info);



//				for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
//				blasfeo_dgemm(&ta, &tb, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				for(ii=0; ii<n*n; ii++) D[ii] = C[ii];
//				blasfeo_dpotrf(&uplo, &n, D, &n);
//				blasfeo_dpotrf(&uplo, &n, B, &n, &info);

//				dgemm_(&c_n, &c_n, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				dgemm_(&c_n, &c_t, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
//				dpotrf_(&c_l, &n, B, &n, &info);

#if 0
				int memsize_A = blasfeo_memsize_dmat(n, n);
				int memsize_B = blasfeo_memsize_dmat(n, n);
				int memsize_C = blasfeo_memsize_dmat(n, n);

				int memsize = 64+memsize_A+memsize_B+memsize_C;

				void *mem = calloc(memsize, 1);
				void *mem_align = (void *) ( ( ( (unsigned long long) mem ) + 63) / 64 * 64 );

				struct blasfeo_dmat sA, sB, sC;

				blasfeo_create_dmat(n, n, &sA, mem_align);
				blasfeo_create_dmat(n, n, &sB, mem_align+memsize_A);
				blasfeo_create_dmat(n, n, &sC, mem_align+memsize_A+memsize_B);

				blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
//				blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
				blasfeo_pack_tran_dmat(n, n, B, n, &sB, 0, 0);
				blasfeo_pack_dmat(n, n, C, n, &sC, 0, 0);

				blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sC, 0, 0);

				blasfeo_unpack_dmat(n, n, &sC, 0, 0, C, n);

				free(mem);
#endif

				}

			tmp_time_blas_api = blasfeo_toc(&timer) / nrep;
			time_blas_api = tmp_time_blas_api<time_blas_api ? tmp_time_blas_api : time_blas_api;
			// BENCHMARK_BLASFEO_END

			}
#endif

//		d_print_mat(n, n, C, ldc);
//		d_print_mat(n, n, D, ldd);





#if 1
		/* call blasfeo */
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{
				
//				blasfeo_dgemm_nn(n, n, n, alpha, &sA, 0, 0, &sB, 0, 0, beta, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dgemm_nt(n, n, n, alpha, &sA, 0, 0, &sB, 0, 0, beta, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dgemm_tn(n, n, n, alpha, &sA, 0, 0, &sB, 0, 0, beta, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dgemm_tt(n, n, n, alpha, &sA, 0, 0, &sB, 0, 0, beta, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dsyrk_ln(n, n, alpha, &sA, 0, 0, &sB, 0, 0, beta, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dpotrf_l(n, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dtrsm_rltn(n, n, alpha, &sB, 0, 0, &sC, 0, 0, &sC, 0, 0);
//				blasfeo_dtrmm_rlnn(n, n, alpha, &sB, 0, 0, &sC, 0, 0, &sC, 0, 0);

				}

			tmp_time_blasfeo = blasfeo_toc(&timer) / nrep;
			time_blasfeo = tmp_time_blasfeo<time_blasfeo ? tmp_time_blasfeo : time_blasfeo;
			// BENCHMARK_BLASFEO_END

			}
#endif

//		d_print_mat(n, n, C, ldc);
//		blasfeo_print_dmat(n, n, &sC, 0, 0);



		double Gflops_max = flops_max * GHz_max;

		double flop_operation = 2.0*n*n*n; // gemm
//		double flop_operation = 1.0*n*n*n; // syrk trsm
//		double flop_operation = 1.0/3.0*n*n*n; // potrf
//		double flop_operation = 2.0/3.0*n*n*n; // getrf

		double Gflops_blas      = 1e-9*flop_operation/time_blas;
		double Gflops_blas_api  = 1e-9*flop_operation/time_blas_api;
		double Gflops_blasfeo   = 1e-9*flop_operation/time_blasfeo;

		printf("%d\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\n",
			n,
			Gflops_blas_api, 100.0*Gflops_blas_api/Gflops_max,
			Gflops_blas, 100.0*Gflops_blas/Gflops_max,
			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max);
//		fprintf(f, "%d\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\n",
//			n,
//			Gflops_blas_api, 100.0*Gflops_blas_api/Gflops_max,
//			Gflops_blas, 100.0*Gflops_blas/Gflops_max,
//			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max);

		d_free_align(A);
		d_free_align(B);
		d_free_align(C);
		d_free_align(D);
		free(ipiv);
		blasfeo_free_dmat(&sA);
		blasfeo_free_dmat(&sB);
		blasfeo_free_dmat(&sC);
		blasfeo_free_dmat(&sD);

		}

	printf("\n");
//	fprintf(f, "];\n");

	fclose(f);

	return 0;

	}
