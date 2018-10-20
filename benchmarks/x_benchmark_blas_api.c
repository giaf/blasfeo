/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/


#if defined(BENCHMARKS_MODE)

#include <stdlib.h>
#include <stdio.h>

//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
//#endif



#include "../include/blasfeo.h"
#include "benchmark_x_common.h"



#if defined(REF_BLAS_NETLIB)
//#include "cblas.h"
//#include "lapacke.h"
#include "../include/d_blas.h"
#endif

#if defined(REF_BLAS_OPENBLAS)
void openblas_set_num_threads(int num_threads);
//#include "cblas.h"
//#include "lapacke.h"
#include "../include/d_blas.h"
#endif

#if defined(REF_BLAS_BLIS)
//void omp_set_num_threads(int num_threads);
#include "blis.h"
#endif

#if defined(REF_BLAS_MKL)
#include "mkl.h"
#endif






int main()
	{

#if defined(REF_BLAS_OPENBLAS)
	openblas_set_num_threads(1);
#endif
#if defined(REF_BLAS_BLIS)
//	omp_set_num_threads(1);
#endif
#if defined(REF_BLAS_MKL)
	mkl_set_num_threads(1);
#endif

//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
//#endif

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;

#if defined(DOUBLE_PRECISION)

	// maximum flops per cycle, double precision
#if defined(TARGET_X64_INTEL_HASWELL)
	const double flops_max = 16;
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const double flops_max = 8;
#elif defined(TARGET_X64_INTEL_CORE)
	const double flops_max = 4;
#elif defined(TARGET_X64_AMD_BULLDOZER)
	const double flops_max = 8;
#elif defined(TARGET_X86_AMD_JAGUAR)
	const double flops_max = 2;
#elif defined(TARGET_X86_AMD_BARCELONA)
	const double flops_max = 4;
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const double flops_max = 4;
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	const double flops_max = 4;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	const double flops_max = 0.5;
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const double flops_max = 2;
#elif defined(TARGET_GENERIC)
	const double flops_max = 2;
#else
#error wrong target
#endif

#elif defined(SINGLE_PRECISION)

#if defined(TARGET_X64_INTEL_HASWELL)
	const double flops_max = 32; // 2x256 bit fma
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const double flops_max = 16; // 1x256 bit mul + 1x256 bit add
#elif defined(TARGET_X64_INTEL_CORE)
	const double flops_max = 8; // 1x128 bit mul + 1x128 bit add
#elif defined(TARGET_X64_AMD_BULLDOZER)
	const double flops_max = 16; // 2x128 bit fma
#elif defined(TARGET_X86_AMD_JAGUAR)
	const double flops_max = 8; // 1x128 bit mul + 1x128 bit add
#elif defined(TARGET_X86_AMD_BARCELONA)
	const double flops_max = 8; // 1x128 bit mul + 1x128 bit add
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const double flops_max = 8; // 1x128 bit fma
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	const double flops_max = 8; // 1x128 bit fma
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	const double flops_max = 2; // 1x32 bit fma
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const double flops_max = 8; // 1x128 bit fma
#elif defined(TARGET_GENERIC)
	const double flops_max = 2; // 1x32 bit mul + 1x32 bit add ???
#endif

#else

#error wrong precision

#endif



//	FILE *f;
//	f = fopen("./test_problems/results/test_blas.m", "w"); // a

	printf("A = [%f %f];\n", GHz_max, flops_max);
//	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
	printf("\n");
//	fprintf(f, "\n");

	printf("B = [\n");
//	fprintf(f, "B = [\n");



	int ii, jj, ll;
	int rep;

	int nrep_in = 10; // number of benchmark batches

#if 1
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};

//	for(ll=0; ll<24; ll++)
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll]/nrep_in;
		nrep = nrep>1 ? nrep : 1;
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<12 ? 12 : n;
//		n = n<8 ? 8 : n;

#elif 1
	int nn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	for(ll=0; ll<24; ll++)

		{

		int n = nn[ll];
		int nrep = 40000; //nnrep[ll];
#else
// TODO  ll<1 !!!!!

	for(ll=0; ll<1; ll++)

		{

		int n = 24;
		int nrep = 40000; //nnrep[ll];
#endif

		int rep_in;

#if defined(DOUBLE_PRECISION)
		double *A; d_zeros_align(&A, n, n); // = malloc(n*n*sizeof(double));
		double *B; d_zeros_align(&B, n, n); // = malloc(n*n*sizeof(double));
		double *D; d_zeros_align(&D, n, n); // = malloc(n*n*sizeof(double));
#elif defined(SINGLE_PRECISION)
		float *A; s_zeros_align(&A, n, n); // = malloc(n*n*sizeof(float));
		float *B; s_zeros_align(&B, n, n); // = malloc(n*n*sizeof(float));
		float *D; s_zeros_align(&D, n, n); // = malloc(n*n*sizeof(float));
#endif

		// A
		for(ii=0; ii<n*n; ii++)
			A[ii] = ii;

		// B
		for(ii=0; ii<n*n; ii++)
			B[ii] = 0.0;
		for(ii=0; ii<n; ii++)
			B[ii+n*ii] = 1.0;


		int info;

#if defined(DOUBLE_PRECISION)
		double r_1 = 1.0;
		double r_0 = 0.0;
#elif defined(SINGLE_PRECISION)
		float r_1 = 1.0;
		float r_0 = 0.0;
#endif
		char c_l = 'l';
		char c_n = 'n';
		char c_t = 't';
		char c_u = 'u';

		/* timing */
		blasfeo_timer timer;

		double time_blasfeo  = 1e15;
		double time_blas     = 1e15;
		double tmp_time_blasfeo;
		double tmp_time_blas;

		/* benchmarks */

		// batches repetion, find minimum averaged time
		// discard batch interrupted by the scheduler
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{
#if defined(DOUBLE_PRECISION)

#if defined(GEMM_NN)
				blasfeo_dgemm(&c_n, &c_n, &n, &n, &n, &r_1, A, &n, B, &n, &r_0, D, &n);
#elif defined(GEMM_NT)
				blasfeo_dgemm(&c_n, &c_t, &n, &n, &n, &r_1, A, &n, B, &n, &r_0, D, &n);
#elif defined(GEMM_TN)
				blasfeo_dgemm(&c_t, &c_n, &n, &n, &n, &r_1, A, &n, B, &n, &r_0, D, &n);
#elif defined(GEMM_TT)
				blasfeo_dgemm(&c_t, &c_t, &n, &n, &n, &r_1, A, &n, B, &n, &r_0, D, &n);
#elif defined(SYRK_LN)
#elif defined(TRMM_RLNN)
#elif defined(TRMM_RUTN)
#elif defined(TRSM_LUNN)
#elif defined(TRSM_LLNU)
#elif defined(TRSM_RLTN)
#elif defined(TRSM_RLTU)
#elif defined(TRSM_RUTN)
#elif defined(GELQF)
#elif defined(GEQRF)
#elif defined(GETRF_NOPIVOT)
#elif defined(GETRF_ROWPIVOT)
#elif defined(POTRF_L)
				blasfeo_dpotrf(&c_l, &n, B, &n, &info);
#elif defined(POTRF_U)
				blasfeo_dpotrf(&c_u, &n, B, &n, &info);
#elif defined(GEMV_N)
#elif defined(GEMV_T)
#elif defined(TRMV_LNN)
#elif defined(TRMV_LTN)
#elif defined(TRSV_LNN)
#elif defined(TRSV_LTN)
#elif defined(GEMV_NT)
#elif defined(SYMV_L)
#else
#error wrong routine
#endif

#elif defined(SINGLE_PRECISION)

#if defined(GEMM_NN)
#elif defined(GEMM_NT)
#elif defined(GEMM_TN)
#elif defined(GEMM_TT)
#elif defined(SYRK_LN)
#elif defined(TRMM_RLNN)
#elif defined(TRMM_RUTN)
#elif defined(TRSM_LUNN)
#elif defined(TRSM_LLNU)
#elif defined(TRSM_RLTN)
#elif defined(TRSM_RLTU)
#elif defined(TRSM_RUTN)
#elif defined(GELQF)
#elif defined(GEQRF)
#elif defined(GETRF_NOPIVOT)
#elif defined(GETRF_ROWPIVOT)
#elif defined(POTRF_L)
#elif defined(POTRF_U)
#elif defined(GEMV_N)
#elif defined(GEMV_T)
#elif defined(TRMV_LNN)
#elif defined(TRMV_LTN)
#elif defined(TRSV_LNN)
#elif defined(TRSV_LTN)
#elif defined(GEMV_NT)
#elif defined(SYMV_L)
#else
#error wrong routine
#endif

#endif

				}

			tmp_time_blasfeo = blasfeo_toc(&timer) / nrep;
			time_blasfeo = tmp_time_blasfeo<time_blasfeo ? tmp_time_blasfeo : time_blasfeo;
			// BENCHMARK_BLASFEO_END

			}

		double Gflops_max = flops_max * GHz_max;

#if defined(GEMM_NN) | defined(GEMM_NT) | defined(GEMM_TN) | defined(GEMM_TT)
		double flop_operation = 2.0*n*n*n;
#elif defined(SYRK_LN) | defined(TRMM_RLNN) | defined(TRMM_RUTN) | defined(TRSM_LLNU) | defined(TRSM_LUNN) | defined(TRSM_RLTN) | defined(TRSM_RLTU) | defined(TRSM_RUTN)
		double flop_operation = 1.0*n*n*n;
#elif defined(GELQF) | defined(GEQRF)
		double flop_operation = 4.0/3.0*n*n*n;
#elif defined(GETRF_NOPIVOT) | defined(GETRF_ROWPIVOT)
		double flop_operation = 2.0/3.0*n*n*n;
#elif defined(POTRF_L) | defined(POTRF_U)
		double flop_operation = 1.0/3.0*n*n*n;
#elif defined(GEMV_NT)
		double flop_operation = 4.0*n*n;
#elif defined(GEMV_N) | defined(GEMV_T) | defined(SYMV_L)
		double flop_operation = 2.0*n*n;
#elif defined(TRMV_LNN) | defined(TRMV_LTN) | defined(TRSV_LNN) | defined(TRSV_LTN)
		double flop_operation = 1.0*n*n;
#else
#error wrong routine
#endif

		double Gflops_blasfeo  = 1e-9*flop_operation/time_blasfeo;

		printf("%d\t%7.3f\t%7.3f\n",
			n,
			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max);

//		free(A);
//		free(B);
//		free(D);
#if defined(DOUBLE_PRECISION)
		d_free_align(A);
		d_free_align(B);
		d_free_align(D);
#elif defined(SINGLE_PRECISION)
		s_free_align(A);
		s_free_align(B);
		s_free_align(D);
#endif

		}

	printf("];\n");

	return 0;

	}

#else

#include <stdio.h>

int main()
	{
	printf("\n\n Recompile BLASFEO with BENCHMARKS_MODE=1 to run this benchmark.\n");
	printf("On CMake use -DBLASFEO_BENCHMARKS=ON .\n\n");
	return 0;
	}

#endif

