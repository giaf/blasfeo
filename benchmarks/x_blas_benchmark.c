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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_d_blas.h"
#include "../include/blasfeo_s_blas.h"
#include "../include/blasfeo_timing.h"



#include "cpu_freq.h"



int main()
	{

#if defined(REF_BLAS_OPENBLAS)
	openblas_set_num_threads(1);
#endif
#if defined(REF_BLAS_BLIS)
	omp_set_num_threads(1);
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
		struct blasfeo_dmat sA; blasfeo_allocate_dmat(n, n, &sA);
		struct blasfeo_dmat sB; blasfeo_allocate_dmat(n, n, &sB);
//		struct blasfeo_dmat sC; blasfeo_allocate_dmat(n, n, &sC);
		struct blasfeo_dmat sD; blasfeo_allocate_dmat(n, n, &sD);
//		struct blasfeo_dmat sE; blasfeo_allocate_dmat(n, n, &sE);

		struct blasfeo_dvec sx; blasfeo_allocate_dvec(n, &sx);
//		struct blasfeo_dvec sy; blasfeo_allocate_dvec(n, &sy);
		struct blasfeo_dvec sz; blasfeo_allocate_dvec(n, &sz);
#elif defined(SINGLE_PRECISION)
		struct blasfeo_smat sA; blasfeo_allocate_smat(n, n, &sA);
		struct blasfeo_smat sB; blasfeo_allocate_smat(n, n, &sB);
//		struct blasfeo_smat sC; blasfeo_allocate_smat(n, n, &sC);
		struct blasfeo_smat sD; blasfeo_allocate_smat(n, n, &sD);
//		struct blasfeo_smat sE; blasfeo_allocate_smat(n, n, &sE);

		struct blasfeo_svec sx; blasfeo_allocate_svec(n, &sx);
//		struct blasfeo_svec sy; blasfeo_allocate_svec(n, &sy);
		struct blasfeo_svec sz; blasfeo_allocate_svec(n, &sz);
#endif

		// A
		for(ii=0; ii<n; ii++)
			for(jj=0; jj<n; jj++)
#if defined(DOUBLE_PRECISION)
				blasfeo_dgein1(ii+n*jj+0.0, &sA, ii, jj);
#elif defined(SINGLE_PRECISION)
				blasfeo_sgein1(ii+n*jj+0.0, &sA, ii, jj);
#endif

		// B
#if defined(DOUBLE_PRECISION)
		blasfeo_dgese(n, n, 0.0, &sB, 0, 0);
#elif defined(SINGLE_PRECISION)
		blasfeo_sgese(n, n, 0.0, &sB, 0, 0);
#endif
		for(ii=0; ii<n; ii++)
#if defined(DOUBLE_PRECISION)
			blasfeo_dgein1(1.0, &sB, ii, ii);
#elif defined(SINGLE_PRECISION)
			blasfeo_sgein1(1.0, &sB, ii, ii);
#endif


		int info;

#if defined(DOUBLE_PRECISION)
		double alpha = 1.0;
		double beta = 0.0;
#elif defined(SINGLE_PRECISION)
		float alpha = 1.0;
		float beta = 0.0;
#endif

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

#if defined(GEMM_NT)
				blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(GEMM_NN)
				blasfeo_dgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(SYRK_LN)
				blasfeo_dsyrk_ln(n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(TRMM_RLNN)
				blasfeo_dtrmm_rlnn(n, n, 1.0, &sB, 0, 0, &sA, 0, 0, &sD, 0, 0);
#elif defined(TRSM_RLTN)
				blasfeo_dtrsm_rltn(n, n, 1.0, &sB, 0, 0, &sA, 0, 0, &sD, 0, 0);
#elif defined(POTRF_L)
				blasfeo_dpotrf_l(n, &sB, 0, 0, &sB, 0, 0);
#elif defined(GEMV_N)
				blasfeo_dgemv_n(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz, 0, &sz, 0);
#elif defined(GEMV_T)
				blasfeo_dgemv_t(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz, 0, &sz, 0);
#else
#error wrong routine
#endif

#elif defined(SINGLE_PRECISION)

#if defined(GEMM_NT)
				blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(GEMM_NN)
				blasfeo_sgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(SYRK_LN)
				blasfeo_ssyrk_ln(n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
#elif defined(TRMM_RLNN)
				blasfeo_strmm_rlnn(n, n, 1.0, &sB, 0, 0, &sA, 0, 0, &sD, 0, 0);
#elif defined(TRSM_RLTN)
				blasfeo_strsm_rltn(n, n, 1.0, &sB, 0, 0, &sA, 0, 0, &sD, 0, 0);
#elif defined(POTRF_L)
				blasfeo_spotrf_l(n, &sB, 0, 0, &sB, 0, 0);
#elif defined(GEMV_N)
				blasfeo_sgemv_n(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz, 0, &sz, 0);
#elif defined(GEMV_T)
				blasfeo_sgemv_t(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz, 0, &sz, 0);
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

#if defined(GEMM_NT) | defined(GEMM_NN)
		double flop_operation = 2.0*n*n*n;
#elif defined(SYRK_LN) | defined(TRMM_RLNN) | defined(TRSM_RLTN)
		double flop_operation = 1.0*n*n*n;
#elif defined(POTRF_L)
		double flop_operation = 1.0/3.0*n*n*n;
#elif defined(GEMV_N) | defined(GEMV_T)
		double flop_operation = 2.0*n*n;
#else
#error wrong routine
#endif

		double Gflops_blasfeo  = 1e-9*flop_operation/time_blasfeo;

		printf("%d\t%7.2f\t%7.2f\n",
			n,
			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max);

#if defined(DOUBLE_PRECISION)
		blasfeo_free_dmat(&sA);
		blasfeo_free_dmat(&sB);
//		blasfeo_free_dmat(&sB2);
//		blasfeo_free_dmat(&sB3);
//		blasfeo_free_dmat(&sC);
		blasfeo_free_dmat(&sD);
//		blasfeo_free_dmat(&sE);
		blasfeo_free_dvec(&sx);
//		blasfeo_free_dvec(&sy);
		blasfeo_free_dvec(&sz);
#elif defined(SINGLE_PRECISION)
		blasfeo_free_smat(&sA);
		blasfeo_free_smat(&sB);
//		blasfeo_free_smat(&sB2);
//		blasfeo_free_smat(&sB3);
//		blasfeo_free_smat(&sC);
		blasfeo_free_smat(&sD);
//		blasfeo_free_smat(&sE);
		blasfeo_free_svec(&sx);
//		blasfeo_free_svec(&sy);
		blasfeo_free_svec(&sz);
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
