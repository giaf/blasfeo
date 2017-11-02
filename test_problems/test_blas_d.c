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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
//#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"

#ifndef D_PS
#define D_PS 1
#endif
#ifndef D_NC
#define D_NC 1
#endif



#if defined(REF_BLAS_OPENBLAS)
void openblas_set_num_threads(int num_threads);
#endif
#if defined(REF_BLAS_BLIS)
void omp_set_num_threads(int num_threads);
#endif
#if defined(REF_BLAS_MKL)
#include "mkl.h"
#endif



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

	printf("\n");
	printf("\n");
	printf("\n");

	printf("BLAS performance test - double precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit test_param.h to modify this value).\n", GHz_max);
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
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const float flops_max = 4;
	printf("Testing BLAS version for NEONv2 instruction set, 64 bit (optimized for ARM Cortex A57): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const float flops_max = 2;
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A15): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_GENERIC)
	const float flops_max = 2;
	printf("Testing BLAS version for generic scalar instruction set: theoretical peak %5.1f Gflops ???\n", flops_max*GHz_max);
#endif

//	FILE *f;
//	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_INTEL_HASWELL)
//	fprintf(f, "C = 'd_x64_intel_haswell';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	fprintf(f, "C = 'd_x64_intel_sandybridge';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_CORE)
//	fprintf(f, "C = 'd_x64_intel_core';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_AMD_BULLDOZER)
//	fprintf(f, "C = 'd_x64_amd_bulldozer';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
//	fprintf(f, "C = 'd_armv8a_arm_cortex_a57';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
//	fprintf(f, "C = 'd_armv7a_arm_cortex_a15';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_GENERIC)
//	fprintf(f, "C = 'd_generic';\n");
//	fprintf(f, "\n");
#endif

//	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
//	fprintf(f, "\n");

//	fprintf(f, "B = [\n");



	int i, j, rep, ll;

	const int bsd = D_PS;
	const int ncd = D_NC;

/*	int info = 0;*/

	printf("\nn\t  dgemm_blasfeo\t  dgemm_blas\n");
	printf("\nn\t Gflops\t    %%\t Gflops\n\n");

#if 1
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};

//	for(ll=0; ll<24; ll++)
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll];
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<12 ? 12 : n;
//		n = n<8 ? 8 : n;

#else
	int nn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	for(ll=0; ll<24; ll++)

		{

		int n = nn[ll];
		int nrep = 40000; //nnrep[ll];
#endif

		double *A; d_zeros(&A, n, n);
		double *B; d_zeros(&B, n, n);
		double *C; d_zeros(&C, n, n);
		double *M; d_zeros(&M, n, n);

		char c_n = 'n';
		char c_l = 'l';
		char c_r = 'r';
		char c_t = 't';
		char c_u = 'u';
		int i_1 = 1;
		int i_t;
		double d_1 = 1;
		double d_0 = 0;

		for(i=0; i<n*n; i++)
			A[i] = i;

		for(i=0; i<n; i++)
			B[i*(n+1)] = 1;

		for(i=0; i<n*n; i++)
			M[i] = 1;

		int n2 = n*n;
		double *B2; d_zeros(&B2, n, n);
		for(i=0; i<n*n; i++)
			B2[i] = 1e-15;
		for(i=0; i<n; i++)
			B2[i*(n+1)] = 1;

		int pnd = ((n+bsd-1)/bsd)*bsd;
		int cnd = ((n+ncd-1)/ncd)*ncd;
		int cnd2 = 2*((n+ncd-1)/ncd)*ncd;

		double *x; d_zeros_align(&x, pnd, 1);
		double *y; d_zeros_align(&y, pnd, 1);
		double *x2; d_zeros_align(&x2, pnd, 1);
		double *y2; d_zeros_align(&y2, pnd, 1);
		double *diag; d_zeros_align(&diag, pnd, 1);
		int *ipiv; int_zeros(&ipiv, n, 1);

		for(i=0; i<pnd; i++) x[i] = 1;
		for(i=0; i<pnd; i++) x2[i] = 1;

		// matrix struct
#if 0
		struct d_strmat sA; d_allocate_strmat(n+4, n+4, &sA);
		struct d_strmat sB; d_allocate_strmat(n+4, n+4, &sB);
		struct d_strmat sC; d_allocate_strmat(n+4, n+4, &sC);
		struct d_strmat sD; d_allocate_strmat(n+4, n+4, &sD);
		struct d_strmat sE; d_allocate_strmat(n+4, n+4, &sE);
#else
		struct d_strmat sA; d_allocate_strmat(n, n, &sA);
		struct d_strmat sB; d_allocate_strmat(n, n, &sB);
		struct d_strmat sB2; d_allocate_strmat(n, n, &sB2);
		struct d_strmat sB3; d_allocate_strmat(n, n, &sB3);
		struct d_strmat sC; d_allocate_strmat(n, n, &sC);
		struct d_strmat sD; d_allocate_strmat(n, n, &sD);
		struct d_strmat sE; d_allocate_strmat(n, n, &sE);
#endif
		struct d_strvec sx; d_allocate_strvec(n, &sx);
		struct d_strvec sy; d_allocate_strvec(n, &sy);
		struct d_strvec sz; d_allocate_strvec(n, &sz);

		d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
		d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
		d_cvt_mat2strmat(n, n, B2, n, &sB2, 0, 0);
		d_cvt_vec2strvec(n, x, &sx, 0);
		int ii;
		for(ii=0; ii<n; ii++)
			{
			DMATEL_LIBSTR(&sB3, ii, ii) = 1.0;
//			DMATEL_LIBSTR(&sB3, n-1, ii) = 1.0;
			DMATEL_LIBSTR(&sB3, ii, n-1) = 1.0;
			DVECEL_LIBSTR(&sx, ii) = 1.0;
			}
//		d_print_strmat(n, n, &sB3, 0, 0);
//		if(n==20) return;

		int qr_work_size = 0;//dgeqrf_work_size_libstr(n, n);
		void *qr_work;
		v_zeros_align(&qr_work, qr_work_size);

		int lq_work_size = 0;//dgelqf_work_size_libstr(n, n);
		void *lq_work;
		v_zeros_align(&lq_work, lq_work_size);

		// create matrix to pivot all the time
//		dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);

		double *dummy;

		int info;

		/* timing */
		struct timeval tvm1, tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13, tv14, tv15, tv16;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0);
			}

		double alpha = 1.0;
		double beta = 0.0;

		gettimeofday(&tv0, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

//			dgemm_nt_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 0.0, pC, cnd, pC, cnd);
//			dgemm_nn_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 0.0, pC, cnd, pC, cnd);
//			dsyrk_nt_l_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 1.0, pC, cnd, pD, cnd);
//			dtrmm_nt_ru_lib(n, n, pA, cnd, pB, cnd, 0, pC, cnd, pD, cnd);
//			dpotrf_nt_l_lib(n, n, pB, cnd, pD, cnd, diag);
//			dsyrk_dpotrf_nt_l_lib(n, n, n, pA, cnd, pA, cnd, 1, pB, cnd, pD, cnd, diag);
//			dsyrk_nt_l_lib(n, n, n, pA, cnd, pA, cnd, 1, pB, cnd, pD, cnd);
//			dpotrf_nt_l_lib(n, n, pD, cnd, pD, cnd, diag);
//			dgetrf_nn_nopivot_lib(n, n, pB, cnd, pB, cnd, diag); //TODO
//			dgetrf_nn_lib(n, n, pB, cnd, pB, cnd, diag, ipiv); //TODO
//			dtrsm_nn_ll_one_lib(n, n, pD, cnd, pB, cnd, pB, cnd);
//			dtrsm_nn_lu_inv_lib(n, n, pD, cnd, diag, pB, cnd, pB, cnd);
			}

		gettimeofday(&tv1, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{
//			kernel_dgemm_nt_12x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//			kernel_dgemm_nt_8x8_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//			kernel_dsyrk_nt_l_8x8_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//			kernel_dgemm_nt_8x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//			kernel_dgemm_nt_4x8_lib4(n, &alpha, sA.pA, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//			kernel_dgemm_nt_4x4_lib4(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//			kernel_dger4_12_sub_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//			kernel_dger4_sub_12r_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//			kernel_dger4_sub_8r_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//			kernel_dger12_add_4r_lib4(n, sA.pA, sB.pA, sB.cn, sD.pA);
//			kernel_dger8_add_4r_lib4(n, sA.pA, sB.pA, sB.cn, sD.pA);
//			kernel_dger4_sub_4r_lib4(n, sA.pA, sB.pA, sD.pA);
//			kernel_dger2_sub_4r_lib4(n, sA.pA, sB.pA, sD.pA);
//			kernel_dger4_sub_8c_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//			kernel_dger4_sub_4c_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//			kernel_dgemm_nn_4x12_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//			kernel_dgemm_nn_4x8_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//			kernel_dgemm_nn_4x4_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);

			dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
//			dgemm_nn_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
//			dsyrk_ln_libstr(n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
//			dsyrk_ln_mn_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
//			dpotrf_l_mn_libstr(n, n, &sB, 0, 0, &sB, 0, 0);
//			dpotrf_l_libstr(n, &sB, 0, 0, &sB, 0, 0);
//			dgetrf_nopivot_libstr(n, n, &sB, 0, 0, &sB, 0, 0);
//			dgetrf_libstr(n, n, &sB, 0, 0, &sB, 0, 0, ipiv);
//			dgeqrf_libstr(n, n, &sC, 0, 0, &sD, 0, 0, qr_work);
//			dcolin_libstr(n, &sx, 0, &sB3, 0, n-1);
//			dgelqf_libstr(n, n, &sB3, 0, 0, &sB3, 0, 0, lq_work);
//			dtrmm_rlnn_libstr(n, n, 1.0, &sA, 0, 0, &sD, 0, 0, &sD, 0, 0); //
//			dtrmm_rutn_libstr(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
//			dtrsm_llnu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//			dtrsm_lunn_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//			dtrsm_rltn_libstr(n, n, 1.0, &sB2, 0, 0, &sD, 0, 0, &sD, 0, 0); //
//			dtrsm_rltu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//			dtrsm_rutn_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//			dgemv_n_libstr(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//			dgemv_t_libstr(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//			dsymv_l_libstr(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//			dgemv_nt_libstr(n, n, 1.0, 1.0, &sA, 0, 0, &sx, 0, &sx, 0, 0.0, 0.0, &sy, 0, &sy, 0, &sz, 0, &sz, 0);
			}

//		d_print_strmat(n, n, &sD, 0, 0);

		gettimeofday(&tv2, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{
#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_NETLIB) || defined(REF_BLAS_MKL)
			dgemm_(&c_n, &c_t, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
//			dpotrf_(&c_l, &n, B2, &n, &info);
//			dgemm_(&c_n, &c_n, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
//			dsyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_0, C, &n);
//			dtrmm_(&c_r, &c_u, &c_t, &c_n, &n, &n, &d_1, A, &n, C, &n);
//			dgetrf_(&n, &n, B2, &n, ipiv, &info);
//			dtrsm_(&c_l, &c_l, &c_n, &c_u, &n, &n, &d_1, B2, &n, B, &n);
//			dtrsm_(&c_l, &c_u, &c_n, &c_n, &n, &n, &d_1, B2, &n, B, &n);
//			dtrtri_(&c_l, &c_n, &n, B2, &n, &info);
//			dlauum_(&c_l, &n, B, &n, &info);
//			dgemv_(&c_n, &n, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);
//			dgemv_(&c_t, &n, &n, &d_1, A, &n, x2, &i_1, &d_0, y2, &i_1);
//			dtrmv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
//			dtrsv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
//			dsymv_(&c_l, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);

//			for(i=0; i<n; i++)
//				{
//				i_t = n-i;
//				dcopy_(&i_t, &B[i*(n+1)], &i_1, &C[i*(n+1)], &i_1);
//				}
//			dsyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_1, C, &n);
//			dpotrf_(&c_l, &n, C, &n, &info);

#endif

#if defined(REF_BLAS_BLIS)
//			dgemm_(&c_n, &c_t, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
//			dgemm_(&c_n, &c_n, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
//			dsyrk_(&c_l, &c_n, &n77, &n77, &d_1, A, &n77, &d_0, C, &n77);
//			dtrmm_(&c_r, &c_u, &c_t, &c_n, &n77, &n77, &d_1, A, &n77, C, &n77);
//			dpotrf_(&c_l, &n77, B, &n77, &info);
//			dtrtri_(&c_l, &c_n, &n77, B, &n77, &info);
//			dlauum_(&c_l, &n77, B, &n77, &info);
#endif
			}

		gettimeofday(&tv3, NULL); // stop

		float Gflops_max = flops_max * GHz_max;

//		float flop_operation = 4*16.0*2*n; // kernel 12x4
//		float flop_operation = 3*16.0*2*n; // kernel 12x4
//		float flop_operation = 2*16.0*2*n; // kernel 8x4
//		float flop_operation = 1*16.0*2*n; // kernel 4x4
//		float flop_operation = 0.5*16.0*2*n; // kernel 2x4

		float flop_operation = 2.0*n*n*n; // dgemm
//		float flop_operation = 1.0*n*n*n; // dsyrk dtrmm dtrsm
//		float flop_operation = 1.0/3.0*n*n*n; // dpotrf dtrtri
//		float flop_operation = 2.0/3.0*n*n*n; // dgetrf
//		float flop_operation = 4.0/3.0*n*n*n; // dgeqrf
//		float flop_operation = 2.0*n*n; // dgemv dsymv
//		float flop_operation = 1.0*n*n; // dtrmv dtrsv
//		float flop_operation = 4.0*n*n; // dgemv_nt

//		float flop_operation = 4.0/3.0*n*n*n; // dsyrk+dpotrf

		float time_hpmpc    = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float time_blasfeo  = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float time_blas     = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

		float Gflops_hpmpc    = 1e-9*flop_operation/time_hpmpc;
		float Gflops_blasfeo  = 1e-9*flop_operation/time_blasfeo;
		float Gflops_blas     = 1e-9*flop_operation/time_blas;


//		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_hpmpc, 100.0*Gflops_hpmpc/Gflops_max, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
//		fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_hpmpc, 100.0*Gflops_hpmpc/Gflops_max, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
//		fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);


		d_free(A);
		d_free(B);
		d_free(B2);
		d_free(M);
		d_free_align(x);
		d_free_align(y);
		d_free_align(x2);
		d_free_align(y2);
		int_free(ipiv);
		free(qr_work);
		free(lq_work);

		d_free_strmat(&sA);
		d_free_strmat(&sB);
		d_free_strmat(&sB2);
		d_free_strmat(&sB3);
		d_free_strmat(&sC);
		d_free_strmat(&sD);
		d_free_strmat(&sE);
		d_free_strvec(&sx);
		d_free_strvec(&sy);
		d_free_strvec(&sz);

		}

	printf("\n");

//	fprintf(f, "];\n");
//	fclose(f);

	return 0;

	}
