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


#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"

#ifndef S_PS
#define S_PS 1
#endif
#ifndef S_NC
#define S_NC 1
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

	printf("\n");
	printf("\n");
	printf("\n");

	printf("BLAS performance test - float precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit test_param.h to modify this value).\n", GHz_max);
	printf("\n");

	// maximum flops per cycle, single precision
	// maxumum memops (sustained load->store of floats) per cycle, single precision
#if defined(TARGET_X64_INTEL_HASWELL)
	const float flops_max = 32; // 2x256 bit fma
	const float memops_max = 8; // 2x256 bit load + 1x256 bit store
	printf("Testing BLAS version for AVX2 and FMA instruction sets, 64 bit (optimized for Intel Haswell): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const float flops_max = 16; // 1x256 bit mul + 1x256 bit add
	const float memops_max = 4; // 1x256 bit load + 1x128 bit store
	printf("Testing BLAS version for AVX instruction set, 64 bit (optimized for Intel Sandy Bridge): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_CORE)
	const float flops_max = 8; // 1x128 bit mul + 1x128 bit add
	const float memops_max = 4; // 1x128 bit load + 1x128 bit store;
	printf("Testing BLAS version for SSE3 instruction set, 64 bit (optimized for Intel Core): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_AMD_BULLDOZER)
	const float flops_max = 16; // 2x128 bit fma
	const float memops_max = 4; // 1x256 bit load + 1x128 bit store
	printf("Testing BLAS version for SSE3 and FMA instruction set, 64 bit (optimized for AMD Bulldozer): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const float flops_max = 8; // 1x128 bit fma
	const float memops_max = 4; // ???
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A15): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const float flops_max = 8; // 1x128 bit fma
	const float memops_max = 4; // ???
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A15): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_GENERIC)
	const float flops_max = 2; // 1x32 bit mul + 1x32 bit add ???
	const float memops_max = 1; // ???
	printf("Testing BLAS version for generic scalar instruction set: theoretical peak %5.1f Gflops ???\n", flops_max*GHz_max);
#endif
	
//	FILE *f;
//	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_INTEL_HASWELL)
//	fprintf(f, "C = 's_x64_intel_haswell';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	fprintf(f, "C = 's_x64_intel_sandybridge';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_CORE)
//	fprintf(f, "C = 's_x64_intel_core';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_AMD_BULLDOZER)
//	fprintf(f, "C = 's_x64_amd_bulldozer';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
//	fprintf(f, "C = 's_armv7a_arm_cortex_a15';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
//	fprintf(f, "C = 's_armv7a_arm_cortex_a15';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_GENERIC)
//	fprintf(f, "C = 's_generic';\n");
//	fprintf(f, "\n");
#endif

//	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
//	fprintf(f, "\n");

//	fprintf(f, "B = [\n");
	


	int i, j, rep, ll;
	
	const int bss = S_PS;
	const int ncs = S_NC;

/*	int info = 0;*/
	
	printf("\nn\t  sgemm_blasfeo\t  sgemm_blas\n");
	printf("\nn\t Gflops\t    %%\t Gflops\t    %%\n\n");
	
#if 1
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};
	
//	for(ll=0; ll<24; ll++)
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll]/2;
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<16 ? 16 : n;

		int n2 = n*n;

#else
	int nn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	
	for(ll=0; ll<24; ll++)

		{

		int n = nn[ll];
		int nrep = 40000; //nnrep[ll];
#endif

		int rep_in;
		int nrep_in = 10;

		float *A; s_zeros_align(&A, n, n);
		float *B; s_zeros_align(&B, n, n);
		float *C; s_zeros_align(&C, n, n);
		float *M; s_zeros_align(&M, n, n);

		char c_n = 'n';
		char c_l = 'l';
		char c_r = 'r';
		char c_t = 't';
		char c_u = 'u';
		int i_1 = 1;
		int i_t;
		float d_1 = 1;
		float d_0 = 0;
	
		for(i=0; i<n*n; i++)
			A[i] = i;
	
		for(i=0; i<n; i++)
			B[i*(n+1)] = 1;
	
		for(i=0; i<n*n; i++)
			M[i] = 1;
	
		float *B2; s_zeros(&B2, n, n);
		for(i=0; i<n*n; i++)
			B2[i] = 1e-15;
		for(i=0; i<n; i++)
			B2[i*(n+1)] = 1;

		float *x; s_zeros(&x, n, 1);
		float *y; s_zeros(&y, n, 1);
		float *x2; s_zeros(&x2, n, 1);
		float *y2; s_zeros(&y2, n, 1);
		float *diag; s_zeros(&diag, n, 1);
		int *ipiv; int_zeros(&ipiv, n, 1);

//		for(i=0; i<n; i++) x[i] = 1;
//		for(i=0; i<n; i++) x2[i] = 1;

		// matrix struct
#if 0
		struct blasfeo_smat sA; blasfeo_allocate_smat(n+4, n+4, &sA);
		struct blasfeo_smat sB; blasfeo_allocate_smat(n+4, n+4, &sB);
		struct blasfeo_smat sC; blasfeo_allocate_smat(n+4, n+4, &sC);
		struct blasfeo_smat sD; blasfeo_allocate_smat(n+4, n+4, &sD);
		struct blasfeo_smat sE; blasfeo_allocate_smat(n+4, n+4, &sE);
#else
		struct blasfeo_smat sA; blasfeo_allocate_smat(n, n, &sA);
		struct blasfeo_smat sB; blasfeo_allocate_smat(n, n, &sB);
		struct blasfeo_smat sC; blasfeo_allocate_smat(n, n, &sC);
		struct blasfeo_smat sD; blasfeo_allocate_smat(n, n, &sD);
		struct blasfeo_smat sE; blasfeo_allocate_smat(n, n, &sE);
#endif
		struct blasfeo_svec sx; blasfeo_allocate_svec(n, &sx);
		struct blasfeo_svec sy; blasfeo_allocate_svec(n, &sy);
		struct blasfeo_svec sz; blasfeo_allocate_svec(n, &sz);

		blasfeo_pack_smat(n, n, A, n, &sA, 0, 0);
		blasfeo_pack_smat(n, n, B, n, &sB, 0, 0);
		blasfeo_pack_svec(n, x, &sx, 0);


		// create matrix to pivot all the time
//		blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);

		float *dummy;

		int info;

		/* timing */
		struct timeval tvm1, tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13, tv14, tv15, tv16;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
			}

		float alpha = 1.0;
		float beta = 0.0;

		float time_hpmpc    = 1e15;
		float time_blasfeo  = 1e15;
		float time_blas     = 1e15;

		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			gettimeofday(&tv0, NULL); // stop

			for(rep=0; rep<nrep; rep++)
				{
	//			kernel_sgemm_nt_24x4_lib8(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
	//			kernel_sgemm_nt_16x4_lib8(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
	//			kernel_sgemm_nt_8x8_lib8(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
	//			kernel_sgemm_nt_8x4_lib8(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
	//			kernel_sgemm_nt_4x8_gen_lib8(n, &alpha, sA.pA, sB.pA, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 4, 0, 8);
	//			kernel_sgemm_nt_4x8_vs_lib8(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA, 4, 8);
	//			kernel_sgemm_nt_4x8_lib8(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
	//			kernel_sgemm_nt_12x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
	//			kernel_sgemm_nt_8x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
	//			kernel_sgemm_nt_4x4_lib4(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
	//			kernel_sgemm_nn_16x4_lib8(n, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
	//			kernel_sgemm_nn_8x8_lib8(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
	//			kernel_sgemm_nn_8x4_lib8(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);

//				blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//				blasfeo_sgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//				blasfeo_ssyrk_ln(n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
	//			blasfeo_spotrf_l_mn(n, n, &sB, 0, 0, &sB, 0, 0);
				blasfeo_spotrf_l(n, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_sgetr(n, n, &sA, 0, 0, &sB, 0, 0);
	//			blasfeo_sgetrf_nopivot(n, n, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_sgetrf_rowpivot(n, n, &sB, 0, 0, &sB, 0, 0, ipiv);
	//			blasfeo_strmm_rlnn(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
	//			blasfeo_strmm_rutn(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
	//			blasfeo_strsm_llnu(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_strsm_lunn(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_strsm_rltn(n, n, 1.0, &sB, 0, 0, &sD, 0, 0, &sD, 0, 0);
	//			blasfeo_strsm_rltu(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_strsm_rutn(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
	//			blasfeo_sgemv_n(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
	//			blasfeo_sgemv_t(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
	//			blasfeo_ssymv_l(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
	//			blasfeo_sgemv_nt(n, n, 1.0, 1.0, &sA, 0, 0, &sx, 0, &sx, 0, 0.0, 0.0, &sy, 0, &sy, 0, &sz, 0, &sz, 0);
				}

	//		blasfeo_print_dmat(n, n, &sD, 0, 0);

			gettimeofday(&tv1, NULL); // stop

			for(rep=0; rep<nrep; rep++)
				{
	#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_NETLIB) || defined(REF_BLAS_MKL)
	//			sgemm_(&c_n, &c_t, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
	//			sgemm_(&c_n, &c_n, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
	//			scopy_(&n2, A, &i_1, B, &i_1);
	//			ssyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_0, C, &n);
	//			strmm_(&c_r, &c_u, &c_t, &c_n, &n, &n, &d_1, A, &n, C, &n);
	//			spotrf_(&c_l, &n, B2, &n, &info);
	//			sgetrf_(&n, &n, B2, &n, ipiv, &info);
	//			strsm_(&c_l, &c_l, &c_n, &c_u, &n, &n, &d_1, B2, &n, B, &n);
	//			strsm_(&c_l, &c_u, &c_n, &c_n, &n, &n, &d_1, B2, &n, B, &n);
	//			strtri_(&c_l, &c_n, &n, B2, &n, &info);
	//			slauum_(&c_l, &n, B, &n, &info);
	//			sgemv_(&c_n, &n, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);
	//			sgemv_(&c_t, &n, &n, &d_1, A, &n, x2, &i_1, &d_0, y2, &i_1);
	//			strmv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
	//			strsv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
	//			ssymv_(&c_l, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);

	//			for(i=0; i<n; i++)
	//				{
	//				i_t = n-i;
	//				scopy_(&i_t, &B[i*(n+1)], &i_1, &C[i*(n+1)], &i_1);
	//				}
	//			ssyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_1, C, &n);
	//			spotrf_(&c_l, &n, C, &n, &info);

	#endif

	#if defined(REF_BLAS_BLIS)
	//			sgemm_(&c_n, &c_t, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
	//			sgemm_(&c_n, &c_n, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
	//			ssyrk_(&c_l, &c_n, &n77, &n77, &d_1, A, &n77, &d_0, C, &n77);
	//			strmm_(&c_r, &c_u, &c_t, &c_n, &n77, &n77, &d_1, A, &n77, C, &n77);
	//			spotrf_(&c_l, &n77, B, &n77, &info);
	//			strtri_(&c_l, &c_n, &n77, B, &n77, &info);
	//			slauum_(&c_l, &n77, B, &n77, &info);
	#endif
				}

			gettimeofday(&tv2, NULL); // stop

			float tmp_blasfeo  = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
			float tmp_blas     = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);

			time_blasfeo = tmp_blasfeo<time_blasfeo ? tmp_blasfeo : time_blasfeo;
			time_blas = tmp_blas<time_blas ? tmp_blas : time_blas;

			}

		// flops
		if(1)
			{

			float Gflops_max = flops_max * GHz_max;

//			float flop_operation = 6*16.0*2*n; // kernel 24x4
//			float flop_operation = 4*16.0*2*n; // kernel 16x4
//			float flop_operation = 3*16.0*2*n; // kernel 12x4
//			float flop_operation = 2*16.0*2*n; // kernel 8x4
//			float flop_operation = 1*16.0*2*n; // kernel 4x4

//			float flop_operation = 2.0*n*n*n; // dgemm
//			float flop_operation = 1.0*n*n*n; // dsyrk dtrmm dtrsm
			float flop_operation = 1.0/3.0*n*n*n; // dpotrf dtrtri
//			float flop_operation = 2.0/3.0*n*n*n; // dgetrf
//			float flop_operation = 2.0*n*n; // dgemv dsymv
//			float flop_operation = 1.0*n*n; // dtrmv dtrsv
//			float flop_operation = 4.0*n*n; // dgemv_nt
//			float flop_operation = 3*16.0*2*n; // kernel 12x4

//			float flop_operation = 4.0/3.0*n*n*n; // dsyrk+dpotrf

			float Gflops_blasfeo  = 1e-9*flop_operation/time_blasfeo;
			float Gflops_blas     = 1e-9*flop_operation/time_blas;


			printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
//			fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);

			}
		// memops
		else
			{

			float Gmemops_max = memops_max * GHz_max;

			float memop_operation = 1.0*n*n; // dgecp

			float time_hpmpc    = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
			float time_blasfeo  = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
			float time_blas     = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

			float Gmemops_hpmpc    = 1e-9*memop_operation/time_hpmpc;
			float Gmemops_blasfeo  = 1e-9*memop_operation/time_blasfeo;
			float Gmemops_blas     = 1e-9*memop_operation/time_blas;


			printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gmemops_blasfeo, 100.0*Gmemops_blasfeo/Gmemops_max, Gmemops_blas, 100.0*Gmemops_blas/Gmemops_max);
//			fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gmemops_blasfeo, 100.0*Gmemops_blasfeo/Gmemops_max, Gmemops_blas, 100.0*Gmemops_blas/Gmemops_max);

			}


		free(A);
		free(B);
		free(B2);
		free(M);
		free(x);
		free(y);
		free(x2);
		free(y2);
		free(ipiv);
		
		blasfeo_free_smat(&sA);
		blasfeo_free_smat(&sB);
		blasfeo_free_smat(&sC);
		blasfeo_free_smat(&sD);
		blasfeo_free_smat(&sE);
		blasfeo_free_svec(&sx);
		blasfeo_free_svec(&sy);
		blasfeo_free_svec(&sz);

		}

	printf("\n");

//	fprintf(f, "];\n");
//	fclose(f);

	return 0;
	
	}

