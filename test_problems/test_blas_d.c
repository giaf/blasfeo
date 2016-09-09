/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_blas.h"



#if defined(REF_BLAS_OPENBLAS)
#include <f77blas.h>
void openblas_set_num_threads(int n_thread);
#endif
#if defined(REF_BLAS_BLIS)
#include <blis/blis.h>
#endif
#if defined(REF_BLAS_NETLIB)
#include "../reference_code/blas.h"
#endif



#define GHZ_MAX 3.6



int main()
	{
		
#if defined(REF_BLAS_OPENBLAS)
	openblas_set_num_threads(1);
#endif
#if defined(REF_BLAS_BLIS)
	omp_set_num_threads(1);
#endif

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
	printf("Testing BLAS version for AVX2 and FMA instruction sets, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const float flops_max = 8;
	printf("Testing BLAS version for AVX instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_CORE)
	const float flops_max = 4;
	printf("Testing BLAS version for AVX instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#endif
	
	FILE *f;
	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_INTEL_HASWELL)
	fprintf(f, "C = 'd_x64_haswell';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	fprintf(f, "C = 'd_x64_sandybridge';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_CORE)
	fprintf(f, "C = 'd_x64_core';\n");
	fprintf(f, "\n");
#endif

	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
	fprintf(f, "\n");

	fprintf(f, "B = [\n");
	


	int i, j, rep, ll;
	
	const int bsd = D_BS;
	const int ncd = D_NC;

/*	int info = 0;*/
	
	printf("\nn\t  kernel_dgemm\t  dgemm\t\t  dsyrk_dpotrf\t  dtrmm\t\t  dtrtr\t\t  dgemv_n\t  dgemv_t\t  dtrmv_n\t  dtrmv_t\t  dtrsv_n\t  dtrsv_t\t  dsymv\t\t  dgemv_nt\t\t  dsyrk+dpotrf\t  BLAS dgemm\t  BLAS dgemv_n\t  BLAS dgemv_t\n");
	printf("\nn\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\t Gflops\t    %%\n\n");
	
#if 1
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};
	
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll];

#else
	int nn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	
	for(ll=0; ll<24; ll++)

		{

		int n = nn[ll];
		int nrep = 40000; //nnrep[ll];
#endif

#if defined(LOW_RANK)
		int m = LOW_RANK;
#endif

#if defined(REF_BLAS_BLIS)
		f77_int n77 = n;
#endif
	
		double *A; d_zeros(&A, n, n);
		double *B; d_zeros(&B, n, n);
		double *C; d_zeros(&C, n, n);
		double *M; d_zeros(&M, n, n);

#if defined(LOW_RANK)
		double *Al; d_zeros(&Al, m, n);
		double *Cl; d_zeros(&Cl, m, m);
		for(i=0; i<m*n; i++)
			Al[i] = i;
#endif

		char c_n = 'n';
		char c_l = 'l';
		char c_r = 'r';
		char c_t = 't';
		char c_u = 'u';
		int i_1 = 1;
		int i_t;
#if defined(REF_BLAS_BLIS)
		f77_int i77_1 = i_1;
#endif
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
		int pad = (ncd-n%ncd)%ncd;

		double *pA; d_zeros_align(&pA, pnd, cnd);
		double *pB; d_zeros_align(&pB, pnd, cnd);
		double *pC; d_zeros_align(&pC, pnd, cnd);
		double *pD; d_zeros_align(&pD, pnd, cnd);
		double *pE; d_zeros_align(&pE, pnd, cnd2);
		double *pF; d_zeros_align(&pF, 2*pnd, cnd);
		double *pL; d_zeros_align(&pL, pnd, cnd);
		double *pM; d_zeros_align(&pM, pnd, cnd);
		double *x; d_zeros_align(&x, pnd, 1);
		double *y; d_zeros_align(&y, pnd, 1);
		double *x2; d_zeros_align(&x2, pnd, 1);
		double *y2; d_zeros_align(&y2, pnd, 1);
		double *diag; d_zeros_align(&diag, pnd, 1);
		int *ipiv; i_zeros(&ipiv, n, 1);

#if defined(LOW_RANK)
		int pmd = ((m+bsd-1)/bsd)*bsd;	
		int cmd = ((m+ncd-1)/ncd)*ncd;	
		double *pAl; d_zeros_align(&pAl, pmd, cnd);
		double *pCl; d_zeros_align(&pCl, pmd, cmd);
#endif

	
		d_cvt_mat2pmat(n, n, A, n, 0, pA, cnd);
		d_cvt_mat2pmat(n, n, B, n, 0, pB, cnd);
		d_cvt_mat2pmat(n, n, B, n, 0, pD, cnd);
		d_cvt_mat2pmat(n, n, A, n, 0, pE, cnd2);
		d_cvt_mat2pmat(n, n, M, n, 0, pM, cnd);
/*		d_cvt_mat2pmat(n, n, B, n, 0, pE+n*bsd, pnd);*/
		
/*		d_print_pmat(n, 2*n, bsd, pE, 2*pnd);*/
/*		exit(2);*/
	
		for(i=0; i<pnd*cnd; i++) pC[i] = -1;
		
		for(i=0; i<pnd; i++) x[i] = 1;
		for(i=0; i<pnd; i++) x2[i] = 1;

		double *dummy;

		int info;

		/* timing */
		struct timeval tvm1, tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13, tv14, tv15, tv16;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			dgemm_nt_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 0.0, pC, cnd, pC, cnd);
			}

		gettimeofday(&tv0, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

#if defined(LOW_RANK)
#else
//			dgemm_nt_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 0.0, pC, cnd, pC, cnd);
			dgemm_nn_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 0.0, pC, cnd, pC, cnd);
//			dsyrk_nt_l_lib(n, n, n, 1.0, pA, cnd, pB, cnd, 1.0, pC, cnd, pD, cnd);
//			dtrmm_nt_ru_lib(n, n, pA, cnd, pB, cnd, 0, pC, cnd, pD, cnd);
//			dpotrf_nt_l_lib(n, n, pB, cnd, pD, cnd, diag);
//			dsyrk_dpotrf_nt_l_lib(n, n, n, pA, cnd, pA, cnd, 1, pB, cnd, pD, cnd, diag);
//			dsyrk_nt_l_lib(n, n, n, pA, cnd, pA, cnd, 1, pB, cnd, pD, cnd);
//			dpotrf_nt_l_lib(n, n, pD, cnd, pD, cnd, diag);
#endif
			}
	
		gettimeofday(&tv1, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{
#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_NETLIB) || defined(REF_BLAS_MKL)
#if defined(LOW_RANK)
//			dgemm_(&c_n, &c_t, &m, &m, &n, &d_1, Al, &m, Al, &m, &d_0, Cl, &m);
//			dsyrk_(&c_l, &c_n, &m, &n, &d_1, Al, &m, &d_0, Cl, &m);
#else
			dgemm_(&c_n, &c_t, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
//			dgemm_(&c_n, &c_n, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
//			dsyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_0, C, &n);
//			dtrmm_(&c_r, &c_u, &c_t, &c_n, &n, &n, &d_1, A, &n, C, &n);
//			dpotrf_(&c_l, &n, B2, &n, &info);
//			dgetrf_(&n, &n, B2, &n, ipiv, &info);
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

		gettimeofday(&tv2, NULL); // stop

		float Gflops_max = flops_max * GHz_max;

#if defined(LOW_RANK)
//		float flop_operation = 2.0*m*m*n; // dgemm
		float flop_operation = 1.0*m*m*n; // dsyrk dtrmm
#else
		float flop_operation = 2.0*n*n*n; // dgemm
//		float flop_operation = 1.0*n*n*n; // dsyrk dtrmm
//		float flop_operation = 1.0/3.0*n*n*n; // dpotrf dtrtri
//		float flop_operation = 2.0/3.0*n*n*n; // dgetrf
//		float flop_operation = 2.0*n*n; // dgemv dsymv
//		float flop_operation = 1.0*n*n; // dtrmv dtrsv
//		float flop_operation = 4.0*n*n; // dgemv_nt

//		float flop_operation = 4.0/3.0*n*n*n; // dsyrk+dpotrf
#endif

		float time_hpmpc    = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float time_blas     = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
#ifdef N_CODEGEN
		float time_codegen  = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);
#endif

		float Gflops_hpmpc    = 1e-9*flop_operation/time_hpmpc;
		float Gflops_blas     = 1e-9*flop_operation/time_blas;
#ifdef N_CODEGEN
		float Gflops_codegen  = 1e-9*flop_operation/time_codegen;
#endif


#ifdef N_CODEGEN
		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_hpmpc, 100.0*Gflops_hpmpc/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max, Gflops_codegen, 100.0*Gflops_codegen/Gflops_max);
#else
		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_hpmpc, 100.0*Gflops_hpmpc/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
		fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_hpmpc, 100.0*Gflops_hpmpc/Gflops_max, Gflops_blas, 100.0*Gflops_blas/Gflops_max);
#endif


		free(A);
		free(B);
		free(B2);
		free(M);
		free(pA);
		free(pB);
		free(pC);
		free(pD);
		free(pE);
		free(pF);
		free(pL);
		free(pM);
		free(x);
		free(y);
		free(x2);
		free(y2);
		free(ipiv);
#if defined(LOW_RANK)
		free(Al);
		free(Cl);
		free(pAl);
		free(pCl);
#endif
		
		}

	printf("\n");

	fprintf(f, "];\n");
	fclose(f);

	return 0;
	
	}
