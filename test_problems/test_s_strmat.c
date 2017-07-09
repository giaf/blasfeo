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
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"


int main()
	{

#if defined(LA_HIGH_PERFORMANCE)

	printf("\nLA provided by HIGH_PERFORMANCE\n\n");

#elif defined(LA_REFERENCE)

	printf("\nLA provided by REFERENCE\n\n");

#elif defined(LA_BLAS)

	printf("\nLA provided by BLAS\n\n");

#else

	printf("\nLA provided by ???\n\n");
	exit(2);

#endif

	int ii, jj;

	int n = 16;

	//
	// matrices in column-major format
	//
	float *A; s_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
//	for(jj=0; jj<n; jj++)
//		for(ii=0; ii<jj; ii++)
//			A[ii+n*jj] = 0.0/0.0;
//	s_print_mat(n, n, A, n);

	float *B; s_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	s_print_mat(n, n, B, n);

	float *D; s_zeros(&D, n, n);
	for(ii=0; ii<n*n; ii++) D[ii] = -1.0;
//	s_print_mat(n, n, B, n);


	//
	// matrices in matrix struct format
	//

	struct s_strmat sA;
	s_allocate_strmat(n, n, &sA);
	s_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
	s_print_strmat(n, n, &sA, 0, 0);

	struct s_strmat sB;
	s_allocate_strmat(n, n, &sB);
	s_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	s_print_strmat(n, n, &sB, 0, 0);

	struct s_strmat sD;
	s_allocate_strmat(n, n, &sD);
	s_cvt_mat2strmat(n, n, D, n, &sD, 0, 0);

	struct s_strvec sx;
	s_allocate_strvec(n, &sx);
	sx.pa[7] = 1.0;
	s_print_tran_strvec(n, &sx, 0);

	struct s_strvec sz0;
	s_allocate_strvec(n, &sz0);

	struct s_strvec sz1;
	s_allocate_strvec(n, &sz1);

	//
	// tests
	//

	float alpha = 1.0;
	float beta = 0.0;
//	kernel_sgemm_nt_24x4_lib8(4, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_16x4_lib8(4, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_8x8_lib8(5, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//	kernel_sgemm_nt_8x4_lib8(5, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//	kernel_sgemm_nt_4x8_gen_lib8(8, &alpha, sA.pA, sB.pA, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 4, 0, 8);
//	kernel_sgemm_nt_8x4_vs_lib8(8, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA, 7, 4);
//	kernel_sgemm_nt_8x4_lib8(8, &alpha, sB.pA, sA.pA+4, &beta, sA.pA+4*8, sD.pA+4*8);
//	kernel_sgemm_nn_16x4_lib8(4, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_12x4_lib4(4, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_8x8_lib4(8, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_8x4_lib4(2, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	s_print_strmat(n, n, &sD, 0, 0);
//	return 0;
//	sgemm_nt_libstr(n, n, 5, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	ssyrk_ln_libstr(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	ssyrk_ln_mn_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	kernel_ssyrk_nt_l_8x8_lib8(n, &alpha, sA.pA, sA.pA, &beta, sB.pA, sD.pA);
//	sgecp_libstr(16, 16, &sA, 2, 0, &sD, 1, 0);
//	sgetr_libstr(16, 16, &sA, 2, 0, &sD, 2, 0);
//	s_print_strmat(n, n, &sD, 0, 0);
//	sgemv_n_libstr(6, 6, 1.0, &sA, 1, 0, &sx, 0, 0.0, &sz0, 0, &sz0, 0);
//	sgemv_t_libstr(11, 8, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz0, 0, &sz0, 0);
//	strmv_lnn_libstr(6, 6, &sA, 1, 0, &sx, 0, &sz0, 0);
//	strmv_ltn_libstr(10, 10, &sA, 1, 0, &sx, 0, &sz0, 0);
//	sA.pA[0] = 1.0;
//	strsv_lnn_libstr(10, &sA, 0, 0, &sx, 0, &sz0, 0);
//	for(ii=0; ii<8; ii++) sA.dA[ii] = 1.0/sgeex1_libstr(&sA, ii, ii);
//	kernel_strsv_lt_inv_8_lib8(0, sA.pA, sA.cn, sA.dA, sx.pa, sx.pa, sz0.pa);
//	kernel_strsv_lt_inv_8_vs_lib8(0, sA.pA, sA.cn, sA.dA, sx.pa, sx.pa, sz0.pa, 3);
//	s_print_strmat(n, n, &sA, 0, 0);
//	strsv_ltn_libstr(12, &sA, 0, 0, &sx, 0, &sz0, 0);
//	strsv_ltn_mn_libstr(11, 3, &sA, 0, 0, &sx, 0, &sz0, 0);
//	s_print_strmat(n, n, &sA, 0, 0);
//	kernel_sgemv_nt_4_lib8(n, &alpha, &alpha, sA.pA, sA.cn, sx.pa, sx.pa, &beta, sz1.pa, sz0.pa, sz1.pa);
//	kernel_sgemv_nt_4_vs_lib8(n, &alpha, &alpha, sA.pA, sA.cn, sx.pa, sx.pa, &beta, sz1.pa, sz0.pa, sz1.pa, 3);
//	sgemv_nt_libstr(5, 2, alpha, alpha, &sA, 0, 0, &sx, 0, &sx, 0, beta, beta, &sz0, 0, &sz1, 0, &sz0, 0, &sz1, 0);
//	ssymv_l_libstr(10, 10, alpha, &sA, 1, 0, &sx, 0, beta, &sz0, 0, &sz1, 0);
//	s_print_tran_strvec(n, &sz0, 0);
//	s_print_tran_strvec(n, &sz1, 0);
//	return 0;
//	sgesc_libstr(16, 9, 2.0, &sD, 0, 0);
//	s_print_strmat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(0, sD.pA, sD.pA, sD.pA, sD.pA, sx.pa);
//	s_print_strmat(n, n, &sD, 0, 0);
//	s_print_tran_strvec(n, &sx, 0);
//	kernel_strsm_nt_rl_inv_8x8_lib8(0, sD.pA, sD.pA, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA, sx.pa);
//	s_print_strmat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(8, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA+8*sD.cn+8*8, sD.pA+8*sD.cn+8*8, sx.pa+8);
//	spotrf_l_mn_libstr(23, 17, &sD, 0, 0, &sD, 0, 0);
//	spotrf_l_libstr(n, &sD, 0, 0, &sD, 0, 0);
//	kernel_strmm_nn_rl_8x4_lib8(3, &alpha, sB.pA, 7, sA.pA, sA.cn, sD.pA);
	strmm_rlnn_libstr(12, 8, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
	s_print_strmat(n, n, &sD, 0, 0);
	return 0;



	//
	// free memory
	//

	free(A);
	free(B);
	free(D);
	s_free_strmat(&sA);
	s_free_strmat(&sB);
	s_free_strmat(&sD);

	return 0;
	
	}
