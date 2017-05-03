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

	int n = 12;

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

	//
	// tests
	//

	float alpha = 1.0;
	float beta = 0.0;
//	kernel_sgemm_nt_24x4_lib8(4, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_16x4_lib8(4, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	kernel_sgemm_nt_8x8_lib8(5, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//	kernel_sgemm_nt_8x4_lib8(5, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
	kernel_sgemm_nt_4x8_gen_lib8(8, &alpha, sA.pA, sB.pA, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 4, 0, 8);
//	kernel_sgemm_nt_8x4_vs_lib8(8, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA, 7, 4);
//	kernel_sgemm_nt_8x4_lib8(8, &alpha, sB.pA, sA.pA+4, &beta, sA.pA+4*8, sD.pA+4*8);
//	kernel_sgemm_nn_16x4_lib8(4, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//	s_print_strmat(n, n, &sD, 0, 0);
//	return 0;
//	sgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	kernel_ssyrk_nt_l_8x8_lib8(n, &alpha, sA.pA, sA.pA, &beta, sB.pA, sD.pA);
//	sgecp_libstr(16, 16, &sA, 2, 0, &sD, 1, 0);
//	sgetr_libstr(16, 16, &sA, 2, 0, &sD, 2, 0);
	s_print_strmat(n, n, &sD, 0, 0);
	return 0;
//	sgesc_libstr(16, 9, 2.0, &sD, 0, 0);
//	s_print_strmat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(0, sD.pA, sD.pA, sD.pA, sD.pA, sx.pa);
//	s_print_strmat(n, n, &sD, 0, 0);
//	s_print_tran_strvec(n, &sx, 0);
//	kernel_strsm_nt_rl_inv_8x8_lib8(0, sD.pA, sD.pA, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA, sx.pa);
//	s_print_strmat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(8, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA+8*sD.cn+8*8, sD.pA+8*sD.cn+8*8, sx.pa+8);
//	spotrf_l_mn_libstr(23, 17, &sD, 0, 0, &sD, 0, 0);
	spotrf_l_libstr(n, &sD, 0, 0, &sD, 0, 0);
	s_print_strmat(n, n, &sD, 0, 0);



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
