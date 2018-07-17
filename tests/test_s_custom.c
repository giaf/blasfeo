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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"

//#include "test_s_common.h"
//#include "test_x_common.c"

int main()
	{
//	print_compilation_flags();

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

	struct blasfeo_smat sA;
	blasfeo_allocate_smat(n, n, &sA);
	blasfeo_pack_smat(n, n, A, n, &sA, 0, 0);
	blasfeo_print_smat(n, n, &sA, 0, 0);

	struct blasfeo_smat sB;
	blasfeo_allocate_smat(n, n, &sB);
	blasfeo_pack_smat(n, n, B, n, &sB, 0, 0);
	blasfeo_print_smat(n, n, &sB, 0, 0);

	struct blasfeo_smat sD;
	blasfeo_allocate_smat(n, n, &sD);
	blasfeo_pack_smat(n, n, D, n, &sD, 0, 0);

	struct blasfeo_svec sx;
	blasfeo_allocate_svec(n, &sx);
	sx.pa[7] = 1.0;
	blasfeo_print_tran_svec(n, &sx, 0);

	struct blasfeo_svec sz0;
	blasfeo_allocate_svec(n, &sz0);

	struct blasfeo_svec sz1;
	blasfeo_allocate_svec(n, &sz1);

	//
	// tests
	//

	// copy scale
#if 0
	blasfeo_print_smat(n, n, &sA, 0, 0);
	blasfeo_sgecpsc(10, 10, 0.1, &sA, 0, 0, &sD, 0, 0);
	blasfeo_print_smat(n, n, &sD, 0, 0);
	return 0;
#endif

	float alpha = 1.0;
	float beta = 0.0;
	
//	kernel_sgemm_nt_4x4_lib4(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//	kernel_sgemm_nn_4x4_lib4(n, &alpha, sA.pA, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//	blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
	blasfeo_sgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
	blasfeo_print_smat(n, n, &sD, 0, 0);
	return 0;

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
//	blasfeo_print_smat(n, n, &sD, 0, 0);
//	return 0;
	blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
//	blasfeo_ssyrk_ln(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	blasfeo_ssyrk_ln_mn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sB, 0, 0, &sD, 0, 0);
//	kernel_ssyrk_nt_l_8x8_lib8(n, &alpha, sA.pA, sA.pA, &beta, sB.pA, sD.pA);
//	blasfeo_sgecp(16, 16, &sA, 2, 0, &sD, 1, 0);
//	blasfeo_sgetr(16, 16, &sA, 2, 0, &sD, 2, 0);
//	blasfeo_print_smat(n, n, &sD, 0, 0);
//	blasfeo_sgemv_n(6, 6, 1.0, &sA, 1, 0, &sx, 0, 0.0, &sz0, 0, &sz0, 0);
//	blasfeo_sgemv_t(11, 8, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sz0, 0, &sz0, 0);
//	blasfeo_strmv_lnn(6, 6, &sA, 1, 0, &sx, 0, &sz0, 0);
//	blasfeo_strmv_ltn(10, 10, &sA, 1, 0, &sx, 0, &sz0, 0);
//	sA.pA[0] = 1.0;
//	blasfeo_strsv_lnn(10, &sA, 0, 0, &sx, 0, &sz0, 0);
//	for(ii=0; ii<8; ii++) sA.dA[ii] = 1.0/blasfeo_sgeex1(&sA, ii, ii);
//	kernel_strsv_lt_inv_8_lib8(0, sA.pA, sA.cn, sA.dA, sx.pa, sx.pa, sz0.pa);
//	kernel_strsv_lt_inv_8_vs_lib8(0, sA.pA, sA.cn, sA.dA, sx.pa, sx.pa, sz0.pa, 3);
//	blasfeo_print_smat(n, n, &sA, 0, 0);
//	blasfeo_strsv_ltn(12, &sA, 0, 0, &sx, 0, &sz0, 0);
//	blasfeo_strsv_ltn_mn(11, 3, &sA, 0, 0, &sx, 0, &sz0, 0);
//	blasfeo_print_smat(n, n, &sA, 0, 0);
//	kernel_sgemv_nt_4_lib8(n, &alpha, &alpha, sA.pA, sA.cn, sx.pa, sx.pa, &beta, sz1.pa, sz0.pa, sz1.pa);
//	kernel_sgemv_nt_4_vs_lib8(n, &alpha, &alpha, sA.pA, sA.cn, sx.pa, sx.pa, &beta, sz1.pa, sz0.pa, sz1.pa, 3);
//	blasfeo_sgemv_nt(5, 2, alpha, alpha, &sA, 0, 0, &sx, 0, &sx, 0, beta, beta, &sz0, 0, &sz1, 0, &sz0, 0, &sz1, 0);
//	blasfeo_ssymv_l(10, 10, alpha, &sA, 1, 0, &sx, 0, beta, &sz0, 0, &sz1, 0);
//	blasfeo_print_tran_svec(n, &sz0, 0);
//	blasfeo_print_tran_svec(n, &sz1, 0);
//	return 0;
//	blasfeo_sgesc(16, 9, 2.0, &sD, 0, 0);
//	blasfeo_print_smat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(0, sD.pA, sD.pA, sD.pA, sD.pA, sx.pa);
//	blasfeo_print_smat(n, n, &sD, 0, 0);
//	blasfeo_print_tran_svec(n, &sx, 0);
//	kernel_strsm_nt_rl_inv_8x8_lib8(0, sD.pA, sD.pA, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA, sx.pa);
//	blasfeo_print_smat(n, n, &sD, 0, 0);
//	kernel_spotrf_nt_l_8x8_lib8(8, sD.pA+8*sD.cn, sD.pA+8*sD.cn, sD.pA+8*sD.cn+8*8, sD.pA+8*sD.cn+8*8, sx.pa+8);
//	blasfeo_spotrf_l_mn(23, 17, &sD, 0, 0, &sD, 0, 0);
	blasfeo_spotrf_l(n, &sD, 0, 0, &sD, 0, 0);
//	kernel_strmm_nn_rl_8x4_lib8(3, &alpha, sB.pA, 7, sA.pA, sA.cn, sD.pA);
//	blasfeo_strmm_rlnn(16, 12, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
	blasfeo_print_smat(n, n, &sD, 0, 0);
	return 0;



	//
	// free memory
	//

	free(A);
	free(B);
	free(D);
	blasfeo_free_smat(&sA);
	blasfeo_free_smat(&sB);
	blasfeo_free_smat(&sD);

	return 0;

	print_compilation_flags();
	}
