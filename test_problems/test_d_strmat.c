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
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"


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

	int ii;

	int n = 8;

	//
	// matrices in column-major format
	//
	double *A; d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
//	d_print_mat(n, n, A, n);

	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	d_print_mat(n, n, B, n);

	double *C; d_zeros(&C, n, n);

	double *D; d_zeros(&D, n, n);
	for(ii=0; ii<n*n; ii++) D[ii] = -1;

	double *x_n; d_zeros(&x_n, n, 1);
//	for(ii=0; ii<n; ii++) x_n[ii] = 1.0;
	x_n[1] = 1.0;
//	x_n[1] = 1.0;
//	x_n[2] = 2.0;
//	x_n[3] = 3.0;
	double *x_t; d_zeros(&x_t, n, 1);
//	for(ii=0; ii<n; ii++) x_n[ii] = 1.0;
	x_t[0] = 1.0;
	double *y_n; d_zeros(&y_n, n, 1);
	double *y_t; d_zeros(&y_t, n, 1);
	double *z_n; d_zeros(&z_n, n, 1);
	double *z_t; d_zeros(&z_t, n, 1);

	double *x0; d_zeros(&x0, n, 1); x0[0] = 1.0;
	double *x1; d_zeros(&x1, n, 1); x1[1] = 1.0;
	double *x2; d_zeros(&x2, n, 1); x2[2] = 1.0;
	double *x3; d_zeros(&x3, n, 1); x3[3] = 1.0;
	double *x4; d_zeros(&x4, n, 1); x4[4] = 1.0;
	double *x5; d_zeros(&x5, n, 1); x5[5] = 1.0;
	double *x6; d_zeros(&x6, n, 1); x6[6] = 1.0;
	double *x7; d_zeros(&x7, n, 1); x7[7] = 1.0;
//	double *x8; d_zeros(&x8, n, 1); x8[8] = 1.0;
//	double *x9; d_zeros(&x9, n, 1); x9[9] = 1.0;

	int *ipiv; int_zeros(&ipiv, n, 1);
	int *ipiv_tmp; int_zeros(&ipiv_tmp, n, 1);
	int *ipiv_inv; int_zeros(&ipiv_inv, n, 1);
	double d_tmp;

	//
	// matrices in matrix struct format
	//
	int size_strmat = 5*d_size_strmat(n, n);
	void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
	char *ptr_memory_strmat = (char *) memory_strmat;

	struct d_strmat sA;
//	d_allocate_strmat(n, n, &sA);
	d_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
//	d_cast_mat2strmat(A, &sA);
	d_print_strmat(n, n, &sA, 0, 0);

	struct d_strmat sB;
//	d_allocate_strmat(n, n, &sB);
	d_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	d_print_strmat(n, n, &sB, 0, 0);

	struct d_strmat sC;
//	d_allocate_strmat(n, n, &sC);
	d_create_strmat(n, n, &sC, ptr_memory_strmat);
	ptr_memory_strmat += sC.memory_size;

	struct d_strmat sD;
//	d_allocate_strmat(n, n, &sD);
	d_create_strmat(n, n, &sD, ptr_memory_strmat);
	ptr_memory_strmat += sD.memory_size;
	d_cvt_mat2strmat(n, n, D, n, &sD, 0, 0);

	struct d_strmat sE;
//	d_allocate_strmat(n, n, &sE);
	d_create_strmat(n, n, &sE, ptr_memory_strmat);
	ptr_memory_strmat += sE.memory_size;

	struct d_strvec sx_n;
	d_allocate_strvec(n, &sx_n);
	d_cvt_vec2strvec(n, x_n, &sx_n, 0);

	struct d_strvec sx_t;
	d_allocate_strvec(n, &sx_t);
	d_cvt_vec2strvec(n, x_t, &sx_t, 0);

	struct d_strvec sy_n;
	d_allocate_strvec(n, &sy_n);
	d_cvt_vec2strvec(n, y_n, &sy_n, 0);

	struct d_strvec sy_t;
	d_allocate_strvec(n, &sy_t);
	d_cvt_vec2strvec(n, y_t, &sy_t, 0);

	struct d_strvec sz_n;
	d_allocate_strvec(n, &sz_n);
	d_cvt_vec2strvec(n, z_n, &sz_n, 0);

	struct d_strvec sz_t;
	d_allocate_strvec(n, &sz_t);
	d_cvt_vec2strvec(n, z_t, &sz_t, 0);

	struct d_strvec sx0; d_create_strvec(n, &sx0, x0);
	struct d_strvec sx1; d_create_strvec(n, &sx1, x1);
	struct d_strvec sx2; d_create_strvec(n, &sx2, x2);
	struct d_strvec sx3; d_create_strvec(n, &sx3, x3);
	struct d_strvec sx4; d_create_strvec(n, &sx4, x4);
	struct d_strvec sx5; d_create_strvec(n, &sx5, x5);
	struct d_strvec sx6; d_create_strvec(n, &sx6, x6);
	struct d_strvec sx7; d_create_strvec(n, &sx7, x7);
//	struct d_strvec sx8; d_create_strvec(n, &sx8, x8);
//	struct d_strvec sx9; d_create_strvec(n, &sx9, x9);

	struct d_strvec sz0; d_allocate_strvec(n, &sz0);
	struct d_strvec sz1; d_allocate_strvec(n, &sz1);
	struct d_strvec sz2; d_allocate_strvec(n, &sz2);
	struct d_strvec sz3; d_allocate_strvec(n, &sz3);
	struct d_strvec sz4; d_allocate_strvec(n, &sz4);
	struct d_strvec sz5; d_allocate_strvec(n, &sz5);
	struct d_strvec sz6; d_allocate_strvec(n, &sz6);
	struct d_strvec sz7; d_allocate_strvec(n, &sz7);
//	struct d_strvec sz8; d_allocate_strvec(n, &sz8);
//	struct d_strvec sz9; d_allocate_strvec(n, &sz9);

	// tests
	double *v; d_zeros(&v, n, 1);
	double *vp; d_zeros(&vp, n, 1);
	double *vm; d_zeros(&vm, n, 1);
	double *m; d_zeros(&m, n, 1);
	double *r; d_zeros(&r, n, 1);

	for(ii=0; ii<n; ii++) v[ii] = ii; // x
	for(ii=0; ii<n; ii++) vp[ii] = 8.0; // upper
	for(ii=0; ii<n; ii++) vm[ii] = 3.0; // lower
	for(ii=0; ii<n; ii++) r[ii] = 2*ii+1; // x

	d_print_mat(1, n, v, 1);
	d_print_mat(1, n, vp, 1);
	d_print_mat(1, n, vm, 1);
	d_print_mat(1, n, r, 1);

	struct d_strvec sv; d_create_strvec(n, &sv, v);
	struct d_strvec svp; d_create_strvec(n, &svp, vp);
	struct d_strvec svm; d_create_strvec(n, &svm, vm);
	struct d_strvec sm; d_create_strvec(n, &sm, m);
	struct d_strvec sr; d_create_strvec(n, &sr, r);

//	d_print_tran_strvec(n, &sv, 0);
//	d_print_tran_strvec(n, &svp, 0);
//	d_print_tran_strvec(n, &svm, 0);
//	d_print_tran_strvec(n, &sm, 0);
//	d_print_tran_strvec(n, &sr, 0);

//	d_print_tran_strvec(n, &sm, 0);
//	DVECEL_LIBSTR(&sm, 0) = 0.0;
//	DVECEL_LIBSTR(&sm, 1) = 1.0;
//	DVECEL_LIBSTR(&sm, 2) = 2.0;
//	d_print_tran_strvec(n, &sm, 0);
//	return 0;

	double alpha = 1.0;
	double beta = 1.0;
	dgemm_nt_libstr(n, n, n, alpha, &sA, 0, 0, &sA, 0, 0, beta, &sB, 0, 0, &sD, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);
//	dgetrf_nopivot_libstr(n, n, &sD, 0, 0, &sD, 0, 0);
	dgetrf_libstr(n, n, &sD, 0, 0, &sD, 0, 0, ipiv);
	int_print_mat(1, n, ipiv, 1);
	d_print_strmat(n, n, &sD, 0, 0);
	//
#if 0
	// N scheme
#if 1
	dvecpe_libstr(n, ipiv, &sx0, 0);
	dvecpe_libstr(n, ipiv, &sx1, 0);
	dvecpe_libstr(n, ipiv, &sx2, 0);
	dvecpe_libstr(n, ipiv, &sx3, 0);
	dvecpe_libstr(n, ipiv, &sx4, 0);
	dvecpe_libstr(n, ipiv, &sx5, 0);
	dvecpe_libstr(n, ipiv, &sx6, 0);
	dvecpe_libstr(n, ipiv, &sx7, 0);
#endif
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx0, 0, &sz0, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx1, 0, &sz1, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx2, 0, &sz2, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx3, 0, &sz3, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx4, 0, &sz4, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx5, 0, &sz5, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx6, 0, &sz6, 0);
	dtrsv_lnu_libstr(n, &sD, 0, 0, &sx7, 0, &sz7, 0);
	//
	d_print_tran_strvec(n, &sz0, 0);
	d_print_tran_strvec(n, &sz1, 0);
	d_print_tran_strvec(n, &sz2, 0);
	d_print_tran_strvec(n, &sz3, 0);
	d_print_tran_strvec(n, &sz4, 0);
	d_print_tran_strvec(n, &sz5, 0);
	d_print_tran_strvec(n, &sz6, 0);
	d_print_tran_strvec(n, &sz7, 0);
	//
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz0, 0, &sz0, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz1, 0, &sz1, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz2, 0, &sz2, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz3, 0, &sz3, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz4, 0, &sz4, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz5, 0, &sz5, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz6, 0, &sz6, 0);
	dtrsv_unn_libstr(n, &sD, 0, 0, &sz7, 0, &sz7, 0);
	//
	d_print_tran_strvec(n, &sz0, 0);
	d_print_tran_strvec(n, &sz1, 0);
	d_print_tran_strvec(n, &sz2, 0);
	d_print_tran_strvec(n, &sz3, 0);
	d_print_tran_strvec(n, &sz4, 0);
	d_print_tran_strvec(n, &sz5, 0);
	d_print_tran_strvec(n, &sz6, 0);
	d_print_tran_strvec(n, &sz7, 0);
#else
	// T scheme
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx0, 0, &sz0, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx1, 0, &sz1, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx2, 0, &sz2, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx3, 0, &sz3, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx4, 0, &sz4, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx5, 0, &sz5, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx6, 0, &sz6, 0);
	dtrsv_utn_libstr(n, &sD, 0, 0, &sx7, 0, &sz7, 0);
	//
	d_print_tran_strvec(n, &sz0, 0);
	d_print_tran_strvec(n, &sz1, 0);
	d_print_tran_strvec(n, &sz2, 0);
	d_print_tran_strvec(n, &sz3, 0);
	d_print_tran_strvec(n, &sz4, 0);
	d_print_tran_strvec(n, &sz5, 0);
	d_print_tran_strvec(n, &sz6, 0);
	d_print_tran_strvec(n, &sz7, 0);
	//
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz0, 0, &sz0, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz1, 0, &sz1, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz2, 0, &sz2, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz3, 0, &sz3, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz4, 0, &sz4, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz5, 0, &sz5, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz6, 0, &sz6, 0);
	dtrsv_ltu_libstr(n, &sD, 0, 0, &sz7, 0, &sz7, 0);
	//
#if 1
	dvecpei_libstr(n, ipiv, &sz0, 0);
	dvecpei_libstr(n, ipiv, &sz1, 0);
	dvecpei_libstr(n, ipiv, &sz2, 0);
	dvecpei_libstr(n, ipiv, &sz3, 0);
	dvecpei_libstr(n, ipiv, &sz4, 0);
	dvecpei_libstr(n, ipiv, &sz5, 0);
	dvecpei_libstr(n, ipiv, &sz6, 0);
	dvecpei_libstr(n, ipiv, &sz7, 0);
#endif
	//
	d_print_tran_strvec(n, &sz0, 0);
	d_print_tran_strvec(n, &sz1, 0);
	d_print_tran_strvec(n, &sz2, 0);
	d_print_tran_strvec(n, &sz3, 0);
	d_print_tran_strvec(n, &sz4, 0);
	d_print_tran_strvec(n, &sz5, 0);
	d_print_tran_strvec(n, &sz6, 0);
	d_print_tran_strvec(n, &sz7, 0);
#endif
	return 0;
//	kernel_dgemm_nt_4x4_gen_lib4(4, &alpha, sA.pA, sB.pA, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 4, 0, 4);
//	kernel_dsyrk_nt_l_4x4_gen_lib4(4, &alpha, sA.pA, sB.pA, &beta, 0, sD.pA, sD.cn, 3, sD.pA, sD.cn, 0, 4, 0, 4);
//	kernel_dtrmm_nn_rl_4x4_gen_lib4(4, &alpha, sB.pA, 3, sA.pA, sB.cn, 0, sD.pA, sD.cn, 0, 4, 0, 4);
	d_print_strmat(n, n, &sD, 0, 0);
//	kernel_dgemv_n_4_lib4(4, &alpha, sA.pA, sx0.pa, &beta, sx0.pa, sz0.pa);
//	kernel_dgemv_n_4_vs_lib4(4, &alpha, sA.pA, sx1.pa, &beta, sx0.pa, sz0.pa, 5);
//	kernel_dgemv_t_4_lib4(3, &alpha, sA.pA, sA.cn, sx2.pa, &beta, sx0.pa, sz0.pa);
//	kernel_dgemv_t_4_vs_lib4(3, &alpha, sA.pA, sA.cn, sx2.pa, &beta, sx0.pa, sz0.pa, 3);
//	kernel_dgemv_nt_4_lib4(4, &alpha, &alpha, sA.pA+4, sA.cn, sx0.pa, sx0.pa, &beta, sz0.pa, sz0.pa, sz1.pa);
//	kernel_dsymv_l_4_lib4(4, &alpha, sA.pA+0, sA.cn, sx3.pa, sz0.pa);
//	d_print_tran_strvec(n, &sz0, 0);
//	d_print_tran_strvec(n, &sz1, 0);
	return 0;
	dtrmm_rlnn_libstr(8, 8, alpha, &sA, 3, 0, &sB, 0, 0, &sD, 0, 0);
//	dgemm_nn_libstr(8, 8, 8, alpha, &sB, 0, 0, &sA, 1, 0, beta, &sA, 0, 0, &sD, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);
	return 0;
//	dsyrk_ln_libstr(n, 15, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
//	dpotrf_l_mn_libstr(n, 15, &sD, 0, 0, &sD, 0, 0);
//	dsyrk_dpotrf_ln_libstr(n, 15, n, &sA, 0, 0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
//	dtrmm_rlnn_libstr(n, n, alpha, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
//	dgese_libstr(n, n, 0.0/0.0, &sD, 0, 0);
//	kernel_dgemm_nt_4x8_lib4(n, &alpha, sA.pA, sB.pA, sB.cn, &beta, sC.pA, sD.pA);
//	kernel_dgemm_nn_4x8_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sC.pA, sD.pA);
//	kernel_dsyrk_nt_l_4x4_gen_lib4(n, &alpha, sA.pA, sB.pA, &beta, 0, sC.pA, sC.cn, 3, sD.pA, sD.cn, 0, 4, 0, 4);
//	kernel_dsyrk_nt_l_8x4_gen_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, 0, sC.pA, sC.cn, 3, sD.pA, sD.cn, 0, 8, 0, 8);
//	dsyrk_ln_libstr(10, 10, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sD, 1, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx0, 0, beta, &sz0, 0, &sz0, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx1, 0, beta, &sz1, 0, &sz1, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx2, 0, beta, &sz2, 0, &sz2, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx3, 0, beta, &sz3, 0, &sz3, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx4, 0, beta, &sz4, 0, &sz4, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx5, 0, beta, &sz5, 0, &sz5, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx6, 0, beta, &sz6, 0, &sz6, 0);
	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx7, 0, beta, &sz7, 0, &sz7, 0);
//	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx8, 0, beta, &sz8, 0, &sz8, 0);
//	dsymv_l_libstr(10, 10, alpha, &sA, 0, 0, &sx9, 0, beta, &sz9, 0, &sz9, 0);
	d_print_tran_strvec(n, &sz0, 0);
	d_print_tran_strvec(n, &sz1, 0);
	d_print_tran_strvec(n, &sz2, 0);
	d_print_tran_strvec(n, &sz3, 0);
	d_print_tran_strvec(n, &sz4, 0);
	d_print_tran_strvec(n, &sz5, 0);
	d_print_tran_strvec(n, &sz6, 0);
	d_print_tran_strvec(n, &sz7, 0);
//	d_print_tran_strvec(n, &sz8, 0);
//	d_print_tran_strvec(n, &sz9, 0);
	return 0;

//	d_print_strmat(n, n, &sC, 0, 0);
//	dgese_libstr(n, n, 1.0, &sB, 0, 0);
//	kernel_dger4_sub_4_lib4(6, sB.pA, sA.pA, sC.pA);
//	kernel_dger4_sub_4_vs_lib4(6, sB.pA, sA.pA, sC.pA, 1);
	return 0;

//	d_print_strmat(n, n, &sC, 0, 0);
//	dgese_libstr(n, n, 1.0, &sB, 0, 0);
//	kernel_dger4_sub_4_lib4(6, sB.pA, sA.pA, sC.pA);
//	kernel_dger4_sub_4_vs_lib4(6, sB.pA, sA.pA, sC.pA, 1);
//	kernel_dger4_sub_8_lib4(5, sB.pA, sB.cn, sA.pA, sC.pA, sC.cn);
//	kernel_dger4_sub_8_vs_lib4(5, sB.pA, sB.cn, sA.pA, sC.pA, sC.cn, 5);
//	kernel_dger4_sub_12_lib4(5, sB.pA, sB.cn, sA.pA, sC.pA, sC.cn);
//	kernel_dger4_sub_12_vs_lib4(5, sB.pA, sB.cn, sA.pA, sC.pA, sC.cn, 9);
//	kernel_dger4_sub_8c_lib4(9, sB.pA, sA.cn, sA.pA, sC.pA, sC.cn);
//	kernel_dger4_sub_4c_lib4(9, sB.pA, sA.cn, sA.pA, sC.pA, sC.cn);
//	d_print_strmat(n, n, &sC, 0, 0);
//	return 0;

#if 1
	dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0);
#else
	dgese_libstr(n, n, 0.1, &sC, 0, 0);
	DMATEL_LIBSTR(&sC, 0, 0) = 1.0;
//	DMATEL_LIBSTR(&sC, 0, 1) = 1.0;
	for(ii=1; ii<n-1; ii++)
		{
//		DMATEL_LIBSTR(&sC, ii, ii-1) = 1.0;
		DMATEL_LIBSTR(&sC, ii, ii) = 1.0;
//		DMATEL_LIBSTR(&sC, ii, ii+1) = 1.0;
		}
//	DMATEL_LIBSTR(&sC, n-1, n-2) = 1.0;
	DMATEL_LIBSTR(&sC, n-1, n-1) = 1.0;
#endif
	d_print_strmat(n, n, &sC, 0, 0);
	dgese_libstr(n, n, 0.0/0.0, &sD, 0, 0);
//	d_print_strmat(n, n, &sA, 0, 0);
//	dgein1_libstr(12.0, &sA, 0, 0);
//	DMATEL_LIBSTR(&sA, 0, 0) =   12.0;
//	DMATEL_LIBSTR(&sA, 1, 0) =    6.0;
//	DMATEL_LIBSTR(&sA, 2, 0) = -  4.0;
//	DMATEL_LIBSTR(&sA, 0, 1) = - 51.0;
//	DMATEL_LIBSTR(&sA, 1, 1) =  167.0;
//	DMATEL_LIBSTR(&sA, 2, 1) =   24.0;
//	DMATEL_LIBSTR(&sA, 0, 2) =    4.0;
//	DMATEL_LIBSTR(&sA, 1, 2) = - 68.0;
//	DMATEL_LIBSTR(&sA, 2, 2) = - 41.0;
//	d_print_strmat(n, n, &sA, 0, 0);
	d_print_strmat(n, n, &sC, 0, 0);
//	printf("\n%f\n", DGEEL_LIBSTR(&sA, 0, 0));
//	int qr_work_size = dgeqrf_work_size_libstr(n, n);
	int qr_work_size = dgelqf_work_size_libstr(n, n);
	void *qr_work;
	v_zeros_align(&qr_work, qr_work_size);
//	dgeqrf_libstr(10, 10, &sC, 0, 0, &sD, 0, 0, qr_work);
	dgelqf_libstr(17, 17, &sC, 0, 0, &sD, 0, 0, qr_work);
//	dgecp_libstr(10, 10, &sC, 0, 0, &sD, 0, 0);
//	kernel_dgeqrf_4_lib4(16, 12, sD.pA, sD.cn, sD.dA, qr_work);
//	d_print_strmat(n, n, &sA, 0, 0);
//	kernel_dgeqrf_vs_lib4(10, 16, 0, sD.pA+0, sD.cn, sD.dA);
//	kernel_dgelqf_vs_lib4(10, 10, 10, 0, sD.pA+0, sD.cn, sD.dA);
	d_print_strmat(n, n, &sD, 0, 0);
	free(qr_work);
	return 0;

//	dveccl_mask_libstr(n, &svm, 0, &sv, 0, &svp, 0, &sv, 0, &sm, 0);
//	veccl_libstr(n, &svm, 0, &sv, 0, &svp, 0, &sv, 0);
//	d_print_tran_strvec(12, &sv, 0);
//	d_print_tran_strvec(12, &sm, 0);
//	dvecze_libstr(n, &sm, 0, &sr, 0, &sr, 0);
//	d_print_tran_strvec(12, &sr, 0);
//	return 0;

//	d_print_strmat(n, n, &sA, 0, 0);
//	dtrsv_unn_libstr(n, &sA, 1, 0, &sx0, 0, &sz0, 0);
//	d_print_tran_strvec(n, &sz0, 0);
//	dtrsv_unn_libstr(n, &sA, 1, 0, &sx1, 0, &sz1, 0);
//	d_print_tran_strvec(n, &sz1, 0);
//	dtrsv_unn_libstr(n, &sA, 1, 0, &sx2, 0, &sz2, 0);
//	d_print_tran_strvec(n, &sz2, 0);
//	dtrsv_unn_libstr(n, &sA, 1, 0, &sx3, 0, &sz3, 0);
//	d_print_tran_strvec(n, &sz3, 0);
//	return 0;

//	double alpha = 1.0;
//	double beta = 1.0;
//	kernel_dgemm_nt_4x12_vs_lib4(n, &alpha, sA.pA, sB.pA, sB.cn, &beta, sD.pA, sD.pA, 3, 10);
//	kernel_dgemm_nt_8x8u_vs_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn, 7, 6);
	dgemm_nn_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);
	dpotrf_l_libstr(16, &sD, 0, 0, &sD, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);
	return 0;;

//	dmatse_libstr(n, n, 100.0, &sD, 0, 0);

//	for(ii=0; ii<n; ii++)
//		dvecin1_libstr(ii+1, &sx_n, ii);
//	d_print_tran_strvec(n, &sx_n, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	// ddiain_libstr(4, -1.0, &sx_n, 1, &sD, 3, 2);
//	ddiaad_libstr(4, -1.0, &sx_n, 1, &sD, 3, 2);
//	d_print_strmat(n, n, &sD, 0, 0);
//	return 0;

//	d_print_tran_strvec(n, &sx_n, 0);
//	dgemm_l_diag_libstr(n, n, 1.0, &sx_n, 0, &sA, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//	dgemm_r_diag_libstr(n, n, 1.0, &sA, 0, 0, &sx_n, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	exit(1);

//	dsetmat_libstr(n, n, 0.0, &sD, 0, 0);
//	dmatin1_libstr(2.0, &sD, 0, 0);
//	dmatin1_libstr(2.0, &sD, 1, 1);
//	dmatin1_libstr(2.0, &sD, 2, 2);
//	dmatin1_libstr(1.0, &sD, 1, 0);
//	dmatin1_libstr(1.0, &sD, 2, 1);
//	dmatin1_libstr(0.5, &sD, 2, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	d_print_tran_strvec(n, &sx_n, 0);
//	dtrsv_lnn_libstr(n, n, &sD, 0, 0, &sx_n, 0, &sz_n, 0);
//	d_print_tran_strvec(n, &sz_n, 0);
//	exit(1);

//	dgemm_nt_libstr(8, 8, 8, 1.0, &sB, 0, 0, &sA, 1, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	return 0;

//	double alpha = 1.0;
//	kernel_dtrmm_nn_rl_4x4_gen_lib4(7, &alpha, sB.pA, 2, sA.pA, sA.cn, 1, sD.pA, sD.cn, 0, 4, 1, 4);
//	kernel_dtrmm_nn_rl_4x4_gen_lib4(7, &alpha, sB.pA+sB.cn*4, 2, sA.pA, sA.cn, 1, sD.pA+sD.cn*4, sD.cn, 0, 4, 1, 4);
//	kernel_dtrmm_nn_rl_4x4_lib4(4, &alpha, sB.pA, sA.pA, sA.cn+4*4, sD.pA+4*4);
//	kernel_dtrmm_nn_rl_4x4_gen_lib4(3, &alpha, sB.pA+sB.cn*4+4*4, 2, sA.pA+sB.cn*4+4*4, sA.cn, 1, sD.pA+sD.cn*4+4*4, sD.cn, 0, 4, 0, 4);
	dtrmm_rlnn_libstr(8, 8, 1.0, &sB, 0, 0, &sA, 3, 0, &sD, 2, 1);
	d_print_strmat(n, n, &sD, 0, 0);
	return 0;

	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx0, 0, &sx0, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx1, 0, &sx1, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx2, 0, &sx2, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx3, 0, &sx3, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx4, 0, &sx4, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx5, 0, &sx5, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx6, 0, &sx6, 0);
	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx7, 0, &sx7, 0);
//	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx8, 0, &sx8, 0);
//	dtrmv_lnn_libstr(8, 8, &sA, 0, 0, &sx9, 0, &sx9, 0);
	d_print_tran_strvec(n, &sx0, 0);
	d_print_tran_strvec(n, &sx1, 0);
	d_print_tran_strvec(n, &sx2, 0);
	d_print_tran_strvec(n, &sx3, 0);
	d_print_tran_strvec(n, &sx4, 0);
	d_print_tran_strvec(n, &sx5, 0);
	d_print_tran_strvec(n, &sx6, 0);
	d_print_tran_strvec(n, &sx7, 0);
//	d_print_tran_strvec(n, &sx8, 0);
//	d_print_tran_strvec(n, &sx9, 0);
	return 0;

	dgemv_t_libstr(2, 8, 1.0, &sA, 2, 0, &sx_n, 0, 0.0, &sy_n, 0, &sz_n, 0);
	d_print_tran_strvec(n, &sz_n, 0);
	return 0;

	dgemm_nt_libstr(4, 8, 8, 1.0, &sB, 0, 0, &sA, 0, 0, 0.0, &sB, 0, 0, &sD, 3, 0);
//	d_print_strmat(n, n, &sB, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);
	exit(1);

	dpotrf_l_libstr(n, &sD, 0, 0, &sD, 0, 0);
//	dgetrf_nopivot_libstr(n, n, &sD, 0, 0, &sD, 0, 0);
//	dgetrf_libstr(n, n, &sD, 0, 0, &sD, 0, 0, ipiv);
	d_print_strmat(n, n, &sD, 0, 0);
#if defined(LA_HIGH_PERFORMANCE) | defined(LA_REFERENCE)
	d_print_mat(1, n, sD.dA, 1);
#endif
	int_print_mat(1, n, ipiv, 1);
	dtrsm_rltn_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
	exit(1);

#if 1 // solve P L U X = P B
	d_print_strmat(n, n, &sB, 0, 0);
	drowpe_libstr(n, ipiv, &sB);
	d_print_strmat(n, n, &sB, 0, 0);

	dtrsm_llnu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
	dtrsm_lunn_libstr(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
#else // solve X^T (P L U)^T = B^T P^T
	d_print_strmat(n, n, &sB, 0, 0);
	dcolpe_libstr(n, ipiv, &sB);
	d_print_strmat(n, n, &sB, 0, 0);

	dtrsm_rltu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
	dtrsm_rutn_libstr(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
#endif

//	d_print_strmat(n, n, &sA, 0, 0);
//	d_print_strmat(n, n, &sB, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);

//	d_cvt_strmat2mat(n, n, &sE, 0, 0, C, n);
//	d_print_mat(n, n, C, n);

	dtrtr_u_libstr(6, &sE, 2, 0, &sB, 1, 0);
	d_print_strmat(n, n, &sB, 0, 0);

	d_print_strmat(n, n, &sA, 0, 0);
	dgemv_nt_libstr(6, n, 1.0, 1.0, &sA, 0, 0, &sx_n, 0, &sx_t, 0, 0.0, 0.0, &sy_n, 0, &sy_t, 0, &sz_n, 0, &sz_t, 0);
//	dsymv_l_libstr(5, 5, 1.0, &sA, 0, 0, x_n, 0.0, y_n, z_n);
	d_print_mat(1, n, z_n, 1);
	d_print_mat(1, n, z_t, 1);




//	for(ii=0; ii<sE.pm*sE.cn; ii++) sE.pA[ii] = 0.0;
//	double alpha = 0.0;
//	double beta = 1.0;
//	kernel_dgemm_nt_4x4_gen_lib4(4, &alpha, sA.pA, sB.pA, &beta, 3, sA.pA, sA.cn, 0, sE.pA, sE.cn, 0, 4, 2, 2);
//	d_print_strmat(n, n, &sE, 0, 0);

	// free memory
	free(A);
	free(B);
	free(C);
	free(D);
	free(ipiv);
//	d_free_strmat(&sA);
//	d_free_strmat(&sB);
//	d_free_strmat(&sD);
	v_free_align(memory_strmat);

	return 0;

	}
