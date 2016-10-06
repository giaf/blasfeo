/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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

#include <stdio.h>

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_blas.h"



int main()
	{

	const int bs = D_BS;
	const int nc = D_NC;

	int ii, jj;

	int n = 16;

	int pn = (n+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;

	double *A; d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++)
		A[ii] = ii;
	
	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++)
		B[ii*(n+1)] = 2.0;
	for(ii=0; ii<n-1; ii++)
		B[1+ii*(n+1)] = 1.0;
	for(ii=0; ii<n-1; ii++)
		B[(ii+1)*n+ii*1] = 1.0;
	for(ii=0; ii<n-2; ii++)
		B[2+ii*(n+1)] = 0.5;
	for(ii=0; ii<n-2; ii++)
		B[(ii+2)*n+ii*1] = 0.5;
	for(ii=0; ii<n-3; ii++)
		B[3+ii*(n+1)] = 0.25;
	for(ii=0; ii<n-3; ii++)
		B[(ii+3)*n+ii*1] = 0.25;
	
	double *D; d_zeros(&D, n, n);
	for(ii=0; ii<n*n; ii++)
		D[ii] = 1.0;
	
	double *x; d_zeros_align(&x, n, 1);
	x[0] = 0.0;
	x[1] = 0.0;
	x[2] = 0.0;
	x[3] = 0.0;
	x[4] = 0.0;
	x[5] = 0.0;
	x[6] = 0.0;
	x[7] = 0.0;
	x[8] = 0.0;
	x[9] = 1.0;
	
	double *y; d_zeros_align(&y, n, 1);
	for(ii=0; ii<n; ii++) y[ii] = ii;

	double *z; d_zeros_align(&z, n, 1);

//	d_print_mat(n, n, A, n);
//	d_print_mat(n, n, B, n);

	double *pA; d_zeros_align(&pA, pn, cn);
	d_cvt_mat2pmat(n, n, A, n, 0, pA, cn);
	
	double *pB; d_zeros_align(&pB, pn, cn);
	d_cvt_mat2pmat(n, n, B, n, 0, pB, cn);
	
	double *pC; d_zeros_align(&pC, pn, cn);

	double *pD; d_zeros_align(&pD, pn, cn);
	d_cvt_mat2pmat(n, n, D, n, 0, pD, cn);

	double *inv_diag_D; d_zeros(&inv_diag_D, pn, 1);
	for(ii=0; ii<n; ii++) inv_diag_D[ii] = 0.5;
	
//	dgemm_nt_lib(11, 11, 16, 1.0, pA, cn, pB, cn, 0.0, pD, cn, pD, cn);
//	dgemm_nn_lib(11, 11, 16, 1.0, pB, cn, pA, cn, 0.0, pC, cn, pD, cn);
//	dsyrk_nt_l_lib(15, 15, 16, 1.0, pA, cn, pA, cn, 1.0, pB, cn, pD, cn);
//	dtrmm_nt_ru_lib(12, 12, -1.0, pA, cn, pB, cn, 1.0, pC, cn, pD, cn);
//	dpotrf_nt_l_lib(15, 15, pD, cn, pD, cn, inv_diag_D);
//	dsyrk_dpotrf_nt_l_lib(15, 15, 16, pA, cn, pA, cn, pB, cn, pD, cn, inv_diag_D);
//	dgemv_t_lib(3, 6, 1.0, pA, cn, x, 0.0, y, z);
//	dtrmv_un_lib(n, pA, cn, x, 0, y, z);
//	dtrsv_ln_inv_lib_b(4, 4, pB, cn, inv_diag_D, x, z);
//	dtrsv_lt_inv_lib_b(11, 3, pB, cn, inv_diag_D, x, z);
//	kernel_dgetrf_nn_4x4_lib4(0, pB, pB, pn, pB, pD, inv_diag_D);
//	kernel_dtrsm_nn_ru_inv_4x4_lib4(0, pB, pB, pn, pB+4*pn, pD+4*pn, pD, inv_diag_D);
//	kernel_dtrsm_nn_ll_one_4x4_lib4(0, pB, pB, pn, pB+4*bs, pD+4*bs, pD);
//	kernel_dgetrf_nn_4x4_lib4(4, pD+4*pn, pD+4*bs, pn, pB+4*pn+4*bs, pD+4*pn+4*bs, inv_diag_D+4);
	dgetrf_nn_nopivot_lib(n, n, pB, pn, pD, pn, inv_diag_D);

	d_print_pmat(n, n, pA, n);
	d_print_pmat(n, n, pB, n);
	d_print_pmat(n, n, pC, n);
	d_print_mat(1, n, inv_diag_D, 1);
	d_print_pmat(n, n, pD, n);
//	d_print_mat(1, n, x, 1);
//	d_print_mat(1, n, y, 1);
//	d_print_mat(1, n, z, 1);

	d_free(A);
	d_free(B);
	d_free(D);
	d_free_align(pA);
	d_free_align(pB);
	d_free_align(pC);
	d_free_align(pD);
	d_free_align(x);
	d_free_align(y);
	d_free_align(z);

	return 0;

	}
