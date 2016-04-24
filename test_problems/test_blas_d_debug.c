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

#include <stdio.h>

#include "../include/block_size.h"
#include "../include/d_aux.h"
#include "../include/d_blas.h"



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
		B[ii*(n+1)] = 1.0;
	
	double *D; d_zeros(&D, n, n);
	for(ii=0; ii<n*n; ii++)
		D[ii] = 1.0;
	
	double *x; d_zeros_align(&x, n, 1);
	x[0] = 1.0;
	x[1] = 1.0;
	x[2] = 1.0;
	
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
	
//	dgemm_ntnt_lib(14, 15, n, pA, cn, pB, cn, 0, pC, cn, pD, cn);
//	dgemm_ntnt_lib(14, 15, n, pA, cn, pB, cn, 0, pC, cn, pD, cn);
//	dgemm_nttn_lib(14, 15, n, pA, cn, pB, cn, 0, pC, cn, pD, cn);
//	dgemm_nttt_lib(14, 15, n, pA, cn, pB, cn, 0, pC, cn, pD, cn);
//	dsyrk_ntnn_l_lib(16, 16, n, pA, cn, pB, cn, 1, pB, cn, pD, cn);
//	dtrmm_ntnn_lu_lib(15, 15, pB, cn, pA, cn, 0, pC, cn, pD, cn);
//	dpotrf_ntnn_l_lib(16, 16, pD, cn, pC, cn, inv_diag_D);
//	dsyrk_dpotrf_ntnn_l_lib(15, 15, n, pA, cn, pB, cn, 1, pB, cn, pD, cn, inv_diag_D);
	dgemv_t_lib_b(3, 6, pA, cn, x, 0, y, z);

	d_print_pmat(n, n, pA, n);
//	d_print_pmat(n, n, pB, n);
//	d_print_pmat(n, n, pC, n);
//	d_print_pmat(n, n, pD, n);
//	d_print_mat(1, n, inv_diag_D, 1);
	d_print_mat(1, n, x, 1);
	d_print_mat(1, n, y, 1);
	d_print_mat(1, n, z, 1);

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
