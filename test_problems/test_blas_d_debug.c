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
	
//	d_print_mat(n, n, A, n);
//	d_print_mat(n, n, B, n);

	double *pA; d_zeros_align(&pA, pn, cn);
	d_cvt_mat2pmat(n, n, A, n, 0, pA, cn);
	
	double *pB; d_zeros_align(&pB, pn, cn);
	d_cvt_mat2pmat(n, n, B, n, 0, pB, cn);
	
	double *pC; d_zeros_align(&pC, pn, cn);

	double *pD; d_zeros_align(&pD, pn, cn);
	d_cvt_mat2pmat(n, n, D, n, 0, pD, cn);
	
	d_print_pmat(n, n, pA, n);
	d_print_pmat(n, n, pB, n);

//	kernel_dgemm_nt_4x4_lib4(3, pA, pB, -1, 0, pC, 1, pD);
//	kernel_dgemm_nt_4x4_vs_lib4(4, pA, pB, -1, 0, pC, 0, pD, 4, 4);
	dgemm_nt_lib(14, 15, n, pA, cn, pB, cn, 0, 0, pC, cn, 0, pD, cn);

	d_print_pmat(n, n, pC, n);
	d_print_pmat(n, n, pD, n);

	d_free(A);
	d_free(B);
	d_free(D);
	d_free_align(pA);
	d_free_align(pB);
	d_free_align(pC);
	d_free_align(pD);

	return 0;

	}
