/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_target.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_timing.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"

#include "../include/d_blas.h"



int main()
	{

#if !defined(BLAS_API)
	printf("\nRecompile with BLAS_API=1 to run this benchmark!\n\n");
	exit(1);
#endif

	int n = 16;

	int ii;

	double *A = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		A[ii] = ii;
	int lda = n;
//	d_print_mat(n, n, A, n);

	double *B = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		B[ii] = 0;
	for(ii=0; ii<n; ii++)
		B[ii*(n+1)] = 1.0;
	int ldb = n;
//	d_print_mat(n, n, B, ldb);

	double *C = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		C[ii] = -1;
	int ldc = n;
//	d_print_mat(n, n, C, ldc);

	double *D = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		D[ii] = -1;
	int ldd = n;
//	d_print_mat(n, n, C, ldc);


	int bs = 4;

	double d_0 = 0.0;
	double d_1 = 1.0;

	char c_l = 'l';
	char c_n = 'n';
	char c_r = 'r';
	char c_t = 't';
	char c_u = 'u';

	double alpha = 2.0;
	double beta = 1.0;

	char ta = 'n';
	char tb = 't';
	char uplo = 'u';
	int info = 0;

	int m0 = 15;
	int n0 = 15;
	int k0 = 15;



	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
//	blasfeo_dsyrk(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_1, D, &n);
//	blasfeo_dpotrf(&c_l, &n, D, &n, &info);
	dsyrk_(&c_u, &c_n, &n, &n, &d_1, A, &n, &d_1, D, &n);
	dpotrf_(&c_u, &n, D, &n, &info);
	d_print_mat(n, n, D, n);
//	return 0;


	// blas

	for(ii=0; ii<n*n; ii++) C[ii] = -1;

#if 0
//	dgemm_(&ta, &tb, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	dgemm_(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);
	dpotrf_(&uplo, &m0, C, &n, &info);
#endif

#if 0
	dsyrk_(&uplo, &ta, &m0, &k0, &alpha, A, &n, &beta, C, &n);
#endif

#if 1
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	dtrsm_(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, D, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	dtrmm_(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
#endif

	printf("\ninfo %d\n", info);
//	d_print_mat(n, n, B, ldb);
	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);


	// blasfeo blas

	for(ii=0; ii<n*n; ii++) C[ii] = -1;

#if 0
//	blasfeo_dgemm(&ta, &tb, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	blasfeo_dgemm(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);
	blasfeo_dpotrf(&uplo, &m0, C, &n, &info);
#endif

#if 0
	blasfeo_dsyrk(&uplo, &ta, &m0, &k0, &alpha, A, &n, &beta, C, &n);
#endif

#if 1
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, D, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
#endif

	printf("\ninfo %d\n", info);
//	d_print_mat(n, n, B, ldc);
	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);


	// free memory

	free(A);
	free(B);
	free(C);
	free(D);


	// return

	return 0;

	}
