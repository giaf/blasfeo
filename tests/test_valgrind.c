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

	int n = 32;

	int ii;

	double *A = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		A[ii] = ii;
	int lda = n;
//	d_print_mat(n, n, A, n);

	double *B = malloc(n*n*sizeof(double));
	for(ii=0; ii<n*n; ii++)
		B[ii] = -0.0;
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

	int *ipiv = malloc(n*sizeof(int));


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

	int m0 = 11;
	int n0 = 11;
	int k0 = 11;



//	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
//	blasfeo_dsyrk(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_1, D, &n);
//	blasfeo_dpotrf(&c_l, &n, D, &n, &info);
//	dsyrk_(&c_u, &c_n, &n, &n, &d_1, A, &n, &d_1, D, &n);
//	dpotrf_(&c_u, &n, D, &n, &info);
//	d_print_mat(n, n, D, n);
//	return 0;
//	d_print_mat(n, n, A, n);


	// blas

	for(ii=0; ii<n*n; ii++) C[ii] = -1;

#if 0
//	dgemm_(&ta, &tb, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
	dgemm_(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);
//	dpotrf_(&c_l, &m0, C, &n, &info);
	dposv_(&c_u, &m0, &n0, C, &n, D, &n, &info);
	d_print_mat(n, n, C, ldc);
	d_print_mat(n, n, D, ldd);
#endif

#if 0
	dgemm_(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
#endif

#if 0
	dsyrk_(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	dtrsm_(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, D, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
//	dtrmm_(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	dtrmm_(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	dtrmm_(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
	dgemm_(&c_n, &c_t, &n, &n, &n, &d_1, A, &n, A, &n, &d_1, C, &n);
	dgetrf_(&m0, &n0, C, &n, ipiv, &info);
//	dgetrs_(&c_t, &m0, &n0, C, &n, ipiv, D, &n, &info);
//	dgesv_(&m0, &n0, C, &n, ipiv, D, &n, &info);
	int_print_mat(1, n, ipiv, 1);
	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);
#endif

//	printf("\ninfo %d\n", info);
//	d_print_mat(n, n, B, ldb);
//	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);

#if 0
	for(ii=0; ii<globm*globn; ii++)
		C[ii] = globA[ii];
	d_print_mat(globm, globn, C, globm);
	dgetrf_(&globm, &globn, C, &globm, ipiv, &info);
	d_print_mat(globm, globn, C, globm);
	int_print_mat(1, n, ipiv, 1);
#endif


	// blasfeo blas


	// dgemm
	m0 = 3; n0 = 3; k0 = 3;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 6; n0 = 6; k0 = 6;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 9; n0 = 9; k0 = 9;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 12; n0 = 12; k0 = 12;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 15; n0 = 15; k0 = 15;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 18; n0 = 18; k0 = 18;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 21; n0 = 21; k0 = 21;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	m0 = 24; n0 = 24; k0 = 24;
	blasfeo_dgemm(&c_n, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_n, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	blasfeo_dgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);

	// dsyrk
	m0 = 3; n0 = 3; k0 = 3;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 6; n0 = 6; k0 = 6;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 9; n0 = 9; k0 = 9;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 12; n0 = 12; k0 = 12;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 15; n0 = 15; k0 = 15;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 18; n0 = 18; k0 = 18;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 21; n0 = 21; k0 = 21;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	m0 = 24; n0 = 24; k0 = 24;
	blasfeo_dsyrk(&c_l, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_l, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_n, &m0, &k0, &alpha, A, &n, &beta, C, &n);
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);

	// dtrmm
	m0 = 3; n0 = 3;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 6; n0 = 6;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 9; n0 = 9;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 12; n0 = 12;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 15; n0 = 15;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 18; n0 = 18;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 21; n0 = 21;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 24; n0 = 24;
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);

	// dtrsm
	m0 = 3; n0 = 3;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 6; n0 = 6;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 9; n0 = 9;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 12; n0 = 12;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 15; n0 = 15;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 18; n0 = 18;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 21; n0 = 21;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	m0 = 24; n0 = 24;
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrsm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);

	// dgetrf
	m0 = 3; n0 = 3;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 6; n0 = 6;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 9; n0 = 9;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 12; n0 = 12;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 15; n0 = 15;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 18; n0 = 18;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 21; n0 = 21;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);
	m0 = 24; n0 = 24;
	blasfeo_dgetrf(&m0, &n0, B, &n, ipiv, &info);

	// dgetrs
	m0 = 3; n0 = 3;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 6; n0 = 6;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 9; n0 = 9;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 12; n0 = 12;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 15; n0 = 15;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 18; n0 = 18;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 21; n0 = 21;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);
	m0 = 24; n0 = 24;
	blasfeo_dgetrs(&c_n, &m0, &n0, B, &n, ipiv, D, &n, &info);
	blasfeo_dgetrs(&c_t, &m0, &n0, B, &n, ipiv, D, &n, &info);

	// dpotrf
	m0 = 3;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 6;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 9;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 12;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 15;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 18;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 21;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);
	m0 = 24;
	blasfeo_dpotrf(&c_l, &m0, B, &n, &info);
	blasfeo_dpotrf(&c_u, &m0, B, &n, &info);

	// dgetrs
	m0 = 3; n0 = 3;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 6; n0 = 6;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 9; n0 = 9;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 12; n0 = 12;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 15; n0 = 15;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 18; n0 = 18;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 21; n0 = 21;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);
	m0 = 24; n0 = 24;
	blasfeo_dpotrs(&c_l, &m0, &n0, B, &n, D, &n, &info);
	blasfeo_dpotrs(&c_u, &m0, &n0, B, &n, D, &n, &info);



#if 0
//	blasfeo_dgemm(&ta, &tb, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
	blasfeo_dgemm(&ta, &tb, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);
//	blasfeo_dpotrf(&c_l, &m0, C, &n, &info);
	blasfeo_dposv(&c_u, &m0, &n0, C, &n, D, &n, &info);
	d_print_mat(n, n, C, ldc);
	d_print_mat(n, n, D, ldd);
#endif

#if 0
	blasfeo_dgemm(&c_n, &c_t, &m0, &n0, &k0, &alpha, A, &n, B, &n, &beta, C, &n);
#endif

#if 0
	blasfeo_dsyrk(&c_u, &c_t, &m0, &k0, &alpha, A, &n, &beta, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
	blasfeo_dtrsm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, D, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n;  ii++) C[ii] = B[ii];
//	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
	blasfeo_dtrmm(&c_l, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_l, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_l, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_l, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_u, &c_n, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_n, &m0, &n0, &alpha, A, &n, C, &n);
//	blasfeo_dtrmm(&c_r, &c_u, &c_t, &c_u, &m0, &n0, &alpha, A, &n, C, &n);
#endif

#if 0
	for(ii=0; ii<n*n; ii++) C[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) D[ii] = B[ii];
	blasfeo_dgemm(&c_n, &c_t, &n, &n, &n, &d_1, A, &n, A, &n, &d_1, C, &n);
	blasfeo_dgetrf(&m0, &n0, C, &n, ipiv, &info);
//	blasfeo_dgetrs(&c_t, &m0, &n0, C, &n, ipiv, D, &n, &info);
//	blasfeo_dgesv(&m0, &n0, C, &n, ipiv, D, &n, &info);
	int_print_mat(1, n, ipiv, 1);
	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);
#endif

//	printf("\ninfo %d\n", info);
//	d_print_mat(n, n, A, lda);
//	d_print_mat(n, n, B, ldb);
//	d_print_mat(n, n, C, ldc);
//	d_print_mat(n, n, D, ldd);

#if 0
	for(ii=0; ii<globm*globn; ii++)
		C[ii] = globA[ii];
	d_print_mat(globm, globn, C, globm);
	blasfeo_dgetrf(&globm, &globn, C, &globm, ipiv, &info);
	d_print_mat(globm, globn, C, globm);
	int_print_mat(1, n, ipiv, 1);
#endif

	// free memory

	free(A);
	free(B);
	free(C);
	free(D);
	free(ipiv);


	// return
	printf("\nsuccess!\n\n");

	return 0;

	}

