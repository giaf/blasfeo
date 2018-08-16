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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"

int kernel_dgemm_nt_4x4_lib4_test(int n, double *alpha, double *A, double *B, double *beta, double *C, double *D);

int main()
	{

	printf("\ntest assembly\n");

	int ii;

	int n = 12;

	double *A; d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
	d_print_mat(n, n, A, n);

	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
	d_print_mat(n, n, B, n);

	struct blasfeo_dmat sA;
	blasfeo_allocate_dmat(n, n, &sA);
	blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
	blasfeo_print_dmat(n, n, &sA, 0, 0);

	struct blasfeo_dmat sB;
	blasfeo_allocate_dmat(n, n, &sB);
	blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
	blasfeo_print_dmat(n, n, &sB, 0, 0);

	struct blasfeo_dmat sD;
	blasfeo_allocate_dmat(n, n, &sD);

	struct blasfeo_dmat sC;
	blasfeo_allocate_dmat(n, n, &sC);

	double alpha = 1.0;
	double beta = 0.0;
	int ret = kernel_dgemm_nt_4x4_lib4_test(n, &alpha, sB.pA, sA.pA, &beta, sB.pA, sD.pA);
	blasfeo_print_dmat(n, n, &sD, 0, 0);

//	printf("\n%ld %ld\n", (long long) n, ret);
//	printf("\n%ld %ld\n", (long long) &alpha, ret);
//	printf("\n%ld %ld\n", (long long) sA.pA, ret);
//	printf("\n%ld %ld\n", (long long) sB.pA, ret);
//	printf("\n%ld %ld\n", (long long) &beta, ret);
//	printf("\n%ld %ld\n", (long long) sC.pA, ret);
//	printf("\n%ld %ld\n", (long long) sD.pA, ret);

	return 0;

	}
