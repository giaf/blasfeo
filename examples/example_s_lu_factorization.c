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
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"


int main()
	{

	printf("\nExample of LU factorization and backsolve\n\n");

#if defined(LA_HIGH_PERFORMANCE)

	printf("\nLA provided by BLASFEO\n\n");

#elif defined(LA_REFERENCE)

	printf("\nLA provided by REFERENCE\n\n");

#elif defined(LA_BLAS)

	printf("\nLA provided by BLAS\n\n");

#else

	printf("\nLA provided by ???\n\n");
	exit(2);

#endif

	int ii;

	int n = 16;

	//
	// matrices in column-major format
	//

	float *A; s_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
//	s_print_mat(n, n, A, n);

	// spd matrix
	float *B; s_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	s_print_mat(n, n, B, n);

	// identity
	float *I; s_zeros(&I, n, n);
	for(ii=0; ii<n; ii++) I[ii*(n+1)] = 1.0;
//	s_print_mat(n, n, B, n);

	// result matrix
	float *D; s_zeros(&D, n, n);
//	s_print_mat(n, n, D, n);

	// permutation indeces
	int *ipiv; int_zeros(&ipiv, n, 1);

	//
	// matrices in matrix struct format
	//

	// work space enough for 5 matrix structs for size n times n
	int size_strmat = 5*s_size_strmat(n, n);
	void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
	char *ptr_memory_strmat = (char *) memory_strmat;

	struct blasfeo_smat sA;
//	s_allocate_strmat(n, n, &sA);
	s_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memsize;
	// convert from column major matrix to strmat
	s_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
	printf("\nA = \n");
	s_print_strmat(n, n, &sA, 0, 0);

	struct blasfeo_smat sB;
//	s_allocate_strmat(n, n, &sB);
	s_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memsize;
	// convert from column major matrix to strmat
	s_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	printf("\nB = \n");
	s_print_strmat(n, n, &sB, 0, 0);

	struct blasfeo_smat sI;
//	s_allocate_strmat(n, n, &sI);
	s_create_strmat(n, n, &sI, ptr_memory_strmat);
	ptr_memory_strmat += sI.memsize;
	// convert from column major matrix to strmat

	struct blasfeo_smat sD;
//	s_allocate_strmat(n, n, &sD);
	s_create_strmat(n, n, &sD, ptr_memory_strmat);
	ptr_memory_strmat += sD.memsize;

	struct blasfeo_smat sLU;
//	s_allocate_strmat(n, n, &sD);
	s_create_strmat(n, n, &sLU, ptr_memory_strmat);
	ptr_memory_strmat += sLU.memsize;

	sgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
	printf("\nB+A*A' = \n");
	s_print_strmat(n, n, &sD, 0, 0);

//	sgetrf_nopivot_libstr(n, n, &sD, 0, 0, &sD, 0, 0);
	sgetrf_libstr(n, n, &sD, 0, 0, &sLU, 0, 0, ipiv);
	printf("\nLU = \n");
	s_print_strmat(n, n, &sLU, 0, 0);
	printf("\nipiv = \n");
	int_print_mat(1, n, ipiv, 1);

#if 0 // solve P L U X = P B
	s_cvt_mat2strmat(n, n, I, n, &sI, 0, 0);
	printf("\nI = \n");
	s_print_strmat(n, n, &sI, 0, 0);

	srowpe_libstr(n, ipiv, &sI);
	printf("\nperm(I) = \n");
	s_print_strmat(n, n, &sI, 0, 0);

	strsm_llnu_libstr(n, n, 1.0, &sLU, 0, 0, &sI, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L)) = \n");
	s_print_strmat(n, n, &sD, 0, 0);
	strsm_lunn_libstr(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A) = \n");
	s_print_strmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	s_cvt_strmat2mat(n, n, &sD, 0, 0, D, n);
#else // solve X^T (P L U)^T = B^T P^T
	s_cvt_tran_mat2strmat(n, n, I, n, &sI, 0, 0);
	printf("\nI' = \n");
	s_print_strmat(n, n, &sI, 0, 0);

	scolpe_libstr(n, ipiv, &sB);
	printf("\nperm(I') = \n");
	s_print_strmat(n, n, &sB, 0, 0);

	strsm_rltu_libstr(n, n, 1.0, &sLU, 0, 0, &sB, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L')) = \n");
	s_print_strmat(n, n, &sD, 0, 0);
	strsm_rutn_libstr(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A') = \n");
	s_print_strmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	s_cvt_tran_strmat2mat(n, n, &sD, 0, 0, D, n);
#endif

	// print matrix in column-major format
	printf("\ninv(A) = \n");
	s_print_mat(n, n, D, n);



	//
	// free memory
	//

	s_free(A);
	s_free(B);
	s_free(D);
	s_free(I);
	int_free(ipiv);
//	s_free_strmat(&sA);
//	s_free_strmat(&sB);
//	s_free_strmat(&sD);
//	s_free_strmat(&sI);
	v_free_align(memory_strmat);

	return 0;
	
	}

