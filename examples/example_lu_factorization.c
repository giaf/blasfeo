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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_i_aux.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"


int main()
	{

	printf("\nExample of LU factorization and backsolve\n\n");

#if defined(BLASFEO_LA)

	printf("\nLA provided by BLASFEO\n\n");

#elif defined(BLAS_LA)

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

	double *A; d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
//	d_print_mat(n, n, A, n);

	// spd matrix
	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	d_print_mat(n, n, B, n);

	// identity
	double *I; d_zeros(&I, n, n);
	for(ii=0; ii<n; ii++) I[ii*(n+1)] = 1.0;
//	d_print_mat(n, n, B, n);

	// result matrix
	double *D; d_zeros(&D, n, n);
//	d_print_mat(n, n, D, n);

	// permutation indeces
	int *ipiv; int_zeros(&ipiv, n, 1);

	//
	// matrices in matrix struct format
	//

	// work space enough for 5 matrix structs for size n times n
	int size_strmat = 5*d_size_strmat(n, n);
	void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
	char *ptr_memory_strmat = (char *) memory_strmat;

	struct d_strmat sA;
//	d_allocate_strmat(n, n, &sA);
	d_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	// convert from column major matrix to strmat
	d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
	printf("\nA = \n");
	d_print_strmat(n, n, &sA, 0, 0);

	struct d_strmat sB;
//	d_allocate_strmat(n, n, &sB);
	d_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	// convert from column major matrix to strmat
	d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	printf("\nB = \n");
	d_print_strmat(n, n, &sB, 0, 0);

	struct d_strmat sI;
//	d_allocate_strmat(n, n, &sI);
	d_create_strmat(n, n, &sI, ptr_memory_strmat);
	ptr_memory_strmat += sI.memory_size;
	// convert from column major matrix to strmat

	struct d_strmat sD;
//	d_allocate_strmat(n, n, &sD);
	d_create_strmat(n, n, &sD, ptr_memory_strmat);
	ptr_memory_strmat += sD.memory_size;

	struct d_strmat sLU;
//	d_allocate_strmat(n, n, &sD);
	d_create_strmat(n, n, &sLU, ptr_memory_strmat);
	ptr_memory_strmat += sLU.memory_size;

	dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
	printf("\nB+A*A' = \n");
	d_print_strmat(n, n, &sD, 0, 0);

//	dgetrf_nopivot_libstr(n, n, &sD, 0, 0, &sD, 0, 0);
	dgetrf_libstr(n, n, &sD, 0, 0, &sLU, 0, 0, ipiv);
	printf("\nLU = \n");
	d_print_strmat(n, n, &sLU, 0, 0);
	printf("\nipiv = \n");
	int_print_mat(1, n, ipiv, 1);

#if 0 // solve P L U X = P B
	d_cvt_mat2strmat(n, n, I, n, &sI, 0, 0);
	printf("\nI = \n");
	d_print_strmat(n, n, &sI, 0, 0);

	drowpe_libstr(n, ipiv, &sI);
	printf("\nperm(I) = \n");
	d_print_strmat(n, n, &sI, 0, 0);

	dtrsm_llnu_libstr(n, n, 1.0, &sLU, 0, 0, &sI, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L)) = \n");
	d_print_strmat(n, n, &sD, 0, 0);
	dtrsm_lunn_libstr(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A) = \n");
	d_print_strmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	d_cvt_strmat2mat(n, n, &sD, 0, 0, D, n);
#else // solve X^T (P L U)^T = B^T P^T
	d_cvt_tran_mat2strmat(n, n, I, n, &sI, 0, 0);
	printf("\nI' = \n");
	d_print_strmat(n, n, &sI, 0, 0);

	dcolpe_libstr(n, ipiv, &sB);
	printf("\nperm(I') = \n");
	d_print_strmat(n, n, &sB, 0, 0);

	dtrsm_rltu_libstr(n, n, 1.0, &sLU, 0, 0, &sB, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L')) = \n");
	d_print_strmat(n, n, &sD, 0, 0);
	dtrsm_rutn_libstr(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A') = \n");
	d_print_strmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	d_cvt_tran_strmat2mat(n, n, &sD, 0, 0, D, n);
#endif

	// print matrix in column-major format
	printf("\ninv(A) = \n");
	d_print_mat(n, n, D, n);



	//
	// free memory
	//

	d_free(A);
	d_free(B);
	d_free(D);
	d_free(I);
	int_free(ipiv);
//	d_free_strmat(&sA);
//	d_free_strmat(&sB);
//	d_free_strmat(&sD);
//	d_free_strmat(&sI);
	v_free_align(memory_strmat);

	return 0;
	
	}
