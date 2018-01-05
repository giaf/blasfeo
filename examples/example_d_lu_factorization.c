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
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"


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
	int size_strmat = 5*blasfeo_memsize_dmat(n, n);
	void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
	char *ptr_memory_strmat = (char *) memory_strmat;

	struct blasfeo_dmat sA;
//	blasfeo_allocate_dmat(n, n, &sA);
	blasfeo_create_dmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memsize;
	// convert from column major matrix to strmat
	blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
	printf("\nA = \n");
	blasfeo_print_dmat(n, n, &sA, 0, 0);

	struct blasfeo_dmat sB;
//	blasfeo_allocate_dmat(n, n, &sB);
	blasfeo_create_dmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memsize;
	// convert from column major matrix to strmat
	blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
	printf("\nB = \n");
	blasfeo_print_dmat(n, n, &sB, 0, 0);

	struct blasfeo_dmat sI;
//	blasfeo_allocate_dmat(n, n, &sI);
	blasfeo_create_dmat(n, n, &sI, ptr_memory_strmat);
	ptr_memory_strmat += sI.memsize;
	// convert from column major matrix to strmat

	struct blasfeo_dmat sD;
//	blasfeo_allocate_dmat(n, n, &sD);
	blasfeo_create_dmat(n, n, &sD, ptr_memory_strmat);
	ptr_memory_strmat += sD.memsize;

	struct blasfeo_dmat sLU;
//	blasfeo_allocate_dmat(n, n, &sD);
	blasfeo_create_dmat(n, n, &sLU, ptr_memory_strmat);
	ptr_memory_strmat += sLU.memsize;

	blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
	printf("\nB+A*A' = \n");
	blasfeo_print_dmat(n, n, &sD, 0, 0);

//	blasfeo_dgetrf_nopivot(n, n, &sD, 0, 0, &sD, 0, 0);
	blasfeo_dgetrf_rowpivot(n, n, &sD, 0, 0, &sLU, 0, 0, ipiv);
	printf("\nLU = \n");
	blasfeo_print_dmat(n, n, &sLU, 0, 0);
	printf("\nipiv = \n");
	int_print_mat(1, n, ipiv, 1);

#if 0 // solve P L U X = P B
	blasfeo_pack_dmat(n, n, I, n, &sI, 0, 0);
	printf("\nI = \n");
	blasfeo_print_dmat(n, n, &sI, 0, 0);

	blasfeo_drowpe(n, ipiv, &sI);
	printf("\nperm(I) = \n");
	blasfeo_print_dmat(n, n, &sI, 0, 0);

	blasfeo_dtrsm_llnu(n, n, 1.0, &sLU, 0, 0, &sI, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L)) = \n");
	blasfeo_print_dmat(n, n, &sD, 0, 0);
	blasfeo_dtrsm_lunn(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A) = \n");
	blasfeo_print_dmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	blasfeo_unpack_dmat(n, n, &sD, 0, 0, D, n);
#else // solve X^T (P L U)^T = B^T P^T
	blasfeo_pack_tran_dmat(n, n, I, n, &sI, 0, 0);
	printf("\nI' = \n");
	blasfeo_print_dmat(n, n, &sI, 0, 0);

	blasfeo_dcolpe(n, ipiv, &sB);
	printf("\nperm(I') = \n");
	blasfeo_print_dmat(n, n, &sB, 0, 0);

	blasfeo_dtrsm_rltu(n, n, 1.0, &sLU, 0, 0, &sB, 0, 0, &sD, 0, 0);
	printf("\nperm(inv(L')) = \n");
	blasfeo_print_dmat(n, n, &sD, 0, 0);
	blasfeo_dtrsm_rutn(n, n, 1.0, &sLU, 0, 0, &sD, 0, 0, &sD, 0, 0);
	printf("\ninv(A') = \n");
	blasfeo_print_dmat(n, n, &sD, 0, 0);

	// convert from strmat to column major matrix
	blasfeo_unpack_tran_dmat(n, n, &sD, 0, 0, D, n);
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
//	blasfeo_free_dmat(&sA);
//	blasfeo_free_dmat(&sB);
//	blasfeo_free_dmat(&sD);
//	blasfeo_free_dmat(&sI);
	v_free_align(memory_strmat);

	return 0;
	
	}
