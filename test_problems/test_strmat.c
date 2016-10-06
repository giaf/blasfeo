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

#include "../strmat/d_strmat.h"


int main()
	{

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

	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	d_print_mat(n, n, B, n);

	double *C; d_zeros(&C, n, n);

	//
	// matrices in matrix struct format
	//
	int size_strmat = 4*d_size_strmat(n, n);
	void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
	char *ptr_memory_strmat = (char *) memory_strmat;

	struct d_strmat sA;
//	d_allocate_strmat(n, n, &sA);
	d_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
	d_print_strmat(n, n, &sA, 0, 0);

	struct d_strmat sB;
//	d_allocate_strmat(n, n, &sB);
	d_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	d_print_strmat(n, n, &sB, 0, 0);

	struct d_strmat sD;
//	d_allocate_strmat(n, n, &sD);
	d_create_strmat(n, n, &sD, ptr_memory_strmat);
	ptr_memory_strmat += sD.memory_size;

	struct d_strmat sE;
//	d_allocate_strmat(n, n, &sE);
	d_create_strmat(n, n, &sE, ptr_memory_strmat);
	ptr_memory_strmat += sE.memory_size;

	dgemm_nt_libst(8, 8, 8, 1.0, &sA, 5, 0, &sA, 5, 0, 1.0, &sB, 0, 0, &sD, 5, 2);
	d_print_strmat(n, n, &sB, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);

//	dpotrf_libst(n, 2, &sD, 0, 0, &sD, 0, 0);
//	dgetrf_nopivot_libst(n, n, &sD, 0, 0, &sD, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);

//	dtrsm_llnu_libst(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);
//	dtrsm_lunn_libst(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);

//	dtrsm_rltu_libst(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);
//	dtrsm_rutn_libst(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);

//	d_print_strmat(n, n, &sA, 0, 0);
//	d_print_strmat(n, n, &sB, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);

//	d_cvt_strmat2mat(n, n, &sE, 0, 0, C, n);
//	d_print_mat(n, n, C, n);



//	for(ii=0; ii<sE.pm*sE.cn; ii++) sE.pA[ii] = 0.0;
//	double alpha = 0.0;
//	double beta = 1.0;
//	kernel_dgemm_nt_4x4_gen_lib4(4, &alpha, sA.pA, sB.pA, &beta, 3, sA.pA, sA.cn, 0, sE.pA, sE.cn, 0, 4, 2, 2);
//	d_print_strmat(n, n, &sE, 0, 0);

	// free memory
	free(A);
	free(B);
	free(C);
//	d_free_strmat(&sA);
//	d_free_strmat(&sB);
//	d_free_strmat(&sD);
	v_free_align(memory_strmat);

	return 0;
	
	}
