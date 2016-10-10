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

#if defined(LA_BLASFEO)

	printf("\nLA provided by BLASFEO\n\n");

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

	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
//	d_print_mat(n, n, B, n);

	double *C; d_zeros(&C, n, n);

	double *x_n; d_zeros(&x_n, n, 1); 
//	for(ii=0; ii<n; ii++) x_n[ii] = 1.0;
	x_n[4] = 1.0;
	double *x_t; d_zeros(&x_t, n, 1); 
//	for(ii=0; ii<n; ii++) x_n[ii] = 1.0;
	x_t[0] = 1.0;
	double *y_n; d_zeros(&y_n, n, 1);
	double *y_t; d_zeros(&y_t, n, 1);
	double *z_n; d_zeros(&z_n, n, 1);
	double *z_t; d_zeros(&z_t, n, 1);

	int *ipiv; int_zeros(&ipiv, n, 1);

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

	dgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);
//	d_print_strmat(n, n, &sB, 0, 0);
	d_print_strmat(n, n, &sD, 0, 0);

//	dpotrf_libstr(n, 2, &sD, 0, 0, &sD, 0, 0);
//	dgetrf_nopivot_libstr(n, n, &sD, 0, 0, &sD, 0, 0);
	dgetrf_libstr(n, n, &sD, 0, 0, &sD, 0, 0, ipiv);
	d_print_strmat(n, n, &sD, 0, 0);
	int_print_mat(1, n, ipiv, 1);

#if 1 // solve P L U X = P B
	d_print_strmat(n, n, &sB, 0, 0);
	drowpe_libstr(n, ipiv, &sB);
	d_print_strmat(n, n, &sB, 0, 0);

	dtrsm_llnu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
	dtrsm_lunn_libstr(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
#else // solve X^T (P L U)^T = B^T P^T
	d_print_strmat(n, n, &sB, 0, 0);
	dcolpe_libstr(n, ipiv, &sB);
	d_print_strmat(n, n, &sB, 0, 0);

	dtrsm_rltu_libstr(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
	dtrsm_rutn_libstr(n, n, 1.0, &sD, 0, 0, &sE, 0, 0, &sE, 0, 0);
	d_print_strmat(n, n, &sE, 0, 0);
#endif

//	d_print_strmat(n, n, &sA, 0, 0);
//	d_print_strmat(n, n, &sB, 0, 0);
//	d_print_strmat(n, n, &sD, 0, 0);
//	d_print_strmat(n, n, &sE, 0, 0);

//	d_cvt_strmat2mat(n, n, &sE, 0, 0, C, n);
//	d_print_mat(n, n, C, n);
	
	dtrtr_u_libstr(6, &sE, 2, 0, &sB, 1, 0);
	d_print_strmat(n, n, &sB, 0, 0);

	d_print_strmat(n, n, &sA, 0, 0);
//	dgemv_nt_libstr(n, n, 1.0, 1.0, &sA, 0, 0, x_n, x_t, 0.0, 0.0, y_n, y_t, z_n, z_t);
	dsymv_l_libstr(n, 4, 1.0, &sA, 0, 0, x_n, 0.0, y_n, z_n);
	d_print_mat(1, n, z_n, 1);
//	d_print_mat(1, n, z_t, 1);




//	for(ii=0; ii<sE.pm*sE.cn; ii++) sE.pA[ii] = 0.0;
//	double alpha = 0.0;
//	double beta = 1.0;
//	kernel_dgemm_nt_4x4_gen_lib4(4, &alpha, sA.pA, sB.pA, &beta, 3, sA.pA, sA.cn, 0, sE.pA, sE.cn, 0, 4, 2, 2);
//	d_print_strmat(n, n, &sE, 0, 0);

	// free memory
	free(A);
	free(B);
	free(C);
	free(ipiv);
//	d_free_strmat(&sA);
//	d_free_strmat(&sB);
//	d_free_strmat(&sD);
	v_free_align(memory_strmat);

	return 0;
	
	}
