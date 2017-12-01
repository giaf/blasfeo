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



#if defined(TESTING_MODE)



// standard
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
// BLASFEO
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"
// reference BLASFEO for testing
#include "../include/blasfeo_s_aux_ref.h"
#include "../include/blasfeo_s_aux_ext_dep_ref.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x));

#include "test_s_common.h"
#include "test_x_common.c"


int main()
	{

#ifndef LA
	#error LA undefined
#endif

#ifndef TARGET
	#error TARGET undefined
#endif


	printf("\n\n\n--------------- Single Precision --------------------\n\n\n");

SHOW_DEFINE(LA)
SHOW_DEFINE(TARGET)


	int ii, jj;

	int n = 21;
	int p_n = 15;
	int N = 10;

	//
	// matrices in column-major format
	//
	float *A;
	// standard column major allocation (malloc)
	s_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;

	float *B;
	// standard column major allocation (malloc)
	s_zeros(&B, n, n);
	for(ii=0; ii<n*n; ii++) B[ii] = 2*ii;

	/* -------- instantiate s_strmat */

	// compute memory size
	int size_strmat = 2*s_size_strmat(n, n);
	int size_strmat_ref = 2*blasfeo_s_memsize_strmat_ref(n, n);
	// inizilize void pointer
	void *memory_strmat;
	void *memory_strmat_ref;

	// initialize pointer
	// memory allocation
	v_zeros_align(&memory_strmat, size_strmat);
	v_zeros_align(&memory_strmat_ref, size_strmat_ref);

	// get point to strmat
	char *ptr_memory_strmat = (char *) memory_strmat;
	char *ptr_memory_strmat_ref = (char *) memory_strmat_ref;

	// instantiate s_strmat
	struct s_strmat sA;
	s_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	s_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);

	struct s_strmat sB;
	s_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	s_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);

	// Testing comparison
	// reference matrices, column major

	struct s_strmat_ref rA;
	blasfeo_s_create_strmat_ref(n, n, &rA, ptr_memory_strmat_ref);
	ptr_memory_strmat_ref += rA.memory_size;
	blasfeo_s_cvt_mat2strmat_ref(n, n, A, n, &rA, 0, 0);

	struct s_strmat_ref rB;
	blasfeo_s_create_strmat_ref(n, n, &rB, ptr_memory_strmat_ref);
	ptr_memory_strmat_ref += sB.memory_size;
	blasfeo_s_cvt_mat2strmat_ref(n, n, B, n, &rB, 0, 0);


	// -------- Print matrices

	printf("\nPrint strmat HP A:\n\n");
	s_print_strmat(p_n, p_n, &sA, 0, 0);

	printf("\nPrint strmat REF A:\n\n");
	blasfeo_s_print_strmat_ref(p_n, p_n, &rA, 0, 0);

	printf("\nPrint strmat HP B:\n\n");
	s_print_strmat(p_n, p_n, &sB, 0, 0);

	printf("\nPrint strmat REF B:\n\n");
	blasfeo_s_print_strmat_ref(p_n, p_n, &rB, 0, 0);


	/* ----------- copy and scale */
	printf("\n\n----------- TEST Copy&Scale\n\n");

	float alpha;
	alpha = 1.5;
	int ret, ni, mi;
	ni = 12;
	mi = 10;

	printf("Compute different combinations of submatrix offsets\n\n");

	// loop over A offset
	for (ii = 0; ii < 8; ii++)
		{


		// ---- Scale
		//
		printf("Scale A[%d:%d,%d:%d] by %f\n",
						ii,ni, 0,mi,    alpha);

		sgesc_libstr(     ni, mi, alpha, &sA, ii, 0);
		blasfeo_sgesc_ref(ni, mi, alpha, &rA, ii, 0);

		assert(sgecmp_libstr(n, n, &sA, &rA, &sA, &rA));

		// loop over B offset
		for (jj = 0; jj < 8; jj++)
			{

			// ---- Copy&Scale
			//

			printf("Copy-Scale A[%d:%d,%d:%d] by %f in B[%d:%d,%d:%d]\n",
							     ii,ni, 0,mi,    alpha,  jj,ni, 0,mi);

			// HP submatrix copy&scale
			sgecpsc_libstr(ni, mi, alpha, &sA, ii, 0, &sB, jj, 0);
			// REF submatrix copy&scale
			blasfeo_sgecpsc_ref(ni, mi, alpha, &rA, ii, 0, &rB, jj, 0);
			// check against blas with blasfeo REF
			assert(sgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

			// ---- Copy
			//
			printf("Copy A[%d:%d,%d:%d] in B[%d:%d,%d:%d]\n",
							ii,ni, 0,mi,     jj,ni, 0,mi);

			sgecp_libstr(     ni, mi, &sA, ii, 0, &sB, jj, 0);
			blasfeo_sgecp_ref(ni, mi, &rA, ii, 0, &rB, jj, 0);
			assert(sgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

			printf("\n");
			}

		printf("\n");
		}

	printf("\n\n----------- END TEST Copy&Scale\n\n");

#if defined(LA)
SHOW_DEFINE(LA)
#endif
#if defined(TARGET)
SHOW_DEFINE(TARGET)
#endif
#if defined(PRECISION)
SHOW_DEFINE(PRECISION)
#endif

	printf("\n\n");

	return 0;

	}



#else



#include <stdio.h>

int main()
	{
	printf("\n\n Recompile BLASFEO with TESTING_MODE=1 to run this test.\n\n");
	return 0;
	}



#endif
