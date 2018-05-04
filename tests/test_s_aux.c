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

#include "test_s_common.h"
#include "test_x_common.c"


int main()
	{

	print_compilation_flags();

	int ii, jj;

	int n = 21;
	int p_n = 15;

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

	/* -------- instantiate blasfeo_smat */

	// compute memory size
	int size_smat = 2*blasfeo_memsize_smat(n, n);
	int size_smat_ref = 2*blasfeo_memsize_smat_ref(n, n);
	// inizilize void pointer
	void *memory_smat;
	void *memory_smat_ref;

	// initialize pointer
	// memory allocation
	v_zeros_align(&memory_smat, size_smat);
	v_zeros_align(&memory_smat_ref, size_smat_ref);

	// get point to smat
	char *ptr_memory_smat = (char *) memory_smat;
	char *ptr_memory_smat_ref = (char *) memory_smat_ref;

	// instantiate blasfeo_smat
	struct blasfeo_smat sA;
	blasfeo_create_smat(n, n, &sA, ptr_memory_smat);
	ptr_memory_smat += sA.memsize;
	blasfeo_pack_smat(n, n, A, n, &sA, 0, 0);

	struct blasfeo_smat sB;
	blasfeo_create_smat(n, n, &sB, ptr_memory_smat);
	ptr_memory_smat += sB.memsize;
	blasfeo_pack_smat(n, n, B, n, &sB, 0, 0);

	// Testing comparison
	// reference matrices, column major

	struct blasfeo_smat_ref rA;
	blasfeo_create_smat_ref(n, n, &rA, ptr_memory_smat_ref);
	ptr_memory_smat_ref += rA.memsize;
	blasfeo_pack_smat_ref(n, n, A, n, &rA, 0, 0);

	struct blasfeo_smat_ref rB;
	blasfeo_create_smat_ref(n, n, &rB, ptr_memory_smat_ref);
	ptr_memory_smat_ref += sB.memsize;
	blasfeo_pack_smat_ref(n, n, B, n, &rB, 0, 0);

	// -------- Print matrices

	printf("\nPrint smat HP A:\n\n");
	blasfeo_print_smat(p_n, p_n, &sA, 0, 0);

	printf("\nPrint smat REF A:\n\n");
	blasfeo_print_smat_ref(p_n, p_n, &rA, 0, 0);

	printf("\nPrint smat HP B:\n\n");
	blasfeo_print_smat(p_n, p_n, &sB, 0, 0);

	printf("\nPrint smat REF B:\n\n");
	blasfeo_print_smat_ref(p_n, p_n, &rB, 0, 0);


	/* ----------- copy and scale */
	printf("\n\n----------- TEST Copy&Scale\n\n");

	float alpha;
	int ni, mi, res, ai, aj, bi, bj, err_i, err_j;
	alpha = 1.5;
	ni = 12;
	mi = 10;

	printf("Compute different combinations of submatrix offsets\n\n");

	// loop over A offset
	for (ii = 0; ii < 8; ii++)
		{

		// ---- Scale
		//
		//
		ai = ii;
		aj = 0;
		printf("Scale A[%d:%d,%d:%d] by %f\n",
						ai, ni, aj, mi, alpha);

		blasfeo_sgesc(ni, mi, alpha, &sA, ai, aj);
		blasfeo_sgesc_ref(ni, mi, alpha, &rA, ai, aj);

		res = sgecmp_libstr(ni, mi, ai, aj, &sA, &rA, &sA, &rA, &err_i, &err_j, VERBOSE);
		assert(res);

		// loop over B offset
		for (jj = 0; jj < 8; jj++)
			{

			ai = ii;
			aj = 0;
			bi = jj;
			bj = 0;

			// ---- Copy&Scale
			//
			//
			printf("Copy-Scale A[%d:%d,%d:%d] by %f in B[%d:%d,%d:%d]\n",
							     ai,ni, aj,mi,   alpha,  bi,ni, bj,mi);

			// HP submatrix copy&scale
			blasfeo_sgecpsc(ni, mi, alpha, &sA, ai, aj, &sB, bi, bj);
			// REF submatrix copy&scale
			blasfeo_sgecpsc_ref(ni, mi, alpha, &rA, ai, aj, &rB, bi, bj);

			// check against blas with blasfeo REF
			res = sgecmp_libstr(ni, mi, bi, bj, &sB, &rB, &sA, &rA, &err_i, &err_j, VERBOSE);
			assert(res);

			// ---- Copy
			//
			printf("Copy A[%d:%d,%d:%d] in B[%d:%d,%d:%d]\n",
							ai,ni, aj,mi,     bi,ni, bj,mi);

			blasfeo_sgecp(ni, mi, &sA, ai, aj, &sB, bi, bj);
			blasfeo_sgecp_ref(ni, mi, &rA, ii, 0, &rB, bi, bj);

			int res = sgecmp_libstr(ni, mi, bi, bj, &sB, &rB, &sA, &rA, &err_i, &err_j, VERBOSE);
			assert(res);

			printf("\n");
			}

		printf("\n");
		}

	printf("\n----------- END TEST Copy&Scale\n");
	printf("\n----------- TEST SUCCEEDED\n\n");
	printf("\n\n");

	print_compilation_flags();

	return 0;

	}



#else



#include <stdio.h>

int main()
	{
	printf("\n\n Recompile BLASFEO with TESTING_MODE=1 to run this test.\n");
	printf("On CMake use -DBLASFEO_TESTING=ON .\n\n");
	return 0;
	}

#endif
