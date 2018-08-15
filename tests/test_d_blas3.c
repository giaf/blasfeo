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

#if defined(TESTING_MODE)

// standard
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

// BLASFEO routines
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_blas.h"
#include "../include/blasfeo_d_kernel.h"

// External dependencies
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_timing.h"

// BLASFEO LA:REFERENCE routines
#include "../include/blasfeo_d_blas3_ref.h"
#include "../include/blasfeo_d_aux_ref.h"
#include "../include/blasfeo_d_aux_ext_dep_ref.h"

#include "test_d_common.h"
#include "test_x_common.c"


int main()
	{
	print_compilation_flags();

	int ii, jj, kk, ai, aj, bi, bj, ci, cj, di, dj;
	int n = 60;

	double test_elapsed_time;

	blasfeo_timer timer;

	const char* result_code;

	/* matrices in column-major format */
	/* printf("Allocate C matrices\n"); */

	double *A, *B, *C, *D;
	// standard column major allocation (malloc)
	d_zeros(&A, n, n);
	d_zeros(&B, n, n);
	d_zeros(&C, n, n);
	d_zeros(&D, n, n);

	for(ii=0; ii<n*n; ii++) A[ii] = ii+1;
	for(ii=0; ii<n*n; ii++) B[ii] = 2*(ii+1);
	for(ii=0; ii<n*n; ii++) C[ii] = 0.5*(ii+1);

	/* instantiate blasfeo_dmat */

	/* printf("Allocate HP matrices\n"); */

	struct blasfeo_dmat sA; blasfeo_allocate_dmat(n, n, &sA);
	struct blasfeo_dmat sB; blasfeo_allocate_dmat(n, n, &sB);
	struct blasfeo_dmat sC; blasfeo_allocate_dmat(n, n, &sC);
	struct blasfeo_dmat sD; blasfeo_allocate_dmat(n, n, &sD);

	blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
	blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
	blasfeo_pack_dmat(n, n, C, n, &sC, 0, 0);
	blasfeo_pack_dmat(n, n, D, n, &sD, 0, 0);

	// batch memory allocation
	#if 0
	// compute memory size
	int size_dmat = 4*blasfeo_memsize_dmat(n, n);
	// initialize void pointer
	void *memory_dmat;
	// memory allocation
	v_zeros_align(&memory_dmat, size_dmat);
	// cast memory pointer
	char *ptr_memory_dmat = (char *) memory_dmat;
	// instantiate blasfeo_dmat
	struct blasfeo_dmat sA;
	blasfeo_create_dmat(n, n, &sA, ptr_memory_dmat);
	ptr_memory_dmat += sA.memsize;
	blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
	#endif

	/* printf("Allocate REF matrices\n"); */

	struct blasfeo_dmat_ref rA; blasfeo_allocate_dmat_ref(n, n, &rA);
	struct blasfeo_dmat_ref rB; blasfeo_allocate_dmat_ref(n, n, &rB);
	struct blasfeo_dmat_ref rC; blasfeo_allocate_dmat_ref(n, n, &rC);
	struct blasfeo_dmat_ref rD; blasfeo_allocate_dmat_ref(n, n, &rD);


	blasfeo_pack_dmat_ref(n, n, A, n, &rA, 0, 0);
	blasfeo_pack_dmat_ref(n, n, B, n, &rB, 0, 0);
	blasfeo_pack_dmat_ref(n, n, C, n, &rC, 0, 0);
	blasfeo_pack_dmat_ref(n, n, D, n, &rD, 0, 0);

	// -------- Print matrices
	#if 0
	/* printf("\nPrint dmat HP A:\n\n"); */
	/* blasfeo_print_dmat(p_n, p_n, &sA, 0, 0); */

	/* printf("\nPrint dmat REF A:\n\n"); */
	/* blasfeo_print_dmat_ref(p_n, p_n, &rA, 0, 0); */

	/* printf("\nPrint dmat HP B:\n\n"); */
	/* blasfeo_print_dmat(p_n, p_n, &sB, 0, 0); */

	/* printf("\nPrint dmat REF B:\n\n"); */
	/* blasfeo_print_dmat_ref(p_n, p_n, &rB, 0, 0); */
	#endif

	int ni, nj, nk, total_calls, bad_calls;
	int AB_offset_i;

	int err_i = 0;
	int err_j = 0;

	// sub-mastrix offset, sweep start
	int ii0 = 0;
	int jj0 = 0;
	int kk0 = 0;
	int AB_offset0 = 0;

	// sub-matrix dimensions, sweep start
	int ni0 = 1;
	int nj0 = 1;
	int nk0 = 1;

	#if ROUTINE_CLASS_GEMM
	int AB_offsets = 2;
	int ii0s = 13;
	int jj0s = 1;
	int kk0s = 1;
	int nis = 25;
	int njs = 25;
	int nks = 25;
	int alphas = 1;
	#elif ROUTINE_CLASS_SYRK || ROUTINE_CLASS_TRM
	/* ai=bi=ci=di=0 */
	int AB_offsets = 1;
	int ii0s = 1;
	int jj0s = 9;
	int kk0s = 1;
	int nks = 1;
	int alphas = 1;
	int nis = 17;
	int njs = 17;
	#endif

	double alpha_l[6] = {1.0, 0.0, 0.0001, 0.02, 400.0, 50000.0};
	double beta_l[6] = {1.0, 0.0, 0.0001, 0.02, 400.0, 50000.0};

	total_calls = alphas*nis*njs*nks*ii0s*jj0s*AB_offsets;
	bad_calls = 0;

	// Main test loop
	#if 1


	printf("\n----------- TEST " string(ROUTINE) "\n");

	blasfeo_tic(&timer);

	// loop over alphas/betas
	for (kk = 0; kk < alphas; kk++)
		{
		double alpha = alpha_l[kk];
		double beta = beta_l[kk];

		// try different loop grow order n then m and viceversa
		//

		// loop over column matrix dimension
		/* for (ni = ni0; ni < ni0+nis; ni++) */
		for (nj = nj0; nj < nj0+njs; nj++)
			{
			// loop over row matrix dimension
			/* for (nj = nj0; nj < nj0+njs; nj++) */
			for (ni = ni0; ni < ni0+nis; ni++)
				{

				// loop over column matrix dimension
				for (nk = nk0; nk < nk0+nks; nk++)
					{
					// loop over row offset
					for (ii = ii0; ii < ii0+ii0s; ii++)
						{

						// loop over column offset
						for (jj = jj0; jj < jj0+jj0s; jj++)
							{

							// loop over column offset
							for (kk = kk0; kk < kk0+kk0s; kk++)
								{

								// loop over row AB offset
								for (AB_offset_i = AB_offset0; AB_offset_i < AB_offsets; AB_offset_i++)
									{

									ai = ii;
									aj = jj;
									bi = ii+AB_offset_i;
									bj = jj;
									ci = ii+AB_offset_i;
									cj = jj;
									di = ii+AB_offset_i;
									dj = jj;

									#ifdef ROUTINE_CLASS_GEMM
									#if (VERBOSE>1)
									print_routine_signature(string(ROUTINE),
										alpha, beta, ni, nj, nk,
										ai, aj, bi, bj, ci, cj, di, dj);
									#endif

									ROUTINE(
										ni, nj, nk, alpha,
										&sA, ai, aj,
										&sB, bi, bj, beta,
										&sC, ci, cj,
										&sD, di, dj);

									REF(ROUTINE)(
										ni, nj, nk, alpha,
										&rA, ai, aj,
										&rB, bi, bj, beta,
										&rC, ci, cj,
										&rD, di, dj);

									int res = dgecmp_libstr(ni, nj, ai, aj, &sD, &rD, &sA, &rA, &err_i, &err_j, VERBOSE);

									if (!res) bad_calls += 1;
									#if (VERBOSE==0)
									#else
									if (!res)
										{
										#if (VERBOSE>2)
										print_input_matrices(
											string(ROUTINE), ni, nj, nk,
											&sA, &rA, &sB, &rB, &sC, &rC,
											ii, jj, ii+AB_offset_i, jj, ii, jj);
										#endif

										print_routine_signature(string(ROUTINE),
											alpha, beta, ni, nj, nk,
											ai, aj, bi, bj, ci, cj, di, dj);

										print_compilation_flags();
										assert(0);
										}
									#endif

									#elif ROUTINE_CLASS_SYRK

									#if (VERBOSE>1)
									print_routine_signature(string(ROUTINE),
										alpha, beta, ni, nj, nk,
										ai, aj, bi, bj, ci, cj, di, dj);
									#endif

									ROUTINE(
										ni, nj, alpha,
										&sA, ai, aj,
										&sB, bi, bj, beta,
										&sC, ci, cj,
										&sD, di, dj);

									REF(ROUTINE)(
										ni, nj, alpha,
										&rA, ai, aj,
										&rB, bi, bj, beta,
										&rC, ci, cj,
										&rD, di, dj);

									int res = dgecmp_libstr(ni, nj, ai, aj, &sD, &rD, &sA, &rA, &err_i, &err_j, VERBOSE);

									if (!res) bad_calls += 1;
									#if (VERBOSE>0)
									if (!res)
										{
										#if (VERBOSE>2)
										print_input_matrices(
											string(ROUTINE), ni, nj, nk,
											&sA, &rA, &sB, &rB, &sC, &rC,
											ii, jj, ii+AB_offset_i, jj, ii, jj);
										#endif

										print_routine_signature(string(ROUTINE),
											alpha, beta, ni, nj, nk,
											ai, aj, bi, bj, ci, cj, di, dj);
										print_compilation_flags();
										assert(0);
										}
									#endif

									#elif ROUTINE_CLASS_TRM

									#if (VERBOSE>1)
									print_routine_signature(string(ROUTINE),
										alpha, beta, ni, nj, nk,
										ai, aj, bi, bj, ci, cj, di, dj);
									#endif

									ROUTINE(
										ni, nj, alpha,
										&sA, ai, aj,
										&sB, bi, bj,
										&sD, di, dj);

									REF(ROUTINE)(
										ni, nj, alpha,
										&rA, ai, aj,
										&rB, bi, bj,
										&rD, di, dj);

									int res = dgecmp_libstr(ni, nj, ai, aj, &sD, &rD, &sA, &rA, &err_i, &err_j, VERBOSE);

									if (!res) bad_calls += 1;

									#if (VERBOSE>0)
									if (!res)
										{
										#if (VERBOSE>2)
										print_input_matrices(
											string(ROUTINE), ni, nj, nk,
											&sA, &rA, &sB, &rB, &sC, &rC,
											ii, jj, ii+AB_offset_i, jj, ii, jj);
										#endif

										print_routine_signature(string(ROUTINE),
											alpha, beta, ni, nj, nk,
											ai, aj, bi, bj, ci, cj, di, dj);

										print_compilation_flags();
										assert(0);
										}
									#endif

									#else

										printf("\n\nNo Routine Class defined for "string(ROUTINE)"\n\n");
										exit(0);

									#endif

									}
								}
							}
						}
					}
				}
			}
		}

	test_elapsed_time = blasfeo_toc(&timer);

	if (!bad_calls)
		{
			result_code = "SUCCEEDED";
		}
	else
		{
			result_code = "FAILED";
		}

	printf("\n----------- TEST "string(ROUTINE)" %s, %d/%d Bad calls, Elapsed time: %4.4f s\n\n",
			result_code, bad_calls, total_calls, test_elapsed_time);

	#endif

	#if (VERBOSE>1)
	print_compilation_flags();
	#endif

	d_free(A);
	d_free(B);
	d_free(C);
	d_free(D);

	blasfeo_free_dmat(&sB);
	blasfeo_free_dmat(&sA);
	blasfeo_free_dmat(&sC);
	blasfeo_free_dmat(&sD);

	blasfeo_free_dmat_ref(&rA);
	blasfeo_free_dmat_ref(&rB);
	blasfeo_free_dmat_ref(&rC);
	blasfeo_free_dmat_ref(&rD);

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
