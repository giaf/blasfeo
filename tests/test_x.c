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

void test_routine(struct RoutineArgs *args, int *bad_calls){

	#if (VERBOSE>2)
	print_routine(args);
	#endif

	// execute both HP routine and REF routine
	// templated call
	call_routines(args);

	// routine test
	int res = GECMP_LIBSTR(
		args->n, args->m, args->ai, args->aj,
		args->sD, args->rD,
		&(args->err_i), &(args->err_j), VERBOSE);

	if (!res) *bad_calls += 1;
	#if (VERBOSE==0)
	// increment number of bad calls olny
	#else
	if (!res)
		{
	#if (VERBOSE>1)
		// print input matrices
		// templated call
		print_routine_matrices(args);
	#endif

		// VERBOSE 1 if error
		// print routine name and signature
		// templated call
		print_routine(args);

		print_compilation_flags();
		// assert(0);
		}

	#endif

}

int main()
	{
	print_compilation_flags();

	int ii, jj, kk, aa;
	int ni, nj, nk, bad_calls;
	int AB_offset_i;
	int n = 60;
	double test_elapsed_time;
	const char* result_code;

	// test args
	int ii0, jj0, kk0;
	int ii0s, jj0s, kk0s;
	int AB_offset0, AB_offsets;
	// sub-matrix dimensions, sweep start
	int ni0, nj0, nk0;
	// sub-matrix dimensions, sweep lenght
	int nis, njs, nks;
	int alphas;

	blasfeo_timer timer;

	// matrices in column-major format
	REAL *A, *A_po, *B, *C, *D;

	// standard column major allocation (malloc)
	ZEROS(&A, n, n);
	ZEROS(&A_po, n, n);
	ZEROS(&B, n, n);
	ZEROS(&C, n, n);
	ZEROS(&D, n, n);

	// fill up matrices with not trivial numbers
	for(ii=0; ii<n*n; ii++) A[ii] = ii+1;
	for(ii=0; ii<n*n; ii++) B[ii] = 2*(ii+1);
	for(ii=0; ii<n*n; ii++) C[ii] = 0.5*(ii+1);

	// Create positive definite matrix
	// A_po = A * A'
	REAL c;
	for(jj=0; jj<n; jj++)
	{
		for(ii=0; ii<n; ii++)
		{
			c = 0.0;
			for(kk=0; kk<n; kk++)
				c += A[ii+n*kk] * A[jj+n*kk];
			A_po[ii+n*jj] = c;
		}
	}
	// A_po[i,i] = A_po[i,i] + i
	for(ii=0; ii<n; ii++) A_po[(ii*n)+ii] = A_po[(ii*n)+ii] + ii;

	// Allocate HP matrices
	struct STRMAT sA; ALLOCATE_STRMAT(n, n, &sA);
	struct STRMAT sA_po; ALLOCATE_STRMAT(n, n, &sA_po);
	struct STRMAT sB; ALLOCATE_STRMAT(n, n, &sB);
	struct STRMAT sC; ALLOCATE_STRMAT(n, n, &sC);
	struct STRMAT sD; ALLOCATE_STRMAT(n, n, &sD);
	PACK_STRMAT(n, n, A, n, &sA, 0, 0);
	PACK_STRMAT(n, n, A_po, n, &sA_po, 0, 0);
	PACK_STRMAT(n, n, B, n, &sB, 0, 0);
	PACK_STRMAT(n, n, C, n, &sC, 0, 0);
	PACK_STRMAT(n, n, D, n, &sD, 0, 0);

	// Allocate ref matrices
	struct STRMAT_REF rA; ALLOCATE_STRMAT_REF(n, n, &rA);
	struct STRMAT_REF rA_po; ALLOCATE_STRMAT_REF(n, n, &rA_po);
	struct STRMAT_REF rB; ALLOCATE_STRMAT_REF(n, n, &rB);
	struct STRMAT_REF rC; ALLOCATE_STRMAT_REF(n, n, &rC);
	struct STRMAT_REF rD; ALLOCATE_STRMAT_REF(n, n, &rD);
	PACK_STRMAT_REF(n, n, A, n, &rA, 0, 0);
	PACK_STRMAT_REF(n, n, A_po, n, &rA_po, 0, 0);
	PACK_STRMAT_REF(n, n, B, n, &rB, 0, 0);
	PACK_STRMAT_REF(n, n, C, n, &rC, 0, 0);
	PACK_STRMAT_REF(n, n, D, n, &rD, 0, 0);

	// Allocate row pivot vectors
	int *sipiv;
	int *ripiv;
	int_zeros(&sipiv, n, 1);
	int_zeros(&ripiv, n, 1);

	// Test description structure
	struct TestArgs targs;
	initialize_test_args(&targs);

	// templated call
	set_test_args(&targs);

	int total_calls = compute_total_calls(&targs);

	// unpack Test Args
	ii0 = targs.ii0;
	jj0 = targs.jj0;
	kk0 = targs.kk0;

	ii0s = targs.ii0s;
	jj0s = targs.jj0s;
	kk0s = targs.kk0s;

	AB_offset0 = targs.AB_offset0;
	AB_offsets = targs.AB_offsets;

	// sub-matrix dimensions, sweep start
	ni0 = targs.ni0;
	nj0 = targs.nj0;
	nk0 = targs.nk0;

	// sub-matrix dimensions, sweep lenght
	nis = targs.nis;
	njs = targs.njs;
	nks = targs.nks;
	alphas = targs.alphas;

	bad_calls = 0;

	printf("\n----------- TEST " string(ROUTINE) "\n");

	blasfeo_tic(&timer);

	// Routine single class structure
	struct RoutineArgs args;
	initialize_args(&args);

	// pack matrices
	args.sA = &sA;
	args.sA_po = &sA_po;
	args.sB = &sB;
	args.sC = &sC;
	args.sD = &sD;
	args.sipiv = sipiv;

	args.rA = &rA;
	args.rA_po = &rA_po;
	args.rB = &rB;
	args.rC = &rC;
	args.rD = &rD;
	args.ripiv = ripiv;

	// loop over alphas/betas
	for (aa = 0; aa < alphas; aa++)
		{
		REAL alpha = targs.alpha_l[aa];

		// loop over column matrix dimension
		/* for (ni = ni0; ni < ni0+nis; ni++) */
		for (nj = nj0; nj < nj0+njs; nj++)
			{
			// loop over row matrix dimension
			/* for (nj = nj0; nj < nj0+njs; nj++) */
			for (ni = ni0; ni < ni0+nis; ni++)
				{

				// loop over row matrix dimension
				for (nk = nk0; nk < nk0+nks; nk++)
					{

					// loop over row AB offset
					for (AB_offset_i = AB_offset0; AB_offset_i < AB_offsets; AB_offset_i++)
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

									// load current iteration arguments
									args.ai = ii;
									args.aj = jj;

									args.bi = ii+AB_offset_i;
									args.bj = jj;

									args.ci = ii;
									args.cj = jj;

									args.di = ii;
									args.dj = jj;

									args.n = ni;
									args.m = nj;
									args.k = nk;

									args.alpha = alpha;

									test_routine(&args, &bad_calls);

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

	#if (VERBOSE>1)
	print_compilation_flags();
	#endif

	FREE(A);
	FREE(A_po);
	FREE(B);
	FREE(C);
	FREE(D);

	FREE_STRMAT(&sB);
	FREE_STRMAT(&sA);
	FREE_STRMAT(&sA_po);
	FREE_STRMAT(&sC);
	FREE_STRMAT(&sD);

	FREE_STRMAT_REF(&rA);
	FREE_STRMAT_REF(&rA_po);
	FREE_STRMAT_REF(&rB);
	FREE_STRMAT_REF(&rC);
	FREE_STRMAT_REF(&rD);

	return 0;

	}
