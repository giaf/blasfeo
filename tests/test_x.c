/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

int test_routine(struct RoutineArgs *args, int *bad_calls)
	{

	#if (VERBOSE > 2)
	// (err=0 || err=1) && VERBOSE = 3
	print_routine(args);
	#endif

	// execute both HP routine and REF routine
	// templated call
	call_routines(args);
	// the result is expected to be in matrix D inside struct args

	// routine test
	#ifdef TEST_BLAS_API
	int err = GECMP_BLASAPI(
		args->n, args->m, args->ai, args->aj,
		args->cD, args->cD_lda, args->bD, args->bD_lda,
		&(args->err_i), &(args->err_j), VERBOSE);
	#else
	int err = GECMP_BLASFEOAPI(
		args->n, args->m, args->ai, args->aj,
		args->sD, args->rD,
		&(args->err_i), &(args->err_j), VERBOSE);
	#endif

	if (err)
		{
		#if (VERBOSE>=0)
		// err=1 && VERBOSE = any
		// increment number of bad calls olny
		*bad_calls += 1;
		#endif

		#if (VERBOSE==1) ||  (VERBOSE==2)
		// err=1 && VERBOSE=1
		// print routine name and signature
		// templated call
		print_routine(args);
		# endif

		#if (VERBOSE>1)
		// err=1 && VERBOSE=2
		// print input matrices
		// templated call
		print_routine_matrices(args);
		# endif

		print_compilation_flags();
		}
	// terminate on error
	return err;
	}



int main()
	{
	print_compilation_flags();

	int ii, jj, kk;
	int n = 60;
	int bad_calls;
	double test_elapsed_time;
	const char* result_code;

	// test args
	//
	// sub-matrix offset
	int ai, bi, di, xj;
	int ai0, bi0, di0, xj0;
	int ais, bis, dis, xjs;

	// sub-matrix dimensions
	int ni, nj, nk;
	int ni0, nj0, nk0;
	int nis, njs, nks;

	// coefficients
	int alphas;
	int aa;

	blasfeo_timer timer;

	// matrices in column-major format
	REAL *A, *A_po, *B, *C, *D;
	REAL *cA, *cA_po, *cB, *cC, *cD;
	REAL *bA, *bA_po, *bB, *bC, *bD;

	// standard column major allocation (malloc)
	ZEROS(&A, n, n);
	ZEROS(&A_po, n, n);
	ZEROS(&B, n, n);
	ZEROS(&C, n, n);
	ZEROS(&D, n, n);

	ZEROS(&cA, n, n);
	ZEROS(&cA_po, n, n);
	ZEROS(&cB, n, n);
	ZEROS(&cC, n, n);
	ZEROS(&cD, n, n);

	ZEROS(&bA, n, n);
	ZEROS(&bA_po, n, n);
	ZEROS(&bB, n, n);
	ZEROS(&bC, n, n);
	ZEROS(&bD, n, n);

	// fill up matrices with not trivial numbers
	for(ii=0; ii<n*n; ii++) A[ii] = ii+1;
	for(ii=0; ii<n*n; ii++) B[ii] = 2*(ii+1);
	for(ii=0; ii<n*n; ii++) C[ii] = 0.5*(ii+1);

	// A non singular matrix
	// A[i,i] = A[i,i] + 1
	for(ii=0; ii<n; ii++) A[(ii*n)+ii] = A[(ii*n)+ii] + 1;

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

	// A_po[i,i] = A_po[i,i] + 1E6 + i
	// Well conditioned positive definite matrix
	for(ii=0; ii<n; ii++) A_po[(ii*n)+ii] = A_po[(ii*n)+ii] + 1E6+ii;

	// Allocate HP matrices
	struct MAT sA; ALLOCATE_MAT(n, n, &sA);
	struct MAT sA_po; ALLOCATE_MAT(n, n, &sA_po);
	struct MAT sB; ALLOCATE_MAT(n, n, &sB);
	struct MAT sC; ALLOCATE_MAT(n, n, &sC);
	struct MAT sD; ALLOCATE_MAT(n, n, &sD);
	PACK_MAT(n, n, A, n, &sA, 0, 0);
	PACK_MAT(n, n, A_po, n, &sA_po, 0, 0);
	PACK_MAT(n, n, B, n, &sB, 0, 0);
	PACK_MAT(n, n, C, n, &sC, 0, 0);
	PACK_MAT(n, n, D, n, &sD, 0, 0);
	sA.m = n;
	sA.n = n;
	sB.m = n;
	sB.n = n;
	sC.m = n;
	sC.n = n;
	sD.m = n;
	sD.n = n;

	// Allocate BLASFEO_blasapi matrices
	for(ii=0; ii<n*n; ii++) cA[ii] = A[ii];
	for(ii=0; ii<n*n; ii++) cA_po[ii] = A_po[ii];
	for(ii=0; ii<n*n; ii++) cB[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) cC[ii] = C[ii];
	for(ii=0; ii<n*n; ii++) cD[ii] = D[ii];

	// Allocate blas matrices
	for(ii=0; ii<n*n; ii++) bA[ii] = A[ii];
	for(ii=0; ii<n*n; ii++) bA_po[ii] = A_po[ii];
	for(ii=0; ii<n*n; ii++) bB[ii] = B[ii];
	for(ii=0; ii<n*n; ii++) bC[ii] = C[ii];
	for(ii=0; ii<n*n; ii++) bD[ii] = D[ii];

	// Allocate ref matrices
	struct MAT_REF rA; ALLOCATE_MAT_REF(n, n, &rA);
	struct MAT_REF rA_po; ALLOCATE_MAT_REF(n, n, &rA_po);
	struct MAT_REF rB; ALLOCATE_MAT_REF(n, n, &rB);
	struct MAT_REF rC; ALLOCATE_MAT_REF(n, n, &rC);
	struct MAT_REF rD; ALLOCATE_MAT_REF(n, n, &rD);
	PACK_MAT_REF(n, n, A, n, &rA, 0, 0);
	PACK_MAT_REF(n, n, A_po, n, &rA_po, 0, 0);
	PACK_MAT_REF(n, n, B, n, &rB, 0, 0);
	PACK_MAT_REF(n, n, C, n, &rC, 0, 0);
	PACK_MAT_REF(n, n, D, n, &rD, 0, 0);
	rA.m = n;
	rA.n = n;
	rB.m = n;
	rB.n = n;
	rC.m = n;
	rC.n = n;
	rD.m = n;
	rD.n = n;

	// Allocate row pivot vectors
	int *sipiv;
	int *ripiv;
	int *cipiv;
	int *bipiv;
	int_zeros(&sipiv, n, 1);
	int_zeros(&ripiv, n, 1);
	int_zeros(&cipiv, n, 1);
	int_zeros(&bipiv, n, 1);

	// Test description structure
	struct TestArgs targs;
	initialize_test_args(&targs);

	// templated call
	set_test_args(&targs);

	int total_calls = compute_total_calls(&targs);

	// unpack Test Args
	ai0 = targs.ai0;
	bi0 = targs.bi0;
	di0 = targs.di0;
	xj0 = targs.xj0;

	ais = targs.ais;
	bis = targs.bis;
	dis = targs.dis;
	xjs = targs.xjs;

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

	printf("\n----------- TEST " string(ROUTINE_FULLNAME) "\n");

	blasfeo_tic(&timer);

	// Routine single class structure
	struct RoutineArgs args;
	initialize_args(&args);

	// bind matrices
	// blasfeo matrices
	args.sA = &sA;
	args.sA_po = &sA_po;
	args.sB = &sB;
	args.sC = &sC;
	args.sD = &sD;
	args.sipiv = sipiv;

	// blasfeo reference
	args.rA = &rA;
	args.rA_po = &rA_po;
	args.rB = &rB;
	args.rC = &rC;
	args.rD = &rD;
	args.ripiv = ripiv;

	// blasapi matrices (column major)
	args.cA = cA;
	args.cA_lda = n;
	args.cA_po = cA_po;
	args.cA_po_lda = n;
	args.cB = cB;
	args.cB_lda = n;
	args.cC = cC;
	args.cC_lda = n;
	args.cD = cD;
	args.cD_lda = n;
	args.cipiv = cipiv;

	// blas matrices (column major)
	args.bA = cA;
	args.bA_lda = n;
	args.bA_po = cA_po;
	args.bA_po_lda = n;
	args.bB = cB;
	args.bB_lda = n;
	args.bC = cC;
	args.bC_lda = n;
	args.bD = cD;
	args.bD_lda = n;
	args.bipiv = bipiv;

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

					// loop over A row offset
					for (ai = ai0; ai < ai0+ais; ai++)
						{

						// loop over B row offset
						for (bi = bi0; bi < bi0+bis; bi++)
							{

							// loop over column offset
							for (xj = xj0; xj < xj0+xjs; xj++)
								{

								// loop over D row offset
								for (di = di0; di < di0+dis; di++)
									{

									// reset result D
									GESE_REF(n, n, -1.0, &rD, 0, 0);
									GESE(n, n, -1.0, &sD, 0, 0);
									for(ii=0; ii<n*n; ii++) cD[ii] = -1.0;
									for(ii=0; ii<n*n; ii++) bD[ii] = -1.0;

									// load current iteration arguments
									args.ai = ai;
									args.aj = xj;

									args.bi = bi;
									args.bj = xj;

									args.ci = ai;
									args.cj = xj;

									args.di = di;
									args.dj = xj;

									args.m = ni;
									args.n = nj;
									args.k = nk;

									args.alpha = alpha;

									int error = test_routine(&args, &bad_calls);

									if (!CONTINUE_ON_ERROR && error) return 1;
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

	printf("\n----------- TEST "string(ROUTINE_FULLNAME)" %s, %d/%d Bad calls, Elapsed time: %4.4f s\n\n",
			result_code, bad_calls, total_calls, test_elapsed_time);

	#if (VERBOSE>1)
	print_compilation_flags();
	#endif

	FREE(A);
	FREE(A_po);
	FREE(B);
	FREE(C);
	FREE(D);

	FREE(cA);
	FREE(cA_po);
	FREE(cB);
	FREE(cC);
	FREE(cD);

	FREE(bA);
	FREE(bA_po);
	FREE(bB);
	FREE(bC);
	FREE(bD);

	FREE_MAT(&sB);
	FREE_MAT(&sA);
	FREE_MAT(&sA_po);
	FREE_MAT(&sC);
	FREE_MAT(&sD);

	FREE_MAT_REF(&rA);
	FREE_MAT_REF(&rA_po);
	FREE_MAT_REF(&rB);
	FREE_MAT_REF(&rC);
	FREE_MAT_REF(&rD);

	int_free(sipiv);
	int_free(ripiv);
	int_free(cipiv);
	int_free(bipiv);

	if (bad_calls > 0)
		return 1;
	else
		return 0;

	}
