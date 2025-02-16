// CLASS_POTRF_BLASAPI
//
void call_routines(struct RoutineArgs *args)
	{
	// copy input matrix A in D
	int ii, jj;
	for(jj=0; jj<args->m; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->cD[ii+args->cD_lda*jj] = args->cA_po[ii+args->cA_po_lda*jj];
			}
		}
	for(jj=0; jj<args->m; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->bD[ii+args->bD_lda*jj] = args->bA_po[ii+args->bA_po_lda*jj];
			}
		}

	// routine call
	//
	BLASFEO_LAPACK(ROUTINE)(
		string(UPLO), &(args->m),
		args->cD, &(args->cD_lda),
		&(args->info));

	BLAS(ROUTINE)(
		string(UPLO), &(args->m),
		args->bD, &(args->bD_lda),
		&(args->info));

	// D matrix is overwritten with the solution

	}



void print_routine(struct RoutineArgs *args)
	{
	printf("blas_%s(%s, %d, D, %d, info);\n", string(ROUTINE), string(UPLO), args->m, args->cD_lda);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nInput matrix:\n");
	print_xmat_debug(args->m, args->m, args->cA_po, args->cA_po_lda, 0, 0, 0, 0, 0, "HP");
	print_xmat_debug(args->m, args->m, args->bA_po, args->bA_po_lda, 0, 0, 0, 0, 0, "REF");
	}



void set_test_args(struct TestArgs *targs)
	{
	targs->nis = 21;
	}
