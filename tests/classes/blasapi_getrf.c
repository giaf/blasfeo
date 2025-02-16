// CLASS_GETRF_BLASAPI
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
		&(args->m), &(args->n),
		args->cD, &(args->cD_lda),
		args->cipiv, &(args->info));

	BLAS(ROUTINE)(
		&(args->m), &(args->n),
		args->bD, &(args->bD_lda),
		args->bipiv, &(args->info));

	// D matrix is overwritten with the solution

	}



void print_routine(struct RoutineArgs *args)
	{
	printf("blas_%s(%d, %d, D, %d, ipiv, info);\n", string(ROUTINE), args->m, args->n, args->cD_lda);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nInput matrix:\n");
	print_xmat_debug(args->m, args->m, args->cA, args->cA_lda, 0, 0, 0, 0, 0, "HP");
	print_xmat_debug(args->m, args->m, args->bA, args->bA_lda, 0, 0, 0, 0, 0, "REF");

	printf("\nRow pivot vector:\n");
	int size = args->m < args->n ? args->m : args->n;
	int_print_mat(1, size, args->cipiv, 1);
	int_print_mat(1, size, args->bipiv, 1);
	}



void set_test_args(struct TestArgs *targs)
	{
	targs->nis = 21;
	targs->njs = 21;
	}
