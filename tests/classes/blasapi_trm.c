// CLASS_GEMM

void call_routines(struct RoutineArgs *args)
	{

	// copy input matrix B in D
	int ii, jj;
	for(jj=0; jj<args->n; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->cD[ii+args->cD_lda*jj] = args->cB[ii+args->cB_lda*jj];
			}
		}
	for(jj=0; jj<args->n; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->bD[ii+args->bD_lda*jj] = args->bB[ii+args->bB_lda*jj];
			}
		}

	BLASFEO_BLAS(ROUTINE)(
		string(SIDE), string(UPLO), string(TRANSA), string(DIAG),
		&(args->m), &(args->n), &(args->alpha),
		args->cA, &(args->cA_lda),
		args->cD, &(args->cD_lda));

	BLAS(ROUTINE)(
		string(SIDE), string(UPLO), string(TRANSA), string(DIAG),
		&(args->m), &(args->n), &(args->alpha),
		args->bA, &(args->bA_lda),
		args->bD, &(args->bD_lda));

	// D matrix is overwritten with the solution

	}


void print_routine(struct RoutineArgs *args)
	{
	printf("blas_%s(%s, %s, %s, %s, %d, %d, %f, A, %d, D, %d);\n", string(ROUTINE), string(UPLO), string(SIDE), string(TRANSA), string(DIAG), args->m, args->n, args->alpha, args->cA_lda, args->cD_lda);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nPrint A:\n");
	if(*string(SIDE)=='l' || *string(SIDE)=='L')
		{
		print_xmat_debug(args->m, args->m, args->cA, args->cA_lda, 0, 0, 0, 0, 0, "HP");
		print_xmat_debug(args->m, args->m, args->bA, args->bA_lda, 0, 0, 0, 0, 0, "REF");
		}
		else
		{
		print_xmat_debug(args->n, args->n, args->cA, args->cA_lda, 0, 0, 0, 0, 0, "HP");
		print_xmat_debug(args->n, args->n, args->bA, args->bA_lda, 0, 0, 0, 0, 0, "REF");
		}

	printf("\nPrint B:\n");
	print_xmat_debug(args->m, args->n, args->cB, args->cB_lda, 0, 0, 0, 0, 0, "HP");
	print_xmat_debug(args->m, args->n, args->bB, args->bB_lda, 0, 0, 0, 0, 0, "REF");
	}



void set_test_args(struct TestArgs *targs)
	{
	targs->nis = 21;
	targs->njs = 21;

//	targs->alphas = 1;
	}
