// CLASS_SYRK
//

void call_routines(struct RoutineArgs *args)
	{

	// copy input matrix B in D
	int ii, jj;
	for(jj=0; jj<args->m; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->cD[ii+args->cD_lda*jj] = args->cB[ii+args->cB_lda*jj];
			}
		}
	for(jj=0; jj<args->m; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->bD[ii+args->bD_lda*jj] = args->bB[ii+args->bB_lda*jj];
			}
		}

	// routine call
	//
	BLASFEO_BLAS(ROUTINE)(
		string(UPLO), string(TRANS),
		&(args->m), &(args->k), &(args->alpha),
		args->cA, &(args->cA_lda),
		&(args->beta),
		args->cD, &(args->cD_lda));

	BLAS(ROUTINE)(
		string(UPLO), string(TRANS),
		&(args->m), &(args->k), &(args->alpha),
		args->bA, &(args->bA_lda),
		&(args->beta),
		args->bD, &(args->bD_lda));

	// D matrix is overwritten with the solution

	}



void print_routine(struct RoutineArgs *args)
	{
	printf("blas_%s(%s, %s, %d, %d, %f, A, %d, %f, D, %d);\n", string(ROUTINE), string(UPLO), string(TRANS), args->m, args->k, args->alpha, args->cA_lda, args->beta, args->cD_lda);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nPrint A:\n");
	if(*string(TRANS)=='n' || *string(TRANS)=='N')
		{
		print_xmat_debug(args->m, args->k, args->cA, args->cA_lda, 0, 0, 0, 0, 0, "HP");
		print_xmat_debug(args->m, args->k, args->bA, args->bA_lda, 0, 0, 0, 0, 0, "REF");
		}
		else
		{
		print_xmat_debug(args->k, args->m, args->cA, args->cA_lda, 0, 0, 0, 0, 0, "HP");
		print_xmat_debug(args->k, args->m, args->bA, args->bA_lda, 0, 0, 0, 0, 0, "REF");
		}

	printf("\nPrint B:\n");
	print_xmat_debug(args->m, args->m, args->cB, args->cB_lda, 0, 0, 0, 0, 0, "HP");
	print_xmat_debug(args->m, args->m, args->bB, args->bB_lda, 0, 0, 0, 0, 0, "REF");
	}



void set_test_args(struct TestArgs *targs)
	{
	targs->nis = 17;
	targs->nks = 9;

	targs->alphas = 1;
	}
