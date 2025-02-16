// CLASS_GEMM
//

void call_routines(struct RoutineArgs *args)
	{
	// copy input matrix C in D
	int ii, jj;
	for(jj=0; jj<args->n; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->cD[ii+args->cD_lda*jj] = args->cC[ii+args->cC_lda*jj];
			}
		}
	for(jj=0; jj<args->n; jj++)
		{
		for(ii=0; ii<args->m; ii++)
			{
				args->bD[ii+args->bD_lda*jj] = args->bC[ii+args->bC_lda*jj];
			}
		}

	BLASFEO_BLAS(ROUTINE)(
		string(TRANSA), string(TRANSB),
		&(args->m), &(args->n), &(args->k), &(args->alpha),
		args->cA, &(args->cA_lda),
		args->cB, &(args->cB_lda), &(args->beta),
		args->cD, &(args->cD_lda));

	BLAS(ROUTINE)(
		string(TRANSA), string(TRANSB),
		&(args->m), &(args->n), &(args->k), &(args->alpha),
		args->bA, &(args->bA_lda),
		args->bB, &(args->bB_lda), &(args->beta),
		args->bD, &(args->bD_lda));

	}



void print_routine(struct RoutineArgs *args)
	{
	printf("blas_%s(%s, %s, %d, %d, %d, %f, A, %d, B, %d, D, %d);\n", string(ROUTINE), string(TRANSA), string(TRANSB), args->m, args->n, args->k, args->alpha, args->cA_lda, args->cB_lda, args->cD_lda);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nPrint A:\n");
	if(*string(TRANSA)=='n' || *string(TRANSA)=='N')
		{
		print_xmat_debug(args->m, args->k, args->cA, args->cA_lda, args->ai, args->aj, 0, 0, 0, "HP");
		print_xmat_debug(args->m, args->k, args->bA, args->cA_lda, args->ai, args->aj, 0, 0, 0, "REF");
		}
		else
		{
		print_xmat_debug(args->k, args->m, args->cA, args->cA_lda, args->ai, args->aj, 0, 0, 0, "HP");
		print_xmat_debug(args->k, args->m, args->bA, args->cA_lda, args->ai, args->aj, 0, 0, 0, "REF");
		}

	printf("\nPrint B:\n");
	if(*string(TRANSB)=='n' || *string(TRANSB)=='N')
		{
		print_xmat_debug(args->k, args->n, args->cB, args->cB_lda, args->bi, args->bj, 0, 0, 0, "HP");
		print_xmat_debug(args->k, args->n, args->bB, args->bB_lda, args->bi, args->bj, 0, 0, 0, "REF");
		}
		else
		{
		print_xmat_debug(args->n, args->k, args->cB, args->cB_lda, args->bi, args->bj, 0, 0, 0, "HP");
		print_xmat_debug(args->n, args->k, args->bB, args->bB_lda, args->bi, args->bj, 0, 0, 0, "REF");
		}

	printf("\nPrint C:\n");
	print_xmat_debug(args->m, args->n, args->cC, args->cC_lda, args->di, args->dj, 0, 0, 0, "HP");
	print_xmat_debug(args->m, args->n, args->bC, args->bC_lda, args->di, args->dj, 0, 0, 0, "REF");
	}



void set_test_args(struct TestArgs *targs)
	{
//	targs->ais = 5;
//	targs->bis = 5;
//	targs->dis = 5;
//	targs->xjs = 2;

	targs->nis = 17;
	targs->njs = 8;
	targs->nks = 8;

	targs->alphas = 1;
	}
