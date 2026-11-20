// CLASS_GETRF_FULLPIVOT
//

void call_routines(struct RoutineArgs *args)
	{
	int srank, rrank;
	REAL tol = 0.0;

	// routine call
	//
	BLASFEO(ROUTINE)(
		args->m, args->n, tol,
		args->sA_po, args->ai, args->aj,
		args->sD, args->di, args->dj,
		args->sipiv, args->sjpiv, &srank);

	BLASFEO(REF(ROUTINE))(
		args->m, args->n, tol,
		args->rA_po, args->ai, args->aj,
		args->rD, args->di, args->dj,
		args->ripiv, args->rjpiv, &rrank);
	}



void print_routine(struct RoutineArgs *args)
	{
	printf("blasfeo_%s(%d, %d, tol, A, %d, %d, D, %d, %d, ipiv, jpiv, rank);\n", string(ROUTINE), args->m, args->n, args->ai, args->aj, args->di, args->dj);
	}



void print_routine_matrices(struct RoutineArgs *args)
	{
	printf("\nPrint A:\n");
	blasfeo_print_xmat_debug(args->m, args->n, args->sA_po, args->ai, args->aj, 0, 0, 0, "HP");
	blasfeo_print_xmat_debug(args->m, args->n, args->rA_po, args->ai, args->aj, 0, 0, 0, "REF");

	printf("\nPrint LU:\n");
	blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->di, args->dj, 0, 0, 0, "HP");
	blasfeo_print_xmat_debug(args->m, args->n, args->rD, args->di, args->dj, 0, 0, 0, "REF");
	}



void set_test_args(struct TestArgs *targs)
	{
	targs->ais = 1;
	targs->di0 = 0;
	targs->dis = 8;

	targs->ni0 = 1;
	targs->nj0 = 1;
	targs->nis = 20;
	targs->njs = 20;
	}
