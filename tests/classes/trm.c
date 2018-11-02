// CLASS_TRM
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		args->m, args->n, args->alpha,
		args->sA, args->ai, args->aj,
		args->sB, args->bi, args->bj,
		args->sD, args->di, args->dj);

	REF(BLASFEO(ROUTINE))(
		args->m, args->n, args->alpha,
		args->rA, args->ai, args->aj,
		args->rB, args->bi, args->bj,
		args->rD, args->di, args->dj);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	int maxn = (args->m > args->n)? args->m : args->n;
	printf(
		"Solving X: %f*A[%d:%d,%d:%d]*X[%d:%d,%d:%d] = %f*B[%d:%d,%d:%d]\n",
		args->alpha, args->ai, maxn, args->aj,  maxn,
		args->di, args->m, args->dj, args->n,
		args->beta, args->bi, args->m, args->bj, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sA, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint B:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sB, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rB, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint D:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rD, args->ai, args->aj, 0, 0, 0);
}


void set_test_args(struct TestArgs *targs)
{
	targs->ais = 1;
	targs->bis = 1;
	targs->dis = 1;
	targs->xjs = 9;

	targs->nis = 9;
	targs->njs = 9;
	targs->nks = 9;
}
