// CLASS_TRM
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	ROUTINE(
		args->m, args->n, args->alpha,
		args->sA, args->ai, args->aj,
		args->sB, args->bi, args->bj,
		args->sD, args->di, args->dj);

	REF(ROUTINE)(
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
	targs->AB_offsets = 1;
	targs->ii0s = 1;
	targs->jj0s = 9;
	targs->kk0s = 1;
	targs->nks = 1;
	targs->alphas = 1;
	targs->nis = 17;
	targs->njs = 17;
}
