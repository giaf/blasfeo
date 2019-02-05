// CLASS_SYRK
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		args->m, args->n, args->alpha,
		args->sA, args->ai, args->aj,
		args->sB, args->bi, args->bj, args->beta,
		args->sC, args->ci, args->cj,
		args->sD, args->di, args->dj);

	REF(BLASFEO(ROUTINE))(
		args->m, args->n, args->alpha,
		args->rA, args->ai, args->aj,
		args->rB, args->bi, args->bj, args->beta,
		args->rC, args->ci, args->cj,
		args->rD, args->di, args->dj);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	int maxn = (args->m > args->n)? args->m : args->n;
	printf(
		"D[%d:%d,%d:%d] =  %f*A[%d:%d,%d:%d]'*A[%d:%d,%d:%d] + %f*C[%d:%d,%d:%d]\n",
		args->di, args->m, args->dj, args->n,
		args->alpha, args->aj, args->k, args->ai, args->m,
		args->ai, args->m, args->aj, args->k,
		args->beta, args->ci, args->m, args->cj, args->n
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

		printf("\nPrint C:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sC, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rC, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint D:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rD, args->ai, args->aj, 0, 0, 0);
}


void set_test_args(struct TestArgs *targs)
{
	targs->njs = 9;
	targs->nks = 9;
}
