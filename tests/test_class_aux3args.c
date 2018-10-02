// CLASS_3ARGS
//
// blasfeo_xgecpsc(ni, mi, alpha, &sA, ai, aj, &sB, bi, bj);
// blasfeo_xgead(ni, mi, alpha, &sA, ai, aj, &sB, bi, bj);

void call_routines(struct RoutineArgs *args){
	// call HP and REF routine

	// routine call
	//
	ROUTINE(
		args->n, args->m, args->alpha,
		args->sA, args->ai, args->aj,
		args->sB, args->bi, args->bj
		);

	REF(ROUTINE)(
		args->n, args->m, args->alpha,
		args->rA, args->ai, args->aj,
		args->rB, args->bi, args->bj
		);
}

void print_routine(struct RoutineArgs *args){
	// print current class signature

	printf("%s ", string(ROUTINE));
	printf(
		"B[%d:%d,%d:%d] =  %f*A[%d:%d,%d:%d]\n",
		args->bi, args->m, args->bj, args->n,
		args->alpha, args->ai, args->m, args->aj, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args){

		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sA, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint B:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sB, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rB, args->ai, args->aj, 0, 0, 0);
}

void set_test_args(struct TestArgs *targs)
{
	targs->AB_offsets = 1;
	targs->ii0s = 1;
	targs->jj0s = 9;
	targs->nis = 17;
	targs->njs = 17;
	targs->alphas = 1;
}
