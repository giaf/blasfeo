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

	BLASFEO(REF(ROUTINE))(
		args->m, args->n, args->alpha,
		args->rA, args->ai, args->aj,
		args->rB, args->bi, args->bj,
		args->rD, args->di, args->dj);

}

void print_routine(struct RoutineArgs *args){
	// unpack args
	//
	printf("Called: %s with: ", string(ROUTINE));
	printf(
		"%f*A[%d,%d|%d,%d]*X[%d,%d] = B[%d,%d]\n\n",
		args->alpha, args->m, args->n, args->m, args->n,
		args->m, args->n,args->m, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sA, args->ai, args->aj, 0, 0, 0, "HP");
		blasfeo_print_xmat_debug(args->m, args->n, args->rA, args->ai, args->aj, 0, 0, 0, "REF");

		printf("\nPrint B:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sB, args->ai, args->aj, 0, 0, 0, "HP");
		blasfeo_print_xmat_debug(args->m, args->n, args->rB, args->ai, args->aj, 0, 0, 0, "REF");

		printf("\nPrint D:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->ai, args->aj, 0, 0, 0, "HP");
		blasfeo_print_xmat_debug(args->m, args->n, args->rD, args->ai, args->aj, 0, 0, 0, "REF");
}


void set_test_args(struct TestArgs *targs)
{
	targs->nis = 20;
	targs->njs = 20;
	targs->nks = 20;

	targs->ni0 = 10;
	targs->nj0 = 10;
	targs->nk0 = 10;

	targs->alphas = 1;
}
