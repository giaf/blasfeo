// CLASS_GETRF
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		args->m, args->n,
		args->sA_po, args->ai, args->aj,
		args->sD, args->di, args->dj,
		args->sipiv);

	REF(BLASFEO(ROUTINE))(
		args->m, args->n,
		args->rA_po, args->ai, args->aj,
		args->rD, args->di, args->dj,
		args->ripiv);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	printf(
		"Solving A[%d:%d,%d:%d] = P * LU[%d:%d,%d:%d]\n",
		args->ai, args->m, args->aj,  args->n,
		args->di, args->m, args->dj, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sA_po, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA_po, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint LU:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rD, args->ai, args->aj, 0, 0, 0);
}


void set_test_args(struct TestArgs *targs)
{
	targs->ii0s = 1;
	targs->jj0s = 1;
	targs->kk0s = 1;
	targs->nis = 10;
	targs->njs = 10;
}
