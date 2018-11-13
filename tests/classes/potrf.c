// CLASS_POTRF
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		args->m,
		args->sA_po, args->ai, args->aj,
		args->sD, args->di, args->dj
		);

	REF(BLASFEO(ROUTINE))(
		args->m,
		args->rA_po, args->ai, args->aj,
		args->rD, args->di, args->dj
		);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	printf(
		"Solving A[%d:%d,%d:%d] = LL*[%d:%d,%d:%d]\n",
		args->ai, args->m, args->aj,  args->m,
		args->di, args->m, args->dj, args->m
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sA_po, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA_po, args->ai, args->aj, 0, 0, 0);

		printf("\nPrint LL:\n");
		blasfeo_print_xmat_debug(args->m, args->n, args->sD, args->ai, args->aj, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rD, args->ai, args->aj, 0, 0, 0);
}


void set_test_args(struct TestArgs *targs)
{
	targs->nis = 13;
}
