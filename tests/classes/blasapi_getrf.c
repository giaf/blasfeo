// CLASS_GETRF_BLASAPI
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		&(args->m), &(args->n),
		args->cA->pA, &(args->cA->m),
		args->cipiv, &(args->info));

	BLAS(ROUTINE)(
		&(args->m), &(args->n),
		args->rA->pA, &(args->rA->m),
		args->ripiv, &(args->info));

	// copy result in D
	GECP_REF(args->m, args->m, args->cA, 0, 0, args->cD, 0, 0);
	GECP_REF(args->m, args->m, args->rA, 0, 0, args->rD, 0, 0);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf(
		"blas_%s: solving A[%d,%d] = P * LU[%d,%d]\n",
		string(ROUTINE),
		args->m,  args->n,
		args->m, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		print_xmat_debug(args->m, args->n, args->cA, 0, 0, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA, 0, 0, 0, 0, 0);
}

void set_test_args(struct TestArgs *targs)
{
	targs->nis = 16;
	targs->njs = 16;
}
