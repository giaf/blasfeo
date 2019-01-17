// CLASS_POTRF_BLASAPI
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	BLASFEO(ROUTINE)(
		string(UPLO), &(args->m),
		args->cA_po->pA, &(args->cA_po->m),
		&(args->info));

	BLAS(ROUTINE)(
		string(UPLO), &(args->m),
		args->rA_po->pA, &(args->rA_po->m),
		&(args->info));

	// copy result in D
	GECP_REF(args->m, args->m, args->cA_po, 0, 0, args->cD, 0, 0);
	GECP_REF(args->m, args->m, args->rA_po, 0, 0, args->rD, 0, 0);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("blas_%s_%s: ", string(ROUTINE), string(UPLO));
	printf(
		"solving A[%d,%d] = LL^T[%d,%d]\n",
		args->m, args->m,
		args->m, args->m
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
}
