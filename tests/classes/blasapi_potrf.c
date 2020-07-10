// CLASS_POTRF_BLASAPI
//
void call_routines(struct RoutineArgs *args){

	// copy input matrix A in C
	GECP_REF(args->m, args->m, args->cA_po, 0, 0, args->cC, 0, 0);
	GECP_REF(args->m, args->m, args->rA_po, 0, 0, args->rC, 0, 0);

	// routine call
	//
	BLASFEO(ROUTINE)(
		string(UPLO), &(args->m),
		args->cC->pA, &(args->cC->m),
		&(args->info));

	BLAS(ROUTINE)(
		string(UPLO), &(args->m),
		args->rC->pA, &(args->rC->m),
		&(args->info));

	// C matrix is overwritten with the solution

	// copy result matrix C in D
	GECP_REF(args->m, args->m, args->cC, 0, 0, args->cD, 0, 0);
	GECP_REF(args->m, args->m, args->rC, 0, 0, args->rD, 0, 0);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf(
		"blas_%s_%s: solving A[%d,%d] = LL^T[%d,%d]\n",
		string(ROUTINE), string(UPLO),
		args->m, args->m,
		args->m, args->m
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
	printf("\nInput matrix:\n");
	print_xmat_debug(args->m, args->m, args->cA_po, 0, 0, 0, 0, 0);
	print_xmat_debug(args->m, args->m, args->rA_po, 0, 0, 0, 0, 0);

	printf("\nRow pivot vector:\n");
	int_print_mat(1, args->m, args->cipiv, 1);
	int_print_mat(1, args->m, args->ripiv, 1);
}

void set_test_args(struct TestArgs *targs)
{
	targs->nis = 13;
}
