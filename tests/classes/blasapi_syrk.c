// CLASS_GEMM
//

void call_routines(struct RoutineArgs *args){

	// copy input matrix B in C
	GECP_REF(args->m, args->m, args->cB, 0, 0, args->cC, 0, 0);
	GECP_REF(args->m, args->m, args->rB, 0, 0, args->rC, 0, 0);

	// input matrices A and B are unchanged

	// routine call
	//
	BLASFEO(ROUTINE)(
		string(UPLO), string(TRANS),
		&(args->m), &(args->n), &(args->alpha),
		args->cA->pA, &(args->cA->m),
		&(args->beta),
		args->cC->pA, &(args->cC->m));

	BLAS(ROUTINE)(
		string(UPLO), string(TRANS),
		&(args->m), &(args->n), &(args->alpha),
		args->rA->pA, &(args->rA->m),
		&(args->beta),
		args->rC->pA, &(args->rC->m));

	// C matrix is overwritten with the solution

	// copy result matrix C in D
	GECP_REF(args->m, args->m, args->cC, 0, 0, args->cD, 0, 0);
	GECP_REF(args->m, args->m, args->rC, 0, 0, args->rD, 0, 0);
}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("blas_%s_%s%s: ", string(ROUTINE), string(UPLO), string(TRANS));
	printf(
		"solving D[%d:%d] =  %f*A[%d:%d]*A[%d:%d]' + %f*B[%d:%d]\n",
		args->m, args->n,
		args->alpha, args->m, args->k,
		args->k, args->n,
		args->beta, args->m, args->n
	);

}

void print_routine_matrices(struct RoutineArgs *args)
{
		printf("\nPrint A:\n");
		print_xmat_debug(args->m, args->n, args->cA, 0, 0, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rA, 0, 0, 0, 0, 0);

		printf("\nPrint B:\n");
		print_xmat_debug(args->m, args->m, args->cB, 0, 0, 0, 0, 0);
		print_xmat_debug(args->m, args->m, args->rB, 0, 0, 0, 0, 0);

		printf("\nPrint D:\n");
		print_xmat_debug(args->m, args->n, args->cD, 0, 0, 0, 0, 0);
		print_xmat_debug(args->m, args->n, args->rD, 0, 0, 0, 0, 0);
}


void set_test_args(struct TestArgs *targs)
{
	targs->nis = 9;
	targs->njs = 9;

	targs->alphas = 1;
}
