// CLASS_GETRF
//
void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	// void blasfeo_dgetrf(int *pm, int *pn, double *C, int *pldc, int *ipiv, int *info)
	//
	BLASFEO(ROUTINE)(
		&(args->m), &(args->n),
		args->cA->pA, &(args->cA->m),
		args->cipiv, &(args->info));

	BLAS(ROUTINE)(
		&(args->m), &(args->n),
		args->rA->pA, &(args->rA->m),
		args->ripiv, &(args->info));

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	printf(
		"Solving A[%d,%d] = P * LU[%d,%d]\n",
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
	targs->ais = 1;
	targs->bis = 1;
	targs->dis = 1;
	targs->xjs = 1;

	targs->nis = 12;
	targs->njs = 12;
	targs->nks = 1;

	targs->alphas = 1;
}
