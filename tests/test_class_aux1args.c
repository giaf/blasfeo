// CLASS_1ARGS
//
// blasfeo_xgesc(ni, mi, &sA, ai, aj);

void call_routines(struct RoutineArgs *args){

	// unpack args

	// routine call
	//
	ROUTINE(
		args.n, args.m, alpha, &sA, args.ai, args.aj
		);

	REF(ROUTINE)(
		ni, nj, nk, alpha, &rA, ai, aj,
		);

}

void print_routine(struct RoutineArgs *args){
	// unpack args

	printf("%s\n", string(ROUTINE));
	printf(
		"A[%d:%d,%d:%d] =  %f*A[%d:%d,%d:%d]\n",
		args.ai, args.m, args.aj, args.n,
		args.alpha, args.ai, args.m, args.aj, args.n
	);

}

void print_routine_matrices(struct RoutineArgs *args){

		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(args.m, args.n, args.sA, args.ai, args.aj, 0, 0, 0);
		print_xmat_debug(args.m, args.n, args.rA, args.ai, args.aj, 0, 0, 0);
	}

void set_test_args(struct TestArgs *targs)
{
	targs->AB_offsets = 2;
	targs->ii0s = 13;
	targs->jj0s = 1;
	targs->nis = 25;
	targs->njs = 25;
	targs->alphas = 1;

}

							// Classes:
							//
							// ---- 1ARGS
							// Scale
							//
							// 1 matrix
							// different alphas
							// different size
							// different offset
							//
							// ---- 2ARGA
							// Copy
							// 2 matrix
							// different size
							// different offset
							//
							// ---- 3ARGS
							// Copy&Scale
							// Add&Scale
							//
							// 2 matrix
							// different alphas
							// different size
							// different offset
							//
