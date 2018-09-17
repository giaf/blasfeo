// Test {double,single} precision common

void print_compilation_flags()
{
	SHOW_DEFINE(LA)
	SHOW_DEFINE(TARGET)
	SHOW_DEFINE(PRECISION)
	SHOW_DEFINE(MIN_KERNEL_SIZE)
	SHOW_DEFINE(ROUTINE)
}

void initialize_test_args(struct TestArgs * targs)
{
	// sub-mastrix offset, sweep start
	targs->ii0 = 0;
	targs->jj0 = 0;
	targs->kk0 = 0;

	targs->ii0s = 1;
	targs->jj0s = 1;
	targs->kk0s = 1;

	targs->AB_offset0 = 0;
	targs->AB_offsets = 1;

	// sub-matrix dimensions, sweep start
	targs->ni0 = 4;
	targs->nj0 = 4;
	targs->nk0 = 4;

	// sub-matrix dimensions, sweep lenght
	targs->nis = 1;
	targs->njs = 1;
	targs->nks = 1;

	targs->alphas = 1;
	targs->alpha_l[0] = 1.0;
	targs->alpha_l[1] = 0.0;
	targs->alpha_l[2] = 0.0001;
	targs->alpha_l[3] = 0.02;
	targs->alpha_l[4] = 400.0;
	targs->alpha_l[5] = 50000.0;

	targs->betas = 1;
	targs->beta_l[0] = 1.0;
	targs->beta_l[1] = 0.0;
	targs->beta_l[2] = 0.0001;
	targs->beta_l[3] = 0.02;
	targs->beta_l[4] = 400.0;
	targs->beta_l[5] = 50000.0;

	targs->total_calls = 1;
};

int compute_total_calls(struct TestArgs * targs)
{
	int total_calls =
		targs->alphas *
		targs->betas *
		targs->nis *
		targs->njs *
		targs->nks *
		targs->ii0s *
		targs->jj0s *
		targs->kk0s *
		targs->AB_offsets;

	return total_calls;
}

void initialize_args(struct RoutineArgs * args)
{
	args->alpha = 0;
	args->beta = 0;

	args->err_i = 0;
	args->err_j = 0;

	// sizes
	args->n = 0;
	args->m = 0;
	args->k = 0;

	// offset
	args->ai = 0;
	args->aj = 0;

	args->bi = 0;
	args->bj = 0;

	args->ci = 0;
	args->cj = 0;

	args->di = 0;
	args->dj = 0;
};

/* prints a matrix in column-major format */
void print_xmat_debug(
	int m, int n, struct STRMAT_REF *sA,
	int ai, int aj, int err_i, int err_j, int ERR)
	{

	/* REAL *pA = sA->pA + ai + aj*lda; */
	int lda = sA->m;
	REAL *pA = sA->pA;
	int j0,i0, ie, je;
	int i, j;
	const int max_rows = 16;
	const int max_cols = 9;
	const int offset = 2;

	i0 = (ai - offset >=0 )? ai - offset : 0;
	ie = ai + m + offset;
	j0 = (aj - offset >=0 )? aj - offset : 0;
	je = aj + n + offset;

	if (ie-i0 > max_rows)
	{
		i0 = (err_i - ((int)(max_rows/2)) >=0 )? err_i - ((int)(max_rows/2)) : 0;
		ie = err_i + ((int)(max_rows/2)) ;
	}
	if (je-j0 > max_cols)
	{
		j0 = (err_j - ((int)(max_rows/2)) >=0 )? err_j - ((int)(max_rows/2)) : 0;
		je = err_j + ((int)(max_rows/2)) ;
	}

	/* i0 = err_i-subsize; */
	/* j0 = err_j-subsize; */
	/* if (i0 < ai) i0 = ai; */
	/* if (j0 < aj) j0 = aj; */
	/* ie = err_i+subsize; */
	/* je = err_j+subsize; */
	/* if (ie > ai+m) ie = ai+m; */
	/* if (je > aj+n) je = aj+n; */

	if (!ERR)
	{
		i0 = ai;
		j0 = aj;
		ie = ai+m;
		je = aj+n;
	}

	printf("%s\t", "REF");
	for(j=j0; j<je; j++) printf("%11d\t", j);
	printf("\n");
	for(j=j0; j<je; j++) printf("-----------------");
	printf("\n");

	for(i=i0; i<ie; i++)
		{
		for(j=j0; j<je; j++)
			{
			if (j == j0)  printf("%d\t| ", i);

			if ((i==err_i) && (j==err_j) && ERR)
				printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+lda*j]);
			else if ((i >= ai) && (i < ai+m) && (j >= aj) && (j < aj+n))
				printf(ANSI_COLOR_GREEN"%9.2f\t"ANSI_COLOR_RESET, pA[i+lda*j]);
			else printf("%9.2f\t", pA[i+lda*j]);

			}
		printf("\n");
		}
	printf("\n");
	return;
	}

/* prints a matrix in panel-major format */
void blasfeo_print_xmat_debug(
	int m, int n, struct STRMAT *sA,
	int ai, int aj, int err_i, int err_j, int ERR)
	{
	#if defined(LA_BLAS_WRAPPER) || defined(LA_REFERENCE)
	/* print_xmat_debug(m, n, sA->pA, ai, aj, err_i, err_j); */
	#else
	const int ps = PS;

	int i0, j0, ie, je;
	int ii, j, ip0, ipe, ip;

	const int max_rows = 16;
	const int max_cols = 9;
	const int offset = 2;

	i0 = (ai - offset >=0 )? ai - offset : 0;
	ie = ai + m + offset;
	j0 = (aj - offset >=0 )? aj - offset : 0;
	je = aj + n + offset;

	if (ie-i0 > max_rows)
	{
		i0 = (err_i - ((int)(max_rows/2)) >=0 )? err_i - ((int)(max_rows/2)) : 0;
		ie = err_i + ((int)(max_rows/2)) ;
	}
	if (je-j0 > max_cols)
	{
		j0 = (err_j - ((int)(max_rows/2)) >=0 )? err_j - ((int)(max_rows/2)) : 0;
		je = err_j + ((int)(max_rows/2)) ;
	}

	/* const int subsize = 6; */
	/* i0 = err_i-subsize; */
	/* j0 = err_j-subsize; */

	/* if (i0 < ai) i0 = ai; */
	/* if (j0 < aj) j0 = aj; */

	/* ie = err_i+subsize; */
	/* je = err_j+subsize; */

	/* if (ie > ai+m) ie = ai+m; */
	/* if (je > aj+n) je = aj+n; */

	if (!ERR)
	{
		i0 = ai;
		j0 = aj;
		ie = ai+m;
		je = aj+n;
	}

	int sda = sA->cn;
	REAL *pA = sA->pA;

	printf("%s\t", "HP");
	for(j=j0; j<je; j++) printf("%11d\t", j);
	printf("\n");
	for(j=j0; j<je; j++) printf("-----------------");
	printf("\n");

	ii = i0-i0%ps;
	ip0 = i0%ps;
	for( ; ii<ie; ii+=ps)
		{
		ipe = (ie-ii)<ps ? (ie-ii): ps;
		for(ip=ip0; ip<ipe; ip++)
			{
			for(j=j0; j<je; j++)
				{
				if (j == j0) printf("%d\t| ", ii+ip);

				if ((ii+ip==err_i) && (j==err_j) && ERR)
					printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[ip+ps*j+sda*ii]);
				else if ((ii+ip >= ai) && (ii+ip < ai+m) && (j >= aj) && (j < aj+n))
					printf(ANSI_COLOR_GREEN"%9.2f\t"ANSI_COLOR_RESET, pA[ip+ps*j+sda*ii]);
				else printf("%9.2f\t", pA[ip+ps*j+sda*ii]);

				}
			printf("\n");
			ip0 = 0;
			}
		if (ipe == ps) for(j=j0; j<je; j++) printf(ANSI_COLOR_CYAN"-----------------"ANSI_COLOR_RESET);
		printf("\n");
		}

	#endif
	return;
	}


void print_input_matrices(
	const char* routine, int m, int n, int k, struct STRMAT *sA, struct STRMAT_REF *rA,
	struct STRMAT *sB, struct STRMAT_REF *rB, struct STRMAT *sC, struct STRMAT_REF *rC,
	int ai, int aj, int bi, int bj, int ci, int cj)
{
	if (strcmp(routine, "blasfeo_dgemm_nn") == 0)
	{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(m, k, sA, ai, aj, 0, 0, 0);
		print_xmat_debug(m, k, rA, ai, aj, 0, 0, 0);
		printf("\nPrint B:\n");
		blasfeo_print_xmat_debug(k, n, sB, bi, bj, 0, 0, 0);
		print_xmat_debug(k, n, rB, bi, bj, 0, 0, 0);
		printf("\nPrint C:\n");
		blasfeo_print_xmat_debug(m, n, sC, ci, cj, 0, 0, 0);
		print_xmat_debug(m, n, rC, ci, cj, 0, 0, 0);
	}
	else if (strcmp(routine, "blasfeo_dgemm_nt") == 0)
	{
		printf("\nPrint A:\n");
		blasfeo_print_xmat_debug(m, k, sA, ai, aj, 0, 0, 0);
		print_xmat_debug(m, k, rA, ai, aj, 0, 0, 0);
		printf("\nPrint B:\n");
		blasfeo_print_xmat_debug(n, k, sB, bi, bj, 0, 0, 0);
		print_xmat_debug(n, k, rB, bi, bj, 0, 0, 0);
		printf("\nPrint C:\n");
		blasfeo_print_xmat_debug(m, n, sC, ci, cj, 0, 0, 0);
		print_xmat_debug(m, n, rC, ci, cj, 0, 0, 0);
	}
	else if (strncmp(routine, "blasfeo_trsm", 13) == 0)
	{
		printf("\nPrint A:\n\n");
		int maxn = (m > n)? m : n;
		blasfeo_print_xmat_debug(maxn, maxn, sA, ai, aj, 0, 0, 0);
		print_xmat_debug(maxn, maxn, rA, ai, aj, 0, 0, 0);
		printf("\nPrint B:\n\n");
		blasfeo_print_xmat_debug(m, n, sB, bi, bj, 0, 0, 0);
		print_xmat_debug(m, n, rB, bi, bj, 0, 0, 0);
	}
	else
	{
	}
}


static void printbits(void *c, size_t n)
{
	unsigned char *t = c;
	if (c == NULL)
	return;
	while (n > 0)
	{
		int q;
		--n;
		for(q = 0x80; q; q >>= 1) printf("%x", !!(t[n] & q));
	}
	printf("\n");
}

// 1 to 1 comparison of every element
int GECMP_LIBSTR(
	int m, int n, int bi, int bj,
	struct STRMAT *sD, struct STRMAT_REF *rD,
	int* err_i, int* err_j, int debug)
	{
	int ii, jj;

	for(ii = 0; ii < m; ii++)
		{
		for(jj = 0; jj < n; jj++)
			{

			// strtucture mat
			REAL sbi = MATEL_LIBSTR(sD, ii, jj);
			// reference mat
			REAL rbi = MATEL_REF(rD, ii, jj);

			if ( (sbi != rbi) & ( fabs(sbi-rbi) > 1e-13*(fabs(sbi)+fabs(rbi)) ) & ( fabs(sbi-rbi) > 1e-12))
				{
					*err_i = ii;
					*err_j = jj;
					if (!debug) return 0;

					printf("\n\nFailed at index %d,%d, (HP) %2.18f != %2.18f (RF)\n", ii, jj, sbi, rbi);
					printf("Absolute error: %3.5e\n", fabs(sbi-rbi));
					printf("Relative error: %3.5e\n", fabs(sbi-rbi)/(fabs(sbi)+fabs(rbi)));
					printf("\nBitwise comparison:\n");
					printf("HP:  ");
					printbits(&sbi, sizeof(REAL));
					printf("REF: ");
					printbits(&rbi, sizeof(REAL));
					printf("\n");

					printf("\nResult matrix:\n");
					blasfeo_print_xmat_debug(m, n, sD, bi, bj, ii, jj, 1);
					print_xmat_debug(m, n, rD, bi, bj, ii, jj, 1);

					return 0;
				}
			}
		}

	return 1;
	}
