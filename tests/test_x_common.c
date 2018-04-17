void print_compilation_flags(){
	SHOW_DEFINE(LA)
	SHOW_DEFINE(TARGET)
	SHOW_DEFINE(PRECISION)
	SHOW_DEFINE(MIN_KERNEL_SIZE)
	SHOW_DEFINE(ROUTINE)
}

/* prints a matrix in column-major format */
void print_xmat_debug(int m, int n, struct STRMAT_REF *sA, int ai, int aj, int err_i, int err_j, int ERR)
	{
	const int subsize = 6;
	int lda = sA->m;
	/* REAL *pA = sA->pA + ai + aj*lda; */
	REAL *pA = sA->pA;
	int j0,i0, ie, je;
	int i, j;

	i0 = err_i-subsize;
	j0 = err_j-subsize;

	if (i0 < ai) i0 = ai;
	if (j0 < aj) j0 = aj;

	ie = err_i+subsize;
	je = err_j+subsize;

	if (ie > ai+m) ie = ai+m;
	if (je > aj+n) je = aj+n;

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
			if ((i==err_i) && (j==err_j) && ERR) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+lda*j]);
			else printf("%9.2f\t", pA[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	return;
	}

/* prints a matrix in panel-major format */
void blasfeo_print_xmat_debug(int m, int n, struct STRMAT *sA, int ai, int aj, int err_i, int err_j, int ERR)
	{
	#if defined(LA_BLAS_WRAPPER) || defined(LA_REFERENCE)
	/* print_xmat_debug(m, n, sA->pA, ai, aj, err_i, err_j); */
	#else
	const int ps = PS;
	const int subsize = 6;

	int i0, j0, ie, je;
	int ii, j, ip0, ipe, ip;

	i0 = err_i-subsize;
	j0 = err_j-subsize;

	if (i0 < ai) i0 = ai;
	if (j0 < aj) j0 = aj;

	ie = err_i+subsize;
	je = err_j+subsize;

	if (ie > ai+m) ie = ai+m;
	if (je > aj+n) je = aj+n;

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
				{
					printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[ip+ps*j+sda*ii]);
				}
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


void print_routine_signature(const char* routine, REAL alpha, REAL beta, int m, int n, int k,
							 int ai, int aj, int bi, int bj, int ci, int cj, int di, int dj)
{
	if (strcmp(routine, "blasfeo_dgemm_nn") == 0)
	{
		printf(
			"Calling D[%d:%d,%d:%d] =  %f*A[%d:%d,%d:%d]*B[%d:%d,%d:%d] + %f*C[%d:%d,%d:%d]\n",
			di, m, dj, n,
			alpha, ai, m, aj, k,
			bi, k, bj, n,
			beta, ci, m, cj, n
		);
	}
	else if (strcmp(routine, "blasfeo_dgemm_nt") == 0)
	{
		printf(
			"Calling D[%d:%d,%d:%d] =  %f*A[%d:%d,%d:%d]*B[%d:%d,%d:%d] + %f*C[%d:%d,%d:%d]\n",
			di, m, dj, n,
			alpha, ai, m, aj, k,
			bi, n, bj, k,
			beta, ci, m, cj, n
		);
	}
	else if ((strncmp(routine, "blasfeo_dtrsm", 13) == 0) || (strncmp(routine, "blasfeo_dtrmm", 13) == 0))
	{
		int maxn = (m > n)? m : n;
		printf(
			"Calling %f*A[%d:%d,%d:%d]*X[%d:%d,%d:%d] = %f*B[%d:%d,%d:%d]\n",
			alpha, ai, maxn, aj, maxn,
			di, m, dj, n,
			beta, bi, m, bj, n
		);
	}
	else
	{
		printf("Do not know: %s\n", routine);
		printf(
			"D[%d:%d,%d:%d], %f*A[%d:%d,%d:%d]*B[%d:%d,%d:%d], %f*C[%d:%d,%d:%d]\n",
			di, m, dj, n,
			alpha, ai, m, aj, k,
			bi, k, bj, n,
			beta, ci, m, cj, n
		);
	}
}


void print_input_matrices(const char* routine, int m, int n, int k,
						  struct STRMAT *sA, struct STRMAT_REF *rA,
						  struct STRMAT *sB, struct STRMAT_REF *rB,
						  struct STRMAT *sC, struct STRMAT_REF *rC,
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
int GECMP_LIBSTR(int m, int n,
				 struct STRMAT *sB, struct STRMAT_REF *rB,
				 struct STRMAT *sA, struct STRMAT_REF *rA,
				 int* err_i, int* err_j,
				 int debug
				 )
	{
	int ii, jj;
	const int offset = 8;

	for(ii = 0; ii < m; ii++)
		{
		for(jj = 0; jj < n; jj++)
			{

			// strtucture mat
			REAL sbi = MATEL_LIBSTR(sB, ii, jj);
			// reference mat
			REAL rbi = MATEL_REF(rB, ii, jj);

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
					blasfeo_print_xmat_debug(ii+offset, jj+offset, sB, 0, 0, ii, jj, 1);
					print_xmat_debug(ii+offset, jj+offset, rB, 0, 0, ii, jj, 1);

					return 0;
				}
			}
		}

	return 1;
	}
