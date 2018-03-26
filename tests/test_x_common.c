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
	#if defined(LA_BLAS_WRAPPER)
	/* print_xmat_debug(m, n, sA->pA, ai, aj, err_i, err_j); */
	#else
	const int ps = PS;
	const int subsize = 6;
	int i0, j0, ie, je, ip0, ipe, ip;

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
	int ii, i, j, tmp;

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
				if (j == j0) printf("%d\t| ", ii+i);
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

					/* printf("fabs(sbi-rbi) %2.18f \n", fabs(sbi-rbi)); */
					/* printf("fabs(sbi)+fabs(rbi) %2.18f \n", fabs(sbi)+fabs(rbi)); */

					/* printf("\nPrint D\n"); */
					blasfeo_print_xmat_debug(ii+offset, jj+offset, sB, 0, 0, ii, jj, 1);
					print_xmat_debug(ii+offset, jj+offset, rB, 0, 0, ii, jj, 1);

					#if defined(LA)
					SHOW_DEFINE(LA)
					#endif

					#if defined(TARGET)
					SHOW_DEFINE(TARGET)
					#endif

					#if defined(PRECISION)
					SHOW_DEFINE(PRECISION)
					#endif

					#if defined(MIN_KERNEL_SIZE)
					SHOW_DEFINE(MIN_KERNEL_SIZE)
					#endif


					return 0;
				}
			}
		}

	return 1;
	}
