/* prints a matrix in column-major format */
void print_xmat_debug(int m, int n, struct STRMAT_REF *sA, int ai, int aj, int err_i, int err_j)
	{
	const int subsize = 6;
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int j0,i0, ie, je;
	int i, j;

	i0 = err_i-subsize;
	j0 = err_j-subsize;

	if (i0 < 0) i0 = 0;
	if (j0 < 0) j0 = 0;

	ie = err_i+subsize;
	je = err_j+subsize;

	if (ie > m) ie = m;
	if (je > n) je = n;

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
			if ((i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+lda*j]);
			else printf("%9.2f\t", pA[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	return;
	}

/* prints a matrix in panel-major format */
void blasfeo_print_xmat_debug(int m, int n, struct STRMAT *sA, int ai, int aj, int err_i, int err_j)
	{
	const int ps = PS;
	const int subsize = 6;
	int i0, j0, ie, je;

	i0 = err_i-subsize;
	j0 = err_j-subsize;

	if (i0 < 0) i0 = 0;
	if (j0 < 0) j0 = 0;

	ie = err_i+subsize;
	je = err_j+subsize;

	if (ie > m) ie = m;
	if (je > n) je = n;

	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	ii = i0-i0%ps;
	printf("%s\t", "HP");
	for(j=j0; j<je; j++) printf("%11d\t", j);
	printf("\n");
	for(j=j0; j<je; j++) printf("-----------------");
	printf("\n");
	if(ai%ps>0)
		{
		tmp = ps-ai%ps;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=j0; j<je; j++)
				{
				if (j == j0) printf("%d\t| ", i);
				if ((i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j]);
				else printf("%9.2f\t", pA[i+ps*j]);
				}
			printf("\n");
			}
		pA += tmp + ps*(sda-1);
		m -= tmp;
		}

	int ip0 = i0%ps;
	for( ; ii<ie-(ps-1); ii+=ps)
		{
		for(i=ip0; i<ps; i++)
			{
			for(j=j0; j<je; j++)
				{
				if (j == j0) printf("%d\t| ", ii+i);
				if ((ii+i==err_i) && (j==err_j))
				{
					printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j+sda*ii]);
				}
				else printf("%9.2f\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			ip0 = 0;
			}
		for(j=j0; j<je; j++) printf(ANSI_COLOR_CYAN"-----------------"ANSI_COLOR_RESET);
		printf("\n");
		}
	/* printf("\n"); */

	if(ii<ie)
		{
		tmp = ie-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=j0; j<je; j++)
				{
				if (j == j0) printf("%d\t| ", ii+i);
				if ((ii+i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j+sda*ii]);
				else printf("%9.2f\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
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
					blasfeo_print_xmat_debug(ii+offset, jj+offset, sB, 0, 0, ii, jj);
					print_xmat_debug(ii+offset, jj+offset, rB, 0, 0, ii, jj);

					/* if (debug<2) return 0; */

					/* printf("A matrix \n"); */
					/* [> printf("\nPrint D HP:\n\n"); <] */
					/* blasfeo_print_xmat_debug(ii+offset, jj+offset, sA, 0, 0, ii, jj); */

					/* [> printf("\nPrint D REF:\n\n"); <] */
					/* print_xmat_debug(ii+offset, jj+offset, rA, 0, 0, ii, jj); */

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






int GECMP_BLAS_LIBSTR(int n, int m,
				 struct STRMAT *sB, struct STRMAT_REF *rB
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

			if ( (sbi != rbi) & ( fabs(sbi-rbi) > 1e-10*(fabs(sbi)+fabs(rbi)) ) )
				{
					printf("\n\nFailed at index %d,%d, (HP) %f != %f (RF)\n\n", ii, jj, sbi, rbi);

					printf("\nPrint B HP:\n\n");
					PRINT_STRMAT(ii+offset, jj+offset, sB, 0, 0);

					printf("\nPrint B REF:\n\n");
					PRINT_STRMAT_REF(ii+offset, jj+offset, rB, 0, 0);

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
