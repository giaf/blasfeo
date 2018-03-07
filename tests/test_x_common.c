/* prints a matrix in column-major format */
void print_xmat_debug(int m, int n, struct STRMAT_REF *sA, int ai, int aj, int err_i, int err_j)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int i, j;
	printf("%s\t", "REF");
	for(j=0; j<n; j++) printf("%11d\t", j);
	printf("\n");
	for(j=0; j<n; j++) printf("-----------------");
	printf("\n");

	for(i=0; i<m; i++)
		{
		for(j=0; j<n; j++)
			{
			if (j == 0)  printf("%d\t", i);
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
	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	ii = 0;
	printf("%s\t", "HP");
	for(j=0; j<n; j++) printf("%11d\t", j);
	printf("\n");
	if(ai%ps>0)
		{
		tmp = ps-ai%ps;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				if (j == 0) printf("%d\t", i);
				if ((i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j]);
				else printf("%9.2f\t", pA[i+ps*j]);
				}
			printf("\n");
			}
		pA += tmp + ps*(sda-1);
		m -= tmp;
		}


	for( ; ii<m-(ps-1); ii+=ps)
		{
		for(j=0; j<n; j++) printf("-----------------");
		printf("\n");
		for(i=0; i<ps; i++)
			{
			for(j=0; j<n; j++)
				{
				if (j == 0) printf("%d\t", ii+i);
				if ((ii+i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j+sda*ii]);
				else printf("%9.2f\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	for(j=0; j<n; j++) printf("-----------------");
	printf("\n");

	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				if (j == 0) printf("%d\t", ii+i);
				if ((ii+i==err_i) && (j==err_j)) printf(ANSI_COLOR_RED"%9.2f\t"ANSI_COLOR_RESET, pA[i+ps*j+sda*ii]);
				else printf("%9.2f\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
	return;
	}


// 1 to 1 comparison of every element
int GECMP_LIBSTR(int n, int m,
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

			if ( (sbi != rbi) & ( fabs(sbi-rbi) > 1e-10*(fabs(sbi)+fabs(rbi)) ) )
				{
					if (!debug) return 0;

					printf("\n\nFailed at index %d,%d, (HP) %f != %f (RF)\n\n", ii, jj, sbi, rbi);

					/* printf("\nPrint D HP:\n\n"); */
					blasfeo_print_xmat_debug(ii+offset, jj+offset, sB, 0, 0, ii, jj);

					/* printf("\nPrint D REF:\n\n"); */
					print_xmat_debug(ii+offset, jj+offset, rB, 0, 0, ii, jj);

					/* printf("\nPrint A HP:\n\n"); */
					/* PRINT_STRMAT(ii+offset, jj+offset, sA, 0, 0); */

					/* printf("\nPrint A REF:\n\n"); */
					/* PRINT_STRMAT_REF(ii+offset, jj+offset, rA, 0, 0); */

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
