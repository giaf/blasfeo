// 1 to 1 comparison of every element
int GECMP_LIBSTR(int n, int m,
				 struct XMAT *sB, struct XMAT_REF *rB,
				 struct XMAT *sA, struct XMAT_REF *rA
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
					PRINT_XMAT(ii+offset, jj+offset, sB, 0, 0);

					printf("\nPrint B REF:\n\n");
					PRINT_XMAT_REF(ii+offset, jj+offset, rB, 0, 0);

					printf("\nPrint A HP:\n\n");
					PRINT_XMAT(ii+offset, jj+offset, sA, 0, 0);

					printf("\nPrint A REF:\n\n");
					PRINT_XMAT_REF(ii+offset, jj+offset, rA, 0, 0);

					#if defined(LA)
					SHOW_DEFINE(LA)
					#endif

					#if defined(TARGET)
					SHOW_DEFINE(TARGET)
					#endif

					#if defined(PRECISION)
					SHOW_DEFINE(PRECISION)
					#endif

					return 0;
				}
			}
		}

	return 1;
	}
