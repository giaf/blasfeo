
// 1 to 1 comparison of every element
int GECMP_LIBSTR(int n, int m, struct STRMAT *sA, struct STRMAT *rA)
	{
	int ii, jj;

	for(ii = 0; ii <= n; ii++)
		{
		for(jj = 0; jj <= m; jj++)
			{

			REAL sai = MATEL_LIBSTR(sA, ii, jj);
			REAL rai = MATEL_LIB(rA, ii, jj);

			if ( (sai != rai) & (fabs(sai-rai) > 1e-10*(fabs(sai) + fabs(rai))))
				{
					printf("\n\nFailed at index %d,%d, %f != %f\n\n", ii, jj, sai, rai);
					return 0;
				}
			}
		}

	return 1;
	}
