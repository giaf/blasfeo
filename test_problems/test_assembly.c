#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"

int main()
	{

	printf("\ntest assembly\n");

	int ii;

	int n = 12;

	double *A; d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;
	d_print_mat(n, n, A, n);

	double *B; d_zeros(&B, n, n);
	for(ii=0; ii<n; ii++) B[ii*(n+1)] = 1.0;
	d_print_mat(n, n, B, n);

	struct d_strmat sA;
	d_allocate_strmat(n, n, &sA);
	d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);
	d_print_strmat(n, n, &sA, 0, 0);

	struct d_strmat sB;
	d_allocate_strmat(n, n, &sB);
	d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);
	d_print_strmat(n, n, &sB, 0, 0);

	struct d_strmat sD;
	d_allocate_strmat(n, n, &sD);

	struct d_strmat sC;
	d_allocate_strmat(n, n, &sC);

	double alpha = 1.0;
	double beta = 0.0;
	int ret = kernel_dgemm_nt_4x4_lib4_test(n, &alpha, sB.pA, sA.pA, &beta, sB.pA, sD.pA);
	d_print_strmat(n, n, &sD, 0, 0);
//	printf("\n%ld %ld\n", (long long) n, ret);
//	printf("\n%ld %ld\n", (long long) &alpha, ret);
//	printf("\n%ld %ld\n", (long long) sA.pA, ret);
//	printf("\n%ld %ld\n", (long long) sB.pA, ret);
//	printf("\n%ld %ld\n", (long long) &beta, ret);
//	printf("\n%ld %ld\n", (long long) sC.pA, ret);
//	printf("\n%ld %ld\n", (long long) sD.pA, ret);

	return 0;

	}
