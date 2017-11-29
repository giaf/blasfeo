#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_aux_test.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x));

#include "test_d_common.h"
#include "test_x_common.c"


int main()
	{

#ifndef LA
	#error LA undefined
#endif

#ifndef TARGET
	#error TARGET undefined
#endif


	printf("\n\n\n--------------- Double Precision --------------------\n\n\n");

SHOW_DEFINE(LA)
SHOW_DEFINE(TARGET)


	int ii, jj;

	int n = 21;
	int p_n = 15;
	int N = 10;

	//
	// matrices in column-major format
	//
	double *A;
	// standard column major allocation (malloc)
	d_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;

	double *B;
	// standard column major allocation (malloc)
	d_zeros(&B, n, n);
	for(ii=0; ii<n*n; ii++) B[ii] = 2*ii;

	// standard column major allocation (malloc)
	double *C;
	d_zeros(&C, n, n);

	// using blasfeo allocate
	/* struct d_strmat sD;                          */
	/* d_allocate_strmat(n-1, n-1, &sD);            */

	/* -------- instantiate d_strmat */

	// compute memory size
	int size_strmat = N*d_size_strmat(n, n);
	// inizilize void pointer
	void *memory_strmat;

	// initialize pointer
	// memory allocation
	v_zeros_align(&memory_strmat, size_strmat);

	// get point to strmat
	char *ptr_memory_strmat = (char *) memory_strmat;

	// -------- instantiate d_strmat
	printf("\nInstantiate matrices\n\n");

	// instantiate d_strmat depend on compilation flag LA_BLAS || LA_REFERENCE
	struct d_strmat sA;
	d_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	d_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);

	struct d_strmat sB;
	d_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	d_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);

	// Testing comparison
	// reference matrices, column major

	struct d_strmat_ref rA;
	blasfeo_d_create_strmat_ref(n, n, &rA, ptr_memory_strmat);
	ptr_memory_strmat += rA.memory_size;
	blasfeo_d_cvt_mat2strmat_ref(n, n, A, n, &rA, 0, 0);

	struct d_strmat_ref rB;
	blasfeo_d_create_strmat_ref(n, n, &rB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	blasfeo_d_cvt_mat2strmat_ref(n, n, B, n, &rB, 0, 0);


	// -------- instantiate d_strmat

	printf("\nPrint strmat HP A:\n\n");
	d_print_strmat(p_n, p_n, &sA, 0, 0);

	printf("\nPrint strmat REF A:\n\n");
	blasfeo_d_print_strmat_ref(p_n, p_n, &rA, 0, 0);

	printf("\nPrint strmat HP B:\n\n");
	d_print_strmat(p_n, p_n, &sB, 0, 0);

	printf("\nPrint strmat REF B:\n\n");
	blasfeo_d_print_strmat_ref(p_n, p_n, &rB, 0, 0);

	printf("\n\n----------- TEST Copy&Scale\n\n");

	double alpha;
	alpha = 1.5;
	int ret, ni, mi;
	ni = 12;
	mi = 10;

	printf("Compute different combinations of submatrix offsets\n\n");

	// loop over A offset
	for (ii = 0; ii < 8; ii++)
		{


		// ---- Scale
		//
		printf("Scale A[%d:%d,%d:%d] by %f\n",
						ii,ni, 0,mi,    alpha);

		dgesc_libstr(     ni, mi, alpha, &sA, ii, 0);
		blasfeo_dgesc_ref(ni, mi, alpha, &rA, ii, 0);

		/* printf("value 0,1: %f", MATEL_LIBSTR(&sA, 0,1)); */
		/* printf("PS:%d", PS); */

		assert(dgecmp_libstr(n, n, &sA, &rA, &sA, &rA));

		// loop over B offset
		for (jj = 0; jj < 8; jj++)
			{

			// ---- Copy&Scale
			//

			printf("Copy-Scale A[%d:%d,%d:%d] by %f in B[%d:%d,%d:%d]\n",
							     ii,ni, 0,mi,    alpha,  jj,ni, 0,mi);

			// HP submatrix copy&scale
			dgecpsc_libstr(ni, mi, alpha, &sA, ii, 0, &sB, jj, 0);
			// REF submatrix copy&scale
			blasfeo_dgecpsc_ref(ni, mi, alpha, &rA, ii, 0, &rB, jj, 0);
			// check against blas with blasfeo REF
			assert(dgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

			// ---- Copy
			//
			printf("Copy A[%d:%d,%d:%d] in B[%d:%d,%d:%d]\n",
							ii,ni, 0,mi,     jj,ni, 0,mi);

			dgecp_libstr(     ni, mi, &sA, ii, 0, &sB, jj, 0);
			blasfeo_dgecp_ref(ni, mi, &rA, ii, 0, &rB, jj, 0);
			assert(dgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

			printf("\n");
			}

		printf("\n");
		}

	printf("\n\n----------- END TEST Copy&Scale\n\n");

#if defined(LA)
SHOW_DEFINE(LA)
#endif
#if defined(TARGET)
SHOW_DEFINE(TARGET)
#endif
#if defined(PRECISION)
SHOW_DEFINE(PRECISION)
#endif

	printf("\n\n");

	return 0;

	}
