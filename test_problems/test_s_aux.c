#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_aux_test.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x));

#include "test_s_common.h"
#include "test_x_common.c"


int main()
	{

#ifndef LA
	#error LA undefined
#endif

#ifndef TARGET
	#error TARGET undefined
#endif


	printf("\n\n\n--------------- Single Precision --------------------\n\n\n");

SHOW_DEFINE(LA)
SHOW_DEFINE(TARGET)


	int ii, jj;

	int n = 21;
	int p_n = 15;
	int N = 10;

	//
	// matrices in column-major format
	//
	float *A;
	// standard column major allocation (malloc)
	s_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;

	float *B;
	// standard column major allocation (malloc)
	s_zeros(&B, n, n);
	for(ii=0; ii<n*n; ii++) B[ii] = 2*ii;

	// standard column major allocation (malloc)
	float *C;
	s_zeros(&C, n, n);

	// using blasfeo allocate
	/* struct s_strmat sD;                          */
	/* s_allocate_strmat(n-1, n-1, &sD);            */

	/* -------- instantiate s_strmat */

	// compute memory size
	int size_strmat = N*s_size_strmat(n, n);
	// inizilize void pointer
	void *memory_strmat;

	// initialize pointer
	// memory allocation
	v_zeros_align(&memory_strmat, size_strmat);

	// get point to strmat
	char *ptr_memory_strmat = (char *) memory_strmat;

	// -------- instantiate s_strmat
	printf("\nInstantiate matrices\n\n");

	// instantiate s_strmat depend on compilation flag LA_BLAS || LA_REFERENCE
	struct s_strmat sA;
	s_create_strmat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memory_size;
	s_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);

	struct s_strmat sB;
	s_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	s_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);

	// Testing comparison
	// reference matrices, column major

	struct s_strmat rA;
	test_s_create_strmat(n, n, &rA, ptr_memory_strmat);
	ptr_memory_strmat += rA.memory_size;
	test_s_cvt_mat2strmat(n, n, A, n, &rA, 0, 0);

	struct s_strmat rB;
	test_s_create_strmat(n, n, &rB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	test_s_cvt_mat2strmat(n, n, B, n, &rB, 0, 0);


	// -------- instantiate s_strmat

	// test operations
	//
	/* sgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0); */

	/* printf("\nPrint mat B:\n\n"); */
	/* s_print_mat(p_n, p_n, B, n); */

	printf("\nPrint strmat HP A:\n\n");
	s_print_strmat(p_n, p_n, &sA, 0, 0);

	printf("\nPrint strmat REF A:\n\n");
	test_s_print_strmat(p_n, p_n, &rA, 0, 0);

	printf("\nPrint strmat HP B:\n\n");
	s_print_strmat(p_n, p_n, &sB, 0, 0);

	printf("\nPrint strmat REF B:\n\n");
	test_s_print_strmat(p_n, p_n, &rB, 0, 0);


	/* printf("\nPrint stored strmat A:\n\n"); */
	/* s_print_strmat((&sA)->pm, (&sA)->cn, &sA, 0, 0); */

	/* printf("\nPrint strmat B:\n\n"); */
	/* s_print_strmat(p_n, p_n, &sB, 0, 0); */

	/* AUX */

	/* ----------- memory */
	/* printf("----------- STRMAT memory\n\n"); */
	/* for (int i=0; i<12; i++) */
	/* { */
		/* printf("%d: %f, %f\n", i, sA.pA[i], A[i]); */
	/* } */
	/* printf("...\n\n"); */


	/* ---------- extraction */
	/* printf("----------- Extraction\n\n"); */

	/* int ai = 8; */
	/* int aj = 1; */

	// ---- strmat
	/* float ex_val = sgeex1_libstr(&sA, ai, aj); */
	/* printf("Extract %d,%d for A: %f\n\n", ai, aj, ex_val); */

	/* ---- column major */
	/* struct s_strmat* ssA = &sA; */
	/* int lda = (&sA)->m; */
	/* float pointer + n_rows + n_col*leading_dimension; */
	/* float *pA = (&sA)->pA + ai + aj*lda; */
	/* float val = pA[0]; */

	/* ----------- copy and scale */
	printf("\n\n----------- TEST Copy&Scale\n\n");

	float alpha;
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

		sgesc_libstr(     ni, mi, alpha, &sA, ii, 0);
		test_sgesc_libstr(ni, mi, alpha, &rA, ii, 0);

		assert(sgecmp_libstr(n, n, &sA, &rA, &sA, &rA));

		// loop over B offset
		for (jj = 0; jj < 8; jj++)
			{

			// ---- Copy&Scale
			//

			printf("Copy-Scale A[%d:%d,%d:%d] by %f in B[%d:%d,%d:%d]\n",
							     ii,ni, 0,mi,    alpha,  jj,ni, 0,mi);

			// HP submatrix copy&scale
			sgecpsc_libstr(ni, mi, alpha, &sA, ii, 0, &sB, jj, 0);
			// REF submatrix copy&scale
			test_sgecpsc_libstr(ni, mi, alpha, &rA, ii, 0, &rB, jj, 0);
			// check against blas with blasfeo REF
			assert(sgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

			// ---- Copy
			//
			printf("Copy A[%d:%d,%d:%d] in B[%d:%d,%d:%d]\n",
							ii,ni, 0,mi,     jj,ni, 0,mi);

			sgecp_libstr(     ni, mi, &sA, ii, 0, &sB, jj, 0);
			test_sgecp_libstr(ni, mi, &rA, ii, 0, &rB, jj, 0);
			assert(sgecmp_libstr(n, n, &sB, &rB, &sA, &rA));

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
