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

SHOW_DEFINE(LA)
SHOW_DEFINE(TARGET)


	int ii, jj;

	int n = 25;
	int p_n = 16;
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

	float *C;
	// standard column major allocation (malloc)
	s_zeros(&C, n, n);

	struct s_strmat sD;

	s_allocate_strmat(n-1, n-1, &sD);

	// -------- instantiate s_strmat

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
	// allocate memory for strtmat
	/* s_allocate_strmat(n, n, &sA); */
	// use pre-allocated memory for strmat
	s_create_strmat(n, n, &sA, ptr_memory_strmat);
	// update memory pointer
	ptr_memory_strmat += sA.memory_size;
	// use A data and sA memeory
	s_cvt_mat2strmat(n, n, A, n, &sA, 0, 0);

	struct s_strmat sB;
	s_create_strmat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	s_cvt_mat2strmat(n, n, B, n, &sB, 0, 0);

	struct s_strmat sC;
	s_create_strmat(n, n, &sC, ptr_memory_strmat);
	ptr_memory_strmat += sC.memory_size;
	s_cvt_mat2strmat(n, n, C, n, &sC, 0, 0);

	// reference matrices

	struct s_strmat rA;
	s_create_strmat(n, n, &rA, ptr_memory_strmat);
	ptr_memory_strmat += rA.memory_size;
	test_s_cvt_mat2strmat(n, n, A, n, &rA, 0, 0);

	struct s_strmat rB;
	s_create_strmat(n, n, &rB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memory_size;
	test_s_cvt_mat2strmat(n, n, B, n, &rB, 0, 0);


	// -------- instantiate s_strmat

	// test operations
	//
	/* sgemm_nt_libstr(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0); */

	printf("\nPrint strmat A:\n\n");
	s_print_strmat(n, n, &sA, 0, 0);

	printf("\nPrint strmat B:\n\n");
	s_print_strmat(n, n, &sB, 0, 0);

	/* AUX */

	/* ----------- memory */
	printf("----------- STRMAT memory\n\n");
	for (int i=0; i<9; i++)
	{
		printf("%d: %f\n", i, sA.pA[i]);
	}
	printf("...\n\n");


	/* ---------- extraction */
	printf("----------- Extraction\n\n");

	int ai = 8;
	int aj = 1;

	// ---- strmat
	float ex_val = sgeex1_libstr(&sA, ai, aj);
	printf("Extract %d,%d for A: %f\n\n", ai, aj, ex_val);

	// ---- column major
	/* struct s_strmat *ssA = &sA; */
	/* int lda = ssA->m; */
	/* float pointer + n_rows + n_col*leading_dimension; */
	/* float *pA = ssA->pA + ai + aj*lda; */
	/* float val = pA[0]; */

	/* ----------- copy and scale */
	printf("----------- Copy&Scale\n\n");

	float alpha;
	alpha = 4;
	int ret, ni, mi;
	ni = 12;
	mi = 12;

	printf("Compute different combinations of submatrix offsets\n\n");

	// loop over A offset
	for (ii = 0; ii < 8; ii++)
		{
		// loop over B offset
		printf("A, B offset");
		for (jj = 0; jj < 8; jj++)
			{

			sgecpsc_libstr(ni, mi, alpha, &sA, ii, 0, &sB, jj, 0);

			// compute rA with reference routine
			test_sgecpsc_libstr(ni, mi, alpha, &rA, ii, 0, &rB, jj, 0);

			// check against blas with blasfeo REF
			printf(", %d_%d", ii, jj);
			assert(sgecmp_libstr(n, n, &sB, &rB));


			}
			printf("\n");
		}
		printf("\n");
	printf("-------\n\n\n");

	printf("Scale A by %f and copy in B, print B:\n\n", alpha);
	sgecpsc_libstr(n, n, alpha, &sA, 0, 0, &sB, 0, 0);
	s_print_strmat(p_n, p_n, &sB, 0, 0);

	printf("----------- Scale\n\n");
	alpha = 0.3;
	printf("Scale A by %f, print A:\n\n", alpha);
	sgesc_libstr(n, n, alpha, &sA, 0, 0);
	s_print_strmat(p_n, p_n, &sA, 0, 0);

	printf("----------- Copy\n\n");
	printf("Copy submatrix A[3:5, 3:5] in B[0:5, 0:5], print B:\n\n");
	sgecp_libstr(5, 5, &sA, 3, 3, &sB, 0, 0);
	s_print_strmat(p_n, p_n, &sB, 0, 0);


	/* ----------- copy scale tringular */
	/* printf("----------- Copy&Scale\n\n"); */

	/* sgecp_libstr(n, n, &sC, 0, 0, &sB, 0, 0); */
	/* strcpsc_l_libstr(n, 0.1, &sA, 0, 0, &sB, 0, 0); */
	/* printf("Scale trl A by 0.1 and copy in B, print B:\n\n"); */
	/* s_print_strmat(n, n, &sB, 0, 0); */

	/* strcp_l_libstr(5, &sA, 3, 3, &sB, 0, 0); */
	/* printf("Copy trl submatrix A[3:5, 3:5] in B[0:5, 0:5], print B:\n\n"); */
	/* s_print_strmat(n, n, &sB, 0, 0); */

	/* strsc_l_libstr(n, 0.3, &sB, 0, 0); */
	/* printf("Scale trl B by 0.3, print B:\n\n"); */
	/* s_print_strmat(n, n, &sB, 0, 0); */


#if defined(LA)
SHOW_DEFINE(LA)
#endif
#if defined(TARGET)
SHOW_DEFINE(TARGET)
#endif

	}
