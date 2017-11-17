#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x));

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

	int ii;

	int n = 9;

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

	s_allocate_strmat(6, 6, &sD);

	// -------- instantiate s_strmat

	// compute memory size
	int size_strmat = 5*s_size_strmat(n, n);
	// inizilize void pointer
	void *memory_strmat;
	// initialize pointer
	v_zeros_align(&memory_strmat, size_strmat);
	// get point to strmat
	char *ptr_memory_strmat = (char *) memory_strmat;

	// -------- instantiate s_strmat

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

	float alpha = 0.1;
	printf("Scale A by %f and copy in B, print B:\n\n", alpha);

	sgecpsc_libstr(n, n, alpha, &sA, 0, 0, &sB, 0, 0);
	s_print_strmat(n, n, &sB, 0, 0);

	printf("----------- Scale\n\n");
	alpha = 0.3;
	printf("Scale A by %f, print A:\n\n", alpha);
	sgesc_libstr(n, n, alpha, &sA, 0, 0);
	s_print_strmat(n, n, &sA, 0, 0);

	printf("----------- Copy\n\n");
	printf("Copy submatrix A[3:5, 3:5] in B[0:5, 0:5], print B:\n\n");
	sgecp_libstr(5, 5, &sA, 3, 3, &sB, 0, 0);
	s_print_strmat(n, n, &sB, 0, 0);


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
