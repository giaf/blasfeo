#include <stdlib.h>
#include <stdio.h>
#include <blasfeo_common.h>         // matrix and vector struct definition
#include <blasfeo_d_aux.h>          // auxiliary routines (e.g. pack, copy)
#include <blasfeo_d_aux_ext_dep.h>  // allocation and printing routines, double precision
#include <blasfeo_v_aux_ext_dep.h>  // allocation, void
#include <blasfeo_d_blas.h>         // linear algebra routines

int main()
    {

    int ii;  // loop index

    int n = 12;  // matrix size

    // A
    struct blasfeo_dmat sA;            // matrix structure
    blasfeo_allocate_dmat(n, n, &sA);  // allocate and assign memory needed by A

    // B
    struct blasfeo_dmat sB;                       // matrix structure
    int B_size = blasfeo_memsize_dmat(n, n);      // size of memory needed by B
    void *B_mem_align;
    v_zeros_align(&B_mem_align, B_size);          // allocate memory needed by B
    blasfeo_create_dmat(n, n, &sB, B_mem_align);  // assign aligned memory to struct

    // C
    struct blasfeo_dmat sC;                                                  // matrix structure
    int C_size = blasfeo_memsize_dmat(n, n);                                 // size of memory needed by C
    C_size += 64;                                                            // 64-bytes alignment
    void *C_mem = malloc(C_size);
    void *C_mem_align = (void *) ((((unsigned long long) C_mem)+63)/64*64);  // align memory pointer
    blasfeo_create_dmat(n, n, &sC, C_mem_align);                             // assign aligned memory to struct

    // A
    double *A = malloc(n*n*sizeof(double));
    for(ii=0; ii<n*n; ii++)
        A[ii] = ii;
    int lda = n;
    blasfeo_pack_dmat(n, n, A, lda, &sA, 0, 0);  // convert from column-major to BLASFEO dmat
    free(A);

    // B
    blasfeo_dgese(n, n, 0.0, &sB, 0, 0);    // set B to zero
    for(ii=0; ii<n; ii++)
        BLASFEO_DMATEL(&sB, ii, ii) = 1.0;  // set B diagonal to 1.0 accessing dmat elements

    // C
    blasfeo_dgese(n, n, -1.0, &sC, 0, 0);  // set C to -1.0

    blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sC, 0, 0);

    printf("\nC = \n");
    blasfeo_print_dmat(n, n, &sC, 0, 0);

    blasfeo_free_dmat(&sA);
    v_free_align(B_mem_align);
    free(C_mem);

    return 0;

    }

