/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

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

