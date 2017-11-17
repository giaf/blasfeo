/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

/*
 * auxiliary algebra operation external dependancies header
 *
 * include/blasfeo_d_aux_ext_dep.h
 *
 * - dynamic memory allocation
 * - print
 *
 */


#if defined(EXT_DEP)



#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif


/* column-major matrices */

// dynamically allocate row*col doubles of memory and set accordingly a pointer to double; set allocated memory to zero
void d_zeros(double **pA, int row, int col);
// dynamically allocate row*col doubles of memory aligned to 64-byte boundaries and set accordingly a pointer to double; set allocated memory to zero
void d_zeros_align(double **pA, int row, int col);
// dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to double; set allocated memory to zero
void d_zeros_align_bytes(double **pA, int size);
// free the memory allocated by d_zeros
void d_free(double *pA);
// free the memory allocated by d_zeros_align or d_zeros_align_bytes
void d_free_align(double *pA);
// print a column-major matrix
void d_print_mat(int m, int n, double *A, int lda);
// print the transposed of a column-major matrix
void d_print_tran_mat(int row, int col, double *A, int lda);
// print to file a column-major matrix
void d_print_to_file_mat(FILE *file, int row, int col, double *A, int lda);
// print to file the transposed of a column-major matrix
void d_print_tran_to_file_mat(FILE *file, int row, int col, double *A, int lda);
// print in exponential notation a column-major matrix
void d_print_e_mat(int m, int n, double *A, int lda);
// print in exponential notation the transposed of a column-major matrix
void d_print_e_tran_mat(int row, int col, double *A, int lda);

/* strmat and strvec */

#ifdef BLASFEO_COMMON
// create a strmat for a matrix of size m*n by dynamically allocating memory
void d_allocate_strmat(int m, int n, struct d_strmat *sA);
// create a strvec for a vector of size m by dynamically allocating memory
void d_allocate_strvec(int m, struct d_strvec *sa);
// free the memory allocated by d_allocate_strmat
void d_free_strmat(struct d_strmat *sA);
// free the memory allocated by d_allocate_strvec
void d_free_strvec(struct d_strvec *sa);
// print a strmat
void d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);
// print in exponential notation a strmat
void d_print_e_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);
// print to file a strmat
void d_print_to_file_strmat(FILE *file, int m, int n, struct d_strmat *sA, int ai, int aj);
// print a strvec
void d_print_strvec(int m, struct d_strvec *sa, int ai);
// print in exponential notation a strvec
void d_print_e_strvec(int m, struct d_strvec *sa, int ai);
// print to file a strvec
void d_print_to_file_strvec(FILE *file, int m, struct d_strvec *sa, int ai);
// print the transposed of a strvec
void d_print_tran_strvec(int m, struct d_strvec *sa, int ai);
// print in exponential notation the transposed of a strvec
void d_print_e_tran_strvec(int m, struct d_strvec *sa, int ai);
// print to file the transposed of a strvec
void d_print_tran_to_file_strvec(FILE *file, int m, struct d_strvec *sa, int ai);
#endif



#ifdef __cplusplus
}
#endif



#endif // EXT_DEP
