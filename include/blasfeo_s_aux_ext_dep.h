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

#ifndef BLASFEO_S_AUX_EXT_DEP_H_
#define BLASFEO_S_AUX_EXT_DEP_H_


#if defined(EXT_DEP)



#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif



/************************************************
* s_aux_extern_depend_lib.c
************************************************/

/* column-major matrices */

// dynamically allocate row*col floats of memory and set accordingly a pointer to float; set allocated memory to zero
void s_zeros(float **pA, int row, int col);
// dynamically allocate row*col floats of memory aligned to 64-byte boundaries and set accordingly a pointer to float; set allocated memory to zero
void s_zeros_align(float **pA, int row, int col);
// dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to float; set allocated memory to zero
void s_zeros_align_bytes(float **pA, int size);
// free the memory allocated by d_zeros
void s_free(float *pA);
// free the memory allocated by d_zeros_align or d_zeros_align_bytes
void s_free_align(float *pA);
// print a column-major matrix
void s_print_mat(int m, int n, float *A, int lda);
// print the transposed of a column-major matrix
void s_print_tran_mat(int row, int col, float *A, int lda);
// print to file a column-major matrix
void s_print_to_file_mat(FILE *file, int row, int col, float *A, int lda);
// print to file a column-major matrix in exponential format
void s_print_to_file_exp_mat(FILE *file, int row, int col, float *A, int lda);
// print to string a column-major matrix
int s_print_to_string_mat(char **buf_out, int row, int col, float *A, int lda);
// print to file the transposed of a column-major matrix
void s_print_tran_to_file_mat(FILE *file, int row, int col, float *A, int lda);
// print to file the transposed of a column-major matrix in exponential format
void s_print_tran_to_file_exp_mat(FILE *file, int row, int col, float *A, int lda);
// print in exponential notation a column-major matrix
void s_print_exp_mat(int m, int n, float *A, int lda);
// print in exponential notation the transposed of a column-major matrix
void s_print_exp_tran_mat(int row, int col, float *A, int lda);

/* strmat and strvec */

// create a strmat for a matrix of size m*n by dynamically allocating memory
void blasfeo_allocate_smat(int m, int n, struct blasfeo_smat *sA);
// create a strvec for a vector of size m by dynamically allocating memory
void blasfeo_allocate_svec(int m, struct blasfeo_svec *sa);
// free the memory allocated by blasfeo_allocate_dmat
void blasfeo_free_smat(struct blasfeo_smat *sA);
// free the memory allocated by blasfeo_allocate_dvec
void blasfeo_free_svec(struct blasfeo_svec *sa);
// print a strmat
void blasfeo_print_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj);
// print in exponential notation a strmat
void blasfeo_print_exp_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj);
// print to file a strmat
void blasfeo_print_to_file_smat(FILE *file, int m, int n, struct blasfeo_smat *sA, int ai, int aj);
// print to file a strmat in exponential format
void blasfeo_print_to_file_exp_smat(FILE *file, int m, int n, struct blasfeo_smat *sA, int ai, int aj);
// print to string a strmat
void blasfeo_print_to_string_smat(char **buf_out, int m, int n, struct blasfeo_smat *sA, int ai, int aj);
// print a strvec
void blasfeo_print_svec(int m, struct blasfeo_svec *sa, int ai);
// print in exponential notation a strvec
void blasfeo_print_exp_svec(int m, struct blasfeo_svec *sa, int ai);
// print to file a strvec
void blasfeo_print_to_file_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
// print to string a strvec
void blasfeo_print_to_string_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);
// print the transposed of a strvec
void blasfeo_print_tran_svec(int m, struct blasfeo_svec *sa, int ai);
// print in exponential notation the transposed of a strvec
void blasfeo_print_exp_tran_svec(int m, struct blasfeo_svec *sa, int ai);
// print to file the transposed of a strvec
void blasfeo_print_to_file_tran_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
// print to string the transposed of a strvec
void blasfeo_print_to_string_tran_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);



#ifdef __cplusplus
}
#endif



#endif  // EXT_DEP

#endif  // BLASFEO_S_AUX_EXT_DEP_H_
