/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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



// d_aux_extern_depend_lib
void d_zeros(double **pA, int row, int col);
void d_zeros_align(double **pA, int row, int col);
void d_zeros_align_bytes(double **pA, int size);
void d_free(double *pA);
void d_free_align(double *pA);
void d_print_mat(int row, int col, double *A, int lda);
void d_print_mat_e(int row, int col, double *A, int lda);
void d_print_pmat(int row, int col, double *pA, int sda);
void d_print_pmat_e(int row, int col, double *pA, int sda);
void v_zeros(void **ptrA, int size);
void v_zeros_align(void **ptrA, int size);
void v_free(void *ptrA);
void v_free_align(void *ptrA);
void d_allocate_strmat(int m, int n, struct d_strmat *sA);
void d_free_strmat(struct d_strmat *sA);
void d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);

// d_aux_lib
void d_cvt_mat2pmat(int row, int col, double *A, int lda, int offset, double *pA, int sda);
void d_cvt_tran_mat2pmat(int row, int col, double *A, int lda, int offset, double *pA, int sda);
void d_cvt_pmat2mat(int row, int col, int offset, double *pA, int sda, double *A, int lda);
void d_cvt_tran_pmat2mat(int row, int col, int offset, double *pA, int sda, double *A, int lda);
int d_size_strmat(int m, int n);
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory);
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
