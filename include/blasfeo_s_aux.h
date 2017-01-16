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



#ifdef __cplusplus
extern "C" {
#endif



// s_aux_extern_depend_lib
void s_zeros(float **pA, int row, int col);
void s_zeros_align(float **pA, int row, int col);
void s_free(float *pA);
void s_free_align(float *pA);
void s_print_mat(int row, int col, float *A, int lda);
void s_print_mat_e(int row, int col, float *A, int lda);
void s_print_pmat(int row, int col, float *pA, int sda);
void s_print_pmat_e(int row, int col, float *pA, int sda);

// s_aux_lib
void s_cvt_mat2pmat(int row, int col, float *A, int lda, int offset, float *pA, int sda);
void s_cvt_tran_mat2pmat(int row, int col, float *A, int lda, int offset, float *pA, int sda);
void s_cvt_pmat2mat(int row, int col, int offset, float *pA, int sda, float *A, int lda);
void s_cvt_tran_pmat2mat(int row, int col, int offset, float *pA, int sda, float *A, int lda);



#ifdef __cpluspluc
}
#endif
