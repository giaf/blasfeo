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

// A panel-major bs=4; B, C, D column-major
// 12x4
void kernel_dgemm_nn_12x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_12x4_vs_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_12x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_12x4_vs_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_12x4_lib4cc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dpotrf_nt_l_12x4_lib4cc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *dD);
// 8x4
void kernel_dgemm_nn_8x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_8x4_vs_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_8x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_8x4_vs_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_8x4_lib4cc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dpotrf_nt_l_8x4_lib4cc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *dD);
// 4x4
void kernel_dgemm_nn_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_4x4_vs_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_4x4_vs_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_4x4_lib4cc(int kmax, double *A, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4cc(int kmax, double *A, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
void kernel_dpotrf_nt_l_4x4_lib4cc(int kmax, double *A, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *dD);

// A, B panel-major bs=4; C, D column-major
// 12x4
void kernel_dgemm_nt_12x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_12x4_vs_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_12x4_lib44c(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dpotrf_nt_l_12x4_lib44c(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD);
// 8x4
void kernel_dgemm_nt_8x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_8x4_vs_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_8x4_lib44c(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dpotrf_nt_l_8x4_lib44c(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD);
// 4x4
void kernel_dgemm_nt_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_4x4_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
void kernel_dpotrf_nt_l_4x4_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *dD);
void kernel_dpotrf_nt_l_4x4_vs_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *dD, int m1, int n1);

// pack
// 12
void kernel_dpack_nn_12_lib4(int kmax, double *A, int lda, double *B, int sdb);
void kernel_dpack_nn_12_vs_lib4(int kmax, double *A, int lda, double *B, int sdb, int m1);
// 8
void kernel_dpack_nn_8_lib4(int kmax, double *A, int lda, double *B, int sdb);
void kernel_dpack_nn_8_vs_lib4(int kmax, double *A, int lda, double *B, int sdb, int m1);
// 4
void kernel_dpack_nn_4_lib4(int kmax, double *A, int lda, double *B);
void kernel_dpack_nn_4_vs_lib4(int kmax, double *A, int lda, double *B, int m1);
void kernel_dpack_tn_4_lib4(int kmax, double *A, int lda, double *B);
void kernel_dpack_tn_4_vs_lib4(int kmax, double *A, int lda, double *B, int n1);
