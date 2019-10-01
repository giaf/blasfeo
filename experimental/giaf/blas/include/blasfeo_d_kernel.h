/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

// A panel-major bs=4; B, C, D column-major
// 12x4
void kernel_dgemm_nn_12x4_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_12x4_vs_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_12x4_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_12x4_vs_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_12x4_lib4ccc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
// 8x4
void kernel_dgemm_nn_8x4_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_8x4_vs_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_8x4_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_8x4_vs_lib4ccc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_8x4_lib4ccc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(int kmax, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
// 4x4
void kernel_dgemm_nn_4x4_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nn_4x4_vs_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dgemm_nt_4x4_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_4x4_vs_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);

// A, B panel-major bs=4; C, D column-major
// 12x4
void kernel_dgemm_nt_12x4_lib44cc(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_12x4_vs_lib44cc(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_12x4_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_12x4_vs_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
void kernel_dpotrf_nt_l_12x4_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD);
void kernel_dpotrf_nt_l_12x4_vs_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD, int m1, int n1);
// 8x4
void kernel_dgemm_nt_8x4_lib44cc(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_8x4_vs_lib44cc(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_8x4_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_8x4_vs_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
void kernel_dpotrf_nt_l_8x4_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD);
void kernel_dpotrf_nt_l_8x4_vs_lib44cc(int kmax, double *A, int sda, double *B, double *C, int ldc, double *D, int ldd, double *dD, int m1, int n1);
// 4x4
void kernel_dgemm_nt_4x4_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);
void kernel_dgemm_nt_4x4_vs_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);
void kernel_dtrsm_nt_rl_inv_4x4_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE);
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *dE, int m1, int n1);
void kernel_dpotrf_nt_l_4x4_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *dD);
void kernel_dpotrf_nt_l_4x4_vs_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *dD, int m1, int n1);

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
