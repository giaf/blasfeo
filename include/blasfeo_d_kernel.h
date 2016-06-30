/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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



// level 2 BLAS
// 12
void kernel_dgemv_n_12_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
void kernel_dgemv_t_12_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
// 8
void kernel_dgemv_n_8_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
void kernel_dgemv_t_8_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
// 4
void kernel_dgemv_n_4_lib4(int k, double *A, double *x, int alg, double *y, double *z);
void kernel_dgemv_n_4_vs_lib4(int k, double *A, double *x, int alg, double *y, double *z, int km);
void kernel_dgemv_t_4_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
void kernel_dgemv_t_4_vs_lib4(int k, double *A, int sda, double *x, int alg, double *C, double *D, int km);
void kernel_dtrsv_ln_inv_4_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z);
void kernel_dtrsv_ln_inv_4_vs_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z, int km, int kn);
void kernel_dtrsv_lt_inv_4_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);
void kernel_dtrsv_lt_inv_3_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);
void kernel_dtrsv_lt_inv_2_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);
void kernel_dtrsv_lt_inv_1_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);
void kernel_dtrmv_un_4_lib4(int k, double *A, double *x, int alg, double *y, double *z);
void kernel_dtrmv_ut_4_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);
void kernel_dtrmv_ut_4_vs_lib4(int k, double *A, int sda, double *x, int alg, double *C, double *D, int km);



// level 3 BLAS
// 8x4
void kernel_dgemm_ntnn_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dgemm_ntnn_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dgemm_ntnt_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dgemm_ntnt_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dgemm_nttn_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dgemm_nttn_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dgemm_nttt_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dgemm_nttt_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dsyrk_ntnn_l_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dsyrk_ntnn_l_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dtrmm_ntnn_ru_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);
void kernel_dtrmm_ntnn_ru_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);
void kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);
void kernel_dtrsm_ntnn_rl_inv_8x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);
void kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km_, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);
void kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km_, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);
// 4x4
void kernel_dgemm_ntnn_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dgemm_ntnn_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dgemm_ntnt_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dgemm_ntnt_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dgemm_nttn_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dgemm_nttn_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dgemm_nttt_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dgemm_nttt_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dsyrk_ntnn_l_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dsyrk_ntnn_l_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dtrmm_ntnn_ru_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);
void kernel_dtrmm_ntnn_ru_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
void kernel_dtrsm_ntnn_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);
void kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);
void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E);
void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);



// LAPACK
// 8x4
void kernel_dpotrf_ntnn_l_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);
void kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km_, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);
// 4x4
void kernel_dpotrf_ntnn_l_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn);
void kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *inv_diag_D, int km, int kn);



