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



// level 2 BLAS
// 4
void kernel_sgemv_n_4_lib4(int k, float *A, float *x, int alg, float *y, float *z);
void kernel_sgemv_n_4_vs_lib4(int k, float *A, float *x, int alg, float *y, float *z, int km);
void kernel_sgemv_t_4_lib4(int k, float *A, int sda, float *x, int alg, float *y, float *z);
void kernel_sgemv_t_4_vs_lib4(int k, float *A, int sda, float *x, int alg, float *C, float *D, int km);
void kernel_strsv_ln_inv_4_lib4(int k, float *A, float *inv_diag_A, float *x, float *y, float *z);
void kernel_strsv_ln_inv_4_vs_lib4(int k, float *A, float *inv_diag_A, float *x, float *y, float *z, int km, int kn);
void kernel_strsv_lt_inv_4_lib4(int k, float *A, int sda, float *inv_diag_A, float *x, float *y, float *z);
void kernel_strsv_lt_inv_3_lib4(int k, float *A, int sda, float *inv_diag_A, float *x, float *y, float *z);
void kernel_strsv_lt_inv_2_lib4(int k, float *A, int sda, float *inv_diag_A, float *x, float *y, float *z);
void kernel_strsv_lt_inv_1_lib4(int k, float *A, int sda, float *inv_diag_A, float *x, float *y, float *z);
void kernel_strmv_un_4_lib4(int k, float *A, float *x, int alg, float *y, float *z);
void kernel_strmv_ut_4_lib4(int k, float *A, int sda, float *x, int alg, float *y, float *z);
void kernel_strmv_ut_4_vs_lib4(int k, float *A, int sda, float *x, int alg, float *C, float *D, int km);



// level 3 BLAS
// 4x4
void kernel_sgemm_nt_4x4_a0_lib4(int k, float *alpha, float *A, float *B, float *D); //
void kernel_sgemm_nt_4x4_lib4(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D); //
void kernel_sgemm_nt_4x4_vs_lib4(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D, int km, int kn); //
void kernel_sgemm_nn_4x4_lib4(int k, float *alpha, float *A, float *B, int sdb, float *beta, float *C, float *D); //
void kernel_sgemm_nn_4x4_vs_lib4(int k, float *alpha, float *A, float *B, int sdb, float *beta, float *C, float *D, int km, int kn); //
void kernel_ssyrk_nt_l_4x4_lib4(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D); //
void kernel_ssyrk_nt_l_4x4_vs_lib4(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D, int km, int kn); //
void kernel_strmm_nt_ru_4x4_lib4(int k, float *A, float *B, int alg, float *C, float *D);
void kernel_strmm_nt_ru_4x4_vs_lib4(int k, float *A, float *B, int alg, float *C, float *D, int km, int kn);
void kernel_strsm_nt_rl_inv_4x4_lib4(int k, float *A, float *B, float *C, float *D, float *E, float *inv_diag_E);
void kernel_strsm_nt_rl_inv_4x4_vs_lib4(int k, float *A, float *B, float *C, float *D, float *E, float *inv_diag_E, int km, int kn);



// LAPACK
// 4x4
void kernel_spotrf_nt_l_4x4_lib4(int k, float *A, float *B, float *C, float *D, float *inv_diag_D);
void kernel_spotrf_nt_l_4x4_vs_lib4(int k, float *A, float *B, float *C, float *D, float *inv_diag_D, int km, int kn);



// merged routines
// 4x4
void kernel_sgemm_strsm_nt_rl_inv_4x4_lib4(int kp, float *Ap, float *Bp, int km_, float *Am, float *Bm, float *C, float *D, float *E, float *inv_diag_E);
void kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(int kp, float *Ap, float *Bp, int km_, float *Am, float *Bm, float *C, float *D, float *E, float *inv_diag_E, int km, int kn);
void kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(int kp, float *Ap, float *Bp, int km_, float *Am, float *Bm, float *C, float *D, float *inv_diag_D, int km, int kn);
