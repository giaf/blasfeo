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



//
// level 1 BLAS
//

// y = y + alpha*x
void saxpy_lib(int kmax, float alpha, float *x, float *y);
void saxpy_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi);
// z = y, y = y + alpha*x
void saxpy_bkp_lib(int kmax, float alpha, float *x, float *y, float *z);
void saxpy_bkp_libstr(int m, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);



//
// level 2 BLAS
//

// z <= beta * y + alpha * A * x
void sgemv_n_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A' * x
void sgemv_t_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// y <= inv( A ) * x
void strsv_ln_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
void strsv_lnn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x
void strsv_lt_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
void strsv_ltn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A * x ; A upper triangular
void strmv_unn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A' * x ; A upper triangular
void strmv_utn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A * x ; A lower triangular
void strmv_lnn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A' * x ; A lower triangular
void strmv_ltn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z_n <= beta_n * y_n + alpha_n * A  * x_n
// z_t <= beta_t * y_t + alpha_t * A' * x_t
void sgemv_nt_lib(int m, int n, float alpha_n, float alpha_t, float *pA, int sda, float *x_n, float *x_t, float beta_n, float beta_t, float *y_n, float *y_t, float *z_n, float *z_t);
void sgemv_nt_libstr(int m, int n, float alpha_n, float alpha_t, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx_n, int xi_n, struct s_strvec *sx_t, int xi_t, float beta_n, float beta_t, struct s_strvec *sy_n, int yi_n, struct s_strvec *sy_t, int yi_t, struct s_strvec *sz_n, int zi_n, struct s_strvec *sz_t, int zi_t);
// z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
void ssymv_l_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
	


//
// level 3 BLAS
//

// dense

// D <= beta * C + alpha * A * B^T
void sgemm_nt_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void sgemm_nt_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B
void sgemm_nn_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void sgemm_nn_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void ssyrk_nt_l_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void ssyrk_ln_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^T ; B upper triangular
void strmm_nt_ru_lib(int m, int n, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void strmm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A ; A lower triangular
void strmm_rlnn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
void strsm_nt_rl_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *pB, int sdb, float *pD, int sdd);
void strsm_rltn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular with unit diagonal
void strsm_nt_rl_one_lib(int m, int n, float *pA, int sda, float *pB, int sdb, float *pD, int sdd);
void strsm_rltu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void strsm_nt_ru_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *pB, int sdb, float *pD, int sdd);
void strsm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A lower triangular with unit diagonal
void strsm_nn_ll_one_lib(int m, int n, float *pA, int sda, float *pB, int sdb, float *pD, int sdd);
void strsm_llnu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void strsm_nn_lu_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *pB, int sdb, float *pD, int sdd);
void strsm_lunn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);

// diagonal

// D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
void sgemm_diag_left_ib(int m, int n, float alpha, float *dA, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void sgemm_l_diag_libstr(int m, int n, float alpha, struct s_strvec *sA, int ai, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
void sgemm_diag_right_lib(int m, int n, float alpha, float *pA, int sda, float *dB, float beta, float *pC, int sdc, float *pD, int sdd);
void sgemm_r_diag_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sB, int bi, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void spotrf_nt_l_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
void spotrf_l_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= chol( C + A * B' ) ; C, D lower triangular
void ssyrk_dpotrf_nt_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
void ssyrk_dpotrf_ln_libstr(int m, int n, int k, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= lu( C ) ; no pivoting
void sgetrf_nn_nopivot_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
void sgetrf_nopivot_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= lu( C ) ; pivoting
void sgetrf_nn_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D, int *ipiv);
void sgetrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, int *ipiv);



#ifdef __cplusplus
}
#endif
