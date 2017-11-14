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
void saxpy_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// z = x .* y, return sum(z) = x^T * y
float svecmuldot_libstr(int m, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// return x^T * y
float sdot_libstr(int m, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi);
// apply plane rotation [a b] [c -s; s; c] to two consecutive columns of A
void dcolrot_libstr(int m, struct s_strmat *sA, int ai, int aj, float c, float s);
// apply plane rotation [c s; -s c] [a; b] to two consecutive rows of A
void drowrot_libstr(int m, struct s_strmat *sA, int ai, int aj, float c, float s);



//
// level 2 BLAS
//

// dense

// z <= beta * y + alpha * A * x
void sgemv_n_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A' * x
void sgemv_t_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);
// y <= inv( A ) * x, A (m)x(n)
void strsv_lnn_mn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x, A (m)x(n)
void strsv_ltn_mn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A ) * x, A (m)x(m) lower, not_transposed, not_unit
void strsv_lnn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A ) * x, A (m)x(m) lower, not_transposed, unit
void strsv_lnu_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x, A (m)x(m) lower, transposed, not_unit
void strsv_ltn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x, A (m)x(m) lower, transposed, unit
void strsv_ltu_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x, A (m)x(m) upper, not_transposed, not_unit
void strsv_unn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// y <= inv( A' ) * x, A (m)x(m) upper, transposed, not_unit
void strsv_utn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= beta * y + alpha * A * x ; A upper triangular
void strmv_unn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= A' * x ; A upper triangular
void strmv_utn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= A * x ; A lower triangular
void strmv_lnn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z <= A' * x ; A lower triangular
void strmv_ltn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi);
// z_n <= beta_n * y_n + alpha_n * A  * x_n
// z_t <= beta_t * y_t + alpha_t * A' * x_t
void sgemv_nt_libstr(int m, int n, float alpha_n, float alpha_t, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx_n, int xi_n, struct s_strvec *sx_t, int xi_t, float beta_n, float beta_t, struct s_strvec *sy_n, int yi_n, struct s_strvec *sy_t, int yi_t, struct s_strvec *sz_n, int zi_n, struct s_strvec *sz_t, int zi_t);
// z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
void ssymv_l_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);

// diagonal

// z <= beta * y + alpha * A * x, A diagonal
void sgemv_diag_libstr(int m, float alpha, struct s_strvec *sA, int ai, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi);



//
// level 3 BLAS
//

// dense

// D <= beta * C + alpha * A * B^T
void sgemm_nt_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B
void sgemm_nn_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void ssyrk_ln_libstr(int m, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
void ssyrk_ln_mn_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^T ; B upper triangular
void strmm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A ; A lower triangular
void strmm_rlnn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
void strsm_rltn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular with unit diagonal
void strsm_rltu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void strsm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A lower triangular with unit diagonal
void strsm_llnu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void strsm_lunn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj);

// diagonal

// D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
void sgemm_diag_left_ib(int m, int n, float alpha, float *dA, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void sgemm_l_diag_libstr(int m, int n, float alpha, struct s_strvec *sA, int ai, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
void sgemm_r_diag_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sB, int bi, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void spotrf_l_libstr(int m, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
void spotrf_l_mn_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= chol( C + A * B' ) ; C, D lower triangular
void ssyrk_spotrf_ln_libstr(int m, int n, int k, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= lu( C ) ; no pivoting
void sgetrf_nopivot_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj);
// D <= lu( C ) ; pivoting
void sgetrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, int *ipiv);
// D <= qr( C )
void sgeqrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, void *work);
int sgeqrf_work_size_libstr(int m, int n); // in bytes
// D <= lq( C )
void sgelqf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, void *work);
int sgelqf_work_size_libstr(int m, int n); // in bytes




#ifdef __cplusplus
}
#endif
