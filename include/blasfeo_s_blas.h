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

#ifndef BLASFEO_S_BLAS_H_
#define BLASFEO_S_BLAS_H_

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// level 1 BLAS
//

// z = y + alpha*x
void blasfeo_saxpy(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z = beta*y + alpha*x
void blasfeo_saxpby(int kmax, float alpha, struct blasfeo_svec *sx, int xi, float beta, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z = x .* y
void blasfeo_svecmul(int m, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z += x .* y
void blasfeo_svecmulacc(int m, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z = x .* y, return sum(z) = x^T * y
float blasfeo_svecmuldot(int m, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// return x^T * y
float blasfeo_sdot(int m, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi);
// construct givens plane rotation
void blasfeo_srotg(float a, float b, float *c, float *s);
// apply plane rotation [a b] [c -s; s; c] to the aj0 and aj1 columns of A at row index ai
void blasfeo_scolrot(int m, struct blasfeo_smat *sA, int ai, int aj0, int aj1, float c, float s);
// apply plane rotation [c s; -s c] [a; b] to the ai0 and ai1 rows of A at column index aj
void blasfeo_srowrot(int m, struct blasfeo_smat *sA, int ai0, int ai1, int aj, float c, float s);



//
// level 2 BLAS
//

// dense

// z <= beta * y + alpha * A * x
void blasfeo_sgemv_n(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, float beta, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z <= beta * y + alpha * A' * x
void blasfeo_sgemv_t(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, float beta, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);
// z <= inv( A ) * x, A (m)x(n)
void blasfeo_strsv_lnn_mn(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(n)
void blasfeo_strsv_ltn_mn(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A ) * x, A (m)x(m) lower, not_transposed, not_unit
void blasfeo_strsv_lnn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A ) * x, A (m)x(m) lower, not_transposed, unit
void blasfeo_strsv_lnu(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) lower, transposed, not_unit
void blasfeo_strsv_ltn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) lower, transposed, unit
void blasfeo_strsv_ltu(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) upper, not_transposed, not_unit
void blasfeo_strsv_unn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) upper, transposed, not_unit
void blasfeo_strsv_utn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= beta * y + alpha * A * x ; A upper triangular
void blasfeo_strmv_unn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= A' * x ; A upper triangular
void blasfeo_strmv_utn(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= A * x ; A lower triangular
void blasfeo_strmv_lnn(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z <= A' * x ; A lower triangular
void blasfeo_strmv_ltn(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
// z_n <= beta_n * y_n + alpha_n * A  * x_n
// z_t <= beta_t * y_t + alpha_t * A' * x_t
void blasfeo_sgemv_nt(int m, int n, float alpha_n, float alpha_t, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx_n, int xi_n, struct blasfeo_svec *sx_t, int xi_t, float beta_n, float beta_t, struct blasfeo_svec *sy_n, int yi_n, struct blasfeo_svec *sy_t, int yi_t, struct blasfeo_svec *sz_n, int zi_n, struct blasfeo_svec *sz_t, int zi_t);
// z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
void blasfeo_ssymv_l(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi, float beta, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);

// diagonal

// z <= beta * y + alpha * A * x, A diagonal
void blasfeo_sgemv_d(int m, float alpha, struct blasfeo_svec *sA, int ai, struct blasfeo_svec *sx, int xi, float beta, struct blasfeo_svec *sy, int yi, struct blasfeo_svec *sz, int zi);



//
// level 3 BLAS
//

// dense

// D <= beta * C + alpha * A * B^T
void blasfeo_sgemm_nt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= beta * C + alpha * A * B
void blasfeo_sgemm_nn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void blasfeo_ssyrk_ln(int m, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_ssyrk_ln_mn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * B * A^T ; B upper triangular
void blasfeo_strmm_rutn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * B * A ; A lower triangular
void blasfeo_strmm_rlnn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
void blasfeo_strsm_rltn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular with unit diagonal
void blasfeo_strsm_rltu(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void blasfeo_strsm_rutn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A lower triangular with unit diagonal
void blasfeo_strsm_llnu(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void blasfeo_strsm_lunn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj);

// diagonal

// D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
void sgemm_diag_left_ib(int m, int n, float alpha, float *dA, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void blasfeo_sgemm_dn(int m, int n, float alpha, struct blasfeo_svec *sA, int ai, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
void blasfeo_sgemm_nd(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sB, int bi, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void blasfeo_spotrf_l(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_spotrf_l_mn(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= chol( C + A * B' ) ; C, D lower triangular
void blasfeo_ssyrk_spotrf_ln(int m, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_ssyrk_spotrf_ln_mn(int m, int n, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= lu( C ) ; no pivoting
void blasfeo_sgetrf_nopivot(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj);
// D <= lu( C ) ; pivoting
void blasfeo_sgetrf_rowpivot(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv);
// D <= qr( C )
void blasfeo_sgeqrf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work);
int blasfeo_sgeqrf_worksize(int m, int n); // in bytes
// D <= lq( C )
void blasfeo_sgelqf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work);
int blasfeo_sgelqf_worksize(int m, int n); // in bytes
// D <= lq( C ), positive diagonal elements
void blasfeo_sgelqf_pd(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work);
// [L, A] <= lq( [L, A] ), positive diagonal elements, array of matrices, with
// L lower triangular, of size (m)x(m)
// A full, of size (m)x(n1)
void blasfeo_sgelqf_pd_la(int m, int n1, struct blasfeo_smat *sL, int li, int lj, struct blasfeo_smat *sA, int ai, int aj, void *work);
// [L, L, A] <= lq( [L, L, A] ), positive diagonal elements, array of matrices, with:
// L lower triangular, of size (m)x(m)
// A full, of size (m)x(n1)
void blasfeo_sgelqf_pd_lla(int m, int n1, struct blasfeo_smat *sL0, int l0i, int l0j, struct blasfeo_smat *sL1, int l1i, int l1j, struct blasfeo_smat *sA, int ai, int aj, void *work);




#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_BLAS_H_
