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

#ifndef BLASFEO_D_BLASFEO_API_H_
#define BLASFEO_D_BLASFEO_API_H_



#include "blasfeo_common.h"



#ifdef __cplusplus
extern "C" {
#endif



//
// level 1 BLAS
//

// z = y + alpha*x
// z[zi:zi+n] = alpha*x[xi:xi+n] + y[yi:yi+n]
// NB: Different arguments semantic compare to equivalent standard BLAS routine
void blasfeo_daxpy(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z = beta*y + alpha*x
void blasfeo_daxpby(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z = x .* y
void blasfeo_dvecmul(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z += x .* y
void blasfeo_dvecmulacc(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z = x .* y, return sum(z) = x^T * y
double blasfeo_dvecmuldot(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// return x^T * y
double blasfeo_ddot(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);
// construct givens plane rotation
void blasfeo_drotg(double a, double b, double *c, double *s);
// apply plane rotation [a b] [c -s; s; c] to the aj0 and aj1 columns of A at row index ai
void blasfeo_dcolrot(int m, struct blasfeo_dmat *sA, int ai, int aj0, int aj1, double c, double s);
// apply plane rotation [c s; -s c] [a; b] to the ai0 and ai1 rows of A at column index aj
void blasfeo_drowrot(int m, struct blasfeo_dmat *sA, int ai0, int ai1, int aj, double c, double s);



//
// level 2 BLAS
//

// dense

// z <= beta * y + alpha * A * x
void blasfeo_dgemv_n(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z <= beta * y + alpha * A' * x
void blasfeo_dgemv_t(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A ) * x, A (m)x(n)
void blasfeo_dtrsv_lnn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(n)
void blasfeo_dtrsv_ltn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A ) * x, A (m)x(m) lower, not_transposed, not_unit
void blasfeo_dtrsv_lnn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A ) * x, A (m)x(m) lower, not_transposed, unit
void blasfeo_dtrsv_lnu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) lower, transposed, not_unit
void blasfeo_dtrsv_ltn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) lower, transposed, unit
void blasfeo_dtrsv_ltu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) upper, not_transposed, not_unit
void blasfeo_dtrsv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= inv( A' ) * x, A (m)x(m) upper, transposed, not_unit
void blasfeo_dtrsv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= beta * y + alpha * A * x ; A upper triangular
void blasfeo_dtrmv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= A' * x ; A upper triangular
void blasfeo_dtrmv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= A * x ; A lower triangular
void blasfeo_dtrmv_lnn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= A' * x ; A lower triangular
void blasfeo_dtrmv_ltn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= A * x ; A lower triangular, unit diagonal
void blasfeo_dtrmv_lnu(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z <= A' * x ; A lower triangular, unit diagonal
void blasfeo_dtrmv_ltu(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z_n <= beta_n * y_n + alpha_n * A  * x_n
// z_t <= beta_t * y_t + alpha_t * A' * x_t
void blasfeo_dgemv_nt(int m, int n, double alpha_n, double alpha_t, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx_n, int xi_n, struct blasfeo_dvec *sx_t, int xi_t, double beta_n, double beta_t, struct blasfeo_dvec *sy_n, int yi_n, struct blasfeo_dvec *sy_t, int yi_t, struct blasfeo_dvec *sz_n, int zi_n, struct blasfeo_dvec *sz_t, int zi_t);
// z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
void blasfeo_dsymv_l(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);

// diagonal

// z <= beta * y + alpha * A * x, A diagonal
void blasfeo_dgemv_d(int m, double alpha, struct blasfeo_dvec *sA, int ai, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi);



//
// level 3 BLAS
//

// dense

// D <= beta * C + alpha * A * B
void blasfeo_dgemm_nn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T
void blasfeo_dgemm_nt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A^T * B
void blasfeo_dgemm_tn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A^T * B^T
void blasfeo_dgemm_tt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void blasfeo_dsyrk_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_dsyrk_ln_mn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A^T * B ; C, D lower triangular
void blasfeo_dsyrk_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A * B^T ; C, D upper triangular
void blasfeo_dsyrk_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= beta * C + alpha * A^T * B ; C, D upper triangular
void blasfeo_dsyrk_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * B * A^T ; B upper triangular
void blasfeo_dtrmm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * B * A ; A lower triangular
void blasfeo_dtrmm_rlnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A lower triangular employint explicit inverse of diagonal
void blasfeo_dtrsm_llnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A lower triangular with unit diagonal
void blasfeo_dtrsm_llnu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void blasfeo_dtrsm_lunn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
void blasfeo_dtrsm_rltn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A lower triangular with unit diagonal
void blasfeo_dtrsm_rltu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void blasfeo_dtrsm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj);

// diagonal

// D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
void dgemm_diag_left_lib(int m, int n, double alpha, double *dA, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);
void blasfeo_dgemm_dn(int m, int n, double alpha, struct blasfeo_dvec *sA, int ai, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
void blasfeo_dgemm_nd(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sB, int bi, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void blasfeo_dpotrf_l(int m, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_dpotrf_l_mn(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= chol( C + A * B' ) ; C, D lower triangular
void blasfeo_dsyrk_dpotrf_ln(int m, int k, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_dsyrk_dpotrf_ln_mn(int m, int n, int k, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= lu( C ) ; no pivoting
void blasfeo_dgetrf_nopivot(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
// D <= lu( C ) ; pivoting
void blasfeo_dgetrf_rowpivot(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj, int *ipiv);
// D <= qr( C )
void blasfeo_dgeqrf(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj, void *work);
int blasfeo_dgeqrf_worksize(int m, int n); // in bytes
// D <= lq( C )
void blasfeo_dgelqf(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj, void *work);
int blasfeo_dgelqf_worksize(int m, int n); // in bytes
// D <= lq( C ), positive diagonal elements
void blasfeo_dgelqf_pd(int m, int n, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj, void *work);
// [L, A] <= lq( [L, A] ), positive diagonal elements, array of matrices, with
// L lower triangular, of size (m)x(m)
// A full, of size (m)x(n1)
void blasfeo_dgelqf_pd_la(int m, int n1, struct blasfeo_dmat *sL, int li, int lj, struct blasfeo_dmat *sA, int ai, int aj, void *work);
// [L, L, A] <= lq( [L, L, A] ), positive diagonal elements, array of matrices, with:
// L lower triangular, of size (m)x(m)
// A full, of size (m)x(n1)
void blasfeo_dgelqf_pd_lla(int m, int n1, struct blasfeo_dmat *sL0, int l0i, int l0j, struct blasfeo_dmat *sL1, int l1i, int l1j, struct blasfeo_dmat *sA, int ai, int aj, void *work);



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLASFEO_API_H_
