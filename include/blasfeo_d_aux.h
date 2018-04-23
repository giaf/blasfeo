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

/*
 * auxiliary algebra operations header
 *
 * include/blasfeo_aux_lib*.h
 *
 */

#include "blasfeo_common.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifndef BLASFEO_D_AUX
#define BLASFEO_D_AUX


#include "blasfeo_common.h"
#include "blasfeo_d_aux_old.h"



// --- memory calculations
//
// returns the memory size (in bytes) needed for a dmat
int blasfeo_memsize_dmat(int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a dmat
int blasfeo_memsize_diag_dmat(int m, int n);
// returns the memory size (in bytes) needed for a dvec
int blasfeo_memsize_dvec(int m);

// --- creation
//
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_dmat(int m, int n, struct blasfeo_dmat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_dvec(int m, struct blasfeo_dvec *sA, void *memory);

// --- packing
// pack the column-major matrix A into the matrix struct B
void blasfeo_pack_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// transpose and pack the column-major matrix A into the matrix struct B
void blasfeo_pack_tran_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// pack the vector x into the vector structure y
void blasfeo_pack_dvec(int m, double *x, struct blasfeo_dvec *sy, int yi);
// unpack the matrix structure A into the column-major matrix B
void blasfeo_unpack_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *B, int ldb);
// transpose and unpack the matrix structure A into the column-major matrix B
void blasfeo_unpack_tran_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *B, int ldb);
// pack the vector structure x into the vector y
void blasfeo_unpack_dvec(int m, struct blasfeo_dvec *sx, int xi, double *y);

// --- cast
//
void d_cast_mat2strmat(double *A, struct blasfeo_dmat *sA); // TODO
void d_cast_diag_mat2strmat(double *dA, struct blasfeo_dmat *sA); // TODO
void d_cast_vec2vecmat(double *a, struct blasfeo_dvec *sx); // TODO

// --- insert/extract
//
// sA[ai, aj] <= a
void blasfeo_dgein1(double a, struct blasfeo_dmat *sA, int ai, int aj);
// <= sA[ai, aj]
double blasfeo_dgeex1(struct blasfeo_dmat *sA, int ai, int aj);
// sx[xi] <= a
void blasfeo_dvecin1(double a, struct blasfeo_dvec *sx, int xi);
// <= sx[xi]
double blasfeo_dvecex1(struct blasfeo_dvec *sx, int xi);

// --- set
// A <= alpha
void blasfeo_dgese(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// a <= alpha
void blasfeo_dvecse(int m, double alpha, struct blasfeo_dvec *sx, int xi);

// --- copy / scale
// B <= A
void blasfeo_dgecp(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// A <= alpha*A
void blasfeo_dgesc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// B <= alpha*A
void blasfeo_dgecpsc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// y <= x
void blasfeo_dveccp(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);
// x <= alpha*x
void blasfeo_dvecsc(int m, double alpha, struct blasfeo_dvec *sx, int xi);
// y <= alpha*x
void blasfeo_dveccpsc(int m, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);
// B <= A, A lower triangular
void blasfeo_dtrcp_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrcpsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj);

// --- sum
// B <= B + alpha*A
void blasfeo_dgead(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int yi, int cj);
// y <= y + alpha*x
void blasfeo_dvecad(int m, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);

// --- traspositions
// B <= A'
void blasfeo_dgetr(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// B <= A', A lower triangular
void blasfeo_dtrtr_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// B <= A', A upper triangular
void blasfeo_dtrtr_u(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);

// --- operations on diagonal
// diag(A) += alpha
void blasfeo_ddiare(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A) <= alpha*x
void blasfeo_ddiain(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A)[idx] <= alpha*x
void blasfeo_ddiain_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// x <= diag(A)
void blasfeo_ddiaex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
// x <= diag(A)[idx]
void blasfeo_ddiaex_sp(int kmax, double alpha, int *idx, struct blasfeo_dmat *sD, int di, int dj, struct blasfeo_dvec *sx, int xi);
// diag(A) += alpha*x
void blasfeo_ddiaad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A)[idx] += alpha*x
void blasfeo_ddiaad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// diag(A)[idx] = y + alpha*x
void blasfeo_ddiaadin_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, int *idx, struct blasfeo_dmat *sD, int di, int dj);

// TODO comment

void blasfeo_drowin(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_drowex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void blasfeo_drowad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_drowad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_drowsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void blasfeo_drowpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_drowpei(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_dcolex(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void blasfeo_dcolin(int kmax, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_dcolsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void blasfeo_dcolpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_dcolpei(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_dvecad_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
void blasfeo_dvecin_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
void blasfeo_dvecex_sp(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int x, struct blasfeo_dvec *sz, int zi);
void blasfeo_dveccl(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi);
void blasfeo_dveccl_mask(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi, struct blasfeo_dvec *sm, int mi);
void blasfeo_dvecze(int m, struct blasfeo_dvec *sm, int mi, struct blasfeo_dvec *sv, int vi, struct blasfeo_dvec *se, int ei);
void blasfeo_dvecnrm_inf(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm);
void blasfeo_dvecpe(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);
void blasfeo_dvecpei(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);

#endif

#ifdef __cplusplus
}
#endif
