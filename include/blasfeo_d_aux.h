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

#ifdef __cplusplus
extern "C" {
#endif


/*
 * ----------- TOMOVE
 *
 * expecting column major matrices
 *
 */

void dtrcp_l_lib(int m, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
void dgead_lib(int m, int n, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
// TODO remove ???
void ddiain_sqrt_lib(int kmax, double *x, int offset, double *pD, int sdd);
// TODO ddiaad1
void ddiareg_lib(int kmax, double reg, int offset, double *pD, int sdd);

/*
 * Data format: STRMAT
 *
 */


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

// --- conversion
//
void blasfeo_pack_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_pack_dvec(int m, double *a, struct blasfeo_dvec *sa, int ai);
void blasfeo_pack_tran_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_unpack_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *A, int lda);
void blasfeo_unpack_dvec(int m, struct blasfeo_dvec *sa, int ai, double *a);
void blasfeo_unpack_tran_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *A, int lda);

// --- cast
//
void d_cast_mat2strmat(double *A, struct blasfeo_dmat *sA);
void d_cast_diag_mat2strmat(double *dA, struct blasfeo_dmat *sA);
void d_cast_vec2vecmat(double *a, struct blasfeo_dvec *sa);

// --- insert/extract
//
// <= sA[ai, aj]
void blasfeo_dgein1(double a, struct blasfeo_dmat *sA, int ai, int aj);
// <= sA[ai, aj]
double blasfeo_dgeex1(struct blasfeo_dmat *sA, int ai, int aj);
// sx[xi] <= a
void blasfeo_dvecin1(double a, struct blasfeo_dvec *sx, int xi);
// <= sx[xi]
double blasfeo_dvecex1(struct blasfeo_dvec *sx, int xi);
// A <= alpha

// --- set
void blasfeo_dgese(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// a <= alpha
void blasfeo_dvecse(int m, double alpha, struct blasfeo_dvec *sx, int xi);
// B <= A

// ------ copy / scale
//
// --- matrix

// B <= A
void blasfeo_dgecp(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// A <= alpha*A
void blasfeo_dgesc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// B <= alpha*A
void blasfeo_dgecpsc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);

// --- vector
// y <= x
void blasfeo_dveccp(int m, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci);
// x <= alpha*x
void blasfeo_dvecsc(int m, double alpha, struct blasfeo_dvec *sa, int ai);
// TODO
// x <= alpha*x
void blasfeo_dveccpsc(int m, double alpha, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci);

// B <= A, A lower triangular
void blasfeo_dtrcp_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrcpsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj);

// B <= B + alpha*A
void blasfeo_dgead(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// y <= y + alpha*x
void blasfeo_dvecad(int m, double alpha, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci);

// --- traspositions
void dgetr_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dgetr_libstr(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void dtrtr_l_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_l_libstr(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void dtrtr_u_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_u_libstr(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void ddiare_libstr(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
void ddiain_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void ddiaex_lib(int kmax, double alpha, int offset, double *pD, int sdd, double *x);
void ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void ddiain_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiain_sp_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void ddiaex_libsp(int kmax, int *idx, double alpha, double *pD, int sdd, double *x);
void ddiaex_libstr(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void ddiaex_sp_libstr(int kmax, double alpha, int *idx, struct blasfeo_dmat *sD, int di, int dj, struct blasfeo_dvec *sx, int xi);
void ddiaad_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiaad_sp_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd);
void ddiaadin_sp_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void drowin_lib(int kmax, double alpha, double *x, double *pD);
void drowin_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void drowex_lib(int kmax, double alpha, double *pD, double *x);
void drowex_libstr(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void drowad_lib(int kmax, double alpha, double *x, double *pD);
void drowad_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void drowin_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
void drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD);
void drowad_sp_libstr(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD);
void drowsw_lib(int kmax, double *pA, double *pC);
void drowsw_libstr(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void drowpe_libstr(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void drowpei_libstr(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void dcolex_libstr(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
void dcolin_libstr(int kmax, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
void dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
void dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dcolsw_libstr(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void dcolpe_libstr(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void dcolpei_libstr(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void dvecin_libsp(int kmax, int *idx, double *x, double *y);
void dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y);
void dvecad_sp_libstr(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
void dvecin_sp_libstr(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
void dvecex_sp_libstr(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int x, struct blasfeo_dvec *sz, int zi);
void dveccl_libstr(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi);
void dveccl_mask_libstr(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi, struct blasfeo_dvec *sm, int mi);
void dvecze_libstr(int m, struct blasfeo_dvec *sm, int mi, struct blasfeo_dvec *sv, int vi, struct blasfeo_dvec *se, int ei);
void dvecnrm_inf_libstr(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm);
void dvecpe_libstr(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);
void dvecpei_libstr(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);



#ifdef __cplusplus
}
#endif
