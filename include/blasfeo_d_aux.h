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

#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif



/************************************************
* d_aux_lib.c
************************************************/

// returns the memory size (in bytes) needed for a strmat
int d_size_strmat(int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a strmat
int d_size_diag_strmat(int m, int n);
// returns the memory size (in bytes) needed for a strvec
int d_size_strvec(int m);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void d_create_strvec(int m, struct d_strvec *sA, void *memory);
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_vec2strvec(int m, double *a, struct d_strvec *sa, int ai);
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void d_cvt_strvec2vec(int m, struct d_strvec *sa, int ai, double *a);
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void d_cast_mat2strmat(double *A, struct d_strmat *sA);
void d_cast_diag_mat2strmat(double *dA, struct d_strmat *sA);
void d_cast_vec2vecmat(double *a, struct d_strvec *sa);
void dmatin1_libstr(double a, struct d_strmat *sA, int ai, int aj);
double dmatex1_libstr(struct d_strmat *sA, int ai, int aj);
void dvecin1_libstr(double a, struct d_strvec *sx, int xi);
double dvecex1_libstr(struct d_strvec *sx, int xi);
void dmatse_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj);
void dvecse_libstr(int m, double alpha, struct d_strvec *sx, int xi);
void dgecp_lib(int m, int n, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
void dgecp_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dveccp_libstr(int m, double alpha, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci);
void dtrcp_l_lib(int m, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
void dtrcp_l_libstr(int m, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dgead_lib(int m, int n, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
void dgead_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dvecad_libstr(int m, double alpha, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci);
void dgetr_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dgetr_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dtrtr_l_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_l_libstr(int m, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dtrtr_u_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_u_libstr(int m, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void ddiareg_lib(int kmax, double reg, int offset, double *pD, int sdd);
void ddiain_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void ddiain_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
void ddiain_sqrt_lib(int kmax, double *x, int offset, double *pD, int sdd);
void ddiaex_lib(int kmax, double alpha, int offset, double *pD, int sdd, double *x);
void ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void ddiain_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiain_libspstr(int kmax, int *idx, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sD, int di, int dj);
void ddiaex_libsp(int kmax, int *idx, double alpha, double *pD, int sdd, double *x);
void ddiaex_libspstr(int kmax, int *idx, double alpha, struct d_strmat *sD, int di, int dj, struct d_strvec *sx, int xi);
void ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiaad_libspstr(int kmax, int *idx, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sD, int di, int dj);
void ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd);
void ddiaadin_libspstr(int kmax, int *idx, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi, struct d_strmat *sD, int di, int dj);
void drowin_lib(int kmax, double alpha, double *x, double *pD);
void drowin_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
void drowex_lib(int kmax, double alpha, double *pD, double *x);
void drowex_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi);
void drowad_lib(int kmax, double alpha, double *x, double *pD);
void drowad_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
void drowin_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
void drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD);
void drowad_libspstr(int kmax, int *idx, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sD, int di, int dj);
void drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD);
void drowsw_lib(int kmax, double *pA, double *pC);
void drowsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void drowpe_libstr(int kmax, int *ipiv, struct d_strmat *sA);
void dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
void dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
void dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
void dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dcolsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
void dcolpe_libstr(int kmax, int *ipiv, struct d_strmat *sA);
void dvecin_libsp(int kmax, int *idx, double *x, double *y);
void dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y);
void dvecad_libspstr(int kmax, int *idx, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi);



#ifdef __cplusplus
}
#endif
