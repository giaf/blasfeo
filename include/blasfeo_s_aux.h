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
int s_size_strmat(int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a strmat
int s_size_diag_strmat(int m, int n);
// returns the memory size (in bytes) needed for a strvec
int s_size_strvec(int m);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void s_create_strmat(int m, int n, struct s_strmat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void s_create_strvec(int m, struct s_strvec *sA, void *memory);
void s_cvt_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj);
void s_cvt_vec2strvec(int m, float *a, struct s_strvec *sa, int ai);
void s_cvt_tran_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj);
void s_cvt_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda);
void s_cvt_strvec2vec(int m, struct s_strvec *sa, int ai, float *a);
void s_cvt_tran_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda);
void s_cast_mat2strmat(float *A, struct s_strmat *sA);
void s_cast_diag_mat2strmat(float *dA, struct s_strmat *sA);
void s_cast_vec2vecmat(float *a, struct s_strvec *sa);

void sgein1_libstr(float a, struct s_strmat *sA, int ai, int aj);
float sgeex1_libstr(struct s_strmat *sA, int ai, int aj);
void svecin1_libstr(float a, struct s_strvec *sx, int xi);
float svecex1_libstr(struct s_strvec *sx, int xi);

// A <= alpha
void sgese_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj);
// a <= alpha
void svecse_libstr(int m, float alpha, struct s_strvec *sx, int xi);

// copy and scale
// void sgecp_lib(int m, int n, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
//
void sgecpsc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void sgecp_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void sgesc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj);

void sveccp_libstr(int m, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci);
void svecsc_libstr(int m, float alpha, struct s_strvec *sa, int ai);
void sveccpsc_libstr(int m, float alpha, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci);

void strcp_l_lib(int m, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void strcp_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

void sgead_lib(int m, int n, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void sgead_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void svecad_libstr(int m, float alpha, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci);

void sgetr_lib(int m, int n, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void sgetr_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

void strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_u_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

void sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd);
void sdiaex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi);
void sdiain_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
void sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd);
void sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x);
void sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiain_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
void sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x);
void sdiaex_sp_libstr(int kmax, float alpha, int *idx, struct s_strmat *sD, int di, int dj, struct s_strvec *sx, int xi);
void sdiaad_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
void sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiaad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
void sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd);
void sdiaadin_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, int *idx, struct s_strmat *sD, int di, int dj);
void srowin_lib(int kmax, float alpha, float *x, float *pD);
void srowin_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
void srowex_lib(int kmax, float alpha, float *pD, float *x);
void srowex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi);
void srowad_lib(int kmax, float alpha, float *x, float *pD);
void srowad_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
void srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD);
void srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD);
void srowad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
void srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD);
void srowsw_lib(int kmax, float *pA, float *pC);
void srowsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void srowpe_libstr(int kmax, int *ipiv, struct s_strmat *sA);
void srowpei_libstr(int kmax, int *ipiv, struct s_strmat *sA);
void scolin_lib(int kmax, float *x, int offset, float *pD, int sdd);
void scolin_libstr(int kmax, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
void scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd);
void scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd);
void scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void scolsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void scolpe_libstr(int kmax, int *ipiv, struct s_strmat *sA);
void scolpei_libstr(int kmax, int *ipiv, struct s_strmat *sA);
void svecin_libsp(int kmax, int *idx, float *x, float *y);
void svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y);
void svecad_sp_libstr(int m, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sz, int zi);
void svecin_sp_libstr(int m, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sz, int zi);
void svecex_sp_libstr(int m, float alpha, int *idx, struct s_strvec *sx, int x, struct s_strvec *sz, int zi);
void sveccl_libstr(int m, struct s_strvec *sxm, int xim, struct s_strvec *sx, int xi, struct s_strvec *sxp, int xip, struct s_strvec *sz, int zi);
void sveccl_mask_libstr(int m, struct s_strvec *sxm, int xim, struct s_strvec *sx, int xi, struct s_strvec *sxp, int xip, struct s_strvec *sz, int zi, struct s_strvec *sm, int mi);
void svecze_libstr(int m, struct s_strvec *sm, int mi, struct s_strvec *sv, int vi, struct s_strvec *se, int ei);
void svecnrm_inf_libstr(int m, struct s_strvec *sx, int xi, float *ptr_norm);
void svecpe_libstr(int kmax, int *ipiv, struct s_strvec *sx, int xi);
void svecpei_libstr(int kmax, int *ipiv, struct s_strvec *sx, int xi);



#ifdef __cplusplus
}
#endif

