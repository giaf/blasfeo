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

int test_s_size_strmat(int m, int n);
int test_s_size_diag_strmat(int m, int n);
int test_s_size_strvec(int m);

void test_s_create_strmat(int m, int n, struct s_strmat *sA, void *memory);
void test_s_create_strvec(int m, struct s_strvec *sA, void *memory);

void test_s_cvt_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj);
void test_s_cvt_vec2strvec(int m, float *a, struct s_strvec *sa, int ai);
void test_s_cvt_tran_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj);
void test_s_cvt_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda);
void test_s_cvt_strvec2vec(int m, struct s_strvec *sa, int ai, float *a);
void test_s_cvt_tran_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda);

void test_s_cast_mat2strmat(float *A, struct s_strmat *sA);
void test_s_cast_diag_mat2strmat(float *dA, struct s_strmat *sA);
void test_s_cast_vec2vecmat(float *a, struct s_strvec *sa);
// copy and scale
void test_sgecpsc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void test_sgecp_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
void test_sgesc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj);

// void test_sgein1_libstr(float a, struct s_strmat *sA, int ai, int aj);
// float test_sgeex1_libstr(struct s_strmat *sA, int ai, int aj);
// void test_svecin1_libstr(float a, struct s_strvec *sx, int xi);
// float test_svecex1_libstr(struct s_strvec *sx, int xi);

// // A <= alpha
// void test_sgese_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj);
// // a <= alpha
// void test_svecse_libstr(int m, float alpha, struct s_strvec *sx, int xi);


// void test_sveccp_libstr(int m, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci);
// void test_svecsc_libstr(int m, float alpha, struct s_strvec *sa, int ai);

// void test_strcp_l_lib(int m, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
// void test_strcp_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

// void test_sgead_lib(int m, int n, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
// void test_sgead_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
// void test_svecad_libstr(int m, float alpha, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci);

// void test_sgetr_lib(int m, int n, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_sgetr_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

// void test_strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_strtr_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
// void test_strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_strtr_u_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);

// void test_sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd);
// void test_sdiaex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi);
// void test_sdiain_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
// void test_sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd);
// void test_sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x);
// void test_sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
// void test_sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
// void test_sdiain_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
// void test_sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x);
// void test_sdiaex_sp_libstr(int kmax, float alpha, int *idx, struct s_strmat *sD, int di, int dj, struct s_strvec *sx, int xi);
// void test_sdiaad_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
// void test_sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
// void test_sdiaad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
// void test_sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd);
// void test_sdiaadin_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, int *idx, struct s_strmat *sD, int di, int dj);
// void test_srowin_lib(int kmax, float alpha, float *x, float *pD);
// void test_srowin_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
// void test_srowex_lib(int kmax, float alpha, float *pD, float *x);
// void test_srowex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi);
// void test_srowad_lib(int kmax, float alpha, float *x, float *pD);
// void test_srowad_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
// void test_srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD);
// void test_srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD);
// void test_srowad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj);
// void test_srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD);
// void test_srowsw_lib(int kmax, float *pA, float *pC);
// void test_srowsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
// void test_srowpe_libstr(int kmax, int *ipiv, struct s_strmat *sA);
// void test_scolin_lib(int kmax, float *x, int offset, float *pD, int sdd);
// void test_scolin_libstr(int kmax, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj);
// void test_scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
// void test_scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd);
// void test_scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd);
// void test_scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_scolsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj);
// void test_scolpe_libstr(int kmax, int *ipiv, struct s_strmat *sA);
// void test_svecin_libsp(int kmax, int *idx, float *x, float *y);
// void test_svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y);
// void test_svecad_sp_libstr(int m, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sz, int zi);
// void test_svecin_sp_libstr(int m, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sz, int zi);
// void test_svecex_sp_libstr(int m, float alpha, int *idx, struct s_strvec *sx, int x, struct s_strvec *sz, int zi);
// void test_sveccl_libstr(int m, struct s_strvec *sxm, int xim, struct s_strvec *sx, int xi, struct s_strvec *sxp, int xip, struct s_strvec *sz, int zi);
// void test_sveccl_mask_libstr(int m, struct s_strvec *sxm, int xim, struct s_strvec *sx, int xi, struct s_strvec *sxp, int xip, struct s_strvec *sz, int zi, struct s_strvec *sm, int mi);
// void test_svecze_libstr(int m, struct s_strvec *sm, int mi, struct s_strvec *sv, int vi, struct s_strvec *se, int ei);
// void test_svecnrm_inf_libstr(int m, struct s_strvec *sx, int xi, float *ptr_norm);
// void test_svecpe_libstr(int kmax, int *ipiv, struct s_strvec *sx, int xi);


// ext_dep

void test_s_allocate_strmat(int m, int n, struct s_strmat *sA);
void test_s_allocate_strvec(int m, struct s_strvec *sa);

void test_s_free_strmat(struct s_strmat *sA);
void test_s_free_strvec(struct s_strvec *sa);

void test_s_print_strmat(int m, int n, struct s_strmat *sA, int ai, int aj);
void test_s_print_strvec(int m, struct s_strvec *sa, int ai);
void test_s_print_tran_strvec(int m, struct s_strvec *sa, int ai);

void test_s_print_to_file_strmat(FILE *file, int m, int n, struct s_strmat *sA, int ai, int aj);
void test_s_print_to_file_strvec(FILE *file, int m, struct s_strvec *sa, int ai);
void test_s_print_tran_to_file_strvec(FILE *file, int m, struct s_strvec *sa, int ai);

void test_s_print_e_strmat(int m, int n, struct s_strmat *sA, int ai, int aj);
void test_s_print_e_strvec(int m, struct s_strvec *sa, int ai);
void test_s_print_e_tran_strvec(int m, struct s_strvec *sa, int ai);


#ifdef __cplusplus
}
#endif

