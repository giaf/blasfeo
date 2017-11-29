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


// --- memory calculations
int test_d_size_strmat(int m, int n);
int test_d_size_diag_strmat(int m, int n);
int test_d_size_strvec(int m);

// --- creation
void test_d_create_strmat(int m, int n, struct d_strmat *sA, void *memory);
void test_d_create_strvec(int m, struct d_strvec *sA, void *memory);

// --- conversion
void test_d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void test_d_cvt_vec2strvec(int m, double *a, struct d_strvec *sa, int ai);
void test_d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void test_d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void test_d_cvt_strvec2vec(int m, struct d_strvec *sa, int ai, double *a);
void test_d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);

// --- cast
void test_d_cast_mat2strmat(double *A, struct d_strmat *sA);
void test_d_cast_diag_mat2strmat(double *dA, struct d_strmat *sA);
void test_d_cast_vec2vecmat(double *a, struct d_strvec *sa);

// ------ copy / scale

// B <= A
void test_dgecp_libstr(int m, int n,
					struct d_strmat *sA, int ai, int aj,
					struct d_strmat *sB, int bi, int bj);

// A <= alpha*A
void test_dgesc_libstr(int m, int n,
					double alpha,
					struct d_strmat *sA, int ai, int aj);

// B <= alpha*A
void test_dgecpsc_libstr(int m, int n,
					double alpha,
					struct d_strmat *sA, int ai, int aj,
					struct d_strmat *sB, int bi, int bj);

// // --- insert/extract
// //
// // <= sA[ai, aj]
// void test_dgein1_libstr(double a, struct d_strmat *sA, int ai, int aj);
// // <= sA[ai, aj]
// double dgeex1_libstr(struct d_strmat *sA, int ai, int aj);
// // sx[xi] <= a
// void test_dvecin1_libstr(double a, struct d_strvec *sx, int xi);
// // <= sx[xi]
// double dvecex1_libstr(struct d_strvec *sx, int xi);
// // A <= alpha

// // --- set
// void test_dgese_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj);
// // a <= alpha
// void test_dvecse_libstr(int m, double alpha, struct d_strvec *sx, int xi);
// // B <= A


// // --- vector
// // y <= x
// void test_dveccp_libstr(int m, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci);
// // x <= alpha*x
// void test_dvecsc_libstr(int m, double alpha, struct d_strvec *sa, int ai);
// // TODO
// // x <= alpha*x
// void test_dveccpsc_libstr(int m, double alpha, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci);


// // B <= A, A lower triangular
// void test_dtrcp_l_libstr(int m,
//                     struct d_strmat *sA, int ai, int aj,
//                     struct d_strmat *sB, int bi, int bj);

// void test_dtrcpsc_l_libstr(int m, double alpha,
//                     struct d_strmat *sA, int ai, int aj,
//                     struct d_strmat *sB, int bi, int bj);

// void test_dtrsc_l_libstr(int m, double alpha,
//                     struct d_strmat *sA, int ai, int aj);


// // B <= B + alpha*A
// void test_dgead_libstr(int m, int n, double alpha,
//                     struct d_strmat *sA, int ai, int aj,
//                     struct d_strmat *sC, int ci, int cj);

// // y <= y + alpha*x
// void test_dvecad_libstr(int m, double alpha,
//                     struct d_strvec *sa, int ai,
//                     struct d_strvec *sc, int ci);

// // --- traspositions
// void test_dgetr_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_dgetr_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
// void test_dtrtr_l_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_dtrtr_l_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
// void test_dtrtr_u_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_dtrtr_u_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
// void test_ddiare_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj);
// void test_ddiain_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
// void test_ddiaex_lib(int kmax, double alpha, int offset, double *pD, int sdd, double *x);
// void test_ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
// void test_ddiain_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
// void test_ddiain_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj);
// void test_ddiaex_libsp(int kmax, int *idx, double alpha, double *pD, int sdd, double *x);
// void test_ddiaex_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi);
// void test_ddiaex_sp_libstr(int kmax, double alpha, int *idx, struct d_strmat *sD, int di, int dj, struct d_strvec *sx, int xi);
// void test_ddiaad_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
// void test_ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
// void test_ddiaad_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj);
// void test_ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd);
// void test_ddiaadin_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi, int *idx, struct d_strmat *sD, int di, int dj);
// void test_drowin_lib(int kmax, double alpha, double *x, double *pD);
// void test_drowin_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
// void test_drowex_lib(int kmax, double alpha, double *pD, double *x);
// void test_drowex_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi);
// void test_drowad_lib(int kmax, double alpha, double *x, double *pD);
// void test_drowad_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
// void test_drowin_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
// void test_drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD);
// void test_drowad_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj);
// void test_drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD);
// void test_drowsw_lib(int kmax, double *pA, double *pC);
// void test_drowsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
// void test_drowpe_libstr(int kmax, int *ipiv, struct d_strmat *sA);
// void test_dcolex_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi);
// void test_dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
// void test_dcolin_libstr(int kmax, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj);
// void test_dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
// void test_dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
// void test_dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
// void test_dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_dcolsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj);
// void test_dcolpe_libstr(int kmax, int *ipiv, struct d_strmat *sA);
// void test_dvecin_libsp(int kmax, int *idx, double *x, double *y);
// void test_dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y);
// void test_dvecad_sp_libstr(int m, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strvec *sz, int zi);
// void test_dvecin_sp_libstr(int m, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strvec *sz, int zi);
// void test_dvecex_sp_libstr(int m, double alpha, int *idx, struct d_strvec *sx, int x, struct d_strvec *sz, int zi);
// void test_dveccl_libstr(int m, struct d_strvec *sxm, int xim, struct d_strvec *sx, int xi, struct d_strvec *sxp, int xip, struct d_strvec *sz, int zi);
// void test_dveccl_mask_libstr(int m, struct d_strvec *sxm, int xim, struct d_strvec *sx, int xi, struct d_strvec *sxp, int xip, struct d_strvec *sz, int zi, struct d_strvec *sm, int mi);
// void test_dvecze_libstr(int m, struct d_strvec *sm, int mi, struct d_strvec *sv, int vi, struct d_strvec *se, int ei);
// void test_dvecnrm_inf_libstr(int m, struct d_strvec *sx, int xi, double *ptr_norm);
// void test_dvecpe_libstr(int kmax, int *ipiv, struct d_strvec *sx, int xi);
// void test_dvecpei_libstr(int kmax, int *ipiv, struct d_strvec *sx, int xi);

// ext_dep

void test_d_allocate_strmat(int m, int n, struct d_strmat *sA);
void test_d_allocate_strvec(int m, struct d_strvec *sa);

void test_d_free_strmat(struct d_strmat *sA);
void test_d_free_strvec(struct d_strvec *sa);

void test_d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);
void test_d_print_strvec(int m, struct d_strvec *sa, int ai);
void test_d_print_tran_strvec(int m, struct d_strvec *sa, int ai);

void test_d_print_to_file_strmat(FILE *file, int m, int n, struct d_strmat *sA, int ai, int aj);
void test_d_print_to_file_strvec(FILE *file, int m, struct d_strvec *sa, int ai);
void test_d_print_tran_to_file_strvec(FILE *file, int m, struct d_strvec *sa, int ai);

void test_d_print_e_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);
void test_d_print_e_strvec(int m, struct d_strvec *sa, int ai);
void test_d_print_e_tran_strvec(int m, struct d_strvec *sa, int ai);



// #ifdef __cplusplus
// }
// #endif
