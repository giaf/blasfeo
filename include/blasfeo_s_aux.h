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

#include "../include/blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif



#include "blasfeo_s_aux_old.h"



/************************************************
* d_aux_lib.c
************************************************/

// returns the memory size (in bytes) needed for a smat
int blasfeo_memsize_smat(int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a smat
int blasfeo_memsize_diag_smat(int m, int n);
// returns the memory size (in bytes) needed for a svec
int blasfeo_memsize_svec(int m);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_smat(int m, int n, struct blasfeo_smat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_svec(int m, struct blasfeo_svec *sA, void *memory);
void blasfeo_pack_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_pack_svec(int m, float *a, struct blasfeo_svec *sa, int ai);
void blasfeo_pack_tran_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_unpack_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);
void blasfeo_unpack_svec(int m, struct blasfeo_svec *sa, int ai, float *a);
void blasfeo_unpack_tran_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);
void s_cast_mat2strmat(float *A, struct blasfeo_smat *sA);
void s_cast_diag_mat2strmat(float *dA, struct blasfeo_smat *sA);
void s_cast_vec2vecmat(float *a, struct blasfeo_svec *sa);

void blasfeo_sgein1(float a, struct blasfeo_smat *sA, int ai, int aj);
float blasfeo_sgeex1(struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_svecin1(float a, struct blasfeo_svec *sx, int xi);
float blasfeo_svecex1(struct blasfeo_svec *sx, int xi);

// A <= alpha
void blasfeo_sgese(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);
// a <= alpha
void blasfeo_svecse(int m, float alpha, struct blasfeo_svec *sx, int xi);

// copy and scale
// void sgecp_lib(int m, int n, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
//
void blasfeo_sgecpsc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_sgecp(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_sgesc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);

void blasfeo_sveccp(int m, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);
void blasfeo_svecsc(int m, float alpha, struct blasfeo_svec *sa, int ai);
void blasfeo_sveccpsc(int m, float alpha, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);

void blasfeo_strcp_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

void blasfeo_sgead(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_svecad(int m, float alpha, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);

void blasfeo_sgetr(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

void blasfeo_strtr_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_strtr_u(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

void blasfeo_sdiare(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_sdiaex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
void blasfeo_sdiain(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_sdiain_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_sdiaex_sp(int kmax, float alpha, int *idx, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_svec *sx, int xi);
void blasfeo_sdiaad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_sdiaad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_sdiaadin_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_srowin(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_srowex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
void blasfeo_srowad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_srowad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_srowsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_srowpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_srowpei(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_scolin(int kmax, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_scolsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_scolpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_scolpei(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_svecad_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
void blasfeo_svecin_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
void blasfeo_svecex_sp(int m, float alpha, int *idx, struct blasfeo_svec *sx, int x, struct blasfeo_svec *sz, int zi);
void blasfeo_sveccl(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi);
void blasfeo_sveccl_mask(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi, struct blasfeo_svec *sm, int mi);
void blasfeo_svecze(int m, struct blasfeo_svec *sm, int mi, struct blasfeo_svec *sv, int vi, struct blasfeo_svec *se, int ei);
void blasfeo_svecnrm_inf(int m, struct blasfeo_svec *sx, int xi, float *ptr_norm);
void blasfeo_svecpe(int kmax, int *ipiv, struct blasfeo_svec *sx, int xi);
void blasfeo_svecpei(int kmax, int *ipiv, struct blasfeo_svec *sx, int xi);



#ifdef __cplusplus
}
#endif

