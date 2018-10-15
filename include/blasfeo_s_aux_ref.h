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

#ifndef BLASFEO_S_AUX_REF_H_
#define BLASFEO_S_AUX_REF_H_

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

void blasfeo_create_smat_ref  (int m, int n, struct blasfeo_smat_ref *sA, void *memory);
void blasfeo_create_svec_ref  (int m, int n, struct blasfeo_svec_ref *sA, void *memory);

void blasfeo_free_smat_ref(struct blasfeo_smat_ref *sA);
void blasfeo_free_svec_ref(struct blasfeo_svec_ref *sa);

void blasfeo_pack_smat_ref    (int m, int n, float *A, int lda, struct blasfeo_smat_ref *sA, int ai, int aj);
int  blasfeo_memsize_smat_ref (int m, int n);

void blasfeo_sgead_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_sgecp_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_sgesc_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sgecpsc_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_sgese_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);

// ge
void blasfeo_sgecp_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_sgesc_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sgecpsc_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_strcp_l_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_strcpsc_l_ref(int m, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_strsc_l_ref(int m, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sgead_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sC, int yi, int cj);

void blasfeo_sgetr_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_strtr_l_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_strtr_u_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj);

// dia
void blasfeo_sdiare_ref(int kmax, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sdiain_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sdiain_sp_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, int *idx, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sdiaex_ref(int kmax, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_sdiaex_sp_ref(int kmax, float alpha, int *idx, struct blasfeo_smat_ref *sD, int di, int dj, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_sdiaad_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sdiaad_sp_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, int *idx, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sdiaadin_sp_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi, int *idx, struct blasfeo_smat_ref *sD, int di, int dj);

// row
void blasfeo_srowin_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_srowex_ref(int kmax, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_srowad_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_srowad_sp_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, int *idx, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_srowsw_ref(int kmax, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sC, int ci, int cj);
void blasfeo_srowpe_ref(int kmax, int *ipiv, struct blasfeo_smat_ref *sA);
void blasfeo_srowpei_ref(int kmax, int *ipiv, struct blasfeo_smat_ref *sA);

// col
void blasfeo_scolex_ref(int kmax, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_scolin_ref(int kmax, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_scolsc_ref(int kmax, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_scolsw_ref(int kmax, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sC, int ci, int cj);
void blasfeo_scolpe_ref(int kmax, int *ipiv, struct blasfeo_smat_ref *sA);
void blasfeo_scolpei_ref(int kmax, int *ipiv, struct blasfeo_smat_ref *sA);

// vec
void blasfeo_svecad_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi);
void blasfeo_svecse_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_svecin1_ref(float a, struct blasfeo_svec_ref *sx, int xi);
float blasfeo_svecex1_ref(struct blasfeo_svec_ref *sx, int xi);
void blasfeo_sveccp_ref(int m, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi);
void blasfeo_svecsc_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_sveccpsc_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi);
void blasfeo_svecad_sp_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi, int *idx, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_svecin_sp_ref(int m, float alpha, struct blasfeo_svec_ref *sx, int xi, int *idx, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_svecex_sp_ref(int m, float alpha, int *idx, struct blasfeo_svec_ref *sx, int x, struct blasfeo_svec_ref *sz, int zi);

void blasfeo_sveccl_ref(int m,
	struct blasfeo_svec_ref *sxm, int xim, struct blasfeo_svec_ref *sx, int xi,
	struct blasfeo_svec_ref *sxp, int xip, struct blasfeo_svec_ref *sz, int zi);

void blasfeo_sveccl_mask_ref(int m,
	struct blasfeo_svec_ref *sxm, int xim, struct blasfeo_svec_ref *sx, int xi,
	struct blasfeo_svec_ref *sxp, int xip, struct blasfeo_svec_ref *sz, int zi,
	struct blasfeo_svec_ref *sm, int mi);

void blasfeo_svecze_ref(int m, struct blasfeo_svec_ref *sm, int mi, struct blasfeo_svec_ref *sv, int vi, struct blasfeo_svec_ref *se, int ei);
void blasfeo_svecnrm_inf_ref(int m, struct blasfeo_svec_ref *sx, int xi, float *ptr_norm);
void blasfeo_svecpe_ref(int kmax, int *ipiv, struct blasfeo_svec_ref *sx, int xi);
void blasfeo_svecpei_ref(int kmax, int *ipiv, struct blasfeo_svec_ref *sx, int xi);


#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_AUX_REF_H_
