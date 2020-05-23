/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
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

size_t blasfeo_memsize_smat_ref (int m, int n);
size_t blasfeo_memsize_svec_ref (int m);
void blasfeo_create_smat_ref  (int m, int n, struct blasfeo_smat_ref *sA, void *memory);
void blasfeo_create_svec_ref  (int m, struct blasfeo_svec_ref *sA, void *memory);

void blasfeo_free_smat_ref(struct blasfeo_smat_ref *sA);
void blasfeo_free_svec_ref(struct blasfeo_svec_ref *sa);

void blasfeo_pack_smat_ref    (int m, int n, float *A, int lda, struct blasfeo_smat_ref *sA, int ai, int aj);

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
