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

#ifndef BLASFEO_D_AUX_REF_H_
#define BLASFEO_D_AUX_REF_H_

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

void blasfeo_create_dmat_ref  (int m, int n, struct blasfeo_dmat_ref *sA, char *memory);
void blasfeo_create_dvec_ref  (int m, int n, struct blasfeo_dvec_ref *sA, char *memory);

void blasfeo_free_dmat_ref(struct blasfeo_dmat_ref *sA);
void blasfeo_free_dvec_ref(struct blasfeo_dvec_ref *sa);

void blasfeo_pack_dmat_ref    (int m, int n, double *A, int lda, struct blasfeo_dmat_ref *sA, int ai, int aj);
int  blasfeo_memsize_dmat_ref (int m, int n);

void blasfeo_dgead_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgecp_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgesc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dgecpsc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgese_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);

// ge
void blasfeo_dgecp_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgesc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dgecpsc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dtrcp_l_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dtrcpsc_l_ref(int m, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dtrsc_l_ref(int m, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dgead_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sC, int yi, int cj);

void blasfeo_dgetr_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dtrtr_l_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dtrtr_u_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);

// dia
void blasfeo_ddiare_ref(int kmax, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_ddiain_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_ddiain_sp_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, int *idx, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_ddiaex_ref(int kmax, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_ddiaex_sp_ref(int kmax, double alpha, int *idx, struct blasfeo_dmat_ref *sD, int di, int dj, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_ddiaad_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_ddiaad_sp_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, int *idx, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_ddiaadin_sp_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi, int *idx, struct blasfeo_dmat_ref *sD, int di, int dj);

// row
void blasfeo_drowin_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_drowex_ref(int kmax, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_drowad_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_drowad_sp_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, int *idx, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_drowsw_ref(int kmax, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sC, int ci, int cj);
void blasfeo_drowpe_ref(int kmax, int *ipiv, struct blasfeo_dmat_ref *sA);
void blasfeo_drowpei_ref(int kmax, int *ipiv, struct blasfeo_dmat_ref *sA);

// col
void blasfeo_dcolex_ref(int kmax, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_dcolin_ref(int kmax, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dcolsc_ref(int kmax, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dcolsw_ref(int kmax, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sC, int ci, int cj);
void blasfeo_dcolpe_ref(int kmax, int *ipiv, struct blasfeo_dmat_ref *sA);
void blasfeo_dcolpei_ref(int kmax, int *ipiv, struct blasfeo_dmat_ref *sA);

// vec
void blasfeo_dvecad_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi);
void blasfeo_dvecse_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_dvecin1_ref(double a, struct blasfeo_dvec_ref *sx, int xi);
double blasfeo_dvecex1_ref(struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_dveccp_ref(int m, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi);
void blasfeo_dvecsc_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_dveccpsc_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi);
void blasfeo_dvecad_sp_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi, int *idx, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dvecin_sp_ref(int m, double alpha, struct blasfeo_dvec_ref *sx, int xi, int *idx, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dvecex_sp_ref(int m, double alpha, int *idx, struct blasfeo_dvec_ref *sx, int x, struct blasfeo_dvec_ref *sz, int zi);

void blasfeo_dveccl_ref(int m,
	struct blasfeo_dvec_ref *sxm, int xim, struct blasfeo_dvec_ref *sx, int xi,
	struct blasfeo_dvec_ref *sxp, int xip, struct blasfeo_dvec_ref *sz, int zi);

void blasfeo_dveccl_mask_ref(int m,
	struct blasfeo_dvec_ref *sxm, int xim, struct blasfeo_dvec_ref *sx, int xi,
	struct blasfeo_dvec_ref *sxp, int xip, struct blasfeo_dvec_ref *sz, int zi,
	struct blasfeo_dvec_ref *sm, int mi);

void blasfeo_dvecze_ref(int m, struct blasfeo_dvec_ref *sm, int mi, struct blasfeo_dvec_ref *sv, int vi, struct blasfeo_dvec_ref *se, int ei);
void blasfeo_dvecnrm_inf_ref(int m, struct blasfeo_dvec_ref *sx, int xi, double *ptr_norm);
void blasfeo_dvecpe_ref(int kmax, int *ipiv, struct blasfeo_dvec_ref *sx, int xi);
void blasfeo_dvecpei_ref(int kmax, int *ipiv, struct blasfeo_dvec_ref *sx, int xi);


#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_AUX_REF_H_
