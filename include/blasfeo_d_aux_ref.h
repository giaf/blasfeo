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

#ifndef BLASFEO_D_AUX_REF_H_
#define BLASFEO_D_AUX_REF_H_

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

void blasfeo_create_dmat_ref  (int m, int n, struct blasfeo_dmat_ref *sA, void *memory);
void blasfeo_create_dvec_ref  (int m, int n, struct blasfeo_dvec_ref *sA, void *memory);

void blasfeo_free_dmat_ref(struct blasfeo_dmat_ref *sA);
void blasfeo_free_dvec_ref(struct blasfeo_dvec_ref *sa);

void blasfeo_pack_dmat_ref    (int m, int n, double *A, int lda, struct blasfeo_dmat_ref *sA, int ai, int aj);
int  blasfeo_memsize_dmat_ref (int m, int n);

void blasfeo_dgead_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgecp_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgesc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dgecpsc_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj);

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
