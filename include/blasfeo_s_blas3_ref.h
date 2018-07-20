/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* BLASFEO is free software; you can redistribute it and/or                                        *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* BLASFEO is distributed in the hope that it will be useful,                                      *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with BLASFEO; if not, write to the Free Software                                  *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

/*
 * level3 algebra routines header
 *
 * include/blasfeo_blas3_lib*.h
 *
 */

#ifndef BLASFEO_S_BLAS3_REF_H_
#define BLASFEO_S_BLAS3_REF_H_

#include "blasfeo_common.h"


#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing


// Class GEMM

// D <= beta * C + alpha * A * B
void blasfeo_sgemm_nn_ref(
	int m, int n, int k, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj, float beta,
	struct blasfeo_smat_ref *sC, int ci, int cj,
	struct blasfeo_smat_ref *sD, int di, int dj);

void blasfeo_sgemm_nt_ref(
	int m, int n, int k, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj, float beta,
	struct blasfeo_smat_ref *sC, int ci, int cj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void blasfeo_ssyrk_ln_mn_ref(
	int m, int n, int k, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj, float beta,
	struct blasfeo_smat_ref *sC, int ci, int cj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// Class SYRK

// D <= beta * C + alpha * A * B^T ; C, D lower triangular
void blasfeo_ssyrk_ln_ref(
	int m, int k, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj, float beta,
	struct blasfeo_smat_ref *sC, int ci, int cj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// Class TRMM

// D <= alpha * B * A^T ; B upper triangular
void blasfeo_strmm_rutn_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * B * A ; A lower triangular
void blasfeo_strmm_rlnn_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
void blasfeo_strsm_rltn_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * B * A^{-T} , with A lower triangular with unit diagonal
void blasfeo_strsm_rltu_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void blasfeo_strsm_rutn_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * A^{-1} * B , with A lower triangular with unit diagonal
void blasfeo_strsm_llnu_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

// D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void blasfeo_strsm_lunn_ref(
	int m, int n, float alpha,
	struct blasfeo_smat_ref *sA, int ai, int aj,
	struct blasfeo_smat_ref *sB, int bi, int bj,
	struct blasfeo_smat_ref *sD, int di, int dj);

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_BLAS3_REF_H_
