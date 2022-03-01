/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
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

#include <stdlib.h>
#include <stdio.h>

#include <blasfeo_target.h>
#include <blasfeo_block_size.h>
#include <blasfeo_common.h>
#include <blasfeo_stdlib.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>
#include <blasfeo_memory.h>

#include <blasfeo_timing.h>



#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )
#define blasfeo_dmat blasfeo_cm_dmat
#define blasfeo_hp_dsyr2k_ln blasfeo_hp_cm_dsyr2k_ln
#define blasfeo_hp_dsyr2k_lt blasfeo_hp_cm_dsyr2k_lt
#define blasfeo_hp_dsyr2k_un blasfeo_hp_cm_dsyr2k_un
#define blasfeo_hp_dsyr2k_ut blasfeo_hp_cm_dsyr2k_ut
#define blasfeo_dsyr2k_ln blasfeo_cm_dsyr2k_ln
#define blasfeo_dsyr2k_lt blasfeo_cm_dsyr2k_lt
#define blasfeo_dsyr2k_un blasfeo_cm_dsyr2k_un
#define blasfeo_dsyr2k_ut blasfeo_cm_dsyr2k_ut

#define blasfeo_dsyrk_ln blasfeo_cm_dsyrk_ln
#define blasfeo_dsyrk_lt blasfeo_cm_dsyrk_lt
#define blasfeo_dsyrk_un blasfeo_cm_dsyrk_un
#define blasfeo_dsyrk_ut blasfeo_cm_dsyrk_ut

void blasfeo_cm_dsyrk_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_cm_dsyrk_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_cm_dsyrk_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_cm_dsyrk_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj);
#endif



#define CACHE_LINE_EL D_CACHE_LINE_EL
#define L1_CACHE_EL D_L1_CACHE_EL
#define L2_CACHE_EL D_L2_CACHE_EL
#define LLC_CACHE_EL D_LLC_CACHE_EL
#define PS D_PS
#define M_KERNEL D_M_KERNEL
#define KC D_KC
#define NC D_NC
#define MC D_MC



// dsyr2k lower not-transposed
void blasfeo_hp_dsyr2k_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	double d_1 = 1.0;
	blasfeo_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	blasfeo_dsyrk_ln(m, k, alpha, sB, bi, bj, sA, ai, aj, d_1, sD, di, dj, sD, di, dj);
	return;
	}



// dsyr2k lower transposed
void blasfeo_hp_dsyr2k_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	double d_1 = 1.0;
	blasfeo_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	blasfeo_dsyrk_lt(m, k, alpha, sB, bi, bj, sA, ai, aj, d_1, sD, di, dj, sD, di, dj);
	return;
	}



// dsyr2k upper not-transposed
void blasfeo_hp_dsyr2k_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	double d_1 = 1.0;
	blasfeo_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	blasfeo_dsyrk_un(m, k, alpha, sB, bi, bj, sA, ai, aj, d_1, sD, di, dj, sD, di, dj);
	return;
	}



// dsyr2k upper transposed
void blasfeo_hp_dsyr2k_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	double d_1 = 1.0;
	blasfeo_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	blasfeo_dsyrk_ut(m, k, alpha, sB, bi, bj, sA, ai, aj, d_1, sD, di, dj, sD, di, dj);
	return;
	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_dsyr2k_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyr2k_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyr2k_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyr2k_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



#endif

