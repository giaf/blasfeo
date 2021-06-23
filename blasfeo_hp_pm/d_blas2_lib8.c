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

#include <blasfeo_common.h>
#include <blasfeo_d_kernel.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>
#if defined(BLASFEO_REF_API)
#include <blasfeo_d_blasfeo_ref_api.h>
#endif



void blasfeo_hp_dgemv_n(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
#else
	printf("\nblasfeo_dgemv_n: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dgemv_t(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
#else
	printf("\nblasfeo_dgemv_t: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dgemv_nt(int m, int n, double alpha_n, double alpha_t, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx_n, int xi_n, struct blasfeo_dvec *sx_t, int xi_t, double beta_n, double beta_t, struct blasfeo_dvec *sy_n, int yi_n, struct blasfeo_dvec *sy_t, int yi_t, struct blasfeo_dvec *sz_n, int zi_n, struct blasfeo_dvec *sz_t, int zi_t)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t);
#else
	printf("\nblasfeo_dgemv_nt: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsymv_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
#else
	printf("\nblasfeo_dsymv_l: feature not implemented yet\n");
	exit(1);
#endif
	}


// m >= n
void blasfeo_hp_dsymv_l_mn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsymv_l_mn(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
#else
	printf("\nblasfeo_dsymv_l_mn: feature not implemented yet\n");
	exit(1);
#endif
	}



// m >= n
void blasfeo_hp_dtrmv_lnn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_lnn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_lnn: feature not implemented yet\n");
	exit(1);
#endif
	}



// m >= n
void blasfeo_hp_dtrmv_lnu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_lnu(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_lnu: feature not implemented yet\n");
	exit(1);
#endif
	}



// m >= n
void blasfeo_hp_dtrmv_ltn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_ltn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_ltn: feature not implemented yet\n");
	exit(1);
#endif
	}



// m >= n
void blasfeo_hp_dtrmv_ltu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_ltu(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_ltu: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrmv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_unn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_unn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrmv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmv_utn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrmv_utn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_lnn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_lnn_mn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_lnn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_lnn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_lnn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_lnu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_lnu(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_lnu: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_ltn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_ltn_mn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_ltn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_ltn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_ltn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_ltu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_ltu(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_ltu: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_unn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_unn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dtrsv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsv_utn(m, sA, ai, aj, sx, xi, sz, zi);
#else
	printf("\nblasfeo_dtrsv_utn: feature not implemented yet\n");
	exit(1);
#endif
	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_dgemv_n(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
	}



void blasfeo_dgemv_t(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
	}



void blasfeo_dgemv_nt(int m, int n, double alpha_n, double alpha_t, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx_n, int xi_n, struct blasfeo_dvec *sx_t, int xi_t, double beta_n, double beta_t, struct blasfeo_dvec *sy_n, int yi_n, struct blasfeo_dvec *sy_t, int yi_t, struct blasfeo_dvec *sz_n, int zi_n, struct blasfeo_dvec *sz_t, int zi_t)
	{
	blasfeo_hp_dgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t);
	}



void blasfeo_dsymv_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dsymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
	}



void blasfeo_dsymv_l_mn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dsymv_l_mn(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi);
	}



void blasfeo_dtrmv_lnn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_lnn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrmv_lnu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_lnu(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrmv_ltn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_ltn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrmv_ltu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_ltu(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrmv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_unn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrmv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrmv_utn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_lnn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_lnn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_lnn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_lnu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_lnu(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_ltn_mn(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_ltn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_ltn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_ltu(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_ltu(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_unn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_unn(m, sA, ai, aj, sx, xi, sz, zi);
	}



void blasfeo_dtrsv_utn(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	blasfeo_hp_dtrsv_utn(m, sA, ai, aj, sx, xi, sz, zi);
	}



#endif

