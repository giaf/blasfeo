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
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blasfeo_api.h>
#if defined(BLASFEO_REF_API)
#include <blasfeo_d_blasfeo_ref_api.h>
#endif



// dgemm nn
void blasfeo_hp_dgemm_nn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	if(m<=0 || n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int ps = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;

	int air = ai & (ps-1);
	int bir = bi & (ps-1);

	// pA, pB point to panels edges
	double *pA = sA->pA + aj*ps + (ai-air)*sda;
	double *pB = sB->pA + bj*ps + (bi-bir)*sdb;
	double *pC = sC->pA + cj*ps;
	double *pD = sD->pA + dj*ps;

	int offsetB = bir;

	int ci0 = ci-air;
	int di0 = di-air;
	int offsetC;
	int offsetD;
	if(ci0>=0)
		{
		pC += ci0/ps*ps*sdd;
		offsetC = ci0%ps;
		}
	else
		{
		pC += -ps*sdc;
		offsetC = ps+ci0;
		}

	if(di0>=0)
		{
		pD += di0/ps*ps*sdd;
		offsetD = di0%ps;
		}
	else
		{
		pD += -ps*sdd;
		offsetD = ps+di0;
		}

	int i, j, l;



	// algorithm scheme
	if(air!=0)
		{
		goto clear_air;
		}
select_loop:
	if(offsetC==0 & offsetD==0)
		{
		goto loop_00;
		}
	else
		{
		goto loop_CD;
		}
	// should never get here
	return;



	// clean up at the beginning
clear_air:
	j = 0;
	for(; j<n; j+=8)
		{
		kernel_dgemm_nn_8x8_gen_lib8(k, &alpha, &pA[0], offsetB, &pB[j*ps], sdb, &beta, offsetC, &pC[j*ps], sdc, offsetD, &pD[j*ps], sdd, air, air+m, 0, n-j);
		}
	m -= 1*ps-air;
	pA += 1*ps*sda;
	pC += 1*ps*sdc;
	pD += 1*ps*sdd;
	goto select_loop;



	// main loop aligned
loop_00:
	i = 0;
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_dgemm_nn_8x8_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_nn_8x8_vs_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_8;
		}
	// common return if i==m
	return;



	// main loop C, D not aligned
loop_CD:
	i = 0;
	for(; i<m; i+=8)
		{
		j = 0;
		for(; j<n; j+=8)
			{
			kernel_dgemm_nn_8x8_gen_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, offsetC, &pC[j*ps+i*sdc], sdc, offsetD, &pD[j*ps+i*sdd], sdd, 0, m-i, 0, n-j);
			}
		}
	// common return if i==m
	return;



	// clean up loops definitions

	left_8:
	j = 0;
	for(; j<n; j+=8)
		{
		kernel_dgemm_nn_8x8_vs_lib8(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
		}
	return;

	return;

	}



// dgemm nt
void blasfeo_hp_dgemm_nt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int ps = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	int air = ai & (ps-1);
	int bir = bi & (ps-1);
	double *pA = sA->pA + aj*ps + (ai-air)*sda;
	double *pB = sB->pA + bj*ps + (bi-bir)*sdb;
	double *pC = sC->pA + cj*ps;
	double *pD = sD->pA + dj*ps;

	int ci0 = ci-air;
	int di0 = di-air;
	int offsetC;
	int offsetD;
	if(ci0>=0)
		{
		pC += ci0/ps*ps*sdd;
		offsetC = ci0%ps;
		}
	else
		{
		pC += -ps*sdc;
		offsetC = ps+ci0;
		}
	if(di0>=0)
		{
		pD += di0/ps*ps*sdd;
		offsetD = di0%ps;
		}
	else
		{
		pD += -ps*sdd;
		offsetD = ps+di0;
		}

	int i, j;

	int idxB;



	// algorithm scheme
	if(air!=0)
		{
		goto clear_air;
		// TODO instaed use buffer to align A !!!
		}
select_loop:
	if(offsetC==0 & offsetD==0)
		{
		goto loop_00;
		}
	else
		{
		goto loop_CD;
		}
	// should never get here
	return;



	// clean up at the beginning
clear_air:
	j = 0;
	idxB = 0;
	// clean up at the beginning
	if(bir!=0)
		{
		kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[0], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps]-bir*ps, sdc, offsetD, &pD[j*ps]-bir*ps, sdd, air, air+m, bir, bir+n-j);
		j += ps-bir;
		idxB += 8;
		}
	// main loop
	for(; j<n; j+=8, idxB+=8)
		{
		kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[0], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps], sdc, offsetD, &pD[j*ps], sdd, air, air+m, 0, n-j);
		}
	m -= ps-air;
	pA += ps*sda;
	pC += ps*sdc;
	pD += ps*sdd;
	goto select_loop;



	// main loop aligned
loop_00:
	i = 0;
	for(; i<m-7; i+=8)
		{
		j = 0;
		idxB = 0;
		// clean up at the beginning
		if(bir!=0)
			{
			kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps+i*sdc]-bir*ps, sdc, offsetD, &pD[j*ps+i*sdd]-bir*ps, sdd, 0, m-i, bir, bir+n-j);
			j += ps-bir;
			idxB += 8;
			}
		// main loop
		for(; j<n-7; j+=8, idxB+=8)
			{
			kernel_dgemm_nt_8x8_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd]);
			}
		if(j<n)
			{
			kernel_dgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_8;
		}
	// common return if i==m
	return;



	// main loop C, D not aligned
loop_CD:
	i = 0;
	for(; i<m; i+=8)
		{
		j = 0;
		idxB = 0;
		// clean up at the beginning
		if(bir!=0)
			{
			kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps+i*sdc]-bir*ps, sdc, offsetD, &pD[j*ps+i*sdd]-bir*ps, sdd, 0, m-i, bir, bir+n-j);
			j += ps-bir;
			idxB += 8;
			}
		// main loop
		for(; j<n; j+=8, idxB+=8)
			{
			kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps+i*sdc], sdc, offsetD, &pD[j*ps+i*sdd], sdd, 0, m-i, 0, n-j);
			}
		}
	// common return if i==m
	return;



	// clean up loops definitions

	left_8:
	j = 0;
	idxB = 0;
	// clean up at the beginning
	if(bir!=0)
		{
		kernel_dgemm_nt_8x8_gen_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*ps+i*sdc]-bir*ps, sdc, offsetD, &pD[j*ps+i*sdd]-bir*ps, sdd, 0, m-i, bir, bir+n-j);
		j += ps-bir;
		idxB += 8;
		}
	// main loop
	for(; j<n; j+=8, idxB+=8)
		{
		kernel_dgemm_nt_8x8_vs_lib8(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
		}
	return;

	}



// dgemm_tn
void blasfeo_hp_dgemm_tn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dgemm_tn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dgemm_tt
void blasfeo_hp_dgemm_tt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	if(m<=0 || n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int ps = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;

	int air = ai & (ps-1);
	int bir = bi & (ps-1);

	// pA, pB point to panels edges
	double *pA = sA->pA + aj*ps + (ai-air)*sda;
	double *pB = sB->pA + bj*ps + (bi-bir)*sdb;
	double *pC = sC->pA + (cj-bir)*ps;
	double *pD = sD->pA + (dj-bir)*ps;

	int offsetA = air;

	int ci0 = ci; //-bir;
	int di0 = di; //-bir;
	int offsetC;
	int offsetD;
	if(ci0>=0)
		{
		pC += ci0/ps*ps*sdd;
		offsetC = ci0%ps;
		}
	else
		{
		pC += -ps*sdc;
		offsetC = ps+ci0;
		}

	if(di0>=0)
		{
		pD += di0/ps*ps*sdd;
		offsetD = di0%ps;
		}
	else
		{
		pD += -ps*sdd;
		offsetD = ps+di0;
		}

	int i, j, l;



	// algorithm scheme
	if(bir!=0)
		{
		goto clear_bir;
		}
select_loop:
	if(offsetC==0 & offsetD==0)
		{
		goto loop_00;
		}
	else
		{
		goto loop_CD;
		}
	// should never get here
	return;



	// clean up at the beginning
clear_bir:
	i = 0;
	for(; i<m; i+=8)
		{
		kernel_dgemm_tt_8x8_gen_lib8(k, &alpha, offsetA, &pA[i*ps], sda, &pB[0], &beta, offsetC, &pC[i*sdc], sdc, offsetD, &pD[i*sdd], sdd, 0, m-i, bir, bir+n);
		}
	n -= 1*ps-bir;
	pB += 1*ps*sdb;
	pC += 1*8*ps;
	pD += 1*8*ps;
	goto select_loop;



	// main loop aligned
loop_00:
	j = 0;
	for(; j<n-7; j+=8)
		{
		i = 0;
		for(; i<m-7; i+=8)
			{
			kernel_dgemm_tt_8x8_lib8(k, &alpha, offsetA, &pA[i*ps], sda, &pB[j*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd]);
			}
		if(i<m)
			{
			kernel_dgemm_tt_8x8_vs_lib8(k, &alpha, offsetA, &pA[i*ps], sda, &pB[j*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
			}
		}
	if(n>j)
		{
		goto left_8;
		}
	// common return if i==m
	return;



	// main loop C, D not aligned
loop_CD:
	j = 0;
	for(; j<n; j+=8)
		{
		i = 0;
		for(; i<m; i+=8)
			{
			kernel_dgemm_tt_8x8_gen_lib8(k, &alpha, offsetA, &pA[i*ps], sda, &pB[j*sdb], &beta, offsetC, &pC[j*ps+i*sdc], sdc, offsetD, &pD[j*ps+i*sdd], sdd, 0, m-i, 0, n-j);
			}
		}
	// common return if i==m
	return;



	// clean up loops definitions

	left_8:
	i = 0;
	for(; i<m; i+=8)
		{
		kernel_dgemm_tt_8x8_vs_lib8(k, &alpha, offsetA, &pA[i*ps], sda, &pB[j*sdb], &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
		}
	return;

	return;

	}



// dtrsm_llnn
void blasfeo_hp_dtrsm_llnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_llnn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_llnu
void blasfeo_hp_dtrsm_llnu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_llnu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lltn
void blasfeo_hp_dtrsm_lltn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lltn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lltu
void blasfeo_hp_dtrsm_lltu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lltu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lunn
void blasfeo_hp_dtrsm_lunn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lunn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lunu
void blasfeo_hp_dtrsm_lunu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lunu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lutn
void blasfeo_hp_dtrsm_lutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lutn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_lutu
void blasfeo_hp_dtrsm_lutu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_lutu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_rlnn
void blasfeo_hp_dtrsm_rlnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rlnn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_rlnu
void blasfeo_hp_dtrsm_rlnu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rlnu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_right_lower_transposed_notunit
void blasfeo_hp_dtrsm_rltn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rltn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_right_lower_transposed_unit
void blasfeo_hp_dtrsm_rltu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rltu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_runn
void blasfeo_hp_dtrsm_runn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_runn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_runu
void blasfeo_hp_dtrsm_runu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_runu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_right_upper_transposed_notunit
void blasfeo_hp_dtrsm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rutn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrsm_rutu
void blasfeo_hp_dtrsm_rutu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrsm_rutu: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrmm_right_upper_transposed_notunit (B, i.e. the first matrix, is triangular !!!)
void blasfeo_hp_dtrmm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrmm_rutn: feature not implemented yet\n");
	exit(1);
#endif
	}



// dtrmm_right_lower_nottransposed_notunit (B, i.e. the first matrix, is triangular !!!)
void blasfeo_hp_dtrmm_rlnn(int m, int n, double alpha, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dtrmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
#else
	printf("\nblasfeo_dtrmm_rlnn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsyrk_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dsyrk_ln: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsyrk_ln_mn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsyrk_ln_mn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dsyrk_ln_mn: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsyrk_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dsyrk_lt: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsyrk_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dsyrk_un: feature not implemented yet\n");
	exit(1);
#endif
	}



void blasfeo_hp_dsyrk_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
#else
	printf("\nblasfeo_dsyrk_ut: feature not implemented yet\n");
	exit(1);
#endif
	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_dgemm_nn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_nt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_tn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dgemm_tt(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dtrsm_llnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_llnu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lltn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lltu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lunn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lunu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_lutu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rlnn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rlnu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rltn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rltu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_runn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_runu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrsm_rutu(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj);
	}



void blasfeo_dtrmm_rutn(int m, int n, double alpha, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrmm_rutn(m, n, alpha, sB, bi, bj, sA, ai, aj, sD, di, dj);
	}



void blasfeo_dtrmm_rlnn(int m, int n, double alpha, struct blasfeo_dmat *sB, int bi, int bj, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dtrmm_rlnn(m, n, alpha, sB, bi, bj, sA, ai, aj, sD, di, dj);
	}



void blasfeo_dsyrk_ln(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk_ln_mn(int m, int n, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk_ln_mn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



void blasfeo_dsyrk_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
	blasfeo_hp_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj);
	}



#endif




