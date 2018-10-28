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

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_aux.h"



/****************************
* old interface
****************************/

void sgemm_nt_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	const int bs = 4;

	int i, j, l;

	i = 0;

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nt_16x4_lib4(k, &alpha, &pA[i*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+0)*sdc], &pD[j*bs+(i+0)*sdd], m-(i+0), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], m-(i+4), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+8)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], m-(i+8), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+12)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+12)*sdc], &pD[j*bs+(i+12)*sdd], m-(i+12), n-j);
			}
		}
#endif
#if defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nt_12x4_lib4(k, &alpha, &pA[i*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+0)*sdc], &pD[j*bs+(i+0)*sdd], m-(i+0), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], m-(i+4), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+8)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], m-(i+8), n-j);
			}
		}
#endif
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53) | defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	for(; i<m-7; i+=8)
		{
		j = 0;
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_nt_8x8_lib4(k, &alpha, &pA[i*sda], sda, &pB[j*sdb], sdb, &beta, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
#endif
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nt_8x4_lib4(k, &alpha, &pA[i*sda], sda, &pB[j*sdb], &beta, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+0)*sdc], &pD[j*bs+(i+0)*sdd], m-(i+0), n-j);
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], m-(i+4), n-j);
			}
		}
#endif
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nt_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_12:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+0)*sdc], &pD[j*bs+(i+0)*sdd], m-(i+0), n-j);
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], m-(i+4), n-j);
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+8)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], m-(i+8), n-j);
		}
	return;

	left_8:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+0)*sdc], &pD[j*bs+(i+0)*sdd], m-(i+0), n-j);
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], &pB[j*sdb], &beta, &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], m-(i+4), n-j);
		}
	return;

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



// D <= B * A^{-T} , with A lower triangular with unit diagonal
void strsm_nt_rl_one_lib(int m, int n, float *pA, int sda, float *pB, int sdb, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_one_4x4_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda]);
			}
		if(j<n)
			{
			kernel_strsm_nt_rl_one_4x4_vs_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_strsm_nt_rl_one_4x4_vs_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda], m-i, n-j);
		}
	return;

	}



// D <= B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
void strsm_nt_ru_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *pB, int sdb, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, idx;

	int rn = n%4;

	// float *dummy;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		// clean at the end
		if(rn>0)
			{
			idx = n-rn;
			// XXX pA & pD are dummy and should not be used internally !!!
			kernel_strsm_nt_ru_inv_4x4_vs_lib4(0, pD, pA, &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx], m-i, rn);
			j += rn;
			}
		for(; j<n; j+=4)
			{
			idx = n-j-4;
			kernel_strsm_nt_ru_inv_4x4_lib4(j, &pD[i*sdd+(idx+4)*bs], &pA[idx*sda+(idx+4)*bs], &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx]);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	left_4:
	j = 0;
	// TODO
	// clean at the end
	if(rn>0)
		{
		idx = n-rn;
		// XXX pA & pD are dummy and should not be used internally !!!
		kernel_strsm_nt_ru_inv_4x4_vs_lib4(0, pD, pA, &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx], m-i, rn);
		j += rn;
		}
	for(; j<n; j+=4)
		{
		idx = n-j-4;
		kernel_strsm_nt_ru_inv_4x4_vs_lib4(j, &pD[i*sdd+(idx+4)*bs], &pA[idx*sda+(idx+4)*bs], &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx], m-i, 4);
		}
	return;

	}



// D <= A^{-1} * B , with A lower triangular with unit diagonal
void strsm_nn_ll_one_lib(int m, int n, float *pA, int sda, float *pB, int sdb, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;

	for( ; i<m-3; i+=4)
		{
		j = 0;
		for( ; j<n-3; j+=4)
			{
			kernel_strsm_nn_ll_one_4x4_lib4(i, pA+i*sda, pD+j*bs, sdd, pB+i*sdb+j*bs, pD+i*sdd+j*bs, pA+i*sda+i*bs);
			}
		if(j<n)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(i, pA+i*sda, pD+j*bs, sdd, pB+i*sdb+j*bs, pD+i*sdd+j*bs, pA+i*sda+i*bs, m-i, n-j);
			}
		}
	if(i<m)
		{
		goto left_4;
		}

	// common return
	return;

	left_4:
	j = 0;
	for( ; j<n; j+=4)
		{
		kernel_strsm_nn_ll_one_4x4_vs_lib4(i, pA+i*sda, pD+j*bs, sdd, pB+i*sdb+j*bs, pD+i*sdd+j*bs, pA+i*sda+i*bs, m-i, n-j);
		}
	return;

	}



// D <= A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
void strsm_nn_lu_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *pB, int sdb, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j, idx;
	float *dummy;
	
	i = 0;
	int rm = m%4;
	if(rm>0)
		{
		// TODO code expliticly the final case
		idx = m-rm; // position of the part to do
		j = 0;
		for( ; j<n; j+=4)
			{
			// XXX pA & pD are dummy and should not be used internally !!!
			kernel_strsm_nn_lu_inv_4x4_vs_lib4(0, pD, pA, 0, pB+idx*sdb+j*bs, pD+idx*sdd+j*bs, pA+idx*sda+idx*bs, inv_diag_A+idx, rm, n-j);
			}
		// TODO
		i += rm;
		}
//	int em = m-rm;
	for( ; i<m; i+=4)
		{
		idx = m-i; // position of already done part
		j = 0;
		for( ; j<n-3; j+=4)
			{
			kernel_strsm_nn_lu_inv_4x4_lib4(i, pA+(idx-4)*sda+idx*bs, pD+idx*sdd+j*bs, sdd, pB+(idx-4)*sdb+j*bs, pD+(idx-4)*sdd+j*bs, pA+(idx-4)*sda+(idx-4)*bs, inv_diag_A+(idx-4));
			}
		if(j<n)
			{
			kernel_strsm_nn_lu_inv_4x4_vs_lib4(i, pA+(idx-4)*sda+idx*bs, pD+idx*sdd+j*bs, sdd, pB+(idx-4)*sdb+j*bs, pD+(idx-4)*sdd+j*bs, pA+(idx-4)*sda+(idx-4)*bs, inv_diag_A+(idx-4), 4, n-j);
			}
		}

	// common return
	return;

	}



/****************************
* new interface
****************************/



#if defined(LA_HIGH_PERFORMANCE)



// dgemm nt
void blasfeo_sgemm_nt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	
	const int bs = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	if(ai==0 & bi==0 & ci==0 & di==0)
		{
		sgemm_nt_lib(m, n, k, alpha, pA, sda, pB, sdb, beta, pC, sdc, pD, sdd); 
		return;
		}
	
	pA += ai/bs*bs*sda;
	pB += bi/bs*bs*sda;
	int ci0 = ci-ai%bs;
	int di0 = di-ai%bs;
	int offsetC;
	int offsetD;
	if(ci0>=0)
		{
		pC += ci0/bs*bs*sdd;
		offsetC = ci0%bs;
		}
	else
		{
		pC += -4*sdc;
		offsetC = bs+ci0;
		}
	if(di0>=0)
		{
		pD += di0/bs*bs*sdd;
		offsetD = di0%bs;
		}
	else
		{
		pD += -4*sdd;
		offsetD = bs+di0;
		}
	
	int i, j, l;

	int idxB;

	i = 0;
	// clean up at the beginning
	if(ai%bs!=0)
		{
		j = 0;
		idxB = 0;
		// clean up at the beginning
		if(bi%bs!=0)
			{
			kernel_sgemm_nt_4x4_gen_lib4(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*bs+i*sdc]-bi%bs*bs, sdc, offsetD, &pD[j*bs+i*sdd]-bi%bs*bs, sdd, ai%bs, m-i, bi%bs, n-j);
			j += bs-bi%bs;
			idxB += 4;
			}
		// main loop
		for(; j<n; j+=4)
			{
			kernel_sgemm_nt_4x4_gen_lib4(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*bs+i*sdc], sdc, offsetD, &pD[j*bs+i*sdd], sdd, ai%bs, m-i, 0, n-j);
			idxB += 4;
			}
		m -= bs-ai%bs;
		pA += bs*sda;
		pC += bs*sdc;
		pD += bs*sdd;
		}
	// main loop
	for(; i<m; i+=4)
		{
		j = 0;
		idxB = 0;
		// clean up at the beginning
		if(bi%bs!=0)
			{
			kernel_sgemm_nt_4x4_gen_lib4(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*bs+i*sdc]-bi%bs*bs, sdc, offsetD, &pD[j*bs+i*sdd]-bi%bs*bs, sdd, 0, m-i, bi%bs, n-j);
			j += bs-bi%bs;
			idxB += 4;
			}
		// main loop
		for(; j<n; j+=4)
			{
			kernel_sgemm_nt_4x4_gen_lib4(k, &alpha, &pA[i*sda], &pB[idxB*sdb], &beta, offsetC, &pC[j*bs+i*sdc], sdc, offsetD, &pD[j*bs+i*sdd], sdd, 0, m-i, 0, n-j);
			idxB += 4;
			}
		}

	return;

	}



// dgemm nn
void blasfeo_sgemm_nn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ai!=0 | ci!=0 | di!=0)
		{
		printf("\nblasfeo_sgemm_nn: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int ps = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;

	int bir = bi & (ps-1);

	float *pA = sA->pA + aj*ps;
	float *pB = sB->pA + bj*ps + (bi-bir)*sdb;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;

	int offsetB = bir;

//	sgemm_nn_lib(m, n, k, alpha, pA, sda, pB, sdb, beta, pC, sdc, pD, sdd); 
	int i, j, l;

	i = 0;

#if defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nn_8x4_lib4(k, &alpha, &pA[i*sda], sda, offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd);
			}
		if(j<n)
			{
			kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[(i+0)*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+(i+0)*sdc], &pD[j*ps+(i+0)*sdd], m-(i+0), n-j);
			kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[(i+4)*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+(i+4)*sdc], &pD[j*ps+(i+4)*sdd], m-(i+4), n-j);
			}
		}
#endif
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nn_4x4_lib4(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[i*sda], offsetB, &pB[j*ps], sdb, &beta, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], m-i, n-j);
		}
	return;


	return;

	}



// sgemm_tn
void blasfeo_sgemm_tn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_sgemm_tn: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// sgemm_tt
void blasfeo_sgemm_tt(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_sgemm_tt: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_nn_llu
void blasfeo_strsm_llnu(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nblasfeo_strsm_llnu: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;
	// TODO alpha
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	strsm_nn_ll_one_lib(m, n, pA, sda, pB, sdb, pD, sdd);
	return;
	}



// dtrsm_nn_lun
void blasfeo_strsm_lunn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nblasfeo_strsm_lunn: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;
	// TODO alpha
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	float *dA = sA->dA;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(n, 1.0, ai, pA, sda, dA);
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(n, 1.0, ai, pA, sda, dA);
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	strsm_nn_lu_inv_lib(m, n, pA, sda, dA, pB, sdb, pD, sdd);
	return;
	}



// dtrsm_right_lower_transposed_notunit
void blasfeo_strsm_rltn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nblasfeo_strsm_rltn: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;

	// TODO alpha

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	float *dA = sA->dA;

	int i, j;
	
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(n, 1.0, ai, pA, sda, dA);
			for(i=0; i<n; i++)
				dA[i] = 1.0 / dA[i];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(n, 1.0, ai, pA, sda, dA);
		for(i=0; i<n; i++)
			dA[i] = 1.0 / dA[i];
		sA->use_dA = 0;
		}

	if(m<=0 || n<=0)
		return;
	
	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda], &dA[j]);
			}
		if(j<n)
			{
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda], &dA[j], m-i, n-j);
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pA[j*sda], &pB[j*bs+i*sdb], &pD[j*bs+i*sdd], &pA[j*bs+j*sda], &dA[j], m-i, n-j);
		}
	return;

	}



// dtrsm_right_lower_transposed_unit
void blasfeo_strsm_rltu(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nblasfeo_strsm_rltu: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;
	// TODO alpha
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	strsm_nt_rl_one_lib(m, n, pA, sda, pB, sdb, pD, sdd); 
	return;
	}



// dtrsm_right_upper_transposed_notunit
void blasfeo_strsm_rutn(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nblasfeo_strsm_rutn: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;
	// TODO alpha
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	float *dA = sA->dA;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(n, 1.0, ai, pA, sda, dA);
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(n, 1.0, ai, pA, sda, dA);
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	strsm_nt_ru_inv_lib(m, n, pA, sda, dA, pB, sdb, pD, sdd); 
	return;
	}



// dtrmm_right_upper_transposed_notunit (B, i.e. the first matrix, is triangular !!!)
void blasfeo_strmm_rutn(int m, int n, float alpha, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0)
		{
		printf("\nblasfeo_strmm_rutn: feature not implemented yet: ai=%d, bi=%d, di=%d\n", ai, bi, di);
		exit(1);
		}

	if(m<=0 || n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;

	int i, j;
	
//	strmm_nt_ru_lib(m, n, alpha, pA, sda, pB, sdb, 0.0, pD, sdd, pD, sdd); 
	i = 0;
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_strmm_nt_ru_4x4_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &pD[j*bs+i*sdd]);
			}
		if(j<n) // TODO specialized edge routine
			{
			kernel_strmm_nt_ru_4x4_vs_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &pD[j*bs+i*sdd], m-i, n-j);
			}
		}
	if(i<m)
		{
		goto left_4;
		}
	
	// common return
	return;

	left_4:
	j = 0;
	for(; j<n; j+=4)
		{
		kernel_strmm_nt_ru_4x4_vs_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &pD[j*bs+i*sdd], m-i, n-j);
		}
	// TODO specialized edge routine

	return;

	}



// dtrmm_right_lower_nottransposed_notunit (B, i.e. the first matrix, is triangular !!!)
void blasfeo_strmm_rlnn(int m, int n, float alpha, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;

	pA += ai/bs*bs*sda;
	pB += bi/bs*bs*sdb;
	int offsetB = bi%bs;
	int di0 = di-ai%bs;
	int offsetD;
	if(di0>=0)
		{
		pD += di0/bs*bs*sdd;
		offsetD = di0%bs;
		}
	else
		{
		pD += -4*sdd;
		offsetD = bs+di0;
		}
	
	int ii, jj;

	ii = 0;
	if(ai%bs!=0)
		{
		jj = 0;
		for(; jj<n; jj+=4)
			{
			kernel_strmm_nn_rl_4x4_gen_lib4(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, ai%bs, m-ii, 0, n-jj);
			}
		m -= bs-ai%bs;
		pA += bs*sda;
		pD += bs*sdd;
		}
	if(offsetD==0)
		{
		for(; ii<m-3; ii+=4)
			{
			jj = 0;
			for(; jj<n-5; jj+=4)
				{
				kernel_strmm_nn_rl_4x4_lib4(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, &pD[ii*sdd+jj*bs]);
				}
			for(; jj<n; jj+=4)
				{
				kernel_strmm_nn_rl_4x4_gen_lib4(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, 0, &pD[ii*sdd+jj*bs], sdd, 0, 4, 0, n-jj);
				}
			}
		if(ii<m)
			{
			goto left_4;
			}
		}
	else
		{
		for(; ii<m; ii+=4)
			{
			jj = 0;
			for(; jj<n; jj+=4)
				{
				kernel_strmm_nn_rl_4x4_gen_lib4(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
				}
			}
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	jj = 0;
	for(; jj<n; jj+=4)
		{
		kernel_strmm_nn_rl_4x4_gen_lib4(n-jj, &alpha, &pA[ii*sda+jj*bs], offsetB, &pB[jj*sdb+jj*bs], sdb, offsetD, &pD[ii*sdd+jj*bs], sdd, 0, m-ii, 0, n-jj);
		}
	return;

	}



void blasfeo_ssyrk_ln(int m, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsryrk_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

//	ssyrk_nt_l_lib(m, n, k, alpha, pA, sda, pB, sdb, beta, pC, sdc, pD, sdd);

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_sgemm_nt_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		kernel_ssyrk_nt_l_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<i; j+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, m-j);
		}
	kernel_ssyrk_nt_l_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, m-j);
	return;

	}



void blasfeo_ssyrk_ln_mn(int m, int n, int k, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsryrk_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	const int bs = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

//	ssyrk_nt_l_lib(m, n, k, alpha, pA, sda, pB, sdb, beta, pC, sdc, pD, sdd);

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_sgemm_nt_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			if(j<i) // dgemm
				{
				kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-3)
					{
					kernel_ssyrk_nt_l_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
					}
				else
					{
					kernel_ssyrk_nt_l_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
					}
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<i && j<n; j+=4)
		{
		kernel_sgemm_nt_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	if(j<n)
		{
		kernel_ssyrk_nt_l_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void blasfeo_ssyrk_lt(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_ssyrk_lt: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



void blasfeo_ssyrk_un(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_ssyrk_un: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



void blasfeo_ssyrk_ut(int m, int k, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj, double beta, struct blasfeo_dmat *sC, int ci, int cj, struct blasfeo_dmat *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_ssyrk_ut: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



#else

#error : wrong LA choice

#endif




