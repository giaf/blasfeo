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

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)
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
#if defined(TARGET_ARMV7A_ARM_CORTEX_A15)  | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
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
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	for(; i<m-7; i+=8)
		{
		j = 0;
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)
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



void sgemm_nn_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_sgemm_nn_4x4_lib4(k, &alpha, &pA[i*sda], &pB[j*bs], sdb, &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n)
			{
			kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*bs], sdb, &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
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
		kernel_sgemm_nn_4x4_vs_lib4(k, &alpha, &pA[i*sda], &pB[j*bs], sdb, &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
	return;

	}



void strmm_nt_ru_lib(int m, int n, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd)
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
			kernel_strmm_nt_ru_4x4_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd]);
			}
		if(j<n) // TODO specialized edge routine
			{
			kernel_strmm_nt_ru_4x4_vs_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
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
//	for(; j<n-3; j+=4)
	for(; j<n; j+=4)
		{
		kernel_strmm_nt_ru_4x4_vs_lib4(n-j, &alpha, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], &beta, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
		}
//	if(j<n) // TODO specialized edge routine
//		{
//		kernel_strmm_nt_ru_4x4_vs_lib4(n-j, &pA[j*bs+i*sda], &pB[j*bs+j*sdb], alg, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], m-i, n-j);
//		}
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

	float *dummy;
	
	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		// clean at the end
		if(rn>0)
			{
			idx = n-rn;
			kernel_strsm_nt_ru_inv_4x4_vs_lib4(0, dummy, dummy, &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx], m-i, rn);
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
		kernel_strsm_nt_ru_inv_4x4_vs_lib4(0, dummy, dummy, &pB[i*sdb+idx*bs], &pD[i*sdd+idx*bs], &pA[idx*sda+idx*bs], &inv_diag_A[idx], m-i, rn);
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
			kernel_strsm_nn_lu_inv_4x4_vs_lib4(0, dummy, dummy, 0, pB+idx*sdb+j*bs, pD+idx*sdd+j*bs, pA+idx*sda+idx*bs, inv_diag_A+idx, rm, n-j);
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
void sgemm_nt_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0 | n<=0)
		return;
	
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
void sgemm_nn_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{
	if(m<=0 || n<=0)
		return;
	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsgemm_nn_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	sgemm_nn_lib(m, n, k, alpha, pA, sda, pB, sdb, beta, pC, sdc, pD, sdd); 
	return;
	}
	


// dtrsm_nn_llu
void strsm_llnu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_llnu_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}
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
void strsm_lunn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_lunn_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}
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
void strsm_rltn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj)
	{

	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_rltn_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}

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
void strsm_rltu_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_rltu_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}
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
void strsm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nstrsm_rutn_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d, alpha=%f\n", ai, bi, di, alpha);
		exit(1);
		}
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
void strmm_rutn_libstr(int m, int n, float alpha, struct s_strmat *sB, int bi, int bj, struct s_strmat *sA, int ai, int aj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0)
		{
		printf("\nstrmm_rutn_libstr: feature not implemented yet: ai=%d, bi=%d, di=%d\n", ai, bi, di);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pD = sD->pA + dj*bs;
	strmm_nt_ru_lib(m, n, alpha, pA, sda, pB, sdb, 0.0, pD, sdd, pD, sdd); 
	return;
	}



// dtrmm_right_lower_nottransposed_notunit (B, i.e. the first matrix, is triangular !!!)
void strmm_rlnn_libstr(int m, int n, float alpha, struct s_strmat *sB, int bi, int bj, struct s_strmat *sA, int ai, int aj, struct s_strmat *sD, int di, int dj)
	{

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



void ssyrk_ln_libstr(int m, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsryrk_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}

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



void ssyrk_ln_mn_libstr(int m, int n, int k, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, float beta, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsryrk_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}

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
			if(i<j) // dgemm
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



#else

#error : wrong LA choice

#endif




