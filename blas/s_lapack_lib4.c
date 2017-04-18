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
#include <math.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"



/****************************
* old interface
****************************/

void ssyrk_spotrf_nt_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;

	int alg = 1; // XXX

	const int bs = 4;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_sgemm_strsm_nt_rl_inv_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j]);
			}
		if(j<n)
			{
			if(i<j) // dgemm
				{
				kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-3)
					{
					kernel_ssyrk_spotrf_nt_l_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j]);
					}
				else
					{
					kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
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
	for(; j<i && j<n-3; j+=4)
		{
		kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dgemm
			{
			kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		else // dsyrk
			{
			kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &inv_diag_D[j], m-i, n-j);
			}
		}
	return;

	}



void sgetrf_nn_nopivot_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;

	int ii, jj, ie;

	// main loop
	ii = 0;
	for( ; ii<m-3; ii+=4)
		{
		jj = 0;
		// solve lower
		ie = n<ii ? n : ii; // ie is multiple of 4
		for( ; jj<ie-3; jj+=4)
			{
			kernel_strsm_nn_ru_inv_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj]);
			}
		if(jj<ie)
			{
			kernel_strsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
			jj+=4;
			}
		// factorize
		if(jj<n-3)
			{
			kernel_sgetrf_nn_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_sgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		// solve upper 
		for( ; jj<n-3; jj+=4)
			{
			kernel_strsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd]);
			}
		if(jj<n)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
			}
		}
	if(m>ii)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	left_4:
	jj = 0;
	// solve lower
	ie = n<ii ? n : ii; // ie is multiple of 4
	for( ; jj<ie; jj+=4)
		{
		kernel_strsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[jj*bs+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
		}
	// factorize
	if(jj<n)
		{
		kernel_sgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	// solve upper 
	for( ; jj<n; jj+=4)
		{
		kernel_strsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*bs], sdd, &pC[jj*bs+ii*sdc], &pD[jj*bs+ii*sdd], &pD[ii*bs+ii*sdd], m-ii, n-jj);
		}
	return;

	}



void sgetrf_nn_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D, int *ipiv)
	{

	if(m<=0)
		return;
	
	const int bs = 4;

	int ii, jj, i0, i1, j0, ll, p;

	float d1 = 1.0;
	float dm1 = -1.0;

//	// needs to perform row-excanges on the yet-to-be-factorized matrix too
//	if(pC!=pD)
//		sgecp_lib(m, n, 1.0, 0, pC, sdc, 0, pD, sdd);

	// minimum matrix size
	p = n<m ? n : m; // XXX

	// main loop
	// 4 columns at a time
	jj = 0;
	for(; jj<p-3; jj+=4) // XXX
		{
		// pivot & factorize & solve lower
		ii = jj;
		i0 = ii;
		for( ; ii<m-3; ii+=4)
			{
			kernel_sgemm_nn_4x4_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd]);
			}
		if(m-ii>0)
			{
			kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, 4);
			}
		kernel_sgetrf_pivot_4_lib4(m-i0, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			srowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
			srowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			srowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			srowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
			srowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			srowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
			srowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
			}

		// solve upper
		ll = jj+4;
		for( ; ll<n-3; ll+=4)
			{
			kernel_strsm_nn_ll_one_4x4_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd]);
			}
		if(n-ll>0)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], 4, n-ll);
			}
		}
	if(m>=n)
		{
		if(n-jj>0)
			{
			goto left_n_4;
			}
		}
	else
		{
		if(m-jj>0)
			{
			goto left_m_4;
			}
		}

	// common return if jj==n
	return;

	// clean up

	left_n_4:
	// 1-4 columns at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
	for( ; ii<m; ii+=4)
		{
		kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, n-jj);
		}
	kernel_sgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		srowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		srowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	if(n-jj>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			srowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		if(n-jj>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				srowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				srowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
				}
			if(n-jj>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					srowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					srowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
					}
				}
			}
		}

	// solve upper
	if(0) // there is no upper
		{
		ll = jj+4;
		for( ; ll<n; ll+=4)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], m-i0, n-ll);
			}
		}
	return;


	left_m_4:
	// 1-4 rows at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
	kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], &pD[jj*bs], sdd, &d1, &pD[jj*bs+ii*sdd], &pD[jj*bs+ii*sdd], m-ii, n-jj);
	kernel_sgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*bs+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		srowsw_lib(jj, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs);
		srowsw_lib(n-jj-4, pD+(i0+0)/bs*bs*sdd+(i0+0)%bs+(jj+4)*bs, pD+(ipiv[i0+0])/bs*bs*sdd+(ipiv[i0+0])%bs+(jj+4)*bs);
		}
	if(m-i0>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs);
			srowsw_lib(n-jj-4, pD+(i0+1)/bs*bs*sdd+(i0+1)%bs+(jj+4)*bs, pD+(ipiv[i0+1])/bs*bs*sdd+(ipiv[i0+1])%bs+(jj+4)*bs);
			}
		if(m-i0>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				srowsw_lib(jj, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs);
				srowsw_lib(n-jj-4, pD+(i0+2)/bs*bs*sdd+(i0+2)%bs+(jj+4)*bs, pD+(ipiv[i0+2])/bs*bs*sdd+(ipiv[i0+2])%bs+(jj+4)*bs);
				}
			if(m-i0>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					srowsw_lib(jj, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs);
					srowsw_lib(n-jj-4, pD+(i0+3)/bs*bs*sdd+(i0+3)%bs+(jj+4)*bs, pD+(ipiv[i0+3])/bs*bs*sdd+(ipiv[i0+3])%bs+(jj+4)*bs);
					}
				}
			}
		}

	// solve upper
	ll = jj+4;
	for( ; ll<n; ll+=4)
		{
		kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*bs], sdd, &pD[ll*bs+i0*sdd], &pD[ll*bs+i0*sdd], &pD[i0*bs+i0*sdd], m-i0, n-ll);
		}
	return;

	}



/****************************
* new interface
****************************/



#if defined(LA_HIGH_PERFORMANCE)



// dpotrf
void spotrf_l_libstr(int m, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ci!=0 | di!=0)
		{
		printf("\nspotrf_l_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}

	const int bs = 4;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA;
	if(di==0 && dj==0) // XXX what to do if di and dj are not zero
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j]);
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4: // 1 - 3
	j = 0;
	for(; j<i; j+=4)
		{
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
		}
	kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, m-j);
	return;

	return;
	}



// dpotrf
void spotrf_l_mn_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ci!=0 | di!=0)
		{
		printf("\nspotrf_l_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}

	const int bs = 4;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA;
	if(di==0 && dj==0) // XXX what to do if di and dj are not zero
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(i<j) // dtrsm
				{
				kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-3)
					{
					kernel_spotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j]);
					}
				else
					{
					kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
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
	for(; j<i && j<n-3; j+=4)
		{
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
			}
		else // dpotrf
			{
			kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+j*sdc], &pD[j*bs+j*sdd], &dD[j], m-i, n-j);
			}
		}
	return;

	return;
	}



// dsyrk dpotrf
void ssyrk_spotrf_ln_libstr(int m, int n, int k, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
		printf("\nssyrk_spotrf_ln_libstr: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
		exit(1);
		}
	const int bs = S_BS;
	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero
	ssyrk_spotrf_nt_l_lib(m, n, k, pA, sda, pB, sdb, pC, sdc, pD, sdd, dD);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}



// dgetrf without pivoting
void sgetrf_nopivot_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
		printf("\nsgetf_nopivot_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}
	const int bs = S_BS;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero
	sgetrf_nn_nopivot_lib(m, n, pC, sdc, pD, sdd, dD);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}




// dgetrf pivoting
void sgetrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, int *ipiv)
	{
	if(ci!=0 | di!=0)
		{
		printf("\nsgetrf_libstr: feature not implemented yet: ci=%d, di=%d\n", ci, di);
		exit(1);
		}
	const int bs = S_BS;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero
	// needs to perform row-excanges on the yet-to-be-factorized matrix too
	if(pC!=pD)
		sgecp_libstr(m, n, sC, ci, cj, sD, di, dj);
	sgetrf_nn_lib(m, n, pC, sdc, pD, sdd, dD, ipiv);
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;
	return;
	}



int sgeqrf_work_size_libstr(int m, int n)
	{
	printf("\nsgeqrf_work_size_libstr: feature not implemented yet\n");
	exit(1);
	return 0;
	}



void sgeqrf_libstr(int m, int n, struct s_strmat *sC, int ci, int cj, struct s_strmat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
	printf("\nsgeqrf_libstr: feature not implemented yet\n");
	exit(1);
	return;
	}



#else

#error : wrong LA choice

#endif



