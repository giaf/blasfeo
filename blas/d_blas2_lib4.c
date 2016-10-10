/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"


void dgemv_n_lib(int m, int n, double alpha, double *pA, int sda, double *x, double beta, double *y, double *z)
	{

	if(m<=0)
		return;
	
	const int bs = 4;

	int i;

	i = 0;
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for( ; i<m-11; i+=12)
		{
		kernel_dgemv_n_12_lib4(n, &alpha, &pA[i*sda], sda, x, &beta, &y[i], &z[i]);
		}
#endif
	for( ; i<m-7; i+=8)
		{
		kernel_dgemv_n_8_lib4(n, &alpha, &pA[i*sda], sda, x, &beta, &y[i], &z[i]);
		}
	if(i<m-3)
		{
		kernel_dgemv_n_4_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i]);
		i+=4;
		}
#else
	for( ; i<m-3; i+=4)
		{
		kernel_dgemv_n_4_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i]);
		}
#endif
	if(i<m)
		{
		kernel_dgemv_n_4_vs_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i], m-i);
		}
	
	}



void dgemv_t_lib(int m, int n, double alpha, double *pA, int sda, double *x, double beta, double *y, double *z)
	{

	if(n<=0)
		return;
	
	const int bs = 4;

	int i;

	i = 0;
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for( ; i<n-11; i+=12)
		{
		kernel_dgemv_t_12_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
		}
#endif
	for( ; i<n-7; i+=8)
		{
		kernel_dgemv_t_8_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
		}
	if(i<n-3)
		{
		kernel_dgemv_t_4_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
		i+=4;
		}
#else
	for( ; i<n-3; i+=4)
		{
		kernel_dgemv_t_4_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
		}
#endif
	if(i<n)
		{
		kernel_dgemv_t_4_vs_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i], n-i);
		}
	
	}



void dtrsv_ln_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y)
	{

	if(m<=0 || n<=0)
		return;
	
	// suppose m>=n
	if(m<n)
		m = n;

	const int bs = 4;

	double alpha = -1.0;
	double beta = 1.0;

	int i;

	if(x!=y)
		{
		for(i=0; i<m; i++)
			y[i] = x[i];
		}
	
	i = 0;
	for( ; i<n-3; i+=4)
		{
		kernel_dtrsv_ln_inv_4_lib4(i, &pA[i*sda], &inv_diag_A[i], x, &y[i], &y[i]);
		}
	if(i<n)
		{
		kernel_dtrsv_ln_inv_4_vs_lib4(i, &pA[i*sda], &inv_diag_A[i], x, &y[i], &y[i], m-i, n-i);
		i+=4;
		}
#if defined(TARGET_X64_INTEL_SANDY_BRIDGE) || defined(TARGET_X64_INTEL_HASWELL)
	for( ; i<m-7; i+=8)
		{
		kernel_dgemv_n_8_lib4(n, &alpha, &pA[i*sda], sda, x, &beta, &y[i], &y[i]);
		}
	if(i<m-3)
		{
		kernel_dgemv_n_4_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &y[i]);
		i+=4;
		}
#else
	for( ; i<m-3; i+=4)
		{
		kernel_dgemv_n_4_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &y[i]);
		}
#endif
	if(i<m)
		{
		kernel_dgemv_n_4_vs_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &y[i], m-i);
		i+=4;
		}

	}



void dtrsv_lt_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y)
	{

	if(m<=0 || n<=0)
		return;

	if(n>m)
		n = m;
	
	const int bs = 4;
	
	int i;
	
	if(x!=y)
		for(i=0; i<m; i++)
			y[i] = x[i];
			
	i=0;
	if(n%4==1)
		{
		kernel_dtrsv_lt_inv_1_lib4(m-n+i+1, &pA[n/bs*bs*sda+(n-i-1)*bs], sda, &inv_diag_A[n-i-1], &y[n-i-1], &y[n-i-1], &y[n-i-1]);
		i++;
		}
	else if(n%4==2)
		{
		kernel_dtrsv_lt_inv_2_lib4(m-n+i+2, &pA[n/bs*bs*sda+(n-i-2)*bs], sda, &inv_diag_A[n-i-2], &y[n-i-2], &y[n-i-2], &y[n-i-2]);
		i+=2;
		}
	else if(n%4==3)
		{
		kernel_dtrsv_lt_inv_3_lib4(m-n+i+3, &pA[n/bs*bs*sda+(n-i-3)*bs], sda, &inv_diag_A[n-i-3], &y[n-i-3], &y[n-i-3], &y[n-i-3]);
		i+=3;
		}
	for(; i<n-3; i+=4)
		{
		kernel_dtrsv_lt_inv_4_lib4(m-n+i+4, &pA[(n-i-4)/bs*bs*sda+(n-i-4)*bs], sda, &inv_diag_A[n-i-4], &y[n-i-4], &y[n-i-4], &y[n-i-4]);
		}

	}



void dtrmv_un_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z)
	{

	if(m<=0)
		return;

	const int bs = 4;
	
	int i;
	
	i=0;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; i<m-7; i+=8)
		{
		kernel_dtrmv_un_8_lib4(m-i, pA, sda, x, alg, y, z);
		pA += 8*sda+8*bs;
		x  += 8;
		y  += 8;
		z  += 8;
		}
#endif
	for(; i<m-3; i+=4)
		{
		kernel_dtrmv_un_4_lib4(m-i, pA, x, alg, y, z);
		pA += 4*sda+4*bs;
		x  += 4;
		y  += 4;
		z  += 4;
		}
	if(m>i)
		{
		if(m-i==1)
			{
			if(alg==0)
				{
				z[0] = pA[0+bs*0]*x[0];
				}
			else if(alg==1)
				{
				z[0] = y[0] + pA[0+bs*0]*x[0];
				}
			else
				{
				z[0] = y[0] - pA[0+bs*0]*x[0];
				}
			}
		else if(m-i==2)
			{
			if(alg==0)
				{
				z[0] = pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1];
				z[1] = pA[1+bs*1]*x[1];
				}
			else if(alg==1)
				{
				z[0] = y[0] + pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1];
				z[1] = y[1] + pA[1+bs*1]*x[1];
				}
			else
				{
				z[0] = y[0] - pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1];
				z[1] = y[1] - pA[1+bs*1]*x[1];
				}
			}
		else // if(m-i==3)
			{
			if(alg==0)
				{
				z[0] = pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2];
				z[1] = pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2];
				z[2] = pA[2+bs*2]*x[2];
				}
			else if(alg==1)
				{
				z[0] = y[0] + pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2];
				z[1] = y[1] + pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2];
				z[2] = y[2] + pA[2+bs*2]*x[2];
				}
			else
				{
				z[0] = y[0] - pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2];
				z[1] = y[1] - pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2];
				z[2] = y[2] - pA[2+bs*2]*x[2];
				}
			}
		}

	}



void dtrmv_ut_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z)
	{

	if(m<=0)
		return;

	const int bs = 4;
	
	int i;
	
	double *ptrA;
	
	i=0;
	for(; i<m-3; i+=4)
		{
		kernel_dtrmv_ut_4_lib4(i+4, pA, sda, x, alg, y, z);
		pA += 4*bs;
		y  += bs;
		z  += bs;
		}
	if(i<m)
		{
		kernel_dtrmv_ut_4_vs_lib4(m, pA, sda, x, alg, y, z, m-i);
		}

	}



void dgemv_nt_lib(int m, int n, double alpha_n, double alpha_t, double *pA, int sda, double *x_n, double *x_t, double beta_n, double beta_t, double *y_n, double *y_t, double *z_n, double *z_t)
	{

	if(m<=0 | n<=0)
		return;

	const int bs = 4;

	int ii;

	// copy and scale y_n int z_n
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z_n[ii+0] = beta_n*y_n[ii+0];
		z_n[ii+1] = beta_n*y_n[ii+1];
		z_n[ii+2] = beta_n*y_n[ii+2];
		z_n[ii+3] = beta_n*y_n[ii+3];
		}
	for(; ii<m; ii++)
		{
		z_n[ii+0] = beta_n*y_n[ii+0];
		}
	
	ii = 0;
	for(; ii<n-3; ii+=4)
		{
		kernel_dgemv_nt_4_lib4(m, &alpha_n, &alpha_t, pA+ii*bs, sda, x_n+ii, x_t, &beta_t, y_t+ii, z_n, z_t+ii);
		}
	if(ii<n)
		{
		kernel_dgemv_nt_4_vs_lib4(m, &alpha_n, &alpha_t, pA+ii*bs, sda, x_n+ii, x_t, &beta_t, y_t+ii, z_n, z_t+ii, n-ii);
		}
	
	return;

	}



void dsymv_l_lib(int m, int n, double alpha, double *pA, int sda, double *x, double beta, double *y, double *z)
	{

	if(m<=0 | n<=0)
		return;
	
	const int bs = 0;

	int ii;

	// copy and scale y int z
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = beta*y[ii+0];
		z[ii+1] = beta*y[ii+1];
		z[ii+2] = beta*y[ii+2];
		z[ii+3] = beta*y[ii+3];
		}
	for(; ii<m; ii++)
		{
		z[ii+0] = beta*y[ii+0];
		}
	
	ii = 0;
	for(; ii<n-3; ii+=4)
		{
		kernel_dsymv_l_4_lib4(m, &alpha, pA+ii*bs, sda, x+ii, x, z, z+ii);
		return;
		}
	if(ii<n)
		{
		kernel_dsymv_l_4_vs_lib4(m, &alpha, pA+ii*bs, sda, x+ii, x, z, z+ii, n-ii);
		}
	
	return;

	}
	
#if defined(LA_BLASFEO)



void dgemv_nt_libstr(int m, int n, double alpha_n, double alpha_t, struct d_strmat *sA, int ai, int aj, double *x_n, double *x_t, double beta_n, double beta_t, double *y_n, double *y_t, double *z_n, double *z_t)
	{
	if(ai!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	dgemv_nt_lib(m, n, alpha_n, alpha_t, pA, sda, x_n, x_t, beta_n, beta_t, y_n, y_t, z_n, z_t);
	return;
	}



void dsymv_l_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, double *x, double beta, double *y, double *z)
	{
	if(ai!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	dsymv_l_lib(m, n, alpha, pA, sda, x, beta, y, z);
	return;
	}



#elif defined(LA_BLAS)



void dgemv_nt_libstr(int m, int n, double alpha_n, double alpha_t, struct d_strmat *sA, int ai, int aj, double *x_n, double *x_t, double beta_n, double beta_t, double *y_n, double *y_t, double *z_n, double *z_t)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	// n
	dcopy_(&m, y_n, &i1, z_n, &i1);
	dgemv_(&cn, &m, &n, &alpha_n, pA, &lda, x_n, &i1, &beta_n, z_n, &i1);
	// n
	dcopy_(&n, y_t, &i1, z_t, &i1);
	dgemv_(&ct, &m, &n, &alpha_t, pA, &lda, x_t, &i1, &beta_t, z_t, &i1);
	return;
	}



void dsymv_l_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, double *x, double beta, double *y, double *z)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	dcopy_(&m, y, &i1, z, &i1);
	dsymv_(&cl, &n, &alpha, pA, &lda, x, &i1, &beta, z, &i1);
	int tmp = m-n;
	dgemv_(&cn, &tmp, &n, &alpha, pA+n, &lda, x, &i1, &beta, z+n, &i1);
	dgemv_(&ct, &tmp, &n, &alpha, pA+n, &lda, x+n, &i1, &d1, z, &i1);
	return;
	}



#else

#error : wrong LA choice

#endif
