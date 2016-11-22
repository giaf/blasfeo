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

#if defined(LA_BLAS)
#include <f77blas.h>
#endif

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_aux.h"


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
	
	int ii, idx;
	
	double *ptrA;
	
	ii=0;
	idx = m/bs*bs;
	if(m%bs!=0)
		{
		kernel_dtrmv_ut_4_vs_lib4(m, pA+idx*bs, sda, x, alg, y+idx, z+idx, m%bs);
		ii += m%bs;
		}
	idx -= 4;
	for(; ii<m; ii+=4)
		{
		kernel_dtrmv_ut_4_lib4(idx+4, pA+idx*bs, sda, x, alg, y+idx, z+idx);
		idx -= 4;
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
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<n-5; ii+=6)
		{
		kernel_dgemv_nt_6_lib4(m, &alpha_n, &alpha_t, pA+ii*bs, sda, x_n+ii, x_t, &beta_t, y_t+ii, z_n, z_t+ii);
		}
#endif
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
	
	const int bs = 4;

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
		kernel_dsymv_l_4_lib4(m-ii, &alpha, pA+ii*bs+ii*sda, sda, x+ii, x+ii, z+ii, z+ii);
		}
	if(ii<n)
		{
		kernel_dsymv_l_4_vs_lib4(m-ii, &alpha, pA+ii*bs+ii*sda, sda, x+ii, x+ii, z+ii, z+ii, n-ii);
		}
	
	return;

	}
	
#if defined(LA_BLASFEO)



void dgemv_n_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	dgemv_n_lib(m, n, alpha, pA, sda, x, beta, y, z);
	return;
	}



void dgemv_t_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	dgemv_t_lib(m, n, alpha, pA, sda, x, beta, y, z);
	return;
	}



void dgemv_nt_libstr(int m, int n, double alpha_n, double alpha_t, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx_n, int xi_n, struct d_strvec *sx_t, int xi_t, double beta_n, double beta_t, struct d_strvec *sy_n, int yi_n, struct d_strvec *sy_t, int yi_t, struct d_strvec *sz_n, int zi_n, struct d_strvec *sz_t, int zi_t)
	{
	if(ai!=0 | xi_n%4!=0 | xi_t%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x_n = sx_n->pa + xi_n;
	double *x_t = sx_t->pa + xi_t;
	double *y_n = sy_n->pa + yi_n;
	double *y_t = sy_t->pa + yi_t;
	double *z_n = sz_n->pa + zi_n;
	double *z_t = sz_t->pa + zi_t;
	dgemv_nt_lib(m, n, alpha_n, alpha_t, pA, sda, x_n, x_t, beta_n, beta_t, y_n, y_t, z_n, z_t);
	return;
	}



void dsymv_l_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	dsymv_l_lib(m, n, alpha, pA, sda, x, beta, y, z);
	return;
	}



void dtrmv_unn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dtrmv_un_lib(m, pA, sda, x, 0, z, z);
	return;
	}



void dtrmv_utn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dtrmv_ut_lib(m, pA, sda, x, 0, z, z);
	return;
	}



void dtrsv_lnn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *dA = sA->dA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			ddiaex_lib(n, ai, pA, sda, dA);
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		ddiaex_lib(n, ai, pA, sda, dA);
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	dtrsv_ln_inv_lib(m, n, pA, sda, dA, x, z);
	return;
	}



void dtrsv_ltn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	if(ai!=0 | xi%4!=0)
		{
		printf("\nfeature not implemented yet\n");
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	double *pA = sA->pA + aj*bs; // TODO ai
	double *dA = sA->dA + aj*bs; // TODO ai
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			ddiaex_lib(n, ai, pA, sda, dA);
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		ddiaex_lib(n, ai, pA, sda, dA);
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	dtrsv_lt_inv_lib(m, n, pA, sda, dA, x, z);
	return;
	}



#elif defined(LA_BLAS)



void dgemv_n_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	// n
	dcopy_(&m, y, &i1, z, &i1);
	dgemv_(&cn, &m, &n, &alpha, pA, &lda, x, &i1, &beta, z, &i1);
	return;
	}



void dgemv_t_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	// n
	dcopy_(&n, y, &i1, z, &i1);
	dgemv_(&ct, &m, &n, &alpha, pA, &lda, x, &i1, &beta, z, &i1);
	return;
	}



void dgemv_nt_libstr(int m, int n, double alpha_n, double alpha_t, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx_n, int xi_n, struct d_strvec *sx_t, int xi_t, double beta_n, double beta_t, struct d_strvec *sy_n, int yi_n, struct d_strvec *sy_t, int yi_t, struct d_strvec *sz_n, int zi_n, struct d_strvec *sz_t, int zi_t)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x_n = sx_n->pa + xi_n;
	double *x_t = sx_t->pa + xi_t;
	double *y_n = sy_n->pa + yi_n;
	double *y_t = sy_t->pa + yi_t;
	double *z_n = sz_n->pa + zi_n;
	double *z_t = sz_t->pa + zi_t;
	// n
	dcopy_(&m, y_n, &i1, z_n, &i1);
	dgemv_(&cn, &m, &n, &alpha_n, pA, &lda, x_n, &i1, &beta_n, z_n, &i1);
	// t
	dcopy_(&n, y_t, &i1, z_t, &i1);
	dgemv_(&ct, &m, &n, &alpha_t, pA, &lda, x_t, &i1, &beta_t, z_t, &i1);
	return;
	}



void dsymv_l_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
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
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	dcopy_(&m, y, &i1, z, &i1);
	dsymv_(&cl, &n, &alpha, pA, &lda, x, &i1, &beta, z, &i1);
	int tmp = m-n;
	dgemv_(&cn, &tmp, &n, &alpha, pA+n, &lda, x, &i1, &beta, z+n, &i1);
	dgemv_(&ct, &tmp, &n, &alpha, pA+n, &lda, x+n, &i1, &d1, z, &i1);
	return;
	}



void dtrmv_lnn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dtrmv_(&cl, &cn, &cn, &m, pA, &lda, z, &i1);
	return;
	}



void dtrmv_ltn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dtrmv_(&cl, &ct, &cn, &m, pA, &lda, z, &i1);
	return;
	}



void dtrmv_unn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dtrmv_(&cu, &cn, &cn, &m, pA, &lda, z, &i1);
	return;
	}



void dtrmv_utn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dtrmv_(&cu, &ct, &cn, &m, pA, &lda, z, &i1);
	return;
	}



void dtrsv_lnn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int mmn = m-n;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dtrsv_(&cl, &cn, &cn, &n, pA, &lda, z, &i1);
	dgemv_(&cn, &mmn, &n, &dm1, pA+n, &lda, z, &i1, &d1, z+n, &i1);
	return;
	}



void dtrsv_ltn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double d1 = 1.0;
	double dm1 = -1.0;
	int mmn = m-n;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	dcopy_(&m, x, &i1, z, &i1);
	dgemv_(&ct, &mmn, &n, &dm1, pA+n, &lda, z+n, &i1, &d1, z, &i1);
	dtrsv_(&cl, &ct, &cn, &n, pA, &lda, z, &i1);
	return;
	}



#elif defined(LA_REFERENCE)



void dgemv_n_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double 
		y_0, y_1, y_2, y_3,
		x_0, x_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
#if 1 // y reg version
	ii = 0;
	for(; ii<m-1; ii+=2)
		{
		y_0 = 0.0;
		y_1 = 0.0;
		jj = 0;
		for(; jj<n-1; jj+=2)
			{
			y_0 += pA[ii+0+lda*(jj+0)] * x[jj+0] + pA[ii+0+lda*(jj+1)] * x[jj+1];
			y_1 += pA[ii+1+lda*(jj+0)] * x[jj+0] + pA[ii+1+lda*(jj+1)] * x[jj+1];
			}
		if(jj<n)
			{
			y_0 += pA[ii+0+lda*jj] * x[jj];
			y_1 += pA[ii+1+lda*jj] * x[jj];
			}
		z[ii+0] = beta * y[ii+0] + alpha * y_0;
		z[ii+1] = beta * y[ii+1] + alpha * y_1;
		}
	for(; ii<m; ii++)
		{
		y_0 = 0.0;
		for(jj=0; jj<n; jj++)
			{
			y_0 += pA[ii+lda*jj] * x[jj];
			}
		z[ii] = beta * y[ii] + alpha * y_0;
		}
#else // x reg version
	for(ii=0; ii<n; ii++)
		{
		z[ii] = beta * y[ii];
		}
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		x_0 = alpha * x[jj+0];
		x_1 = alpha * x[jj+1];
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			z[ii+0] += pA[ii+0+lda*(jj+0)] * x_0 + pA[ii+0+lda*(jj+1)] * x_1;
			z[ii+1] += pA[ii+1+lda*(jj+0)] * x_0 + pA[ii+1+lda*(jj+1)] * x_1;
			}
		for(; ii<m; ii++)
			{
			z[ii] += pA[ii+lda*(jj+0)] * x_0;
			z[ii] += pA[ii+lda*(jj+1)] * x_1;
			}
		}
	for(; jj<n; jj++)
		{
		x_0 = alpha * x[jj+0];
		for(ii=0; ii<m; ii++)
			{
			z[ii] += pA[ii+lda*(jj+0)] * x_0;
			}
		}
#endif
	return;
	}



void dgemv_t_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double 
		y_0, y_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		y_0 = 0.0;
		y_1 = 0.0;
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			y_0 += pA[ii+0+lda*(jj+0)] * x[ii+0] + pA[ii+1+lda*(jj+0)] * x[ii+1];
			y_1 += pA[ii+0+lda*(jj+1)] * x[ii+0] + pA[ii+1+lda*(jj+1)] * x[ii+1];
			}
		if(ii<m)
			{
			y_0 += pA[ii+lda*(jj+0)] * x[ii];
			y_1 += pA[ii+lda*(jj+1)] * x[ii];
			}
		z[jj+0] = beta * y[jj+0] + alpha * y_0;
		z[jj+1] = beta * y[jj+1] + alpha * y_1;
		}
	for(; jj<n; jj++)
		{
		y_0 = 0.0;
		for(ii=0; ii<m; ii++)
			{
			y_0 += pA[ii+lda*(jj+0)] * x[ii];
			}
		z[jj+0] = beta * y[jj+0] + alpha * y_0;
		}
	return;
	}



void dgemv_nt_libstr(int m, int n, double alpha_n, double alpha_t, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx_n, int xi_n, struct d_strvec *sx_t, int xi_t, double beta_n, double beta_t, struct d_strvec *sy_n, int yi_n, struct d_strvec *sy_t, int yi_t, struct d_strvec *sz_n, int zi_n, struct d_strvec *sz_t, int zi_t)
	{
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x_n = sx_n->pa + xi_n;
	double *x_t = sx_t->pa + xi_t;
	double *y_n = sy_n->pa + yi_n;
	double *y_t = sy_t->pa + yi_t;
	double *z_n = sz_n->pa + zi_n;
	double *z_t = sz_t->pa + zi_t;
	// n
	printf("\nfeature not implemented yet\n");
	exit(1);
//	dcopy_(&m, y_n, &i1, z_n, &i1);
//	dgemv_(&cn, &m, &n, &alpha_n, pA, &lda, x_n, &i1, &beta_n, z_n, &i1);
	// t
//	dcopy_(&n, y_t, &i1, z_t, &i1);
//	dgemv_(&ct, &m, &n, &alpha_t, pA, &lda, x_t, &i1, &beta_t, z_t, &i1);
	return;
	}



void dsymv_l_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, double beta, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
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
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	printf("\nfeature not implemented yet\n");
	exit(1);
//	dcopy_(&m, y, &i1, z, &i1);
//	dsymv_(&cl, &n, &alpha, pA, &lda, x, &i1, &beta, z, &i1);
//	int tmp = m-n;
//	dgemv_(&cn, &tmp, &n, &alpha, pA+n, &lda, x, &i1, &beta, z+n, &i1);
//	dgemv_(&ct, &tmp, &n, &alpha, pA+n, &lda, x+n, &i1, &d1, z, &i1);
	return;
	}



void dtrmv_lnn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double
		y_0, y_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	if(m%2!=0)
		{
		ii = m-1;
		y_0 = x[ii];
		y_0 *= pA[ii+lda*ii];
		for(jj=0; jj<ii; jj++)
			{
			y_0 += pA[ii+lda*jj] * x[jj];
			}
		z[ii] = y_0;
		m -= 1;
		}
	for(ii=m-2; ii>=0; ii-=2)
		{
		y_0 = x[ii+0];
		y_1 = x[ii+1];
		y_1 *= pA[ii+1+lda*(ii+1)];
		y_1 += pA[ii+1+lda*(ii+0)] * y_0;
		y_0 *= pA[ii+0+lda*(ii+0)];
		jj = 0;
		for(; jj<ii-1; jj+=2)
			{
			y_0 += pA[ii+0+lda*(jj+0)] * x[jj+0] + pA[ii+0+lda*(jj+1)] * x[jj+1];
			y_1 += pA[ii+1+lda*(jj+0)] * x[jj+0] + pA[ii+1+lda*(jj+1)] * x[jj+1];
			}
//	XXX there is no clean up loop !!!!!
//		for(; jj<ii; jj++)
//			{
//			y_0 += pA[ii+0+lda*jj] * x[jj];
//			y_1 += pA[ii+1+lda*jj] * x[jj];
//			}
		z[ii+0] = y_0;
		z[ii+1] = y_1;
		}
	return;
	}


	
void dtrmv_ltn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double
		y_0, y_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		y_0 = x[jj+0];
		y_1 = x[jj+1];
		y_0 *= pA[jj+0+lda*(jj+0)];
		y_0 += pA[jj+1+lda*(jj+0)] * y_1;
		y_1 *= pA[jj+1+lda*(jj+1)];
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			y_0 += pA[ii+0+lda*(jj+0)] * x[ii+0] + pA[ii+1+lda*(jj+0)] * x[ii+1];
			y_1 += pA[ii+0+lda*(jj+1)] * x[ii+0] + pA[ii+1+lda*(jj+1)] * x[ii+1];
			}
		for(; ii<m; ii++)
			{
			y_0 += pA[ii+lda*(jj+0)] * x[ii];
			y_1 += pA[ii+lda*(jj+1)] * x[ii];
			}
		z[jj+0] = y_0;
		z[jj+1] = y_1;
		}
	for(; jj<m; jj++)
		{
		y_0 = x[jj];
		y_0 *= pA[jj+lda*jj];
		for(ii=jj+1; ii<m; ii++)
			{
			y_0 += pA[ii+lda*jj] * x[ii];
			}
		z[jj] = y_0;
		}
	return;
	}



void dtrmv_unn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double
		y_0, y_1,
		x_0, x_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
#if 1 // y reg version
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		y_0 = x[jj+0];
		y_1 = x[jj+1];
		y_0 = pA[jj+0+lda*(jj+0)] * y_0;
		y_0 += pA[jj+0+lda*(jj+1)] * y_1;
		y_1 = pA[jj+1+lda*(jj+1)] * y_1;
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			y_0 += pA[jj+0+lda*(ii+0)] * x[ii+0] + pA[jj+0+lda*(ii+1)] * x[ii+1];
			y_1 += pA[jj+1+lda*(ii+0)] * x[ii+0] + pA[jj+1+lda*(ii+1)] * x[ii+1];
			}
		if(ii<m)
			{
			y_0 += pA[jj+0+lda*(ii+0)] * x[ii+0];
			y_1 += pA[jj+1+lda*(ii+0)] * x[ii+0];
			}
		z[jj+0] = y_0;
		z[jj+1] = y_1;
		}
	for(; jj<m; jj++)
		{
		y_0 = pA[jj+lda*jj] * x[jj];
		for(ii=jj+1; ii<m; ii++)
			{
			y_0 += pA[jj+lda*ii] * x[ii];
			}
		z[jj] = y_0;
		}
#else // x reg version
	if(x != z)
		{
		for(ii=0; ii<m; ii++)
			z[ii] = x[ii];
		}
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		x_0 = z[jj+0];
		x_1 = z[jj+1];
		ii = 0;
		for(; ii<jj-1; ii+=2)
			{
			z[ii+0] += pA[ii+0+lda*(jj+0)] * x_0 + pA[ii+0+lda*(jj+1)] * x_1;
			z[ii+1] += pA[ii+1+lda*(jj+0)] * x_0 + pA[ii+1+lda*(jj+1)] * x_1;
			}
//	XXX there is no clean-up loop, since jj+=2 !!!!!
//		for(; ii<jj; ii++)
//			{
//			z[ii+0] += pA[ii+0+lda*(jj+0)] * x_0 + pA[ii+0+lda*(jj+1)] * x_1;
//			}
		x_0 *= pA[jj+0+lda*(jj+0)];
		x_0 += pA[jj+0+lda*(jj+1)] * x_1;
		x_1 *= pA[jj+1+lda*(jj+1)];
		z[jj+0] = x_0;
		z[jj+1] = x_1;
		}
	for(; jj<m; jj++)
		{
		x_0 = z[jj];
		for(ii=0; ii<jj; ii++)
			{
			z[ii] += pA[ii+lda*jj] * x_0;
			}
		x_0 *= pA[jj+lda*jj];
		z[jj] = x_0;
		}
#endif
	return;
	}



void dtrmv_utn_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double
		y_0, y_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	if(m%2!=0)
		{
		jj = m-1;
		y_0 = pA[jj+lda*jj] * x[jj];
		for(ii=0; ii<jj; ii++)
			{
			y_0 += pA[ii+lda*jj] * x[ii];
			}
		z[jj] = y_0;
		m -= 1; // XXX
		}
	for(jj=m-2; jj>=0; jj-=2)
		{
		y_1 = pA[jj+1+lda*(jj+1)] * x[jj+1];
		y_1 += pA[jj+0+lda*(jj+1)] * x[jj+0];
		y_0 = pA[jj+0+lda*(jj+0)] * x[jj+0];
		for(ii=0; ii<jj-1; ii+=2)
			{
			y_0 += pA[ii+0+lda*(jj+0)] * x[ii+0] + pA[ii+1+lda*(jj+0)] * x[ii+1];
			y_1 += pA[ii+0+lda*(jj+1)] * x[ii+0] + pA[ii+1+lda*(jj+1)] * x[ii+1];
			}
//	XXX there is no clean-up loop !!!!!
//		if(ii<jj)
//			{
//			y_0 += pA[ii+lda*(jj+0)] * x[ii];
//			y_1 += pA[ii+lda*(jj+1)] * x[ii];
//			}
		z[jj+0] = y_0;
		z[jj+1] = y_1;
		}
	return;
	}



void dtrsv_lnn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj, j1;
	double
		y_0, y_1,
		x_0, x_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *dA = sA->dA;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / pA[ii+lda*ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0;
		}
#if 1 // y reg version
	ii = 0;
	for(; ii<n-1; ii+=2)
		{
		y_0 = x[ii+0];
		y_1 = x[ii+1];
		jj = 0;
		for(; jj<ii-1; jj+=2)
			{
			y_0 -= pA[ii+0+lda*(jj+0)] * x[jj+0] + pA[ii+0+lda*(jj+1)] * x[jj+1];
			y_1 -= pA[ii+1+lda*(jj+0)] * x[jj+0] + pA[ii+1+lda*(jj+1)] * x[jj+1];
			}
//	XXX there is no clean-up loop !!!!!
//		if(jj<ii)
//			{
//			y_0 -= pA[ii+0+lda*(jj+0)] * x[jj+0];
//			y_1 -= pA[ii+1+lda*(jj+0)] * x[jj+0];
//			}
		y_0 *= dA[ii+0];
		y_1 -= pA[ii+1+lda*(jj+0)] * y_0;
		y_1 *= dA[ii+1];
		z[ii+0] = y_0;
		z[ii+1] = y_1;
		}
	for(; ii<n; ii++)
		{
		y_0 = x[ii];
		for(jj=0; jj<ii; jj++)
			{
			y_0 -= pA[ii+lda*jj] * x[jj];
			}
		y_0 *= dA[ii];
		z[ii] = y_0;
		}
	for(; ii<m-1; ii+=2)
		{
		y_0 = x[ii+0];
		y_1 = x[ii+1];
		jj = 0;
		for(; jj<n-1; jj+=2)
			{
			y_0 -= pA[ii+0+lda*(jj+0)] * x[jj+0] + pA[ii+0+lda*(jj+1)] * x[jj+1];
			y_1 -= pA[ii+1+lda*(jj+0)] * x[jj+0] + pA[ii+1+lda*(jj+1)] * x[jj+1];
			}
		if(jj<n)
			{
			y_0 -= pA[ii+0+lda*(jj+0)] * x[jj+0];
			y_1 -= pA[ii+1+lda*(jj+0)] * x[jj+0];
			}
		z[ii+0] = y_0;
		z[ii+1] = y_1;
		}
	for(; ii<m; ii++)
		{
		y_0 = x[ii];
		for(jj=0; jj<n; jj++)
			{
			y_0 -= pA[ii+lda*jj] * x[jj];
			}
		z[ii] = y_0;
		}
#else // x reg version
	if(x != z)
		{
		for(ii=0; ii<m; ii++)
			z[ii] = x[ii];
		}
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		x_0 = dA[jj+0] * z[jj+0];
		x_1 = z[jj+1] - pA[jj+1+lda*(jj+0)] * x_0;
		x_1 = dA[jj+1] * x_1;
		z[jj+0] = x_0;
		z[jj+1] = x_1;
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			z[ii+0] -= pA[ii+0+lda*(jj+0)] * x_0 + pA[ii+0+lda*(jj+1)] * x_1;
			z[ii+1] -= pA[ii+1+lda*(jj+0)] * x_0 + pA[ii+1+lda*(jj+1)] * x_1;
			}
		for(; ii<m; ii++)
			{
			z[ii] -= pA[ii+lda*(jj+0)] * x_0 + pA[ii+lda*(jj+1)] * x_1;
			}
		}
	for(; jj<n; jj++)
		{
		x_0 = dA[jj] * z[jj];
		z[jj] = x_0;
		for(ii=jj+1; ii<m; ii++)
			{
			z[ii] -= pA[ii+lda*jj] * x_0;
			}
		}
#endif
	return;
	}



void dtrsv_ltn_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	int ii, jj;
	double
		y_0, y_1;
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *dA = sA->dA;
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / pA[ii+lda*ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0;
		}
	if(n%2!=0)
		{
		jj = n-1;
		y_0 = x[jj];
		for(ii=jj+1; ii<m; ii++)
			{
			y_0 -= pA[ii+lda*jj] * z[ii];
			}
		y_0 *= dA[jj];
		z[jj] = y_0;
		jj -= 2;
		}
	else
		{
		jj = n-2;
		}
	for(; jj>=0; jj-=2)
		{
		y_0 = x[jj+0];
		y_1 = x[jj+1];
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			y_0 -= pA[ii+0+lda*(jj+0)] * z[ii+0] + pA[ii+1+lda*(jj+0)] * z[ii+1];
			y_1 -= pA[ii+0+lda*(jj+1)] * z[ii+0] + pA[ii+1+lda*(jj+1)] * z[ii+1];
			}
		if(ii<m)
			{
			y_0 -= pA[ii+lda*(jj+0)] * z[ii];
			y_1 -= pA[ii+lda*(jj+1)] * z[ii];
			}
		y_1 *= dA[jj+1];
		y_0 -= pA[jj+1+lda*(jj+0)] * y_1;
		y_0 *= dA[jj+0];
		z[jj+0] = y_0;
		z[jj+1] = y_1;
		}
	return;
	}



#else

#error : wrong LA choice

#endif
