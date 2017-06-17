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



void strsv_ln_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y)
	{

	if(m<=0 || n<=0)
		return;
	
	// suppose m>=n
	if(m<n)
		m = n;

	const int bs = 4;

	float alpha = -1.0;
	float beta = 1.0;

	int i;

	if(x!=y)
		{
		for(i=0; i<m; i++)
			y[i] = x[i];
		}
	
	i = 0;
	for( ; i<n-3; i+=4)
		{
		kernel_strsv_ln_inv_4_lib4(i, &pA[i*sda], &inv_diag_A[i], y, &y[i], &y[i]);
		}
	if(i<n)
		{
		kernel_strsv_ln_inv_4_vs_lib4(i, &pA[i*sda], &inv_diag_A[i], y, &y[i], &y[i], m-i, n-i);
		i+=4;
		}
	for( ; i<m-3; i+=4)
		{
		kernel_sgemv_n_4_lib4(n, &alpha, &pA[i*sda], y, &beta, &y[i], &y[i]);
		}
	if(i<m)
		{
		kernel_sgemv_n_4_gen_lib4(n, &alpha, &pA[i*sda], y, &beta, &y[i], &y[i], 0, m-i);
		i+=4;
		}

	}



void strsv_lt_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y)
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
		kernel_strsv_lt_inv_1_lib4(m-n+i+1, &pA[n/bs*bs*sda+(n-i-1)*bs], sda, &inv_diag_A[n-i-1], &y[n-i-1], &y[n-i-1], &y[n-i-1]);
		i++;
		}
	else if(n%4==2)
		{
		kernel_strsv_lt_inv_2_lib4(m-n+i+2, &pA[n/bs*bs*sda+(n-i-2)*bs], sda, &inv_diag_A[n-i-2], &y[n-i-2], &y[n-i-2], &y[n-i-2]);
		i+=2;
		}
	else if(n%4==3)
		{
		kernel_strsv_lt_inv_3_lib4(m-n+i+3, &pA[n/bs*bs*sda+(n-i-3)*bs], sda, &inv_diag_A[n-i-3], &y[n-i-3], &y[n-i-3], &y[n-i-3]);
		i+=3;
		}
	for(; i<n-3; i+=4)
		{
		kernel_strsv_lt_inv_4_lib4(m-n+i+4, &pA[(n-i-4)/bs*bs*sda+(n-i-4)*bs], sda, &inv_diag_A[n-i-4], &y[n-i-4], &y[n-i-4], &y[n-i-4]);
		}

	}



void sgemv_nt_lib(int m, int n, float alpha_n, float alpha_t, float *pA, int sda, float *x_n, float *x_t, float beta_n, float beta_t, float *y_n, float *y_t, float *z_n, float *z_t)
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
		kernel_sgemv_nt_4_lib4(m, &alpha_n, &alpha_t, pA+ii*bs, sda, x_n+ii, x_t, &beta_t, y_t+ii, z_n, z_t+ii);
		}
	if(ii<n)
		{
		kernel_sgemv_nt_4_vs_lib4(m, &alpha_n, &alpha_t, pA+ii*bs, sda, x_n+ii, x_t, &beta_t, y_t+ii, z_n, z_t+ii, n-ii);
		}
	
	return;

	}


	
#if defined(LA_HIGH_PERFORMANCE)



void sgemv_n_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(m<0)
		return;

	const int bs = 4;

	int i;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;

	i = 0;
	// clean up at the beginning
	if(ai%bs!=0)
		{
		kernel_sgemv_n_4_gen_lib4(n, &alpha, pA, x, &beta, y-ai%bs, z-ai%bs, ai%bs, m+ai%bs);
		pA += bs*sda;
		y += 4 - ai%bs;
		z += 4 - ai%bs;
		m -= 4 - ai%bs;
		}
	// main loop
	for( ; i<m-3; i+=4)
		{
		kernel_sgemv_n_4_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i]);
		}
	if(i<m)
		{
		kernel_sgemv_n_4_vs_lib4(n, &alpha, &pA[i*sda], x, &beta, &y[i], &z[i], m-i);
		}
		
	return;

	}



void sgemv_t_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(n<=0)
		return;
	
	const int bs = 4;

	int i;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;

	if(ai%bs==0)
		{
		i = 0;
		for( ; i<n-3; i+=4)
			{
			kernel_sgemv_t_4_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i]);
			}
		if(i<n)
			{
			kernel_sgemv_t_4_vs_lib4(m, &alpha, &pA[i*bs], sda, x, &beta, &y[i], &z[i], n-i);
			}
		}
	else // TODO kernel 8
		{
		i = 0;
		for( ; i<n; i+=4)
			{
			kernel_sgemv_t_4_gen_lib4(m, &alpha, ai%bs, &pA[i*bs], sda, x, &beta, &y[i], &z[i], n-i);
			}
		}
	
	return;

	}



void sgemv_nt_libstr(int m, int n, float alpha_n, float alpha_t, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx_n, int xi_n, struct s_strvec *sx_t, int xi_t, float beta_n, float beta_t, struct s_strvec *sy_n, int yi_n, struct s_strvec *sy_t, int yi_t, struct s_strvec *sz_n, int zi_n, struct s_strvec *sz_t, int zi_t)
	{
	if(ai!=0 | xi_n%4!=0 | xi_t%4!=0)
		{
		printf("\nsgemv_nt_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *x_n = sx_n->pa + xi_n;
	float *x_t = sx_t->pa + xi_t;
	float *y_n = sy_n->pa + yi_n;
	float *y_t = sy_t->pa + yi_t;
	float *z_n = sz_n->pa + zi_n;
	float *z_t = sz_t->pa + zi_t;
	sgemv_nt_lib(m, n, alpha_n, alpha_t, pA, sda, x_n, x_t, beta_n, beta_t, y_n, y_t, z_n, z_t);
	return;
	}



void ssymv_l_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, float beta, struct s_strvec *sy, int yi, struct s_strvec *sz, int zi)
	{

	if(m<=0 | n<=0)
		return;
	
	const int bs = 4;

	int ii, n1;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	float *z = sz->pa + zi;

//	dsymv_l_lib(m, n, alpha, pA, sda, x, beta, y, z);

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
	
	// clean up at the beginning
	if(ai%bs!=0) // 1, 2, 3
		{
		n1 = 4-ai%bs;
		kernel_ssymv_l_4_gen_lib4(m, &alpha, ai%bs, &pA[0], sda, &x[0], &x[0], &z[0], &z[0], n<n1 ? n : n1);
		pA += n1 + n1*bs + (sda-1)*bs;
		x += n1;
		z += n1;
		m -= n1;
		n -= n1;
		}
	// main loop
	ii = 0;
	for(; ii<n-3; ii+=4)
		{
		kernel_ssymv_l_4_lib4(m-ii, &alpha, &pA[ii*bs+ii*sda], sda, &x[ii], &x[ii], &z[ii], &z[ii]);
		}
	// clean up at the end
	if(ii<n)
		{
		kernel_ssymv_l_4_gen_lib4(m-ii, &alpha, 0, &pA[ii*bs+ii*sda], sda, &x[ii], &x[ii], &z[ii], &z[ii], n-ii);
		}
	
	return;
	}



// m >= n
void strmv_lnn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;

	if(m-n>0)
		sgemv_n_libstr(m-n, n, 1.0, sA, ai+n, aj, sx, xi, 0.0, sz, zi+n, sz, zi+n);

	float *pA2 = pA;
	float *z2 = z;
	int m2 = n;
	int n2 = 0;
	float *pA3, *x3;

	float alpha = 1.0;
	float beta = 1.0;

	float zt[4];

	int ii, jj, jj_end;

	ii = 0;

	if(ai%4!=0)
		{
		pA2 += sda*bs - ai%bs;
		z2 += bs-ai%bs;
		m2 -= bs-ai%bs;
		n2 += bs-ai%bs;
		}
	
	pA2 += m2/bs*bs*sda;
	z2 += m2/bs*bs;
	n2 += m2/bs*bs;

	if(m2%bs!=0)
		{
		//
		pA3 = pA2 + bs*n2;
		x3 = x + n2;
		zt[3] = pA3[3+bs*0]*x3[0] + pA3[3+bs*1]*x3[1] + pA3[3+bs*2]*x3[2] + pA3[3+bs*3]*x3[3];
		zt[2] = pA3[2+bs*0]*x3[0] + pA3[2+bs*1]*x3[1] + pA3[2+bs*2]*x3[2];
		zt[1] = pA3[1+bs*0]*x3[0] + pA3[1+bs*1]*x3[1];
		zt[0] = pA3[0+bs*0]*x3[0];
		kernel_sgemv_n_4_lib4(n2, &alpha, pA2, x, &beta, zt, zt);
		for(jj=0; jj<m2%bs; jj++)
			z2[jj] = zt[jj];
		}
	for(; ii<m2-3; ii+=4)
		{
		pA2 -= bs*sda;
		z2 -= 4;
		n2 -= 4;
		pA3 = pA2 + bs*n2;
		x3 = x + n2;
		z2[3] = pA3[3+bs*0]*x3[0] + pA3[3+bs*1]*x3[1] + pA3[3+bs*2]*x3[2] + pA3[3+bs*3]*x3[3];
		z2[2] = pA3[2+bs*0]*x3[0] + pA3[2+bs*1]*x3[1] + pA3[2+bs*2]*x3[2];
		z2[1] = pA3[1+bs*0]*x3[0] + pA3[1+bs*1]*x3[1];
		z2[0] = pA3[0+bs*0]*x3[0];
		kernel_sgemv_n_4_lib4(n2, &alpha, pA2, x, &beta, z2, z2);
		}
	if(ai%4!=0)
		{
		if(ai%bs==1)
			{
			zt[2] = pA[2+bs*0]*x[0] + pA[2+bs*1]*x[1] + pA[2+bs*2]*x[2];
			zt[1] = pA[1+bs*0]*x[0] + pA[1+bs*1]*x[1];
			zt[0] = pA[0+bs*0]*x[0];
			jj_end = 4-ai%bs<n ? 4-ai%bs : n;
			for(jj=0; jj<jj_end; jj++)
				z[jj] = zt[jj];
			}
		else if(ai%bs==2)
			{
			zt[1] = pA[1+bs*0]*x[0] + pA[1+bs*1]*x[1];
			zt[0] = pA[0+bs*0]*x[0];
			jj_end = 4-ai%bs<n ? 4-ai%bs : n;
			for(jj=0; jj<jj_end; jj++)
				z[jj] = zt[jj];
			}
		else // if (ai%bs==3)
			{
			z[0] = pA[0+bs*0]*x[0];
			}
		}

	return;

	}



// m >= n
void strmv_ltn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;

	float xt[4];
	float zt[4];

	float alpha = 1.0;
	float beta = 1.0;

	int ii, jj, ll, ll_max;

	jj = 0;

	if(ai%bs!=0)
		{

		if(ai%bs==1)
			{
			ll_max = m-jj<3 ? m-jj : 3;
			for(ll=0; ll<ll_max; ll++)
				xt[ll] = x[ll];
			for(; ll<3; ll++)
				xt[ll] = 0.0;
			zt[0] = pA[0+bs*0]*xt[0] + pA[1+bs*0]*xt[1] + pA[2+bs*0]*xt[2];
			zt[1] = pA[1+bs*1]*xt[1] + pA[2+bs*1]*xt[2];
			zt[2] = pA[2+bs*2]*xt[2];
			pA += bs*sda - 1;
			x += 3;
			kernel_sgemv_t_4_lib4(m-3-jj, &alpha, pA, sda, x, &beta, zt, zt);
			ll_max = n-jj<3 ? n-jj : 3;
			for(ll=0; ll<ll_max; ll++)
				z[ll] = zt[ll];
			pA += bs*3;
			z += 3;
			jj += 3;
			}
		else if(ai%bs==2)
			{
			ll_max = m-jj<2 ? m-jj : 2;
			for(ll=0; ll<ll_max; ll++)
				xt[ll] = x[ll];
			for(; ll<2; ll++)
				xt[ll] = 0.0;
			zt[0] = pA[0+bs*0]*xt[0] + pA[1+bs*0]*xt[1];
			zt[1] = pA[1+bs*1]*xt[1];
			pA += bs*sda - 2;
			x += 2;
			kernel_sgemv_t_4_lib4(m-2-jj, &alpha, pA, sda, x, &beta, zt, zt);
			ll_max = n-jj<2 ? n-jj : 2;
			for(ll=0; ll<ll_max; ll++)
				z[ll] = zt[ll];
			pA += bs*2;
			z += 2;
			jj += 2;
			}
		else // if(ai%bs==3)
			{
			ll_max = m-jj<1 ? m-jj : 1;
			for(ll=0; ll<ll_max; ll++)
				xt[ll] = x[ll];
			for(; ll<1; ll++)
				xt[ll] = 0.0;
			zt[0] = pA[0+bs*0]*xt[0];
			pA += bs*sda - 3;
			x += 1;
			kernel_sgemv_t_4_lib4(m-1-jj, &alpha, pA, sda, x, &beta, zt, zt);
			ll_max = n-jj<1 ? n-jj : 1;
			for(ll=0; ll<ll_max; ll++)
				z[ll] = zt[ll];
			pA += bs*1;
			z += 1;
			jj += 1;
			}

		}
	
	for(; jj<n-3; jj+=4)
		{
		zt[0] = pA[0+bs*0]*x[0] + pA[1+bs*0]*x[1] + pA[2+bs*0]*x[2] + pA[3+bs*0]*x[3];
		zt[1] = pA[1+bs*1]*x[1] + pA[2+bs*1]*x[2] + pA[3+bs*1]*x[3];
		zt[2] = pA[2+bs*2]*x[2] + pA[3+bs*2]*x[3];
		zt[3] = pA[3+bs*3]*x[3];
		pA += bs*sda;
		x += 4;
		kernel_sgemv_t_4_lib4(m-4-jj, &alpha, pA, sda, x, &beta, zt, z);
		pA += bs*4;
		z += 4;
		}
	if(jj<n)
		{
		ll_max = m-jj<4 ? m-jj : 4;
		for(ll=0; ll<ll_max; ll++)
			xt[ll] = x[ll];
		for(; ll<4; ll++)
			xt[ll] = 0.0;
		zt[0] = pA[0+bs*0]*xt[0] + pA[1+bs*0]*xt[1] + pA[2+bs*0]*xt[2] + pA[3+bs*0]*xt[3];
		zt[1] = pA[1+bs*1]*xt[1] + pA[2+bs*1]*xt[2] + pA[3+bs*1]*xt[3];
		zt[2] = pA[2+bs*2]*xt[2] + pA[3+bs*2]*xt[3];
		zt[3] = pA[3+bs*3]*xt[3];
		pA += bs*sda;
		x += 4;
		kernel_sgemv_t_4_lib4(m-4-jj, &alpha, pA, sda, x, &beta, zt, zt);
		for(ll=0; ll<n-jj; ll++)
			z[ll] = zt[ll];
//		pA += bs*4;
//		z += 4;
		}

	return;

	}



void strmv_unn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{

	if(m<=0)
		return;

	if(ai!=0)
		{
		printf("\ndtrmv_unn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}

	const int bs = 4;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;

	int i;
	
	i=0;
	for(; i<m-3; i+=4)
		{
		kernel_strmv_un_4_lib4(m-i, pA, x, z);
		pA += 4*sda+4*bs;
		x  += 4;
		z  += 4;
		}
	if(m>i)
		{
		if(m-i==1)
			{
			z[0] = pA[0+bs*0]*x[0];
			}
		else if(m-i==2)
			{
			z[0] = pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1];
			z[1] = pA[1+bs*1]*x[1];
			}
		else // if(m-i==3)
			{
			z[0] = pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2];
			z[1] = pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2];
			z[2] = pA[2+bs*2]*x[2];
			}
		}

	return;

	}



void strmv_utn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{

	if(m<=0)
		return;

	if(ai!=0)
		{
		printf("\nstrmv_utn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}

	const int bs = 4;

	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;

	int ii, idx;
	
	float *ptrA;
	
	ii=0;
	idx = m/bs*bs;
	if(m%bs!=0)
		{
		kernel_strmv_ut_4_vs_lib4(m, pA+idx*bs, sda, x, z+idx, m%bs);
		ii += m%bs;
		}
	idx -= 4;
	for(; ii<m; ii+=4)
		{
		kernel_strmv_ut_4_lib4(idx+4, pA+idx*bs, sda, x, z+idx);
		idx -= 4;
		}

	return;

	}



void strsv_lnn_mn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0 | n==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_lnn_mn_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** strsv_lnn_mn_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_lnn_mn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_lnn_mn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_lnn_mn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_lnn_mn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_lnn_mn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** strsv_lnn_mn_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_lnn_mn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_lnn_mn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	if(ai!=0 | xi%4!=0)
		{
		printf("\nstrsv_lnn_mn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *dA = sA->dA;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
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
	strsv_ln_inv_lib(m, n, pA, sda, dA, x, z);
	return;
	}



void strsv_ltn_mn_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_ltn_mn_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** strsv_ltn_mn_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_ltn_mn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_ltn_mn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_ltn_mn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_ltn_mn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_ltn_mn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** strsv_ltn_mn_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_ltn_mn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_ltn_mn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	if(ai!=0 | xi%4!=0)
		{
		printf("\nstrsv_ltn_mn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *dA = sA->dA;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
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
	strsv_lt_inv_lib(m, n, pA, sda, dA, x, z);
	return;
	}



void strsv_lnn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_lnn_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_lnn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_lnn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_lnn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_lnn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_lnn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_lnn_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_lnn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_lnn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	if(ai!=0 | xi%4!=0)
		{
		printf("\nstrsv_lnn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *dA = sA->dA;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(m, 1.0, ai, pA, sda, dA);
			for(ii=0; ii<m; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(m, 1.0, ai, pA, sda, dA);
		for(ii=0; ii<m; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	strsv_ln_inv_lib(m, m, pA, sda, dA, x, z);
	return;
	}



void strsv_lnu_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_lnu_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_lnu_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_lnu_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_lnu_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_lnu_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_lnu_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_lnu_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_lnu_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_lnu_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	printf("\n***** strsv_lnu_libstr : feature not implemented yet *****\n");
	exit(1);
	}



void strsv_ltn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_ltn_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_ltn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_ltn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_ltn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_ltn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_ltn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_ltn_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_ltn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_ltn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	if(ai!=0 | xi%4!=0)
		{
		printf("\nstrsv_ltn_libstr: feature not implemented yet: ai=%d\n", ai);
		exit(1);
		}
	const int bs = 4;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs; // TODO ai
	float *dA = sA->dA;
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
	int ii;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA!=1)
			{
			sdiaex_lib(m, 1.0, ai, pA, sda, dA);
			for(ii=0; ii<m; ii++)
				dA[ii] = 1.0 / dA[ii];
			sA->use_dA = 1;
			}
		}
	else
		{
		sdiaex_lib(m, 1.0, ai, pA, sda, dA);
		for(ii=0; ii<m; ii++)
			dA[ii] = 1.0 / dA[ii];
		sA->use_dA = 0;
		}
	strsv_lt_inv_lib(m, m, pA, sda, dA, x, z);
	return;
	}



void strsv_ltu_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_ltu_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_ltu_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_ltu_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_ltu_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_ltu_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_ltu_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_ltu_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_ltu_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_ltu_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	printf("\n***** strsv_ltu_libstr : feature not implemented yet *****\n");
	exit(1);
	}



void strsv_unn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_unn_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_unn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_unn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_unn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_unn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_unn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_unn_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_unn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_unn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	printf("\n***** strsv_unn_libstr : feature not implemented yet *****\n");
	exit(1);
	}



void strsv_utn_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	if(m==0)
		return;
#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** strsv_utn_libstr : m<0 : %d<0 *****\n", m);
	// non-negative offset
	if(ai<0) printf("\n****** strsv_utn_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** strsv_utn_libstr : aj<0 : %d<0 *****\n", aj);
	if(xi<0) printf("\n****** strsv_utn_libstr : xi<0 : %d<0 *****\n", xi);
	if(zi<0) printf("\n****** strsv_utn_libstr : zi<0 : %d<0 *****\n", zi);
	// inside matrix
	// A: m x k
	if(ai+m > sA->m) printf("\n***** strsv_utn_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+m > sA->n) printf("\n***** strsv_utn_libstr : aj+m > col(A) : %d+%d > %d *****\n", aj, m, sA->n);
	// x: m
	if(xi+m > sx->m) printf("\n***** strsv_utn_libstr : xi+m > size(x) : %d+%d > %d *****\n", xi, m, sx->m);
	// z: m
	if(zi+m > sz->m) printf("\n***** strsv_utn_libstr : zi+m > size(z) : %d+%d > %d *****\n", zi, m, sz->m);
#endif
	printf("\n***** strsv_utn_libstr : feature not implemented yet *****\n");
	exit(1);
	}



#else

#error : wrong LA choice

#endif
