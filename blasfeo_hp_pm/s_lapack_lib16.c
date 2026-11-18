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
#include <math.h>

#include <blasfeo_common.h>
#include <blasfeo_s_aux.h>
#include <blasfeo_s_kernel.h>
#include <blasfeo_s_blasfeo_api.h>
#if defined(BLASFEO_REF_API)
#include <blasfeo_s_blasfeo_ref_api.h>
#endif



// spotrf
void blasfeo_hp_spotrf_l(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_spotrf_l(m, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_spotrf_l: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// spotrf
void blasfeo_hp_spotrf_l_mn(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_spotrf_l_mn(m, n, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_spotrf_l_mn: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// spotrf
void blasfeo_hp_spotrf_u(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_spotrf_u(m, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_spotrf_u: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	
	}



// dsyrk spotrf
void blasfeo_hp_ssyrk_spotrf_ln_mn(int m, int n, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_ssyrk_spotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_ssyrk_spotrf_ln_mn: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



void blasfeo_hp_ssyrk_spotrf_ln(int m, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_ssyrk_spotrf_ln: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// dgetrf no pivoting
void blasfeo_hp_sgetrf_np(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgetrf_np(m, n, sC, ci, cj, sD, di, dj);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgetf_np: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// dgetrf row pivoting
void blasfeo_hp_sgetrf_rp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgetrf_rp: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// sgetrf full (complete) pivoting, rank revealing
void blasfeo_hp_sgetrf_fp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank, float tol)
	{

	const int ps = 16;

	int cir = ci & (ps-1);
	int dir = di & (ps-1);

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps + (ci - cir)*sdc;
	float *pD = sD->pA + dj*ps + (di - dir)*sdd;
	float *dD = sD->dA;

	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	*rank = 0;

	if(m<=0 | n<=0)
		return;

	if(pC+cir!=pD+dir)
		blasfeo_sgecp(m, n, sC, ci, cj, sD, di, dj);

	int ii, jj, kk, ll, ip, jp, k0, k1, i0;
	float tmp, dmax, d_00_inv, d_ij, c_0, m_0;
	float l_v[16], m_v[16];
	float *pDi, *pDj, *pDk;

	int iimax = m<n ? m : n;
	// rows are indexed from the beginning of the first panel of D
	int me = m + dir;

	// the pivot is the element with largest absolute value, first in column-major order;
	// the trailing matrix is processed one panel of rows at a time, keeping track of the
	// largest absolute value in the panel; only if this is not smaller than the current
	// pivot, the panel is searched again for the column (jp) and later the row (ip)

	// pivot search
	dmax = 0.0;
	i0 = dir;
	jp = 0;
	for(k0=dir; k0<me; k0=k1)
		{
		k1 = (k0 | (ps-1)) + 1;
		k1 = k1<me ? k1 : me;
		pDk = pD + (k0 & ~(ps-1))*sdd;
		m_0 = 0.0;
		if(k1-k0==ps)
			{
			for(ll=0; ll<ps; ll++)
				{
				m_v[ll] = 0.0;
				}
			for(jj=0; jj<n; jj++)
				{
				for(ll=0; ll<ps; ll++)
					{
					c_0 = fabsf(pDk[ll+jj*ps]);
					m_v[ll] = m_v[ll]>c_0 ? m_v[ll] : c_0;
					}
				}
			for(ll=0; ll<ps; ll++)
				{
				m_0 = m_0>m_v[ll] ? m_0 : m_v[ll];
				}
			}
		else
			{
			for(jj=0; jj<n; jj++)
				{
				for(kk=k0; kk<k1; kk++)
					{
					c_0 = fabsf(pDk[(kk&(ps-1))+jj*ps]);
					m_0 = m_0>c_0 ? m_0 : c_0;
					}
				}
			}
		if(m_0>=dmax & m_0>0.0)
			{
			// search the panel for the first column containing the maximum
			for(jj=0; jj<n; jj++)
				{
				for(kk=k0; kk<k1; kk++)
					{
					if(fabsf(pDk[(kk&(ps-1))+jj*ps])==m_0)
						goto found_0;
					}
				}
			found_0:
			if(jj<n & (m_0>dmax | jj<jp))
				{
				dmax = m_0;
				i0 = k0;
				jp = jj;
				}
			}
		}

	// factorize
	for(ii=0; ii<iimax; ii++)
		{
		if(!(dmax>tol) | dmax==0.0)
			break;
		// search the panel for the row of the pivot
		ip = i0;
		k1 = (i0 | (ps-1)) + 1;
		k1 = k1<me ? k1 : me;
		pDk = pD + (i0 & ~(ps-1))*sdd + jp*ps;
		for(kk=i0; kk<k1; kk++)
			{
			if(fabsf(pDk[kk&(ps-1)])==dmax)
				{
				ip = kk;
				break;
				}
			}
		ipiv[ii] = ip - dir;
		jpiv[ii] = jp;
		// pointer to row ii
		kk = ii + dir;
		pDi = pD + (kk & ~(ps-1))*sdd + (kk&(ps-1));
		// swap rows
		if(ip!=kk)
			{
			pDj = pD + (ip & ~(ps-1))*sdd + (ip&(ps-1));
			for(jj=0; jj<n; jj++)
				{
				tmp = pDi[jj*ps];
				pDi[jj*ps] = pDj[jj*ps];
				pDj[jj*ps] = tmp;
				}
			}
		// swap columns
		if(jp!=ii)
			{
			for(k0=dir; k0<me; k0=k1)
				{
				k1 = (k0 | (ps-1)) + 1;
				k1 = k1<me ? k1 : me;
				pDk = pD + (k0 & ~(ps-1))*sdd;
				for(kk=k0; kk<k1; kk++)
					{
					tmp = pDk[(kk&(ps-1))+ii*ps];
					pDk[(kk&(ps-1))+ii*ps] = pDk[(kk&(ps-1))+jp*ps];
					pDk[(kk&(ps-1))+jp*ps] = tmp;
					}
				}
			}
		d_00_inv = 1.0/pDi[ii*ps];
		dD[ii] = d_00_inv;
		// scale column of L, rank-1 update of the trailing matrix and pivot search
		dmax = 0.0;
		i0 = ii+1+dir;
		jp = ii+1;
		for(k0=ii+1+dir; k0<me; k0=k1)
			{
			k1 = (k0 | (ps-1)) + 1;
			k1 = k1<me ? k1 : me;
			pDk = pD + (k0 & ~(ps-1))*sdd;
			m_0 = 0.0;
			if(k1-k0==ps)
				{
				for(ll=0; ll<ps; ll++)
					{
					l_v[ll] = pDk[ll+ii*ps] * d_00_inv;
					pDk[ll+ii*ps] = l_v[ll];
					m_v[ll] = 0.0;
					}
				for(jj=ii+1; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(ll=0; ll<ps; ll++)
						{
						c_0 = pDk[ll+jj*ps] - l_v[ll] * d_ij;
						pDk[ll+jj*ps] = c_0;
						c_0 = fabsf(c_0);
						m_v[ll] = m_v[ll]>c_0 ? m_v[ll] : c_0;
						}
					}
				for(ll=0; ll<ps; ll++)
					{
					m_0 = m_0>m_v[ll] ? m_0 : m_v[ll];
					}
				}
			else
				{
				for(kk=k0; kk<k1; kk++)
					{
					pDk[(kk&(ps-1))+ii*ps] *= d_00_inv;
					}
				for(jj=ii+1; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(kk=k0; kk<k1; kk++)
						{
						c_0 = pDk[(kk&(ps-1))+jj*ps] - pDk[(kk&(ps-1))+ii*ps] * d_ij;
						pDk[(kk&(ps-1))+jj*ps] = c_0;
						c_0 = fabsf(c_0);
						m_0 = m_0>c_0 ? m_0 : c_0;
						}
					}
				}
			if(m_0>=dmax & m_0>0.0)
				{
				// search the panel for the first column containing the maximum
				for(jj=ii+1; jj<n; jj++)
					{
					for(kk=k0; kk<k1; kk++)
						{
						if(fabsf(pDk[(kk&(ps-1))+jj*ps])==m_0)
							goto found_1;
						}
					}
				found_1:
				if(jj<n & (m_0>dmax | jj<jp))
					{
					dmax = m_0;
					i0 = k0;
					jp = jj;
					}
				}
			}
		}

	*rank = ii;

	// no pivoting on the rank deficient part
	for(; ii<iimax; ii++)
		{
		ipiv[ii] = ii;
		jpiv[ii] = ii;
		dD[ii] = 0.0;
		}

	return;

	}



int blasfeo_hp_sgeqrf_worksize(int m, int n)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgeqrf_worksize(m, n);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgeqrf_worksize: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



void blasfeo_hp_sgeqrf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgeqrf(m, n, sC, ci, cj, sD, di, dj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgeqrf: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



int blasfeo_hp_sgelqf_worksize(int m, int n)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_worksize(m, n);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_worksize: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



void blasfeo_hp_sgelqf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf(m, n, sC, ci, cj, sD, di, dj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



int blasfeo_hp_sorglq_worksize(int m, int n, int k)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sorglq_worksize(m, n, k);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sorglq_worksize: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



void blasfeo_hp_sorglq(int m, int n, int k, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sorglq(m, n, k, sC, ci, cj, sD, di, dj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sorglq: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// LQ factorization with positive diagonal elements
void blasfeo_hp_sgelqf_pd(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// LQ factorization with positive diagonal elements, array of matrices
// [L, A] <= lq( [L. A] )
// L lower triangular, of size (m)x(m)
// A full of size (m)x(n1)
void blasfeo_hp_sgelqf_pd_la(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd_la(m, n1, sD, di, dj, sA, ai, aj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd_la: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



// LQ factorization with positive diagonal elements, array of matrices
// [L, L, A] <= lq( [L. L, A] )
// L lower triangular, of size (m)x(m)
// A full of size (m)x(n1)
void blasfeo_hp_sgelqf_pd_lla(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sL, int li, int lj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd_lla(m, n1, sD, di, dj, sL, li, lj, sA, ai, aj, work);
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd_lla: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_spotrf_l(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_l(m, sC, ci, cj, sD, di, dj);
	}



void blasfeo_spotrf_l_mn(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_l_mn(m, n, sC, ci, cj, sD, di, dj);
	}



void blasfeo_spotrf_u(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_u(m, sC, ci, cj, sD, di, dj);
	}



void blasfeo_ssyrk_spotrf_ln_mn(int m, int n, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_ssyrk_spotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
	}



void blasfeo_ssyrk_spotrf_ln(int m, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgetrf_np(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgetrf_np(m, n, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgetrf_rp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv)
	{
	blasfeo_hp_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv);
	}



void blasfeo_sgetrf_fp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank, float tol)
	{
	blasfeo_hp_sgetrf_fp(m, n, sC, ci, cj, sD, di, dj, ipiv, jpiv, rank, tol);
	}



int blasfeo_sgeqrf_worksize(int m, int n)
	{
	return blasfeo_hp_sgeqrf_worksize(m, n);
	}



void blasfeo_sgeqrf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *v_work)
	{
	blasfeo_hp_sgeqrf(m, n, sC, ci, cj, sD, di, dj, v_work);
	}



int blasfeo_sgelqf_worksize(int m, int n)
	{
	return blasfeo_hp_sgelqf_worksize(m, n);
	}



void blasfeo_sgelqf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sgelqf(m, n, sC, ci, cj, sD, di, dj, work);
	}



int blasfeo_sorglq_worksize(int m, int n, int k)
	{
	return blasfeo_hp_sorglq_worksize(m, n, k);
	}



void blasfeo_sorglq(int m, int n, int k, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sorglq(m, n, k, sC, ci, cj, sD, di, dj, work);
	}



void blasfeo_sgelqf_pd(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sgelqf_pd(m, n, sC, ci, cj, sD, di, cj, work);
	}



void blasfeo_sgelqf_pd_la(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	blasfeo_hp_sgelqf_pd_la(m, n1, sD, di, dj, sA, ai, aj, work);
	}



void blasfeo_sgelqf_pd_lla(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sL, int li, int lj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	blasfeo_hp_sgelqf_pd_lla(m, n1, sD, di, dj, sL, li, lj, sA, ai, aj, work);
	}



#endif


