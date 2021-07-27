/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
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

#include <math.h>
#include <stdio.h>

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>



// unblocked algorithm
void kernel_dgelqf_vs_lib8(int m, int n, int k, int offD, double *pD, int sdd, double *dD)
	{
	if(m<=0 | n<=0)
		return;
	int ii, jj, kk, ll, imax, jmax, jmax0, kmax, kmax0;
	const int ps = 8;
	imax = k;//m<n ? m : n;
	double alpha, beta, tmp;
	double w00, w01,
		   w10, w11,
		   w20, w21,
		   w30, w31;
	__m512d
		_a0, _b0, _t0, _w0, _w1;
	double *pC00, *pC10, *pC10a, *pC20, *pC20a, *pC01, *pC11;
	double pT[4];
	int ldt = 2;
	double *pD0 = pD-offD;
	ii = 0;
#if 1 // rank 2
	for(; ii<imax-1; ii+=2)
		{
		// first row
		pC00 = &pD0[((offD+ii)&(ps-1))+((offD+ii)-((offD+ii)&(ps-1)))*sdd+ii*ps];
		beta = 0.0;
		for(jj=1; jj<n-ii; jj++)
			{
			tmp = pC00[0+ps*jj];
			beta += tmp*tmp;
			}
		if(beta==0.0)
			{
			dD[ii] = 0.0;
			}
		else
			{
			alpha = pC00[0];
			beta += alpha*alpha;
			beta = sqrt(beta);
			if(alpha>0)
				beta = -beta;
			dD[ii] = (beta-alpha) / beta;
			tmp = 1.0 / (alpha-beta);
			pC00[0] = beta;
			for(jj=1; jj<n-ii; jj++)
				pC00[0+ps*jj] *= tmp;
			}
		pC10 = &pD0[((offD+ii+1)&(ps-1))+((offD+ii+1)-((offD+ii+1)&(ps-1)))*sdd+ii*ps];
		kmax = n-ii;
		w00 = pC10[0+ps*0]; // pC00[0+ps*0] = 1.0
		for(kk=1; kk<kmax; kk++)
			{
			w00 += pC10[0+ps*kk] * pC00[0+ps*kk];
			}
		w00 = - w00*dD[ii];
		pC10[0+ps*0] += w00; // pC00[0+ps*0] = 1.0
		for(kk=1; kk<kmax; kk++)
			{
			pC10[0+ps*kk] += w00 * pC00[0+ps*kk];
			}
		// second row
		pC11 = pC10+ps*1;
		beta = 0.0;
		for(jj=1; jj<n-(ii+1); jj++)
			{
			tmp = pC11[0+ps*jj];
			beta += tmp*tmp;
			}
		if(beta==0.0)
			{
			dD[(ii+1)] = 0.0;
			}
		else
			{
			alpha = pC11[0+ps*0];
			beta += alpha*alpha;
			beta = sqrt(beta);
			if(alpha>0)
				beta = -beta;
			dD[(ii+1)] = (beta-alpha) / beta;
			tmp = 1.0 / (alpha-beta);
			pC11[0+ps*0] = beta;
			for(jj=1; jj<n-(ii+1); jj++)
				pC11[0+ps*jj] *= tmp;
			}
		// compute T
		kmax = n-ii;
		tmp = 1.0*0.0 + pC00[0+ps*1]*1.0;
		for(kk=2; kk<kmax; kk++)
			tmp += pC00[0+ps*kk]*pC10[0+ps*kk];
		pT[0+ldt*0] = - dD[ii+0];
		pT[0+ldt*1] = + dD[ii+1] * tmp * dD[ii+0];
		pT[1+ldt*1] = - dD[ii+1];
		// downgrade
		kmax = n-ii;
		jmax = m-ii-2;
		jmax0 = (ps-((ii+2+offD)&(ps-1)))&(ps-1);
		jmax0 = jmax<jmax0 ? jmax : jmax0;
		jj = 0;
		pC20a = &pD0[((offD+ii+2)&(ps-1))+((offD+ii+2)-((offD+ii+2)&(ps-1)))*sdd+ii*ps];
		pC20 = pC20a;
		if(jmax0>0)
			{
			for( ; jj<jmax0; jj++)
				{
				w00 = pC20[0+ps*0]*1.0 + pC20[0+ps*1]*pC00[0+ps*1];
				w01 = pC20[0+ps*0]*0.0 + pC20[0+ps*1]*1.0;
				for(kk=2; kk<kmax; kk++)
					{
					w00 += pC20[0+ps*kk]*pC00[0+ps*kk];
					w01 += pC20[0+ps*kk]*pC10[0+ps*kk];
					}
				w01 = w00*pT[0+ldt*1] + w01*pT[1+ldt*1];
				w00 = w00*pT[0+ldt*0];
				pC20[0+ps*0] += w00*1.0          + w01*0.0;
				pC20[0+ps*1] += w00*pC00[0+ps*1] + w01*1.0;
				for(kk=2; kk<kmax; kk++)
					{
					pC20[0+ps*kk] += w00*pC00[0+ps*kk] + w01*pC10[0+ps*kk];
					}
				pC20 += 1;
				}
			pC20 += -ps+ps*sdd;
			}
		for( ; jj<jmax-7; jj+=8)
			{
			//
			_w0 = _mm512_load_pd( &pC20[0+ps*0] );
			_a0 = _mm512_load_pd( &pC20[0+ps*1] );
			_b0 = _mm512_set1_pd( pC00[0+ps*1] );
			_t0 = _mm512_mul_pd( _a0, _b0 );
			_w0 = _mm512_add_pd( _w0, _t0 );
			_w1 = _mm512_load_pd( &pC20[0+ps*1] );
			for(kk=2; kk<kmax; kk++)
				{
				_a0 = _mm512_load_pd( &pC20[0+ps*kk] );
				_b0 = _mm512_set1_pd( pC00[0+ps*kk] );
				_t0 = _mm512_mul_pd( _a0, _b0 );
				_w0 = _mm512_add_pd( _w0, _t0 );
				_b0 = _mm512_set1_pd( pC10[0+ps*kk] );
				_t0 = _mm512_mul_pd( _a0, _b0 );
				_w1 = _mm512_add_pd( _w1, _t0 );
				}
			//
			_b0 = _mm512_set1_pd( pT[1+ldt*1] );
			_w1 = _mm512_mul_pd( _w1, _b0 );
			_b0 = _mm512_set1_pd( pT[0+ldt*1] );
			_t0 = _mm512_mul_pd( _w0, _b0 );
			_w1 = _mm512_add_pd( _w1, _t0 );
			_b0 = _mm512_set1_pd( pT[0+ldt*0] );
			_w0 = _mm512_mul_pd( _w0, _b0 );
			//
			_a0 = _mm512_load_pd( &pC20[0+ps*0] );
			_a0 = _mm512_add_pd( _a0, _w0 );
			_mm512_store_pd( &pC20[0+ps*0], _a0 );
			_a0 = _mm512_load_pd( &pC20[0+ps*1] );
			_b0 = _mm512_set1_pd( pC00[0+ps*1] );
			_t0 = _mm512_mul_pd( _w0, _b0 );
			_a0 = _mm512_add_pd( _a0, _t0 );
			_a0 = _mm512_add_pd( _a0, _w1 );
			_mm512_store_pd( &pC20[0+ps*1], _a0 );
			for(kk=2; kk<kmax; kk++)
				{
				_a0 = _mm512_load_pd( &pC20[0+ps*kk] );
				_b0 = _mm512_set1_pd( pC00[0+ps*kk] );
				_t0 = _mm512_mul_pd( _w0, _b0 );
				_a0 = _mm512_add_pd( _a0, _t0 );
				_b0 = _mm512_set1_pd( pC10[0+ps*kk] );
				_t0 = _mm512_mul_pd( _w1, _b0 );
				_a0 = _mm512_add_pd( _a0, _t0 );
				_mm512_store_pd( &pC20[0+ps*kk], _a0 );
				}
			pC20 += ps*sdd;
			}
		for(ll=0; ll<jmax-jj; ll++)
			{
			w00 = pC20[0+ps*0]*1.0 + pC20[0+ps*1]*pC00[0+ps*1];
			w01 = pC20[0+ps*0]*0.0 + pC20[0+ps*1]*1.0;
			for(kk=2; kk<kmax; kk++)
				{
				w00 += pC20[0+ps*kk]*pC00[0+ps*kk];
				w01 += pC20[0+ps*kk]*pC10[0+ps*kk];
				}
			w01 = w00*pT[0+ldt*1] + w01*pT[1+ldt*1];
			w00 = w00*pT[0+ldt*0];
			pC20[0+ps*0] += w00*1.0          + w01*0.0;
			pC20[0+ps*1] += w00*pC00[0+ps*1] + w01*1.0;
			for(kk=2; kk<kmax; kk++)
				{
				pC20[0+ps*kk] += w00*pC00[0+ps*kk] + w01*pC10[0+ps*kk];
				}
			pC20 += 1;
			}
		}
#endif
	for(; ii<imax; ii++)
		{
		pC00 = &pD0[((offD+ii)&(ps-1))+((offD+ii)-((offD+ii)&(ps-1)))*sdd+ii*ps];
		beta = 0.0;
		for(jj=1; jj<n-ii; jj++)
			{
			tmp = pC00[0+ps*jj];
			beta += tmp*tmp;
			}
		if(beta==0.0)
			{
			dD[ii] = 0.0;
			}
		else
			{
			alpha = pC00[0];
			beta += alpha*alpha;
			beta = sqrt(beta);
			if(alpha>0)
				beta = -beta;
			dD[ii] = (beta-alpha) / beta;
			tmp = 1.0 / (alpha-beta);
			pC00[0] = beta;
			for(jj=1; jj<n-ii; jj++)
				pC00[0+ps*jj] *= tmp;
			}
		if(ii<n)
			{
			// compute T
			pT[0+ldt*0] = - dD[ii+0];
			// downgrade
			kmax = n-ii;
			jmax = m-ii-1;
			jmax0 = (ps-((ii+1+offD)&(ps-1)))&(ps-1);
			jmax0 = jmax<jmax0 ? jmax : jmax0;
			jj = 0;
			pC10a = &pD0[((offD+ii+1)&(ps-1))+((offD+ii+1)-((offD+ii+1)&(ps-1)))*sdd+ii*ps];
			pC10 = pC10a;
			if(jmax0>0)
				{
				for( ; jj<jmax0; jj++)
					{
					w00 = pC10[0+ps*0];
					for(kk=1; kk<kmax; kk++)
						{
						w00 += pC10[0+ps*kk] * pC00[0+ps*kk];
						}
					w00 = w00*pT[0+ldt*0];
					pC10[0+ps*0] += w00;
					for(kk=1; kk<kmax; kk++)
						{
						pC10[0+ps*kk] += w00 * pC00[0+ps*kk];
						}
					pC10 += 1;
					}
				pC10 += -ps+ps*sdd;
				}
			for( ; jj<jmax-7; jj+=8)
				{
				//
				_w0 = _mm512_load_pd( &pC10[0+ps*0] );
				for(kk=1; kk<kmax; kk++)
					{
					_a0 = _mm512_load_pd( &pC10[0+ps*kk] );
					_b0 = _mm512_set1_pd( pC00[0+ps*kk] );
					_t0 = _mm512_mul_pd( _a0, _b0 );
					_w0 = _mm512_add_pd( _w0, _t0 );
					}
				//
				_b0 = _mm512_set1_pd( pT[0+ldt*0] );
				_w0 = _mm512_mul_pd( _w0, _b0 );
				//
				_a0 = _mm512_load_pd( &pC10[0+ps*0] );
				_a0 = _mm512_add_pd( _a0, _w0 );
				_mm512_store_pd( &pC10[0+ps*0], _a0 );
				for(kk=1; kk<kmax; kk++)
					{
					_a0 = _mm512_load_pd( &pC10[0+ps*kk] );
					_b0 = _mm512_set1_pd( pC00[0+ps*kk] );
					_t0 = _mm512_mul_pd( _w0, _b0 );
					_a0 = _mm512_add_pd( _a0, _t0 );
					_mm512_store_pd( &pC10[0+ps*kk], _a0 );
					}
				pC10 += ps*sdd;
				}
			for(ll=0; ll<jmax-jj; ll++)
				{
				w00 = pC10[0+ps*0];
				for(kk=1; kk<kmax; kk++)
					{
					w00 += pC10[0+ps*kk] * pC00[0+ps*kk];
					}
				w00 = w00*pT[0+ldt*0];
				pC10[0+ps*0] += w00;
				for(kk=1; kk<kmax; kk++)
					{
					pC10[0+ps*kk] += w00 * pC00[0+ps*kk];
					}
				pC10 += 1;
				}
			}
		}
	return;
	}




