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



#if defined(LA_REFERENCE) | defined(LA_EXTERNAL_BLAS_WRAPPER) | defined(TESTING_MODE)



// dgemm with A diagonal matrix (stored as strvec)
void GEMM_L_DIAG_LIBSTR(int m, int n, REAL alpha, struct STRVEC *sA, int ai, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *dA = sA->pa + ai;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL a0, a1;
	if(beta==0.0)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			a0 = alpha * dA[ii+0];
			a1 = alpha * dA[ii+1];
			for(jj=0; jj<n; jj++)
				{
				pD[ii+0+ldd*jj] = a0 * pB[ii+0+ldb*jj];
				pD[ii+1+ldd*jj] = a1 * pB[ii+1+ldb*jj];
				}
			}
		for(; ii<m; ii++)
			{
			a0 = alpha * dA[ii];
			for(jj=0; jj<n; jj++)
				{
				pD[ii+0+ldd*jj] = a0 * pB[ii+0+ldb*jj];
				}
			}
		}
	else
		{
		int ldc = sC->m;
		REAL *pC = sC->pA + ci + cj*ldc;
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			a0 = alpha * dA[ii+0];
			a1 = alpha * dA[ii+1];
			for(jj=0; jj<n; jj++)
				{
				pD[ii+0+ldd*jj] = a0 * pB[ii+0+ldb*jj] + beta * pC[ii+0+ldc*jj];
				pD[ii+1+ldd*jj] = a1 * pB[ii+1+ldb*jj] + beta * pC[ii+1+ldc*jj];
				}
			}
		for(; ii<m; ii++)
			{
			a0 = alpha * dA[ii];
			for(jj=0; jj<n; jj++)
				{
				pD[ii+0+ldd*jj] = a0 * pB[ii+0+ldb*jj] + beta * pC[ii+0+ldc*jj];
				}
			}
		}
	return;
	}



// dgemm with B diagonal matrix (stored as strvec)
void GEMM_R_DIAG_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRVEC *sB, int bi, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj;
	int lda = sA->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *dB = sB->pa + bi;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL a0, a1;
	if(beta==0.0)
		{
		jj = 0;
		for(; jj<n-1; jj+=2)
			{
			a0 = alpha * dB[jj+0];
			a1 = alpha * dB[jj+1];
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*(jj+0)] = a0 * pA[ii+lda*(jj+0)];
				pD[ii+ldd*(jj+1)] = a1 * pA[ii+lda*(jj+1)];
				}
			}
		for(; jj<n; jj++)
			{
			a0 = alpha * dB[jj+0];
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*(jj+0)] = a0 * pA[ii+lda*(jj+0)];
				}
			}
		}
	else
		{
		int ldc = sC->m;
		REAL *pC = sC->pA + ci + cj*ldc;
		jj = 0;
		for(; jj<n-1; jj+=2)
			{
			a0 = alpha * dB[jj+0];
			a1 = alpha * dB[jj+1];
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*(jj+0)] = a0 * pA[ii+lda*(jj+0)] + beta * pC[ii+ldc*(jj+0)];
				pD[ii+ldd*(jj+1)] = a1 * pA[ii+lda*(jj+1)] + beta * pC[ii+ldc*(jj+1)];
				}
			}
		for(; jj<n; jj++)
			{
			a0 = alpha * dB[jj+0];
			for(ii=0; ii<m; ii++)
				{
				pD[ii+ldd*(jj+0)] = a0 * pA[ii+lda*(jj+0)] + beta * pC[ii+ldc*(jj+0)];
				}
			}
		}
	return;
	}



#else

#error : wrong LA choice

#endif





