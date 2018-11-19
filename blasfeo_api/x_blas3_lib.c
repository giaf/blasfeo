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



#if defined(LA_REFERENCE) | defined(TESTING_MODE)


// dgemm nn
void GEMM_NN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL 
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			c_01 = 0.0; ;
			c_11 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+0)];
				c_10 += pA[(ii+1)+lda*kk] * pB[kk+ldb*(jj+0)];
				c_01 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+1)];
				c_11 += pA[(ii+1)+lda*kk] * pB[kk+ldb*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11 + beta * pC[(ii+1)+ldc*(jj+1)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			c_01 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+0)];
				c_01 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+0)];
				c_10 += pA[(ii+1)+lda*kk] * pB[kk+ldb*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[kk+ldb*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			}
		}
	return;
	}



// dgemm nt
void GEMM_NT(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL 
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[(jj+0)+ldb*kk];
				c_10 += pA[(ii+1)+lda*kk] * pB[(jj+0)+ldb*kk];
				c_01 += pA[(ii+0)+lda*kk] * pB[(jj+1)+ldb*kk];
				c_11 += pA[(ii+1)+lda*kk] * pB[(jj+1)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11 + beta * pC[(ii+1)+ldc*(jj+1)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			c_01 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[(jj+0)+ldb*kk];
				c_01 += pA[(ii+0)+lda*kk] * pB[(jj+1)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[(jj+0)+ldb*kk];
				c_10 += pA[(ii+1)+lda*kk] * pB[(jj+0)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[(ii+0)+lda*kk] * pB[(jj+0)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			}
		}
	return;
	}



// dgemm tn
void GEMM_TN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL 
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			c_01 = 0.0; ;
			c_11 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_10 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+0)];
				c_01 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+1)];
				c_11 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11 + beta * pC[(ii+1)+ldc*(jj+1)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			c_01 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_01 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_10 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			}
		}
	return;
	}



// dgemm tt
void GEMM_TT(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL 
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			c_01 = 0.0; ;
			c_11 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[(jj+0)+ldb*kk];
				c_10 += pA[kk+lda*(ii+1)] * pB[(jj+0)+ldb*kk];
				c_01 += pA[kk+lda*(ii+0)] * pB[(jj+1)+ldb*kk];
				c_11 += pA[kk+lda*(ii+1)] * pB[(jj+1)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11 + beta * pC[(ii+1)+ldc*(jj+1)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			c_01 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[(jj+0)+ldb*kk];
				c_01 += pA[kk+lda*(ii+0)] * pB[(jj+1)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01 + beta * pC[(ii+0)+ldc*(jj+1)];
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[(jj+0)+ldb*kk];
				c_10 += pA[kk+lda*(ii+1)] * pB[(jj+0)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10 + beta * pC[(ii+1)+ldc*(jj+0)];
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[(jj+0)+ldb*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00 + beta * pC[(ii+0)+ldc*(jj+0)];
			}
		}
	return;
	}



// dtrsm_left_lower_nottransposed_notunit
void TRSM_LLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		d_00, d_01,
		d_10, d_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda; // triangular
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL *dA = sA->dA;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA<n)
			{
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / pA[ii+lda*ii];
			sA->use_dA = n;
			}
		}
	else
		{
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0; // nonzero offset makes diagonal dirty
		}
	// solve
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*(jj+0)];
			d_10 = alpha * pB[ii+1+ldb*(jj+0)];
			d_01 = alpha * pB[ii+0+ldb*(jj+1)];
			d_11 = alpha * pB[ii+1+ldb*(jj+1)];
			kk = 0;
			for(; kk<ii; kk++)
				{
				d_00 -= pA[ii+0+lda*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pA[ii+1+lda*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+0+lda*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pA[ii+1+lda*kk] * pD[kk+ldd*(jj+1)];
				}
			d_00 *= dA[ii+0];
			d_01 *= dA[ii+0];
			d_10 -= pA[ii+1+lda*ii] * d_00;
			d_11 -= pA[ii+1+lda*ii] * d_01;
			d_10 *= dA[ii+1];
			d_11 *= dA[ii+1];
			pD[ii+0+ldd*(jj+0)] = d_00;
			pD[ii+1+ldd*(jj+0)] = d_10;
			pD[ii+0+ldd*(jj+1)] = d_01;
			pD[ii+1+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+ldb*(jj+0)];
			d_01 = alpha * pB[ii+ldb*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+lda*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+lda*kk] * pD[kk+ldd*(jj+1)];
				}
			d_00 *= dA[ii+0];
			d_01 *= dA[ii+0];
			pD[ii+ldd*(jj+0)] = d_00;
			pD[ii+ldd*(jj+1)] = d_01;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*jj];
			d_10 = alpha * pB[ii+1+ldb*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+0+lda*kk] * pD[kk+ldd*jj];
				d_10 -= pA[ii+1+lda*kk] * pD[kk+ldd*jj];
				}
			d_00 *= dA[ii+0];
			d_10 -= pA[ii+1+lda*kk] * d_00;
			d_10 *= dA[ii+1];
			pD[ii+0+ldd*jj] = d_00;
			pD[ii+1+ldd*jj] = d_10;
			}
		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+ldb*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+lda*kk] * pD[kk+ldd*jj];
				}
			d_00 *= dA[ii+0];
			pD[ii+ldd*jj] = d_00;
			}
		}
	return;
	}



// dtrsm_left_lower_nottransposed_unit
void TRSM_LLNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		d_00, d_01,
		d_10, d_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda; // triangular
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	// solve
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*(jj+0)];
			d_10 = alpha * pB[ii+1+ldb*(jj+0)];
			d_01 = alpha * pB[ii+0+ldb*(jj+1)];
			d_11 = alpha * pB[ii+1+ldb*(jj+1)];
			kk = 0;
			for(; kk<ii; kk++)
				{
				d_00 -= pA[ii+0+lda*kk] * pD[kk+ldd*(jj+0)];
				d_10 -= pA[ii+1+lda*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+0+lda*kk] * pD[kk+ldd*(jj+1)];
				d_11 -= pA[ii+1+lda*kk] * pD[kk+ldd*(jj+1)];
				}
			d_10 -= pA[ii+1+lda*kk] * d_00;
			d_11 -= pA[ii+1+lda*kk] * d_01;
			pD[ii+0+ldd*(jj+0)] = d_00;
			pD[ii+1+ldd*(jj+0)] = d_10;
			pD[ii+0+ldd*(jj+1)] = d_01;
			pD[ii+1+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+ldb*(jj+0)];
			d_01 = alpha * pB[ii+ldb*(jj+1)];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+lda*kk] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+lda*kk] * pD[kk+ldd*(jj+1)];
				}
			pD[ii+ldd*(jj+0)] = d_00;
			pD[ii+ldd*(jj+1)] = d_01;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*jj];
			d_10 = alpha * pB[ii+1+ldb*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+0+lda*kk] * pD[kk+ldd*jj];
				d_10 -= pA[ii+1+lda*kk] * pD[kk+ldd*jj];
				}
			d_10 -= pA[ii+1+lda*kk] * d_00;
			pD[ii+0+ldd*jj] = d_00;
			pD[ii+1+ldd*jj] = d_10;
			}
		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+ldb*jj];
			for(kk=0; kk<ii; kk++)
				{
				d_00 -= pA[ii+lda*kk] * pD[kk+ldd*jj];
				}
			pD[ii+ldd*jj] = d_00;
			}
		}
	return;
	}



// dtrsm_lltn
void TRSM_LLTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_lltn: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_lltu
void TRSM_LLTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_lltu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_left_upper_nottransposed_notunit
void TRSM_LUNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk, id;
	REAL
		d_00, d_01,
		d_10, d_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda; // triangular
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL *dA = sA->dA;
	if(ai==0 & aj==0)
		{
		if (sA->use_dA<m)
			{
			// invert diagonal of pA
			for(ii=0; ii<m; ii++)
				dA[ii] = 1.0/pA[ii+lda*ii];
			// use only now
			sA->use_dA = m;
			}
		}
	else
		{
		for(ii=0; ii<m; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0; // nonzero offset makes diagonal dirty
		}

	jj = 0;

	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			id = m-ii-2;
			d_00 = alpha * pB[id+0+ldb*(jj+0)];
			d_10 = alpha * pB[id+1+ldb*(jj+0)];
			d_01 = alpha * pB[id+0+ldb*(jj+1)];
			d_11 = alpha * pB[id+1+ldb*(jj+1)];
			kk = id+2;

			for(; kk<m; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_10 -= pA[id+1+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_01 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+1)];
				d_11 -= pA[id+1+lda*(kk+0)] * pD[kk+0+ldd*(jj+1)];
				}

			d_10 *= dA[id+1];
			d_11 *= dA[id+1];

			d_00 -= pA[id+0+lda*(id+1)] * d_10;
			d_01 -= pA[id+0+lda*(id+1)] * d_11;

			d_00 *= dA[id+0];
			d_01 *= dA[id+0];

			pD[id+0+ldd*(jj+0)] = d_00;
			pD[id+1+ldd*(jj+0)] = d_10;
			pD[id+0+ldd*(jj+1)] = d_01;
			pD[id+1+ldd*(jj+1)] = d_11;
			}
		for(; ii<m; ii++)
			{
			id = m-ii-1;
			d_00 = alpha * pB[id+0+ldb*(jj+0)];
			d_01 = alpha * pB[id+0+ldb*(jj+1)];
			kk = id+1;
			for(; kk<m; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_01 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+1)];
				}
			d_00 *= dA[id+0];
			d_01 *= dA[id+0];
			pD[id+0+ldd*(jj+0)] = d_00;
			pD[id+0+ldd*(jj+1)] = d_01;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			id = m-ii-2;
			d_00 = alpha * pB[id+0+ldb*(jj+0)];
			d_10 = alpha * pB[id+1+ldb*(jj+0)];
			kk = id+2;
			for(; kk<m; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_10 -= pA[id+1+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				}
			d_10 *= dA[id+1];
			d_00 -= pA[id+0+lda*(id+1)] * d_10;
			d_00 *= dA[id+0];
			pD[id+0+ldd*(jj+0)] = d_00;
			pD[id+1+ldd*(jj+0)] = d_10;
			}
		for(; ii<m; ii++)
			{
			id = m-ii-1;
			d_00 = alpha * pB[id+0+ldb*(jj+0)];
			kk = id+1;
			for(; kk<m; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				}
			d_00 *= dA[id+0];
			pD[id+0+ldd*(jj+0)] = d_00;
			}
		}
	return;
	}



// dtrsm_lunu
void TRSM_LUNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_lunu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_lutn
void TRSM_LUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_lutn: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_lutu
void TRSM_LUTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_lutu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_rlnn
void TRSM_RLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_rlnn: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_rlnu
void TRSM_RLNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_rlnu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_right_lower_transposed_not-unit
void TRSM_RLTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL *dA = sA->dA;
	if(ai==0 & aj==0)
		{
		if(sA->use_dA<n)
			{
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0 / pA[ii+lda*ii];
			sA->use_dA = n;
			}
		}
	else
		{
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0; // nonzero offset makes diagonal dirty
		}
	REAL
		f_00_inv,
		f_10, f_11_inv,
		c_00, c_01,
		c_10, c_11;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		f_00_inv = dA[jj+0];
		f_10 = pA[jj+1+lda*(jj+0)];
		f_11_inv = dA[jj+1];
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = alpha * pB[ii+0+ldb*(jj+0)];
			c_10 = alpha * pB[ii+1+ldb*(jj+0)];
			c_01 = alpha * pB[ii+0+ldb*(jj+1)];
			c_11 = alpha * pB[ii+1+ldb*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pA[jj+0+lda*kk];
				c_10 -= pD[ii+1+ldd*kk] * pA[jj+0+lda*kk];
				c_01 -= pD[ii+0+ldd*kk] * pA[jj+1+lda*kk];
				c_11 -= pD[ii+1+ldd*kk] * pA[jj+1+lda*kk];
				}
			c_00 *= f_00_inv;
			c_10 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			pD[ii+1+ldd*(jj+0)] = c_10;
			c_01 -= c_00 * f_10;
			c_11 -= c_10 * f_10;
			c_01 *= f_11_inv;
			c_11 *= f_11_inv;
			pD[ii+0+ldd*(jj+1)] = c_01;
			pD[ii+1+ldd*(jj+1)] = c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = alpha * pB[ii+0+ldb*(jj+0)];
			c_01 = alpha * pB[ii+0+ldb*(jj+1)];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pA[jj+0+lda*kk];
				c_01 -= pD[ii+0+ldd*kk] * pA[jj+1+lda*kk];
				}
			c_00 *= f_00_inv;
			pD[ii+0+ldd*(jj+0)] = c_00;
			c_01 -= c_00 * f_10;
			c_01 *= f_11_inv;
			pD[ii+0+ldd*(jj+1)] = c_01;
			}
		}
	for(; jj<n; jj++)
		{
		// factorize diagonal
		f_00_inv = dA[jj];
		for(ii=0; ii<m; ii++)
			{
			c_00 = alpha * pB[ii+ldb*jj];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+ldd*kk] * pA[jj+lda*kk];
				}
			c_00 *= f_00_inv;
			pD[ii+ldd*jj] = c_00;
			}
		}
	return;
	}



// dtrsm_right_lower_transposed_unit
void TRSM_RLTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	REAL
		f_10,
		c_00, c_01,
		c_10, c_11;

	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		f_10 = pA[jj+1+lda*(jj+0)];
		ii = 0;
		for(; ii<m-1; ii+=2)
			{

			c_00 = alpha * pB[ii+0+ldb*(jj+0)];
			c_10 = alpha * pB[ii+1+ldb*(jj+0)];
			c_01 = alpha * pB[ii+0+ldb*(jj+1)];
			c_11 = alpha * pB[ii+1+ldb*(jj+1)];

			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pA[jj+0+lda*kk];
				c_10 -= pD[ii+1+ldd*kk] * pA[jj+0+lda*kk];
				c_01 -= pD[ii+0+ldd*kk] * pA[jj+1+lda*kk];
				c_11 -= pD[ii+1+ldd*kk] * pA[jj+1+lda*kk];
				}

			pD[ii+0+ldd*(jj+0)] = c_00;
			pD[ii+1+ldd*(jj+0)] = c_10;
			c_01 -= c_00 * f_10;
			c_11 -= c_10 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01;
			pD[ii+1+ldd*(jj+1)] = c_11;
			}

		for(; ii<m; ii++)
			{
			c_00 = alpha * pB[ii+0+ldb*(jj+0)];
			c_01 = alpha * pB[ii+0+ldb*(jj+1)];

			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+0+ldd*kk] * pA[jj+0+lda*kk];
				c_01 -= pD[ii+0+ldd*kk] * pA[jj+1+lda*kk];
				}

			pD[ii+0+ldd*(jj+0)] = c_00;
			c_01 -= c_00 * f_10;
			pD[ii+0+ldd*(jj+1)] = c_01;
			}
		}

	for(; jj<n; jj++)
		{

		for(ii=0; ii<m; ii++)
			{
			c_00 = alpha * pB[ii+ldb*jj];
			for(kk=0; kk<jj; kk++)
				{
				c_00 -= pD[ii+ldd*kk] * pA[jj+lda*kk];
				}
			pD[ii+ldd*jj] = c_00;
			}
		}
	return;
	}



// dtrsm_runn
void TRSM_RUNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_runn: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_runu
void TRSM_RUNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_runu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrsm_right_upper_transposed_notunit
void TRSM_RUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	int ii, jj, kk, id;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL
		d_00, d_01,
		d_10, d_11;
	REAL *pA = sA->pA+ai+aj*lda;
	REAL *pB = sB->pA+bi+bj*ldb;
	REAL *pD = sD->pA+di+dj*ldd;
	REAL *dA = sA->dA;

	if(ai==0 & aj==0)
		{
		if (sA->use_dA<n)
			{
			// invert diagonal of pA
			for(ii=0; ii<n; ii++)
				dA[ii] = 1.0/pA[ii+lda*ii];
			// use only now
			sA->use_dA = n;
			}
		}
	else
		{
		for(ii=0; ii<n; ii++)
			dA[ii] = 1.0 / pA[ii+lda*ii];
		sA->use_dA = 0; // nonzero offset makes diagonal dirty
		}

	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		id = n-jj-2;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*(id+0)];
			d_10 = alpha * pB[ii+1+ldb*(id+0)];
			d_01 = alpha * pB[ii+0+ldb*(id+1)];
			d_11 = alpha * pB[ii+1+ldb*(id+1)];
			kk = id+2;

			for(; kk<n; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[ii+0+ldd*(kk+0)];
				d_10 -= pA[id+0+lda*(kk+0)] * pD[ii+1+ldd*(kk+0)];
				d_01 -= pA[id+1+lda*(kk+0)] * pD[ii+0+ldd*(kk+0)];
				d_11 -= pA[id+1+lda*(kk+0)] * pD[ii+1+ldd*(kk+0)];
				}

			d_01 *= dA[id+1];
			d_11 *= dA[id+1];

			d_00 -= pA[id+0+lda*(id+1)] * d_01;
			d_10 -= pA[id+0+lda*(id+1)] * d_11;

			d_00 *= dA[id+0];
			d_10 *= dA[id+0];

			pD[ii+0+ldd*(id+0)] = d_00;
			pD[ii+1+ldd*(id+0)] = d_10;
			pD[ii+0+ldd*(id+1)] = d_01;
			pD[ii+1+ldd*(id+1)] = d_11;
			}

		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+0+ldb*(id+0)];
			d_01 = alpha * pB[ii+0+ldb*(id+1)];
			kk = id+2;
			for(; kk<n; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[ii+0+ldd*(kk+0)];
				d_01 -= pA[id+1+lda*(kk+0)] * pD[ii+0+ldd*(kk+0)];
				}

			d_01 *= dA[id+1];
			d_00 -= pA[id+0+lda*(id+1)] * d_01;
			d_00 *= dA[id+0];

			pD[ii+0+ldd*(id+0)] = d_00;
			pD[ii+0+ldd*(id+1)] = d_01;

			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		id = n-jj-1;
		for(; ii<m-1; ii+=2)
			{
			d_00 = alpha * pB[ii+0+ldb*(id+0)];
			d_10 = alpha * pB[ii+1+ldb*(id+0)];
			kk = id+1;

			for(; kk<n; kk++)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[ii+0+ldd*(kk+0)];
				d_10 -= pA[id+0+lda*(kk+0)] * pD[ii+1+ldd*(kk+0)];
				}

			d_00 *= dA[id+0];
			d_10 *= dA[id+0];

			pD[ii+0+ldd*(id+0)] = d_00;
			pD[ii+1+ldd*(id+0)] = d_10;
			}
		for(; ii<m; ii++)
			{
			d_00 = alpha * pB[ii+ldb*(id)];
			kk = id+1;
			for(; kk<n; kk++)
				d_00 -= pA[id+lda*(kk)] * pD[ii+ldd*(kk)];

			pD[ii+ldd*(id)] = d_00 * dA[id];
			}
		}
	return;
	}



// dtrsm_rutu
void TRSM_RUTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
#ifndef BENCHMARKS_MODE
	printf("\nblasfeo_xtrsm_rutu: feature not implemented yet\n");
	exit(1);
#endif
	return;
	}



// dtrmm_right_upper_transposed_notunit (A triangular !!!)
void TRMM_RUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			kk = jj;
			c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
			c_10 += pB[(ii+1)+ldb*kk] * pA[(jj+0)+lda*kk];
			kk++;
			for(; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
				c_10 += pB[(ii+1)+ldb*kk] * pA[(jj+0)+lda*kk];
				c_01 += pB[(ii+0)+ldb*kk] * pA[(jj+1)+lda*kk];
				c_11 += pB[(ii+1)+ldb*kk] * pA[(jj+1)+lda*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10;
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01;
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			c_01 = 0.0;
			kk = jj;
			c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
			kk++;
			for(; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
				c_01 += pB[(ii+0)+ldb*kk] * pA[(jj+1)+lda*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			for(kk=jj; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
				c_10 += pB[(ii+1)+ldb*kk] * pA[(jj+0)+lda*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			for(kk=jj; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[(jj+0)+lda*kk];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			}
		}	
	return;
	}



// dtrmm_right_lower_nottransposed_notunit (A triangular !!!)
void TRMM_RLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL 
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			c_01 = 0.0; ;
			c_11 = 0.0; ;
			kk = jj;
			c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
			c_10 += pB[(ii+1)+ldb*kk] * pA[kk+lda*(jj+0)];
			kk++;
			for(; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
				c_10 += pB[(ii+1)+ldb*kk] * pA[kk+lda*(jj+0)];
				c_01 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+1)];
				c_11 += pB[(ii+1)+ldb*kk] * pA[kk+lda*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10;
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01;
			pD[(ii+1)+ldd*(jj+1)] = alpha * c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			c_01 = 0.0; ;
			kk = jj;
			c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
			kk++;
			for(; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
				c_01 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+1)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+0)+ldd*(jj+1)] = alpha * c_01;
			}
		}
	for(; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0; ;
			c_10 = 0.0; ;
			for(kk=jj; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
				c_10 += pB[(ii+1)+ldb*kk] * pA[kk+lda*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			pD[(ii+1)+ldd*(jj+0)] = alpha * c_10;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0; ;
			for(kk=jj; kk<n; kk++)
				{
				c_00 += pB[(ii+0)+ldb*kk] * pA[kk+lda*(jj+0)];
				}
			pD[(ii+0)+ldd*(jj+0)] = alpha * c_00;
			}
		}
	return;
	}



// dsyrk_lower not-transposed
void SYRK_LN(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		// diagonal
		c_00 = 0.0;
		c_10 = 0.0;
		c_11 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+0+lda*kk] * pB[jj+0+ldb*kk];
			c_10 += pA[jj+1+lda*kk] * pB[jj+0+ldb*kk];
			c_11 += pA[jj+1+lda*kk] * pB[jj+1+ldb*kk];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		pD[jj+1+ldd*(jj+0)] = beta * pC[jj+1+ldc*(jj+0)] + alpha * c_10;
		pD[jj+1+ldd*(jj+1)] = beta * pC[jj+1+ldc*(jj+1)] + alpha * c_11;
		// lower
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_10 += pA[ii+1+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				c_11 += pA[ii+1+lda*kk] * pB[jj+1+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			pD[ii+1+ldd*(jj+1)] = beta * pC[ii+1+ldc*(jj+1)] + alpha * c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			c_01 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			}
		}
	if(jj<m)
		{
		// diagonal
		c_00 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+lda*kk] * pB[jj+ldb*kk];
			}
		pD[jj+ldd*jj] = beta * pC[jj+ldc*jj] + alpha * c_00;
		}
	return;
	}



// dsyrk_lower not-transposed
void SYRK_LN_MN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<n-1; jj+=2)
		{
		// diagonal
		c_00 = 0.0;
		c_10 = 0.0;
		c_11 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+0+lda*kk] * pB[jj+0+ldb*kk];
			c_10 += pA[jj+1+lda*kk] * pB[jj+0+ldb*kk];
			c_11 += pA[jj+1+lda*kk] * pB[jj+1+ldb*kk];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		pD[jj+1+ldd*(jj+0)] = beta * pC[jj+1+ldc*(jj+0)] + alpha * c_10;
		pD[jj+1+ldd*(jj+1)] = beta * pC[jj+1+ldc*(jj+1)] + alpha * c_11;
		// lower
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_10 += pA[ii+1+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				c_11 += pA[ii+1+lda*kk] * pB[jj+1+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			pD[ii+1+ldd*(jj+1)] = beta * pC[ii+1+ldc*(jj+1)] + alpha * c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			c_01 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			}
		}
	for(; jj<n; jj++)
		{
		// diagonal
		c_00 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+lda*kk] * pB[jj+ldb*kk];
			}
		pD[jj+ldd*jj] = beta * pC[jj+ldc*jj] + alpha * c_00;
		// lower
		for(ii=jj+1; ii<m; ii++)
			{
			c_00 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+lda*kk] * pB[jj+ldb*kk];
				}
			pD[ii+ldd*jj] = beta * pC[ii+ldc*jj] + alpha * c_00;
			}
		}
	return;
	}



// dsyrk_lower transposed
void SYRK_LT(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		// diagonal
		c_00 = 0.0;
		c_10 = 0.0;
		c_11 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[kk+lda*(jj+0)] * pB[kk+ldb*(jj+0)];
			c_10 += pA[kk+lda*(jj+1)] * pB[kk+ldb*(jj+0)];
			c_11 += pA[kk+lda*(jj+1)] * pB[kk+ldb*(jj+1)];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		pD[jj+1+ldd*(jj+0)] = beta * pC[jj+1+ldc*(jj+0)] + alpha * c_10;
		pD[jj+1+ldd*(jj+1)] = beta * pC[jj+1+ldc*(jj+1)] + alpha * c_11;
		// lower
		ii = jj+2;
		for(; ii<m-1; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_10 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+0)];
				c_01 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+1)];
				c_11 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+1)];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			pD[ii+1+ldd*(jj+1)] = beta * pC[ii+1+ldc*(jj+1)] + alpha * c_11;
			}
		for(; ii<m; ii++)
			{
			c_00 = 0.0;
			c_01 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_01 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+1)];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			}
		}
	if(jj<m)
		{
		// diagonal
		c_00 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[kk+lda*jj] * pB[kk+ldb*jj];
			}
		pD[jj+ldd*jj] = beta * pC[jj+ldc*jj] + alpha * c_00;
		}
	return;
	}



// dsyrk_upper not-transposed
void SYRK_UN(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		// upper
		ii = 0;
		for(; ii<jj; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_10 += pA[ii+1+lda*kk] * pB[jj+0+ldb*kk];
				c_01 += pA[ii+0+lda*kk] * pB[jj+1+ldb*kk];
				c_11 += pA[ii+1+lda*kk] * pB[jj+1+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			pD[ii+1+ldd*(jj+1)] = beta * pC[ii+1+ldc*(jj+1)] + alpha * c_11;
			}
		// diagonal
		c_00 = 0.0;
		c_01 = 0.0;
		c_11 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+0+lda*kk] * pB[jj+0+ldb*kk];
			c_01 += pA[jj+0+lda*kk] * pB[jj+1+ldb*kk];
			c_11 += pA[jj+1+lda*kk] * pB[jj+1+ldb*kk];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		pD[jj+0+ldd*(jj+1)] = beta * pC[jj+0+ldc*(jj+1)] + alpha * c_01;
		pD[jj+1+ldd*(jj+1)] = beta * pC[jj+1+ldc*(jj+1)] + alpha * c_11;
		}
	if(jj<m)
		{
		// upper
		ii = 0;
		for(; ii<jj; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[ii+0+lda*kk] * pB[jj+0+ldb*kk];
				c_10 += pA[ii+1+lda*kk] * pB[jj+0+ldb*kk];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			}
		// diagonal
		c_00 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[jj+0+lda*kk] * pB[jj+0+ldb*kk];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		}
	return;
	}



// dsyrk_upper transposed
void SYRK_UT(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{
	if(m<=0)
		return;

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int ii, jj, kk;
	REAL
		c_00, c_01,
		c_10, c_11;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	REAL *pC = sC->pA + ci + cj*ldc;
	REAL *pD = sD->pA + di + dj*ldd;
	jj = 0;
	for(; jj<m-1; jj+=2)
		{
		// upper
		ii = 0;
		for(; ii<jj; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			c_01 = 0.0;
			c_11 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_10 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+0)];
				c_01 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+1)];
				c_11 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+1)];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			pD[ii+0+ldd*(jj+1)] = beta * pC[ii+0+ldc*(jj+1)] + alpha * c_01;
			pD[ii+1+ldd*(jj+1)] = beta * pC[ii+1+ldc*(jj+1)] + alpha * c_11;
			}
		// diagonal
		c_00 = 0.0;
		c_01 = 0.0;
		c_11 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[kk+lda*(jj+0)] * pB[kk+ldb*(jj+0)];
			c_01 += pA[kk+lda*(jj+0)] * pB[kk+ldb*(jj+1)];
			c_11 += pA[kk+lda*(jj+1)] * pB[kk+ldb*(jj+1)];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		pD[jj+0+ldd*(jj+1)] = beta * pC[jj+0+ldc*(jj+1)] + alpha * c_01;
		pD[jj+1+ldd*(jj+1)] = beta * pC[jj+1+ldc*(jj+1)] + alpha * c_11;
		}
	if(jj<m)
		{
		// upper
		ii = 0;
		for(; ii<jj; ii+=2)
			{
			c_00 = 0.0;
			c_10 = 0.0;
			for(kk=0; kk<k; kk++)
				{
				c_00 += pA[kk+lda*(ii+0)] * pB[kk+ldb*(jj+0)];
				c_10 += pA[kk+lda*(ii+1)] * pB[kk+ldb*(jj+0)];
				}
			pD[ii+0+ldd*(jj+0)] = beta * pC[ii+0+ldc*(jj+0)] + alpha * c_00;
			pD[ii+1+ldd*(jj+0)] = beta * pC[ii+1+ldc*(jj+0)] + alpha * c_10;
			}
		// diagonal
		c_00 = 0.0;
		for(kk=0; kk<k; kk++)
			{
			c_00 += pA[kk+lda*(jj+0)] * pB[kk+ldb*(jj+0)];
			}
		pD[jj+0+ldd*(jj+0)] = beta * pC[jj+0+ldc*(jj+0)] + alpha * c_00;
		}
	return;
	}



#elif defined(LA_BLAS_WRAPPER)



// dgemm nn
void GEMM_NN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cn = 'n';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&cn, &cn, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
	return;
	}



// dgemm nt
void GEMM_NT(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cn = 'n';
	char ct = 't';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&cn, &ct, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
	return;
	}



// dgemm tn
void GEMM_TN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cn = 'n';
	char ct = 't';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&ct, &cn, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
	return;
	}



// dgemm tt
void GEMM_TT(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cn = 'n';
	char ct = 't';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&ct, &ct, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
	return;
	}



// dtrsm_left_lower_nottransposed_notunit
void TRSM_LLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*sD->m, &i1);
		}
	TRSM(&cl, &cl, &cn, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_lower_nottransposed_unit
void TRSM_LLNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*sD->m, &i1);
		}
	TRSM(&cl, &cl, &cn, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_lower_transposed_notunit
void TRSM_LLTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char ct = 't';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*sD->m, &i1);
		}
	TRSM(&cl, &cl, &ct, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_lower_transposed_unit
void TRSM_LLTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*sD->m, &i1);
		}
	TRSM(&cl, &cl, &ct, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_upper_nottransposed_notunit
void TRSM_LUNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cu, &cn, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_upper_nottransposed_unit
void TRSM_LUNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cu, &cn, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_upper_transposed_notunit
void TRSM_LUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cu, &ct, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_left_upper_transposed_unit
void TRSM_LUTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cu, &ct, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_lower_nottransposed_notunit
void TRSM_RLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &cn, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_lower_nottransposed_unit
void TRSM_RLNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &cn, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_lower_transposed_notunit
void TRSM_RLTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &ct, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_lower_transposed_unit
void TRSM_RLTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &ct, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_upper_nottransposed_notunit
void TRSM_RUNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cu, &cn, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_upper_nottransposed_unit
void TRSM_RUNU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cu, &cn, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_upper_transposed_notunit
void TRSM_RUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cu, &ct, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrsm_right_upper_transposed_unit
void TRSM_RUTU(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cu, &ct, &cu, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrmm_right_upper_transposed_notunit (A triangular !!!)
void TRMM_RUTN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRMM(&cr, &cu, &ct, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dtrmm_right_lower_nottransposed_notunit (A triangular !!!)
void TRMM_RLNN(int m, int n, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRMM(&cr, &cl, &cn, &cn, &m, &n, &alpha, pA, &lda, pD, &ldd);
	return;
	}



// dsyrk lower not-transposed (allowing for different factors => use dgemm !!!)
void SYRK_LN(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA + ai + aj*sA->m;
	REAL *pB = sB->pA + bi + bj*sB->m;
	REAL *pC = sC->pA + ci + cj*sC->m;
	REAL *pD = sD->pA + di + dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<m; jj++)
			COPY(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cl, &cn, &m, &k, &alpha, pA, &lda, &beta, pD, &ldd);
		}
	else
		{
		GEMM(&cn, &ct, &m, &m, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
	return;
	}



// dsyrk lower not-transposed (allowing for different factors => use dgemm !!!)
void SYRK_LN_MN(int m, int n, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA + ai + aj*sA->m;
	REAL *pB = sB->pA + bi + bj*sB->m;
	REAL *pC = sC->pA + ci + cj*sC->m;
	REAL *pD = sD->pA + di + dj*sD->m;
	int i1 = 1;
	int mmn = m-n;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cl, &cn, &n, &k, &alpha, pA, &lda, &beta, pD, &ldd);
		GEMM(&cn, &ct, &mmn, &n, &k, &alpha, pA+n, &lda, pB, &ldb, &beta, pD+n, &ldd);
		}
	else
		{
		GEMM(&cn, &ct, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
	return;
	}



// dsyrk lower transposed (allowing for different factors => use dgemm !!!)
void SYRK_LT(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA + ai + aj*sA->m;
	REAL *pB = sB->pA + bi + bj*sB->m;
	REAL *pC = sC->pA + ci + cj*sC->m;
	REAL *pD = sD->pA + di + dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<m; jj++)
			COPY(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cl, &ct, &m, &k, &alpha, pA, &lda, &beta, pD, &ldd);
		}
	else
		{
		GEMM(&ct, &cn, &m, &m, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
	return;
	}



// dsyrk upper not-transposed (allowing for different factors => use dgemm !!!)
void SYRK_UN(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA + ai + aj*sA->m;
	REAL *pB = sB->pA + bi + bj*sB->m;
	REAL *pC = sC->pA + ci + cj*sC->m;
	REAL *pD = sD->pA + di + dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<m; jj++)
			COPY(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cu, &cn, &m, &k, &alpha, pA, &lda, &beta, pD, &ldd);
		}
	else
		{
		GEMM(&cn, &ct, &m, &m, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
	return;
	}



// dsyrk upper transposed (allowing for different factors => use dgemm !!!)
void SYRK_UT(int m, int k, REAL alpha, struct XMAT *sA, int ai, int aj, struct XMAT *sB, int bi, int bj, REAL beta, struct XMAT *sC, int ci, int cj, struct XMAT *sD, int di, int dj)
	{

	// invalidate stored inverse diagonal of result matrix
	sD->use_dA = 0;

	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA + ai + aj*sA->m;
	REAL *pB = sB->pA + bi + bj*sB->m;
	REAL *pC = sC->pA + ci + cj*sC->m;
	REAL *pD = sD->pA + di + dj*sD->m;
	int i1 = 1;
	int lda = sA->m;
	int ldb = sB->m;
	int ldc = sC->m;
	int ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<m; jj++)
			COPY(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cu, &ct, &m, &k, &alpha, pA, &lda, &beta, pD, &ldd);
		}
	else
		{
		GEMM(&ct, &cn, &m, &m, &k, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
	return;
	}



#else

#error : wrong LA choice

#endif
