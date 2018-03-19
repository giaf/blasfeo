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



#if defined(LA_REFERENCE) | defined(TESTING_MODE)


// dgemm nt
void GEMM_NT_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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



// dgemm nn
void GEMM_NN_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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



// dtrsm_left_lower_nottransposed_unit
void TRSM_LLNU_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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
#if 1
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
#if 0
			for(; kk<ii-1; kk+=2)
				{
				d_00 -= pA[ii+0+lda*(kk+0)] * pD[kk+ldd*(jj+0)];
				d_10 -= pA[ii+1+lda*(kk+0)] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+0+lda*(kk+0)] * pD[kk+ldd*(jj+1)];
				d_11 -= pA[ii+1+lda*(kk+0)] * pD[kk+ldd*(jj+1)];
				d_00 -= pA[ii+0+lda*(kk+1)] * pD[kk+ldd*(jj+0)];
				d_10 -= pA[ii+1+lda*(kk+1)] * pD[kk+ldd*(jj+0)];
				d_01 -= pA[ii+0+lda*(kk+1)] * pD[kk+ldd*(jj+1)];
				d_11 -= pA[ii+1+lda*(kk+1)] * pD[kk+ldd*(jj+1)];
				}
			if(kk<ii)
#else
			for(; kk<ii; kk++)
#endif
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
#else
	// copy
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			for(ii=0; ii<m; ii++)
				pD[ii+ldd*jj] = alpha * pB[ii+ldb*jj];
		}
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m; ii++)
			{
			d_00 = pD[ii+ldd*jj];
			for(kk=ii+1; kk<m; kk++)
				{
				pD[kk+ldd*jj] -= pA[kk+lda*ii] * d_00;
				}
			}
		}
#endif
	return;
	}



// dtrsm_left_upper_nottransposed_notunit
void TRSM_LUNN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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
	if(!(sA->use_dA==1 & ai==0 & aj==0))
		{
		// inverte diagonal of pA
		for(ii=0; ii<m; ii++)
			dA[ii] = 1.0/pA[ii+lda*ii];
		// use only now
		sA->use_dA = 0;
		}
#if 1
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
#if 0
			for(; kk<m-1; kk+=2)
				{
				d_00 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_10 -= pA[id+1+lda*(kk+0)] * pD[kk+0+ldd*(jj+0)];
				d_01 -= pA[id+0+lda*(kk+0)] * pD[kk+0+ldd*(jj+1)];
				d_11 -= pA[id+1+lda*(kk+0)] * pD[kk+0+ldd*(jj+1)];
				d_00 -= pA[id+0+lda*(kk+1)] * pD[kk+1+ldd*(jj+0)];
				d_10 -= pA[id+1+lda*(kk+1)] * pD[kk+1+ldd*(jj+0)];
				d_01 -= pA[id+0+lda*(kk+1)] * pD[kk+1+ldd*(jj+1)];
				d_11 -= pA[id+1+lda*(kk+1)] * pD[kk+1+ldd*(jj+1)];
				}
			if(kk<m)
#else
			for(; kk<m; kk++)
#endif
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
#else
	// copy
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			for(ii=0; ii<m; ii++)
				pD[ii+ldd*jj] = alpha * pB[ii+ldb*jj];
		}
	// solve
	for(jj=0; jj<n; jj++)
		{
		for(ii=m-1; ii>=0; ii--)
			{
			d_00 = pD[ii+ldd*jj] * dA[ii];
			pD[ii+ldd*jj] = d_00;
			for(kk=0; kk<ii; kk++)
				{
				pD[kk+ldd*jj] -= pA[kk+lda*ii] * d_00;
				}
			}
		}
#endif
	return;
	}



// dtrsm_right_lower_transposed_unit
void TRSM_RLTU_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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
		// factorize diagonal
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



// dtrsm_right_lower_transposed_unit
void TRSM_RLTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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



// dtrsm_right_upper_transposed_notunit
void TRSM_RUTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
	printf("\nblasfeo_dtrsm_rutn: feature not implemented yet\n");
	exit(1);
//	if(!(pB==pD))
//		{
//		for(jj=0; jj<n; jj++)
//			COPY(&m, pB+jj*sB->m, &i1, pD+jj*sD->m, &i1);
//		}
//	TRSM(&cr, &cu, &ct, &cn, &m, &n, &alpha, pA, &(sA->m), pD, &(sD->m));
	return;
	}



// dtrmm_right_upper_transposed_notunit (A triangular !!!)
void TRMM_RUTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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
void TRMM_RLNN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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



// dsyrk_lower_nortransposed (allowing for different factors => use dgemm !!!)
void SYRK_LN_LIBSTR(int m, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0)
		return;
	int ii, jj, kk;
	int n = m; // TODO optimize for this case !!!!!!!!!
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



// dsyrk_lower_nortransposed (allowing for different factors => use dgemm !!!)
void SYRK_LN_MN_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	if(m<=0 | n<=0)
		return;
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



#elif defined(LA_BLAS)



// dgemm nt
void GEMM_NT_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cn = 'n';
	char ct = 't';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long kk = k;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldc = sC->m;
	long long ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&cn, &ct, &mm, &nn, &kk, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
#else
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
#endif
	return;
	}



// dgemm nn
void GEMM_NN_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cn = 'n';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pC = sC->pA+ci+cj*sC->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long kk = k;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldc = sC->m;
	long long ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	GEMM(&cn, &cn, &mm, &nn, &kk, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrsm_left_lower_nottransposed_unit
void TRSM_LLNU_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cl, &cn, &cu, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrsm_left_upper_nottransposed_notunit
void TRSM_LUNN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cl, &cu, &cn, &cn, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrsm_right_lower_transposed_unit
void TRSM_RLTU_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &ct, &cu, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrsm_right_lower_transposed_notunit
void TRSM_RLTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cl, &ct, &cn, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrsm_right_upper_transposed_notunit
void TRSM_RUTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRSM(&cr, &cu, &ct, &cn, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrmm_right_upper_transposed_notunit (A triangular !!!)
void TRMM_RUTN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRMM(&cr, &cu, &ct, &cn, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dtrmm_right_lower_nottransposed_notunit (A triangular !!!)
void TRMM_RLNN_LIBSTR(int m, int n, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, struct STRMAT *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	REAL *pA = sA->pA+ai+aj*sA->m;
	REAL *pB = sB->pA+bi+bj*sB->m;
	REAL *pD = sD->pA+di+dj*sD->m;
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldd = sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pB+jj*ldb, &i1, pD+jj*ldd, &i1);
		}
	TRMM(&cr, &cl, &cn, &cn, &mm, &nn, &alpha, pA, &lda, pD, &ldd);
#else
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
#endif
	return;
	}



// dsyrk_lower_nortransposed (allowing for different factors => use dgemm !!!)
void SYRK_LN_LIBSTR(int m, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
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
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long kk = k;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldc = sC->m;
	long long ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<m; jj++)
			COPY(&mm, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cl, &cn, &mm, &kk, &alpha, pA, &lda, &beta, pD, &ldd);
		}
	else
		{
		GEMM(&cn, &ct, &mm, &mm, &kk, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
#else
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
#endif
	return;
	}

// dsyrk_lower_nortransposed (allowing for different factors => use dgemm !!!)
void SYRK_LN_MN_LIBSTR(int m, int n, int k, REAL alpha, struct STRMAT *sA, int ai, int aj, struct STRMAT *sB, int bi, int bj, REAL beta, struct STRMAT *sC, int ci, int cj, struct STRMAT *sD, int di, int dj)
	{
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
#if defined(REF_BLAS_BLIS)
	long long i1 = 1;
	long long mm = m;
	long long nn = n;
	long long kk = k;
	long long mmn = mm-nn;
	long long lda = sA->m;
	long long ldb = sB->m;
	long long ldc = sC->m;
	long long ldd = sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			COPY(&mm, pC+jj*ldc, &i1, pD+jj*ldd, &i1);
		}
	if(pA==pB)
		{
		SYRK(&cl, &cn, &nn, &kk, &alpha, pA, &lda, &beta, pD, &ldd);
		GEMM(&cn, &ct, &mmn, &nn, &kk, &alpha, pA+n, &lda, pB, &ldb, &beta, pD+n, &ldd);
		}
	else
		{
		GEMM(&cn, &ct, &mm, &nn, &kk, &alpha, pA, &lda, pB, &ldb, &beta, pD, &ldd);
		}
#else
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
#endif
	return;
	}

#else

#error : wrong LA choice

#endif
