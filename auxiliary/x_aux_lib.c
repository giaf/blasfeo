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


// return memory size (in bytes) needed for a strmat
int MEMSIZE_MAT(int m, int n)
	{
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	int size = (m*n+tmp)*sizeof(REAL);
	return size;
	}



// return memory size (in bytes) needed for the diagonal of a strmat
int MEMSIZE_DIAG_MAT(int m, int n)
	{
	int size = 0;
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	size = tmp*sizeof(REAL);
	return size;
	}



// return memory size (in bytes) needed for a strvec
int MEMSIZE_VEC(int m)
	{
	int size = m*sizeof(REAL);
	return size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void CREATE_MAT(int m, int n, struct MAT *sA, void *memory)
	{
	sA->m = m;
	sA->n = n;
	REAL *ptr = (REAL *) memory;
	sA->pA = ptr;
	ptr += m*n;
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->use_dA = 0;
	sA->memsize = (m*n+tmp)*sizeof(REAL);
	return;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void CREATE_VEC(int m, struct VEC *sa, void *memory)
	{
	sa->m = m;
	REAL *ptr = (REAL *) memory;
	sa->pa = ptr;
//	ptr += m * n;
	sa->memsize = m*sizeof(REAL);
	return;
	}



// convert a matrix into a matrix structure
void PACK_MAT(int m, int n, REAL *A, int lda, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii, jj;
	int lda2 = sA->m;
	REAL *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pA[ii+0+jj*lda2] = A[ii+0+jj*lda];
			pA[ii+1+jj*lda2] = A[ii+1+jj*lda];
			pA[ii+2+jj*lda2] = A[ii+2+jj*lda];
			pA[ii+3+jj*lda2] = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pA[ii+0+jj*lda2] = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert and transpose a matrix into a matrix structure
void PACK_TRAN_MAT(int m, int n, REAL *A, int lda, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii, jj;
	int lda2 = sA->m;
	REAL *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pA[jj+(ii+0)*lda2] = A[ii+0+jj*lda];
			pA[jj+(ii+1)*lda2] = A[ii+1+jj*lda];
			pA[jj+(ii+2)*lda2] = A[ii+2+jj*lda];
			pA[jj+(ii+3)*lda2] = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pA[jj+(ii+0)*lda2] = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert a vector into a vector structure
void PACK_VEC(int m, REAL *a, struct VEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	int ii;
	for(ii=0; ii<m; ii++)
		pa[ii] = a[ii];
	return;
	}



// convert a matrix structure into a matrix
void UNPACK_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	REAL *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			A[ii+0+jj*lda] = pA[ii+0+jj*lda2];
			A[ii+1+jj*lda] = pA[ii+1+jj*lda2];
			A[ii+2+jj*lda] = pA[ii+2+jj*lda2];
			A[ii+3+jj*lda] = pA[ii+3+jj*lda2];
			}
		for(; ii<m; ii++)
			{
			A[ii+0+jj*lda] = pA[ii+0+jj*lda2];
			}
		}
	return;
	}



// convert and transpose a matrix structure into a matrix
void UNPACK_TRAN_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	REAL *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			A[jj+(ii+0)*lda] = pA[ii+0+jj*lda2];
			A[jj+(ii+1)*lda] = pA[ii+1+jj*lda2];
			A[jj+(ii+2)*lda] = pA[ii+2+jj*lda2];
			A[jj+(ii+3)*lda] = pA[ii+3+jj*lda2];
			}
		for(; ii<m; ii++)
			{
			A[jj+(ii+0)*lda] = pA[ii+0+jj*lda2];
			}
		}
	return;
	}



// convert a vector structure into a vector
void UNPACK_VEC(int m, struct VEC *sa, int ai, REAL *a)
	{
	REAL *pa = sa->pa + ai;
	int ii;
	for(ii=0; ii<m; ii++)
		a[ii] = pa[ii];
	return;
	}



// cast a matrix into a matrix structure
void CAST_MAT2STRMAT(REAL *A, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	sA->pA = A;
	return;
	}



// cast a matrix into the diagonal of a matrix structure
void CAST_DIAG_MAT2STRMAT(REAL *dA, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	sA->dA = dA;
	return;
	}



// cast a vector into a vector structure
void CAST_VEC2VECMAT(REAL *a, struct VEC *sa)
	{
	sa->pa = a;
	return;
	}



// copy a generic strmat into a generic strmat
void GECP_LIBSTR(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	REAL *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pC[ii+0+jj*ldc] = pA[ii+0+jj*lda];
			pC[ii+1+jj*ldc] = pA[ii+1+jj*lda];
			pC[ii+2+jj*ldc] = pA[ii+2+jj*lda];
			pC[ii+3+jj*ldc] = pA[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pC[ii+0+jj*ldc] = pA[ii+0+jj*lda];
			}
		}
	return;
	}



// scale a generic strmat
void GESC_LIBSTR(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pA[ii+0+jj*lda] *= alpha;
			pA[ii+1+jj*lda] *= alpha;
			pA[ii+2+jj*lda] *= alpha;
			pA[ii+3+jj*lda] *= alpha;
			}
		for(; ii<m; ii++)
			{
			pA[ii+0+jj*lda] *= alpha;
			}
		}
	return;
	}



// scale an generic strmat and copy into generic strmat
void GECPSC_LIBSTR(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;

	int ldb = sB->m;
	REAL *pB = sB->pA + bi + bj*ldb;

	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pB[ii+0+jj*ldb] = pA[ii+0+jj*lda] * alpha;
			pB[ii+1+jj*ldb] = pA[ii+1+jj*lda] * alpha;
			pB[ii+2+jj*ldb] = pA[ii+2+jj*lda] * alpha;
			pB[ii+3+jj*ldb] = pA[ii+3+jj*lda] * alpha;
			}
		for(; ii<m; ii++)
			{
			pB[ii+0+jj*ldb] = pA[ii+0+jj*lda] * alpha;
			}
		}
	return;
	}



// scale and add a generic strmat into a generic strmat
void GEAD_LIBSTR(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	REAL *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pC[ii+0+jj*ldc] += alpha*pA[ii+0+jj*lda];
			pC[ii+1+jj*ldc] += alpha*pA[ii+1+jj*lda];
			pC[ii+2+jj*ldc] += alpha*pA[ii+2+jj*lda];
			pC[ii+3+jj*ldc] += alpha*pA[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pC[ii+0+jj*ldc] += alpha*pA[ii+0+jj*lda];
			}
		}
	return;
	}


// set all elements of a strmat to a value
void GESE(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		for(ii=0; ii<m; ii++)
			{
			pA[ii+lda*jj] = alpha;
			}
		}
	return;
	}




