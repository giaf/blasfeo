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

#include "../include/blasfeo_d_aux.h"


// return memory size (in bytes) needed for a strmat
size_t MEMSIZE_MAT(int m, int n)
	{
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	size_t size = (m*n+tmp)*sizeof(REAL);
	return size;
	}



// return memory size (in bytes) needed for the diagonal of a strmat
size_t MEMSIZE_DIAG_MAT(int m, int n)
	{
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	size_t size = tmp*sizeof(REAL);
	return size;
	}



// return memory size (in bytes) needed for a strvec
size_t MEMSIZE_VEC(int m)
	{
	size_t size = m*sizeof(REAL);
	return size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void CREATE_MAT(int m, int n, struct MAT *sA, void *memory)
	{
	sA->mem = memory;
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
	sa->mem = memory;
	sa->m = m;
	REAL *ptr = (REAL *) memory;
	sa->pa = ptr;
//	ptr += m * n;
	sa->memsize = m*sizeof(REAL);
	return;
	}



// convert a matrix into a matrix structure
void PACK_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

	int ii, jj;
#if defined(MF_COLMAJ)
	int ldb = sB->m;
	REAL *pB = sB->pA + bi + bj*ldb;
	const int bbi=0; const int bbj=0;
#else
	int bbi=bi; int bbj=bj;
#endif
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = A[ii+0+jj*lda];
			XMATEL_B(bbi+ii+1, bbj+jj) = A[ii+1+jj*lda];
			XMATEL_B(bbi+ii+2, bbj+jj) = A[ii+2+jj*lda];
			XMATEL_B(bbi+ii+3, bbj+jj) = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert and transpose a matrix into a matrix structure
void PACK_TRAN_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

	int ii, jj;
#if defined(MF_COLMAJ)
	int ldb = sB->m;
	REAL *pB = sB->pA + bi + bj*ldb;
	const int bbi=0; const int bbj=0;
#else
	int bbi=bi; int bbj=bj;
#endif
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_B(bbi+jj, bbj+(ii+0)) = A[ii+0+jj*lda];
			XMATEL_B(bbi+jj, bbj+(ii+1)) = A[ii+1+jj*lda];
			XMATEL_B(bbi+jj, bbj+(ii+2)) = A[ii+2+jj*lda];
			XMATEL_B(bbi+jj, bbj+(ii+3)) = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+jj, bbj+(ii+0)) = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert a vector into a vector structure
void PACK_VEC(int m, REAL *x, int xi, struct VEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	int ii;
	if(xi==1)
		{
		for(ii=0; ii<m; ii++)
			pa[ii] = x[ii];
		}
	else
		{
		for(ii=0; ii<m; ii++)
			pa[ii] = x[ii*xi];
		}
	return;
	}



// convert a matrix structure into a matrix
void UNPACK_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
	{
	int ii, jj;
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			B[ii+0+jj*ldb] = XMATEL_A(aai+ii+0, aaj+jj);
			B[ii+1+jj*ldb] = XMATEL_A(aai+ii+1, aaj+jj);
			B[ii+2+jj*ldb] = XMATEL_A(aai+ii+2, aaj+jj);
			B[ii+3+jj*ldb] = XMATEL_A(aai+ii+3, aaj+jj);
			}
		for(; ii<m; ii++)
			{
			B[ii+0+jj*ldb] = XMATEL_A(aai+ii+0, aaj+jj);
			}
		}
	return;
	}



// convert and transpose a matrix structure into a matrix
void UNPACK_TRAN_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
	{
	int ii, jj;
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			B[jj+(ii+0)*ldb] = XMATEL_A(aai+ii+0, aaj+jj);
			B[jj+(ii+1)*ldb] = XMATEL_A(aai+ii+1, aaj+jj);
			B[jj+(ii+2)*ldb] = XMATEL_A(aai+ii+2, aaj+jj);
			B[jj+(ii+3)*ldb] = XMATEL_A(aai+ii+3, aaj+jj);
			}
		for(; ii<m; ii++)
			{
			B[jj+(ii+0)*ldb] = XMATEL_A(aai+ii+0, aaj+jj);
			}
		}
	return;
	}



// convert a vector structure into a vector
void UNPACK_VEC(int m, struct VEC *sa, int ai, REAL *x, int xi)
	{
	REAL *pa = sa->pa + ai;
	int ii;
	if(xi==1)
		{
		for(ii=0; ii<m; ii++)
			x[ii] = pa[ii];
		}
	else
		{
		for(ii=0; ii<m; ii++)
			x[ii*xi] = pa[ii];
		}
	return;
	}



// cast a matrix into a matrix structure
void CAST_MAT2STRMAT(REAL *A, struct MAT *sB)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

	sB->pA = A;
	return;
	}



// cast a matrix into the diagonal of a matrix structure
void CAST_DIAG_MAT2STRMAT(REAL *dA, struct MAT *sB)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

	sB->dA = dA;
	return;
	}



// cast a vector into a vector structure
void CAST_VEC2VECMAT(REAL *a, struct VEC *sa)
	{
	sa->pa = a;
	return;
	}



// copy a generic strmat into a generic strmat
void GECP(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

#if defined(MF_COLMAJ)
	int lda = sA->m;
	int ldb = sB->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	const int aai=0; const int aaj=0;
	const int bbi=0; const int bbj=0;
#else
	int aai=ai; int aaj=aj;
	int bbi=bi; int bbj=bj;
#endif
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = XMATEL_A(aai+ii+0, aaj+jj);
			XMATEL_B(bbi+ii+1, bbj+jj) = XMATEL_A(aai+ii+1, aaj+jj);
			XMATEL_B(bbi+ii+2, bbj+jj) = XMATEL_A(aai+ii+2, aaj+jj);
			XMATEL_B(bbi+ii+3, bbj+jj) = XMATEL_A(aai+ii+3, aaj+jj);
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = XMATEL_A(aai+ii+0, aaj+jj);
			}
		}
	return;
	}



// scale a generic strmat
void GESC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_A(aai+ii+0, aaj+jj) *= alpha;
			XMATEL_A(aai+ii+1, aaj+jj) *= alpha;
			XMATEL_A(aai+ii+2, aaj+jj) *= alpha;
			XMATEL_A(aai+ii+3, aaj+jj) *= alpha;
			}
		for(; ii<m; ii++)
			{
			XMATEL_A(aai+ii+0, aaj+jj) *= alpha;
			}
		}
	return;
	}



// scale an generic strmat and copy into generic strmat
void GECPSC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

#if defined(MF_COLMAJ)
	int lda = sA->m;
	int ldb = sB->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	const int aai=0; const int aaj=0;
	const int bbi=0; const int bbj=0;
#else
	int aai=ai; int aaj=aj;
	int bbi=bi; int bbj=bj;
#endif
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = XMATEL_A(aai+ii+0, aaj+jj) * alpha;
			XMATEL_B(bbi+ii+1, bbj+jj) = XMATEL_A(aai+ii+1, aaj+jj) * alpha;
			XMATEL_B(bbi+ii+2, bbj+jj) = XMATEL_A(aai+ii+2, aaj+jj) * alpha;
			XMATEL_B(bbi+ii+3, bbj+jj) = XMATEL_A(aai+ii+3, aaj+jj) * alpha;
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = XMATEL_A(aai+ii+0, aaj+jj) * alpha;
			}
		}
	return;
	}



// scale and add a generic strmat into a generic strmat
void GEAD(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;

#if defined(MF_COLMAJ)
	int lda = sA->m;
	int ldb = sB->m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL *pB = sB->pA + bi + bj*ldb;
	const int aai=0; const int aaj=0;
	const int bbi=0; const int bbj=0;
#else
	int aai=ai; int aaj=aj;
	int bbi=bi; int bbj=bj;
#endif
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) += alpha*XMATEL_A(aai+ii+0, aaj+jj);
			XMATEL_B(bbi+ii+1, bbj+jj) += alpha*XMATEL_A(aai+ii+1, aaj+jj);
			XMATEL_B(bbi+ii+2, bbj+jj) += alpha*XMATEL_A(aai+ii+2, aaj+jj);
			XMATEL_B(bbi+ii+3, bbj+jj) += alpha*XMATEL_A(aai+ii+3, aaj+jj);
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) += alpha*XMATEL_A(aai+ii+0, aaj+jj);
			}
		}
	return;
	}


// set all elements of a strmat to a value
void GESE(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		for(ii=0; ii<m-3; ii+=4)
			{
			XMATEL_A(aai+ii+0, aaj+jj) = alpha;
			XMATEL_A(aai+ii+1, aaj+jj) = alpha;
			XMATEL_A(aai+ii+2, aaj+jj) = alpha;
			XMATEL_A(aai+ii+3, aaj+jj) = alpha;
			}
		for(; ii<m; ii++)
			{
			XMATEL_A(aai+ii+0, aaj+jj) = alpha;
			}
		}
	return;
	}
