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



// copy and transpose a generic strmat into a generic strmat
void GETR(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
			XMATEL_B(bbi+jj, bbj+ii+0) = XMATEL_A(aai+ii+0, aaj+jj);
			XMATEL_B(bbi+jj, bbj+ii+1) = XMATEL_A(aai+ii+1, aaj+jj);
			XMATEL_B(bbi+jj, bbj+ii+2) = XMATEL_A(aai+ii+2, aaj+jj);
			XMATEL_B(bbi+jj, bbj+ii+3) = XMATEL_A(aai+ii+3, aaj+jj);
			}
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+jj, bbj+ii+0) = XMATEL_A(aai+ii+0, aaj+jj);
			}
		}
	return;
	}



// insert element into strmat
void GEIN1(REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	MATEL(sA, ai, aj) = alpha;
	return;
	}



// extract element from strmat
REAL GEEX1(struct MAT *sA, int ai, int aj)
	{
	return MATEL(sA, ai, aj);
	}



// copy a lower triangular strmat into a lower triangular strmat
void TRCP_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
	for(jj=0; jj<m; jj++)
		{
		ii = jj;
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+ii, bbj+jj) = XMATEL_A(aai+ii, aaj+jj);
			}
		}
	return;
	}



// copy and transpose a lower triangular strmat into an upper triangular strmat
void TRTR_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
	for(jj=0; jj<m; jj++)
		{
		ii = jj;
		for(; ii<m; ii++)
			{
			XMATEL_B(bbi+jj, bbj+ii) = XMATEL_A(aai+ii, aaj+jj);
			}
		}
	return;
	}



// copy and transpose an upper triangular strmat into a lower triangular strmat
void TRTR_U(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
	for(jj=0; jj<m; jj++)
		{
		ii = 0;
		for(; ii<=jj; ii++)
			{
			XMATEL_B(bbi+jj, bbj+ii) = XMATEL_A(aai+ii, aaj+jj);
			}
		}
	return;
	}



// set all elements of a strvec to a value
void VECSE(int m, REAL alpha, struct VEC *sx, int xi)
	{
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<m; ii++)
		x[ii] = alpha;
	return;
	}



// copy a strvec into a strvec
void VECCP(int m, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	REAL *pa = sa->pa + ai;
	REAL *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] = pa[ii+0];
		pc[ii+1] = pa[ii+1];
		pc[ii+2] = pa[ii+2];
		pc[ii+3] = pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] = pa[ii+0];
		}
	return;
	}



// scale a strvec
void VECSC(int m, REAL alpha, struct VEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pa[ii+0] *= alpha;
		pa[ii+1] *= alpha;
		pa[ii+2] *= alpha;
		pa[ii+3] *= alpha;
		}
	for(; ii<m; ii++)
		{
		pa[ii+0] *= alpha;
		}
	return;
	}



// copy and scale a strvec into a strvec
void VECCPSC(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	REAL *pa = sa->pa + ai;
	REAL *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] = alpha*pa[ii+0];
		pc[ii+1] = alpha*pa[ii+1];
		pc[ii+2] = alpha*pa[ii+2];
		pc[ii+3] = alpha*pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] = alpha*pa[ii+0];
		}
	return;
	}



// scales and adds a strvec into a strvec
void VECAD(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	REAL *pa = sa->pa + ai;
	REAL *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] += alpha*pa[ii+0];
		pc[ii+1] += alpha*pa[ii+1];
		pc[ii+2] += alpha*pa[ii+2];
		pc[ii+3] += alpha*pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] += alpha*pa[ii+0];
		}
	return;
	}



// add scaled strvec to strvec, sparse formulation
void VECAD_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] += alpha * x[ii];
	return;
	}


// insert scaled strvec to strvec, sparse formulation
void VECIN_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] = alpha * x[ii];
	return;
	}



// extract scaled strvec to strvec, sparse formulation
void VECEX_SP(int m, REAL alpha, int *idx, struct VEC *sx, int xi, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[ii] = alpha * x[idx[ii]];
	return;
	}


// insert element into strvec
void VECIN1(REAL alpha, struct VEC *sx, int xi)
	{
	VECEL(sx, xi) = alpha;
	return;
	}



// extract element from strvec
REAL VECEX1(struct VEC *sx, int xi)
	{
	return VECEL(sx, xi);
	}



// permute elements of a vector struct
void VECPE(int kmax, int *ipiv, struct VEC *sx, int xi)
	{
	int ii;
	REAL tmp;
	REAL *x = sx->pa + xi;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			tmp = x[ipiv[ii]];
			x[ipiv[ii]] = x[ii];
			x[ii] = tmp;
			}
		}
	return;
	}



// inverse permute elements of a vector struct
void VECPEI(int kmax, int *ipiv, struct VEC *sx, int xi)
	{
	int ii;
	REAL tmp;
	REAL *x = sx->pa + xi;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			{
			tmp = x[ipiv[ii]];
			x[ipiv[ii]] = x[ii];
			x[ii] = tmp;
			}
		}
	return;
	}



// clip strvec between two strvec
void VECCL(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi)
	{
	REAL *xm = sxm->pa + xim;
	REAL *x  = sx->pa + xi;
	REAL *xp = sxp->pa + xip;
	REAL *z  = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		{
		if(x[ii]>=xp[ii])
			{
			z[ii] = xp[ii];
			}
		else if(x[ii]<=xm[ii])
			{
			z[ii] = xm[ii];
			}
		else
			{
			z[ii] = x[ii];
			}
		}
	return;
	}



// clip strvec between two strvec, with mask
void VECCL_MASK(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi, struct VEC *sm, int mi)
	{
	REAL *xm = sxm->pa + xim;
	REAL *x  = sx->pa + xi;
	REAL *xp = sxp->pa + xip;
	REAL *z  = sz->pa + zi;
	REAL *mask  = sm->pa + mi;
	int ii;
	for(ii=0; ii<m; ii++)
		{
		if(x[ii]>=xp[ii])
			{
			z[ii] = xp[ii];
			mask[ii] = 1.0;
			}
		else if(x[ii]<=xm[ii])
			{
			z[ii] = xm[ii];
			mask[ii] = -1.0;
			}
		else
			{
			z[ii] = x[ii];
			mask[ii] = 0.0;
			}
		}
	return;
	}


// zero out strvec, with mask
void VECZE(int m, struct VEC *sm, int mi, struct VEC *sv, int vi, struct VEC *se, int ei)
	{
	REAL *mask = sm->pa + mi;
	REAL *v = sv->pa + vi;
	REAL *e = se->pa + ei;
	int ii;
	for(ii=0; ii<m; ii++)
		{
		if(mask[ii]==0)
			{
			e[ii] = v[ii];
			}
		else
			{
			e[ii] = 0;
			}
		}
	return;
	}


// compute inf norm of strvec
void VECNRM_INF(int m, struct VEC *sx, int xi, REAL *ptr_norm)
	{
	int ii;
	REAL *x = sx->pa + xi;
	REAL norm = 0.0;
	REAL tmp;
	for(ii=0; ii<m; ii++)
		{
#ifdef USE_C99_MATH
		norm = fmax(norm, fabs(x[ii]));
#else
		tmp = fabs(x[ii]);
		norm = tmp>norm ? tmp : norm;
#endif
		}
	*ptr_norm = norm;
	return;
	}



// insert a vector into diagonal
void DIAIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj+ii) = alpha*x[ii];
	return;
	}



// insert a strvec to the diagonal of a strmat, sparse formulation
void DIAIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		XMATEL_A(aai+ii, aaj+ii) = alpha * x[jj];
		}
	return;
	}



// extract a vector from diagonal
void DIAEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*XMATEL_A(aai+ii, aaj+ii);
	return;
	}



// extract the diagonal of a strmat from a strvec, sparse formulation
void DIAEX_SP(int kmax, REAL alpha, int *idx, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	REAL *x = sx->pa + xi;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		x[jj] = alpha * XMATEL_A(aai+ii, aaj+ii);
		}
	return;
	}



// add a vector to diagonal
void DIAAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj+ii) += alpha*x[ii];
	return;
	}



// add scaled strvec to another strvec and add to diagonal of strmat, sparse formulation
void DIAAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		XMATEL_A(aai+ii, aaj+ii) += alpha * x[jj];
		}
	return;
	}



// add scaled strvec to another strvec and insert to diagonal of strmat, sparse formulation
void DIAADIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, struct VEC *sy, int yi, int *idx, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		XMATEL_A(aai+ii, aaj+ii) = y[jj] + alpha * x[jj];
		}
	return;
	}



// add scalar to diagonal
void DIARE(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
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
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj+ii) += alpha;
	return;
	}



// extract a row into a vector
void ROWEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*XMATEL_A(aai, aaj+ii);
	return;
	}



// insert a vector into a row
void ROWIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai, aaj+ii) = alpha*x[ii];
	return;
	}



// add a vector to a row
void ROWAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai, aaj+ii) += alpha*x[ii];
	return;
	}



// add scaled strvec to row of strmat, sparse formulation
void ROWAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	REAL *x = sx->pa + xi;
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		XMATEL_A(aai, aaj+ii) += alpha * x[jj];
		}
	return;
	}


// swap two rows of two matrix structs
void ROWSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
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
	int ii;
	REAL tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = XMATEL_A(aai, aaj+ii);
		XMATEL_A(aai, aaj+ii) = XMATEL_B(bbi, bbj+ii);
		XMATEL_B(bbi, bbj+ii) = tmp;
		}
	return;
	}



// permute the rows of a matrix struct
void ROWPE(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			ROWSW(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// inverse permute the rows of a matrix struct
void ROWPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			ROWSW(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// extract vector from column
void COLEX(int kmax, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
#if defined(MF_COLMAJ)
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	const int aai=0; const int aaj=0;
#else
	int aai=ai; int aaj=aj;
#endif
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = XMATEL_A(aai+ii, aaj);
	return;
	}



// insert a vector into a calumn
void COLIN(int kmax, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj) = x[ii];
	return;
	}



// add a scaled vector to a calumn
void COLAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj) += alpha*x[ii];
	return;
	}



// scale a column
void COLSC(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
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
	int ii;
	for(ii=0; ii<kmax; ii++)
		XMATEL_A(aai+ii, aaj) *= alpha;
	return;
	}



// swap two cols of two matrix structs
void COLSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
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
	int ii;
	REAL tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = XMATEL_A(aai+ii, aaj);
		XMATEL_A(aai+ii, aaj) = XMATEL_B(bbi+ii, bbj);
		XMATEL_B(bbi+ii, bbj) = tmp;
		}
	return;
	}



// permute the cols of a matrix struct
void COLPE(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			COLSW(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// inverse permute the cols of a matrix struct
void COLPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			COLSW(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// 1 norm, lower triangular, non-unit
#if 0
REAL dtrcon_1ln_libstr(int n, struct MAT *sA, int ai, int aj, REAL *work, int *iwork)
	{
	if(n<=0)
		return 1.0;
	int ii, jj;
	int lda = sA.m;
	REAL *pA = sA->pA + ai + aj*lda;
	REAL a00, sum;
	REAL rcond = 0.0;
	// compute norm 1 of A
	REAL anorm = 0.0;
	for(jj=0; jj<n; jj++)
		{
		sum = 0.0;
		for(ii=jj; ii<n; ii++)
			{
			sum += abs(pA[ii+lda*jj]);
			}
		anorm = sum>anorm ? sum : anorm;
		}
	if(anorm>0)
		{
		// estimate norm 1 of inv(A)
		ainorm = 0.0;
		}
	return rcond;
	}
#endif




