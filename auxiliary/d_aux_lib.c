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

/*
 * auxiliary functions for LA:REFERENCE (column major)
 *
 * auxiliary/d_aux_lib*.c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"



#define REAL double
#define MAT blasfeo_dmat
#define VEC blasfeo_dvec


#if defined(LA_REFERENCE) | defined(LA_EXTERNAL_BLAS_WRAPPER)


#define MEMSIZE_MAT blasfeo_memsize_dmat
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_dmat
#define MEMSIZE_VEC blasfeo_memsize_dvec

#define CREATE_MAT blasfeo_create_dmat
#define CREATE_VEC blasfeo_create_dvec

#define PACK_MAT blasfeo_pack_dmat
#define PACK_TRAN_MAT blasfeo_pack_tran_dmat
#define PACK_VEC blasfeo_pack_dvec
#define UNPACK_MAT blasfeo_unpack_dmat
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_dmat
#define UNPACK_VEC blasfeo_unpack_dvec

#define CAST_MAT2STRMAT d_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT d_cast_diag_mat2strmat
#define CAST_VEC2VECMAT d_cast_vec2vecmat


#define GEAD_LIBSTR blasfeo_dgead
#define GECP_LIBSTR blasfeo_dgecp
#define GECPSC_LIBSTR blasfeo_dgecpsc
#define GESC_LIBSTR blasfeo_dgesc
#define GESE_LIBSTR blasfeo_dgese



// insert element into strmat
void blasfeo_dgein1(double a, struct blasfeo_dmat *sA, int ai, int aj)
	{

	if (ai==aj)
		{
		// invalidate stored inverse diagonal
		sA->use_dA = 0;
		}

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	pA[0] = a;
	return;
	}



// extract element from strmat
double blasfeo_dgeex1(struct blasfeo_dmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	return pA[0];
	}


// insert element into strvec
void blasfeo_dvecin1(double a, struct blasfeo_dvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	x[0] = a;
	return;
	}



// extract element from strvec
double blasfeo_dvecex1(struct blasfeo_dvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	return x[0];
	}



// set all elements of a strvec to a value
void blasfeo_dvecse(int m, double alpha, struct blasfeo_dvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<m; ii++)
		x[ii] = alpha;
	return;
	}



// insert a vector into diagonal
void blasfeo_ddiain(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] = alpha*x[ii];
	return;
	}



// add scalar to diagonal
void blasfeo_ddiare(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] += alpha;
	return;
	}



// extract a row into a vector
void blasfeo_drowex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*pA[ii*lda];
	return;
	}



// insert a vector into a row
void blasfeo_drowin(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*lda] = alpha*x[ii];
	return;
	}



// add a vector to a row
void blasfeo_drowad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*lda] += alpha*x[ii];
	return;
	}



// swap two rows of two matrix structs
void blasfeo_drowsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*lda;
	int ii;
	double tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = pA[ii*lda];
		pA[ii*lda] = pC[ii*ldc];
		pC[ii*ldc] = tmp;
		}
	return;
	}



// permute the rows of a matrix struct
void blasfeo_drowpe(int kmax, int *ipiv, struct blasfeo_dmat *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			blasfeo_drowsw(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// inverse permute the rows of a matrix struct
void blasfeo_drowpei(int kmax, int *ipiv, struct blasfeo_dmat *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			blasfeo_drowsw(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// extract vector from column
void blasfeo_dcolex(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = pA[ii];
	return;
	}



// insert a vector into a calumn
void blasfeo_dcolin(int kmax, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii] = x[ii];
	return;
	}



// scale a column
void blasfeo_dcolsc(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii] *= alpha;
	return;
	}



// swap two cols of two matrix structs
void blasfeo_dcolsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*lda;
	int ii;
	double tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = pA[ii];
		pA[ii] = pC[ii];
		pC[ii] = tmp;
		}
	return;
	}



// permute the cols of a matrix struct
void blasfeo_dcolpe(int kmax, int *ipiv, struct blasfeo_dmat *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			blasfeo_dcolsw(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// inverse permute the cols of a matrix struct
void blasfeo_dcolpei(int kmax, int *ipiv, struct blasfeo_dmat *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			blasfeo_dcolsw(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// copy a lower triangular strmat into a lower triangular strmat
void blasfeo_dtrcp_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<m; jj++)
		{
		ii = jj;
		for(; ii<m; ii++)
			{
			pC[ii+0+jj*ldc] = pA[ii+0+jj*lda];
			}
		}
	return;
	}



// copy and transpose a generic strmat into a generic strmat
void blasfeo_dgetr(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pC[jj+(ii+0)*ldc] = pA[ii+0+jj*lda];
			pC[jj+(ii+1)*ldc] = pA[ii+1+jj*lda];
			pC[jj+(ii+2)*ldc] = pA[ii+2+jj*lda];
			pC[jj+(ii+3)*ldc] = pA[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pC[jj+(ii+0)*ldc] = pA[ii+0+jj*lda];
			}
		}
	return;
	}



// copy and transpose a lower triangular strmat into an upper triangular strmat
void blasfeo_dtrtr_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<m; jj++)
		{
		ii = jj;
		for(; ii<m; ii++)
			{
			pC[jj+(ii+0)*ldc] = pA[ii+0+jj*lda];
			}
		}
	return;
	}



// copy and transpose an upper triangular strmat into a lower triangular strmat
void blasfeo_dtrtr_u(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj)
	{
	// invalidate stored inverse diagonal
	sC->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*ldc;
	int ii, jj;
	for(jj=0; jj<m; jj++)
		{
		ii = 0;
		for(; ii<=jj; ii++)
			{
			pC[jj+(ii+0)*ldc] = pA[ii+0+jj*lda];
			}
		}
	return;
	}



// --- vector

// copy a strvec into a strvec
void blasfeo_dveccp(int m, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci)
	{
	double *pa = sa->pa + ai;
	double *pc = sc->pa + ci;
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
void blasfeo_dvecsc(int m, double alpha, struct blasfeo_dvec *sa, int ai)
	{
	double *pa = sa->pa + ai;
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
void blasfeo_dveccpsc(int m, double alpha, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci)
	{
	double *pa = sa->pa + ai;
	double *pc = sc->pa + ci;
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
void blasfeo_dvecad(int m, double alpha, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci)
	{
	double *pa = sa->pa + ai;
	double *pc = sc->pa + ci;
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



// insert a strvec to the diagonal of a strmat, sparse formulation
void blasfeo_ddiain_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj)
	{
	// invalidate stored inverse diagonal
	sD->use_dA = 0;

	double *x = sx->pa + xi;
	int ldd = sD->m;
	double *pD = sD->pA + di + dj*ldd;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*(ldd+1)] = alpha * x[jj];
		}
	return;
	}



// extract a vector from diagonal
void blasfeo_ddiaex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*pA[ii*(lda+1)];
	return;
	}



// extract the diagonal of a strmat from a strvec, sparse formulation
void blasfeo_ddiaex_sp(int kmax, double alpha, int *idx, struct blasfeo_dmat *sD, int di, int dj, struct blasfeo_dvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	int ldd = sD->m;
	double *pD = sD->pA + di + dj*ldd;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		x[jj] = alpha * pD[ii*(ldd+1)];
		}
	return;
	}



// add a vector to diagonal
void blasfeo_ddiaad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] += alpha*x[ii];
	return;
	}



// add scaled strvec to another strvec and add to diagonal of strmat, sparse formulation
void blasfeo_ddiaad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj)
	{
	// invalidate stored inverse diagonal
	sD->use_dA = 0;

	double *x = sx->pa + xi;
	int ldd = sD->m;
	double *pD = sD->pA + di + dj*ldd;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*(ldd+1)] += alpha * x[jj];
		}
	return;
	}



// add scaled strvec to another strvec and insert to diagonal of strmat, sparse formulation
void blasfeo_ddiaadin_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, int *idx, struct blasfeo_dmat *sD, int di, int dj)
	{
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	int ldd = sD->m;
	double *pD = sD->pA + di + dj*ldd;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*(ldd+1)] = y[jj] + alpha * x[jj];
		}
	return;
	}



// add scaled strvec to row of strmat, sparse formulation
void blasfeo_drowad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj)
	{
	// invalidate stored inverse diagonal
	sD->use_dA = 0;

	double *x = sx->pa + xi;
	int ldd = sD->m;
	double *pD = sD->pA + di + dj*ldd;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*ldd] += alpha * x[jj];
		}
	return;
	}


// add scaled strvec to strvec, sparse formulation
void blasfeo_dvecad_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] += alpha * x[ii];
	return;
	}


// insert scaled strvec to strvec, sparse formulation
void blasfeo_dvecin_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] = alpha * x[ii];
	return;
	}



// extract scaled strvec to strvec, sparse formulation
void blasfeo_dvecex_sp(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[ii] = alpha * x[idx[ii]];
	return;
	}


// clip strvec between two strvec
void blasfeo_dveccl(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi)
	{
	double *xm = sxm->pa + xim;
	double *x  = sx->pa + xi;
	double *xp = sxp->pa + xip;
	double *z  = sz->pa + zi;
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
void blasfeo_dveccl_mask(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi, struct blasfeo_dvec *sm, int mi)
	{
	double *xm = sxm->pa + xim;
	double *x  = sx->pa + xi;
	double *xp = sxp->pa + xip;
	double *z  = sz->pa + zi;
	double *mask  = sm->pa + mi;
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
void blasfeo_dvecze(int m, struct blasfeo_dvec *sm, int mi, struct blasfeo_dvec *sv, int vi, struct blasfeo_dvec *se, int ei)
	{
	double *mask = sm->pa + mi;
	double *v = sv->pa + vi;
	double *e = se->pa + ei;
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
void blasfeo_dvecnrm_inf(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm)
	{
	int ii;
	double *x = sx->pa + xi;
	double norm = 0.0;
	double tmp;
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



// permute elements of a vector struct
void blasfeo_dvecpe(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi)
	{
	int ii;
	double tmp;
	double *x = sx->pa + xi;
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
void blasfeo_dvecpei(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi)
	{
	int ii;
	double tmp;
	double *x = sx->pa + xi;
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



// 1 norm, lower triangular, non-unit
#if 0
double dtrcon_1ln_libstr(int n, struct blasfeo_dmat *sA, int ai, int aj, double *work, int *iwork)
	{
	if(n<=0)
		return 1.0;
	int ii, jj;
	int lda = sA.m;
	double *pA = sA->pA + ai + aj*lda;
	double a00, sum;
	double rcond = 0.0;
	// compute norm 1 of A
	double anorm = 0.0;
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



#else

#error : wrong LA choice

#endif

// LA_REFERENCE | LA_EXTERNAL_BLAS_WRAPPER
#include "x_aux_lib.c"
