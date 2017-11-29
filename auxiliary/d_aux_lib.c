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


#define REAL double
#define STRMAT d_strmat
#define STRVEC d_strvec


#if defined(LA_REFERENCE) | defined(LA_BLAS)


#define SIZE_STRMAT d_size_strmat
#define SIZE_DIAG_STRMAT d_size_diag_strmat
#define SIZE_STRVEC d_size_strvec

#define CREATE_STRMAT d_create_strmat
#define CREATE_STRVEC d_create_strvec

#define CVT_MAT2STRMAT d_cvt_mat2strmat
#define CVT_TRAN_MAT2STRMAT d_cvt_tran_mat2strmat
#define CVT_VEC2STRVEC d_cvt_vec2strvec
#define CVT_STRMAT2MAT d_cvt_strmat2mat
#define CVT_TRAN_STRMAT2MAT d_cvt_tran_strmat2mat
#define CVT_STRVEC2VEC d_cvt_strvec2vec

#define CAST_MAT2STRMAT d_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT d_cast_diag_mat2strmat
#define CAST_VEC2VECMAT d_cast_vec2vecmat


#define GECP_LIBSTR dgecp_libstr
#define GESC_LIBSTR dgesc_libstr
#define GECPSC_LIBSTR dgecpsc_libstr



// insert element into strmat
void dgein1_libstr(double a, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	pA[0] = a;
	return;
	}



// extract element from strmat
double dgeex1_libstr(struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	return pA[0];
	}


// insert element into strvec
void dvecin1_libstr(double a, struct d_strvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	x[0] = a;
	return;
	}



// extract element from strvec
double dvecex1_libstr(struct d_strvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	return x[0];
	}



// set all elements of a strmat to a value
void dgese_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
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



// set all elements of a strvec to a value
void dvecse_libstr(int m, double alpha, struct d_strvec *sx, int xi)
	{
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<m; ii++)
		x[ii] = alpha;
	return;
	}



// insert a vector into diagonal
void ddiain_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] = alpha*x[ii];
	return;
	}



// add scalar to diagonal
void ddiare_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] += alpha;
	return;
	}



// extract a row into a vector
void drowex_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*pA[ii*lda];
	return;
	}



// insert a vector  into a row
void drowin_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*lda] = alpha*x[ii];
	return;
	}



// add a vector to a row
void drowad_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*lda] += alpha*x[ii];
	return;
	}



// swap two rows of a matrix struct
void drowsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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
void drowpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			drowsw_libstr(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// extract vector from column
void dcolex_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = pA[ii];
	return;
	}



// insert a vector  into a rcol
void dcolin_libstr(int kmax, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii] = x[ii];
	return;
	}



// swap two cols of a matrix struct
void dcolsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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
void dcolpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			dcolsw_libstr(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// --- vector

// copy a strvec into a strvec
void dveccp_libstr(int m, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci)
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
void dvecsc_libstr(int m, double alpha, struct d_strvec *sa, int ai)
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



// copy a lower triangular strmat into a lower triangular strmat
void dtrcp_l_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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



// scale and add a generic strmat into a generic strmat
void dgead_libstr(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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



// scales and adds a strvec into a strvec
void dvecad_libstr(int m, double alpha, struct d_strvec *sa, int ai, struct d_strvec *sc, int ci)
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



// copy and transpose a generic strmat into a generic strmat
void dgetr_libstr(int m, int n, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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
void dtrtr_l_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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
void dtrtr_u_libstr(int m, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
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



// insert a strvec to the diagonal of a strmat, sparse formulation
void ddiain_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj)
	{
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
void ddiaex_libstr(int kmax, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strvec *sx, int xi)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		x[ii] = alpha*pA[ii*(lda+1)];
	return;
	}



// extract the diagonal of a strmat from a strvec , sparse formulation
void ddiaex_sp_libstr(int kmax, double alpha, int *idx, struct d_strmat *sD, int di, int dj, struct d_strvec *sx, int xi)
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
void ddiaad_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	double *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<kmax; ii++)
		pA[ii*(lda+1)] += alpha*x[ii];
	return;
	}



// add scaled strvec to another strvec and insert to diagonal of strmat, sparse formulation
void ddiaad_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj)
	{
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
void ddiaadin_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi, int *idx, struct d_strmat *sD, int di, int dj)
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
void drowad_sp_libstr(int kmax, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strmat *sD, int di, int dj)
	{
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




void dvecad_sp_libstr(int m, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] += alpha * x[ii];
	return;
	}



void dvecin_sp_libstr(int m, double alpha, struct d_strvec *sx, int xi, int *idx, struct d_strvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] = alpha * x[ii];
	return;
	}



void dvecex_sp_libstr(int m, double alpha, int *idx, struct d_strvec *sx, int xi, struct d_strvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[ii] = alpha * x[idx[ii]];
	return;
	}


// clip without mask return
void dveccl_libstr(int m, struct d_strvec *sxm, int xim, struct d_strvec *sx, int xi, struct d_strvec *sxp, int xip, struct d_strvec *sz, int zi)
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



// clip with mask return
void dveccl_mask_libstr(int m, struct d_strvec *sxm, int xim, struct d_strvec *sx, int xi, struct d_strvec *sxp, int xip, struct d_strvec *sz, int zi, struct d_strvec *sm, int mi)
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


// zero out components using mask
void dvecze_libstr(int m, struct d_strvec *sm, int mi, struct d_strvec *sv, int vi, struct d_strvec *se, int ei)
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



void dvecnrm_inf_libstr(int m, struct d_strvec *sx, int xi, double *ptr_norm)
	{
	int ii;
	double *x = sx->pa + xi;
	double norm = 0.0;
	for(ii=0; ii<m; ii++)
		norm = fmax(norm, fabs(x[ii]));
	*ptr_norm = norm;
	return;
	}



// permute elements of a vector struct
void dvecpe_libstr(int kmax, int *ipiv, struct d_strvec *sx, int xi)
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
void dvecpei_libstr(int kmax, int *ipiv, struct d_strvec *sx, int xi)
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
double dtrcon_1ln_libstr(int n, struct d_strmat *sA, int ai, int aj, double *work, int *iwork)
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


#elif defined(TESTING)

#define SIZE_STRMAT test_d_size_strmat
#define SIZE_DIAG_STRMAT test_d_size_diag_strmat
#define SIZE_STRVEC test_d_size_strvec

#define CREATE_STRMAT test_d_create_strmat
#define CREATE_STRVEC test_d_create_strvec

#define CVT_MAT2STRMAT test_d_cvt_mat2strmat
#define CVT_TRAN_MAT2STRMAT test_d_cvt_tran_mat2strmat
#define CVT_VEC2STRVEC test_d_cvt_vec2strvec
#define CVT_STRMAT2MAT test_d_cvt_strmat2mat
#define CVT_TRAN_STRMAT2MAT test_d_cvt_tran_strmat2mat
#define CVT_STRVEC2VEC test_d_cvt_strvec2vec
#define CAST_MAT2STRMAT test_d_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT test_d_cast_diag_mat2strmat
#define CAST_VEC2VECMAT test_d_cast_vec2vecmat

#define GECP_LIBSTR test_dgecp_libstr
#define GESC_LIBSTR test_dgesc_libstr
#define GECPSC_LIBSTR test_dgecpsc_libstr

#else

#error : wrong LA choice

#endif

// TESTING | LA_REFERENCE | LA_BLAS
#include "x_aux_lib.c"
