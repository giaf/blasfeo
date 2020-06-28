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

#include <blasfeo_common.h>



#if defined(MF_COLMAJ)
	#define XMATEL_A(X, Y) pA[(X)+lda*(Y)]
	#define XMATEL_B(X, Y) pB[(X)+ldb*(Y)]
#else // MF_PANELMAJ
	#define XMATEL_A(X, Y) MATEL(sA, X, Y)
	#define XMATEL_B(X, Y) MATEL(sB, X, Y)
	#define PS D_PS
	#define NC D_NC
#endif



#define REAL double
#define MAT blasfeo_dmat
#define MATEL BLASFEO_DMATEL
#define VEC blasfeo_dvec
#define VECEL BLASFEO_DVECEL



#define HP_MEMSIZE_MAT blasfeo_hp_memsize_dmat
#define HP_MEMSIZE_DIAG_MAT blasfeo_hp_memsize_diag_dmat
#define HP_MEMSIZE_VEC blasfeo_hp_memsize_dvec
#define HP_CREATE_MAT blasfeo_hp_create_dmat
#define HP_CREATE_VEC blasfeo_hp_create_dvec
#define HP_PACK_MAT blasfeo_hp_pack_dmat
#define HP_PACK_L_MAT blasfeo_hp_pack_l_dmat
#define HP_PACK_U_MAT blasfeo_hp_pack_u_dmat
#define HP_PACK_TRAN_MAT blasfeo_hp_pack_tran_dmat
#define HP_PACK_VEC blasfeo_hp_pack_dvec
#define HP_UNPACK_MAT blasfeo_hp_unpack_dmat
#define HP_UNPACK_TRAN_MAT blasfeo_hp_unpack_tran_dmat
#define HP_UNPACK_VEC blasfeo_hp_unpack_dvec
#define HP_CAST_MAT2STRMAT ref_d_cast_mat2strmat
#define HP_CAST_DIAG_MAT2STRMAT ref_d_cast_diag_mat2strmat
#define HP_CAST_VEC2VECMAT ref_d_cast_vec2vecmat
#define HP_GECP blasfeo_hp_dgecp
#define HP_GESC blasfeo_hp_dgesc
#define HP_GECPSC blasfeo_hp_dgecpsc
#define HP_GEAD blasfeo_hp_dgead
#define HP_GESE blasfeo_hp_dgese
#define HP_GETR blasfeo_hp_dgetr
#define HP_GEIN1 blasfeo_hp_dgein1
#define HP_GEEX1 blasfeo_hp_dgeex1
#define HP_TRCP_L blasfeo_hp_dtrcp_l
#define HP_TRTR_L blasfeo_hp_dtrtr_l
#define HP_TRTR_U blasfeo_hp_dtrtr_u
#define HP_VECSE blasfeo_hp_dvecse
#define HP_VECCP blasfeo_hp_dveccp
#define HP_VECSC blasfeo_hp_dvecsc
#define HP_VECCPSC blasfeo_hp_dveccpsc
#define HP_VECAD blasfeo_hp_dvecad
#define HP_VECAD_SP blasfeo_hp_dvecad_sp
#define HP_VECIN_SP blasfeo_hp_dvecin_sp
#define HP_VECEX_SP blasfeo_hp_dvecex_sp
#define HP_VECIN1 blasfeo_hp_dvecin1
#define HP_VECEX1 blasfeo_hp_dvecex1
#define HP_VECPE blasfeo_hp_dvecpe
#define HP_VECPEI blasfeo_hp_dvecpei
#define HP_VECCL blasfeo_hp_dveccl
#define HP_VECCL_MASK blasfeo_hp_dveccl_mask
#define HP_VECZE blasfeo_hp_dvecze
#define HP_VECNRM_INF blasfeo_hp_dvecnrm_inf
#define HP_DIAIN blasfeo_hp_ddiain
#define HP_DIAIN_SP blasfeo_hp_ddiain_sp
#define HP_DIAEX blasfeo_hp_ddiaex
#define HP_DIAEX_SP blasfeo_hp_ddiaex_sp
#define HP_DIAAD blasfeo_hp_ddiaad
#define HP_DIAAD_SP blasfeo_hp_ddiaad_sp
#define HP_DIAADIN_SP blasfeo_hp_ddiaadin_sp
#define HP_DIARE blasfeo_hp_ddiare
#define HP_ROWEX blasfeo_hp_drowex
#define HP_ROWIN blasfeo_hp_drowin
#define HP_ROWAD blasfeo_hp_drowad
#define HP_ROWAD_SP blasfeo_hp_drowad_sp
#define HP_ROWSW blasfeo_hp_drowsw
#define HP_ROWPE blasfeo_hp_drowpe
#define HP_ROWPEI blasfeo_hp_drowpei
#define HP_COLEX blasfeo_hp_dcolex
#define HP_COLIN blasfeo_hp_dcolin
#define HP_COLAD blasfeo_hp_dcolad
#define HP_COLSC blasfeo_hp_dcolsc
#define HP_COLSW blasfeo_hp_dcolsw
#define HP_COLPE blasfeo_hp_dcolpe
#define HP_COLPEI blasfeo_hp_dcolpei

#define MEMSIZE_MAT blasfeo_memsize_dmat
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_dmat
#define MEMSIZE_VEC blasfeo_memsize_dvec
#define CREATE_MAT blasfeo_create_dmat
#define CREATE_VEC blasfeo_create_dvec
#define PACK_MAT blasfeo_pack_dmat
#define PACK_L_MAT blasfeo_pack_l_dmat
#define PACK_U_MAT blasfeo_pack_u_dmat
#define PACK_TRAN_MAT blasfeo_pack_tran_dmat
#define PACK_VEC blasfeo_pack_dvec
#define UNPACK_MAT blasfeo_unpack_dmat
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_dmat
#define UNPACK_VEC blasfeo_unpack_dvec
#define CAST_MAT2STRMAT d_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT d_cast_diag_mat2strmat
#define CAST_VEC2VECMAT d_cast_vec2vecmat
#define GECP blasfeo_dgecp
#define GESC blasfeo_dgesc
#define GECPSC blasfeo_dgecpsc
#define GEAD blasfeo_dgead
#define GESE blasfeo_dgese
#define GETR blasfeo_dgetr
#define GEIN1 blasfeo_dgein1
#define GEEX1 blasfeo_dgeex1
#define TRCP_L blasfeo_dtrcp_l
#define TRTR_L blasfeo_dtrtr_l
#define TRTR_U blasfeo_dtrtr_u
#define VECSE blasfeo_dvecse
#define VECCP blasfeo_dveccp
#define VECSC blasfeo_dvecsc
#define VECCPSC blasfeo_dveccpsc
#define VECAD blasfeo_dvecad
#define VECAD_SP blasfeo_dvecad_sp
#define VECIN_SP blasfeo_dvecin_sp
#define VECEX_SP blasfeo_dvecex_sp
#define VECIN1 blasfeo_dvecin1
#define VECEX1 blasfeo_dvecex1
#define VECPE blasfeo_dvecpe
#define VECPEI blasfeo_dvecpei
#define VECCL blasfeo_dveccl
#define VECCL_MASK blasfeo_dveccl_mask
#define VECZE blasfeo_dvecze
#define VECNRM_INF blasfeo_dvecnrm_inf
#define DIAIN blasfeo_ddiain
#define DIAIN_SP blasfeo_ddiain_sp
#define DIAEX blasfeo_ddiaex
#define DIAEX_SP blasfeo_ddiaex_sp
#define DIAAD blasfeo_ddiaad
#define DIAAD_SP blasfeo_ddiaad_sp
#define DIAADIN_SP blasfeo_ddiaadin_sp
#define DIARE blasfeo_ddiare
#define ROWEX blasfeo_drowex
#define ROWIN blasfeo_drowin
#define ROWAD blasfeo_drowad
#define ROWAD_SP blasfeo_drowad_sp
#define ROWSW blasfeo_drowsw
#define ROWPE blasfeo_drowpe
#define ROWPEI blasfeo_drowpei
#define COLEX blasfeo_dcolex
#define COLIN blasfeo_dcolin
#define COLAD blasfeo_dcolad
#define COLSC blasfeo_dcolsc
#define COLSW blasfeo_dcolsw
#define COLPE blasfeo_dcolpe
#define COLPEI blasfeo_dcolpei



// LA_REFERENCE | LA_EXTERNAL_BLAS_WRAPPER
//#include "x_aux_ref.c"


// return memory size (in bytes) needed for a strmat
size_t HP_MEMSIZE_MAT(int m, int n)
	{
#if defined(MF_COLMAJ)
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	size_t size = (m*n+tmp)*sizeof(REAL);
#else // MF_PANELMAJ
	const int bs = PS;
	const int nc = NC;
	const int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	size_t size = (pm*cn+tmp)*sizeof(REAL);
#endif
	return size;
	}



// return memory size (in bytes) needed for the diagonal of a strmat
size_t HP_MEMSIZE_DIAG_MAT(int m, int n)
	{
#if defined(MF_COLMAJ)
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	size_t size = tmp*sizeof(REAL);
#else // MF_PANELMAJ
	const int bs = PS;
	const int nc = NC;
	const int al = bs*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	size_t size = tmp*sizeof(REAL);
#endif
	return size;
	}



// return memory size (in bytes) needed for a strvec
size_t HP_MEMSIZE_VEC(int m)
	{
#if defined(MF_COLMAJ)
	size_t size = m*sizeof(REAL);
#else // MF_PANELMAJ
	const int bs = PS;
//	const int nc = NC;
//	const int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	size_t size = pm*sizeof(REAL);
#endif
	return size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void HP_CREATE_MAT(int m, int n, struct MAT *sA, void *memory)
	{
	sA->mem = memory;
	sA->m = m;
	sA->n = n;
	sA->use_dA = 0; // invalidate stored inverse diagonal
#if defined(MF_COLMAJ)
	REAL *ptr = (REAL *) memory;
	sA->pA = ptr;
	ptr += m*n;
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->use_dA = 0;
	sA->memsize = (m*n+tmp)*sizeof(REAL);
#else // MF_PANELMAJ
	const int bs = PS; // 4
	const int nc = NC;
	const int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	REAL *ptr = (REAL *) memory;
	sA->pA = ptr;
	ptr += pm*cn;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->memsize = (pm*cn+tmp)*sizeof(REAL);
#endif
	return;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void HP_CREATE_VEC(int m, struct VEC *sa, void *memory)
	{
	sa->mem = memory;
	sa->m = m;
#if defined(MF_COLMAJ)
	REAL *ptr = (REAL *) memory;
	sa->pa = ptr;
	sa->memsize = m*sizeof(REAL);
#else // MF_PANELMAJ
	const int bs = PS; // 4
//	const int nc = NC;
//	const int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	sa->pm = pm;
	REAL *ptr = (REAL *) memory;
	sa->pa = ptr;
//	ptr += pm;
	sa->memsize = pm*sizeof(REAL);
#endif
	return;
	}



// convert a matrix into a matrix structure
void HP_PACK_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
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



// convert a lower triangualr matrix into a matrix structure
void HP_PACK_L_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
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
		for(ii=jj; ii<m; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert an upper triangualr matrix into a matrix structure
void HP_PACK_U_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
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
		for(ii=0; ii<=jj; ii++)
			{
			XMATEL_B(bbi+ii+0, bbj+jj) = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert and transpose a matrix into a matrix structure
void HP_PACK_TRAN_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
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
void HP_PACK_VEC(int m, REAL *x, int xi, struct VEC *sa, int ai)
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
void HP_UNPACK_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
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
void HP_UNPACK_TRAN_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
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
void HP_UNPACK_VEC(int m, struct VEC *sa, int ai, REAL *x, int xi)
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
void HP_CAST_MAT2STRMAT(REAL *A, struct MAT *sB)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;
	sB->pA = A;
	return;
	}



// cast a matrix into the diagonal of a matrix structure
void HP_CAST_DIAG_MAT2STRMAT(REAL *dA, struct MAT *sB)
	{
	// invalidate stored inverse diagonal
	sB->use_dA = 0;
	sB->dA = dA;
	return;
	}



// cast a vector into a vector structure
void HP_CAST_VEC2VECMAT(REAL *a, struct VEC *sa)
	{
	sa->pa = a;
	return;
	}



// copy a generic strmat into a generic strmat
void HP_GECP(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_GESC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
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
void HP_GECPSC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_GEAD(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_GESE(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
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
void HP_GETR(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_GEIN1(REAL alpha, struct MAT *sA, int ai, int aj)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	MATEL(sA, ai, aj) = alpha;
	return;
	}



// extract element from strmat
REAL HP_GEEX1(struct MAT *sA, int ai, int aj)
	{
	return MATEL(sA, ai, aj);
	}



// copy a lower triangular strmat into a lower triangular strmat
void HP_TRCP_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_TRTR_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_TRTR_U(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_VECSE(int m, REAL alpha, struct VEC *sx, int xi)
	{
	REAL *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<m; ii++)
		x[ii] = alpha;
	return;
	}



// copy a strvec into a strvec
void HP_VECCP(int m, struct VEC *sa, int ai, struct VEC *sc, int ci)
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
void HP_VECSC(int m, REAL alpha, struct VEC *sa, int ai)
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
void HP_VECCPSC(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
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
void HP_VECAD(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
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
void HP_VECAD_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] += alpha * x[ii];
	return;
	}


// insert scaled strvec to strvec, sparse formulation
void HP_VECIN_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] = alpha * x[ii];
	return;
	}



// extract scaled strvec to strvec, sparse formulation
void HP_VECEX_SP(int m, REAL alpha, int *idx, struct VEC *sx, int xi, struct VEC *sz, int zi)
	{
	REAL *x = sx->pa + xi;
	REAL *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[ii] = alpha * x[idx[ii]];
	return;
	}


// insert element into strvec
void HP_VECIN1(REAL alpha, struct VEC *sx, int xi)
	{
	VECEL(sx, xi) = alpha;
	return;
	}



// extract element from strvec
REAL HP_VECEX1(struct VEC *sx, int xi)
	{
	return VECEL(sx, xi);
	}



// permute elements of a vector struct
void HP_VECPE(int kmax, int *ipiv, struct VEC *sx, int xi)
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
void HP_VECPEI(int kmax, int *ipiv, struct VEC *sx, int xi)
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
void HP_VECCL(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi)
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
void HP_VECCL_MASK(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi, struct VEC *sm, int mi)
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
void HP_VECZE(int m, struct VEC *sm, int mi, struct VEC *sv, int vi, struct VEC *se, int ei)
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
void HP_VECNRM_INF(int m, struct VEC *sx, int xi, REAL *ptr_norm)
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
void HP_DIAIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_DIAIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
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
void HP_DIAEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
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
void HP_DIAEX_SP(int kmax, REAL alpha, int *idx, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
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
void HP_DIAAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_DIAAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
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
void HP_DIAADIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, struct VEC *sy, int yi, int *idx, struct MAT *sA, int ai, int aj)
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
void HP_DIARE(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
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
void HP_ROWEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
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
void HP_ROWIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_ROWAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_ROWAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
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
void HP_ROWSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_ROWPE(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			HP_ROWSW(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// inverse permute the rows of a matrix struct
void HP_ROWPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			HP_ROWSW(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// extract vector from column
void HP_COLEX(int kmax, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
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
void HP_COLIN(int kmax, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_COLAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
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
void HP_COLSC(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
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
void HP_COLSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
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
void HP_COLPE(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			HP_COLSW(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// inverse permute the cols of a matrix struct
void HP_COLPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	// invalidate stored inverse diagonal
	sA->use_dA = 0;
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			HP_COLSW(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
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



#if defined(LA_HIGH_PERFORMANCE)



size_t MEMSIZE_MAT(int m, int n)
	{
	return HP_MEMSIZE_MAT(m, n);
	}



size_t MEMSIZE_DIAG_MAT(int m, int n)
	{
	return HP_MEMSIZE_DIAG_MAT(m, n);
	}



size_t MEMSIZE_VEC(int m)
	{
	return HP_MEMSIZE_VEC(m);
	}



void CREATE_MAT(int m, int n, struct MAT *sA, void *memory)
	{
	HP_CREATE_MAT(m, n, sA, memory);
	}



void CREATE_VEC(int m, struct VEC *sa, void *memory)
	{
	HP_CREATE_VEC(m, sa, memory);
	}



void PACK_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	HP_PACK_MAT(m, n, A, lda, sB, bi, bj);
	}



void PACK_L_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	HP_PACK_L_MAT(m, n, A, lda, sB, bi, bj);
	}



void PACK_U_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	HP_PACK_U_MAT(m, n, A, lda, sB, bi, bj);
	}



void PACK_TRAN_MAT(int m, int n, REAL *A, int lda, struct MAT *sB, int bi, int bj)
	{
	HP_PACK_TRAN_MAT(m, n, A, lda, sB, bi, bj);
	}



void PACK_VEC(int m, REAL *x, int xi, struct VEC *sa, int ai)
	{
	HP_PACK_VEC(m, x, xi, sa, ai);
	}



void UNPACK_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
	{
	HP_UNPACK_MAT(m, n, sA, ai, aj, B, ldb);
	}



void UNPACK_TRAN_MAT(int m, int n, struct MAT *sA, int ai, int aj, REAL *B, int ldb)
	{
	HP_UNPACK_TRAN_MAT(m, n, sA, ai, aj, B, ldb);
	}



void UNPACK_VEC(int m, struct VEC *sa, int ai, REAL *x, int xi)
	{
	HP_UNPACK_VEC(m, sa, ai, x, xi);
	}



void CAST_MAT2STRMAT(REAL *A, struct MAT *sB)
	{
	HP_CAST_MAT2STRMAT(A, sB);
	}



void CAST_DIAG_MAT2STRMAT(REAL *dA, struct MAT *sB)
	{
	HP_CAST_DIAG_MAT2STRMAT(dA, sB);
	}



void CAST_VEC2VECMAT(REAL *a, struct VEC *sa)
	{
	HP_CAST_VEC2VECMAT(a, sa);
	}



void GECP(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_GECP(m, n, sA, ai, aj, sB, bi, bj);
	}



void GESC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	HP_GESC(m, n, alpha, sA, ai, aj);
	}



void GECPSC(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_GECPSC(m, n, alpha, sA, ai, aj, sB, bi, bj);
	}



void GEAD(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_GEAD(m, n, alpha, sA, ai, aj, sB, bi, bj);
	}



void GESE(int m, int n, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	HP_GESE(m, n, alpha, sA, ai, aj);
	}



void GETR(int m, int n, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_GETR(m, n, sA, ai, aj, sB, bi, bj);
	}



void GEIN1(REAL alpha, struct MAT *sA, int ai, int aj)
	{
	HP_GEIN1(alpha, sA, ai, aj);
	}



REAL GEEX1(struct MAT *sA, int ai, int aj)
	{
	return HP_GEEX1(sA, ai, aj);
	}



void TRCP_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	TRCP_L(m, sA, ai, aj, sB, bi, bj);
	}



void TRTR_L(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	TRTR_L(m, sA, ai, aj, sB, bi, bj);
	}



void TRTR_U(int m, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	TRTR_U(m, sA, ai, aj, sB, bi, bj);
	}



void VECSE(int m, REAL alpha, struct VEC *sx, int xi)
	{
	HP_VECSE(m, alpha, sx, xi);
	}



void VECCP(int m, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	HP_VECCP(m, sa, ai, sc, ci);
	}



void VECSC(int m, REAL alpha, struct VEC *sa, int ai)
	{
	HP_VECSC(m, alpha, sa, ai);
	}



void VECCPSC(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	HP_VECCPSC(m, alpha, sa, ai, sc, ci);
	}



void VECAD(int m, REAL alpha, struct VEC *sa, int ai, struct VEC *sc, int ci)
	{
	HP_VECAD(m, alpha, sa, ai, sc, ci);
	}



void VECAD_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	HP_VECAD_SP(m, alpha, sx, xi, idx, sz, zi);
	}



void VECIN_SP(int m, REAL alpha, struct VEC *sx, int xi, int *idx, struct VEC *sz, int zi)
	{
	HP_VECIN_SP(m, alpha, sx, xi, idx, sz, zi);
	}



void VECEX_SP(int m, REAL alpha, int *idx, struct VEC *sx, int xi, struct VEC *sz, int zi)
	{
	HP_VECEX_SP(m, alpha, idx, sx, xi, sz, zi);
	}



void VECIN1(REAL alpha, struct VEC *sx, int xi)
	{
	HP_VECIN1(alpha, sx, xi);
	}



REAL VECEX1(struct VEC *sx, int xi)
	{
	return HP_VECEX1(sx, xi);
	}



void VECPE(int kmax, int *ipiv, struct VEC *sx, int xi)
	{
	HP_VECPE(kmax, ipiv, sx, xi);
	}



void VECPEI(int kmax, int *ipiv, struct VEC *sx, int xi)
	{
	HP_VECPEI(kmax, ipiv, sx, xi);
	}



void VECCL(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi)
	{
	HP_VECCL(m, sxm, xim, sx, xi, sxp, xip, sz, zi);
	}



void VECCL_MASK(int m, struct VEC *sxm, int xim, struct VEC *sx, int xi, struct VEC *sxp, int xip, struct VEC *sz, int zi, struct VEC *sm, int mi)
	{
	HP_VECCL_MASK(m, sxm, xim, sx, xi, sxp, xip, sz, zi, sm, mi);
	}



void VECZE(int m, struct VEC *sm, int mi, struct VEC *sv, int vi, struct VEC *se, int ei)
	{
	HP_VECZE(m, sm, mi, sv, vi, se, ei);
	}



void VECNRM_INF(int m, struct VEC *sx, int xi, REAL *ptr_norm)
	{
	HP_VECNRM_INF(m, sx, xi, ptr_norm);
	}



void DIAIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_DIAIN(kmax, alpha, sx, xi, sA, ai, aj);
	}



void DIAIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
	{
	HP_DIAIN_SP(kmax, alpha, sx, xi, idx, sA, ai, aj);
	}



void DIAEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
	HP_DIAEX(kmax, alpha, sA, ai, aj, sx, xi);
	}



void DIAEX_SP(int kmax, REAL alpha, int *idx, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
	HP_DIAEX_SP(kmax, alpha, idx, sA, ai, aj, sx, xi);
	}



void DIAAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_DIAAD(kmax, alpha, sx, xi, sA, ai, aj);
	}



void DIAAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
	{
	HP_DIAAD_SP(kmax, alpha, sx, xi, idx, sA, ai, aj);
	}



void DIAADIN_SP(int kmax, REAL alpha, struct VEC *sx, int xi, struct VEC *sy, int yi, int *idx, struct MAT *sA, int ai, int aj)
	{
	HP_DIAADIN_SP(kmax, alpha, sx, xi, sy, yi, idx, sA, ai, aj);
	}



void DIARE(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	HP_DIARE(kmax, alpha, sA, ai, aj);
	}



void ROWEX(int kmax, REAL alpha, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
	HP_ROWEX(kmax, alpha, sA, ai, aj, sx, xi);
	}



void ROWIN(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_ROWIN(kmax, alpha, sx, xi, sA, ai, aj);
	}



void ROWAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_ROWAD(kmax, alpha, sx, xi, sA, ai, aj);
	}



void ROWAD_SP(int kmax, REAL alpha, struct VEC *sx, int xi, int *idx, struct MAT *sA, int ai, int aj)
	{
	HP_ROWAD_SP(kmax, alpha, sx, xi, idx, sA, ai, aj);
	}



void ROWSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_ROWSW(kmax, sA, ai, aj, sB, bi, bj);
	}



void ROWPE(int kmax, int *ipiv, struct MAT *sA)
	{
	HP_ROWPE(kmax, ipiv, sA);
	}



void ROWPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	HP_ROWPEI(kmax, ipiv, sA);
	}



void COLEX(int kmax, struct MAT *sA, int ai, int aj, struct VEC *sx, int xi)
	{
	HP_COLEX(kmax, sA, ai, aj, sx, xi);
	}



void COLIN(int kmax, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_COLIN(kmax, sx, xi, sA, ai, aj);
	}



void COLAD(int kmax, REAL alpha, struct VEC *sx, int xi, struct MAT *sA, int ai, int aj)
	{
	HP_COLAD(kmax, alpha, sx, xi, sA, ai, aj);
	}



void COLSC(int kmax, REAL alpha, struct MAT *sA, int ai, int aj)
	{
	HP_COLSC(kmax, alpha, sA, ai, aj);
	}



void COLSW(int kmax, struct MAT *sA, int ai, int aj, struct MAT *sB, int bi, int bj)
	{
	HP_COLSW(kmax, sA, ai, aj, sB, bi, bj);
	}



void COLPE(int kmax, int *ipiv, struct MAT *sA)
	{
	HP_COLPE(kmax, ipiv, sA);
	}



void COLPEI(int kmax, int *ipiv, struct MAT *sA)
	{
	HP_COLPEI(kmax, ipiv, sA);
	}



#endif
