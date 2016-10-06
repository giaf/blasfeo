/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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

#include <stdlib.h>
#include <stdio.h>

#include "d_strmat.h"



// linear algebra provided by BLASFEO
#if defined(BLASFEO_LA)



#include "../include/blasfeo_block_size.h"

// create a matrix structure for a matrix of size m*n by dynamically allocating the memory
void d_allocate_strmat(int m, int n, struct d_strmat *sA)
	{
	int bs = D_BS;
	int nc = D_NC;
	int al = bs*nc;
	sA->bs = bs;
	sA->m = m;
	sA->n = n;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	d_zeros_align(&(sA->pA), sA->pm, sA->cn);
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	d_zeros_align(&(sA->dA), tmp, 1);
	sA->use_dA = 0;
	sA->memory_size = (pm*cn+tmp)*sizeof(double);
	return;
	}

// free memory of a matrix structure
void d_free_strmat(struct d_strmat *sA)
	{
	free(sA->pA);
	free(sA->dA);
	return;
	}

// return memory size (in bytes) needed for a strmat
int d_size_strmat(int m, int n)
	{
	int bs = D_BS;
	int nc = D_NC;
	int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	int memory_size = (pm*cn+tmp)*sizeof(double);
	return memory_size;
	}

// create a matrix structure for a matrix of size m*n by using memory passed by a pointer (and update it)
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory)
	{
	int bs = D_BS;
	int nc = D_NC;
	int al = bs*nc;
	sA->bs = bs;
	sA->m = m;
	sA->n = n;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	double *ptr = (double *) memory;
	sA->pA = ptr;
	ptr += pm*cn;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->use_dA = 0;
	sA->memory_size = (pm*cn+tmp)*sizeof(double);
	return;
	}

// convert a matrix into a matrix structure
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_mat2pmat(m, n, A, lda, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn);
	return;
	}

// convert and transpose a matrix into a matrix structure
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_tran_mat2pmat(m, n, A, lda, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn);
	return;
	}

// convert a matrix structure into a matrix
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_pmat2mat(m, n, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn, A, lda);
	return;
	}

// convert and transpose a matrix structure into a matrix
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_tran_pmat2mat(m, n, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn, A, lda);
	return;
	}

// print a matrix structure
void d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj)
	{
	// TODO ai and aj
	if(ai!=0 | aj!=0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	d_print_pmat(m, n, sA->pA, sA->cn);
	return;
	}

// dgemm nt
//void dgemm_nt_libst(int m, int n, int k, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, double beta, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
//	{
//	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
//		{
//		printf("\nfeature not implemented yet\n\n");
//		exit(1);
//		}
//	dgemm_nt_lib(m, n, k, alpha, sA->pA+aj*sA->bs, sA->cn, sB->pA+bj*sB->bs, sB->cn, beta, sC->pA+cj*sC->bs, sC->cn, sD->pA+dj*sD->bs, sD->cn); 
//	return;
//	}

// dtrsm_nn_llu
void dtrsm_llnu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	// TODO alpha
	dtrsm_nn_ll_one_lib(m, n, sA->pA+aj*sA->bs, sA->cn, sB->pA+bj*sB->bs, sB->cn, sD->pA+dj*sD->bs, sD->cn); 
	return;
	}

// dtrsm_nn_lun
void dtrsm_lunn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	// TODO alpha
	dtrsm_nn_lu_inv_lib(m, n, sA->pA+aj*sA->bs, sA->cn, sA->dA, sB->pA+bj*sB->bs, sB->cn, sD->pA+dj*sD->bs, sD->cn); 
	return;
	}

// dtrsm_right_lower_transposed_unit
void dtrsm_rltu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	// TODO alpha
	dtrsm_nt_rl_one_lib(m, n, sA->pA+aj*sA->bs, sA->cn, sB->pA+bj*sB->bs, sB->cn, sD->pA+dj*sD->bs, sD->cn); 
	return;
	}

// dtrsm_right_upper_transposed_notunit
void dtrsm_rutn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	if(ai!=0 | bi!=0 | di!=0 | alpha!=1.0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	// TODO alpha
	dtrsm_nt_ru_inv_lib(m, n, sA->pA+aj*sA->bs, sA->cn, sA->dA, sB->pA+bj*sB->bs, sB->cn, sD->pA+dj*sD->bs, sD->cn); 
	return;
	}

// dpotrf
void dpotrf_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	dpotrf_nt_l_lib(m, n, sC->pA+cj*sC->bs, sD->cn, sD->pA+dj*sC->bs, sD->cn, sD->dA);
	sC->use_dA = 1;
	return;
	}

// dgetrf without pivoting
void dgetrf_nopivot_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	dgetrf_nn_nopivot_lib(m, n, sC->pA+cj*sC->bs, sD->cn, sD->pA+dj*sC->bs, sD->cn, sD->dA);
	sC->use_dA = 1;
	return;
	}




// linear algebra provided by BLAS
#elif defined(BLAS_LA)



// create a matrix structure for a matrix of size m*n // TODO pass work space instead of dynamic alloc
void d_allocate_strmat(int m, int n, struct d_strmat *sA)
	{
	sA->m = m;
	sA->n = n;
	d_zeros(&(sA->pA), sA->m, sA->n);
	sA->memory_size = (m*n)*sizeof(double);
	return;
	}

// free memory of a matrix structure
void d_free_strmat(struct d_strmat *sA)
	{
	free(sA->pA);
	return;
	}

// return memory size (in bytes) needed for a strmat
int d_size_strmat(int m, int n)
	{
	int size = (m*n)*sizeof(double);
	return size;
	}

// create a matrix structure for a matrix of size m*n by using memory passed by a pointer (and update it)
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory)
	{
	sA->m = m;
	sA->n = n;
	double *ptr = (double *) memory;
	sA->pA = ptr;
	ptr += m * n;
	sA->memory_size = (m*n)*sizeof(double);
	return;
	}

// convert a matrix into a matrix structure
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
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
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m; ii++)
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

// convert a matrix structure into a matrix 
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
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
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m; ii++)
			{
			A[ii+0+jj*lda] = pA[jj+(ii+0)*lda2];
			A[ii+1+jj*lda] = pA[jj+(ii+1)*lda2];
			A[ii+2+jj*lda] = pA[jj+(ii+2)*lda2];
			A[ii+3+jj*lda] = pA[jj+(ii+3)*lda2];
			}
		for(; ii<m; ii++)
			{
			A[ii+0+jj*lda] = pA[jj+(ii+0)*lda2];
			}
		}
	return;
	}

// print a matrix structure
void d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj)
	{
	// TODO ai and aj
	if(ai!=0 || aj!=0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	double *pA = sA->pA;
	int lda = sA->m;
	d_print_mat(m, n, pA, lda);
	return;
	}

// dgemm nt
void dgemm_nt_libst(int m, int n, int k, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, double beta, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char ta = 'n';
	char tb = 't';
	int i1 = 1;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(beta==0.0 || pC==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	dgemm_(&ta, &tb, &m, &n, &k, &alpha, pA, &(sA->m), pB, &(sB->m), &beta, pD, &(sD->m));
	return;
	}

// dtrsm_left_lower_nottransposed_unit
void dtrsm_llnu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	int i1 = 1;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pB+jj*sB->m, &i1, pD+jj*sD->m, &i1);
		}
	dtrsm_(&cl, &cl, &cn, &cu, &m, &n, &alpha, pA, &(sA->m), pD, &(sD->m));
	return;
	}

// dtrsm_left_upper_nottransposed_notunit
void dtrsm_lunn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cu = 'u';
	int i1 = 1;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pB+jj*sB->m, &i1, pD+jj*sD->m, &i1);
		}
	dtrsm_(&cl, &cu, &cn, &cn, &m, &n, &alpha, pA, &(sA->m), pD, &(sD->m));
	return;
	}

// dtrsm_right_lower_transposed_unit
void dtrsm_rltu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pB+jj*sB->m, &i1, pD+jj*sD->m, &i1);
		}
	dtrsm_(&cr, &cl, &ct, &cu, &m, &n, &alpha, pA, &(sA->m), pD, &(sD->m));
	return;
	}

// dtrsm_right_upper_transposed_notunit
void dtrsm_rutn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	char cu = 'u';
	int i1 = 1;
	double *pA = sA->pA+ai+aj*sA->m;
	double *pB = sB->pA+bi+bj*sB->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pB==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pB+jj*sB->m, &i1, pD+jj*sD->m, &i1);
		}
	dtrsm_(&cr, &cu, &ct, &cn, &m, &n, &alpha, pA, &(sA->m), pD, &(sD->m));
	return;
	}

// dpotrf
void dpotrf_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	int jj;
	char cl = 'l';
	char cn = 'n';
	char cr = 'r';
	char ct = 't';
	int mmn = m-n;
	int info;
	int i1 = 1;
	double d1 = 1.0;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	dpotrf_(&cl, &n, pD, &(sD->m), &info);
	dtrsm_(&cr, &cl, &ct, &cn, &mmn, &n, &d1, pD, &(sD->m), pD+n, &(sD->m));
	return;
	}

// dgetrf without pivoting
void dgetrf_nopivot_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj)
	{
	// TODO with custom level 2 LAPACK + level 3 BLAS
//	printf("\nfeature not implemented yet\n\n");
//	exit(1);
	int jj;
	int i1 = 1;
	double d1 = 1.0;
	double *pC = sC->pA+ci+cj*sC->m;
	double *pD = sD->pA+di+dj*sD->m;
	if(!(pC==pD))
		{
		for(jj=0; jj<n; jj++)
			dcopy_(&m, pC+jj*sC->m, &i1, pD+jj*sD->m, &i1);
		}
	dgetf2_nopivot(m, n, pD, sD->m);
	return;
	}

#endif


