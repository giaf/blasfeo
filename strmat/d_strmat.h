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


#if defined(BLASFEO_LA)

// matrix structure
struct d_strmat 
	{
	int bs; // heigh of panels
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	double *pA; // pointer to a pm*pn array of doubles, the first is aligned to cache line size
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memory_size; // size of needed memory
	};

#elif defined(BLAS_LA)

// matrix structure
struct d_strmat 
	{
	int m; // rows
	int n; // cols
	double *pA; // pointer to a m*n array of doubles
	int memory_size; // size of needed memory
	};

#endif

void d_allocate_strmat(int m, int n, struct d_strmat *sA);
void d_free_strmat(struct d_strmat *sA);
int d_size_strmat(int m, int n);
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory);
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj);
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda);
void d_print_strmat(int m, int n, struct d_strmat *sA, int ai, int aj);
//void dgemm_nt_libst(int m, int n, int k, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, double beta, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj);
void dtrsm_llnu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj);
void dtrsm_lunn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj);
void dtrsm_rltu_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj);
void dtrsm_rutn_libst(int m, int n, double alpha, struct d_strmat *sA, int ai, int aj, struct d_strmat *sB, int bi, int bj, struct d_strmat *sD, int di, int dj);
void dpotrf_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj);
void dgetrf_nopivot_libst(int m, int n, struct d_strmat *sC, int ci, int cj, struct d_strmat *sD, int di, int dj);
