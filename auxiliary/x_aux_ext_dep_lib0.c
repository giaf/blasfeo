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


#if defined(LA_BLAS_WRAPPER) | defined(LA_REFERENCE) | defined(TESTING_MODE)



// create a matrix structure for a matrix of size m*n
void ALLOCATE_STRMAT(int m, int n, struct STRMAT *sA)
	{
	sA->m = m;
	sA->n = n;
	ZEROS(&(sA->pA), sA->m, sA->n);
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	ZEROS(&(sA->dA), tmp, 1);
	sA->memsize = (m*n+tmp)*sizeof(REAL);
	return;
	}



// free memory of a matrix structure
void FREE_STRMAT(struct STRMAT *sA)
	{
	free(sA->pA);
	free(sA->dA);
	return;
	}



// create a vector structure for a vector of size m
void ALLOCATE_STRVEC(int m, struct STRVEC *sa)
	{
	sa->m = m;
	ZEROS(&(sa->pa), sa->m, 1);
	sa->memsize = m*sizeof(REAL);
	return;
	}



// free memory of a vector structure
void FREE_STRVEC(struct STRVEC *sa)
	{
	free(sa->pa);
	return;
	}



// print a matrix structure
void PRINT_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_MAT(m, n, pA, lda);
	return;
	}



// print a vector structure
void PRINT_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_MAT(m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void PRINT_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_MAT(1, m, pa, 1);
	return;
	}



// print a matrix structure
void PRINT_TO_FILE_STRMAT(FILE *file, int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_TO_FILE_MAT(file, m, n, pA, lda);
	return;
	}



// print a matrix structure
void PRINT_TO_STRING_STRMAT(char **out_buf, int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_TO_STRING_MAT(out_buf, m, n, pA, lda);
	return;
	}



// print a vector structure
void PRINT_TO_FILE_STRVEC(FILE *file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, m, 1, pa, m);
	return;
	}


// print a vector structure
void PRINT_TO_STRING_STRVEC(char **out_buf, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_STRING_MAT(out_buf, m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void PRINT_TO_FILE_TRAN_STRVEC(FILE *file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, 1, m, pa, 1);
	return;
	}



// print and transpose a vector structure
void PRINT_TO_STRING_TRAN_STRVEC(char **buf_out, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_STRING_MAT(buf_out, 1, m, pa, 1);
	return;
	}


// print a matrix structure
void PRINT_E_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_E_MAT(m, n, pA, lda);
	return;
	}



// print a vector structure
void PRINT_E_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_E_MAT(m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void PRINT_E_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_E_MAT(1, m, pa, 1);
	return;
	}


#endif


