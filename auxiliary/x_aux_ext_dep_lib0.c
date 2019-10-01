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


#if defined(LA_EXTERNAL_BLAS_WRAPPER) | defined(LA_REFERENCE) | defined(TESTING_MODE)



// create a matrix structure for a matrix of size m*n
void ALLOCATE_STRMAT(int m, int n, struct STRMAT *sA)
	{
	sA->m = m;
	sA->n = n;
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
#if defined(LA_EXTERNAL_BLAS_WRAPPER)
	ZEROS_ALIGN(&(sA->pA), sA->m, sA->n);
	ZEROS_ALIGN(&(sA->dA), tmp, 1);
#else
	ZEROS(&(sA->pA), sA->m, sA->n);
	ZEROS(&(sA->dA), tmp, 1);
#endif
	sA->memsize = (m*n+tmp)*sizeof(REAL);
	sA->use_dA = 0;
	return;
	}



// free memory of a matrix structure
void FREE_STRMAT(struct STRMAT *sA)
	{
#if defined(LA_EXTERNAL_BLAS_WRAPPER)
	FREE_ALIGN(sA->pA);
	FREE_ALIGN(sA->dA);
#else
	FREE(sA->pA);
	FREE(sA->dA);
#endif
	return;
	}



// create a vector structure for a vector of size m
void ALLOCATE_STRVEC(int m, struct STRVEC *sa)
	{
	sa->m = m;
#if defined(LA_EXTERNAL_BLAS_WRAPPER)
	ZEROS_ALIGN(&(sa->pa), sa->m, 1);
#else
	ZEROS(&(sa->pa), sa->m, 1);
#endif
	sa->memsize = m*sizeof(REAL);
	return;
	}



// free memory of a vector structure
void FREE_STRVEC(struct STRVEC *sa)
	{
#if defined(LA_EXTERNAL_BLAS_WRAPPER)
	FREE_ALIGN(sa->pa);
#else
	FREE(sa->pa);
#endif
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
void PRINT_TO_FILE_EXP_STRMAT(FILE *file, int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_TO_FILE_EXP_MAT(file, m, n, pA, lda);
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
void PRINT_EXP_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	int lda = sA->m;
	REAL *pA = sA->pA + ai + aj*lda;
	PRINT_EXP_MAT(m, n, pA, lda);
	return;
	}



// print a vector structure
void PRINT_EXP_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_EXP_MAT(m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void PRINT_EXP_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_EXP_MAT(1, m, pa, 1);
	return;
	}


#endif
