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



#ifndef BLASFEO_D_BLAS_API_H_
#define BLASFEO_D_BLAS_API_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#ifdef BLAS_API



#ifdef FORTRAN_BLAS_API



// BLAS 1
//
void dcopy_(int *n, double *x, int *incx, double *y, int *incy);

// BLAS 3
//
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void dsyrk_(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);



// LAPACK
//
void dgesv_(int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
//
void dgetrs_(char *trans, int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void dlaswp_(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *incx);
//
void dposv_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
//
void dpotrs_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void dtrtrs_(char *uplo, char *trans, char *diag, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);



#else // BLASFEO_API



// BLAS 1
//
void blasfeo_dcopy(int *n, double *x, int *incx, double *y, int *incy);

// BLAS 3
//
void blasfeo_dgemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void blasfeo_dsyrk(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void blasfeo_dtrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void blasfeo_dtrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);



// LAPACK
//
void blasfeo_dgesv(int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blasfeo_dgetrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
//
void blasfeo_dgetrs(char *trans, int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blasfeo_dlaswp(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *incx);
//
void blasfeo_dposv(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blasfeo_dpotrf(char *uplo, int *m, double *A, int *lda, int *info);
//
void blasfeo_dpotrs(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blasfeo_dtrtrs(char *uplo, char *trans, char *diag, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);



#endif // BLASFEO_API



#endif // BLAS_API



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLAS_API_H_
