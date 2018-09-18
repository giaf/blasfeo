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

#ifdef __cplusplus
extern "C" {
#endif



// headers to reference BLAS and LAPACK routines employed in BLASFEO WR

// level 1
void dcopy_(int *m, double *x, int *incx, double *y, int *incy);
void daxpy_(int *m, double *alpha, double *x, int *incx, double *y, int *incy);
void dscal_(int *m, double *alpha, double *x, int *incx);
void drot_(int *m, double *x, int *incx, double *y, int *incy, double *c, double *s);
void drotg_(double *a, double *b, double *c, double *s);

// level 2
void dgemv_(char *ta, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dsymv_(char *uplo, int *m, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dtrmv_(char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx);
void dtrsv_(char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *A, int *lda);

// level 3
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
void dtrmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);

// lapack
int dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
int dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
void dgeqrf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dgeqr2_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *info);
void dgelqf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);



#ifdef __cplusplus
}
#endif
