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
void scopy_(long long *m, float *x, long long *incx, float *y, long long *incy);
void saxpy_(long long *m, float *alpha, float *x, long long *incx, float *y, long long *incy);
void sscal_(long long *m, float *alpha, float *x, long long *incx);

// level 2
void sgemv_(char *ta, long long *m, long long *n, float *alpha, float *A, long long *lda, float *x, long long *incx, float *beta, float *y, long long *incy);
void ssymv_(char *uplo, long long *m, float *alpha, float *A, long long *lda, float *x, long long *incx, float *beta, float *y, long long *incy);
void strmv_(char *uplo, char *trans, char *diag, long long *n, float *A, long long *lda, float *x, long long *incx);
void strsv_(char *uplo, char *trans, char *diag, long long *n, float *A, long long *lda, float *x, long long *incx);
void sger_(long long *m, long long *n, float *alpha, float *x, long long *incx, float *y, long long *incy, float *A, long long *lda);

// level 3
void sgemm_(char *ta, char *tb, long long *m, long long *n, long long *k, float *alpha, float *A, long long *lda, float *B, long long *ldb, float *beta, float *C, long long *ldc);
void ssyrk_(char *uplo, char *trans, long long *n, long long *k, float *alpha, float *A, long long *lda, float *beta, float *C, long long *ldc);
void strmm_(char *side, char *uplo, char *transa, char *diag, long long *m, long long *n, float *alpha, float *A, long long *lda, float *B, long long *ldb);
void strsm_(char *side, char *uplo, char *transa, char *diag, long long *m, long long *n, float *alpha, float *A, long long *lda, float *B, long long *ldb);

// lapack
long long spotrf_(char *uplo, long long *m, float *A, long long *lda, long long *info);
long long sgetrf_(long long *m, long long *n, float *A, long long *lda, long long *ipiv, long long *info);
void sgeqrf_(long long *m, long long *n, float *A, long long *lda, float *tau, float *work, long long *lwork, long long *info);
void sgeqr2_(long long *m, long long *n, float *A, long long *lda, float *tau, float *work, long long *info);



#ifdef __cplusplus
}
#endif
