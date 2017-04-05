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

#ifdef __cplusplus
extern "C" {
#endif



// headers to reference BLAS and LAPACK routines employed in BLASFEO WR

// level 1
void dcopy_(long long *m, double *x, long long *incx, double *y, long long *incy);
void daxpy_(long long *m, double *alpha, double *x, long long *incx, double *y, long long *incy);
void dscal_(long long *m, double *alpha, double *x, long long *incx);

// level 2
void dgemv_(char *ta, long long *m, long long *n, double *alpha, double *A, long long *lda, double *x, long long *incx, double *beta, double *y, long long *incy);
void dsymv_(char *uplo, long long *m, double *alpha, double *A, long long *lda, double *x, long long *incx, double *beta, double *y, long long *incy);
void dtrmv_(char *uplo, char *trans, char *diag, long long *n, double *A, long long *lda, double *x, long long *incx);
void dtrsv_(char *uplo, char *trans, char *diag, long long *n, double *A, long long *lda, double *x, long long *incx);
void dger_(long long *m, long long *n, double *alpha, double *x, long long *incx, double *y, long long *incy, double *A, long long *lda);

// level 3
void dgemm_(char *ta, char *tb, long long *m, long long *n, long long *k, double *alpha, double *A, long long *lda, double *B, long long *ldb, double *beta, double *C, long long *ldc);
void dsyrk_(char *uplo, char *trans, long long *n, long long *k, double *alpha, double *A, long long *lda, double *beta, double *C, long long *ldc);
void dtrmm_(char *side, char *uplo, char *trans, char *diag, long long *m, long long *n, double *alpha, double *A, long long *lda, double *B, long long *ldb);
void dtrsm_(char *side, char *uplo, char *trans, char *diag, long long *m, long long *n, double *alpha, double *A, long long *lda, double *B, long long *ldb);

// lapack
long long dpotrf_(char *uplo, long long *m, double *A, long long *lda, long long *info);
long long dgetrf_(long long *m, long long *n, double *A, long long *lda, long long *ipiv, long long *info);



#ifdef __cplusplus
}
#endif
