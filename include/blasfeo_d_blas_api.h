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

// BLAS 3
//
void blasfeo_dgemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);



// LAPACK
//
void blasfeo_dpotrf(char *uplo, int *m, double *A, int *lda, int *info);

#endif // BLAS_API



#ifdef FORTRAN_BLAS_API

// BLAS 3
//
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);



// LAPACK
//
void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);

#endif // BLASFEO_API



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLAS_API_H_
