/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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



//
// level 2 BLAS
//

// z <= beta * y + alpha * A * x
void dgemv_n_lib(int m, int n, double alpha, double *pA, int sda, double *x, double beta, double *y, double *z);
// z <= beta * y + alpha * A' * x
void dgemv_t_lib(int m, int n, double alpha, double *pA, int sda, double *x, double beta, double *y, double *z);
// y <= inv( A ) * x
void dtrsv_ln_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y);
// y <= inv( A' ) * x
void dtrsv_lt_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y);
// z <= beta * y + alpha * A * x ; A upper triangular
void dtrmv_un_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z);
// z <= beta * y + alpha * A' * x ; A upper triangular
void dtrmv_ut_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z);
	


//
// level 3 BLAS
//

// D <= beta * C + alpha * A * B'
void dgemm_nt_lib(int m, int n, int k, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);
// D <= beta * C + alpha * A * B
void dgemm_nn_lib(int m, int n, int k, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);
// D <= beta * C + alpha * A * B' ; C, D lower triangular
void dsyrk_nt_l_lib(int m, int n, int k, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);
// D <= beta * C + alpha * A * B' ; B upper triangular
void dtrmm_nt_ru_lib(int m, int n, double alpha, double *pA, int sda, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void dpotrf_nt_l_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
// D <= chol( C + A * B' ) ; C, D lower triangular
void dsyrk_dpotrf_nt_l_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
