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
void sgemv_n_lib(int m, int n, float *pA, int sda, float *x, int alg, float *y, float *z);
// z <= beta * y + alpha * A' * x
void sgemv_t_lib(int m, int n, float *pA, int sda, float *x, int alg, float *y, float *z);
// y <= inv( A ) * x
void strsv_ln_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
// y <= inv( A' ) * x
void strsv_lt_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
// z <= beta * y + alpha * A * x ; A upper triangular
void strmv_un_lib(int m, float *pA, int sda, float *x, int alg, float *y, float *z);
// z <= beta * y + alpha * A' * x ; A upper triangular
void strmv_ut_lib(int m, float *pA, int sda, float *x, int alg, float *y, float *z);
	


//
// level 3 BLAS
//

// D <= beta * C + alpha * A * B'
void sgemm_nt_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
// D <= beta * C + alpha * A * B
void sgemm_nn_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
// D <= beta * C + alpha * A * B' ; C, D lower triangular
void ssyrk_nt_l_lib(int m, int n, int k, float alpha, float *pA, int sda, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
// D <= beta * C + alpha * A * B' ; B upper triangular
void strmm_nt_ru_lib(int m, int n, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);



//
// LAPACK
//

// D <= chol( C ) ; C, D lower triangular
void spotrf_nt_l_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
// D <= chol( C + A * B' ) ; C, D lower triangular
void ssyrk_spotrf_nt_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
