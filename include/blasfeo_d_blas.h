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



// level 2 BLAS
void dgemv_n_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z);
void dgemv_t_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z);
void dtrsv_ln_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y);
void dtrsv_lt_inv_lib(int m, int n, double *pA, int sda, double *inv_diag_A, double *x, double *y);
void dtrmv_un_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z);
void dtrmv_ut_lib(int m, double *pA, int sda, double *x, int alg, double *y, double *z);
	
// level 3 BLAS
void dgemm_ntnn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dgemm_ntnt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dgemm_nttn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dgemm_nttt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dsyrk_ntnn_l_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dtrmm_ntnn_lu_lib(int m, int n, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);

// LAPACK
void dpotrf_ntnn_l_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
void dsyrk_dpotrf_ntnn_l_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
