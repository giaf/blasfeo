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
void sgemv_n_lib(int m, int n, float *pA, int sda, float *x, int alg, float *y, float *z);
void sgemv_t_lib(int m, int n, float *pA, int sda, float *x, int alg, float *y, float *z);
void strsv_ln_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
void strsv_lt_inv_lib(int m, int n, float *pA, int sda, float *inv_diag_A, float *x, float *y);
void strmv_un_lib(int m, float *pA, int sda, float *x, int alg, float *y, float *z);
void strmv_ut_lib(int m, float *pA, int sda, float *x, int alg, float *y, float *z);
	
// level 3 BLAS
void sgemm_ntnn_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);
void sgemm_ntnt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);
void sgemm_nttn_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);
void sgemm_nttt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);
void ssyrk_ntnn_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);
void strmm_ntnn_lu_lib(int m, int n, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd);

// LAPACK
void spotrf_ntnn_l_lib(int m, int n, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);
void ssyrk_spotrf_ntnn_l_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, int alg, float *pC, int sdc, float *pD, int sdd, float *inv_diag_D);

