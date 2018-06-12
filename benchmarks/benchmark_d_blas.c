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


#if defined(BENCHMARKS_MODE)

#include <stdlib.h>
#include <stdio.h>

//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
//#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"
#include "../include/blasfeo_timing.h"

#ifndef D_PS
#define D_PS 1
#endif
#ifndef D_NC
#define D_NC 1
#endif



#if defined(REF_BLAS_NETLIB)
#include "cblas.h"
#include "lapacke.h"
#endif

#if defined(REF_BLAS_OPENBLAS)
void openblas_set_num_threads(int num_threads);
#include "cblas.h"
#include "lapacke.h"
#endif

#if defined(REF_BLAS_BLIS)
void omp_set_num_threads(int num_threads);
#include "blis.h"
#endif

#if defined(REF_BLAS_MKL)
#include "mkl.h"
#endif


#include "cpu_freq.h"


#if defined(LA_HIGH_PERFORMANCE) & (defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE))
void dgemm_nn_1_1_1(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_4x2_vs_lib4(1, &alpha, A, 0, B, sdb, &beta, C, D, 1, 1);
	return;
	}

void dgemm_nn_2_2_2(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_2x2_lib4(2, &alpha, A, 0, B, sdb, &beta, C, D);
	return;
	}

void dgemm_nn_3_3_3(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_4x4_vs_lib4(3, &alpha, A, 0, B, sdb, &beta, C, D, 3, 3);
	return;
	}

void dgemm_nn_4_4_4(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_4x4_lib4(4, &alpha, A, 0, B, sdb, &beta, C, D);
	return;
	}

void dgemm_nn_5_5_5(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_6x6_vs_lib4(5, &alpha, A, sda, 0, B, sdb, &beta, C, sdc, D, sdd, 5, 5);
	return;
	}

void dgemm_nn_6_6_6(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_6x6_lib4(6, &alpha, A, sda, 0, B, sdb, &beta, C, sdc, D, sdd);
	return;
	}

void dgemm_nn_7_7_7(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_vs_lib4(7, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(7, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd, 7, 3);
	return;
	}

void dgemm_nn_8_8_8(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_lib4(8, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(8, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	return;
	}

void dgemm_nn_9_9_9(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_10x4_vs_lib4(9, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(9, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd, 9, 4);
	kernel_dgemm_nn_10x2_vs_lib4(9, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd, 9, 1);
	return;
	}

void dgemm_nn_10_10_10(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_10x4_lib4(10, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_10x4_lib4(10, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_10x2_lib4(10, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	return;
	}

void dgemm_nn_11_11_11(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_12x4_vs_lib4(11, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(11, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(11, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd, 11, 3);
	return;
	}

void dgemm_nn_12_12_12(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_12x4_lib4(12, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(12, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(12, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	return;
	}

void dgemm_nn_13_13_13(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_lib4(13, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(13, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x6_vs_lib4(13, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd, 8, 5);

	kernel_dgemm_nn_6x8_vs_lib4(13, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdc, sdd, D+8*sdd, sdd, 5, 8);
	kernel_dgemm_nn_6x6_vs_lib4(13, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdc+8*4, sdd, D+8*sdd+8*4, sdd, 5, 5);
	return;
	}

void dgemm_nn_14_14_14(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_lib4(14, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(14, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x6_lib4(14, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);

	kernel_dgemm_nn_6x8_lib4(14, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdc, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_6x6_lib4(14, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdc+8*4, sdd, D+8*sdd+8*4, sdd);
	return;
	}

void dgemm_nn_15_15_15(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_8x6_lib4(15, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x6_lib4(15, &alpha, A, sda, 0, B+6*4, sdb, &beta, C+6*4, sdd, D+6*4, sdd);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd, 8, 3);

	kernel_dgemm_nn_8x6_vs_lib4(15, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd, 7, 6);
	kernel_dgemm_nn_8x6_vs_lib4(15, &alpha, A+8*sda, sda, 0, B+6*4, sdb, &beta, C+8*sdd+6*4, sdd, D+8*sdd+6*4, sdd, 7, 6);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd, 7, 3);
#else
	kernel_dgemm_nn_8x4_lib4(15, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(15, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(15, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd, 8, 3);

	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(15, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd, 7, 3);
#endif
	return;
	}

void dgemm_nn_16_16_16(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_8x6_lib4(16, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x6_lib4(16, &alpha, A, sda, 0, B+6*4, sdb, &beta, C+6*4, sdd, D+6*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);

	kernel_dgemm_nn_8x6_lib4(16, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x6_lib4(16, &alpha, A+8*sda, sda, 0, B+6*4, sdb, &beta, C+8*sdd+6*4, sdd, D+8*sdd+6*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
#else
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);

	kernel_dgemm_nn_8x4_lib4(16, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(16, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
#endif
	return;
	}

void dgemm_nn_17_17_17(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(17, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(17, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(17, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(17, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_vs_lib4(17, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd, 12, 1);

	kernel_dgemm_nn_6x8_vs_lib4(17, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd, 5, 8);
	kernel_dgemm_nn_6x8_vs_lib4(17, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd, 5, 8);
	kernel_dgemm_nn_6x2_vs_lib4(17, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd, 5, 1);
#else
	kernel_dgemm_nn_8x4_lib4(17, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(17, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(17, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x6_vs_lib4(17, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd, 8, 5);

	kernel_dgemm_nn_10x4_vs_lib4(17, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(17, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(17, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(17, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd, 9, 4);
	kernel_dgemm_nn_10x2_vs_lib4(17, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd, 9, 1);
#endif
	return;
	}

void dgemm_nn_18_18_18(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(18, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(18, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(18, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(18, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_vs_lib4(18, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd, 12, 2);

	kernel_dgemm_nn_6x8_lib4(18, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd);
	kernel_dgemm_nn_6x8_lib4(18, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd);
	kernel_dgemm_nn_6x2_lib4(18, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd);
#else
	kernel_dgemm_nn_8x4_lib4(18, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(18, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(18, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x6_lib4(18, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);

	kernel_dgemm_nn_10x4_lib4(18, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_10x4_lib4(18, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_10x4_lib4(18, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_10x4_lib4(18, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_10x2_lib4(18, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd);
#endif
	return;
	}

void dgemm_nn_19_19_19(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_lib4(19, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(19, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(19, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(19, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x4_vs_lib4(19, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd, 8, 3);

	kernel_dgemm_nn_12x4_vs_lib4(19, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(19, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(19, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(19, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(19, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd, 11, 3);
	return;
	}

void dgemm_nn_20_20_20(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
	kernel_dgemm_nn_8x4_lib4(20, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(20, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(20, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(20, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(20, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);

	kernel_dgemm_nn_12x4_lib4(20, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_12x4_lib4(20, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(20, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(20, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(20, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd);
	return;
	}

void dgemm_nn_21_21_21(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(21, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(21, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(21, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(21, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(21, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_12x4_vs_lib4(21, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd, 12, 1);

	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B+4*4, sdb, &beta, C+12*sdd+4*4, sdd, D+12*sdd+4*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B+12*4, sdb, &beta, C+12*sdd+12*4, sdd, D+12*sdd+12*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd, 9, 4);
	kernel_dgemm_nn_10x4_vs_lib4(21, &alpha, A+12*sda, sda, 0, B+20*4, sdb, &beta, C+12*sdd+20*4, sdd, D+12*sdd+20*4, sdd, 9, 1);
#else
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x6_vs_lib4(21, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd, 8, 5);

	kernel_dgemm_nn_8x4_lib4(21, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(21, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_8x6_vs_lib4(21, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd, 8, 5);

	kernel_dgemm_nn_6x8_vs_lib4(21, &alpha, A+16*sda, sda, 0, B, sdb, &beta, C+16*sdd, sdd, D+16*sdd, sdd, 5, 8);
	kernel_dgemm_nn_6x8_vs_lib4(21, &alpha, A+16*sda, sda, 0, B+8*4, sdb, &beta, C+16*sdd+8*4, sdd, D+16*sdd+8*4, sdd, 5, 8);
	kernel_dgemm_nn_6x6_vs_lib4(21, &alpha, A+16*sda, sda, 0, B+16*4, sdb, &beta, C+16*sdd+16*4, sdd, D+16*sdd+16*4, sdd, 5, 5);
#endif
	return;
	}

void dgemm_nn_22_22_22(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(22, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(22, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(22, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(22, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(22, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_12x4_vs_lib4(22, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd, 12, 2);

	kernel_dgemm_nn_10x4_lib4(22, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd);
	kernel_dgemm_nn_10x4_lib4(22, &alpha, A+12*sda, sda, 0, B+4*4, sdb, &beta, C+12*sdd+4*4, sdd, D+12*sdd+4*4, sdd);
	kernel_dgemm_nn_10x4_lib4(22, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd);
	kernel_dgemm_nn_10x4_lib4(22, &alpha, A+12*sda, sda, 0, B+12*4, sdb, &beta, C+12*sdd+12*4, sdd, D+12*sdd+12*4, sdd);
	kernel_dgemm_nn_10x4_lib4(22, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd);
	kernel_dgemm_nn_10x4_vs_lib4(22, &alpha, A+12*sda, sda, 0, B+20*4, sdb, &beta, C+12*sdd+20*4, sdd, D+12*sdd+20*4, sdd, 10, 2);
#else
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x6_lib4(22, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);

	kernel_dgemm_nn_8x4_lib4(22, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(22, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_8x6_lib4(22, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd);

	kernel_dgemm_nn_6x8_lib4(22, &alpha, A+16*sda, sda, 0, B, sdb, &beta, C+16*sdd, sdd, D+16*sdd, sdd);
	kernel_dgemm_nn_6x8_lib4(22, &alpha, A+16*sda, sda, 0, B+8*4, sdb, &beta, C+16*sdd+8*4, sdd, D+16*sdd+8*4, sdd);
	kernel_dgemm_nn_6x6_lib4(22, &alpha, A+16*sda, sda, 0, B+16*4, sdb, &beta, C+16*sdd+16*4, sdd, D+16*sdd+16*4, sdd);
#endif
	return;
	}

void dgemm_nn_23_23_23(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(23, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(23, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(23, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(23, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(23, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd, 12, 3);

	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B+4*4, sdb, &beta, C+12*sdd+4*4, sdd, D+12*sdd+4*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B+12*4, sdb, &beta, C+12*sdd+12*4, sdd, D+12*sdd+12*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd, 11, 4);
	kernel_dgemm_nn_12x4_vs_lib4(23, &alpha, A+12*sda, sda, 0, B+20*4, sdb, &beta, C+12*sdd+20*4, sdd, D+12*sdd+20*4, sdd, 11, 3);
#else
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd, 8, 3);

	kernel_dgemm_nn_8x4_lib4(23, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(23, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+8*sda, sda, 0, B+20*4, sdb, &beta, C+8*sdd+20*4, sdd, D+8*sdd+20*4, sdd, 8, 3);

	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B, sdb, &beta, C+16*sdd, sdd, D+16*sdd, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B+4*4, sdb, &beta, C+16*sdd+4*4, sdd, D+16*sdd+4*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B+8*4, sdb, &beta, C+16*sdd+8*4, sdd, D+16*sdd+8*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B+12*4, sdb, &beta, C+16*sdd+12*4, sdd, D+16*sdd+12*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B+16*4, sdb, &beta, C+16*sdd+16*4, sdd, D+16*sdd+16*4, sdd, 7, 4);
	kernel_dgemm_nn_8x4_vs_lib4(23, &alpha, A+16*sda, sda, 0, B+20*4, sdb, &beta, C+16*sdd+20*4, sdd, D+16*sdd+20*4, sdd, 7, 3);
#endif
	return;
	}

void dgemm_nn_24_24_24(double alpha, double *A, int sda, double *B, int sdb, double beta, double *C, int sdc, double *D, int sdd)
	{
#if defined(TARGET_X64_INTEL_HASWELL)
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd);

	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B, sdb, &beta, C+12*sdd, sdd, D+12*sdd, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B+4*4, sdb, &beta, C+12*sdd+4*4, sdd, D+12*sdd+4*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B+8*4, sdb, &beta, C+12*sdd+8*4, sdd, D+12*sdd+8*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B+12*4, sdb, &beta, C+12*sdd+12*4, sdd, D+12*sdd+12*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B+16*4, sdb, &beta, C+12*sdd+16*4, sdd, D+12*sdd+16*4, sdd);
	kernel_dgemm_nn_12x4_lib4(24, &alpha, A+12*sda, sda, 0, B+20*4, sdb, &beta, C+12*sdd+20*4, sdd, D+12*sdd+20*4, sdd);
#else
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B, sdb, &beta, C, sdd, D, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B+4*4, sdb, &beta, C+4*4, sdd, D+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B+8*4, sdb, &beta, C+8*4, sdd, D+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B+12*4, sdb, &beta, C+12*4, sdd, D+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B+16*4, sdb, &beta, C+16*4, sdd, D+16*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A, sda, 0, B+20*4, sdb, &beta, C+20*4, sdd, D+20*4, sdd);

	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B, sdb, &beta, C+8*sdd, sdd, D+8*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B+4*4, sdb, &beta, C+8*sdd+4*4, sdd, D+8*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B+8*4, sdb, &beta, C+8*sdd+8*4, sdd, D+8*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B+12*4, sdb, &beta, C+8*sdd+12*4, sdd, D+8*sdd+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B+16*4, sdb, &beta, C+8*sdd+16*4, sdd, D+8*sdd+16*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+8*sda, sda, 0, B+20*4, sdb, &beta, C+8*sdd+20*4, sdd, D+8*sdd+20*4, sdd);

	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B, sdb, &beta, C+16*sdd, sdd, D+16*sdd, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B+4*4, sdb, &beta, C+16*sdd+4*4, sdd, D+16*sdd+4*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B+8*4, sdb, &beta, C+16*sdd+8*4, sdd, D+16*sdd+8*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B+12*4, sdb, &beta, C+16*sdd+12*4, sdd, D+16*sdd+12*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B+16*4, sdb, &beta, C+16*sdd+16*4, sdd, D+16*sdd+16*4, sdd);
	kernel_dgemm_nn_8x4_lib4(24, &alpha, A+16*sda, sda, 0, B+20*4, sdb, &beta, C+16*sdd+20*4, sdd, D+16*sdd+20*4, sdd);
#endif
	return;
	}
#endif


int main()
	{

#if defined(REF_BLAS_OPENBLAS)
	openblas_set_num_threads(1);
#endif
#if defined(REF_BLAS_BLIS)
	omp_set_num_threads(1);
#endif
#if defined(REF_BLAS_MKL)
	mkl_set_num_threads(1);
#endif

//#if defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
//#endif

	printf("\n");
	printf("\n");
	printf("\n");

	printf("BLAS performance test - double precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit test_param.h to modify this value).\n", GHz_max);
	printf("\n");

	// maximum flops per cycle, double precision
#if defined(TARGET_X64_INTEL_HASWELL)
	const float flops_max = 16;
	printf("Testing BLAS version for AVX2 and FMA instruction sets, 64 bit (optimized for Intel Haswell): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	const float flops_max = 8;
	printf("Testing BLAS version for AVX instruction set, 64 bit (optimized for Intel Sandy Bridge): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_INTEL_CORE)
	const float flops_max = 4;
	printf("Testing BLAS version for SSE3 instruction set, 64 bit (optimized for Intel Core): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_AMD_BULLDOZER)
	const float flops_max = 8;
	printf("Testing BLAS version for SSE3 and FMA instruction set, 64 bit (optimized for AMD Bulldozer): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	const float flops_max = 4;
	printf("Testing BLAS version for NEONv2 instruction set, 64 bit (optimized for ARM Cortex A57): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	const float flops_max = 4;
	printf("Testing BLAS version for NEONv2 instruction set, 64 bit (optimized for ARM Cortex A53): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	const float flops_max = 2;
	printf("Testing BLAS version for VFPv4 instruction set, 32 bit (optimized for ARM Cortex A15): theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_GENERIC)
	const float flops_max = 2;
	printf("Testing BLAS version for generic scalar instruction set: theoretical peak %5.1f Gflops ???\n", flops_max*GHz_max);
#endif

//	FILE *f;
//	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_INTEL_HASWELL)
//	fprintf(f, "C = 'd_x64_intel_haswell';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
//	fprintf(f, "C = 'd_x64_intel_sandybridge';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_INTEL_CORE)
//	fprintf(f, "C = 'd_x64_intel_core';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_X64_AMD_BULLDOZER)
//	fprintf(f, "C = 'd_x64_amd_bulldozer';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
//	fprintf(f, "C = 'd_armv8a_arm_cortex_a57';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
//	fprintf(f, "C = 'd_armv7a_arm_cortex_a15';\n");
//	fprintf(f, "\n");
#elif defined(TARGET_GENERIC)
//	fprintf(f, "C = 'd_generic';\n");
//	fprintf(f, "\n");
#endif

//	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
//	fprintf(f, "\n");

//	fprintf(f, "B = [\n");



	int i, j, rep, ll;

	const int bsd = D_PS;
	const int ncd = D_NC;

/*	int info = 0;*/

	printf("\nn\t  dgemm_blasfeo\t  dgemm_blas\n");
	printf("\nn\t Gflops\t    %%\t Gflops\n\n");

#if 1
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 500, 550, 600, 650, 700};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4};

//	for(ll=0; ll<24; ll++)
	for(ll=0; ll<75; ll++)
//	for(ll=0; ll<115; ll++)
//	for(ll=0; ll<120; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll]/2;
//		int n = ll+1;
//		int nrep = nnrep[0];
//		n = n<12 ? 12 : n;
//		n = n<8 ? 8 : n;

#elif 1
	int nn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	for(ll=0; ll<24; ll++)

		{

		int n = nn[ll];
		int nrep = 40000; //nnrep[ll];
#else
// TODO  ll<1 !!!!!

	for(ll=0; ll<1; ll++)

		{

		int n = 24;
		int nrep = 40000; //nnrep[ll];
#endif

		int rep_in;
		int nrep_in = 10;

		double *A; d_zeros_align(&A, n, n);
		double *B; d_zeros_align(&B, n, n);
		double *C; d_zeros_align(&C, n, n);
		double *M; d_zeros_align(&M, n, n);

		char c_n = 'n';
		char c_l = 'l';
		char c_r = 'r';
		char c_t = 't';
		char c_u = 'u';
		int i_1 = 1;
		int i_t;
		double d_1 = 1;
		double d_0 = 0;

		for(i=0; i<n*n; i++)
			A[i] = i;

		for(i=0; i<n; i++)
			B[i*(n+1)] = 1;

		for(i=0; i<n*n; i++)
			M[i] = 1;

		int n2 = n*n;
		double *B2; d_zeros(&B2, n, n);
		for(i=0; i<n*n; i++)
			B2[i] = 1e-15;
		for(i=0; i<n; i++)
			B2[i*(n+1)] = 1;

		int pnd = ((n+bsd-1)/bsd)*bsd;
		int cnd = ((n+ncd-1)/ncd)*ncd;
		int cnd2 = 2*((n+ncd-1)/ncd)*ncd;

		double *x; d_zeros_align(&x, pnd, 1);
		double *y; d_zeros_align(&y, pnd, 1);
		double *x2; d_zeros_align(&x2, pnd, 1);
		double *y2; d_zeros_align(&y2, pnd, 1);
		double *diag; d_zeros_align(&diag, pnd, 1);
		int *ipiv; int_zeros(&ipiv, n, 1);

		for(i=0; i<pnd; i++) x[i] = 1;
		for(i=0; i<pnd; i++) x2[i] = 1;

		// matrix struct
#if 0
		struct blasfeo_dmat sA; blasfeo_allocate_dmat(n+4, n+4, &sA);
		struct blasfeo_dmat sB; blasfeo_allocate_dmat(n+4, n+4, &sB);
		struct blasfeo_dmat sC; blasfeo_allocate_dmat(n+4, n+4, &sC);
		struct blasfeo_dmat sD; blasfeo_allocate_dmat(n+4, n+4, &sD);
		struct blasfeo_dmat sE; blasfeo_allocate_dmat(n+4, n+4, &sE);
#else
		struct blasfeo_dmat sA; blasfeo_allocate_dmat(n, n, &sA);
		struct blasfeo_dmat sB; blasfeo_allocate_dmat(n, n, &sB);
		struct blasfeo_dmat sB2; blasfeo_allocate_dmat(n, n, &sB2);
		struct blasfeo_dmat sB3; blasfeo_allocate_dmat(n, n, &sB3);
		struct blasfeo_dmat sC; blasfeo_allocate_dmat(n, n, &sC);
		struct blasfeo_dmat sD; blasfeo_allocate_dmat(n, n, &sD);
		struct blasfeo_dmat sE; blasfeo_allocate_dmat(n, n, &sE);
#endif
		struct blasfeo_dvec sx; blasfeo_allocate_dvec(n, &sx);
		struct blasfeo_dvec sy; blasfeo_allocate_dvec(n, &sy);
		struct blasfeo_dvec sz; blasfeo_allocate_dvec(n, &sz);

		blasfeo_pack_dmat(n, n, A, n, &sA, 0, 0);
		blasfeo_pack_dmat(n, n, B, n, &sB, 0, 0);
		blasfeo_pack_dmat(n, n, B2, n, &sB2, 0, 0);
		blasfeo_pack_dvec(n, x, &sx, 0);
		int ii;

		for(ii=0; ii<n; ii++)
			{
			BLASFEO_DMATEL(&sB3, ii, ii) = 1.0;
			// BLASFEO_DMATEL(&sB3, n-1, ii) = 1.0;
			BLASFEO_DMATEL(&sB3, ii, n-1) = 1.0;
			BLASFEO_DVECEL(&sx, ii) = 1.0;
			}

		int qr_work_size = blasfeo_dgeqrf_worksize(n, n);
		void *qr_work;
		v_zeros_align(&qr_work, qr_work_size);

		int lq_work_size = blasfeo_dgelqf_worksize(n, n);
		void *lq_work;
		v_zeros_align(&lq_work, lq_work_size);

		// create matrix to pivot all the time
		// blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sD, 0, 0);

		double *dummy;

		int info;

		double alpha = 1.0;
		double beta = 0.0;

		/* timing */
		blasfeo_timer timer;

		double time_blasfeo  = 1e15;
		double time_blas     = 1e15;
		double tmp_time_blasfeo;
		double tmp_time_blas;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0);
			}

		/* benchmarks */

		// batches repetion, find minimum averaged time
		// discard batch interrupted by the scheduler
		for(rep_in=0; rep_in<nrep_in; rep_in++)
			{

			// BENCHMARK_BLASFEO
			blasfeo_tic(&timer);

			// averaged repetions
			for(rep=0; rep<nrep; rep++)
				{

//				kernel_dgemm_nt_12x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nt_8x8_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dsyrk_nt_l_8x8_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nt_8x4_lib4(n, &alpha, sA.pA, sA.cn, sB.pA, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nt_4x8_lib4(n, &alpha, sA.pA, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nt_4x4_lib4(n, &alpha, sA.pA, sB.pA, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_4x4_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dger4_12_sub_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//				kernel_dger4_sub_12r_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//				kernel_dger4_sub_8r_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//				kernel_dger12_add_4r_lib4(n, sA.pA, sB.pA, sB.cn, sD.pA);
//				kernel_dger8_add_4r_lib4(n, sA.pA, sB.pA, sB.cn, sD.pA);
//				kernel_dger4_sub_4r_lib4(n, sA.pA, sB.pA, sD.pA);
//				kernel_dger2_sub_4r_lib4(n, sA.pA, sB.pA, sD.pA);
//				kernel_dger4_sub_8c_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//				kernel_dger4_sub_4c_lib4(n, sA.pA, sA.cn, sB.pA, sD.pA, sD.cn);
//				kernel_dgemm_nn_4x12_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_4x8_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_2x8_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_4x4_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_12x4_lib4(n, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nn_8x4_lib4(n, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nn_4x4_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, sD.pA, sD.pA);
//				kernel_dgemm_nn_8x6_lib4(n, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, sD.pA, sD.cn, sD.pA, sD.cn);
//				kernel_dgemm_nn_8x4_gen_lib4(n, &alpha, sA.pA, sA.cn, 0, sB.pA, sB.cn, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 8, 0, 4);
//				kernel_dgemm_nn_4x4_gen_lib4(n, &alpha, sA.pA, 0, sB.pA, sB.cn, &beta, 0, sD.pA, sD.cn, 0, sD.pA, sD.cn, 0, 8, 0, 4);

//				blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//				blasfeo_dgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//				blasfeo_dsyrk_ln(n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0);
//				blasfeo_dsyrk_ln_mn(n, n, n, 1.0, &sA, 0, 0, &sA, 0, 0, 0.0, &sC, 0, 0, &sD, 0, 0);
//				blasfeo_dpotrf_l_mn(n, n, &sB, 0, 0, &sB, 0, 0);
				blasfeo_dpotrf_l(n, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dgetrf_nopivot(n, n, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dgetrf_rowpivot(n, n, &sB, 0, 0, &sB, 0, 0, ipiv);
//				blasfeo_dgeqrf(n, n, &sC, 0, 0, &sD, 0, 0, qr_work);
//				blasfeo_dcolin(n, &sx, 0, &sB3, 0, n-1);
//				blasfeo_dgelqf(n, n, &sB3, 0, 0, &sB3, 0, 0, lq_work);
//				blasfeo_dgelqf_pd(n, n, &sB3, 0, 0, &sB3, 0, 0, lq_work);
//				blasfeo_dgelqf_pd_la(n, n, &sB3, 0, 0, &sA, 0, 0, lq_work);
//				blasfeo_dgelqf_pd_lla(n, n, &sB3, 0, 0, &sB, 0, 0, &sA, 0, 0, lq_work);
//				blasfeo_dtrmm_rlnn(n, n, 1.0, &sA, 0, 0, &sD, 0, 0, &sD, 0, 0); //
//				blasfeo_dtrmm_rutn(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0);
//				blasfeo_dtrsm_llnu(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dtrsm_lunn(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dtrsm_rltn(n, n, 1.0, &sB2, 0, 0, &sD, 0, 0, &sD, 0, 0); //
//				blasfeo_dtrsm_rltu(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dtrsm_rutn(n, n, 1.0, &sD, 0, 0, &sB, 0, 0, &sB, 0, 0);
//				blasfeo_dgemv_n(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//				blasfeo_dgemv_t(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//				blasfeo_dsymv_l(n, n, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sz, 0);
//				blasfeo_dgemv_nt(n, n, 1.0, 1.0, &sA, 0, 0, &sx, 0, &sx, 0, 0.0, 0.0, &sy, 0, &sy, 0, &sz, 0, &sz, 0);
				}

			tmp_time_blasfeo = blasfeo_toc(&timer) / nrep;
			time_blasfeo = tmp_time_blasfeo<time_blasfeo ? tmp_time_blasfeo : time_blasfeo;
			// BENCHMARK_BLASFEO_END

			// BENCHMARK_BLAS_REF
			blasfeo_tic(&timer);

			for(rep=0; rep<nrep; rep++)
				{
				#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_NETLIB) || defined(REF_BLAS_MKL)
				dpotrf_(&c_l, &n, B2, &n, &info);
				// dgemm_(&c_n, &c_n, &n, &n, &n, &d_1, A, &n, B, &n, &d_0, C, &n);
				// dgemm_(&c_n, &c_n, &n, &n, &n, &d_1, A, &n, M, &n, &d_0, C, &n);
				// dsyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_0, C, &n);
				// dtrmm_(&c_r, &c_u, &c_t, &c_n, &n, &n, &d_1, A, &n, C, &n);
				// dgetrf_(&n, &n, B2, &n, ipiv, &info);
				// dtrsm_(&c_l, &c_l, &c_n, &c_u, &n, &n, &d_1, B2, &n, B, &n);
				// dtrsm_(&c_l, &c_u, &c_n, &c_n, &n, &n, &d_1, B2, &n, B, &n);
				// dtrtri_(&c_l, &c_n, &n, B2, &n, &info);
				// dlauum_(&c_l, &n, B, &n, &info);
				// dgemv_(&c_n, &n, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);
				// dgemv_(&c_t, &n, &n, &d_1, A, &n, x2, &i_1, &d_0, y2, &i_1);
				// dtrmv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
				// dtrsv_(&c_l, &c_n, &c_n, &n, B, &n, x, &i_1);
				// dsymv_(&c_l, &n, &d_1, A, &n, x, &i_1, &d_0, y, &i_1);
				// for(i=0; i<n; i++)
				// 	{
				// 	i_t = n-i;
				// 	dcopy_(&i_t, &B[i*(n+1)], &i_1, &C[i*(n+1)], &i_1);
				// 	}
				// dsyrk_(&c_l, &c_n, &n, &n, &d_1, A, &n, &d_1, C, &n);
				// dpotrf_(&c_l, &n, C, &n, &info);
				#endif

				#if defined(REF_BLAS_BLIS)
				// dgemm_(&c_n, &c_t, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
				// dgemm_(&c_n, &c_n, &n77, &n77, &n77, &d_1, A, &n77, B, &n77, &d_0, C, &n77);
				// dsyrk_(&c_l, &c_n, &n77, &n77, &d_1, A, &n77, &d_0, C, &n77);
				// dtrmm_(&c_r, &c_u, &c_t, &c_n, &n77, &n77, &d_1, A, &n77, C, &n77);
				// dpotrf_(&c_l, &n77, B, &n77, &info);
				// dtrtri_(&c_l, &c_n, &n77, B, &n77, &info);
				// dlauum_(&c_l, &n77, B, &n77, &info);
				#endif
				}

			tmp_time_blas = blasfeo_toc(&timer) / nrep;
			time_blas = tmp_time_blas<time_blas ? tmp_time_blas : time_blas;

			// BENCHMARK_BLAS_REF_END

			}

		float Gflops_max = flops_max * GHz_max;

//		float flop_operation = 4*16.0*2*n; // kernel 16x4
//		float flop_operation = 3*16.0*2*n; // kernel 12x4
//		float flop_operation = 2*16.0*2*n; // kernel 8x4
//		float flop_operation = 1*16.0*2*n; // kernel 4x4
//		float flop_operation = 0.5*16.0*2*n; // kernel 2x4
//		float flop_operation = 2.0*n*n*n; // gemm
//		float flop_operation = 1.0*n*n*n; // syrk trmm trsm
//		float flop_operation = 2.0/3.0*n*n*n; // getrf
//		float flop_operation = 4.0/3.0*n*n*n; // geqrf
//		float flop_operation = 2.0*n*n*n; // geqrf_la
//		float flop_operation = 8.0/3.0*n*n*n; // geqrf_lla
//		float flop_operation = 2.0*n*n; // gemv symv
//		float flop_operation = 1.0*n*n; // trmv trsv
//		float flop_operation = 4.0*n*n; // gemv_nt
//		float flop_operation = 4.0/3.0*n*n*n; // syrk+potrf
		float flop_operation = 1.0/3.0*n*n*n; // potrf trtri

		float Gflops_blasfeo  = 1e-9*flop_operation/time_blasfeo;
		float Gflops_blas     = 1e-9*flop_operation/time_blas;

		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n",
			n,
			Gflops_blasfeo, 100.0*Gflops_blasfeo/Gflops_max,
			Gflops_blas, 100.0*Gflops_blas/Gflops_max);

		d_free(A);
		d_free(B);
		d_free(B2);
		d_free(M);
		d_free_align(x);
		d_free_align(y);
		d_free_align(x2);
		d_free_align(y2);
		int_free(ipiv);
		free(qr_work);
		free(lq_work);

		blasfeo_free_dmat(&sA);
		blasfeo_free_dmat(&sB);
		blasfeo_free_dmat(&sB2);
		blasfeo_free_dmat(&sB3);
		blasfeo_free_dmat(&sC);
		blasfeo_free_dmat(&sD);
		blasfeo_free_dmat(&sE);
		blasfeo_free_dvec(&sx);
		blasfeo_free_dvec(&sy);
		blasfeo_free_dvec(&sz);

		}

	printf("\n");

	return 0;

	}

#else

#include <stdio.h>

int main()
	{
	printf("\n\n Recompile BLASFEO with BENCHMARKS_MODE=1 to run this benchmark.\n");
	printf("On CMake use -DBLASFEO_BENCHMARKS=ON .\n\n");
	return 0;
	}

#endif
