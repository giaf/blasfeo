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


#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dgemm_nt_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];
		b_2 = B[2+ldb*0];
		b_3 = B[3+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[0+ldb*1];
		b_1 = B[1+ldb*1];
		b_2 = B[2+ldb*1];
		b_3 = B[3+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[0+ldb*2];
		b_1 = B[1+ldb*2];
		b_2 = B[2+ldb*2];
		b_3 = B[3+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[0+ldb*3];
		b_1 = B[1+ldb*3];
		b_2 = B[2+ldb*3];
		b_3 = B[3+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;

		A += 16;
		B += 4*ldb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];
		b_2 = B[2+ldb*0];
		b_3 = B[3+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;

		A += 4;
		B += 1*ldb;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
	D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
	D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



static void kernel_dgemm_nt_4x3_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];
		b_2 = B[2+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[0+ldb*1];
		b_1 = B[1+ldb*1];
		b_2 = B[2+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[0+ldb*2];
		b_1 = B[1+ldb*2];
		b_2 = B[2+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[0+ldb*3];
		b_1 = B[1+ldb*3];
		b_2 = B[2+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		A += 16;
		B += 4*ldb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];
		b_2 = B[2+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		A += 4;
		B += 1*ldb;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	return;

	}



static void kernel_dgemm_nt_4x2_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[0+ldb*1];
		b_1 = B[1+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[0+ldb*2];
		b_1 = B[1+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[0+ldb*3];
		b_1 = B[1+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		A += 16;
		B += 4*ldb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[1+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		A += 4;
		B += 1*ldb;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	return;

	}



static void kernel_dgemm_nt_4x1_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[0+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[0+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[0+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		A += 16;
		B += 4*ldb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		A += 4;
		B += 1*ldb;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	return;

	}



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dgemm_nt_4x4_vs_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	if(n1<=1)
		{
		kernel_dgemm_nt_4x1_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else if(n1==2)
		{
		kernel_dgemm_nt_4x2_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else if(n1==3)
		{
		kernel_dgemm_nt_4x3_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else //if(n1==1)
		{
		kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) | defined(TARGET_X86_AMD_BARCELONA)
void kernel_dgemm_nt_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)
	{

#if defined(TARGET_X86_AMD_BARCELONA)
	kernel_dgemm_nt_4x2_lib44c(kmax, alpha, A, B+0, beta, C+0*ldc, ldc, D+0*ldd, ldd);
	kernel_dgemm_nt_4x2_lib44c(kmax, alpha, A, B+2, beta, C+2*ldc, ldc, D+2*ldd, ldd);
	return;
#endif

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
	D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
	D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dgemm_nt_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dgemm_nt_4x4_libc4c(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, B, A, lda, &beta1, CC, bs, CC, bs);

	double tmp;
	tmp = CC[1+bs*0]; CC[1+bs*0] = CC[0+bs*1]; CC[0+bs*1] = tmp;
	tmp = CC[2+bs*0]; CC[2+bs*0] = CC[0+bs*2]; CC[0+bs*2] = tmp;
	tmp = CC[3+bs*0]; CC[3+bs*0] = CC[0+bs*3]; CC[0+bs*3] = tmp;
	tmp = CC[2+bs*1]; CC[2+bs*1] = CC[1+bs*2]; CC[1+bs*2] = tmp;
	tmp = CC[3+bs*1]; CC[3+bs*1] = CC[1+bs*3]; CC[1+bs*3] = tmp;
	tmp = CC[3+bs*2]; CC[3+bs*2] = CC[2+bs*3]; CC[2+bs*3] = tmp;

	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
	D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
	D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dgemm_nt_4x4_vs_libc4c(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, B, A, lda, &beta1, CC, bs, CC, bs);

	double tmp;
	tmp = CC[1+bs*0]; CC[1+bs*0] = CC[0+bs*1]; CC[0+bs*1] = tmp;
	tmp = CC[2+bs*0]; CC[2+bs*0] = CC[0+bs*2]; CC[0+bs*2] = tmp;
	tmp = CC[3+bs*0]; CC[3+bs*0] = CC[0+bs*3]; CC[0+bs*3] = tmp;
	tmp = CC[2+bs*1]; CC[2+bs*1] = CC[1+bs*2]; CC[1+bs*2] = tmp;
	tmp = CC[3+bs*1]; CC[3+bs*1] = CC[1+bs*3]; CC[1+bs*3] = tmp;
	tmp = CC[3+bs*2]; CC[3+bs*2] = CC[2+bs*3]; CC[2+bs*3] = tmp;

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dgemm_nn_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];
		b_2 = B[0+ldb*2];
		b_3 = B[0+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1+ldb*0];
		b_1 = B[1+ldb*1];
		b_2 = B[1+ldb*2];
		b_3 = B[1+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2+ldb*0];
		b_1 = B[2+ldb*1];
		b_2 = B[2+ldb*2];
		b_3 = B[2+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3+ldb*0];
		b_1 = B[3+ldb*1];
		b_2 = B[3+ldb*2];
		b_3 = B[3+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;

		A += 16;
		B += 4;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];
		b_2 = B[0+ldb*2];
		b_3 = B[0+ldb*3];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		CC[0+bs*3] += a_0 * b_3;
		CC[1+bs*3] += a_1 * b_3;
		CC[2+bs*3] += a_2 * b_3;
		CC[3+bs*3] += a_3 * b_3;

		A += 4;
		B += 1;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
	D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
	D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



static void kernel_dgemm_nn_4x3_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];
		b_2 = B[0+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1+ldb*0];
		b_1 = B[1+ldb*1];
		b_2 = B[1+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2+ldb*0];
		b_1 = B[2+ldb*1];
		b_2 = B[2+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3+ldb*0];
		b_1 = B[3+ldb*1];
		b_2 = B[3+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		A += 16;
		B += 4;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];
		b_2 = B[0+ldb*2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;

		A += 4;
		B += 1;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	return;

	}



static void kernel_dgemm_nn_4x2_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1+ldb*0];
		b_1 = B[1+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2+ldb*0];
		b_1 = B[2+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3+ldb*0];
		b_1 = B[3+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		A += 16;
		B += 4;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];
		b_1 = B[0+ldb*1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;

		A += 4;
		B += 1;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	return;

	}



static void kernel_dgemm_nn_4x1_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		A += 16;
		B += 4;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0+ldb*0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;

		A += 4;
		B += 1;

		}
	
	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	return;

	}



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dgemm_nn_4x4_vs_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	if(n1<=1)
		{
		kernel_dgemm_nn_4x1_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else if(n1==2)
		{
		kernel_dgemm_nn_4x2_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else if(n1==3)
		{
		kernel_dgemm_nn_4x3_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}
	else //if(n1==4)
		{
		kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, &beta1, CC, bs, CC, bs);
		}

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		}

	return;

	}
#endif


#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dsyrk_nt_l_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_AMD_BULLDOZER)  || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dsyrk_nt_l_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dsyrk_nt_u_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
//	D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
//	D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
//	D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

	D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
	D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
//	D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
//	D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

	D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
	D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
	D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
//	D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

	D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
	D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
	D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
	D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7)
void kernel_dsyrk_nt_u_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = 1.0;
	double beta1 = 0.0;

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	if(m1>=4)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
//		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
//		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];
//		D[3+ldd*0] = beta[0]*C[3+ldc*0] + alpha[0]*CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
//		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];
//		D[3+ldd*1] = beta[0]*C[3+ldc*1] + alpha[0]*CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];
//		D[3+ldd*2] = beta[0]*C[3+ldc*2] + alpha[0]*CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		D[3+ldd*3] = beta[0]*C[3+ldc*3] + alpha[0]*CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
//		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];
//		D[2+ldd*0] = beta[0]*C[2+ldc*0] + alpha[0]*CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];
//		D[2+ldd*1] = beta[0]*C[2+ldc*1] + alpha[0]*CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];
		D[2+ldd*2] = beta[0]*C[2+ldc*2] + alpha[0]*CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		D[2+ldd*3] = beta[0]*C[2+ldc*3] + alpha[0]*CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];
//		D[1+ldd*0] = beta[0]*C[1+ldc*0] + alpha[0]*CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];
		D[1+ldd*1] = beta[0]*C[1+ldc*1] + alpha[0]*CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];
		D[1+ldd*2] = beta[0]*C[1+ldc*2] + alpha[0]*CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		D[1+ldd*3] = beta[0]*C[1+ldc*3] + alpha[0]*CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = beta[0]*C[0+ldc*0] + alpha[0]*CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = beta[0]*C[0+ldc*1] + alpha[0]*CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = beta[0]*C[0+ldc*2] + alpha[0]*CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = beta[0]*C[0+ldc*3] + alpha[0]*CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrmm_nn_rl_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 1

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 2

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[0+2*ldb];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 3

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[0+2*ldb];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	b_3 = B[0+3*ldb];
	CC[0+bs*3] += a_0 * b_3;
	CC[1+bs*3] += a_1 * b_3;
	CC[2+bs*3] += a_2 * b_3;
	CC[3+bs*3] += a_3 * b_3;

	A += bs;
	B += 1;
	k += 1;

	store:

	CC[0+bs*0] = alpha[0]*CC[0+bs*0] + beta[0]*C[0+ldc*0];
	CC[1+bs*0] = alpha[0]*CC[1+bs*0] + beta[0]*C[1+ldc*0];
	CC[2+bs*0] = alpha[0]*CC[2+bs*0] + beta[0]*C[2+ldc*0];
	CC[3+bs*0] = alpha[0]*CC[3+bs*0] + beta[0]*C[3+ldc*0];

	CC[0+bs*1] = alpha[0]*CC[0+bs*1] + beta[0]*C[0+ldc*1];
	CC[1+bs*1] = alpha[0]*CC[1+bs*1] + beta[0]*C[1+ldc*1];
	CC[2+bs*1] = alpha[0]*CC[2+bs*1] + beta[0]*C[2+ldc*1];
	CC[3+bs*1] = alpha[0]*CC[3+bs*1] + beta[0]*C[3+ldc*1];

	CC[0+bs*2] = alpha[0]*CC[0+bs*2] + beta[0]*C[0+ldc*2];
	CC[1+bs*2] = alpha[0]*CC[1+bs*2] + beta[0]*C[1+ldc*2];
	CC[2+bs*2] = alpha[0]*CC[2+bs*2] + beta[0]*C[2+ldc*2];
	CC[3+bs*2] = alpha[0]*CC[3+bs*2] + beta[0]*C[3+ldc*2];

	CC[0+bs*3] = alpha[0]*CC[0+bs*3] + beta[0]*C[0+ldc*3];
	CC[1+bs*3] = alpha[0]*CC[1+bs*3] + beta[0]*C[1+ldc*3];
	CC[2+bs*3] = alpha[0]*CC[2+bs*3] + beta[0]*C[2+ldc*3];
	CC[3+bs*3] = alpha[0]*CC[3+bs*3] + beta[0]*C[3+ldc*3];

	double beta1 = 1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax-k, alpha, A, B, ldb, &beta1, CC, bs, D, ldd);

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrmm_nn_rl_4x4_vs_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 1

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 2

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[0+2*ldb];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	A += bs;
	B += 1;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 3

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*ldb];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[0+1*ldb];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[0+2*ldb];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	b_3 = B[0+3*ldb];
	CC[0+bs*3] += a_0 * b_3;
	CC[1+bs*3] += a_1 * b_3;
	CC[2+bs*3] += a_2 * b_3;
	CC[3+bs*3] += a_3 * b_3;

	A += bs;
	B += 1;
	k += 1;

	store:

	CC[0+bs*0] = alpha[0]*CC[0+bs*0] + beta[0]*C[0+ldc*0];
	CC[1+bs*0] = alpha[0]*CC[1+bs*0] + beta[0]*C[1+ldc*0];
	CC[2+bs*0] = alpha[0]*CC[2+bs*0] + beta[0]*C[2+ldc*0];
	CC[3+bs*0] = alpha[0]*CC[3+bs*0] + beta[0]*C[3+ldc*0];

	CC[0+bs*1] = alpha[0]*CC[0+bs*1] + beta[0]*C[0+ldc*1];
	CC[1+bs*1] = alpha[0]*CC[1+bs*1] + beta[0]*C[1+ldc*1];
	CC[2+bs*1] = alpha[0]*CC[2+bs*1] + beta[0]*C[2+ldc*1];
	CC[3+bs*1] = alpha[0]*CC[3+bs*1] + beta[0]*C[3+ldc*1];

	CC[0+bs*2] = alpha[0]*CC[0+bs*2] + beta[0]*C[0+ldc*2];
	CC[1+bs*2] = alpha[0]*CC[1+bs*2] + beta[0]*C[1+ldc*2];
	CC[2+bs*2] = alpha[0]*CC[2+bs*2] + beta[0]*C[2+ldc*2];
	CC[3+bs*2] = alpha[0]*CC[3+bs*2] + beta[0]*C[3+ldc*2];

	CC[0+bs*3] = alpha[0]*CC[0+bs*3] + beta[0]*C[0+ldc*3];
	CC[1+bs*3] = alpha[0]*CC[1+bs*3] + beta[0]*C[1+ldc*3];
	CC[2+bs*3] = alpha[0]*CC[2+bs*3] + beta[0]*C[2+ldc*3];
	CC[3+bs*3] = alpha[0]*CC[3+bs*3] + beta[0]*C[3+ldc*3];

	double beta1 = 1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax-k, alpha, A, B, ldb, &beta1, CC, bs, CC, bs);

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrmm_nt_ru_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 1

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 2

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[2+0*bs];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 3

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[2+0*bs];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	b_3 = B[3+0*bs];
	CC[0+bs*3] += a_0 * b_3;
	CC[1+bs*3] += a_1 * b_3;
	CC[2+bs*3] += a_2 * b_3;
	CC[3+bs*3] += a_3 * b_3;

	A += bs;
	B += bs;
	k += 1;

	store:

	CC[0+bs*0] = alpha[0]*CC[0+bs*0] + beta[0]*C[0+ldc*0];
	CC[1+bs*0] = alpha[0]*CC[1+bs*0] + beta[0]*C[1+ldc*0];
	CC[2+bs*0] = alpha[0]*CC[2+bs*0] + beta[0]*C[2+ldc*0];
	CC[3+bs*0] = alpha[0]*CC[3+bs*0] + beta[0]*C[3+ldc*0];

	CC[0+bs*1] = alpha[0]*CC[0+bs*1] + beta[0]*C[0+ldc*1];
	CC[1+bs*1] = alpha[0]*CC[1+bs*1] + beta[0]*C[1+ldc*1];
	CC[2+bs*1] = alpha[0]*CC[2+bs*1] + beta[0]*C[2+ldc*1];
	CC[3+bs*1] = alpha[0]*CC[3+bs*1] + beta[0]*C[3+ldc*1];

	CC[0+bs*2] = alpha[0]*CC[0+bs*2] + beta[0]*C[0+ldc*2];
	CC[1+bs*2] = alpha[0]*CC[1+bs*2] + beta[0]*C[1+ldc*2];
	CC[2+bs*2] = alpha[0]*CC[2+bs*2] + beta[0]*C[2+ldc*2];
	CC[3+bs*2] = alpha[0]*CC[3+bs*2] + beta[0]*C[3+ldc*2];

	CC[0+bs*3] = alpha[0]*CC[0+bs*3] + beta[0]*C[0+ldc*3];
	CC[1+bs*3] = alpha[0]*CC[1+bs*3] + beta[0]*C[1+ldc*3];
	CC[2+bs*3] = alpha[0]*CC[2+bs*3] + beta[0]*C[2+ldc*3];
	CC[3+bs*3] = alpha[0]*CC[3+bs*3] + beta[0]*C[3+ldc*3];

	double beta1 = 1.0;

	kernel_dgemm_nt_4x4_lib4(kmax-k, alpha, A, B, &beta1, CC, CC);

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrmm_nt_ru_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 1

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 2

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[2+0*bs];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	A += bs;
	B += bs;
	k += 1;

	if(k>=kmax)
		goto store;

	// k = 3

	a_0 = A[0];
	a_1 = A[1];
	a_2 = A[2];
	a_3 = A[3];

	b_0 = B[0+0*bs];
	CC[0+bs*0] += a_0 * b_0;
	CC[1+bs*0] += a_1 * b_0;
	CC[2+bs*0] += a_2 * b_0;
	CC[3+bs*0] += a_3 * b_0;

	b_1 = B[1+0*bs];
	CC[0+bs*1] += a_0 * b_1;
	CC[1+bs*1] += a_1 * b_1;
	CC[2+bs*1] += a_2 * b_1;
	CC[3+bs*1] += a_3 * b_1;

	b_2 = B[2+0*bs];
	CC[0+bs*2] += a_0 * b_2;
	CC[1+bs*2] += a_1 * b_2;
	CC[2+bs*2] += a_2 * b_2;
	CC[3+bs*2] += a_3 * b_2;

	b_3 = B[3+0*bs];
	CC[0+bs*3] += a_0 * b_3;
	CC[1+bs*3] += a_1 * b_3;
	CC[2+bs*3] += a_2 * b_3;
	CC[3+bs*3] += a_3 * b_3;

	A += bs;
	B += bs;
	k += 1;

	store:

	CC[0+bs*0] = alpha[0]*CC[0+bs*0] + beta[0]*C[0+ldc*0];
	CC[1+bs*0] = alpha[0]*CC[1+bs*0] + beta[0]*C[1+ldc*0];
	CC[2+bs*0] = alpha[0]*CC[2+bs*0] + beta[0]*C[2+ldc*0];
	CC[3+bs*0] = alpha[0]*CC[3+bs*0] + beta[0]*C[3+ldc*0];

	CC[0+bs*1] = alpha[0]*CC[0+bs*1] + beta[0]*C[0+ldc*1];
	CC[1+bs*1] = alpha[0]*CC[1+bs*1] + beta[0]*C[1+ldc*1];
	CC[2+bs*1] = alpha[0]*CC[2+bs*1] + beta[0]*C[2+ldc*1];
	CC[3+bs*1] = alpha[0]*CC[3+bs*1] + beta[0]*C[3+ldc*1];

	CC[0+bs*2] = alpha[0]*CC[0+bs*2] + beta[0]*C[0+ldc*2];
	CC[1+bs*2] = alpha[0]*CC[1+bs*2] + beta[0]*C[1+ldc*2];
	CC[2+bs*2] = alpha[0]*CC[2+bs*2] + beta[0]*C[2+ldc*2];
	CC[3+bs*2] = alpha[0]*CC[3+bs*2] + beta[0]*C[3+ldc*2];

	CC[0+bs*3] = alpha[0]*CC[0+bs*3] + beta[0]*C[0+ldc*3];
	CC[1+bs*3] = alpha[0]*CC[1+bs*3] + beta[0]*C[1+ldc*3];
	CC[2+bs*3] = alpha[0]*CC[2+bs*3] + beta[0]*C[2+ldc*3];
	CC[3+bs*3] = alpha[0]*CC[3+bs*3] + beta[0]*C[3+ldc*3];

	double beta1 = 1.0;

	kernel_dgemm_nt_4x4_lib4(kmax-k, alpha, A, B, &beta1, CC, CC);

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dpotrf_nt_l_4x4_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[3+bs*3] = C[3+ldc*3];

	kernel_dpotrf_nt_l_4x4_lib4(kmax, A, B, CC, CC, inv_diag_D);

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dpotrf_nt_l_4x4_vs_lib44c(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D, int m1, int n1)
	{

	const int bs = 4;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	if(m1>=4)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];
		CC[3+bs*0] = C[3+ldc*0];

		if(n1==1)
			goto kernel;

		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];
		CC[3+bs*1] = C[3+ldc*1];

		if(n1==2)
			goto kernel;

		CC[2+bs*2] = C[2+ldc*2];
		CC[3+bs*2] = C[3+ldc*2];

		if(n1==3)
			goto kernel;

		CC[3+bs*3] = C[3+ldc*3];
		}
	else if(m1>=3)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];

		if(n1==1)
			goto kernel;

		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];

		if(n1==2)
			goto kernel;

		CC[2+bs*2] = C[2+ldc*2];
		}
	else if(m1>=2)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];

		if(n1==1)
			goto kernel;

		CC[1+bs*1] = C[1+ldc*1];
		}
	else //if(m1>=1)
		{
		CC[0+bs*0] = C[0+ldc*0];
		}

kernel:
	kernel_dpotrf_nt_l_4x4_vs_lib4(kmax, A, B, CC, CC, inv_diag_D, m1, n1);

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			goto end;

		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			goto end;

		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			goto end;

		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			goto end;

		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			goto end;

		D[2+ldd*2] = CC[2+bs*2];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			goto end;

		D[1+ldd*1] = CC[1+bs*1];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];
		}

end:
	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_inv_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;


	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_inv_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_one_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;


	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_one_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_rl_one_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = E[3+lde*0];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = E[2+lde*0];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = E[1+lde*0];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[1+bs*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[2+bs*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+bs*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[3+bs*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+bs*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+bs*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	if(m1>=4)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];
		CC[3+bs*0] = C[3+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];
		CC[3+bs*1] = C[3+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];
		CC[3+bs*2] = C[3+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		CC[3+bs*3] = C[3+ldc*3];
		}
	else if(m1>=3)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		}
	else if(m1>=2)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		}
	else //if(m1>=1)
		{
		CC[0+bs*0] = C[0+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		}

kernel:
	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[1+bs*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+bs*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+bs*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+bs*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+bs*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+bs*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;
	double beta1  = 1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;
	double beta1  = 1.0;

	if(m1>=4)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];
		CC[3+bs*0] = C[3+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];
		CC[3+bs*1] = C[3+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];
		CC[3+bs*2] = C[3+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		CC[3+bs*3] = C[3+ldc*3];
		}
	else if(m1>=3)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		}
	else if(m1>=2)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		}
	else //if(m1>=1)
		{
		CC[0+bs*0] = C[0+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		}

kernel:
	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, &beta1, CC, CC);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	tmp = E[1+bs*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	tmp = E[2+bs*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+bs*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	tmp = E[3+bs*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+bs*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+bs*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	if(m1>=4)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];
		CC[3+bs*0] = C[3+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];
		CC[3+bs*1] = C[3+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];
		CC[3+bs*2] = C[3+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		CC[3+bs*3] = C[3+ldc*3];
		}
	else if(m1>=3)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];
		CC[2+bs*0] = C[2+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];
		CC[2+bs*1] = C[2+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];
		CC[2+bs*2] = C[2+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		CC[2+bs*3] = C[2+ldc*3];
		}
	else if(m1>=2)
		{
		CC[0+bs*0] = C[0+ldc*0];
		CC[1+bs*0] = C[1+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];
		CC[1+bs*1] = C[1+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];
		CC[1+bs*2] = C[1+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		CC[1+bs*3] = C[1+ldc*3];
		}
	else //if(m1>=1)
		{
		CC[0+bs*0] = C[0+ldc*0];

		if(n1==1)
			goto kernel;

		CC[0+bs*1] = C[0+ldc*1];

		if(n1==2)
			goto kernel;

		CC[0+bs*2] = C[0+ldc*2];

		if(n1==3)
			goto kernel;

		CC[0+bs*3] = C[0+ldc*3];
		}

kernel:
	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	if(n1==1)
		goto store;
	
	tmp = E[1+bs*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+bs*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+bs*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+bs*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+bs*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+bs*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;


	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1==1)
		goto store;
	
	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1==1)
		goto store;
	
	tmp = E[1+lde*0];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	if(n1==2)
		goto store;
	
	tmp = E[2+lde*0];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[2+lde*1];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	if(n1==3)
		goto store;
	
	tmp = E[3+lde*0];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[3+lde*1];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[3+lde*2];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_inv_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_inv_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	if(n1==1)
		goto store;
	
	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;
	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;

	if(n1==2)
		goto store;
	
	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;
	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;

	if(n1==3)
		goto store;
	
	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;
	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_one_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1==1)
		goto store;
	
	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	if(n1==2)
		goto store;
	
	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	if(n1==3)
		goto store;
	
	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_one_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nn_ru_one_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nn_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1==1)
		goto store;
	
	tmp = E[0+lde*1];
	CC[0+bs*1] -= CC[0+bs*0] * tmp;
	CC[1+bs*1] -= CC[1+bs*0] * tmp;
	CC[2+bs*1] -= CC[2+bs*0] * tmp;
	CC[3+bs*1] -= CC[3+bs*0] * tmp;

	if(n1==2)
		goto store;
	
	tmp = E[0+lde*2];
	CC[0+bs*2] -= CC[0+bs*0] * tmp;
	CC[1+bs*2] -= CC[1+bs*0] * tmp;
	CC[2+bs*2] -= CC[2+bs*0] * tmp;
	CC[3+bs*2] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*2] -= CC[0+bs*1] * tmp;
	CC[1+bs*2] -= CC[1+bs*1] * tmp;
	CC[2+bs*2] -= CC[2+bs*1] * tmp;
	CC[3+bs*2] -= CC[3+bs*1] * tmp;

	if(n1==3)
		goto store;
	
	tmp = E[0+lde*3];
	CC[0+bs*3] -= CC[0+bs*0] * tmp;
	CC[1+bs*3] -= CC[1+bs*0] * tmp;
	CC[2+bs*3] -= CC[2+bs*0] * tmp;
	CC[3+bs*3] -= CC[3+bs*0] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*3] -= CC[0+bs*1] * tmp;
	CC[1+bs*3] -= CC[1+bs*1] * tmp;
	CC[2+bs*3] -= CC[2+bs*1] * tmp;
	CC[3+bs*3] -= CC[3+bs*1] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*3] -= CC[0+bs*2] * tmp;
	CC[1+bs*3] -= CC[1+bs*2] * tmp;
	CC[2+bs*3] -= CC[2+bs*2] * tmp;
	CC[3+bs*3] -= CC[3+bs*2] * tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+bs*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+bs*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+bs*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+bs*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+bs*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+bs*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	if(n1<=3)
		goto n3;

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+bs*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+bs*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+bs*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+bs*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+bs*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+bs*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;


	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = inv_diag_E[3];
	CC[0+bs*3] *= tmp;
	CC[1+bs*3] *= tmp;
	CC[2+bs*3] *= tmp;
	CC[3+bs*3] *= tmp;
	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = inv_diag_E[2];
	CC[0+bs*2] *= tmp;
	CC[1+bs*2] *= tmp;
	CC[2+bs*2] *= tmp;
	CC[3+bs*2] *= tmp;
	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = inv_diag_E[1];
	CC[0+bs*1] *= tmp;
	CC[1+bs*1] *= tmp;
	CC[2+bs*1] *= tmp;
	CC[3+bs*1] *= tmp;
	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	tmp = inv_diag_E[0];
	CC[0+bs*0] *= tmp;
	CC[1+bs*0] *= tmp;
	CC[2+bs*0] *= tmp;
	CC[3+bs*0] *= tmp;

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	tmp = E[0+bs*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+bs*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+bs*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = E[0+bs*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+bs*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = E[0+bs*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_vs_lib44c4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	CC[0+bs*0] = C[0+ldc*0];
	CC[1+bs*0] = C[1+ldc*0];
	CC[2+bs*0] = C[2+ldc*0];
	CC[3+bs*0] = C[3+ldc*0];

	CC[0+bs*1] = C[0+ldc*1];
	CC[1+bs*1] = C[1+ldc*1];
	CC[2+bs*1] = C[2+ldc*1];
	CC[3+bs*1] = C[3+ldc*1];

	CC[0+bs*2] = C[0+ldc*2];
	CC[1+bs*2] = C[1+ldc*2];
	CC[2+bs*2] = C[2+ldc*2];
	CC[3+bs*2] = C[3+ldc*2];

	CC[0+bs*3] = C[0+ldc*3];
	CC[1+bs*3] = C[1+ldc*3];
	CC[2+bs*3] = C[2+ldc*3];
	CC[3+bs*3] = C[3+ldc*3];

	kernel_dgemm_nt_4x4_lib4(kmax, &alpha1, A, B, beta, CC, CC);

	if(n1<=3)
		goto n3;

	tmp = E[0+bs*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+bs*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+bs*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = E[0+bs*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+bs*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = E[0+bs*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;


	D[0+bs*0] = CC[0+bs*0];
	D[1+bs*0] = CC[1+bs*0];
	D[2+bs*0] = CC[2+bs*0];
	D[3+bs*0] = CC[3+bs*0];

	D[0+bs*1] = CC[0+bs*1];
	D[1+bs*1] = CC[1+bs*1];
	D[2+bs*1] = CC[2+bs*1];
	D[3+bs*1] = CC[3+bs*1];

	D[0+bs*2] = CC[0+bs*2];
	D[1+bs*2] = CC[1+bs*2];
	D[2+bs*2] = CC[2+bs*2];
	D[3+bs*2] = CC[3+bs*2];

	D[0+bs*3] = CC[0+bs*3];
	D[1+bs*3] = CC[1+bs*3];
	D[2+bs*3] = CC[2+bs*3];
	D[3+bs*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c(int kmax, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, bs, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	store:

	if(m1>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;


	D[0+ldd*0] = CC[0+bs*0];
	D[1+ldd*0] = CC[1+bs*0];
	D[2+ldd*0] = CC[2+bs*0];
	D[3+ldd*0] = CC[3+bs*0];

	D[0+ldd*1] = CC[0+bs*1];
	D[1+ldd*1] = CC[1+bs*1];
	D[2+ldd*1] = CC[2+bs*1];
	D[3+ldd*1] = CC[3+bs*1];

	D[0+ldd*2] = CC[0+bs*2];
	D[1+ldd*2] = CC[1+bs*2];
	D[2+ldd*2] = CC[2+bs*2];
	D[3+ldd*2] = CC[3+bs*2];

	D[0+ldd*3] = CC[0+bs*3];
	D[1+ldd*3] = CC[1+bs*3];
	D[2+ldd*3] = CC[2+bs*3];
	D[3+ldd*3] = CC[3+bs*3];

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X86_AMD_BARCELONA) || defined(TARGET_X86_AMD_JAGUAR) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV7A_ARM_CORTEX_A7) || defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dtrsm_nt_ru_one_4x4_vs_lib4ccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1)
	{

	const int bs = 4;

	double tmp;

#if defined(TARGET_GENERIC)
	double CC[16] = {0};
#else
#if defined (_MSC_VER)
	double CC[16] __declspec(align(64)) = {0};
#else
	double CC[16] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	double alpha1 = -1.0;

	kernel_dgemm_nt_4x4_lib4cc(kmax, &alpha1, A, B, ldb, beta, C, ldc, CC, bs);

	if(n1<=3)
		goto n3;

	tmp = E[0+lde*3];
	CC[0+bs*0] -= CC[0+bs*3] * tmp;
	CC[1+bs*0] -= CC[1+bs*3] * tmp;
	CC[2+bs*0] -= CC[2+bs*3] * tmp;
	CC[3+bs*0] -= CC[3+bs*3] * tmp;
	tmp = E[1+lde*3];
	CC[0+bs*1] -= CC[0+bs*3] * tmp;
	CC[1+bs*1] -= CC[1+bs*3] * tmp;
	CC[2+bs*1] -= CC[2+bs*3] * tmp;
	CC[3+bs*1] -= CC[3+bs*3] * tmp;
	tmp = E[2+lde*3];
	CC[0+bs*2] -= CC[0+bs*3] * tmp;
	CC[1+bs*2] -= CC[1+bs*3] * tmp;
	CC[2+bs*2] -= CC[2+bs*3] * tmp;
	CC[3+bs*2] -= CC[3+bs*3] * tmp;

n3:
	if(n1<=2)
		goto n2;

	tmp = E[0+lde*2];
	CC[0+bs*0] -= CC[0+bs*2] * tmp;
	CC[1+bs*0] -= CC[1+bs*2] * tmp;
	CC[2+bs*0] -= CC[2+bs*2] * tmp;
	CC[3+bs*0] -= CC[3+bs*2] * tmp;
	tmp = E[1+lde*2];
	CC[0+bs*1] -= CC[0+bs*2] * tmp;
	CC[1+bs*1] -= CC[1+bs*2] * tmp;
	CC[2+bs*1] -= CC[2+bs*2] * tmp;
	CC[3+bs*1] -= CC[3+bs*2] * tmp;

n2:
	if(n1<=1)
		goto n1;

	tmp = E[0+lde*1];
	CC[0+bs*0] -= CC[0+bs*1] * tmp;
	CC[1+bs*0] -= CC[1+bs*1] * tmp;
	CC[2+bs*0] -= CC[2+bs*1] * tmp;
	CC[3+bs*0] -= CC[3+bs*1] * tmp;

n1:

	store:

	if(m1>=4)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];
		D[3+ldd*0] = CC[3+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];
		D[3+ldd*1] = CC[3+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];
		D[3+ldd*2] = CC[3+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		D[3+ldd*3] = CC[3+bs*3];
		}
	else if(m1>=3)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];
		D[2+ldd*0] = CC[2+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];
		D[2+ldd*1] = CC[2+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];
		D[2+ldd*2] = CC[2+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		D[2+ldd*3] = CC[2+bs*3];
		}
	else if(m1>=2)
		{
		D[0+ldd*0] = CC[0+bs*0];
		D[1+ldd*0] = CC[1+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];
		D[1+ldd*1] = CC[1+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];
		D[1+ldd*2] = CC[1+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		D[1+ldd*3] = CC[1+bs*3];
		}
	else //if(m1>=1)
		{
		D[0+ldd*0] = CC[0+bs*0];

		if(n1==1)
			return;

		D[0+ldd*1] = CC[0+bs*1];

		if(n1==2)
			return;

		D[0+ldd*2] = CC[0+bs*2];

		if(n1==3)
			return;

		D[0+ldd*3] = CC[0+bs*3];
		}

	return;

	}
#endif




