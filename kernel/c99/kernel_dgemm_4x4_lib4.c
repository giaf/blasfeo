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

#include <math.h>



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgemm_nt_4x4_gen_lib4(int kmax, double *alpha, double *A, double *B, double *beta, int offsetC, double *C0, int sdc, int offsetD, double *D0, int sdd, int m0, int m1, int n0, int n1)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	double
		*C1, *D1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	if(offsetC==0)
		{
		c_00 = beta[0]*C0[0+bs*0] + alpha[0]*c_00;
		c_10 = beta[0]*C0[1+bs*0] + alpha[0]*c_10;
		c_20 = beta[0]*C0[2+bs*0] + alpha[0]*c_20;
		c_30 = beta[0]*C0[3+bs*0] + alpha[0]*c_30;

		c_01 = beta[0]*C0[0+bs*1] + alpha[0]*c_01;
		c_11 = beta[0]*C0[1+bs*1] + alpha[0]*c_11;
		c_21 = beta[0]*C0[2+bs*1] + alpha[0]*c_21;
		c_31 = beta[0]*C0[3+bs*1] + alpha[0]*c_31;

		c_02 = beta[0]*C0[0+bs*2] + alpha[0]*c_02;
		c_12 = beta[0]*C0[1+bs*2] + alpha[0]*c_12;
		c_22 = beta[0]*C0[2+bs*2] + alpha[0]*c_22;
		c_32 = beta[0]*C0[3+bs*2] + alpha[0]*c_32;

		c_03 = beta[0]*C0[0+bs*3] + alpha[0]*c_03;
		c_13 = beta[0]*C0[1+bs*3] + alpha[0]*c_13;
		c_23 = beta[0]*C0[2+bs*3] + alpha[0]*c_23;
		c_33 = beta[0]*C0[3+bs*3] + alpha[0]*c_33;
		}
	else if(offsetC==1)
		{
		C1 = C0 + sdc*bs;

		c_00 = beta[0]*C0[1+bs*0] + alpha[0]*c_00;
		c_10 = beta[0]*C0[2+bs*0] + alpha[0]*c_10;
		c_20 = beta[0]*C0[3+bs*0] + alpha[0]*c_20;
		c_30 = beta[0]*C1[0+bs*0] + alpha[0]*c_30;

		c_01 = beta[0]*C0[1+bs*1] + alpha[0]*c_01;
		c_11 = beta[0]*C0[2+bs*1] + alpha[0]*c_11;
		c_21 = beta[0]*C0[3+bs*1] + alpha[0]*c_21;
		c_31 = beta[0]*C1[0+bs*1] + alpha[0]*c_31;

		c_02 = beta[0]*C0[1+bs*2] + alpha[0]*c_02;
		c_12 = beta[0]*C0[2+bs*2] + alpha[0]*c_12;
		c_22 = beta[0]*C0[3+bs*2] + alpha[0]*c_22;
		c_32 = beta[0]*C1[0+bs*2] + alpha[0]*c_32;

		c_03 = beta[0]*C0[1+bs*3] + alpha[0]*c_03;
		c_13 = beta[0]*C0[2+bs*3] + alpha[0]*c_13;
		c_23 = beta[0]*C0[3+bs*3] + alpha[0]*c_23;
		c_33 = beta[0]*C1[0+bs*3] + alpha[0]*c_33;
		}
	else if(offsetC==2)
		{
		C1 = C0 + sdc*bs;

		c_00 = beta[0]*C0[2+bs*0] + alpha[0]*c_00;
		c_10 = beta[0]*C0[3+bs*0] + alpha[0]*c_10;
		c_20 = beta[0]*C1[0+bs*0] + alpha[0]*c_20;
		c_30 = beta[0]*C1[1+bs*0] + alpha[0]*c_30;

		c_01 = beta[0]*C0[2+bs*1] + alpha[0]*c_01;
		c_11 = beta[0]*C0[3+bs*1] + alpha[0]*c_11;
		c_21 = beta[0]*C1[0+bs*1] + alpha[0]*c_21;
		c_31 = beta[0]*C1[1+bs*1] + alpha[0]*c_31;

		c_02 = beta[0]*C0[2+bs*2] + alpha[0]*c_02;
		c_12 = beta[0]*C0[3+bs*2] + alpha[0]*c_12;
		c_22 = beta[0]*C1[0+bs*2] + alpha[0]*c_22;
		c_32 = beta[0]*C1[1+bs*2] + alpha[0]*c_32;

		c_03 = beta[0]*C0[2+bs*3] + alpha[0]*c_03;
		c_13 = beta[0]*C0[3+bs*3] + alpha[0]*c_13;
		c_23 = beta[0]*C1[0+bs*3] + alpha[0]*c_23;
		c_33 = beta[0]*C1[1+bs*3] + alpha[0]*c_33;
		}
	else //if(offsetC==3)
		{
		C1 = C0 + sdc*bs;

		c_00 = beta[0]*C0[3+bs*0] + alpha[0]*c_00;
		c_10 = beta[0]*C1[0+bs*0] + alpha[0]*c_10;
		c_20 = beta[0]*C1[1+bs*0] + alpha[0]*c_20;
		c_30 = beta[0]*C1[2+bs*0] + alpha[0]*c_30;

		c_01 = beta[0]*C0[3+bs*1] + alpha[0]*c_01;
		c_11 = beta[0]*C1[0+bs*1] + alpha[0]*c_11;
		c_21 = beta[0]*C1[1+bs*1] + alpha[0]*c_21;
		c_31 = beta[0]*C1[2+bs*1] + alpha[0]*c_31;

		c_02 = beta[0]*C0[3+bs*2] + alpha[0]*c_02;
		c_12 = beta[0]*C1[0+bs*2] + alpha[0]*c_12;
		c_22 = beta[0]*C1[1+bs*2] + alpha[0]*c_22;
		c_32 = beta[0]*C1[2+bs*2] + alpha[0]*c_32;

		c_03 = beta[0]*C0[3+bs*3] + alpha[0]*c_03;
		c_13 = beta[0]*C1[0+bs*3] + alpha[0]*c_13;
		c_23 = beta[0]*C1[1+bs*3] + alpha[0]*c_23;
		c_33 = beta[0]*C1[2+bs*3] + alpha[0]*c_33;
		}
	
	// shift sol for cols
	if(n0>0)
		{
		if(n0==1)
			{
			c_00 = c_01;
			c_10 = c_11;
			c_20 = c_21;
			c_30 = c_31;

			c_01 = c_02;
			c_11 = c_12;
			c_21 = c_22;
			c_31 = c_32;

			c_02 = c_03;
			c_12 = c_13;
			c_22 = c_23;
			c_32 = c_33;

			D0 += 1*bs;
			}
		else if(n0==2)
			{
			c_00 = c_02;
			c_10 = c_12;
			c_20 = c_22;
			c_30 = c_32;

			c_01 = c_03;
			c_11 = c_13;
			c_21 = c_23;
			c_31 = c_33;

			D0 += 2*bs;
			}
		else //if(n0==3)
			{
			c_00 = c_03;
			c_10 = c_13;
			c_20 = c_23;
			c_30 = c_33;

			D0 += 3*bs;
			}
		}

	int kn = n1 - n0;

	if(offsetD==0)
		{
		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[0+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[1+bs*0] = c_10;
		if(m0<=2 & m1>2) D0[2+bs*0] = c_20;
		if(m0<=3 & m1>3) D0[3+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[0+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[1+bs*1] = c_11;
		if(m0<=2 & m1>2) D0[2+bs*1] = c_21;
		if(m0<=3 & m1>3) D0[3+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[0+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[1+bs*2] = c_12;
		if(m0<=2 & m1>2) D0[2+bs*2] = c_22;
		if(m0<=3 & m1>3) D0[3+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[0+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[1+bs*3] = c_13;
		if(m0<=2 & m1>2) D0[2+bs*3] = c_23;
		if(m0<=3 & m1>3) D0[3+bs*3] = c_33;
		}
	else if(offsetD==1)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[1+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[2+bs*0] = c_10;
		if(m0<=2 & m1>2) D0[3+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[0+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[1+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[2+bs*1] = c_11;
		if(m0<=2 & m1>2) D0[3+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[0+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[1+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[2+bs*2] = c_12;
		if(m0<=2 & m1>2) D0[3+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[0+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[1+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[2+bs*3] = c_13;
		if(m0<=2 & m1>2) D0[3+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[0+bs*3] = c_33;
		}
	else if(offsetD==2)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[2+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[3+bs*0] = c_10;
		if(m0<=2 & m1>2) D1[0+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[1+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[2+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[3+bs*1] = c_11;
		if(m0<=2 & m1>2) D1[0+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[1+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[2+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[3+bs*2] = c_12;
		if(m0<=2 & m1>2) D1[0+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[1+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[2+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[3+bs*3] = c_13;
		if(m0<=2 & m1>2) D1[0+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[1+bs*3] = c_33;
		}
	else //if(offsetD==3)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[3+bs*0] = c_00;
		if(m0<=1 & m1>1) D1[0+bs*0] = c_10;
		if(m0<=2 & m1>2) D1[1+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[2+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[3+bs*1] = c_01;
		if(m0<=1 & m1>1) D1[0+bs*1] = c_11;
		if(m0<=2 & m1>2) D1[1+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[2+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[3+bs*2] = c_02;
		if(m0<=1 & m1>1) D1[0+bs*2] = c_12;
		if(m0<=2 & m1>2) D1[1+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[2+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[3+bs*3] = c_03;
		if(m0<=1 & m1>1) D1[0+bs*3] = c_13;
		if(m0<=2 & m1>2) D1[1+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[2+bs*3] = c_33;
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgemm_nt_4x4_vs_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = beta[0]*C[0+bs*0] + alpha[0]*c_00;
	c_10 = beta[0]*C[1+bs*0] + alpha[0]*c_10;
	c_20 = beta[0]*C[2+bs*0] + alpha[0]*c_20;
	c_30 = beta[0]*C[3+bs*0] + alpha[0]*c_30;

	c_01 = beta[0]*C[0+bs*1] + alpha[0]*c_01;
	c_11 = beta[0]*C[1+bs*1] + alpha[0]*c_11;
	c_21 = beta[0]*C[2+bs*1] + alpha[0]*c_21;
	c_31 = beta[0]*C[3+bs*1] + alpha[0]*c_31;

	c_02 = beta[0]*C[0+bs*2] + alpha[0]*c_02;
	c_12 = beta[0]*C[1+bs*2] + alpha[0]*c_12;
	c_22 = beta[0]*C[2+bs*2] + alpha[0]*c_22;
	c_32 = beta[0]*C[3+bs*2] + alpha[0]*c_32;

	c_03 = beta[0]*C[0+bs*3] + alpha[0]*c_03;
	c_13 = beta[0]*C[1+bs*3] + alpha[0]*c_13;
	c_23 = beta[0]*C[2+bs*3] + alpha[0]*c_23;
	c_33 = beta[0]*C[3+bs*3] + alpha[0]*c_33;

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dgemm_nt_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	kernel_dgemm_nt_4x4_vs_lib4(kmax, alpha, A, B, beta, C, D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgemm_nn_4x4_vs_lib4(int kmax, double *alpha, double *A, double *B, int sdb, double *beta, double *C, double *D, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[4];
		b_2 = B[8];
		b_3 = B[12];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1];
		b_1 = B[5];
		b_2 = B[9];
		b_3 = B[13];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2];
		b_1 = B[6];
		b_2 = B[10];
		b_3 = B[14];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3];
		b_1 = B[7];
		b_2 = B[11];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 4*sdb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[4];
		b_2 = B[8];
		b_3 = B[12];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 1;

		}
	
	c_00 = beta[0]*C[0+bs*0] + alpha[0]*c_00;
	c_10 = beta[0]*C[1+bs*0] + alpha[0]*c_10;
	c_20 = beta[0]*C[2+bs*0] + alpha[0]*c_20;
	c_30 = beta[0]*C[3+bs*0] + alpha[0]*c_30;

	c_01 = beta[0]*C[0+bs*1] + alpha[0]*c_01;
	c_11 = beta[0]*C[1+bs*1] + alpha[0]*c_11;
	c_21 = beta[0]*C[2+bs*1] + alpha[0]*c_21;
	c_31 = beta[0]*C[3+bs*1] + alpha[0]*c_31;

	c_02 = beta[0]*C[0+bs*2] + alpha[0]*c_02;
	c_12 = beta[0]*C[1+bs*2] + alpha[0]*c_12;
	c_22 = beta[0]*C[2+bs*2] + alpha[0]*c_22;
	c_32 = beta[0]*C[3+bs*2] + alpha[0]*c_32;

	c_03 = beta[0]*C[0+bs*3] + alpha[0]*c_03;
	c_13 = beta[0]*C[1+bs*3] + alpha[0]*c_13;
	c_23 = beta[0]*C[2+bs*3] + alpha[0]*c_23;
	c_33 = beta[0]*C[3+bs*3] + alpha[0]*c_33;

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgemm_nn_4x4_lib4(int kmax, double *alpha, double *A, double *B, int sdb, double *beta, double *C, double *D)
	{
	kernel_dgemm_nn_4x4_vs_lib4(kmax, alpha, A, B, sdb, beta, C, D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dsyrk_nt_l_4x4_vs_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, //c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, //c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, //c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

//		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

//		c_02 += a_0 * b_2;
//		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

//		c_03 += a_0 * b_3;
//		c_13 += a_1 * b_3;
//		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

//		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

//		c_02 += a_0 * b_2;
//		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

//		c_03 += a_0 * b_3;
//		c_13 += a_1 * b_3;
//		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

//		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

//		c_02 += a_0 * b_2;
//		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

//		c_03 += a_0 * b_3;
//		c_13 += a_1 * b_3;
//		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

//		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

//		c_02 += a_0 * b_2;
//		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

//		c_03 += a_0 * b_3;
//		c_13 += a_1 * b_3;
//		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

//		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

//		c_02 += a_0 * b_2;
//		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

//		c_03 += a_0 * b_3;
//		c_13 += a_1 * b_3;
//		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = beta[0]*C[0+bs*0] + alpha[0]*c_00;
	c_10 = beta[0]*C[1+bs*0] + alpha[0]*c_10;
	c_20 = beta[0]*C[2+bs*0] + alpha[0]*c_20;
	c_30 = beta[0]*C[3+bs*0] + alpha[0]*c_30;

//	c_01 = beta[0]*C[0+bs*1] + alpha[0]*c_01;
	c_11 = beta[0]*C[1+bs*1] + alpha[0]*c_11;
	c_21 = beta[0]*C[2+bs*1] + alpha[0]*c_21;
	c_31 = beta[0]*C[3+bs*1] + alpha[0]*c_31;

//	c_02 = beta[0]*C[0+bs*2] + alpha[0]*c_02;
//	c_12 = beta[0]*C[1+bs*2] + alpha[0]*c_12;
	c_22 = beta[0]*C[2+bs*2] + alpha[0]*c_22;
	c_32 = beta[0]*C[3+bs*2] + alpha[0]*c_32;

//	c_03 = beta[0]*C[0+bs*3] + alpha[0]*c_03;
//	c_13 = beta[0]*C[1+bs*3] + alpha[0]*c_13;
//	c_23 = beta[0]*C[2+bs*3] + alpha[0]*c_23;
	c_33 = beta[0]*C[3+bs*3] + alpha[0]*c_33;

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
//		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
//		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

//		if(kn==2)
//			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

//		if(kn==1)
//			return;

//		D[0+bs*1] = c_01;

//		if(kn==2)
//			return;

//		D[0+bs*2] = c_02;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dsyrk_nt_l_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	kernel_dsyrk_nt_l_4x4_vs_lib4(kmax, alpha, A, B, beta, C, D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrmm_nt_ru_4x4_vs_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	k = 0;

	// k = 0
	if(kmax>0)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		A += 4;
		B += 4;
		k++;
		}

	// k = 1
	if(kmax>0)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		A += 4;
		B += 4;
		k++;
		}

	// k = 2
	if(kmax>0)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		A += 4;
		B += 4;
		k++;
		}

	for(; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = beta[0]*C[0+bs*0] + alpha[0]*c_00;
	c_10 = beta[0]*C[1+bs*0] + alpha[0]*c_10;
	c_20 = beta[0]*C[2+bs*0] + alpha[0]*c_20;
	c_30 = beta[0]*C[3+bs*0] + alpha[0]*c_30;

	c_01 = beta[0]*C[0+bs*1] + alpha[0]*c_01;
	c_11 = beta[0]*C[1+bs*1] + alpha[0]*c_11;
	c_21 = beta[0]*C[2+bs*1] + alpha[0]*c_21;
	c_31 = beta[0]*C[3+bs*1] + alpha[0]*c_31;

	c_02 = beta[0]*C[0+bs*2] + alpha[0]*c_02;
	c_12 = beta[0]*C[1+bs*2] + alpha[0]*c_12;
	c_22 = beta[0]*C[2+bs*2] + alpha[0]*c_22;
	c_32 = beta[0]*C[3+bs*2] + alpha[0]*c_32;

	c_03 = beta[0]*C[0+bs*3] + alpha[0]*c_03;
	c_13 = beta[0]*C[1+bs*3] + alpha[0]*c_13;
	c_23 = beta[0]*C[2+bs*3] + alpha[0]*c_23;
	c_33 = beta[0]*C[3+bs*3] + alpha[0]*c_33;

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif




#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrmm_nt_ru_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	kernel_dtrmm_nt_ru_4x4_vs_lib4(k, alpha, A, B, beta, C, D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrmm_nn_rl_4x4_gen_lib4(int kmax, double *alpha, double *A, int offsetB, double *B, int sdb, int offsetD, double *D0, int sdd, int m0, int m1, int n0, int n1)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	double *D1;
	
	int k;

	B += offsetB;

	k = 0;

	if(offsetB==0)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 1

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 2

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 3

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		}
	else if(offsetB==1)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 1

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 2

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		}
	else if(offsetB==2)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 1

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 2

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 3

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 4

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 5

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		}
	else // if(offetB==3)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 1

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 2

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 3

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 1;
		k += 1;

		if(k>=kmax)
			goto store;

		// k = 4

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		b_1 = B[4];
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		b_2 = B[8];
		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		b_3 = B[12];
		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 4*sdb-3;
		k += 1;

		}

	for(; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[4];
		b_2 = B[8];
		b_3 = B[12];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[1];
		b_1 = B[5];
		b_2 = B[9];
		b_3 = B[13];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[2];
		b_1 = B[6];
		b_2 = B[10];
		b_3 = B[14];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[3];
		b_1 = B[7];
		b_2 = B[11];
		b_3 = B[15];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 16;
		B += 4*sdb;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[4];
		b_2 = B[8];
		b_3 = B[12];

		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;

		A += 4;
		B += 1;

		}
	
	store:
	
	c_00 = alpha[0]*c_00;
	c_10 = alpha[0]*c_10;
	c_20 = alpha[0]*c_20;
	c_30 = alpha[0]*c_30;

	c_01 = alpha[0]*c_01;
	c_11 = alpha[0]*c_11;
	c_21 = alpha[0]*c_21;
	c_31 = alpha[0]*c_31;

	c_02 = alpha[0]*c_02;
	c_12 = alpha[0]*c_12;
	c_22 = alpha[0]*c_22;
	c_32 = alpha[0]*c_32;

	c_03 = alpha[0]*c_03;
	c_13 = alpha[0]*c_13;
	c_23 = alpha[0]*c_23;
	c_33 = alpha[0]*c_33;

	// shift sol for cols
	if(n0>0)
		{
		if(n0==1)
			{
			c_00 = c_01;
			c_10 = c_11;
			c_20 = c_21;
			c_30 = c_31;

			c_01 = c_02;
			c_11 = c_12;
			c_21 = c_22;
			c_31 = c_32;

			c_02 = c_03;
			c_12 = c_13;
			c_22 = c_23;
			c_32 = c_33;

			D0 += 1*bs;
			}
		else if(n0==2)
			{
			c_00 = c_02;
			c_10 = c_12;
			c_20 = c_22;
			c_30 = c_32;

			c_01 = c_03;
			c_11 = c_13;
			c_21 = c_23;
			c_31 = c_33;

			D0 += 2*bs;
			}
		else //if(n0==3)
			{
			c_00 = c_03;
			c_10 = c_13;
			c_20 = c_23;
			c_30 = c_33;

			D0 += 3*bs;
			}
		}

	int kn = n1 - n0;

	if(offsetD==0)
		{
		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[0+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[1+bs*0] = c_10;
		if(m0<=2 & m1>2) D0[2+bs*0] = c_20;
		if(m0<=3 & m1>3) D0[3+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[0+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[1+bs*1] = c_11;
		if(m0<=2 & m1>2) D0[2+bs*1] = c_21;
		if(m0<=3 & m1>3) D0[3+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[0+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[1+bs*2] = c_12;
		if(m0<=2 & m1>2) D0[2+bs*2] = c_22;
		if(m0<=3 & m1>3) D0[3+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[0+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[1+bs*3] = c_13;
		if(m0<=2 & m1>2) D0[2+bs*3] = c_23;
		if(m0<=3 & m1>3) D0[3+bs*3] = c_33;
		}
	else if(offsetD==1)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[1+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[2+bs*0] = c_10;
		if(m0<=2 & m1>2) D0[3+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[0+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[1+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[2+bs*1] = c_11;
		if(m0<=2 & m1>2) D0[3+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[0+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[1+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[2+bs*2] = c_12;
		if(m0<=2 & m1>2) D0[3+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[0+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[1+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[2+bs*3] = c_13;
		if(m0<=2 & m1>2) D0[3+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[0+bs*3] = c_33;
		}
	else if(offsetD==2)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[2+bs*0] = c_00;
		if(m0<=1 & m1>1) D0[3+bs*0] = c_10;
		if(m0<=2 & m1>2) D1[0+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[1+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[2+bs*1] = c_01;
		if(m0<=1 & m1>1) D0[3+bs*1] = c_11;
		if(m0<=2 & m1>2) D1[0+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[1+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[2+bs*2] = c_02;
		if(m0<=1 & m1>1) D0[3+bs*2] = c_12;
		if(m0<=2 & m1>2) D1[0+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[1+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[2+bs*3] = c_03;
		if(m0<=1 & m1>1) D0[3+bs*3] = c_13;
		if(m0<=2 & m1>2) D1[0+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[1+bs*3] = c_33;
		}
	else //if(offsetD==3)
		{
		D1 = D0 + sdd*bs;

		if(kn<=0)
			return;

		if(m0<=0 & m1>0) D0[3+bs*0] = c_00;
		if(m0<=1 & m1>1) D1[0+bs*0] = c_10;
		if(m0<=2 & m1>2) D1[1+bs*0] = c_20;
		if(m0<=3 & m1>3) D1[2+bs*0] = c_30;

		if(kn<=1)
			return;

		if(m0<=0 & m1>0) D0[3+bs*1] = c_01;
		if(m0<=1 & m1>1) D1[0+bs*1] = c_11;
		if(m0<=2 & m1>2) D1[1+bs*1] = c_21;
		if(m0<=3 & m1>3) D1[2+bs*1] = c_31;

		if(kn<=2)
			return;

		if(m0<=0 & m1>0) D0[3+bs*2] = c_02;
		if(m0<=1 & m1>1) D1[0+bs*2] = c_12;
		if(m0<=2 & m1>2) D1[1+bs*2] = c_22;
		if(m0<=3 & m1>3) D1[2+bs*2] = c_32;

		if(kn<=3)
			return;

		if(m0<=0 & m1>0) D0[3+bs*3] = c_03;
		if(m0<=1 & m1>1) D1[0+bs*3] = c_13;
		if(m0<=2 & m1>2) D1[1+bs*3] = c_23;
		if(m0<=3 & m1>3) D1[2+bs*3] = c_33;
		}
	
	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrmm_nn_rl_4x4_lib4(int kmax, double *alpha, double *A, int offsetB, double *B, int sdb, double *D)
	{
	kernel_dtrmm_nn_rl_4x4_gen_lib4(kmax, alpha, A, offsetB, B, sdb, 0, D, 0, 0, 4, 0, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dpotrf_nt_l_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		tmp,
		c_00=0, //c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, //c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, //c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

//		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

//		c_02 -= a_0 * b_2;
//		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

//		c_03 -= a_0 * b_3;
//		c_13 -= a_1 * b_3;
//		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

//		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

//		c_02 -= a_0 * b_2;
//		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

//		c_03 -= a_0 * b_3;
//		c_13 -= a_1 * b_3;
//		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

//		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

//		c_02 -= a_0 * b_2;
//		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

//		c_03 -= a_0 * b_3;
//		c_13 -= a_1 * b_3;
//		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

//		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

//		c_02 -= a_0 * b_2;
//		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

//		c_03 -= a_0 * b_3;
//		c_13 -= a_1 * b_3;
//		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

//		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

//		c_02 -= a_0 * b_2;
//		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

//		c_03 -= a_0 * b_3;
//		c_13 -= a_1 * b_3;
//		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = C[0+bs*0] + c_00;
	c_10 = C[1+bs*0] + c_10;
	c_20 = C[2+bs*0] + c_20;
	c_30 = C[3+bs*0] + c_30;

//	c_01 = C[0+bs*1] + c_01;
	c_11 = C[1+bs*1] + c_11;
	c_21 = C[2+bs*1] + c_21;
	c_31 = C[3+bs*1] + c_31;

//	c_02 = C[0+bs*2] + c_02;
//	c_12 = C[1+bs*2] + c_12;
	c_22 = C[2+bs*2] + c_22;
	c_32 = C[3+bs*2] + c_32;

//	c_03 = C[0+bs*3] + c_03;
//	c_13 = C[1+bs*3] + c_13;
//	c_23 = C[2+bs*3] + c_23;
	c_33 = C[3+bs*3] + c_33;

	if(c_00>0)
		{
		c_00 = sqrt(c_00);
		tmp = 1.0/c_00;
		}
	else
		{
		c_00 = 0.0;
		tmp = 0.0;
		}
	c_10 *= tmp;
	c_20 *= tmp;
	c_30 *= tmp;
	inv_diag_D[0] = tmp;

	if(kn==1)
		goto store;
	
	c_11 -= c_10 * c_10;
	c_21 -= c_20 * c_10;
	c_31 -= c_30 * c_10;
	if(c_11>0)
		{
		c_11 = sqrt(c_11);
		tmp = 1.0/c_11;
		}
	else
		{
		c_11 = 0.0;
		tmp = 0.0;
		}
	c_21 *= tmp;
	c_31 *= tmp;
	inv_diag_D[1] = tmp;

	if(kn==2)
		goto store;
	
	c_22 -= c_20 * c_20;
	c_32 -= c_30 * c_20;
	c_22 -= c_21 * c_21;
	c_32 -= c_31 * c_21;
	if(c_22>0)
		{
		c_22 = sqrt(c_22);
		tmp = 1.0/c_22;
		}
	else
		{
		c_22 = 0.0;
		tmp = 0.0;
		}
	c_32 *= tmp;
	inv_diag_D[2] = tmp;

	if(kn==3)
		goto store;
	
	c_33 -= c_30 * c_30;
	c_33 -= c_31 * c_31;
	c_33 -= c_32 * c_32;
	if(c_33>0)
		{
		c_33 = sqrt(c_33);
		tmp = 1.0/c_33;
		}
	else
		{
		c_33 = 0.0;
		tmp = 0.0;
		}
	inv_diag_D[3] = tmp;


	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
//		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
//		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

//		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

//		if(kn==2)
//			return;

//		D[0+bs*2] = c_02;
//		D[1+bs*2] = c_12;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
//		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

//		if(kn==1)
//			return;

//		D[0+bs*1] = c_01;

//		if(kn==2)
//			return;

//		D[0+bs*2] = c_02;

//		if(kn==3)
//			return;

//		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dpotrf_nt_l_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, double *inv_diag_D)
	{
	kernel_dpotrf_nt_l_4x4_vs_lib4(kmax, A, B, C, D, inv_diag_D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, double *C, double *D, double *inv_diag_D, int km, int kn)
	{
	double alpha = 1.0;
	double beta = 1.0;
	kernel_dsyrk_nt_l_4x4_vs_lib4(kp, &alpha, Ap, Bp, &beta, C, D, km, kn);
	kernel_dpotrf_nt_l_4x4_vs_lib4(km_, Am, Bm, D, D, inv_diag_D, km, kn);
	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dsyrk_dpotrf_nt_l_4x4_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, double *C, double *D, double *inv_diag_D)
	{
	double alpha = 1.0;
	double beta = 1.0;
	kernel_dsyrk_nt_l_4x4_lib4(kp, &alpha, Ap, Bp, &beta, C, D);
	kernel_dpotrf_nt_l_4x4_lib4(km_, Am, Bm, D, D, inv_diag_D);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		tmp,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = C[0+bs*0] + c_00;
	c_10 = C[1+bs*0] + c_10;
	c_20 = C[2+bs*0] + c_20;
	c_30 = C[3+bs*0] + c_30;

	c_01 = C[0+bs*1] + c_01;
	c_11 = C[1+bs*1] + c_11;
	c_21 = C[2+bs*1] + c_21;
	c_31 = C[3+bs*1] + c_31;

	c_02 = C[0+bs*2] + c_02;
	c_12 = C[1+bs*2] + c_12;
	c_22 = C[2+bs*2] + c_22;
	c_32 = C[3+bs*2] + c_32;

	c_03 = C[0+bs*3] + c_03;
	c_13 = C[1+bs*3] + c_13;
	c_23 = C[2+bs*3] + c_23;
	c_33 = C[3+bs*3] + c_33;

	tmp = inv_diag_E[0];
	c_00 *= tmp;
	c_10 *= tmp;
	c_20 *= tmp;
	c_30 *= tmp;

	if(kn==1)
		goto store;
	
	tmp = E[1+bs*0];
	c_01 -= c_00 * tmp;
	c_11 -= c_10 * tmp;
	c_21 -= c_20 * tmp;
	c_31 -= c_30 * tmp;
	tmp = inv_diag_E[1];
	c_01 *= tmp;
	c_11 *= tmp;
	c_21 *= tmp;
	c_31 *= tmp;

	if(kn==2)
		goto store;
	
	tmp = E[2+bs*0];
	c_02 -= c_00 * tmp;
	c_12 -= c_10 * tmp;
	c_22 -= c_20 * tmp;
	c_32 -= c_30 * tmp;
	tmp = E[2+bs*1];
	c_02 -= c_01 * tmp;
	c_12 -= c_11 * tmp;
	c_22 -= c_21 * tmp;
	c_32 -= c_31 * tmp;
	tmp = inv_diag_E[2];
	c_02 *= tmp;
	c_12 *= tmp;
	c_22 *= tmp;
	c_32 *= tmp;

	if(kn==3)
		goto store;
	
	tmp = E[3+bs*0];
	c_03 -= c_00 * tmp;
	c_13 -= c_10 * tmp;
	c_23 -= c_20 * tmp;
	c_33 -= c_30 * tmp;
	tmp = E[3+bs*1];
	c_03 -= c_01 * tmp;
	c_13 -= c_11 * tmp;
	c_23 -= c_21 * tmp;
	c_33 -= c_31 * tmp;
	tmp = E[3+bs*2];
	c_03 -= c_02 * tmp;
	c_13 -= c_12 * tmp;
	c_23 -= c_22 * tmp;
	c_33 -= c_32 * tmp;
	tmp = inv_diag_E[3];
	c_03 *= tmp;
	c_13 *= tmp;
	c_23 *= tmp;
	c_33 *= tmp;


	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dtrsm_nt_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(k, A, B, C, D, E, inv_diag_E, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{
	double alpha = 1.0;
	double beta  = 1.0;
	kernel_dgemm_nt_4x4_vs_lib4(kp, &alpha, Ap, Bp, &beta, C, D, km, kn);
	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(km_, Am, Bm, D, D, E, inv_diag_E, km, kn);
	}
#endif



#if defined(TARGET_GENERIC)
void kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E)
	{
	double alpha = 1.0;
	double beta  = 1.0;
	kernel_dgemm_nt_4x4_lib4(kp, &alpha, Ap, Bp, &beta, C, D);
	kernel_dtrsm_nt_rl_inv_4x4_lib4(km_, Am, Bm, D, D, E, inv_diag_E);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nt_rl_one_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		tmp,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = C[0+bs*0] + c_00;
	c_10 = C[1+bs*0] + c_10;
	c_20 = C[2+bs*0] + c_20;
	c_30 = C[3+bs*0] + c_30;

	c_01 = C[0+bs*1] + c_01;
	c_11 = C[1+bs*1] + c_11;
	c_21 = C[2+bs*1] + c_21;
	c_31 = C[3+bs*1] + c_31;

	c_02 = C[0+bs*2] + c_02;
	c_12 = C[1+bs*2] + c_12;
	c_22 = C[2+bs*2] + c_22;
	c_32 = C[3+bs*2] + c_32;

	c_03 = C[0+bs*3] + c_03;
	c_13 = C[1+bs*3] + c_13;
	c_23 = C[2+bs*3] + c_23;
	c_33 = C[3+bs*3] + c_33;

	if(kn==1)
		goto store;
	
	tmp = E[1+bs*0];
	c_01 -= c_00 * tmp;
	c_11 -= c_10 * tmp;
	c_21 -= c_20 * tmp;
	c_31 -= c_30 * tmp;

	if(kn==2)
		goto store;
	
	tmp = E[2+bs*0];
	c_02 -= c_00 * tmp;
	c_12 -= c_10 * tmp;
	c_22 -= c_20 * tmp;
	c_32 -= c_30 * tmp;
	tmp = E[2+bs*1];
	c_02 -= c_01 * tmp;
	c_12 -= c_11 * tmp;
	c_22 -= c_21 * tmp;
	c_32 -= c_31 * tmp;

	if(kn==3)
		goto store;
	
	tmp = E[3+bs*0];
	c_03 -= c_00 * tmp;
	c_13 -= c_10 * tmp;
	c_23 -= c_20 * tmp;
	c_33 -= c_30 * tmp;
	tmp = E[3+bs*1];
	c_03 -= c_01 * tmp;
	c_13 -= c_11 * tmp;
	c_23 -= c_21 * tmp;
	c_33 -= c_31 * tmp;
	tmp = E[3+bs*2];
	c_03 -= c_02 * tmp;
	c_13 -= c_12 * tmp;
	c_23 -= c_22 * tmp;
	c_33 -= c_32 * tmp;


	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nt_rl_one_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E)
	{
	kernel_dtrsm_nt_rl_one_4x4_vs_lib4(k, A, B, C, D, E, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{

	const int bs = 4;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		tmp,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 1

		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];

		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		b_3 = B[7];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 2

		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];

		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		b_3 = B[11];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		// k = 3

		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];

		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		b_3 = B[15];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		// k = 0

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		b_3 = B[3];

		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;

		A += 4;
		B += 4;

		}
	
	c_00 = C[0+bs*0] + c_00;
	c_10 = C[1+bs*0] + c_10;
	c_20 = C[2+bs*0] + c_20;
	c_30 = C[3+bs*0] + c_30;

	c_01 = C[0+bs*1] + c_01;
	c_11 = C[1+bs*1] + c_11;
	c_21 = C[2+bs*1] + c_21;
	c_31 = C[3+bs*1] + c_31;

	c_02 = C[0+bs*2] + c_02;
	c_12 = C[1+bs*2] + c_12;
	c_22 = C[2+bs*2] + c_22;
	c_32 = C[3+bs*2] + c_32;

	c_03 = C[0+bs*3] + c_03;
	c_13 = C[1+bs*3] + c_13;
	c_23 = C[2+bs*3] + c_23;
	c_33 = C[3+bs*3] + c_33;


	if(kn>3)
		{
		tmp = inv_diag_E[3];
		c_03 *= tmp;
		c_13 *= tmp;
		c_23 *= tmp;
		c_33 *= tmp;
		tmp = E[2+bs*3];
		c_02 -= c_03 * tmp;
		c_12 -= c_13 * tmp;
		c_22 -= c_23 * tmp;
		c_32 -= c_33 * tmp;
		tmp = E[1+bs*3];
		c_01 -= c_03 * tmp;
		c_11 -= c_13 * tmp;
		c_21 -= c_23 * tmp;
		c_31 -= c_33 * tmp;
		tmp = E[0+bs*3];
		c_00 -= c_03 * tmp;
		c_10 -= c_13 * tmp;
		c_20 -= c_23 * tmp;
		c_30 -= c_33 * tmp;
		}

	if(kn>2)
		{
		tmp = inv_diag_E[2];
		c_02 *= tmp;
		c_12 *= tmp;
		c_22 *= tmp;
		c_32 *= tmp;
		tmp = E[1+bs*2];
		c_01 -= c_02 * tmp;
		c_11 -= c_12 * tmp;
		c_21 -= c_22 * tmp;
		c_31 -= c_32 * tmp;
		tmp = E[0+bs*2];
		c_00 -= c_02 * tmp;
		c_10 -= c_12 * tmp;
		c_20 -= c_22 * tmp;
		c_30 -= c_32 * tmp;
		}

	if(kn>1)
		{
		tmp = inv_diag_E[1];
		c_01 *= tmp;
		c_11 *= tmp;
		c_21 *= tmp;
		c_31 *= tmp;
		tmp = E[0+bs*1];
		c_00 -= c_01 * tmp;
		c_10 -= c_11 * tmp;
		c_20 -= c_21 * tmp;
		c_30 -= c_31 * tmp;
		}

	tmp = inv_diag_E[0];
	c_00 *= tmp;
	c_10 *= tmp;
	c_20 *= tmp;
	c_30 *= tmp;


	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nt_ru_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dtrsm_nt_ru_inv_4x4_vs_lib4(k, A, B, C, D, E, inv_diag_E, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgetrf_nn_4x4_vs_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *inv_diag_D, int km, int kn)
	{

	const int bs = 4;

	int k;

	double
		tmp,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	if(kmax<=0)
		goto add;

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		A += 4;
		B += 1;

		}
		
	add:

	c_00 += C[0+bs*0];
	c_10 += C[1+bs*0];
	c_20 += C[2+bs*0];
	c_30 += C[3+bs*0];

	c_01 += C[0+bs*1];
	c_11 += C[1+bs*1];
	c_21 += C[2+bs*1];
	c_31 += C[3+bs*1];

	c_02 += C[0+bs*2];
	c_12 += C[1+bs*2];
	c_22 += C[2+bs*2];
	c_32 += C[3+bs*2];

	c_03 += C[0+bs*3];
	c_13 += C[1+bs*3];
	c_23 += C[2+bs*3];
	c_33 += C[3+bs*3];

	// factorization

	// first column
	tmp = 1.0 / c_00;
	c_10 *= tmp;
	c_20 *= tmp;
	c_30 *= tmp;

	inv_diag_D[0] = tmp;

	if(kn==1)
		goto store;

	// second column
	c_11 -= c_10 * c_01;
	c_21 -= c_20 * c_01;
	c_31 -= c_30 * c_01;

	tmp = 1.0 / c_11;
	c_21 *= tmp;
	c_31 *= tmp;
	
	inv_diag_D[1] = tmp;

	if(kn==2)
		goto store;

	// third column
	c_12 -= c_10 * c_02;
	c_22 -= c_20 * c_02;
	c_32 -= c_30 * c_02;

	c_22 -= c_21 * c_12;
	c_32 -= c_31 * c_12;

	tmp = 1.0 / c_22;
	c_32 *= tmp;

	inv_diag_D[2] = tmp;

	if(kn==3)
		goto store;

	// fourth column
	c_13 -= c_10 * c_03;
	c_23 -= c_20 * c_03;
	c_33 -= c_30 * c_03;

	c_23 -= c_21 * c_13;
	c_33 -= c_31 * c_13;

	c_33 -= c_32 * c_23;

	tmp = 1.0 / c_33;

	inv_diag_D[3] = tmp;

	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dgetrf_nn_4x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *inv_diag_D)
	{
	kernel_dgetrf_nn_4x4_vs_lib4(kmax, A, B, sdb, C, D, inv_diag_D, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_ll_one_4x4_vs_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E, int km, int kn)
	{

	const int bs = 4;

	int k;

	double
		tmp,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		e_1, e_2, e_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	if(kmax<=0)
		goto add;

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		A += 4;
		B += 1;

		}
		
	add:

	c_00 += C[0+bs*0];
	c_10 += C[1+bs*0];
	c_20 += C[2+bs*0];
	c_30 += C[3+bs*0];

	c_01 += C[0+bs*1];
	c_11 += C[1+bs*1];
	c_21 += C[2+bs*1];
	c_31 += C[3+bs*1];

	c_02 += C[0+bs*2];
	c_12 += C[1+bs*2];
	c_22 += C[2+bs*2];
	c_32 += C[3+bs*2];

	c_03 += C[0+bs*3];
	c_13 += C[1+bs*3];
	c_23 += C[2+bs*3];
	c_33 += C[3+bs*3];

	// solution

	if(km==1)
		goto store;
	
	e_1 = E[1+bs*0];
	e_2 = E[2+bs*0];
	e_3 = E[3+bs*0];
	c_10 -= e_1 * c_00;
	c_20 -= e_2 * c_00;
	c_30 -= e_3 * c_00;
	c_11 -= e_1 * c_01;
	c_21 -= e_2 * c_01;
	c_31 -= e_3 * c_01;
	c_12 -= e_1 * c_02;
	c_22 -= e_2 * c_02;
	c_32 -= e_3 * c_02;
	c_13 -= e_1 * c_03;
	c_23 -= e_2 * c_03;
	c_33 -= e_3 * c_03;

	if(km==2)
		goto store;
	
	e_2 = E[2+bs*1];
	e_3 = E[3+bs*1];
	c_20 -= e_2 * c_10;
	c_30 -= e_3 * c_10;
	c_21 -= e_2 * c_11;
	c_31 -= e_3 * c_11;
	c_22 -= e_2 * c_12;
	c_32 -= e_3 * c_12;
	c_23 -= e_2 * c_13;
	c_33 -= e_3 * c_13;

	if(km==3)
		goto store;
	
	e_3 = E[3+bs*2];
	c_30 -= e_3 * c_20;
	c_31 -= e_3 * c_21;
	c_32 -= e_3 * c_22;
	c_33 -= e_3 * c_23;

	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_ll_one_4x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E)
	{
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(kmax, A, B, sdb, C, D, E, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{

	const int bs = 4;

	int k;

	double
		tmp,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		e_00, e_01, e_02, e_03,
		      e_11, e_12, e_13,
			        e_22, e_23,
					      e_33,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	if(kmax<=0)
		goto add;

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		A += 4;
		B += 1;

		}
		
	add:

	c_00 += C[0+bs*0];
	c_10 += C[1+bs*0];
	c_20 += C[2+bs*0];
	c_30 += C[3+bs*0];

	c_01 += C[0+bs*1];
	c_11 += C[1+bs*1];
	c_21 += C[2+bs*1];
	c_31 += C[3+bs*1];

	c_02 += C[0+bs*2];
	c_12 += C[1+bs*2];
	c_22 += C[2+bs*2];
	c_32 += C[3+bs*2];

	c_03 += C[0+bs*3];
	c_13 += C[1+bs*3];
	c_23 += C[2+bs*3];
	c_33 += C[3+bs*3];
	
	// solve

	e_00 = inv_diag_E[0];
	c_00 *= e_00;
	c_10 *= e_00;
	c_20 *= e_00;
	c_30 *= e_00;

	if(kn==1)
		goto store;
	
	e_01 = E[0+bs*1];
	e_11 = inv_diag_E[1];
	c_01 -= c_00 * e_01;
	c_11 -= c_10 * e_01;
	c_21 -= c_20 * e_01;
	c_31 -= c_30 * e_01;
	c_01 *= e_11;
	c_11 *= e_11;
	c_21 *= e_11;
	c_31 *= e_11;

	if(kn==2)
		goto store;
	
	e_02 = E[0+bs*2];
	e_12 = E[1+bs*2];
	e_22 = inv_diag_E[2];
	c_02 -= c_00 * e_02;
	c_12 -= c_10 * e_02;
	c_22 -= c_20 * e_02;
	c_32 -= c_30 * e_02;
	c_02 -= c_01 * e_12;
	c_12 -= c_11 * e_12;
	c_22 -= c_21 * e_12;
	c_32 -= c_31 * e_12;
	c_02 *= e_22;
	c_12 *= e_22;
	c_22 *= e_22;
	c_32 *= e_22;

	if(kn==3)
		goto store;
	
	e_03 = E[0+bs*3];
	e_13 = E[1+bs*3];
	e_23 = E[2+bs*3];
	e_33 = inv_diag_E[3];
	c_03 -= c_00 * e_03;
	c_13 -= c_10 * e_03;
	c_23 -= c_20 * e_03;
	c_33 -= c_30 * e_03;
	c_03 -= c_01 * e_13;
	c_13 -= c_11 * e_13;
	c_23 -= c_21 * e_13;
	c_33 -= c_31 * e_13;
	c_03 -= c_02 * e_23;
	c_13 -= c_12 * e_23;
	c_23 -= c_22 * e_23;
	c_33 -= c_32 * e_23;
	c_03 *= e_33;
	c_13 *= e_33;
	c_23 *= e_33;
	c_33 *= e_33;

	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_ru_inv_4x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(kmax, A, B, sdb, C, D, E, inv_diag_E, 4, 4);
	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_lu_inv_4x4_vs_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{

	const int bs = 4;

	int k;

	double
		tmp,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		e_00, e_01, e_02, e_03,
		      e_11, e_12, e_13,
			        e_22, e_23,
					      e_33,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	if(kmax<=0)
		goto add;

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		A += 4;
		B += 1;

		}
		
	add:

	c_00 += C[0+bs*0];
	c_10 += C[1+bs*0];
	c_20 += C[2+bs*0];
	c_30 += C[3+bs*0];

	c_01 += C[0+bs*1];
	c_11 += C[1+bs*1];
	c_21 += C[2+bs*1];
	c_31 += C[3+bs*1];

	c_02 += C[0+bs*2];
	c_12 += C[1+bs*2];
	c_22 += C[2+bs*2];
	c_32 += C[3+bs*2];

	c_03 += C[0+bs*3];
	c_13 += C[1+bs*3];
	c_23 += C[2+bs*3];
	c_33 += C[3+bs*3];

//	printf("\n%f %f %f %f\n", c_00, c_01, c_02, c_03);
//	printf("\n%f %f %f %f\n", c_10, c_11, c_12, c_13);
//	printf("\n%f %f %f %f\n", c_20, c_21, c_22, c_23);
//	printf("\n%f %f %f %f\n", c_30, c_31, c_32, c_33);
	
	// solve

	if(km>3)
		{
		e_03 = E[0+bs*3];
		e_13 = E[1+bs*3];
		e_23 = E[2+bs*3];
		e_33 = inv_diag_E[3];
		c_30 *= e_33;
		c_31 *= e_33;
		c_32 *= e_33;
		c_33 *= e_33;
		c_00 -= e_03 * c_30;
		c_01 -= e_03 * c_31;
		c_02 -= e_03 * c_32;
		c_03 -= e_03 * c_33;
		c_10 -= e_13 * c_30;
		c_11 -= e_13 * c_31;
		c_12 -= e_13 * c_32;
		c_13 -= e_13 * c_33;
		c_20 -= e_23 * c_30;
		c_21 -= e_23 * c_31;
		c_22 -= e_23 * c_32;
		c_23 -= e_23 * c_33;
		}
	
	if(km>2)
		{
		e_02 = E[0+bs*2];
		e_12 = E[1+bs*2];
		e_22 = inv_diag_E[2];
		c_20 *= e_22;
		c_21 *= e_22;
		c_22 *= e_22;
		c_23 *= e_22;
		c_00 -= e_02 * c_20;
		c_01 -= e_02 * c_21;
		c_02 -= e_02 * c_22;
		c_03 -= e_02 * c_23;
		c_10 -= e_12 * c_20;
		c_11 -= e_12 * c_21;
		c_12 -= e_12 * c_22;
		c_13 -= e_12 * c_23;
		}
	
	if(km>1)
		{
		e_01 = E[0+bs*1];
		e_11 = inv_diag_E[1];
		c_10 *= e_11;
		c_11 *= e_11;
		c_12 *= e_11;
		c_13 *= e_11;
		c_00 -= e_01 * c_10;
		c_01 -= e_01 * c_11;
		c_02 -= e_01 * c_12;
		c_03 -= e_01 * c_13;
		}
	
	e_00 = inv_diag_E[0];
	c_00 *= e_00;
	c_01 *= e_00;
	c_02 *= e_00;
	c_03 *= e_00;

	store:

	if(km>=4)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(km>=3)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;
		D[2+bs*2] = c_22;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		D[2+bs*3] = c_23;
		}
	else if(km>=2)
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;
		D[1+bs*1] = c_11;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;
		D[1+bs*2] = c_12;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		D[1+bs*3] = c_13;
		}
	else //if(km>=1)
		{
		D[0+bs*0] = c_00;

		if(kn==1)
			return;

		D[0+bs*1] = c_01;

		if(kn==2)
			return;

		D[0+bs*2] = c_02;

		if(kn==3)
			return;

		D[0+bs*3] = c_03;
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15)
void kernel_dtrsm_nn_lu_inv_4x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dtrsm_nn_lu_inv_4x4_vs_lib4(kmax, A, B, sdb, C, D, E, inv_diag_E, 4, 4);
	}
#endif

