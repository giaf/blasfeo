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

#include <math.h>



void kernel_dgemm_ntnn_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
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

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[1+bs*0] - c_10;
			c_20 = C[2+bs*0] - c_20;
			c_30 = C[3+bs*0] - c_30;

			c_01 = C[0+bs*1] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[2+bs*1] - c_21;
			c_31 = C[3+bs*1] - c_31;

			c_02 = C[0+bs*2] - c_02;
			c_12 = C[1+bs*2] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[3+bs*2] - c_32;

			c_03 = C[0+bs*3] - c_03;
			c_13 = C[1+bs*3] - c_13;
			c_23 = C[2+bs*3] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

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



void kernel_dgemm_ntnn_4x4_lib4(int kmax, double *A, double *B, int alg, double *C, double *D)
	{
	kernel_dgemm_ntnn_4x4_vs_lib4(kmax, A, B, alg, C, D, 4, 4);
	}


void kernel_dgemm_ntnt_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
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

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[1+bs*0] - c_10;
			c_20 = C[2+bs*0] - c_20;
			c_30 = C[3+bs*0] - c_30;

			c_01 = C[0+bs*1] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[2+bs*1] - c_21;
			c_31 = C[3+bs*1] - c_31;

			c_02 = C[0+bs*2] - c_02;
			c_12 = C[1+bs*2] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[3+bs*2] - c_32;

			c_03 = C[0+bs*3] - c_03;
			c_13 = C[1+bs*3] - c_13;
			c_23 = C[2+bs*3] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

	store:
	if(kn>=4)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;
		D[0+bs*2] = c_20;
		D[0+bs*3] = c_30;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;
		D[1+bs*2] = c_21;
		D[1+bs*3] = c_31;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;
		D[2+bs*2] = c_22;
		D[2+bs*3] = c_32;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		D[3+bs*2] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(kn>=3)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;
		D[0+bs*2] = c_20;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;
		D[1+bs*2] = c_21;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;
		D[2+bs*2] = c_22;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		D[3+bs*2] = c_23;
		}
	else if(kn>=2)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		}
	else //if(kn>=1)
		{
		D[0+bs*0] = c_00;

		if(km==1)
			return;

		D[1+bs*0] = c_01;

		if(km==2)
			return;

		D[2+bs*0] = c_02;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		}

	}



void kernel_dgemm_ntnt_4x4_lib4(int kmax, double *A, double *B, int alg, double *C, double *D)
	{
	kernel_dgemm_ntnt_4x4_vs_lib4(kmax, A, B, alg, C, D, 4, 4);
	}



void kernel_dgemm_nttn_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
			c_00 = C[0+bs*0] + c_00;
			c_10 = C[0+bs*1] + c_10;
			c_20 = C[0+bs*2] + c_20;
			c_30 = C[0+bs*3] + c_30;

			c_01 = C[1+bs*0] + c_01;
			c_11 = C[1+bs*1] + c_11;
			c_21 = C[1+bs*2] + c_21;
			c_31 = C[1+bs*3] + c_31;

			c_02 = C[2+bs*0] + c_02;
			c_12 = C[2+bs*1] + c_12;
			c_22 = C[2+bs*2] + c_22;
			c_32 = C[2+bs*3] + c_32;

			c_03 = C[3+bs*0] + c_03;
			c_13 = C[3+bs*1] + c_13;
			c_23 = C[3+bs*2] + c_23;
			c_33 = C[3+bs*3] + c_33;

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[0+bs*1] - c_10;
			c_20 = C[0+bs*2] - c_20;
			c_30 = C[0+bs*3] - c_30;

			c_01 = C[1+bs*0] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[1+bs*2] - c_21;
			c_31 = C[1+bs*3] - c_31;

			c_02 = C[2+bs*0] - c_02;
			c_12 = C[2+bs*1] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[2+bs*3] - c_32;

			c_03 = C[3+bs*0] - c_03;
			c_13 = C[3+bs*1] - c_13;
			c_23 = C[3+bs*2] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

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



void kernel_dgemm_nttn_4x4_lib4(int kmax, double *A, double *B, int alg, double *C, double *D)
	{
	kernel_dgemm_nttn_4x4_vs_lib4(kmax, A, B, alg, C, D, 4, 4);
	}



void kernel_dgemm_nttt_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
			c_00 = C[0+bs*0] + c_00;
			c_10 = C[0+bs*1] + c_10;
			c_20 = C[0+bs*2] + c_20;
			c_30 = C[0+bs*3] + c_30;

			c_01 = C[1+bs*0] + c_01;
			c_11 = C[1+bs*1] + c_11;
			c_21 = C[1+bs*2] + c_21;
			c_31 = C[1+bs*3] + c_31;

			c_02 = C[2+bs*0] + c_02;
			c_12 = C[2+bs*1] + c_12;
			c_22 = C[2+bs*2] + c_22;
			c_32 = C[2+bs*3] + c_32;

			c_03 = C[3+bs*0] + c_03;
			c_13 = C[3+bs*1] + c_13;
			c_23 = C[3+bs*2] + c_23;
			c_33 = C[3+bs*3] + c_33;

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[0+bs*1] - c_10;
			c_20 = C[0+bs*2] - c_20;
			c_30 = C[0+bs*3] - c_30;

			c_01 = C[1+bs*0] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[1+bs*2] - c_21;
			c_31 = C[1+bs*3] - c_31;

			c_02 = C[2+bs*0] - c_02;
			c_12 = C[2+bs*1] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[2+bs*3] - c_32;

			c_03 = C[3+bs*0] - c_03;
			c_13 = C[3+bs*1] - c_13;
			c_23 = C[3+bs*2] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

	store:

	if(kn>=4)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;
		D[0+bs*2] = c_20;
		D[0+bs*3] = c_30;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;
		D[1+bs*2] = c_21;
		D[1+bs*3] = c_31;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;
		D[2+bs*2] = c_22;
		D[2+bs*3] = c_32;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		D[3+bs*2] = c_23;
		D[3+bs*3] = c_33;
		}
	else if(kn>=3)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;
		D[0+bs*2] = c_20;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;
		D[1+bs*2] = c_21;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;
		D[2+bs*2] = c_22;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		D[3+bs*2] = c_23;
		}
	else if(kn>=2)
		{
		D[0+bs*0] = c_00;
		D[0+bs*1] = c_10;

		if(km==1)
			return;

		D[1+bs*0] = c_01;
		D[1+bs*1] = c_11;

		if(km==2)
			return;

		D[2+bs*0] = c_02;
		D[2+bs*1] = c_12;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		D[3+bs*1] = c_13;
		}
	else //if(kn>=1)
		{
		D[0+bs*0] = c_00;

		if(km==1)
			return;

		D[1+bs*0] = c_01;

		if(km==2)
			return;

		D[2+bs*0] = c_02;

		if(km==3)
			return;

		D[3+bs*0] = c_03;
		}

	}



void kernel_dgemm_nttt_4x4_lib4(int kmax, double *A, double *B, int alg, double *C, double *D)
	{
	kernel_dgemm_nttt_4x4_vs_lib4(kmax, A, B, alg, C, D, 4, 4);
	}



void kernel_dsyrk_ntnn_l_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
			c_00 = C[0+bs*0] + c_00;
			c_10 = C[1+bs*0] + c_10;
			c_20 = C[2+bs*0] + c_20;
			c_30 = C[3+bs*0] + c_30;

//			c_01 = C[0+bs*1] + c_01;
			c_11 = C[1+bs*1] + c_11;
			c_21 = C[2+bs*1] + c_21;
			c_31 = C[3+bs*1] + c_31;

//			c_02 = C[0+bs*2] + c_02;
//			c_12 = C[1+bs*2] + c_12;
			c_22 = C[2+bs*2] + c_22;
			c_32 = C[3+bs*2] + c_32;

//			c_03 = C[0+bs*3] + c_03;
//			c_13 = C[1+bs*3] + c_13;
//			c_23 = C[2+bs*3] + c_23;
			c_33 = C[3+bs*3] + c_33;

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[1+bs*0] - c_10;
			c_20 = C[2+bs*0] - c_20;
			c_30 = C[3+bs*0] - c_30;

//			c_01 = C[0+bs*1] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[2+bs*1] - c_21;
			c_31 = C[3+bs*1] - c_31;

//			c_02 = C[0+bs*2] - c_02;
//			c_12 = C[1+bs*2] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[3+bs*2] - c_32;

//			c_03 = C[0+bs*3] - c_03;
//			c_13 = C[1+bs*3] - c_13;
//			c_23 = C[2+bs*3] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

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



void kernel_dtrmm_ntnn_ru_4x4_vs_lib4(int kmax, double *A, double *B, int alg, double *C, double *D, int km, int kn)
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
	
	if(alg==0)
		{
		goto store;
		}
	else
		{
		if(alg==1)
			{
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

			goto store;
			}
		else
			{
			c_00 = C[0+bs*0] - c_00;
			c_10 = C[1+bs*0] - c_10;
			c_20 = C[2+bs*0] - c_20;
			c_30 = C[3+bs*0] - c_30;

			c_01 = C[0+bs*1] - c_01;
			c_11 = C[1+bs*1] - c_11;
			c_21 = C[2+bs*1] - c_21;
			c_31 = C[3+bs*1] - c_31;

			c_02 = C[0+bs*2] - c_02;
			c_12 = C[1+bs*2] - c_12;
			c_22 = C[2+bs*2] - c_22;
			c_32 = C[3+bs*2] - c_32;

			c_03 = C[0+bs*3] - c_03;
			c_13 = C[1+bs*3] - c_13;
			c_23 = C[2+bs*3] - c_23;
			c_33 = C[3+bs*3] - c_33;

			goto store;
			}
		}

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






void kernel_dtrmm_ntnn_ru_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D)
	{
	kernel_dtrmm_ntnn_ru_4x4_vs_lib4(k, A, B, alg, C, D, 4, 4);
	}



void kernel_dpotrf_ntnn_l_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn)
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

	for(k=0; k<kmax; k++)
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



void kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *inv_diag_D, int km, int kn)
	{
	kernel_dsyrk_ntnn_l_4x4_vs_lib4(kp, Ap, Bp, alg, C, D, km, kn);
	kernel_dpotrf_ntnn_l_4x4_vs_lib4(km_, Am, Bm, D, D, inv_diag_D, km, kn);
	}



void kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
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

	for(k=0; k<kmax; k++)
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



void kernel_dtrsm_ntnn_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4(k, A, B, C, D, E, inv_diag_E, 4, 4);
	}



void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E, int km, int kn)
	{
	kernel_dgemm_ntnn_4x4_vs_lib4(kp, Ap, Bp, alg, C, D, km, kn);
	kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4(km_, Am, Bm, D, D, E, inv_diag_E, km, kn);
	}



void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km_, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E)
	{
	kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4(kp, Ap, Bp, km_, Am, Bm, alg, C, D, E, inv_diag_E, 4, 4);
	}



