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



#include "../../include/blasfeo_d_kernel.h"



#if ! ( defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53) )
void kernel_dpack_nn_4_lib4(int kmax, double *A, int lda, double *C)
	{

	const int ps = 4;

	int ii;
	ii = 0;

	for(; ii<kmax-3; ii+=4)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[1+lda*0];
		C[2+ps*0] = A[2+lda*0];
		C[3+ps*0] = A[3+lda*0];

		C[0+ps*1] = A[0+lda*1];
		C[1+ps*1] = A[1+lda*1];
		C[2+ps*1] = A[2+lda*1];
		C[3+ps*1] = A[3+lda*1];

		C[0+ps*2] = A[0+lda*2];
		C[1+ps*2] = A[1+lda*2];
		C[2+ps*2] = A[2+lda*2];
		C[3+ps*2] = A[3+lda*2];

		C[0+ps*3] = A[0+lda*3];
		C[1+ps*3] = A[1+lda*3];
		C[2+ps*3] = A[2+lda*3];
		C[3+ps*3] = A[3+lda*3];

		A += 4*lda;
		C += 4*ps;
		}
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[1+lda*0];
		C[2+ps*0] = A[2+lda*0];
		C[3+ps*0] = A[3+lda*0];

		A += 1*lda;
		C += 1*ps;
		}

	return;

	}
#endif



void kernel_dpack_nn_4_vs_lib4(int kmax, double *A, int lda, double *C, int m1)
	{

	if(m1<=0)
		return;

	const int ps = 4;

	int ii;
	ii = 0;

	if(m1>=4)
		{
		kernel_dpack_nn_4_lib4(kmax, A, lda, C);
		return;
		}
	else if(m1==1)
		{
		goto l1;
		}
	else if(m1==2)
		{
		goto l2;
		}
	else //if(m1==3)
		{
		goto l3;
		}
	return;

l1:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];

		A += 1*lda;
		C += 1*ps;
		}
	return;

l2:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[1+lda*0];

		A += 1*lda;
		C += 1*ps;
		}
	return;

l3:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[1+lda*0];
		C[2+ps*0] = A[2+lda*0];

		A += 1*lda;
		C += 1*ps;
		}
	return;

	}



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dpack_nn_8_vs_lib4(int kmax, double *A, int lda, double *C, int sdc, int m1)
	{

	kernel_dpack_nn_4_lib4(kmax, A+0, lda, C+0*sdc);
	kernel_dpack_nn_4_vs_lib4(kmax, A+4, lda, C+4*sdc, m1-4);

	return;

	}
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dpack_nn_12_vs_lib4(int kmax, double *A, int lda, double *C, int sdc, int m1)
	{

	const int ps = 4;

	kernel_dpack_nn_8_lib4(kmax, A+0, lda, C+0*sdc, sdc);
	kernel_dpack_nn_4_vs_lib4(kmax, A+8, lda, C+8*sdc, m1-8);

	return;

	}
#endif



void kernel_dpack_tn_4_lib4(int kmax, double *A, int lda, double *C)
{

	const int ps = 4;

	int ii;
	ii = 0;

	for(; ii<kmax-3; ii+=4)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[0+lda*1];
		C[2+ps*0] = A[0+lda*2];
		C[3+ps*0] = A[0+lda*3];

		C[0+ps*1] = A[1+lda*0];
		C[1+ps*1] = A[1+lda*1];
		C[2+ps*1] = A[1+lda*2];
		C[3+ps*1] = A[1+lda*3];

		C[0+ps*2] = A[2+lda*0];
		C[1+ps*2] = A[2+lda*1];
		C[2+ps*2] = A[2+lda*2];
		C[3+ps*2] = A[2+lda*3];

		C[0+ps*3] = A[3+lda*0];
		C[1+ps*3] = A[3+lda*1];
		C[2+ps*3] = A[3+lda*2];
		C[3+ps*3] = A[3+lda*3];

		A += 4;
		C += 4*ps;
		}
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[0+lda*1];
		C[2+ps*0] = A[0+lda*2];
		C[3+ps*0] = A[0+lda*3];

		A += 1;
		C += 1*ps;
		}

	return;

	}



void kernel_dpack_tn_4_vs_lib4(int kmax, double *A, int lda, double *C, int m1)
{

	if(m1<=0)
		return;

	const int ps = 4;

	int ii;
	ii = 0;

	if(m1>=4)
		{
		kernel_dpack_tn_4_lib4(kmax, A, lda, C);
		return;
		}
	else if(m1==1)
		{
		goto l1;
		}
	else if(m1==2)
		{
		goto l2;
		}
	else //if(m1==3)
		{
		goto l3;
		}
	return;

l1:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];

		A += 1;
		C += 1*ps;
		}
	return;

l2:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[0+lda*1];

		A += 1;
		C += 1*ps;
		}
	return;

l3:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		C[0+ps*0] = A[0+lda*0];
		C[1+ps*0] = A[0+lda*1];
		C[2+ps*0] = A[0+lda*2];

		A += 1;
		C += 1*ps;
		}
	return;

	}



void kernel_dunpack_nt_4_lib4(int kmax, double *C, double *A, int lda)
{

	const int ps = 4;

	int ii;
	ii = 0;

	for(; ii<kmax-3; ii+=4)
		{
		A[0+lda*0] = C[0+ps*0];
		A[0+lda*1] = C[1+ps*0];
		A[0+lda*2] = C[2+ps*0];
		A[0+lda*3] = C[3+ps*0];

		A[1+lda*0] = C[0+ps*1];
		A[1+lda*1] = C[1+ps*1];
		A[1+lda*2] = C[2+ps*1];
		A[1+lda*3] = C[3+ps*1];

		A[2+lda*0] = C[0+ps*2];
		A[2+lda*1] = C[1+ps*2];
		A[2+lda*2] = C[2+ps*2];
		A[2+lda*3] = C[3+ps*2];

		A[3+lda*0] = C[0+ps*3];
		A[3+lda*1] = C[1+ps*3];
		A[3+lda*2] = C[2+ps*3];
		A[3+lda*3] = C[3+ps*3];

		A += 4;
		C += 4*ps;
		}
	for(; ii<kmax; ii++)
		{
		A[0+lda*0] = C[0+ps*0];
		A[0+lda*1] = C[1+ps*0];
		A[0+lda*2] = C[2+ps*0];
		A[0+lda*3] = C[3+ps*0];

		A += 1;
		C += 1*ps;
		}

	return;

	}



void kernel_dunpack_nt_4_vs_lib4(int kmax, double *C, double *A, int lda, int m1)
{

	if(m1<=0)
		return;

	const int ps = 4;

	int ii;
	ii = 0;

	if(m1>=4)
		{
		kernel_dunpack_nt_4_lib4(kmax, C, A, lda);
		return;
		}
	else if(m1==1)
		{
		goto l1;
		}
	else if(m1==2)
		{
		goto l2;
		}
	else //if(m1==3)
		{
		goto l3;
		}
	return;

l1:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		A[0+lda*0] = C[0+ps*0];

		A += 1;
		C += 1*ps;
		}
	return;

l2:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		A[0+lda*0] = C[0+ps*0];
		A[0+lda*1] = C[1+ps*0];

		A += 1;
		C += 1*ps;
		}
	return;

l3:
	ii = 0;
	for(; ii<kmax; ii++)
		{
		A[0+lda*0] = C[0+ps*0];
		A[0+lda*1] = C[1+ps*0];
		A[0+lda*2] = C[2+ps*0];

		A += 1;
		C += 1*ps;
		}
	return;

	}



// copy transposed panel into normal panel
void kernel_dpacp_tn_4_lib4(int kmax, int offsetA, double *A, int sda, double *B)
	{

	const int ps = 4;

	int k;

	int kna = (ps-offsetA)%ps;
	kna = kmax<kna ? kmax : kna;

	k = 0;
	if(kna>0)
		{
		A += offsetA;
		for( ; k<kna; k++)
			{
			//
			B[0+ps*0] = A[0+ps*0];
			B[1+ps*0] = A[0+ps*1];
			B[2+ps*0] = A[0+ps*2];
			B[3+ps*0] = A[0+ps*3];

			A += 1;
			B += ps;
			}
		A += ps*(sda-1);
		}
	for(; k<kmax-3; k+=4)
		{
		//
		B[0+ps*0] = A[0+ps*0];
		B[0+ps*1] = A[1+ps*0];
		B[0+ps*2] = A[2+ps*0];
		B[0+ps*3] = A[3+ps*0];
		//
		B[1+ps*0] = A[0+ps*1];
		B[1+ps*1] = A[1+ps*1];
		B[1+ps*2] = A[2+ps*1];
		B[1+ps*3] = A[3+ps*1];
		//
		B[2+ps*0] = A[0+ps*2];
		B[2+ps*1] = A[1+ps*2];
		B[2+ps*2] = A[2+ps*2];
		B[2+ps*3] = A[3+ps*2];
		//
		B[3+ps*0] = A[0+ps*3];
		B[3+ps*1] = A[1+ps*3];
		B[3+ps*2] = A[2+ps*3];
		B[3+ps*3] = A[3+ps*3];

		A += ps*sda;
		B += ps*ps;
		}
	for( ; k<kmax; k++)
		{
		//
		B[0+ps*0] = A[0+ps*0];
		B[1+ps*0] = A[0+ps*1];
		B[2+ps*0] = A[0+ps*2];
		B[3+ps*0] = A[0+ps*3];

		A += 1;
		B += ps;
		}
	return;
	}



// copy transposed panel into normal panel
void kernel_dpacp_nt_4_lib4(int kmax, double *A, int offsetB, double *B, int sdb)
	{

	const int ps = 4;

	int k;

	int kna = (ps-offsetB)%ps;
	kna = kmax<kna ? kmax : kna;

	k = 0;
	if(kna>0)
		{
		B += offsetB;
		for( ; k<kna; k++)
			{
			//
			B[0+ps*0] = A[0+ps*0];
			B[0+ps*1] = A[1+ps*0];
			B[0+ps*2] = A[2+ps*0];
			B[0+ps*3] = A[3+ps*0];

			B += 1;
			A += ps;
			}
		B += ps*(sdb-1);
		}
	for(; k<kmax-3; k+=4)
		{
		//
		B[0+ps*0] = A[0+ps*0];
		B[1+ps*0] = A[0+ps*1];
		B[2+ps*0] = A[0+ps*2];
		B[3+ps*0] = A[0+ps*3];
		//
		B[0+ps*1] = A[1+ps*0];
		B[1+ps*1] = A[1+ps*1];
		B[2+ps*1] = A[1+ps*2];
		B[3+ps*1] = A[1+ps*3];
		//
		B[0+ps*2] = A[2+ps*0];
		B[1+ps*2] = A[2+ps*1];
		B[2+ps*2] = A[2+ps*2];
		B[3+ps*2] = A[2+ps*3];
		//
		B[0+ps*3] = A[3+ps*0];
		B[1+ps*3] = A[3+ps*1];
		B[2+ps*3] = A[3+ps*2];
		B[3+ps*3] = A[3+ps*3];

		B += ps*sdb;
		A += ps*ps;
		}
	for( ; k<kmax; k++)
		{
		//
		B[0+ps*0] = A[0+ps*0];
		B[0+ps*1] = A[1+ps*0];
		B[0+ps*2] = A[2+ps*0];
		B[0+ps*3] = A[3+ps*0];

		B += 1;
		A += ps;
		}
	return;
	}



void kernel_dpacp_nn_4_lib4(int kmax, int offsetA, double *A, int sda, double *B)
	{

	const int ps = 4;

	int k;

	int air = offsetA%ps;

	double *A0 = A;
	double *A1 = A0 + ps*sda;

	if(offsetA==0)
		goto air_0;
	else if(offsetA==1)
		goto air_1;
	else if(offsetA==2)
		goto air_2;
	else //if(offsetA==3)
		goto air_3;
	
air_0:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];
		B[2+ps*0] = A0[2+ps*0];
		B[3+ps*0] = A0[3+ps*0];

		B[0+ps*1] = A0[0+ps*1];
		B[1+ps*1] = A0[1+ps*1];
		B[2+ps*1] = A0[2+ps*1];
		B[3+ps*1] = A0[3+ps*1];

		B[0+ps*2] = A0[0+ps*2];
		B[1+ps*2] = A0[1+ps*2];
		B[2+ps*2] = A0[2+ps*2];
		B[3+ps*2] = A0[3+ps*2];

		B[0+ps*3] = A0[0+ps*3];
		B[1+ps*3] = A0[1+ps*3];
		B[2+ps*3] = A0[2+ps*3];
		B[3+ps*3] = A0[3+ps*3];

		A0 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];
		B[2+ps*0] = A0[2+ps*0];
		B[3+ps*0] = A0[3+ps*0];

		A0 += 4;
		B += 4;
		}
	goto end;



air_1:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[1+ps*0];
		B[1+ps*0] = A0[2+ps*0];
		B[2+ps*0] = A0[3+ps*0];
		B[3+ps*0] = A1[0+ps*0];

		B[0+ps*1] = A0[1+ps*1];
		B[1+ps*1] = A0[2+ps*1];
		B[2+ps*1] = A0[3+ps*1];
		B[3+ps*1] = A1[0+ps*1];

		B[0+ps*2] = A0[1+ps*2];
		B[1+ps*2] = A0[2+ps*2];
		B[2+ps*2] = A0[3+ps*2];
		B[3+ps*2] = A1[0+ps*2];

		B[0+ps*3] = A0[1+ps*3];
		B[1+ps*3] = A0[2+ps*3];
		B[2+ps*3] = A0[3+ps*3];
		B[3+ps*3] = A1[0+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[1+ps*0];
		B[1+ps*0] = A0[2+ps*0];
		B[2+ps*0] = A0[3+ps*0];
		B[3+ps*0] = A1[0+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;



air_2:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[2+ps*0];
		B[1+ps*0] = A0[3+ps*0];
		B[2+ps*0] = A1[0+ps*0];
		B[3+ps*0] = A1[1+ps*0];

		B[0+ps*1] = A0[2+ps*1];
		B[1+ps*1] = A0[3+ps*1];
		B[2+ps*1] = A1[0+ps*1];
		B[3+ps*1] = A1[1+ps*1];

		B[0+ps*2] = A0[2+ps*2];
		B[1+ps*2] = A0[3+ps*2];
		B[2+ps*2] = A1[0+ps*2];
		B[3+ps*2] = A1[1+ps*2];

		B[0+ps*3] = A0[2+ps*3];
		B[1+ps*3] = A0[3+ps*3];
		B[2+ps*3] = A1[0+ps*3];
		B[3+ps*3] = A1[1+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[2+ps*0];
		B[1+ps*0] = A0[3+ps*0];
		B[2+ps*0] = A1[0+ps*0];
		B[3+ps*0] = A1[1+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;



air_3:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];
		B[2+ps*0] = A1[1+ps*0];
		B[3+ps*0] = A1[2+ps*0];

		B[0+ps*1] = A0[3+ps*1];
		B[1+ps*1] = A1[0+ps*1];
		B[2+ps*1] = A1[1+ps*1];
		B[3+ps*1] = A1[2+ps*1];

		B[0+ps*2] = A0[3+ps*2];
		B[1+ps*2] = A1[0+ps*2];
		B[2+ps*2] = A1[1+ps*2];
		B[3+ps*2] = A1[2+ps*2];

		B[0+ps*3] = A0[3+ps*3];
		B[1+ps*3] = A1[0+ps*3];
		B[2+ps*3] = A1[1+ps*3];
		B[3+ps*3] = A1[2+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];
		B[2+ps*0] = A1[1+ps*0];
		B[3+ps*0] = A1[2+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;

end:
	return;

	}



void kernel_dpacp_nn_4_vs_lib4(int kmax, int offsetA, double *A, int sda, double *B, int m1)
	{

	if(m1<=0)
		{
		return;
		}

	if(m1>=4)
		{
		kernel_dpacp_nn_4_lib4(kmax, offsetA, A, sda, B);
		return;
		}


	const int ps = 4;

	int k;

	int air = offsetA%ps;

	double *A0 = A;
	double *A1 = A0 + ps*sda;

	if(offsetA==0)
		goto air_0;
	else if(offsetA==1)
		goto air_1;
	else if(offsetA==2)
		goto air_2;
	else //if(offsetA==3)
		goto air_3;
	
air_0:
	
	if(m1==1)
		{
		goto air_0_1;
		}
	else if(m1==2)
		{
		goto air_0_2;
		}
	else //if(m1==3)
		{
		goto air_0_3;
		}

air_1:
	
	if(m1==1)
		{
		A0 += air;
		goto air_0_1;
		}
	else if(m1==2)
		{
		A0 += air;
		goto air_0_2;
		}
	else //if(m1==3)
		{
		A0 += air;
		goto air_0_3;
		}

air_2:
	
	if(m1==1)
		{
		A0 += air;
		goto air_0_1;
		}
	else if(m1==2)
		{
		A0 += air;
		goto air_0_2;
		}
	else //if(m1==3)
		{
		goto air_2_3;
		}

air_3:
	
	if(m1==1)
		{
		A0 += air;
		goto air_0_1;
		}
	else if(m1==2)
		{
		goto air_3_2;
		}
	else //if(m1==3)
		{
		goto air_3_3;
		}



air_0_3:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];
		B[2+ps*0] = A0[2+ps*0];

		B[0+ps*1] = A0[0+ps*1];
		B[1+ps*1] = A0[1+ps*1];
		B[2+ps*1] = A0[2+ps*1];

		B[0+ps*2] = A0[0+ps*2];
		B[1+ps*2] = A0[1+ps*2];
		B[2+ps*2] = A0[2+ps*2];

		B[0+ps*3] = A0[0+ps*3];
		B[1+ps*3] = A0[1+ps*3];
		B[2+ps*3] = A0[2+ps*3];

		A0 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];
		B[2+ps*0] = A0[2+ps*0];

		A0 += 4;
		B += 4;
		}
	goto end;

air_0_2:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];

		B[0+ps*1] = A0[0+ps*1];
		B[1+ps*1] = A0[1+ps*1];

		B[0+ps*2] = A0[0+ps*2];
		B[1+ps*2] = A0[1+ps*2];

		B[0+ps*3] = A0[0+ps*3];
		B[1+ps*3] = A0[1+ps*3];

		A0 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[0+ps*0];
		B[1+ps*0] = A0[1+ps*0];

		A0 += 4;
		B += 4;
		}
	goto end;

air_0_1:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[0+ps*0];

		B[0+ps*1] = A0[0+ps*1];

		B[0+ps*2] = A0[0+ps*2];

		B[0+ps*3] = A0[0+ps*3];

		A0 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[0+ps*0];

		A0 += 4;
		B += 4;
		}
	goto end;

air_2_3:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[2+ps*0];
		B[1+ps*0] = A0[3+ps*0];
		B[2+ps*0] = A1[0+ps*0];

		B[0+ps*1] = A0[2+ps*1];
		B[1+ps*1] = A0[3+ps*1];
		B[2+ps*1] = A1[0+ps*1];

		B[0+ps*2] = A0[2+ps*2];
		B[1+ps*2] = A0[3+ps*2];
		B[2+ps*2] = A1[0+ps*2];

		B[0+ps*3] = A0[2+ps*3];
		B[1+ps*3] = A0[3+ps*3];
		B[2+ps*3] = A1[0+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[2+ps*0];
		B[1+ps*0] = A0[3+ps*0];
		B[2+ps*0] = A1[0+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;

air_3_3:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];
		B[2+ps*0] = A1[1+ps*0];

		B[0+ps*1] = A0[3+ps*1];
		B[1+ps*1] = A1[0+ps*1];
		B[2+ps*1] = A1[1+ps*1];

		B[0+ps*2] = A0[3+ps*2];
		B[1+ps*2] = A1[0+ps*2];
		B[2+ps*2] = A1[1+ps*2];

		B[0+ps*3] = A0[3+ps*3];
		B[1+ps*3] = A1[0+ps*3];
		B[2+ps*3] = A1[1+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];
		B[2+ps*0] = A1[1+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;

air_3_2:
	k = 0;
	for(k=0; k<kmax-3; k+=4)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];

		B[0+ps*1] = A0[3+ps*1];
		B[1+ps*1] = A1[0+ps*1];

		B[0+ps*2] = A0[3+ps*2];
		B[1+ps*2] = A1[0+ps*2];

		B[0+ps*3] = A0[3+ps*3];
		B[1+ps*3] = A1[0+ps*3];

		A0 += 16;
		A1 += 16;
		B += 16;
		}
	for(; k<kmax; k++)
		{
		B[0+ps*0] = A0[3+ps*0];
		B[1+ps*0] = A1[0+ps*0];

		A0 += 4;
		A1 += 4;
		B += 4;
		}
	goto end;

end:
	return;

	}


