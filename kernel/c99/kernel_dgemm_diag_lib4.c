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



// B is the diagonal of a matrix, case beta=0.0
void kernel_dgemm_diag_right_4_a0_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
	
	alpha0 = alpha[0];
		
	b_0 = alpha0 * B[0];
	b_1 = alpha0 * B[1];
	b_2 = alpha0 * B[2];
	b_3 = alpha0 * B[3];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_0;
		c_2 = a_2 * b_0;
		c_3 = a_3 * b_0;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		c_0 = a_0 * b_1;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_1;
		c_3 = a_3 * b_1;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		c_0 = a_0 * b_2;
		c_1 = a_1 * b_2;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_2;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;
		

		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		c_0 = a_0 * b_3;
		c_1 = a_1 * b_3;
		c_2 = a_2 * b_3;
		c_3 = a_3 * b_3;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;
		D[2+bs*3] = c_2;
		D[3+bs*3] = c_3;

		A += 4*sda;
		D += 4*sdd;
		
		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		
		c_0 = a_0 * b_0;

		D[0+bs*0] = c_0;
		

		a_0 = A[0+bs*1];
		
		c_0 = a_0 * b_1;

		D[0+bs*1] = c_0;
		

		a_0 = A[0+bs*2];
		
		c_0 = a_0 * b_2;

		D[0+bs*2] = c_0;
		

		a_0 = A[0+bs*3];
		
		c_0 = a_0 * b_3;

		D[0+bs*3] = c_0;


		A += 1;
		D += 1;
		
		}
	
	}




// B is the diagonal of a matrix
void kernel_dgemm_diag_right_4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
	
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	b_0 = alpha0 * B[0];
	b_1 = alpha0 * B[1];
	b_2 = alpha0 * B[2];
	b_3 = alpha0 * B[3];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_0;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_0;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_0;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*1] + a_2 * b_1;
		c_3 = beta0 * C[3+bs*1] + a_3 * b_1;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_2;
		c_1 = beta0 * C[1+bs*2] + a_1 * b_2;
		c_2 = beta0 * C[2+bs*2] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*2] + a_3 * b_2;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;
		

		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_3;
		c_1 = beta0 * C[1+bs*3] + a_1 * b_3;
		c_2 = beta0 * C[2+bs*3] + a_2 * b_3;
		c_3 = beta0 * C[3+bs*3] + a_3 * b_3;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;
		D[2+bs*3] = c_2;
		D[3+bs*3] = c_3;

		A += 4*sda;
		C += 4*sdc;
		D += 4*sdd;
		
		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
		

		a_0 = A[0+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;

		D[0+bs*1] = c_0;
		

		a_0 = A[0+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_2;

		D[0+bs*2] = c_0;
		

		a_0 = A[0+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_3;

		D[0+bs*3] = c_0;


		A += 1;
		C += 1;
		D += 1;
		
		}
	
	}




// B is the diagonal of a matrix
void kernel_dgemm_diag_right_3_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2,
		c_0, c_1, c_2, c_3;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	b_0 = alpha0 * B[0];
	b_1 = alpha0 * B[1];
	b_2 = alpha0 * B[2];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_0;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_0;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_0;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*1] + a_2 * b_1;
		c_3 = beta0 * C[3+bs*1] + a_3 * b_1;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_2;
		c_1 = beta0 * C[1+bs*2] + a_1 * b_2;
		c_2 = beta0 * C[2+bs*2] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*2] + a_3 * b_2;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;
		

		A += 4*sda;
		C += 4*sdc;
		D += 4*sdd;
		
		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
		

		a_0 = A[0+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;

		D[0+bs*1] = c_0;
		

		a_0 = A[0+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_2;

		D[0+bs*2] = c_0;
		

		A += 1;
		C += 1;
		D += 1;
		
		}
	
	}



// B is the diagonal of a matrix
void kernel_dgemm_diag_right_2_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_0, c_1, c_2, c_3;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	b_0 = alpha0 * B[0];
	b_1 = alpha0 * B[1];

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_0;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_0;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_0;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*1] + a_2 * b_1;
		c_3 = beta0 * C[3+bs*1] + a_3 * b_1;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		A += 4*sda;
		C += 4*sdc;
		D += 4*sdd;
		
		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
		

		a_0 = A[0+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_1;

		D[0+bs*1] = c_0;
		

		A += 1;
		C += 1;
		D += 1;
		
		}
	
	}



// B is the diagonal of a matrix
void kernel_dgemm_diag_right_1_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2, a_3,
		b_0,
		c_0, c_1, c_2, c_3;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	b_0 = alpha0 * B[0];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_0;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_0;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_0;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		A += 4*sda;
		C += 4*sdc;
		D += 4*sdd;
		
		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
		

		A += 1;
		C += 1;
		D += 1;
		
		}
	
	}



// A is the diagonal of a matrix, case beta=0.0
void kernel_dgemm_diag_left_4_a0_lib4(int kmax, double *alpha, double *A, double *B, double *D, int alg)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
		
	alpha0 = alpha[0];
		
	a_0 = alpha0 * A[0];
	a_1 = alpha0 * A[1];
	a_2 = alpha0 * A[2];
	a_3 = alpha0 * A[3];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_3;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_3;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_3;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;
		

		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
		b_3 = B[3+bs*3];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_3;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;
		D[2+bs*3] = c_2;
		D[3+bs*3] = c_3;

		B += 16;
		D += 16;
		
		}
	for(; k<kmax; k++)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = a_0 * b_0;
		c_1 = a_1 * b_1;
		c_2 = a_2 * b_2;
		c_3 = a_3 * b_3;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
	
		B += 4;
		D += 4;
		
		}
	
	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D, int alg)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	a_0 = alpha0 * A[0];
	a_1 = alpha0 * A[1];
	a_2 = alpha0 * A[2];
	a_3 = alpha0 * A[3];
	
	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_3;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
		

		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*1] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*1] + a_3 * b_3;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;
		

		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*2] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*2] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*2] + a_3 * b_3;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;
		

		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
		b_3 = B[3+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*3] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*3] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*3] + a_3 * b_3;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;
		D[2+bs*3] = c_2;
		D[3+bs*3] = c_3;

		B += 16;
		C += 16;
		D += 16;
		
		}
	for(; k<kmax; k++)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_2;
		c_3 = beta0 * C[3+bs*0] + a_3 * b_3;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;
	
		B += 4;
		C += 4;
		D += 4;
		
		}
	
	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_3_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1, a_2,
		b_0, b_1, b_2,
		c_0, c_1, c_2;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	a_0 = alpha0 * A[0];
	a_1 = alpha0 * A[1];
	a_2 = alpha0 * A[2];

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_2;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		

		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*1] + a_2 * b_2;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		

		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*2] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*2] + a_2 * b_2;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		D[2+bs*2] = c_2;
		

		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*3] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*3] + a_2 * b_2;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;
		D[2+bs*3] = c_2;

		B += 16;
		C += 16;
		D += 16;
		
		}
	for(; k<kmax; k++)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;
		c_2 = beta0 * C[2+bs*0] + a_2 * b_2;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
	
		B += 4;
		C += 4;
		D += 4;
		
		}
	
	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_2_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0, a_1,
		b_0, b_1,
		c_0, c_1;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	a_0 = alpha0 * A[0];
	a_1 = alpha0 * A[1];

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		

		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*1] + a_1 * b_1;

		D[0+bs*1] = c_0;
		D[1+bs*1] = c_1;
		

		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*2] + a_1 * b_1;

		D[0+bs*2] = c_0;
		D[1+bs*2] = c_1;
		

		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*3] + a_1 * b_1;

		D[0+bs*3] = c_0;
		D[1+bs*3] = c_1;

		B += 16;
		C += 16;
		D += 16;
		
		}
	for(; k<kmax; k++)
		{
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;
		c_1 = beta0 * C[1+bs*0] + a_1 * b_1;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
	
		B += 4;
		C += 4;
		D += 4;
		
		}
	
	}


// A is the diagonal of a matrix
void kernel_dgemm_diag_left_1_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		alpha0, beta0,
		a_0,
		b_0,
		c_0;
		
	alpha0 = alpha[0];
	beta0  = beta[0];
		
	a_0 = alpha0 * A[0];
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = B[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
		

		b_0 = B[0+bs*1];
		
		c_0 = beta0 * C[0+bs*1] + a_0 * b_0;

		D[0+bs*1] = c_0;
		

		b_0 = B[0+bs*2];
		
		c_0 = beta0 * C[0+bs*2] + a_0 * b_0;

		D[0+bs*2] = c_0;
		

		b_0 = B[0+bs*3];
		
		c_0 = beta0 * C[0+bs*3] + a_0 * b_0;

		D[0+bs*3] = c_0;

		B += 16;
		C += 16;
		D += 16;
		
		}
	for(; k<kmax; k++)
		{
		
		b_0 = B[0+bs*0];
		
		c_0 = beta0 * C[0+bs*0] + a_0 * b_0;

		D[0+bs*0] = c_0;
	
		B += 4;
		C += 4;
		D += 4;
		
		}
		
	}



