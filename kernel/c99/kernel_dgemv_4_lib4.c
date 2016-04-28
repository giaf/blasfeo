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



void kernel_dgemv_n_4_vs_lib4_b(int kmax, double *A, double *x, int alg, double *y, double *z, int km)
	{

	const int bs = 4;

	int k;

	double
		x_0,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];

		y_0 += A[0+bs*0] * x_0;
		y_1 += A[1+bs*0] * x_0;
		y_2 += A[2+bs*0] * x_0;
		y_3 += A[3+bs*0] * x_0;
		
		x_0 = x[1];

		y_0 += A[0+bs*1] * x_0;
		y_1 += A[1+bs*1] * x_0;
		y_2 += A[2+bs*1] * x_0;
		y_3 += A[3+bs*1] * x_0;
		
		x_0 = x[2];

		y_0 += A[0+bs*2] * x_0;
		y_1 += A[1+bs*2] * x_0;
		y_2 += A[2+bs*2] * x_0;
		y_3 += A[3+bs*2] * x_0;
		
		x_0 = x[3];

		y_0 += A[0+bs*3] * x_0;
		y_1 += A[1+bs*3] * x_0;
		y_2 += A[2+bs*3] * x_0;
		y_3 += A[3+bs*3] * x_0;
		
		A += 4*bs;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+bs*0] * x_0;
		y_1 += A[1+bs*0] * x_0;
		y_2 += A[2+bs*0] * x_0;
		y_3 += A[3+bs*0] * x_0;
		
		A += 1*bs;
		x += 1;

		}

	// store_vs
	if(alg==0)
		{
		goto store;
		}
	else if(alg==1)
		{
		y_0 += y[0];
		y_1 += y[1];
		y_2 += y[2];
		y_3 += y[3];

		goto store;
		}
	else // alg==-1
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		y_3 = y[3] - y_3;

		goto store;
		}

	store:
	if(km>=4)
		{
		z[0] = y_0;
		z[1] = y_1;
		z[2] = y_2;
		z[3] = y_3;
		}
	else
		{
		z[0] = y_0;
		if(km>=2)
			{
			z[1] = y_1;
			if(km>2)
				{
				z[2] = y_2;
				}
			}
		}

	}
	
	
	

void kernel_dgemv_n_4_lib4_b(int kmax, double *A, double *x, int alg, double *y, double *z, int km)
	{

	kernel_dgemv_n_4_vs_lib4_b(kmax, A, x, alg, y, z, 4);

	}



void kernel_dgemv_t_4_vs_lib4_b(int kmax, double *A, int sda, double *x, int alg, double *y, double *z, int km)
	{

	if(kmax<=0) 
		return;
	
	const int bs  = 4;
	
	int k;
	
	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	k=0;
	for(; k<kmax-bs+1; k+=bs)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+bs*0] * x_0;
		y_1 += A[0+bs*1] * x_0;
		y_2 += A[0+bs*2] * x_0;
		y_3 += A[0+bs*3] * x_0;

		y_0 += A[1+bs*0] * x_1;
		y_1 += A[1+bs*1] * x_1;
		y_2 += A[1+bs*2] * x_1;
		y_3 += A[1+bs*3] * x_1;
		
		y_0 += A[2+bs*0] * x_2;
		y_1 += A[2+bs*1] * x_2;
		y_2 += A[2+bs*2] * x_2;
		y_3 += A[2+bs*3] * x_2;

		y_0 += A[3+bs*0] * x_3;
		y_1 += A[3+bs*1] * x_3;
		y_2 += A[3+bs*2] * x_3;
		y_3 += A[3+bs*3] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
	
		y_0 += A[0+bs*0] * x_0;
		y_1 += A[0+bs*1] * x_0;
		y_2 += A[0+bs*2] * x_0;
		y_3 += A[0+bs*3] * x_0;
	
		A += 1;
		x += 1;
		
		}

	if(alg==0)
		{
		goto store;
		}
	else if(alg==1)
		{
		y_0 += y[0];
		y_1 += y[1];
		y_2 += y[2];
		y_3 += y[3];

		goto store;
		}
	else // alg==-1
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		y_3 = y[3] - y_3;

		goto store;
		}

	store:
	if(km>=4)
		{
		z[0] = y_0;
		z[1] = y_1;
		z[2] = y_2;
		z[3] = y_3;
		}
	else
		{
		z[0] = y_0;
		if(km>=2)
			{
			z[1] = y_1;
			if(km>2)
				{
				z[2] = y_2;
				}
			}
		}

	}
	
	
	
void kernel_dgemv_t_4_lib4_b(int kmax, double *A, int sda, double *x, int alg, double *y, double *z)
	{

	kernel_dgemv_t_4_vs_lib4_b(kmax, A, sda, x, alg, y, z, 4);

	}




void kernel_dtrsv_ln_inv_4_vs_lib4_b(int kmax, double *A, double *inv_diag_A, double *x, int alg, double *y, double *z, int km, int kn)
	{

	const int bs = 4;
	
	int k;

	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 -= A[0+bs*0] * x_0;
		y_1 -= A[1+bs*0] * x_0;
		y_2 -= A[2+bs*0] * x_0;
		y_3 -= A[3+bs*0] * x_0;

		y_0 -= A[0+bs*1] * x_1;
		y_1 -= A[1+bs*1] * x_1;
		y_2 -= A[2+bs*1] * x_1;
		y_3 -= A[3+bs*1] * x_1;

		y_0 -= A[0+bs*2] * x_2;
		y_1 -= A[1+bs*2] * x_2;
		y_2 -= A[2+bs*2] * x_2;
		y_3 -= A[3+bs*2] * x_2;

		y_0 -= A[0+bs*3] * x_3;
		y_1 -= A[1+bs*3] * x_3;
		y_2 -= A[2+bs*3] * x_3;
		y_3 -= A[3+bs*3] * x_3;
		
		A += 4*bs;
		x += 4;

		}

	if(alg==1)
		{
		y_0 = y[0] + y_0;
		y_1 = y[1] + y_1;
		y_2 = y[2] + y_2;
		y_3 = y[3] + y_3;
		}
	else if(alg==-1)
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		y_3 = y[3] - y_3;
		}

	double
		a_00, a_10, a_20, a_30,
		a_11, a_21, a_31;
	
	// a_00
	a_00 = inv_diag_A[0];
	a_10 = A[1+bs*0];
	a_20 = A[2+bs*0];
	a_30 = A[3+bs*0];
	y_0 *= a_00;
	z[0] = y_0;
	y_1 -= a_10 * y_0;
	y_2 -= a_20 * y_0;
	y_3 -= a_30 * y_0;

	if(kn==1)
		{
		if(km==1)
			return;
		y[1] = y_1;
		if(km==2)
			return;
		y[2] = y_2;
		if(km==3)
			return;
		y[3] = y_3;
		return;
		}

	// a_11
	a_11 = inv_diag_A[1];
	a_21 = A[2+bs*1];
	a_31 = A[3+bs*1];
	y_1 *= a_11;	
	z[1] = y_1;
	y_2 -= a_21 * y_1;
	y_3 -= a_31 * y_1;

	if(kn==2)
		{
		if(km==2)
			return;
		y[2] = y_2;
		if(km==3)
			return;
		y[3] = y_3;
		return;
		}

	// a_22
	a_00 = inv_diag_A[2];
	a_10 = A[3+bs*2];
	y_2 *= a_00;
	z[2] = y_2;
	y_3 -= a_10 * y_2;

	if(kn==3)
		{
		if(km==3)
			return;
		y[3] = y_3;

		return;
		}

	// a_33
	a_11 = inv_diag_A[3];
	y_3 *= a_11;	
	z[3] = y_3;

	}
	

	
void kernel_dtrsv_ln_inv_4_lib4_b(int kmax, double *A, double *inv_diag_A, double *x, int alg, double *y, double *z)
	{

	kernel_dtrsv_ln_inv_4_vs_lib4_b(kmax, A, inv_diag_A, x, alg, y, z, 4, 4);


	}
	
	
		
void kernel_dtrsv_lt_inv_4_lib4_b(int kmax, double *A, int sda, double *inv_diag_A, double *x, int alg, double *y, double *z)
	{

	const int bs = 4;
	
	int
		k;
	
	double *tA, *tx;
	tA = A;
	tx = x;

	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	k=4;
	A += 4 + (sda-1)*bs;
	x += 4;
	for(; k<kmax-3; k+=4)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 -= A[0+bs*0] * x_0;
		y_1 -= A[0+bs*1] * x_0;
		y_2 -= A[0+bs*2] * x_0;
		y_3 -= A[0+bs*3] * x_0;

		y_0 -= A[1+bs*0] * x_1;
		y_1 -= A[1+bs*1] * x_1;
		y_2 -= A[1+bs*2] * x_1;
		y_3 -= A[1+bs*3] * x_1;
		
		y_0 -= A[2+bs*0] * x_2;
		y_1 -= A[2+bs*1] * x_2;
		y_2 -= A[2+bs*2] * x_2;
		y_3 -= A[2+bs*3] * x_2;

		y_0 -= A[3+bs*0] * x_3;
		y_1 -= A[3+bs*1] * x_3;
		y_2 -= A[3+bs*2] * x_3;
		y_3 -= A[3+bs*3] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
		
		y_0 -= A[0+bs*0] * x_0;
		y_1 -= A[0+bs*1] * x_0;
		y_2 -= A[0+bs*2] * x_0;
		y_3 -= A[0+bs*3] * x_0;
		
		A += 1;//sda*bs;
		x += 1;

		}
	
	if(alg==1)
		{
		y_0 = y[0] + y_0;
		y_1 = y[1] + y_1;
		y_2 = y[2] + y_2;
		y_3 = y[3] + y_3;
		}
	else if(alg==-1)
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		y_3 = y[3] - y_3;
		}

	A = tA;
	x = tx;

	// bottom trinagle
	y_3 *= inv_diag_A[3];
	z[3] = y_3;

	y_2 -= A[3+bs*2] * y_3;
	y_2 *= inv_diag_A[2];
	z[2] = y_2;

	// square
	y_0 -= A[2+bs*0]*y_2 + A[3+bs*0]*y_3;
	y_1 -= A[2+bs*1]*y_2 + A[3+bs*1]*y_3;
		
	// top trinagle
	y_1 *= inv_diag_A[1];
	z[1] = y_1;

	y_0 -= A[1+bs*0] * y_1;
	y_0 *= inv_diag_A[0];
	z[0] = y_0;

	}
	
	
	
void kernel_dtrsv_lt_inv_3_lib4_b(int kmax, double *A, int sda, double *inv_diag_A, double *x, int alg, double *y, double *z)
	{

	const int bs = 4;
	
	int
		k;
	
	double *tA, *tx;
	tA = A;
	tx = x;

	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0;
	
	k = 3;
	if(kmax>4)
		{
		// clean up at the beginning
		x_3 = x[3];

		y_0 -= A[3+bs*0] * x_3;
		y_1 -= A[3+bs*1] * x_3;
		y_2 -= A[3+bs*2] * x_3;

		k=4;
		A += 4 + (sda-1)*bs;
		x += 4;
		for(; k<kmax-3; k+=4)
			{
			
			x_0 = x[0];
			x_1 = x[1];
			x_2 = x[2];
			x_3 = x[3];
			
			y_0 -= A[0+bs*0] * x_0;
			y_1 -= A[0+bs*1] * x_0;
			y_2 -= A[0+bs*2] * x_0;

			y_0 -= A[1+bs*0] * x_1;
			y_1 -= A[1+bs*1] * x_1;
			y_2 -= A[1+bs*2] * x_1;
			
			y_0 -= A[2+bs*0] * x_2;
			y_1 -= A[2+bs*1] * x_2;
			y_2 -= A[2+bs*2] * x_2;

			y_0 -= A[3+bs*0] * x_3;
			y_1 -= A[3+bs*1] * x_3;
			y_2 -= A[3+bs*2] * x_3;
			
			A += sda*bs;
			x += 4;

			}
		}
	else
		{
		A += 3;
		x += 1;
		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
		
		y_0 -= A[0+bs*0] * x_0;
		y_1 -= A[0+bs*1] * x_0;
		y_2 -= A[0+bs*2] * x_0;
		
		A += 1;//sda*bs;
		x += 1;

		}

	if(alg==1)
		{
		y_0 = y[0] + y_0;
		y_1 = y[1] + y_1;
		y_2 = y[2] + y_2;
		}
	else if(alg==-1)
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		}

	A = tA;
	x = tx;

	// bottom trinagle
	y_2 *= inv_diag_A[2];
	z[2] = y_2;

	// square
	y_0 -= A[2+bs*0]*y_2;
	y_1 -= A[2+bs*1]*y_2;
		
	// top trinagle
	y_1 *= inv_diag_A[1];
	z[1] = y_1;

	y_0 -= A[1+bs*0] * y_1;
	y_0 *= inv_diag_A[0];
	z[0] = y_0;

	}
	
	
	
void kernel_dtrsv_lt_inv_2_lib4_b(int kmax, double *A, int sda, double *inv_diag_A, double *x, int alg, double *y, double *z)
	{

	const int bs = 4;
	
	int
		k;
	
	double *tA, *tx;
	tA = A;
	tx = x;

	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0;
	
	k = 2;
	if(kmax>4)
		{
		// clean up at the beginning
		x_2 = x[2];
		x_3 = x[3];

		y_0 -= A[2+bs*0] * x_2;
		y_1 -= A[2+bs*1] * x_2;

		y_0 -= A[3+bs*0] * x_3;
		y_1 -= A[3+bs*1] * x_3;

		k=4;
		A += 4 + (sda-1)*bs;
		x += 4;
		for(; k<kmax-3; k+=4)
			{
			
			x_0 = x[0];
			x_1 = x[1];
			x_2 = x[2];
			x_3 = x[3];
			
			y_0 -= A[0+bs*0] * x_0;
			y_1 -= A[0+bs*1] * x_0;

			y_0 -= A[1+bs*0] * x_1;
			y_1 -= A[1+bs*1] * x_1;
			
			y_0 -= A[2+bs*0] * x_2;
			y_1 -= A[2+bs*1] * x_2;

			y_0 -= A[3+bs*0] * x_3;
			y_1 -= A[3+bs*1] * x_3;
			
			A += sda*bs;
			x += 4;

			}
		}
	else
		{
		A += 2;
		x += 2;
		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
		
		y_0 -= A[0+bs*0] * x_0;
		y_1 -= A[0+bs*1] * x_0;
		
		A += 1;//sda*bs;
		x += 1;

		}

	if(alg==1)
		{
		y_0 = y[0] + y_0;
		y_1 = y[1] + y_1;
		}
	else if(alg==-1)
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		}

	A = tA;
	x = tx;

	// top trinagle
	y_1 *= inv_diag_A[1];
	z[1] = y_1;

	y_0 -= A[1+bs*0] * y_1;
	y_0 *= inv_diag_A[0];
	z[0] = y_0;

	}
	
	
	
void kernel_dtrsv_lt_inv_1_lib4_b(int kmax, double *A, int sda, double *inv_diag_A, double *x, int alg, double *y, double *z)
	{

	const int bs = 4;
	
	int
		k;
	
	double *tA, *tx;
	tA = A;
	tx = x;

	double
		x_0, x_1, x_2, x_3,
		y_0=0;
	
	k = 1;
	if(kmax>4)
		{
		// clean up at the beginning
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 -= A[1+bs*0] * x_1;
		y_0 -= A[2+bs*0] * x_2;
		y_0 -= A[3+bs*0] * x_3;

		k=4;
		A += 4 + (sda-1)*bs;
		x += 4;
		for(; k<kmax-3; k+=4)
			{
			
			x_0 = x[0];
			x_1 = x[1];
			x_2 = x[2];
			x_3 = x[3];
			
			y_0 -= A[0+bs*0] * x_0;
			y_0 -= A[1+bs*0] * x_1;
			y_0 -= A[2+bs*0] * x_2;
			y_0 -= A[3+bs*0] * x_3;
			
			A += sda*bs;
			x += 4;

			}
		}
	else
		{
		A += 1;
		x += 1;
		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
		
		y_0 -= A[0+bs*0] * x_0;
		
		A += 1;//sda*bs;
		x += 1;

		}

	if(alg==1)
		{
		y_0 = y[0] + y_0;
		}
	else if(alg==-1)
		{
		y_0 = y[0] - y_0;
		}

	A = tA;
	x = tx;

	// top trinagle
	y_0 *= inv_diag_A[0];
	z[0] = y_0;

	}
	
	
	
void kernel_dtrmv_un_4_lib4(int kmax, double *A, double *x, int alg, double *y, double *z)
	{

	const int bs = 4;
	
	int k;

	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	x_0 = x[0];
	x_1 = x[1];
	x_2 = x[2];
	x_3 = x[3];

	y_0 += A[0+bs*0] * x_0;
/*	y_1 += A[1+bs*0] * x_0;*/
/*	y_2 += A[2+bs*0] * x_0;*/
/*	y_3 += A[3+bs*0] * x_0;*/

	y_0 += A[0+bs*1] * x_1;
	y_1 += A[1+bs*1] * x_1;
/*	y_2 += A[2+bs*1] * x_1;*/
/*	y_3 += A[3+bs*1] * x_1;*/

	y_0 += A[0+bs*2] * x_2;
	y_1 += A[1+bs*2] * x_2;
	y_2 += A[2+bs*2] * x_2;
/*	y_3 += A[3+bs*2] * x_2;*/

	y_0 += A[0+bs*3] * x_3;
	y_1 += A[1+bs*3] * x_3;
	y_2 += A[2+bs*3] * x_3;
	y_3 += A[3+bs*3] * x_3;
	
	A += 4*bs;
	x += 4;

	k=4;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 += A[0+bs*0] * x_0;
		y_1 += A[1+bs*0] * x_0;
		y_2 += A[2+bs*0] * x_0;
		y_3 += A[3+bs*0] * x_0;

		y_0 += A[0+bs*1] * x_1;
		y_1 += A[1+bs*1] * x_1;
		y_2 += A[2+bs*1] * x_1;
		y_3 += A[3+bs*1] * x_1;

		y_0 += A[0+bs*2] * x_2;
		y_1 += A[1+bs*2] * x_2;
		y_2 += A[2+bs*2] * x_2;
		y_3 += A[3+bs*2] * x_2;

		y_0 += A[0+bs*3] * x_3;
		y_1 += A[1+bs*3] * x_3;
		y_2 += A[2+bs*3] * x_3;
		y_3 += A[3+bs*3] * x_3;
		
		A += 4*bs;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+bs*0] * x_0;
		y_1 += A[1+bs*0] * x_0;
		y_2 += A[2+bs*0] * x_0;
		y_3 += A[3+bs*0] * x_0;
		
		A += 1*bs;
		x += 1;

		}

	if(alg==0)
		{
		z[0] = y_0;
		z[1] = y_1;
		z[2] = y_2;
		z[3] = y_3;
		}
	else if(alg==1)
		{
		z[0] = y[0] + y_0;
		z[1] = y[1] + y_1;
		z[2] = y[2] + y_2;
		z[3] = y[3] + y_3;
		}
	else // alg==-1
		{
		z[0] = y[0] - y_0;
		z[1] = y[1] - y_1;
		z[2] = y[2] - y_2;
		z[3] = y[3] - y_3;
		}

	}
	
	
	
void kernel_dtrmv_ut_4_vs_lib4(int kmax, double *A, int sda, double *x, int alg, double *y, double *z, int km)
	{

	const int bs  = 4;
	
	int
		k;
	
	double
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0, y_2=0, y_3=0;
	
	k=0;
	for(; k<kmax-4; k+=4)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+bs*0] * x_0;
		y_1 += A[0+bs*1] * x_0;
		y_2 += A[0+bs*2] * x_0;
		y_3 += A[0+bs*3] * x_0;

		y_0 += A[1+bs*0] * x_1;
		y_1 += A[1+bs*1] * x_1;
		y_2 += A[1+bs*2] * x_1;
		y_3 += A[1+bs*3] * x_1;
		
		y_0 += A[2+bs*0] * x_2;
		y_1 += A[2+bs*1] * x_2;
		y_2 += A[2+bs*2] * x_2;
		y_3 += A[2+bs*3] * x_2;

		y_0 += A[3+bs*0] * x_3;
		y_1 += A[3+bs*1] * x_3;
		y_2 += A[3+bs*2] * x_3;
		y_3 += A[3+bs*3] * x_3;
		
		A += sda*bs;
		x += 4;

		}

	x_0 = x[0];
	x_1 = x[1];
	x_2 = x[2];
	x_3 = x[3];
	
	y_0 += A[0+bs*0] * x_0;
	y_1 += A[0+bs*1] * x_0;
	y_2 += A[0+bs*2] * x_0;
	y_3 += A[0+bs*3] * x_0;

/*	y_0 += A[1+bs*0] * x_1;*/
	y_1 += A[1+bs*1] * x_1;
	y_2 += A[1+bs*2] * x_1;
	y_3 += A[1+bs*3] * x_1;
	
/*	y_0 += A[2+bs*0] * x_2;*/
/*	y_1 += A[2+bs*1] * x_2;*/
	y_2 += A[2+bs*2] * x_2;
	y_3 += A[2+bs*3] * x_2;

/*	y_0 += A[3+bs*0] * x_3;*/
/*	y_1 += A[3+bs*1] * x_3;*/
/*	y_2 += A[3+bs*2] * x_3;*/
	y_3 += A[3+bs*3] * x_3;
	
//	A += sda*bs;
//	x += 4;

	// store_vs
	if(alg==0)
		{
		goto store;
		}
	else if(alg==1)
		{
		y_0 += y[0];
		y_1 += y[1];
		y_2 += y[2];
		y_3 += y[3];

		goto store;
		}
	else // alg==-1
		{
		y_0 = y[0] - y_0;
		y_1 = y[1] - y_1;
		y_2 = y[2] - y_2;
		y_3 = y[3] - y_3;

		goto store;
		}

	store:
	if(km>=4)
		{
		z[0] = y_0;
		z[1] = y_1;
		z[2] = y_2;
		z[3] = y_3;
		}
	else
		{
		z[0] = y_0;
		if(km>=2)
			{
			z[1] = y_1;
			if(km>2)
				{
				z[2] = y_2;
				}
			}
		}

	}
	
	
	
void kernel_dtrmv_ut_4_lib4(int kmax, double *A, int sda, double *x, int alg, double *y, double *z)
	{
	
	kernel_dtrmv_ut_4_vs_lib4(kmax, A, sda, x, alg, y, z, 4);

	}





