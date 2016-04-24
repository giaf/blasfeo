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


