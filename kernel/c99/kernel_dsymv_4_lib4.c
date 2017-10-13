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



// XXX copy and scale y_n into z_n outside the kernel !!!!!
#if defined(TARGET_GENERIC) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV8A_ARM_CORTEX_A57)
void kernel_dgemv_nt_4_vs_lib4(int kmax, double *alpha_n, double *alpha_t, double *A, int sda, double *x_n, double *x_t, double *beta_t, double *y_t, double *z_n, double *z_t, int km)
	{

	if(kmax<=0) 
		return;
	
	const int bs = 4;

	int k;

	double
		a_00, a_01, a_02, a_03,
		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,
		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;
	
	x_n_0 = 0;
	x_n_1 = 0;
	x_n_2 = 0;
	x_n_3 = 0;

	x_n_0 = alpha_n[0]*x_n[0];
	if(km>1)
		{
		x_n_1 = alpha_n[0]*x_n[1];
		if(km>2)
			{
			x_n_2 = alpha_n[0]*x_n[2];
			if(km>3)
				{
				x_n_3 = alpha_n[0]*x_n[3];
				}
			}
		}

	y_t_0 = 0;
	y_t_1 = 0;
	y_t_2 = 0;
	y_t_3 = 0;

	k = 0;
	for(; k<kmax-3; k+=bs)
		{

		// 0

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;


		// 1

		y_n_0 = z_n[1]; 
		x_t_0 = x_t[1];

		a_00 = A[1+bs*0];
		a_01 = A[1+bs*1];
		a_02 = A[1+bs*2];
		a_03 = A[1+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[1] = y_n_0;


		// 2

		y_n_0 = z_n[2]; 
		x_t_0 = x_t[2];

		a_00 = A[2+bs*0];
		a_01 = A[2+bs*1];
		a_02 = A[2+bs*2];
		a_03 = A[2+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[2] = y_n_0;


		// 3

		y_n_0 = z_n[3]; 
		x_t_0 = x_t[3];

		a_00 = A[3+bs*0];
		a_01 = A[3+bs*1];
		a_02 = A[3+bs*2];
		a_03 = A[3+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[3] = y_n_0;


		A += sda*bs;
		z_n += 4;
		x_t += 4;

		}
	for(; k<kmax; k++)
		{

		// 0

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		}
	
	// store t
	z_t[0] = alpha_t[0]*y_t_0 + beta_t[0]*y_t[0];
	if(km>1)
		{
		z_t[1] = alpha_t[0]*y_t_1 + beta_t[0]*y_t[1];
		if(km>2)
			{
			z_t[2] = alpha_t[0]*y_t_2 + beta_t[0]*y_t[2];
			if(km>3)
				{
				z_t[3] = alpha_t[0]*y_t_3 + beta_t[0]*y_t[3];
				}
			}
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV8A_ARM_CORTEX_A57)
// XXX copy and scale y_n into z_n outside the kernel !!!!!
void kernel_dgemv_nt_4_lib4(int kmax, double *alpha_n, double *alpha_t, double *A, int sda, double *x_n, double *x_t, double *beta_t, double *y_t, double *z_n, double *z_t)
	{

	kernel_dgemv_nt_4_vs_lib4(kmax, alpha_n, alpha_t, A, sda, x_n, x_t, beta_t, y_t, z_n, z_t, 4);

	return;

	}
#endif



// XXX copy and scale y_n into z_n outside the kernel !!!!!
#if defined(TARGET_GENERIC) || defined(TARGET_X64_INTEL_CORE) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV8A_ARM_CORTEX_A57)
void kernel_dsymv_l_4_gen_lib4(int kmax, double *alpha, int offA, double *A, int sda, double *x_n, double *z_n, int km)
	{

	if(kmax<=0) 
		return;
	
	double *x_t = x_n;
	double *z_t = z_n;

	const int bs = 4;

	int k;

	double
		a_00, a_01, a_02, a_03,
		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,
		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;
	
	x_n_0 = 0;
	x_n_1 = 0;
	x_n_2 = 0;
	x_n_3 = 0;

	x_n_0 = alpha[0]*x_n[0];
	if(km>1)
		{
		x_n_1 = alpha[0]*x_n[1];
		if(km>2)
			{
			x_n_2 = alpha[0]*x_n[2];
			if(km>3)
				{
				x_n_3 = alpha[0]*x_n[3];
				}
			}
		}

	y_t_0 = 0;
	y_t_1 = 0;
	y_t_2 = 0;
	y_t_3 = 0;

	k = 0;
	if(offA==0)
		{
		if(kmax<4)
			{
			// 0

			x_t_0 = x_t[0];

			a_00 = A[0+bs*0];
			
			y_t_0 += a_00 * x_t_0;

			if(kmax==1)
				goto store_t;

			// 1

			y_n_0 = z_n[1]; 
			x_t_0 = x_t[1];

			a_00 = A[1+bs*0];
			a_01 = A[1+bs*1];
			
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
			y_t_1 += a_01 * x_t_0;

			z_n[1] = y_n_0;

			if(kmax==2)
				goto store_t;

			// 2

			y_n_0 = z_n[2]; 
			x_t_0 = x_t[2];

			a_00 = A[2+bs*0];
			a_01 = A[2+bs*1];
			a_02 = A[2+bs*2];
			
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
			y_n_0 += a_01 * x_n_1;
			y_t_1 += a_01 * x_t_0;
			y_t_2 += a_02 * x_t_0;

			z_n[2] = y_n_0;

			goto store_t;
			}
		else
			{

			// 0

			x_t_0 = x_t[0];

			a_00 = A[0+bs*0];
			
			y_t_0 += a_00 * x_t_0;


			// 1

			y_n_0 = z_n[1]; 
			x_t_0 = x_t[1];

			a_00 = A[1+bs*0];
			a_01 = A[1+bs*1];
			
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
			y_t_1 += a_01 * x_t_0;

			z_n[1] = y_n_0;


			// 2

			y_n_0 = z_n[2]; 
			x_t_0 = x_t[2];

			a_00 = A[2+bs*0];
			a_01 = A[2+bs*1];
			a_02 = A[2+bs*2];
			
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
			y_n_0 += a_01 * x_n_1;
			y_t_1 += a_01 * x_t_0;
			y_t_2 += a_02 * x_t_0;

			z_n[2] = y_n_0;


			// 3

			y_n_0 = z_n[3]; 
			x_t_0 = x_t[3];

			a_00 = A[3+bs*0];
			a_01 = A[3+bs*1];
			a_02 = A[3+bs*2];
			a_03 = A[3+bs*3];
			
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
			y_n_0 += a_01 * x_n_1;
			y_t_1 += a_01 * x_t_0;
			y_n_0 += a_02 * x_n_2;
			y_t_2 += a_02 * x_t_0;
			y_t_3 += a_03 * x_t_0;

			z_n[3] = y_n_0;

			k += 4;
			A += sda*bs;
			z_n += 4;
			x_t += 4;

			}
		}
	else if(offA==1)
		{

		// 0

		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		
		y_t_0 += a_00 * x_t_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==1)
			goto store_t;

		// 1

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_t_1 += a_01 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==2)
			goto store_t;

		// 2

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_t_2 += a_02 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==3)
			goto store_t;

		// 3

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==4)
			goto store_t;

		// 4

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==5)
			goto store_t;

		// 5

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==6)
			goto store_t;

		// 6

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==7)
			goto store_t;

		k += 7;

		}
	else if(offA==2)
		{

		// 0

		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		
		y_t_0 += a_00 * x_t_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==1)
			goto store_t;

		// 1

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_t_1 += a_01 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==2)
			goto store_t;

		// 2

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_t_2 += a_02 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==3)
			goto store_t;

		// 3

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==4)
			goto store_t;

		// 4

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==5)
			goto store_t;

		// 5

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==6)
			goto store_t;

		k += 6;

		}
	else // if(offA==3)
		{

		// 0

		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		
		y_t_0 += a_00 * x_t_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==1)
			goto store_t;

		// 1

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_t_1 += a_01 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==2)
			goto store_t;

		// 2

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_t_2 += a_02 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==3)
			goto store_t;

		// 3

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		if(kmax==4)
			goto store_t;

		// 4

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		A += (sda-1)*bs; // new panel

		if(kmax==5)
			goto store_t;

		k += 5;

		}
	for(; k<kmax-3; k+=bs)
		{

		// 0

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;


		// 1

		y_n_0 = z_n[1]; 
		x_t_0 = x_t[1];

		a_00 = A[1+bs*0];
		a_01 = A[1+bs*1];
		a_02 = A[1+bs*2];
		a_03 = A[1+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[1] = y_n_0;


		// 2

		y_n_0 = z_n[2]; 
		x_t_0 = x_t[2];

		a_00 = A[2+bs*0];
		a_01 = A[2+bs*1];
		a_02 = A[2+bs*2];
		a_03 = A[2+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[2] = y_n_0;


		// 3

		y_n_0 = z_n[3]; 
		x_t_0 = x_t[3];

		a_00 = A[3+bs*0];
		a_01 = A[3+bs*1];
		a_02 = A[3+bs*2];
		a_03 = A[3+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[3] = y_n_0;


		A += sda*bs;
		z_n += 4;
		x_t += 4;

		}
	for(; k<kmax; k++)
		{

		// 0

		y_n_0 = z_n[0]; 
		x_t_0 = x_t[0];

		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		a_02 = A[0+bs*2];
		a_03 = A[0+bs*3];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		y_n_0 += a_02 * x_n_2;
		y_t_2 += a_02 * x_t_0;
		y_n_0 += a_03 * x_n_3;
		y_t_3 += a_03 * x_t_0;

		z_n[0] = y_n_0;

		A += 1;
		z_n += 1;
		x_t += 1;

		}
	
	store_t:
	z_t[0] += alpha[0]*y_t_0;
	if(km>1)
		{
		z_t[1] += alpha[0]*y_t_1;
		if(km>2)
			{
			z_t[2] += alpha[0]*y_t_2;
			if(km>3)
				{
				z_t[3] += alpha[0]*y_t_3;
				}
			}
		}

	return;

	}
#endif



#if defined(TARGET_GENERIC) || defined(TARGET_X64_AMD_BULLDOZER) || defined(TARGET_ARMV7A_ARM_CORTEX_A15) || defined(TARGET_ARMV8A_ARM_CORTEX_A57)
// XXX copy and scale y_n into z_n outside the kernel !!!!!
void kernel_dsymv_l_4_lib4(int kmax, double *alpha, double *A, int sda, double *x_n, double *z_n)
	{

	kernel_dsymv_l_4_gen_lib4(kmax, alpha, 0, A, sda, x_n, z_n, 4);

	return;

	}
#endif




