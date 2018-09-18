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

#include <math.h>

#include "../../include/blasfeo_s_kernel.h"



#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
void kernel_strmm_nt_ru_8x4_lib8(int kmax, float *alpha, float *A, float *B, float *D)
	{

	const int bs = 8;

	float
		a_0, a_1, a_2, a_3,
		a_4, a_5, a_6, a_7,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	float CC[32] = {0};
#else
#if defined (_MSC_VER)
	float CC[32] __declspec(align(64)) = {0};
#else
	float CC[32] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0
	if(kmax>0)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		A += bs;
		B += bs;
		k++;
		}

	// k = 1
	if(kmax>1)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];
		b_1 = B[1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;
		CC[4+bs*1] += a_4 * b_1;
		CC[5+bs*1] += a_5 * b_1;
		CC[6+bs*1] += a_6 * b_1;
		CC[7+bs*1] += a_7 * b_1;

		A += bs;
		B += bs;
		k++;
		}

	// k = 2
	if(kmax>2)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;
		CC[4+bs*1] += a_4 * b_1;
		CC[5+bs*1] += a_5 * b_1;
		CC[6+bs*1] += a_6 * b_1;
		CC[7+bs*1] += a_7 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;
		CC[4+bs*2] += a_4 * b_2;
		CC[5+bs*2] += a_5 * b_2;
		CC[6+bs*2] += a_6 * b_2;
		CC[7+bs*2] += a_7 * b_2;

		A += bs;
		B += bs;
		k++;
		}
	
	kernel_sgemm_nt_8x4_lib8(kmax-k, alpha, A, B, alpha, CC, D);

	return;

	}
#endif



#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
void kernel_strmm_nt_ru_8x4_vs_lib8(int kmax, float *alpha, float *A, float *B, float *D, int km, int kn)
	{

	const int bs = 8;

	float
		a_0, a_1, a_2, a_3,
		a_4, a_5, a_6, a_7,
		b_0, b_1, b_2, b_3;

#if defined(TARGET_GENERIC)
	float CC[32] = {0};
#else
#if defined (_MSC_VER)
	float CC[32] __declspec(align(64)) = {0};
#else
	float CC[32] __attribute__ ((aligned (64))) = {0};
#endif
#endif

	int k;

	k = 0;

	// k = 0
	if(kmax>0)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		A += bs;
		B += bs;
		k++;
		}

	// k = 1
	if(kmax>1)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];
		b_1 = B[1];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;
		CC[4+bs*1] += a_4 * b_1;
		CC[5+bs*1] += a_5 * b_1;
		CC[6+bs*1] += a_6 * b_1;
		CC[7+bs*1] += a_7 * b_1;

		A += bs;
		B += bs;
		k++;
		}

	// k = 2
	if(kmax>2)
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		a_4 = A[4];
		a_5 = A[5];
		a_6 = A[6];
		a_7 = A[7];

		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];

		CC[0+bs*0] += a_0 * b_0;
		CC[1+bs*0] += a_1 * b_0;
		CC[2+bs*0] += a_2 * b_0;
		CC[3+bs*0] += a_3 * b_0;
		CC[4+bs*0] += a_4 * b_0;
		CC[5+bs*0] += a_5 * b_0;
		CC[6+bs*0] += a_6 * b_0;
		CC[7+bs*0] += a_7 * b_0;

		CC[0+bs*1] += a_0 * b_1;
		CC[1+bs*1] += a_1 * b_1;
		CC[2+bs*1] += a_2 * b_1;
		CC[3+bs*1] += a_3 * b_1;
		CC[4+bs*1] += a_4 * b_1;
		CC[5+bs*1] += a_5 * b_1;
		CC[6+bs*1] += a_6 * b_1;
		CC[7+bs*1] += a_7 * b_1;

		CC[0+bs*2] += a_0 * b_2;
		CC[1+bs*2] += a_1 * b_2;
		CC[2+bs*2] += a_2 * b_2;
		CC[3+bs*2] += a_3 * b_2;
		CC[4+bs*2] += a_4 * b_2;
		CC[5+bs*2] += a_5 * b_2;
		CC[6+bs*2] += a_6 * b_2;
		CC[7+bs*2] += a_7 * b_2;

		A += bs;
		B += bs;
		k++;
		}
	
	kernel_sgemm_nt_8x4_lib8(kmax-k, alpha, A, B, alpha, CC, CC);

	if(km>=8)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];
		D[4+bs*0] = CC[4+bs*0];
		D[5+bs*0] = CC[5+bs*0];
		D[6+bs*0] = CC[6+bs*0];
		D[7+bs*0] = CC[7+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];
		D[4+bs*1] = CC[4+bs*1];
		D[5+bs*1] = CC[5+bs*1];
		D[6+bs*1] = CC[6+bs*1];
		D[7+bs*1] = CC[7+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];
		D[4+bs*2] = CC[4+bs*2];
		D[5+bs*2] = CC[5+bs*2];
		D[6+bs*2] = CC[6+bs*2];
		D[7+bs*2] = CC[7+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		D[4+bs*3] = CC[4+bs*3];
		D[5+bs*3] = CC[5+bs*3];
		D[6+bs*3] = CC[6+bs*3];
		D[7+bs*3] = CC[7+bs*3];
		}
	else if(km>=7)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];
		D[4+bs*0] = CC[4+bs*0];
		D[5+bs*0] = CC[5+bs*0];
		D[6+bs*0] = CC[6+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];
		D[4+bs*1] = CC[4+bs*1];
		D[5+bs*1] = CC[5+bs*1];
		D[6+bs*1] = CC[6+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];
		D[4+bs*2] = CC[4+bs*2];
		D[5+bs*2] = CC[5+bs*2];
		D[6+bs*2] = CC[6+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		D[4+bs*3] = CC[4+bs*3];
		D[5+bs*3] = CC[5+bs*3];
		D[6+bs*3] = CC[6+bs*3];
		}
	else if(km>=6)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];
		D[4+bs*0] = CC[4+bs*0];
		D[5+bs*0] = CC[5+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];
		D[4+bs*1] = CC[4+bs*1];
		D[5+bs*1] = CC[5+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];
		D[4+bs*2] = CC[4+bs*2];
		D[5+bs*2] = CC[5+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		D[4+bs*3] = CC[4+bs*3];
		D[5+bs*3] = CC[5+bs*3];
		}
	else if(km>=5)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];
		D[4+bs*0] = CC[4+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];
		D[4+bs*1] = CC[4+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];
		D[4+bs*2] = CC[4+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		D[4+bs*3] = CC[4+bs*3];
		}
	else if(km>=4)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];
		D[3+bs*0] = CC[3+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];
		D[3+bs*1] = CC[3+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];
		D[3+bs*2] = CC[3+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		D[3+bs*3] = CC[3+bs*3];
		}
	else if(km>=3)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];
		D[2+bs*0] = CC[2+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];
		D[2+bs*1] = CC[2+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];
		D[2+bs*2] = CC[2+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		D[2+bs*3] = CC[2+bs*3];
		}
	else if(km>=2)
		{
		D[0+bs*0] = CC[0+bs*0];
		D[1+bs*0] = CC[1+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];
		D[1+bs*1] = CC[1+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];
		D[1+bs*2] = CC[1+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		D[1+bs*3] = CC[1+bs*3];
		}
	else //if(km>=1)
		{
		D[0+bs*0] = CC[0+bs*0];

		if(kn==1)
			return;

		D[0+bs*1] = CC[0+bs*1];

		if(kn==2)
			return;

		D[0+bs*2] = CC[0+bs*2];

		if(kn==3)
			return;

		D[0+bs*3] = CC[0+bs*3];
		}

	return;

	}
#endif

