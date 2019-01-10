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

#include <stdlib.h>
#include <stdio.h>


#include "../../include/blasfeo_target.h"
#include "../../include/blasfeo_common.h"
#include "../../include/blasfeo_d_aux.h"
#include "../../include/blasfeo_d_kernel.h"



#if defined(TARGET_ARMV8A_ARM_CORTEX_A53)
void kernel_dgetrf_pivot_12_lib4(int m, double *pC, int sdc, double *pd, int *ipiv)
	{

	const int ps = 4;

	int ii;

	double *dummy = NULL;

	double d1 = 1.0;
	double dm1 = -1.0;

	// fact left column
	kernel_dgetrf_pivot_8_lib4(m, pC, sdc, pd, ipiv);

	// apply pivot to right column
	for(ii=0; ii<8; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			kernel_drowsw_lib4(4, pC+ii/ps*ps*sdc+ii%ps+8*ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps+8*ps);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_8x4_lib4(0, dummy, 0, dummy, 0, &d1, pC+8*ps, sdc, pC+8*ps, sdc, pC, sdc);

	// correct rigth block
	ii = 8;
	for(; ii<m-11; ii+=12)
		{
		kernel_dgemm_nn_12x4_lib4(8, &dm1, pC+ii*sdc, sdc, 0, pC+8*ps, sdc, &d1, pC+ii*sdc+8*ps, sdc, pC+ii*sdc+8*ps, sdc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(8, &dm1, pC+ii*sdc, 0, pC+8*ps, sdc, &d1, pC+ii*sdc+8*ps, pC+ii*sdc+8*ps, m-ii, 4);
			}
		else if(m-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(8, &dm1, pC+ii*sdc, sdc, 0, pC+8*ps, sdc, &d1, pC+ii*sdc+8*ps, sdc, pC+ii*sdc+8*ps, sdc, m-ii, 4);
			}
		else //if(m-ii<=12)
			{
			kernel_dgemm_nn_12x4_vs_lib4(8, &dm1, pC+ii*sdc, sdc, 0, pC+8*ps, sdc, &d1, pC+ii*sdc+8*ps, sdc, pC+ii*sdc+8*ps, sdc, m-ii, 4);
			}
		}

	// fact right column
	kernel_dgetrf_pivot_4_lib4(m-8, pC+8*sdc+8*ps, sdc, pd+8, ipiv+8);

	for(ii=8; ii<12; ii++)
		ipiv[ii] += 8;

	// apply pivot to left column
	for(ii=8; ii<12; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			kernel_drowsw_lib4(8, pC+ii/ps*ps*sdc+ii%ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps);
			}
		}

	return;

	}
#endif



void kernel_dgetrf_pivot_8_lib4(int m, double *pC, int sdc, double *pd, int *ipiv)
	{

	const int ps = 4;

	int ii;

	double *dummy = NULL;

	double d1 = 1.0;
	double dm1 = -1.0;

	// fact left column
	kernel_dgetrf_pivot_4_lib4(m, pC, sdc, pd, ipiv);

	// apply pivot to right column
	for(ii=0; ii<4; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			kernel_drowsw_lib4(4, pC+ii/ps*ps*sdc+ii%ps+4*ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps+4*ps);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_4x4_lib4(0, dummy, dummy, 0, &d1, pC+4*ps, pC+4*ps, pC);

	// correct rigth block
	ii = 4;
	for(; ii<m-7; ii+=8)
		{
		kernel_dgemm_nn_8x4_lib4(4, &dm1, pC+ii*sdc, sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, sdc, pC+ii*sdc+4*ps, sdc);
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+ii*sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, pC+ii*sdc+4*ps, m-ii, 4);
			}
		else //if(m-ii<=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+ii*sdc, sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, sdc, pC+ii*sdc+4*ps, sdc, m-ii, 4);
			}
		}

	// fact right column
	kernel_dgetrf_pivot_4_lib4(m-4, pC+4*sdc+4*ps, sdc, pd+4, ipiv+4);

	for(ii=4; ii<8; ii++)
		ipiv[ii] += 4;

	// apply pivot to left column
	for(ii=4; ii<8; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			kernel_drowsw_lib4(4, pC+ii/ps*ps*sdc+ii%ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps);
			}
		}

	return;

	}



// m>=1 and n={5,6,7,8}
void kernel_dgetrf_pivot_8_vs_lib4(int m, double *pC, int sdc, double *pd, int *ipiv, int n)
	{

	const int ps = 4;

	int ii;

	double *dummy = NULL;

	double d1 = 1.0;
	double dm1 = -1.0;

	// saturate n to 8
	n = 8<n ? 8 : n;

	int p = m<n ? m : n;

	int n_max;

	// fact left column
	kernel_dgetrf_pivot_4_vs_lib4(m, pC, sdc, pd, ipiv, 4);

	n_max = p<4 ? p : 4;

	// apply pivot to right column
	for(ii=0; ii<n_max; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			kernel_drowsw_lib4(n-4, pC+ii/ps*ps*sdc+ii%ps+4*ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps+4*ps);
			}
		}

	// solve top right block
	kernel_dtrsm_nn_ll_one_4x4_vs_lib4(0, dummy, dummy, 0, &d1, pC+4*ps, pC+4*ps, pC, m, n-4);

	if(m>4)
		{

		// correct rigth block
		ii = 4;
#if defined(TARGET_ARMV8A_ARM_CORTEX_A53)
		for(; ii<m-8; ii+=12)
			{
			kernel_dgemm_nn_12x4_vs_lib4(4, &dm1, pC+ii*sdc, sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, sdc, pC+ii*sdc+4*ps, sdc, m-ii, n-4);
			}
		if(ii<m)
			{
			if(m-ii<=4)
				{
				kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+ii*sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, pC+ii*sdc+4*ps, m-ii, n-4);
				}
			else //if(m-ii<=8)
				{
				kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+ii*sdc, sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, sdc, pC+ii*sdc+4*ps, sdc, m-ii, n-4);
				}
			}
#else
		for(; ii<m-4; ii+=8)
			{
			kernel_dgemm_nn_8x4_vs_lib4(4, &dm1, pC+ii*sdc, sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, sdc, pC+ii*sdc+4*ps, sdc, m-ii, n-4);
			}
		if(ii<m)
			{
			kernel_dgemm_nn_4x4_vs_lib4(4, &dm1, pC+ii*sdc, 0, pC+4*ps, sdc, &d1, pC+ii*sdc+4*ps, pC+ii*sdc+4*ps, m-ii, n-4);
			}
#endif

// TODO
		// fact right column
		kernel_dgetrf_pivot_4_vs_lib4(m-4, pC+4*sdc+4*ps, sdc, pd+4, ipiv+4, n-4);

		n_max = p;

		for(ii=4; ii<n_max; ii++)
			ipiv[ii] += 4;

		// apply pivot to left column
		for(ii=4; ii<n_max; ii++)
			{
			if(ipiv[ii]!=ii)
				{
				kernel_drowsw_lib4(4, pC+ii/ps*ps*sdc+ii%ps, pC+ipiv[ii]/ps*ps*sdc+ipiv[ii]%ps);
				}
			}

		}

	return;

	}


