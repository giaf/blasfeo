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

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX
#endif

#include "../include/blasfeo_target.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"
//#include "../include/blasfeo_d_kernel.h"



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dcopy dcopy_
#endif



void blasfeo_dcopy(int *pn, double *x, int *pincx, double *y, int *pincy)
	{

	int n = *pn;
	int incx = *pincx;
	int incy = *pincy;

	int ii;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		tmp;
#endif

	if(incx==1 & incy==1)
		{
		ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		for(; ii<n-15; ii+=16)
			{
			tmp = _mm256_loadu_pd( &x[0] );
			_mm256_storeu_pd( &y[0], tmp );
			tmp = _mm256_loadu_pd( &x[4] );
			_mm256_storeu_pd( &y[4], tmp );
			tmp = _mm256_loadu_pd( &x[8] );
			_mm256_storeu_pd( &y[8], tmp );
			tmp = _mm256_loadu_pd( &x[12] );
			_mm256_storeu_pd( &y[12], tmp );
			x += 16;
			y += 16;
			}
		for(; ii<n-3; ii+=4)
			{
			tmp = _mm256_loadu_pd( &x[0] );
			_mm256_storeu_pd( &y[0], tmp );
			x += 4;
			y += 4;
			}
#else
		for(; ii<n-3; ii+=4)
			{
			y[0] = x[0];
			y[1] = x[1];
			y[2] = x[2];
			y[3] = x[3];
			x += 4;
			y += 4;
			}
#endif
		for(; ii<n; ii++)
			{
			y[0] = x[0];
			x += 1;
			y += 1;
			}
		}
	else
		{
		ii = 0;
		for(; ii<n; ii++)
			{
			y[0] = x[0];
			x += incx;
			y += incy;
			}
		}

	return;

	}
