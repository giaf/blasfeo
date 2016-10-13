/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX
#endif

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



// y = y + alpha*x, with increments equal to 1
void daxpy_lib(int kmax, double alpha, double *x, double *y)
	{

	int ii;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		v_alpha, v_tmp,
		v_x0, v_y0,
		v_x1, v_y1;
#endif

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	v_alpha = _mm256_broadcast_sd( &alpha );
	for( ; ii<kmax-7; ii+=8)
		{
		v_x0  = _mm256_load_pd( &x[ii+0] );
		v_x1  = _mm256_load_pd( &x[ii+4] );
		v_y0  = _mm256_load_pd( &y[ii+0] );
		v_y1  = _mm256_load_pd( &y[ii+4] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_y0 );
		v_y1  = _mm256_fmadd_pd( v_alpha, v_x1, v_y1 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_alpha, v_x0 );
		v_y0  = _mm256_add_pd( v_tmp, v_y0 );
		v_tmp = _mm256_mul_pd( v_alpha, v_x1 );
		v_y1  = _mm256_add_pd( v_tmp, v_y1 );
#endif
		_mm256_store_pd( &y[ii+0], v_y0 );
		_mm256_store_pd( &y[ii+4], v_y1 );
		}
	for( ; ii<kmax-3; ii+=4)
		{
		v_x0  = _mm256_load_pd( &x[ii] );
		v_y0  = _mm256_load_pd( &y[ii] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_y0 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_alpha, v_x0 );
		v_y0  = _mm256_add_pd( v_tmp, v_y0 );
#endif
		_mm256_store_pd( &y[ii], v_y0 );
		}
#else
	for( ; ii<kmax-3; ii+=4)
		{
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		y[ii+1] = y[ii+1] + alpha*x[ii+1];
		y[ii+2] = y[ii+2] + alpha*x[ii+2];
		y[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
#endif
	for( ; ii<kmax; ii++)
		{
		y[ii+0] = y[ii+0] + alpha*x[ii+0];
		}

	return;

	}



#if defined(LA_BLASFEO)



void daxpy_libstr(int m, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi)
	{
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	daxpy_lib(m, alpha, x, y);
	return;
	}



#else



void daxpy_libstr(int m, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi)
	{
	int i1 = 1;
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	daxpy_(&m, &alpha, x, &i1, y, &i1);
	return;
	}



#endif
