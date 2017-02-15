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

#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



#if defined(LA_HIGH_PERFORMANCE)



void daxpy_libstr(int m, double alpha, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;

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
	for( ; ii<m-7; ii+=8)
		{
		v_x0  = _mm256_loadu_pd( &x[ii+0] );
		v_x1  = _mm256_loadu_pd( &x[ii+4] );
		v_y0  = _mm256_loadu_pd( &y[ii+0] );
		v_y1  = _mm256_loadu_pd( &y[ii+4] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_y0 );
		v_y1  = _mm256_fmadd_pd( v_alpha, v_x1, v_y1 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_alpha, v_x0 );
		v_y0  = _mm256_add_pd( v_tmp, v_y0 );
		v_tmp = _mm256_mul_pd( v_alpha, v_x1 );
		v_y1  = _mm256_add_pd( v_tmp, v_y1 );
#endif
		_mm256_storeu_pd( &z[ii+0], v_y0 );
		_mm256_storeu_pd( &z[ii+4], v_y1 );
		}
	for( ; ii<m-3; ii+=4)
		{
		v_x0  = _mm256_loadu_pd( &x[ii] );
		v_y0  = _mm256_loadu_pd( &y[ii] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_y0 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_alpha, v_x0 );
		v_y0  = _mm256_add_pd( v_tmp, v_y0 );
#endif
		_mm256_storeu_pd( &z[ii], v_y0 );
		}
#else
	for( ; ii<m-3; ii+=4)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		z[ii+1] = y[ii+1] + alpha*x[ii+1];
		z[ii+2] = y[ii+2] + alpha*x[ii+2];
		z[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
#endif
	for( ; ii<m; ii++)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		}

	return;
	}



// multiply two vectors and compute dot product
double dvecmuldot_libstr(int m, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi, struct d_strvec *sz, int zi)
	{
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	int ii;
	double dot = 0.0;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m128d
		u_tmp, u_dot;
	__m256d
		v_tmp,
		v_x0, v_y0, v_z0;
	
	v_tmp = _mm256_setzero_pd();
#endif

	ii = 0;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-3; ii+=4)
		{
		v_x0 = _mm256_loadu_pd( &x[ii+0] );
		v_y0 = _mm256_loadu_pd( &y[ii+0] );
		v_z0 = _mm256_mul_pd( v_x0, v_y0 );
		_mm256_storeu_pd( &z[ii+0], v_z0 );
		v_tmp = _mm256_add_pd( v_tmp, v_z0 );
		}
#endif
	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	// dot product
	u_tmp = _mm_add_pd( _mm256_castpd256_pd128( v_tmp ), _mm256_extractf128_pd( v_tmp, 0x1 ) );
	u_tmp = _mm_hadd_pd( u_tmp, u_tmp);
	u_dot = _mm_load_sd( &dot );
	u_dot = _mm_add_sd( u_dot, u_tmp );
	_mm_store_sd( &dot, u_dot );
#endif
	return dot;
	}



// compute dot product of two vectors
double ddot_libstr(int m, struct d_strvec *sx, int xi, struct d_strvec *sy, int yi)
	{
	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	int ii;
	double dot = 0.0;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m128d
		u_dot0, u_x0, u_y0, u_tmp;
	__m256d
		v_dot0, v_dot1, v_x0, v_x1, v_y0, v_y1, v_tmp;
	
	v_dot0 = _mm256_setzero_pd();
	v_dot1 = _mm256_setzero_pd();
	u_dot0 = _mm_setzero_pd();

	ii = 0;
	for(; ii<m-7; ii+=8)
		{
		v_x0 = _mm256_loadu_pd( &x[ii+0] );
		v_x1 = _mm256_loadu_pd( &x[ii+4] );
		v_y0 = _mm256_loadu_pd( &y[ii+0] );
		v_y1 = _mm256_loadu_pd( &y[ii+4] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_dot0  = _mm256_fmadd_pd( v_x0, v_y0, v_dot0 );
		v_dot1  = _mm256_fmadd_pd( v_x1, v_y1, v_dot1 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_x0, v_y0 );
		v_dot0 = _mm256_add_pd( v_dot0, v_tmp );
		v_tmp = _mm256_mul_pd( v_x1, v_y1 );
		v_dot1 = _mm256_add_pd( v_dot1, v_tmp );
#endif
		}
	for(; ii<m-3; ii+=4)
		{
		v_x0 = _mm256_loadu_pd( &x[ii+0] );
		v_y0 = _mm256_loadu_pd( &y[ii+0] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_dot0  = _mm256_fmadd_pd( v_x0, v_y0, v_dot0 );
#else // sandy bridge
		v_tmp = _mm256_mul_pd( v_x0, v_y0 );
		v_dot0 = _mm256_add_pd( v_dot0, v_tmp );
#endif
		}
	for(; ii<m; ii++)
		{
		u_x0 = _mm_load_sd( &x[ii+0] );
		u_y0 = _mm_load_sd( &y[ii+0] );
#if defined(TARGET_X64_INTEL_HASWELL)
		u_dot0  = _mm_fmadd_sd( u_x0, u_y0, u_dot0 );
#else // sandy bridge
		u_tmp = _mm_mul_sd( u_x0, u_y0 );
		u_dot0 = _mm_add_sd( u_dot0, u_tmp );
#endif
		}
	// reduce
	v_dot0 = _mm256_add_pd( v_dot0, v_dot1 );
	u_tmp = _mm_add_pd( _mm256_castpd256_pd128( v_dot0 ), _mm256_extractf128_pd( v_dot0, 0x1 ) );
	u_tmp = _mm_hadd_pd( u_tmp, u_tmp);
	u_dot0 = _mm_add_sd( u_dot0, u_tmp );
	_mm_store_sd( &dot, u_dot0 );
#else // no haswell, no sandy bridge
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		dot += x[ii+0] * y[ii+0];
		dot += x[ii+1] * y[ii+1];
		dot += x[ii+2] * y[ii+2];
		dot += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		dot += x[ii+0] * y[ii+0];
		}
#endif // haswell, sandy bridge
	return dot;
	}



#else

#error : wrong LA choice

#endif
