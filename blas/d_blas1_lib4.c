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
#include <math.h>

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX
#endif

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_kernel.h"



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_daxpy(int m, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{

	if(m<=0)
		return;

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



void blasfeo_daxpby(int m, double alpha, struct blasfeo_dvec *sx, int xi, double beta, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{

	if(m<=0)
		return;

	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;

	int ii;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		v_beta, v_alpha,
		v_tmp_x, v_tmp_y,
		v_x0, v_y0,
		v_x1, v_y1;
#endif

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	v_alpha = _mm256_broadcast_sd( &alpha );
	v_beta = _mm256_broadcast_sd( &beta );
	for( ; ii<m-7; ii+=8)
		{
		v_x0  = _mm256_loadu_pd( &x[ii+0] );
		v_x1  = _mm256_loadu_pd( &x[ii+4] );
		v_y0  = _mm256_loadu_pd( &y[ii+0] );
		v_y1  = _mm256_loadu_pd( &y[ii+4] );
#if defined(TARGET_X64_INTEL_HASWELL)
		v_tmp_y = _mm256_mul_pd( v_beta, v_y0 );
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_tmp_y );
		v_tmp_y = _mm256_mul_pd( v_beta, v_y1 );
		v_y1  = _mm256_fmadd_pd( v_alpha, v_x1, v_tmp_y );

#else // sandy bridge
		v_tmp_x = _mm256_mul_pd( v_alpha, v_x0 );
		v_tmp_y = _mm256_mul_pd( v_beta, v_y0 );
		v_y0  = _mm256_add_pd( v_tmp_x, v_tmp_y );

		v_tmp_x = _mm256_mul_pd( v_alpha, v_x1 );
		v_tmp_y = _mm256_mul_pd( v_beta, v_y1 );
		v_y1  = _mm256_add_pd( v_tmp_x, v_tmp_y );

#endif
		_mm256_storeu_pd( &z[ii+0], v_y0 );
		_mm256_storeu_pd( &z[ii+4], v_y1 );
		}
	for( ; ii<m-3; ii+=4)
		{
		v_x0  = _mm256_loadu_pd( &x[ii] );
		v_y0  = _mm256_loadu_pd( &y[ii] );

#if defined(TARGET_X64_INTEL_HASWELL)
		v_tmp_y = _mm256_mul_pd( v_beta, v_y0 );
		v_y0  = _mm256_fmadd_pd( v_alpha, v_x0, v_tmp_y );
#else // sandy bridge
		v_tmp_x = _mm256_mul_pd( v_alpha, v_x0 );
		v_tmp_y = _mm256_mul_pd( v_beta, v_y0 );
		v_y0  = _mm256_add_pd( v_tmp_x, v_tmp_y );
#endif
		_mm256_storeu_pd( &z[ii], v_y0 );
		}
#else
	for( ; ii<m-3; ii+=4)
		{
		z[ii+0] = beta*y[ii+0] + alpha*x[ii+0];
		z[ii+1] = beta*y[ii+1] + alpha*x[ii+1];
		z[ii+2] = beta*y[ii+2] + alpha*x[ii+2];
		z[ii+3] = beta*y[ii+3] + alpha*x[ii+3];
		}
#endif
	for( ; ii<m; ii++)
		{
		z[ii+0] = beta*y[ii+0] + alpha*x[ii+0];
		}

	return;
	}



// multiply two vectors
void blasfeo_dvecmulacc(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{

	if(m<=0)
		return;

	double *x = sx->pa + xi;
	double *y = sy->pa + yi;
	double *z = sz->pa + zi;
	int ii;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		v_tmp,
		v_x0, v_y0, v_z0;
#endif

	ii = 0;

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<m-3; ii+=4)
		{
		v_x0 = _mm256_loadu_pd( &x[ii+0] );
		v_y0 = _mm256_loadu_pd( &y[ii+0] );
		v_z0 = _mm256_loadu_pd( &z[ii+0] );
		v_tmp = _mm256_mul_pd( v_x0, v_y0 );
		v_z0 = _mm256_add_pd( v_z0, v_tmp );
		_mm256_storeu_pd( &z[ii+0], v_z0 );
		}
#endif
	for(; ii<m; ii++)
		{
		z[ii+0] += x[ii+0] * y[ii+0];
		}
	return;
	}



// multiply two vectors and compute dot product
double blasfeo_dvecmuldot(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, struct blasfeo_dvec *sz, int zi)
	{

	if(m<=0)
		return 0.0;

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
double blasfeo_ddot(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi)
	{

	if(m<=0)
		return 0.0;

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



void blasfeo_drotg(double a, double b, double *c, double *s)
	{
	double aa = fabs(a);
	double bb = fabs(b);
	double roe = aa>=bb ? a : b;
	double scale = aa + bb;
	double r;
	if(scale==0)
		{
		*c = 1.0;
		*s = 0.0;
		}
	else
		{
		aa = a/scale;
		bb = b/scale;
		r = scale * sqrt(aa*aa + bb*bb);
		r = r * (roe >= 0 ? 1 : -1);
		*c = a / r;
		*s = b / r;	
		}
	return;
	}



void blasfeo_dcolrot(int m, struct blasfeo_dmat *sA, int ai, int aj0, int aj1, double c, double s)
	{
	const int ps = 4;
	int sda = sA->cn;
	double *px = sA->pA + ai/ps*ps*sda + ai%ps + aj0*ps;
	double *py = sA->pA + ai/ps*ps*sda + ai%ps + aj1*ps;
	int mna = (ps-ai%ps)%ps;
	int ii;
	double d_tmp;
	ii = 0;
	if(mna>0)
		{
		for(; ii<mna; ii++)
			{
			d_tmp = c*px[0] + s*py[0];
			py[0] = c*py[0] - s*px[0];
			px[0] = d_tmp;
			px++;
			py++;
			}
		px += ps*(sda-1);
		py += ps*(sda-1);
		}
	for(; ii<m-3; ii+=4)
		{
		//
		d_tmp = c*px[0] + s*py[0];
		py[0] = c*py[0] - s*px[0];
		px[0] = d_tmp;
		//
		d_tmp = c*px[1] + s*py[1];
		py[1] = c*py[1] - s*px[1];
		px[1] = d_tmp;
		//
		d_tmp = c*px[2] + s*py[2];
		py[2] = c*py[2] - s*px[2];
		px[2] = d_tmp;
		//
		d_tmp = c*px[3] + s*py[3];
		py[3] = c*py[3] - s*px[3];
		px[3] = d_tmp;
		//
		px+=ps*sda;
		py+=ps*sda;
		}
	for(; ii<m; ii++)
		{
		//
		d_tmp = c*px[0] + s*py[0];
		py[0] = c*py[0] - s*px[0];
		px[0] = d_tmp;
		//
		px++;
		py++;
		}
	return;
	}
	


void blasfeo_drowrot(int m, struct blasfeo_dmat *sA, int ai0, int ai1, int aj, double c, double s)
	{
	const int ps = 4;
	int sda = sA->cn;
	double *px = sA->pA + ai0/ps*ps*sda + ai0%ps + aj*ps;
	double *py = sA->pA + ai1/ps*ps*sda + ai1%ps + aj*ps;
	int ii;
	double d_tmp;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		//
		d_tmp = c*px[0*ps] + s*py[0*ps];
		py[0*ps] = c*py[0*ps] - s*px[0*ps];
		px[0*ps] = d_tmp;
		//
		d_tmp = c*px[1*ps] + s*py[1*ps];
		py[1*ps] = c*py[1*ps] - s*px[1*ps];
		px[1*ps] = d_tmp;
		//
		d_tmp = c*px[2*ps] + s*py[2*ps];
		py[2*ps] = c*py[2*ps] - s*px[2*ps];
		px[2*ps] = d_tmp;
		//
		d_tmp = c*px[3*ps] + s*py[3*ps];
		py[3*ps] = c*py[3*ps] - s*px[3*ps];
		px[3*ps] = d_tmp;
		//
		px+=4*ps;
		py+=4*ps;
		}
	for(; ii<m; ii++)
		{
		//
		d_tmp = c*px[0*ps] + s*py[0*ps];
		py[0*ps] = c*py[0*ps] - s*px[0*ps];
		px[0*ps] = d_tmp;
		//
		px+=1*ps;
		py+=1*ps;
		}
	return;
	}
	


#else

#error : wrong LA choice

#endif
