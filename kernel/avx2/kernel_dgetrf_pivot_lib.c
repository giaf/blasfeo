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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include "../../include/blasfeo_common.h"
#include "../../include/blasfeo_d_aux.h"



// swap two rows
void kernel_drowsw_lib(int kmax, double *pA, int lda, double *pC, int ldc)
	{

	int ii;
	double tmp;

	for(ii=0; ii<kmax-3; ii+=4)
		{
		tmp = pA[0+lda*0];
		pA[0+lda*0] = pC[0+ldc*0];
		pC[0+ldc*0] = tmp;
		tmp = pA[0+lda*1];
		pA[0+lda*1] = pC[0+ldc*1];
		pC[0+ldc*1] = tmp;
		tmp = pA[0+lda*2];
		pA[0+lda*2] = pC[0+ldc*2];
		pC[0+ldc*2] = tmp;
		tmp = pA[0+lda*3];
		pA[0+lda*3] = pC[0+ldc*3];
		pC[0+ldc*3] = tmp;
		pA += 4*lda;
		pC += 4*ldc;
		}
	for( ; ii<kmax; ii++)
		{
		tmp = pA[0+lda*0];
		pA[0+lda*0] = pC[0+ldc*0];
		pC[0+ldc*0] = tmp;
		pA += 1*lda;
		pC += 1*ldc;
		}

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_12_lib(int m, double *pA, int lda, double *inv_diag_A, int* ipiv)
	{

	// assume m>=4
	int ma = m-4;

	__m128d
		max0, max1, msk0, imx0, imx1,
		inv;
	
		
	__m256d
		lft, msk,
		sgn, vna, max, imx, idx,
		ones,
		tmp,
		a_0, a_i,
		u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11,
		b_0, b_1, b_2,
		scl,
		c_0,
		d_0;
	
	double
		dlft;

	sgn = _mm256_set_pd( -0.0, -0.0, -0.0, -0.0 );
	vna = _mm256_set_pd( 4.0, 4.0, 4.0, 4.0 );
	lft  = _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double
		tmp0;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	int
		ia0, ia1, ia2;
	

	// first column

	// find pivot
	pB = &pA[0+lda*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_loadu_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// pivot & compute scaling
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			{
			kernel_drowsw_lib(12, pA+0, lda, pA+ipiv[0], lda);
			}

		inv = _mm_loaddup_pd( &pA[0+lda*0] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[0], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[0] = 0.0;
		}


	// second column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_1 = _mm256_broadcast_sd( &pA[0+lda*1] );
	u_2 = _mm256_broadcast_sd( &pA[0+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[0+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[0+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[0+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[0+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[0+lda*7] );
	u_8 = _mm256_broadcast_sd( &pA[0+lda*8] );
	u_9 = _mm256_broadcast_sd( &pA[0+lda*9] );
	u_10 = _mm256_broadcast_sd( &pA[0+lda*10] );
	u_11 = _mm256_broadcast_sd( &pA[0+lda*11] );
	// col 0
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pA[0+lda*0], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 1
	c_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
	_mm256_storeu_pd( &pA[0+lda*1], c_0 );
	// col 2
	a_i = _mm256_loadu_pd( &pA[0+lda*2] );
	a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
	_mm256_storeu_pd( &pA[0+lda*2], a_i );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pA[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pA[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pA[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pA[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pA[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pA[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pA[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pA[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[1] = idamax+1;
	if(tmp0!=0)
		{
		if(ipiv[1]!=1)
			{
			kernel_drowsw_lib(12, pA+1, lda, pA+ipiv[1], lda);
			}

		inv = _mm_loaddup_pd( &pA[1+lda*1] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[1], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[1] = 0.0;
		}



	// third column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_2 = _mm256_broadcast_sd( &pA[1+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[1+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[1+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[1+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[1+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[1+lda*7] );
	u_8 = _mm256_broadcast_sd( &pA[1+lda*8] );
	u_9 = _mm256_broadcast_sd( &pA[1+lda*9] );
	u_10 = _mm256_broadcast_sd( &pA[1+lda*10] );
	u_11 = _mm256_broadcast_sd( &pA[1+lda*11] );
	// col 1
	a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pA[0+lda*1], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 2
	c_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
	_mm256_storeu_pd( &pA[0+lda*2], c_0 );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pA[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pA[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pA[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pA[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pA[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pA[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pA[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pA[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[2] = idamax+2;
	if(tmp0!=0)
		{
		if(ipiv[2]!=2)
			{
			kernel_drowsw_lib(12, pA+2, lda, pA+ipiv[2], lda);
			}

		inv = _mm_loaddup_pd( &pA[2+lda*2] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[2], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[2] = 0.0;
		}



	// fourth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_3 = _mm256_broadcast_sd( &pA[2+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[2+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[2+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[2+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[2+lda*7] );
	u_8 = _mm256_broadcast_sd( &pA[2+lda*8] );
	u_9 = _mm256_broadcast_sd( &pA[2+lda*9] );
	u_10 = _mm256_broadcast_sd( &pA[2+lda*10] );
	u_11 = _mm256_broadcast_sd( &pA[2+lda*11] );
	// col 2
	a_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pA[0+lda*2], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 3
	c_0 = _mm256_loadu_pd( &pA[0+lda*3] );
	c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
	_mm256_storeu_pd( &pA[0+lda*3], c_0 );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pA[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pA[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pA[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pA[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pA[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pA[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pA[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pA[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[3] = idamax+3;
	if(tmp0!=0)
		{
		if(ipiv[3]!=3)
			{
			kernel_drowsw_lib(12, pA+3, lda, pA+ipiv[3], lda);
			}

		inv = _mm_loaddup_pd( &pA[3+lda*3] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[3], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[3] = 0.0;
		}
	

	// fifth column

	// scale & correct & find pivot
	// prep
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_4 = _mm256_broadcast_sd( &pA[3+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[3+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[3+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[3+lda*7] );
	u_8 = _mm256_broadcast_sd( &pA[3+lda*8] );
	u_9 = _mm256_broadcast_sd( &pA[3+lda*9] );
	u_10 = _mm256_broadcast_sd( &pA[3+lda*10] );
	u_11 = _mm256_broadcast_sd( &pA[3+lda*11] );
	pB = pA + 4; // XXX
	// col 3
	a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
	a_0 = _mm256_mul_pd( a_0, scl );
//	tmp = _mm256_mul_pd( a_0, scl );
//	a_0 = _mm256_blend_pd( tmp, a_0, 0xf );
	_mm256_storeu_pd( &pB[0+lda*3], a_0 );
//	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0xf );
	// col 4
	c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
	c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
	_mm256_storeu_pd( &pB[0+lda*4], c_0 );
	// col 5
	a_i = _mm256_loadu_pd( &pB[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pB[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pB[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pB[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pB[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pB[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pB[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pB[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
//	c_0 = _mm256_blend_pd( c_0, sgn, 0xf );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 3
		a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*3], a_0 );
		// col 4
		c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
		_mm256_storeu_pd( &pB[0+lda*4], c_0 );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 3
		a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*3], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 4
		c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
		_mm256_storeu_pd( &pB[0+lda*4], c_0 );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[4] = idamax+4;
	if(tmp0!=0)
		{
		if(ipiv[4]!=4)
			{
			kernel_drowsw_lib(12, pA+4, lda, pA+ipiv[4], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+0+lda*4] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[4], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[4] = 0.0;
		}
	

	// sixth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_5 = _mm256_broadcast_sd( &pB[0+lda*5] );
	u_6 = _mm256_broadcast_sd( &pB[0+lda*6] );
	u_7 = _mm256_broadcast_sd( &pB[0+lda*7] );
	u_8 = _mm256_broadcast_sd( &pB[0+lda*8] );
	u_9 = _mm256_broadcast_sd( &pB[0+lda*9] );
	u_10 = _mm256_broadcast_sd( &pB[0+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[0+lda*11] );
	// col 4
	a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pB[0+lda*4], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 5
	c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
	c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
	_mm256_storeu_pd( &pB[0+lda*5], c_0 );
	// col 6
	a_i = _mm256_loadu_pd( &pB[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pB[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pB[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pB[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pB[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pB[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 4
		a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*4], a_0 );
		// col 5
		c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
		_mm256_storeu_pd( &pB[0+lda*5], c_0 );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 4
		a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*4], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 5
		c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
		_mm256_storeu_pd( &pB[0+lda*5], c_0 );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[5] = idamax+5;
	if(tmp0!=0)
		{
		if(ipiv[5]!=5)
			{
			kernel_drowsw_lib(12, pA+5, lda, pA+ipiv[5], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+1+lda*5] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[5], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[5] = 0.0;
		}
	

	// seventh column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_6 = _mm256_broadcast_sd( &pB[1+lda*6] );
	u_7 = _mm256_broadcast_sd( &pB[1+lda*7] );
	u_8 = _mm256_broadcast_sd( &pB[1+lda*8] );
	u_9 = _mm256_broadcast_sd( &pB[1+lda*9] );
	u_10 = _mm256_broadcast_sd( &pB[1+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[1+lda*11] );
	// col 5
	a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pB[0+lda*5], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 6
	c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
	c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
	_mm256_storeu_pd( &pB[0+lda*6], c_0 );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// col 8
	a_i = _mm256_loadu_pd( &pB[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pB[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pB[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pB[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 5
		a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*5], a_0 );
		// col 6
		c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
		_mm256_storeu_pd( &pB[0+lda*6], c_0 );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 5
		a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*5], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 6
		c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
		_mm256_storeu_pd( &pB[0+lda*6], c_0 );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[6] = idamax+6;
	if(tmp0!=0)
		{
		if(ipiv[6]!=6)
			{
			kernel_drowsw_lib(12, pA+6, lda, pA+ipiv[6], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+2+lda*6] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[6], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[6] = 0.0;
		}


	// eight column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_7 = _mm256_broadcast_sd( &pB[2+lda*7] );
	u_8 = _mm256_broadcast_sd( &pB[2+lda*8] );
	u_9 = _mm256_broadcast_sd( &pB[2+lda*9] );
	u_10 = _mm256_broadcast_sd( &pB[2+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[2+lda*11] );
	// col 6
	a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pB[0+lda*6], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 7
	c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
	c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
	_mm256_storeu_pd( &pB[0+lda*7], c_0 );
	// col 8
	a_i = _mm256_loadu_pd( &pB[0+lda*8] );
	a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
	_mm256_storeu_pd( &pB[0+lda*8], a_i );
	// col 9
	a_i = _mm256_loadu_pd( &pB[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pB[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 6
		a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*6], a_0 );
		// col 7
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 6
		a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*6], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 7
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
		// col 8
		a_i = _mm256_loadu_pd( &pB[0+lda*8] );
		a_i = _mm256_fnmadd_pd( a_0, u_8, a_i );
		_mm256_storeu_pd( &pB[0+lda*8], a_i );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[7] = idamax+7;
	if(tmp0!=0)
		{
		if(ipiv[7]!=7)
			{
			kernel_drowsw_lib(12, pA+7, lda, pA+ipiv[7], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+3+lda*7] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[7], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[7] = 0.0;
		}


	// ninth column

	// scale & correct & find pivot
	// prep
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_8 = _mm256_broadcast_sd( &pB[3+lda*8] );
	u_9 = _mm256_broadcast_sd( &pB[3+lda*9] );
	u_10 = _mm256_broadcast_sd( &pB[3+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[3+lda*11] );
	pB += 4; // XXX
	// col 7
	a_0 = _mm256_loadu_pd( &pB[0+lda*7] );
	a_0 = _mm256_mul_pd( a_0, scl );
//	tmp = _mm256_mul_pd( a_0, scl );
//	a_0 = _mm256_blend_pd( tmp, a_0, 0xf );
	_mm256_storeu_pd( &pB[0+lda*7], a_0 );
//	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0xf );
	// col 8
	c_0 = _mm256_loadu_pd( &pB[0+lda*8] );
	c_0 = _mm256_fnmadd_pd( a_0, u_8, c_0 );
	_mm256_storeu_pd( &pB[0+lda*8], c_0 );
	// col 9
	a_i = _mm256_loadu_pd( &pB[0+lda*9] );
	a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
	_mm256_storeu_pd( &pB[0+lda*9], a_i );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
//	c_0 = _mm256_blend_pd( c_0, sgn, 0xf );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 8;
	for(; k<ma-3; k+=4)
		{
		// col 7
		a_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*7], a_0 );
		// col 8
		c_0 = _mm256_loadu_pd( &pB[0+lda*8] );
		c_0 = _mm256_fnmadd_pd( a_0, u_8, c_0 );
		_mm256_storeu_pd( &pB[0+lda*8], c_0 );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 7
		a_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*7], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 8
		c_0 = _mm256_loadu_pd( &pB[0+lda*8] );
		c_0 = _mm256_fnmadd_pd( a_0, u_8, c_0 );
		_mm256_storeu_pd( &pB[0+lda*8], c_0 );
		// col 9
		a_i = _mm256_loadu_pd( &pB[0+lda*9] );
		a_i = _mm256_fnmadd_pd( a_0, u_9, a_i );
		_mm256_storeu_pd( &pB[0+lda*9], a_i );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[8] = idamax+8;
	if(tmp0!=0)
		{
		if(ipiv[8]!=8)
			{
			kernel_drowsw_lib(12, pA+8, lda, pA+ipiv[8], lda);
			}

		inv = _mm_loaddup_pd( &pA[2*4+lda*8] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[8], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[8] = 0.0;
		}


	// tenth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 2*4; // XXX
	u_9 = _mm256_broadcast_sd( &pB[0+lda*9] );
	u_10 = _mm256_broadcast_sd( &pB[0+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[0+lda*11] );
	// col 8
	a_0 = _mm256_loadu_pd( &pB[0+lda*8] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pB[0+lda*8], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 9
	c_0 = _mm256_loadu_pd( &pB[0+lda*9] );
	c_0 = _mm256_fnmadd_pd( a_0, u_9, c_0 );
	_mm256_storeu_pd( &pB[0+lda*9], c_0 );
	// col 10
	a_i = _mm256_loadu_pd( &pB[0+lda*10] );
	a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
	_mm256_storeu_pd( &pB[0+lda*10], a_i );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 8;
	for(; k<ma-3; k+=4)
		{
		// col 8
		a_0 = _mm256_loadu_pd( &pB[0+lda*8] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*8], a_0 );
		// col 9
		c_0 = _mm256_loadu_pd( &pB[0+lda*9] );
		c_0 = _mm256_fnmadd_pd( a_0, u_9, c_0 );
		_mm256_storeu_pd( &pB[0+lda*9], c_0 );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 8
		a_0 = _mm256_loadu_pd( &pB[0+lda*8] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*8], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 9
		c_0 = _mm256_loadu_pd( &pB[0+lda*9] );
		c_0 = _mm256_fnmadd_pd( a_0, u_9, c_0 );
		_mm256_storeu_pd( &pB[0+lda*9], c_0 );
		// col 10
		a_i = _mm256_loadu_pd( &pB[0+lda*10] );
		a_i = _mm256_fnmadd_pd( a_0, u_10, a_i );
		_mm256_storeu_pd( &pB[0+lda*10], a_i );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[9] = idamax+9;
	if(tmp0!=0)
		{
		if(ipiv[9]!=9)
			{
			kernel_drowsw_lib(12, pA+9, lda, pA+ipiv[9], lda);
			}

		inv = _mm_loaddup_pd( &pA[2*4+1+lda*9] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[9], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[9] = 0.0;
		}


	// eleventh column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 2*4; // XXX
	u_10 = _mm256_broadcast_sd( &pB[1+lda*10] );
	u_11 = _mm256_broadcast_sd( &pB[1+lda*11] );
	// col 9
	a_0 = _mm256_loadu_pd( &pB[0+lda*9] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pB[0+lda*9], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 10
	c_0 = _mm256_loadu_pd( &pB[0+lda*10] );
	c_0 = _mm256_fnmadd_pd( a_0, u_10, c_0 );
	_mm256_storeu_pd( &pB[0+lda*10], c_0 );
	// col 11
	a_i = _mm256_loadu_pd( &pB[0+lda*11] );
	a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
	_mm256_storeu_pd( &pB[0+lda*11], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 8;
	for(; k<ma-3; k+=4)
		{
		// col 9
		a_0 = _mm256_loadu_pd( &pB[0+lda*9] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*9], a_0 );
		// col 10
		c_0 = _mm256_loadu_pd( &pB[0+lda*10] );
		c_0 = _mm256_fnmadd_pd( a_0, u_10, c_0 );
		_mm256_storeu_pd( &pB[0+lda*10], c_0 );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 9
		a_0 = _mm256_loadu_pd( &pB[0+lda*9] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*9], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 10
		c_0 = _mm256_loadu_pd( &pB[0+lda*10] );
		c_0 = _mm256_fnmadd_pd( a_0, u_10, c_0 );
		_mm256_storeu_pd( &pB[0+lda*10], c_0 );
		// col 11
		a_i = _mm256_loadu_pd( &pB[0+lda*11] );
		a_i = _mm256_fnmadd_pd( a_0, u_11, a_i );
		_mm256_storeu_pd( &pB[0+lda*11], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[10] = idamax+10;
	if(tmp0!=0)
		{
		if(ipiv[10]!=10)
			{
			kernel_drowsw_lib(12, pA+10, lda, pA+ipiv[10], lda);
			}

		inv = _mm_loaddup_pd( &pA[2*4+2+lda*10] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[10], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[10] = 0.0;
		}


	// twelfth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 2*4; // XXX
	u_11 = _mm256_broadcast_sd( &pB[2+lda*11] );
	// col 10
	a_0 = _mm256_loadu_pd( &pB[0+lda*10] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pB[0+lda*10], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 11
	c_0 = _mm256_loadu_pd( &pB[0+lda*11] );
	c_0 = _mm256_fnmadd_pd( a_0, u_11, c_0 );
	_mm256_storeu_pd( &pB[0+lda*11], c_0 );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 8;
	for(; k<ma-3; k+=4)
		{
		// col 10
		a_0 = _mm256_loadu_pd( &pB[0+lda*10] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*10], a_0 );
		// col 11
		c_0 = _mm256_loadu_pd( &pB[0+lda*11] );
		c_0 = _mm256_fnmadd_pd( a_0, u_11, c_0 );
		_mm256_storeu_pd( &pB[0+lda*11], c_0 );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 10
		a_0 = _mm256_loadu_pd( &pB[0+lda*10] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*10], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 11
		c_0 = _mm256_loadu_pd( &pB[0+lda*11] );
		c_0 = _mm256_fnmadd_pd( a_0, u_11, c_0 );
		_mm256_storeu_pd( &pB[0+lda*11], c_0 );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[11] = idamax+11;
	if(tmp0!=0)
		{
		if(ipiv[11]!=11)
			{
			kernel_drowsw_lib(12, pA+11, lda, pA+ipiv[11], lda);
			}

		inv = _mm_loaddup_pd( &pA[2*4+3+lda*11] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[11], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[11] = 0.0;
		}


	// scale
	pB = pA + 3*4;
	k = 8;
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_loadu_pd( &pB[0+lda*11] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_storeu_pd( &pB[0+lda*11], c_0 );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*11] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_storeu_pd( &pB[0+lda*11], c_0 );
//		pB += 4;
		}


	return;

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_8_lib(int m, double *pA, int lda, double *inv_diag_A, int* ipiv)
	{

	// assume m>=4
	int ma = m-4;

	__m128d
		max0, max1, msk0, imx0, imx1,
		inv;
	
		
	__m256d
		lft, msk,
		sgn, vna, max, imx, idx,
		ones,
		tmp,
		a_0, a_i,
		u_1, u_2, u_3, u_4, u_5, u_6, u_7,
		b_0, b_1, b_2,
		scl,
		c_0,
		d_0;
	
	double
		dlft;

	sgn = _mm256_set_pd( -0.0, -0.0, -0.0, -0.0 );
	vna = _mm256_set_pd( 4.0, 4.0, 4.0, 4.0 );
	lft  = _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double
		tmp0;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	int
		ia0, ia1, ia2;
	

	// first column

	// find pivot
	pB = &pA[0+lda*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_loadu_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// pivot & compute scaling
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			{
			kernel_drowsw_lib(8, pA+0, lda, pA+ipiv[0], lda);
			}

		inv = _mm_loaddup_pd( &pA[0+lda*0] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[0], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[0] = 0.0;
		}


	// second column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_1 = _mm256_broadcast_sd( &pA[0+lda*1] );
	u_2 = _mm256_broadcast_sd( &pA[0+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[0+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[0+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[0+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[0+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[0+lda*7] );
	// col 0
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pA[0+lda*0], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 1
	c_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
	_mm256_storeu_pd( &pA[0+lda*1], c_0 );
	// col 2
	a_i = _mm256_loadu_pd( &pA[0+lda*2] );
	a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
	_mm256_storeu_pd( &pA[0+lda*2], a_i );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[1] = idamax+1;
	if(tmp0!=0)
		{
		if(ipiv[1]!=1)
			{
			kernel_drowsw_lib(8, pA+1, lda, pA+ipiv[1], lda);
			}

		inv = _mm_loaddup_pd( &pA[1+lda*1] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[1], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[1] = 0.0;
		}



	// third column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_2 = _mm256_broadcast_sd( &pA[1+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[1+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[1+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[1+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[1+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[1+lda*7] );
	// col 1
	a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pA[0+lda*1], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 2
	c_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
	_mm256_storeu_pd( &pA[0+lda*2], c_0 );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[2] = idamax+2;
	if(tmp0!=0)
		{
		if(ipiv[2]!=2)
			{
			kernel_drowsw_lib(8, pA+2, lda, pA+ipiv[2], lda);
			}

		inv = _mm_loaddup_pd( &pA[2+lda*2] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[2], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[2] = 0.0;
		}



	// fourth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_3 = _mm256_broadcast_sd( &pA[2+lda*3] );
	u_4 = _mm256_broadcast_sd( &pA[2+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[2+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[2+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[2+lda*7] );
	// col 2
	a_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pA[0+lda*2], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 3
	c_0 = _mm256_loadu_pd( &pA[0+lda*3] );
	c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
	_mm256_storeu_pd( &pA[0+lda*3], c_0 );
	// col 4
	a_i = _mm256_loadu_pd( &pA[0+lda*4] );
	a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
	_mm256_storeu_pd( &pA[0+lda*4], a_i );
	// col 5
	a_i = _mm256_loadu_pd( &pA[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pA[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pA[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pA[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pA[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pA[0+lda*7], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// col 4
		a_i = _mm256_loadu_pd( &pB[0+lda*4] );
		a_i = _mm256_fnmadd_pd( a_0, u_4, a_i );
		_mm256_storeu_pd( &pB[0+lda*4], a_i );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[3] = idamax+3;
	if(tmp0!=0)
		{
		if(ipiv[3]!=3)
			{
			kernel_drowsw_lib(8, pA+3, lda, pA+ipiv[3], lda);
			}

		inv = _mm_loaddup_pd( &pA[3+lda*3] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[3], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[3] = 0.0;
		}
	

	// fifth column

	// scale & correct & find pivot
	// prep
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_4 = _mm256_broadcast_sd( &pA[3+lda*4] );
	u_5 = _mm256_broadcast_sd( &pA[3+lda*5] );
	u_6 = _mm256_broadcast_sd( &pA[3+lda*6] );
	u_7 = _mm256_broadcast_sd( &pA[3+lda*7] );
	pB = pA + 4; // XXX
	// col 3
	a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
	a_0 = _mm256_mul_pd( a_0, scl );
//	tmp = _mm256_mul_pd( a_0, scl );
//	a_0 = _mm256_blend_pd( tmp, a_0, 0xf );
	_mm256_storeu_pd( &pB[0+lda*3], a_0 );
//	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0xf );
	// col 4
	c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
	c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
	_mm256_storeu_pd( &pB[0+lda*4], c_0 );
	// col 5
	a_i = _mm256_loadu_pd( &pB[0+lda*5] );
	a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
	_mm256_storeu_pd( &pB[0+lda*5], a_i );
	// col 6
	a_i = _mm256_loadu_pd( &pB[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pB[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// search pivot
//	c_0 = _mm256_blend_pd( c_0, sgn, 0xf );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 3
		a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*3], a_0 );
		// col 4
		c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
		_mm256_storeu_pd( &pB[0+lda*4], c_0 );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 3
		a_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*3], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 4
		c_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		c_0 = _mm256_fnmadd_pd( a_0, u_4, c_0 );
		_mm256_storeu_pd( &pB[0+lda*4], c_0 );
		// col 5
		a_i = _mm256_loadu_pd( &pB[0+lda*5] );
		a_i = _mm256_fnmadd_pd( a_0, u_5, a_i );
		_mm256_storeu_pd( &pB[0+lda*5], a_i );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[4] = idamax+4;
	if(tmp0!=0)
		{
		if(ipiv[4]!=4)
			{
			kernel_drowsw_lib(8, pA+4, lda, pA+ipiv[4], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+0+lda*4] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[4], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[4] = 0.0;
		}
	

	// sixth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_5 = _mm256_broadcast_sd( &pB[0+lda*5] );
	u_6 = _mm256_broadcast_sd( &pB[0+lda*6] );
	u_7 = _mm256_broadcast_sd( &pB[0+lda*7] );
	// col 4
	a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pB[0+lda*4], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 5
	c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
	c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
	_mm256_storeu_pd( &pB[0+lda*5], c_0 );
	// col 6
	a_i = _mm256_loadu_pd( &pB[0+lda*6] );
	a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
	_mm256_storeu_pd( &pB[0+lda*6], a_i );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 4
		a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*4], a_0 );
		// col 5
		c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
		_mm256_storeu_pd( &pB[0+lda*5], c_0 );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 4
		a_0 = _mm256_loadu_pd( &pB[0+lda*4] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*4], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 5
		c_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		c_0 = _mm256_fnmadd_pd( a_0, u_5, c_0 );
		_mm256_storeu_pd( &pB[0+lda*5], c_0 );
		// col 6
		a_i = _mm256_loadu_pd( &pB[0+lda*6] );
		a_i = _mm256_fnmadd_pd( a_0, u_6, a_i );
		_mm256_storeu_pd( &pB[0+lda*6], a_i );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[5] = idamax+5;
	if(tmp0!=0)
		{
		if(ipiv[5]!=5)
			{
			kernel_drowsw_lib(8, pA+5, lda, pA+ipiv[5], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+1+lda*5] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[5], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[5] = 0.0;
		}
	

	// seventh column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_6 = _mm256_broadcast_sd( &pB[1+lda*6] );
	u_7 = _mm256_broadcast_sd( &pB[1+lda*7] );
	// col 5
	a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pB[0+lda*5], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 6
	c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
	c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
	_mm256_storeu_pd( &pB[0+lda*6], c_0 );
	// col 7
	a_i = _mm256_loadu_pd( &pB[0+lda*7] );
	a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
	_mm256_storeu_pd( &pB[0+lda*7], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 5
		a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*5], a_0 );
		// col 6
		c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
		_mm256_storeu_pd( &pB[0+lda*6], c_0 );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 5
		a_0 = _mm256_loadu_pd( &pB[0+lda*5] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*5], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 6
		c_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		c_0 = _mm256_fnmadd_pd( a_0, u_6, c_0 );
		_mm256_storeu_pd( &pB[0+lda*6], c_0 );
		// col 7
		a_i = _mm256_loadu_pd( &pB[0+lda*7] );
		a_i = _mm256_fnmadd_pd( a_0, u_7, a_i );
		_mm256_storeu_pd( &pB[0+lda*7], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[6] = idamax+6;
	if(tmp0!=0)
		{
		if(ipiv[6]!=6)
			{
			kernel_drowsw_lib(8, pA+6, lda, pA+ipiv[6], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+2+lda*6] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[6], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[6] = 0.0;
		}


	// eight column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	pB = pA + 4; // XXX
	u_7 = _mm256_broadcast_sd( &pB[2+lda*7] );
	// col 6
	a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pB[0+lda*6], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 7
	c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
	c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
	_mm256_storeu_pd( &pB[0+lda*7], c_0 );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB += 4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		// col 6
		a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*6], a_0 );
		// col 7
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 6
		a_0 = _mm256_loadu_pd( &pB[0+lda*6] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*6], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 7
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		c_0 = _mm256_fnmadd_pd( a_0, u_7, c_0 );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[7] = idamax+7;
	if(tmp0!=0)
		{
		if(ipiv[7]!=7)
			{
			kernel_drowsw_lib(8, pA+7, lda, pA+ipiv[7], lda);
			}

		inv = _mm_loaddup_pd( &pA[4+3+lda*7] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[7], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[7] = 0.0;
		}


	// scale
	pB = pA + 2*4;
	k = 4;
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*7] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_storeu_pd( &pB[0+lda*7], c_0 );
//		pB += 4;
		}


	return;

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_4_lib(int m, double *pA, int lda, double *inv_diag_A, int* ipiv)
	{

	// assume m>=4
	int ma = m-4;

	__m128d
		max0, max1, msk0, imx0, imx1,
		inv;
	
		
	__m256d
		lft, msk,
		sgn, vna, max, imx, idx,
		ones,
		tmp,
		a_0, a_i,
		u_1, u_2, u_3,
		b_0, b_1, b_2,
		scl,
		c_0,
		d_0;
	
	double
		dlft;

	sgn = _mm256_set_pd( -0.0, -0.0, -0.0, -0.0 );
	vna = _mm256_set_pd( 4.0, 4.0, 4.0, 4.0 );
	lft  = _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double
		tmp0;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	int
		ia0, ia1, ia2;
	

	// first column

	// find pivot
	pB = &pA[0+lda*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_loadu_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // alda
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// pivot & compute scaling
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			{
			kernel_drowsw_lib(4, pA+0, lda, pA+ipiv[0], lda);
			}

		inv = _mm_loaddup_pd( &pA[0+lda*0] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[0], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[0] = 0.0;
		}


	// second column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_1 = _mm256_broadcast_sd( &pA[0+lda*1] );
	u_2 = _mm256_broadcast_sd( &pA[0+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[0+lda*3] );
	// col 0
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	_mm256_storeu_pd( &pA[0+lda*0], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x1 );
	// col 1
	c_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
	_mm256_storeu_pd( &pA[0+lda*1], c_0 );
	// col 2
	a_i = _mm256_loadu_pd( &pA[0+lda*2] );
	a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
	_mm256_storeu_pd( &pA[0+lda*2], a_i );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 0
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 1
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		c_0 = _mm256_fnmadd_pd( a_0, u_1, c_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		// col 2
		a_i = _mm256_loadu_pd( &pB[0+lda*2] );
		a_i = _mm256_fnmadd_pd( a_0, u_2, a_i );
		_mm256_storeu_pd( &pB[0+lda*2], a_i );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[1] = idamax+1;
	if(tmp0!=0)
		{
		if(ipiv[1]!=1)
			{
			kernel_drowsw_lib(4, pA+1, lda, pA+ipiv[1], lda);
			}

		inv = _mm_loaddup_pd( &pA[1+lda*1] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[1], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[1] = 0.0;
		}



	// third column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_2 = _mm256_broadcast_sd( &pA[1+lda*2] );
	u_3 = _mm256_broadcast_sd( &pA[1+lda*3] );
	// col 1
	a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	_mm256_storeu_pd( &pA[0+lda*1], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x3 );
	// col 2
	c_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
	_mm256_storeu_pd( &pA[0+lda*2], c_0 );
	// col 3
	a_i = _mm256_loadu_pd( &pA[0+lda*3] );
	a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
	_mm256_storeu_pd( &pA[0+lda*3], a_i );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 1
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 2
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		c_0 = _mm256_fnmadd_pd( a_0, u_2, c_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		// col 3
		a_i = _mm256_loadu_pd( &pB[0+lda*3] );
		a_i = _mm256_fnmadd_pd( a_0, u_3, a_i );
		_mm256_storeu_pd( &pB[0+lda*3], a_i );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[2] = idamax+2;
	if(tmp0!=0)
		{
		if(ipiv[2]!=2)
			{
			kernel_drowsw_lib(4, pA+2, lda, pA+ipiv[2], lda);
			}

		inv = _mm_loaddup_pd( &pA[2+lda*2] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[2], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[2] = 0.0;
		}



	// fourth column

	// scale & correct & find pivot
	// prep
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	u_3 = _mm256_broadcast_sd( &pA[2+lda*3] );
	// col 2
	a_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	_mm256_storeu_pd( &pA[0+lda*2], a_0 );
	a_0 = _mm256_blend_pd( a_0, _mm256_setzero_pd(), 0x7 );
	// col 3
	c_0 = _mm256_loadu_pd( &pA[0+lda*3] );
	c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
	_mm256_storeu_pd( &pA[0+lda*3], c_0 );
	// search pivot
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// search pivot
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		// col 2
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		a_0 = _mm256_blendv_pd( a_0, _mm256_setzero_pd(), msk );
		// col 3
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_fnmadd_pd( a_0, u_3, c_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		// search pivot
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // alda
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[3] = idamax+3;
	if(tmp0!=0)
		{
		if(ipiv[3]!=3)
			{
			kernel_drowsw_lib(4, pA+3, lda, pA+ipiv[3], lda);
			}

		inv = _mm_loaddup_pd( &pA[3+lda*3] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[3], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[3] = 0.0;
		}


	// scale
	pB = pA + 1*4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
//		pB += 4;
		}


	return;

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_4_vs_lib(int m, double *pA, int lda, double *inv_diag_A, int* ipiv, int n)
	{

	if(m<=0 || n<=0)
		return;

	// assume m>=4
	int ma = m-4;

	__m128d
		max0, max1, msk0, imx0, imx1,
		inv;
	
		
	__m256d
		lft, msk,
		sgn, vna, max, imx, idx,
		ones,
		tmp,
		a_0,
		b_0, b_1, b_2,
		scl,
		c_0,
		d_0;
	
	double
		dlft;

	sgn = _mm256_set_pd( -0.0, -0.0, -0.0, -0.0 );
	vna = _mm256_set_pd( 4.0, 4.0, 4.0, 4.0 );
	lft  = _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double
		tmp0;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	int
		ia0, ia1, ia2;

	int n4 = n<4 ? n : 4;
	

	// first column

	// find pivot
	pB = &pA[0+lda*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_loadu_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			{
			kernel_drowsw_lib(n4, pA+0, lda, pA+ipiv[0], lda);
			}

		inv = _mm_loaddup_pd( &pA[0+lda*0] );
		inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
		scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
		_mm_store_sd( &inv_diag_A[0], inv );
		}
	else
		{
		scl = ones;
		inv_diag_A[0] = 0.0;
		}
	
	if(n==1)
		{
		// scale & return
		dlft = m;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x1 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pA[0+lda*0], a_0 );
		pB = pA + 4;
		k = 0;
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_storeu_pd( &pB[0+lda*0], a_0 );
			pB += 4;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_storeu_pd( &pB[0+lda*0], a_0 );
	//		pB += 4;
			}

		return;
		}


	// second column

	// scale & correct & find pivot
	dlft = m;
	msk = _mm256_broadcast_sd( &dlft );
	msk = _mm256_cmp_pd( lft, msk, 14 ); // >
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	c_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x1 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_0 );
	d_0 = _mm256_sub_pd( c_0, tmp );
	d_0 = _mm256_blend_pd( d_0, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( d_0, c_0, msk );
	_mm256_storeu_pd( &pA[0+lda*0], a_0 );
	_mm256_storeu_pd( &pA[0+lda*1], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		c_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk );
		_mm256_storeu_pd( &pB[0+lda*0], a_0 );
		_mm256_storeu_pd( &pB[0+lda*1], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	if(m>1)
		{
		ipiv[1] = idamax+1;
		if(tmp0!=0)
			{
			if(ipiv[1]!=1)
				{
				kernel_drowsw_lib(n4, pA+1, lda, pA+ipiv[1], lda);
				}

			inv = _mm_loaddup_pd( &pA[1+lda*1] );
			inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
			scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
			_mm_store_sd( &inv_diag_A[1], inv );
			}
		else
			{
			scl = ones;
			inv_diag_A[1] = 0.0;
			}
		}

	if(n==2)
		{
		// scale & return
		dlft = m;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x3 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pA[0+lda*1], a_0 );
		pB = pA + 4;
		k = 0;
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_storeu_pd( &pB[0+lda*1], a_0 );
			pB += 4;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_storeu_pd( &pB[0+lda*1], a_0 );
	//		pB += 4;
			}

		return;
		}

	// third column

	// scale & correct & find pivot
	dlft = m;
	msk = _mm256_broadcast_sd( &dlft );
	msk = _mm256_cmp_pd( lft, msk, 14 ); // >
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	c_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x3 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x3 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	_mm256_storeu_pd( &pA[0+lda*1], a_0 );
	_mm256_storeu_pd( &pA[0+lda*2], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_storeu_pd( &pB[0+lda*1], a_0 );
		_mm256_storeu_pd( &pB[0+lda*2], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	if(m>2)
		{
		ipiv[2] = idamax+2;
		if(tmp0!=0)
			{
			if(ipiv[2]!=2)
				{
				kernel_drowsw_lib(n4, pA+2, lda, pA+ipiv[2], lda);
				}

			inv = _mm_loaddup_pd( &pA[2+lda*2] );
			inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
			scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
			_mm_store_sd( &inv_diag_A[2], inv );
			}
		else
			{
			scl = ones;
			inv_diag_A[2] = 0.0;
			}
		}

	if(n==3)
		{
		// scale & return
		dlft = m;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_loadu_pd( &pA[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x7 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_storeu_pd( &pA[0+lda*2], a_0 );
		pB = pA + 4;
		k = 0;
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_storeu_pd( &pB[0+lda*2], a_0 );
			pB += 4;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_storeu_pd( &pB[0+lda*2], a_0 );
	//		pB += 4;
			}

		return;
		}

	// fourth column

	// scale & correct & find pivot
	dlft = m;
	msk = _mm256_broadcast_sd( &dlft );
	msk = _mm256_cmp_pd( lft, msk, 14 ); // >
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	c_0 = _mm256_loadu_pd( &pA[0+lda*3] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_loadu_pd( &pA[0+lda*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	a_0 = _mm256_loadu_pd( &pA[0+lda*1] );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x3 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	a_0 = _mm256_loadu_pd( &pA[0+lda*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_2 = _mm256_permute2f128_pd( c_0, c_0, 0x11 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x7 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_2 = _mm256_permute_pd( b_2, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_2 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x7 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	_mm256_storeu_pd( &pA[0+lda*2], a_0 );
	_mm256_storeu_pd( &pA[0+lda*3], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		a_0 = _mm256_loadu_pd( &pB[0+lda*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_loadu_pd( &pB[0+lda*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_loadu_pd( &pB[0+lda*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_2 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_storeu_pd( &pB[0+lda*2], a_0 );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += 4;
		}
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );
	_mm_store_sd( &tmp0, max0 );
	idamax = _mm_cvtsd_si32( imx0 );

	// compute scaling
	if(m>3)
		{
		ipiv[3] = idamax+3;
		if(tmp0!=0)
			{
			if(ipiv[3]!=3)
				{
				kernel_drowsw_lib(n4, pA+3, lda, pA+ipiv[3], lda);
				}

			inv = _mm_loaddup_pd( &pA[3+lda*3] );
			inv = _mm_div_pd( _mm256_castpd256_pd128( ones ), inv );
			scl = _mm256_permute2f128_pd( _mm256_castpd128_pd256( inv ), _mm256_castpd128_pd256( inv ), 0x00 );
			_mm_store_sd( &inv_diag_A[3], inv );
			}
		else
			{
			scl = ones;
			inv_diag_A[3] = 0.0;
			}
		}

	// scale
	pB = pA + 4;
	k = 0;
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
		pB += 4;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_loadu_pd( &pB[0+lda*3] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_storeu_pd( &pB[0+lda*3], c_0 );
//		pB += 4;
		}

	return;

	}



