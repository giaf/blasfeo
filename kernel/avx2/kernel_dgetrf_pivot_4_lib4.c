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

#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include "../../include/blasfeo_d_aux.h"



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_4_lib4(int m, double *pA, int sda, double *inv_diag_A, int* ipiv)
	{

	const int bs = 4;

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
	
	int B_pref = bs*sda;
	

	// first column

	// find pivot
	pB = &pA[0+bs*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0] );
//		__builtin_prefetch( pB+2*B_pref );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0] );
//		__builtin_prefetch( pB+2*B_pref );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_load_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
			drowsw_lib(4, pA+0, pA+ipiv[0]/bs*bs*sda+ipiv[0]%bs);

		inv = _mm_loaddup_pd( &pA[0+bs*0] );
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
	idx = _mm256_set_pd( 2.2, 1.2, 0.2, -0.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	c_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x1 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_0 );
	d_0 = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( d_0, c_0, 0x1 );
	_mm256_store_pd( &pA[0+bs*0], a_0 );
	_mm256_store_pd( &pA[0+bs*1], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
			drowsw_lib(4, pA+1, pA+ipiv[1]/bs*bs*sda+ipiv[1]%bs);

		inv = _mm_loaddup_pd( &pA[1+bs*1] );
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
	idx = _mm256_set_pd( 1.2, 0.2, -0.8, -1.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	c_0 = _mm256_load_pd( &pA[0+bs*2] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( tmp, c_0, 0x1 );
	a_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x3 );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( tmp, c_0, 0x3 );
	_mm256_store_pd( &pA[0+bs*1], a_0 );
	_mm256_store_pd( &pA[0+bs*2], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
			drowsw_lib(4, pA+2, pA+ipiv[2]/bs*bs*sda+ipiv[2]%bs);

		inv = _mm_loaddup_pd( &pA[2+bs*2] );
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
	idx = _mm256_set_pd( 0.2, -0.8, -1.8, -2.8 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	c_0 = _mm256_load_pd( &pA[0+bs*3] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( tmp, c_0, 0x1 );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	a_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( tmp, c_0, 0x3 );
	a_0 = _mm256_load_pd( &pA[0+bs*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_2 = _mm256_permute2f128_pd( c_0, c_0, 0x11 );
	a_0 = _mm256_blend_pd( tmp, a_0, 0x7 );
	b_2 = _mm256_permute_pd( b_2, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_2 );
	tmp = _mm256_sub_pd( c_0, tmp );
	c_0 = _mm256_blend_pd( tmp, c_0, 0x7 );
	_mm256_store_pd( &pA[0+bs*2], a_0 );
	_mm256_store_pd( &pA[0+bs*3], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_2 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
			drowsw_lib(4, pA+3, pA+ipiv[3]/bs*bs*sda+ipiv[3]%bs);

		inv = _mm_loaddup_pd( &pA[3+bs*3] );
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
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
//		__builtin_prefetch( pB+2*B_pref+8 );
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
//		__builtin_prefetch( pB+2*B_pref+8 );
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
//		pB += B_pref;
		}

	return;

	}

	

void kernel_dgetrf_pivot_4_vs_lib4(int m, int n, double *pA, int sda, double *inv_diag_A, int* ipiv)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

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
	
	int B_pref = bs*sda;
	

	// first column

	// find pivot
	pB = &pA[0+bs*0];
	idx = lft; // _mm256_set_pd( 3.2, 2.2, 1.2, 0.2 );
	max = _mm256_setzero_pd();
	imx = _mm256_setzero_pd();
	k = 0;
	for( ; k<m-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0] );
//		__builtin_prefetch( pB+2*B_pref );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0] );
//		__builtin_prefetch( pB+2*B_pref );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for( ; k<m-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0] );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<m)
		{
		dlft = m-k;
		msk = _mm256_broadcast_sd( &dlft );
		a_0 = _mm256_load_pd( &pB[0] );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_blendv_pd( a_0, sgn, msk );
		a_0 = _mm256_andnot_pd( sgn, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
			drowsw_lib(4, pA+0, pA+ipiv[0]/bs*bs*sda+ipiv[0]%bs);

		inv = _mm_loaddup_pd( &pA[0+bs*0] );
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
		a_0 = _mm256_load_pd( &pA[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x1 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_store_pd( &pA[0+bs*0], a_0 );
		pB = pA + B_pref;
		k = 0;
		for(; k<ma-7; k+=8)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*0] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*0], a_0 );
			pB += B_pref;
			a_0 = _mm256_load_pd( &pB[0+bs*0] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*0], a_0 );
			pB += B_pref;
			}
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*0] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*0], a_0 );
			pB += B_pref;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_load_pd( &pB[0+bs*0] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_store_pd( &pB[0+bs*0], a_0 );
	//		pB += B_pref;
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
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	c_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x1 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_0 );
	d_0 = _mm256_sub_pd( c_0, tmp );
	d_0 = _mm256_blend_pd( d_0, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( d_0, c_0, msk );
	_mm256_store_pd( &pA[0+bs*0], a_0 );
	_mm256_store_pd( &pA[0+bs*1], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x1 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk );
		_mm256_store_pd( &pB[0+bs*0], a_0 );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
				drowsw_lib(4, pA+1, pA+ipiv[1]/bs*bs*sda+ipiv[1]%bs);

			inv = _mm_loaddup_pd( &pA[1+bs*1] );
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
		a_0 = _mm256_load_pd( &pA[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x3 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_store_pd( &pA[0+bs*1], a_0 );
		pB = pA + B_pref;
		k = 0;
		for(; k<ma-7; k+=8)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*1] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*1], a_0 );
			pB += B_pref;
			a_0 = _mm256_load_pd( &pB[0+bs*1] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*1], a_0 );
			pB += B_pref;
			}
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*1] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*1], a_0 );
			pB += B_pref;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_load_pd( &pB[0+bs*1] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_store_pd( &pB[0+bs*1], a_0 );
	//		pB += B_pref;
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
	c_0 = _mm256_load_pd( &pA[0+bs*2] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	a_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x3 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x3 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	_mm256_store_pd( &pA[0+bs*1], a_0 );
	_mm256_store_pd( &pA[0+bs*2], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x3 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_store_pd( &pB[0+bs*1], a_0 );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
				drowsw_lib(4, pA+2, pA+ipiv[2]/bs*bs*sda+ipiv[2]%bs);

			inv = _mm_loaddup_pd( &pA[2+bs*2] );
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
		a_0 = _mm256_load_pd( &pA[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_blend_pd( tmp, a_0, 0x7 );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		_mm256_store_pd( &pA[0+bs*2], a_0 );
		pB = pA + B_pref;
		k = 0;
		for(; k<ma-7; k+=8)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*2] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*2], a_0 );
			pB += B_pref;
			a_0 = _mm256_load_pd( &pB[0+bs*2] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*2], a_0 );
			pB += B_pref;
			}
		for(; k<ma-3; k+=4)
			{
			a_0 = _mm256_load_pd( &pB[0+bs*2] );
			a_0 = _mm256_mul_pd( a_0, scl );
			_mm256_store_pd( &pB[0+bs*2], a_0 );
			pB += B_pref;
			}
		if(k<ma)
			{
			dlft = ma-k;
			msk = _mm256_broadcast_sd( &dlft );
			msk = _mm256_cmp_pd( lft, msk, 14 ); // >
			a_0 = _mm256_load_pd( &pB[0+bs*2] );
			tmp = _mm256_mul_pd( a_0, scl );
			a_0 = _mm256_blendv_pd( tmp, a_0, msk );
			_mm256_store_pd( &pB[0+bs*2], a_0 );
	//		pB += B_pref;
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
	c_0 = _mm256_load_pd( &pA[0+bs*3] );
	b_0 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_0 = _mm256_permute_pd( b_0, 0x0 );
	a_0 = _mm256_load_pd( &pA[0+bs*0] );
	tmp = _mm256_mul_pd( a_0, b_0 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x1 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	b_1 = _mm256_permute2f128_pd( c_0, c_0, 0x00 );
	b_1 = _mm256_permute_pd( b_1, 0xf );
	a_0 = _mm256_load_pd( &pA[0+bs*1] );
	tmp = _mm256_mul_pd( a_0, b_1 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x3 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	a_0 = _mm256_load_pd( &pA[0+bs*2] );
	tmp = _mm256_mul_pd( a_0, scl );
	b_2 = _mm256_permute2f128_pd( c_0, c_0, 0x11 );
	tmp = _mm256_blend_pd( tmp, a_0, 0x7 );
	a_0 = _mm256_blendv_pd( tmp, a_0, msk );
	b_2 = _mm256_permute_pd( b_2, 0x0 );
	tmp = _mm256_mul_pd( a_0, b_2 );
	tmp = _mm256_sub_pd( c_0, tmp );
	tmp = _mm256_blend_pd( tmp, c_0, 0x7 );
	c_0 = _mm256_blendv_pd( tmp, c_0, msk );
	_mm256_store_pd( &pA[0+bs*2], a_0 );
	_mm256_store_pd( &pA[0+bs*3], c_0 );
	c_0 = _mm256_blend_pd( c_0, sgn, 0x7 );
	c_0 = _mm256_blendv_pd( c_0, sgn, msk );
	c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
	msk = _mm256_cmp_pd( c_0, max, 14 ); // >
	max = _mm256_blendv_pd( max, c_0, msk );
	imx = _mm256_blendv_pd( imx, idx, msk );
	idx = _mm256_add_pd( idx, vna );
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
//		__builtin_prefetch( pB+2*B_pref );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
//		__builtin_prefetch( pB+2*B_pref+8 );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_mul_pd( a_0, scl );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( c_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, scl );
		a_0 = _mm256_blendv_pd( tmp, a_0, msk );
		tmp = _mm256_mul_pd( a_0, b_2 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		c_0 = _mm256_blendv_pd( d_0, c_0, msk);
		_mm256_store_pd( &pB[0+bs*2], a_0 );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		c_0 = _mm256_blendv_pd( c_0, sgn, msk );
		c_0 = _mm256_andnot_pd( sgn, c_0 ); // abs
		msk = _mm256_cmp_pd( c_0, max, 14 ); // >
		max = _mm256_blendv_pd( max, c_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
//		idx = _mm256_add_pd( idx, vna );
//		pB += B_pref;
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
				drowsw_lib(4, pA+3, pA+ipiv[3]/bs*bs*sda+ipiv[3]%bs);

			inv = _mm_loaddup_pd( &pA[3+bs*3] );
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
	pB = pA + B_pref;
	k = 0;
	for(; k<ma-7; k+=8)
		{
//		__builtin_prefetch( pB+2*B_pref+8 );
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
//		__builtin_prefetch( pB+2*B_pref+8 );
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		c_0 = _mm256_mul_pd( c_0, scl );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += B_pref;
		}
	if(k<ma)
		{
		dlft = ma-k;
		msk = _mm256_broadcast_sd( &dlft );
		msk = _mm256_cmp_pd( lft, msk, 14 ); // >
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		tmp = _mm256_mul_pd( c_0, scl );
		c_0 = _mm256_blendv_pd( tmp, c_0, msk );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
//		pB += B_pref;
		}

	return;

	}


