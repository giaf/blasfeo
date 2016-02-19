/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
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
*                                                                                                 *
**************************************************************************************************/

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX



void kernel_dgemm_nt_4x4_lib4(int kmax, double *A, double *B, int alg, int tc, double *C, int td, double *D)
	{
	
//	if(kmax<=0)
//		return;
	
	const int ldc = 4;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		a_0123, a_2323,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	__m256d
		c_0, c_1, c_2, c_3,
		d_0, d_1, d_2, d_3,
		t_0, t_1, t_2, t_3;

	__m256i 
		mask_m, mask_n;

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();

	if(kmax<=0)
		goto add;

	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	for(k=0; k<kmax-3; k+=4)
		{
		
#if 1
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
#else

// test to compute the lower triangular using 3 fmadd

/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[2] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[6] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[10] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[14] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );

#endif
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		}

	add:

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0)
			{
			t_0 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			t_1 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			t_2 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			t_3 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			d_0 = _mm256_blend_pd( t_0, t_2, 0xc );
			d_2 = _mm256_blend_pd( t_0, t_2, 0x3 );
			d_1 = _mm256_blend_pd( t_1, t_3, 0xc );
			d_3 = _mm256_blend_pd( t_1, t_3, 0x3 );

			goto store_n;
			}
		else // transposed
			{
			t_0 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			t_1 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			t_2 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			t_3 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
			d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
			d_2 = _mm256_permute2f128_pd( t_2, t_0, 0x31 );
			d_3 = _mm256_permute2f128_pd( t_3, t_1, 0x31 );

			goto store_t;
			}
		}
	else 
		{
		if(tc==0) // C
			{

			// AB + C
			t_0 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			t_1 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			t_2 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			t_3 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			c_0 = _mm256_blend_pd( t_0, t_2, 0xc );
			c_2 = _mm256_blend_pd( t_0, t_2, 0x3 );
			c_1 = _mm256_blend_pd( t_1, t_3, 0xc );
			c_3 = _mm256_blend_pd( t_1, t_3, 0x3 );

			d_0 = _mm256_load_pd( &C[0+ldc*0] );
			d_1 = _mm256_load_pd( &C[0+ldc*1] );
			d_2 = _mm256_load_pd( &C[0+ldc*2] );
			d_3 = _mm256_load_pd( &C[0+ldc*3] );
			
			if(alg==1) // AB = A*B'
				{
				d_0 = _mm256_add_pd( d_0, c_0 );
				d_1 = _mm256_add_pd( d_1, c_1 );
				d_2 = _mm256_add_pd( d_2, c_2 );
				d_3 = _mm256_add_pd( d_3, c_3 );
				}
			else // AB = - A*B'
				{
				d_0 = _mm256_sub_pd( d_0, c_0 );
				d_1 = _mm256_sub_pd( d_1, c_1 );
				d_2 = _mm256_sub_pd( d_2, c_2 );
				d_3 = _mm256_sub_pd( d_3, c_3 );
				}

			if(td==0) // t(AB + C)
				{
				goto store_n;
				}
			else // t(AB + C)
				{
				t_0 = _mm256_unpacklo_pd( d_0, d_1 );
				t_1 = _mm256_unpackhi_pd( d_0, d_1 );
				t_2 = _mm256_unpacklo_pd( d_2, d_3 );
				t_3 = _mm256_unpackhi_pd( d_2, d_3 );

				d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
				d_2 = _mm256_permute2f128_pd( t_0, t_2, 0x31 );
				d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
				d_3 = _mm256_permute2f128_pd( t_1, t_3, 0x31 );

				goto store_t;
				}

			}
		else // t(C)
			{

			t_0 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			t_1 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			t_2 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			t_3 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			c_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
			c_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
			c_2 = _mm256_permute2f128_pd( t_2, t_0, 0x31 );
			c_3 = _mm256_permute2f128_pd( t_3, t_1, 0x31 );

			d_0 = _mm256_load_pd( &C[0+ldc*0] );
			d_1 = _mm256_load_pd( &C[0+ldc*1] );
			d_2 = _mm256_load_pd( &C[0+ldc*2] );
			d_3 = _mm256_load_pd( &C[0+ldc*3] );

			if(alg==1) // AB = A*B'
				{
				d_0 = _mm256_add_pd( d_0, c_0 );
				d_1 = _mm256_add_pd( d_1, c_1 );
				d_2 = _mm256_add_pd( d_2, c_2 );
				d_3 = _mm256_add_pd( d_3, c_3 );
				}
			else // AB = - A*B'
				{
				d_0 = _mm256_sub_pd( d_0, c_0 );
				d_1 = _mm256_sub_pd( d_1, c_1 );
				d_2 = _mm256_sub_pd( d_2, c_2 );
				d_3 = _mm256_sub_pd( d_3, c_3 );
				}

			if(td==0) // t( t(AB) + C )
				{
				t_0 = _mm256_unpacklo_pd( d_0, d_1 );
				t_1 = _mm256_unpackhi_pd( d_0, d_1 );
				t_2 = _mm256_unpacklo_pd( d_2, d_3 );
				t_3 = _mm256_unpackhi_pd( d_2, d_3 );

				d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
				d_2 = _mm256_permute2f128_pd( t_0, t_2, 0x31 );
				d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
				d_3 = _mm256_permute2f128_pd( t_1, t_3, 0x31 );

				goto store_n;
				}
			else // t(AB) + C
				{
				goto store_t;
				}

			}
		}

	// store (4) x (4)
	store_n:
	_mm256_store_pd( &D[0+ldc*0], d_0 );
	_mm256_store_pd( &D[0+ldc*1], d_1 );
	_mm256_store_pd( &D[0+ldc*2], d_2 );
	_mm256_store_pd( &D[0+ldc*3], d_3 );
	return;

	store_t:
	_mm256_store_pd( &D[0+ldc*0], d_0 );
	_mm256_store_pd( &D[0+ldc*1], d_1 );
	_mm256_store_pd( &D[0+ldc*2], d_2 );
	_mm256_store_pd( &D[0+ldc*3], d_3 );
	return;

	}



void kernel_dgemm_nt_4x4_vs_lib4(int kmax, double *A, double *B, int alg, int tc, double *C, int td, double *D, int km, int kn)
	{
	
//	if(kmax<=0)
//		return;
	
	const int ldc = 4;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		a_0123, a_2323,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	__m256d
		c_0, c_1, c_2, c_3,
		d_0, d_1, d_2, d_3,
		t_0, t_1, t_2, t_3;

	__m256i 
		mask_m, mask_n;

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();

	if(kmax<=0)
		goto add;

	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	for(k=0; k<kmax-3; k+=4)
		{
		
#if 1
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
#else

// test to compute the lower triangular using 3 fmadd

/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[2] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[6] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[10] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		a_2323        = _mm256_permute2f128_pd( a_0123, a_0123, 0x9 );
		//a_2323        = _mm256_broadcast_pd( (__m128d *) &A[14] );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_0123, b_1032, 0x0 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_2323, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );

#endif
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		}

	add:

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0)
			{
			t_0 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			t_1 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			t_2 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			t_3 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			d_0 = _mm256_blend_pd( t_0, t_2, 0xc );
			d_2 = _mm256_blend_pd( t_0, t_2, 0x3 );
			d_1 = _mm256_blend_pd( t_1, t_3, 0xc );
			d_3 = _mm256_blend_pd( t_1, t_3, 0x3 );

			goto store_n;
			}
		else // transposed
			{
			t_0 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			t_1 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			t_2 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			t_3 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
			d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
			d_2 = _mm256_permute2f128_pd( t_2, t_0, 0x31 );
			d_3 = _mm256_permute2f128_pd( t_3, t_1, 0x31 );

			goto store_t;
			}
		}
	else 
		{
		if(tc==0) // C
			{

			// AB + C
			t_0 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			t_1 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			t_2 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			t_3 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			c_0 = _mm256_blend_pd( t_0, t_2, 0xc );
			c_2 = _mm256_blend_pd( t_0, t_2, 0x3 );
			c_1 = _mm256_blend_pd( t_1, t_3, 0xc );
			c_3 = _mm256_blend_pd( t_1, t_3, 0x3 );

			d_0 = _mm256_load_pd( &C[0+ldc*0] );
			d_1 = _mm256_load_pd( &C[0+ldc*1] );
			d_2 = _mm256_load_pd( &C[0+ldc*2] );
			d_3 = _mm256_load_pd( &C[0+ldc*3] );
			
			if(alg==1) // AB = A*B'
				{
				d_0 = _mm256_add_pd( d_0, c_0 );
				d_1 = _mm256_add_pd( d_1, c_1 );
				d_2 = _mm256_add_pd( d_2, c_2 );
				d_3 = _mm256_add_pd( d_3, c_3 );
				}
			else // AB = - A*B'
				{
				d_0 = _mm256_sub_pd( d_0, c_0 );
				d_1 = _mm256_sub_pd( d_1, c_1 );
				d_2 = _mm256_sub_pd( d_2, c_2 );
				d_3 = _mm256_sub_pd( d_3, c_3 );
				}

			if(td==0) // t(AB + C)
				{
				goto store_n;
				}
			else // t(AB + C)
				{
				t_0 = _mm256_unpacklo_pd( d_0, d_1 );
				t_1 = _mm256_unpackhi_pd( d_0, d_1 );
				t_2 = _mm256_unpacklo_pd( d_2, d_3 );
				t_3 = _mm256_unpackhi_pd( d_2, d_3 );

				d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
				d_2 = _mm256_permute2f128_pd( t_0, t_2, 0x31 );
				d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
				d_3 = _mm256_permute2f128_pd( t_1, t_3, 0x31 );

				goto store_t;
				}

			}
		else // t(C)
			{

			t_0 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			t_1 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			t_2 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			t_3 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			c_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
			c_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
			c_2 = _mm256_permute2f128_pd( t_2, t_0, 0x31 );
			c_3 = _mm256_permute2f128_pd( t_3, t_1, 0x31 );

			d_0 = _mm256_load_pd( &C[0+ldc*0] );
			d_1 = _mm256_load_pd( &C[0+ldc*1] );
			d_2 = _mm256_load_pd( &C[0+ldc*2] );
			d_3 = _mm256_load_pd( &C[0+ldc*3] );

			if(alg==1) // AB = A*B'
				{
				d_0 = _mm256_add_pd( d_0, c_0 );
				d_1 = _mm256_add_pd( d_1, c_1 );
				d_2 = _mm256_add_pd( d_2, c_2 );
				d_3 = _mm256_add_pd( d_3, c_3 );
				}
			else // AB = - A*B'
				{
				d_0 = _mm256_sub_pd( d_0, c_0 );
				d_1 = _mm256_sub_pd( d_1, c_1 );
				d_2 = _mm256_sub_pd( d_2, c_2 );
				d_3 = _mm256_sub_pd( d_3, c_3 );
				}

			if(td==0) // t( t(AB) + C )
				{
				t_0 = _mm256_unpacklo_pd( d_0, d_1 );
				t_1 = _mm256_unpackhi_pd( d_0, d_1 );
				t_2 = _mm256_unpacklo_pd( d_2, d_3 );
				t_3 = _mm256_unpackhi_pd( d_2, d_3 );

				d_0 = _mm256_permute2f128_pd( t_0, t_2, 0x20 );
				d_2 = _mm256_permute2f128_pd( t_0, t_2, 0x31 );
				d_1 = _mm256_permute2f128_pd( t_1, t_3, 0x20 );
				d_3 = _mm256_permute2f128_pd( t_1, t_3, 0x31 );

				goto store_n;
				}
			else // t(AB) + C
				{
				goto store_t;
				}

			}
		}

	// store (1 - 4) x (3 - 4)
	store_n:
	d_temp = km - 0.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );
	_mm256_maskstore_pd( &D[0+ldc*0], mask_m, d_0 );
	_mm256_maskstore_pd( &D[0+ldc*1], mask_m, d_1 );
	_mm256_maskstore_pd( &D[0+ldc*2], mask_m, d_2 );

	if(kn>=4)
		{
		_mm256_maskstore_pd( &D[0+ldc*3], mask_m, d_3 );
		}
	return;

	store_t:
	if(kn==3)
		mask_n = _mm256_set_epi64x( 1, -1, -1, -1 );
	else // kn>=4
		mask_n = _mm256_set_epi64x( -1, -1, -1, -1 );

	if(km>=4)
		{
		_mm256_maskstore_pd( &D[0+ldc*0], mask_n, d_0 );
		_mm256_maskstore_pd( &D[0+ldc*1], mask_n, d_1 );
		_mm256_maskstore_pd( &D[0+ldc*2], mask_n, d_2 );
		_mm256_maskstore_pd( &D[0+ldc*3], mask_n, d_3 );
		}
	else
		{
		_mm256_maskstore_pd( &D[0+ldc*0], mask_n, d_0 );
		if(km>=2)
			{
			_mm256_maskstore_pd( &D[0+ldc*1], mask_n, d_1 );
			if(km>2)
				{
				_mm256_maskstore_pd( &D[0+ldc*2], mask_n, d_2 );
				}
			}
		}
	return;

	}



