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

#include <immintrin.h>  // AVX



// B is the diagonal of a matrix, beta==0.0 case
void kernel_sgemm_diag_right_4_a0_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 8;

	int k;

	__m256
		alpha0,
		mask_f,
		sign,
		a_00,
		b_00, b_11, b_22, b_33,
		d_00, d_01, d_02, d_03;
	
	__m256i
		mask_i;
	
	alpha0 = _mm256_broadcast_ss( alpha );
	
	b_00 = _mm256_broadcast_ss( &B[0] );
	b_00 = _mm256_mul_ps( b_00, alpha0 );
	b_11 = _mm256_broadcast_ss( &B[1] );
	b_11 = _mm256_mul_ps( b_11, alpha0 );
	b_22 = _mm256_broadcast_ss( &B[2] );
	b_22 = _mm256_mul_ps( b_22, alpha0 );
	b_33 = _mm256_broadcast_ss( &B[3] );
	b_33 = _mm256_mul_ps( b_33, alpha0 );
	
	for(k=0; k<kmax-7; k+=8)
		{

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );
		a_00 = _mm256_load_ps( &A[24] );
		d_03 = _mm256_mul_ps( a_00, b_33 );

		_mm256_store_ps( &D[0], d_00 );
		_mm256_store_ps( &D[8], d_01 );
		_mm256_store_ps( &D[16], d_02 );
		_mm256_store_ps( &D[24], d_03 );

		A += 8*sda;
		D += 8*sdd;

		}
	if(k<kmax)
		{

		const float mask_f[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
		float m_f = kmax-k;

		mask_i = _mm256_castps_si256( _mm256_sub_ps( _mm256_loadu_ps( mask_f ), _mm256_broadcast_ss( &m_f ) ) );

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );
		a_00 = _mm256_load_ps( &A[24] );
		d_03 = _mm256_mul_ps( a_00, b_33 );

		_mm256_maskstore_ps( &D[0], mask_i, d_00 );
		_mm256_maskstore_ps( &D[8], mask_i, d_01 );
		_mm256_maskstore_ps( &D[16], mask_i, d_02 );
		_mm256_maskstore_ps( &D[24], mask_i, d_03 );

		}
	
	}



// B is the diagonal of a matrix
void kernel_sgemm_diag_right_4_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 8;

	int k;

	__m256
		alpha0, beta0,
		mask_f,
		sign,
		a_00,
		b_00, b_11, b_22, b_33,
		c_00,
		d_00, d_01, d_02, d_03;
	
	__m256i
		mask_i;
	
	alpha0 = _mm256_broadcast_ss( alpha );
	beta0  = _mm256_broadcast_ss( beta );
	
	b_00 = _mm256_broadcast_ss( &B[0] );
	b_00 = _mm256_mul_ps( b_00, alpha0 );
	b_11 = _mm256_broadcast_ss( &B[1] );
	b_11 = _mm256_mul_ps( b_11, alpha0 );
	b_22 = _mm256_broadcast_ss( &B[2] );
	b_22 = _mm256_mul_ps( b_22, alpha0 );
	b_33 = _mm256_broadcast_ss( &B[3] );
	b_33 = _mm256_mul_ps( b_33, alpha0 );
	
	for(k=0; k<kmax-7; k+=8)
		{

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );
		a_00 = _mm256_load_ps( &A[24] );
		d_03 = _mm256_mul_ps( a_00, b_33 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );
		c_00 = _mm256_load_ps( &C[16] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_02 = _mm256_add_ps( c_00, d_02 );
		c_00 = _mm256_load_ps( &C[24] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_03 = _mm256_add_ps( c_00, d_03 );

		_mm256_store_ps( &D[0], d_00 );
		_mm256_store_ps( &D[8], d_01 );
		_mm256_store_ps( &D[16], d_02 );
		_mm256_store_ps( &D[24], d_03 );

		A += 8*sda;
		C += 8*sdc;
		D += 8*sdd;

		}
	if(k<kmax)
		{

		const float mask_f[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
		float m_f = kmax-k;

		mask_i = _mm256_castps_si256( _mm256_sub_ps( _mm256_loadu_ps( mask_f ), _mm256_broadcast_ss( &m_f ) ) );

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );
		a_00 = _mm256_load_ps( &A[24] );
		d_03 = _mm256_mul_ps( a_00, b_33 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );
		c_00 = _mm256_load_ps( &C[16] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_02 = _mm256_add_ps( c_00, d_02 );
		c_00 = _mm256_load_ps( &C[24] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_03 = _mm256_add_ps( c_00, d_03 );

		_mm256_maskstore_ps( &D[0], mask_i, d_00 );
		_mm256_maskstore_ps( &D[8], mask_i, d_01 );
		_mm256_maskstore_ps( &D[16], mask_i, d_02 );
		_mm256_maskstore_ps( &D[24], mask_i, d_03 );

		}
	
	}



// B is the diagonal of a matrix
void kernel_sgemm_diag_right_3_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 8;

	int k;

	__m256
		alpha0, beta0,
		mask_f,
		sign,
		a_00,
		b_00, b_11, b_22,
		c_00,
		d_00, d_01, d_02;
	
	__m256i
		mask_i;
	
	alpha0 = _mm256_broadcast_ss( alpha );
	beta0  = _mm256_broadcast_ss( beta );
	
	b_00 = _mm256_broadcast_ss( &B[0] );
	b_00 = _mm256_mul_ps( b_00, alpha0 );
	b_11 = _mm256_broadcast_ss( &B[1] );
	b_11 = _mm256_mul_ps( b_11, alpha0 );
	b_22 = _mm256_broadcast_ss( &B[2] );
	b_22 = _mm256_mul_ps( b_22, alpha0 );
	
	for(k=0; k<kmax-7; k+=8)
		{

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );
		c_00 = _mm256_load_ps( &C[16] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_02 = _mm256_add_ps( c_00, d_02 );

		_mm256_store_ps( &D[0], d_00 );
		_mm256_store_ps( &D[8], d_01 );
		_mm256_store_ps( &D[16], d_02 );

		A += 8*sda;
		C += 8*sdc;
		D += 8*sdd;

		}
	if(k<kmax)
		{

		const float mask_f[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
		float m_f = kmax-k;

		mask_i = _mm256_castps_si256( _mm256_sub_ps( _mm256_loadu_ps( mask_f ), _mm256_broadcast_ss( &m_f ) ) );

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );
		a_00 = _mm256_load_ps( &A[16] );
		d_02 = _mm256_mul_ps( a_00, b_22 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );
		c_00 = _mm256_load_ps( &C[16] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_02 = _mm256_add_ps( c_00, d_02 );

		_mm256_maskstore_ps( &D[0], mask_i, d_00 );
		_mm256_maskstore_ps( &D[8], mask_i, d_01 );
		_mm256_maskstore_ps( &D[16], mask_i, d_02 );

		}
	
	}



// B is the diagonal of a matrix
void kernel_sgemm_diag_right_2_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	__m256
		alpha0, beta0,
		mask_f,
		sign,
		a_00,
		b_00, b_11,
		c_00,
		d_00, d_01;
	
	__m256i
		mask_i;
	
	alpha0 = _mm256_broadcast_ss( alpha );
	beta0  = _mm256_broadcast_ss( beta );
	
	b_00 = _mm256_broadcast_ss( &B[0] );
	b_00 = _mm256_mul_ps( b_00, alpha0 );
	b_11 = _mm256_broadcast_ss( &B[1] );
	b_11 = _mm256_mul_ps( b_11, alpha0 );
	
	for(k=0; k<kmax-7; k+=8)
		{

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );

		_mm256_store_ps( &D[0], d_00 );
		_mm256_store_ps( &D[8], d_01 );

		A += 8*sda;
		C += 8*sdc;
		D += 8*sdd;

		}
	if(k<kmax)
		{

		const float mask_f[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
		float m_f = kmax-k;

		mask_i = _mm256_castps_si256( _mm256_sub_ps( _mm256_loadu_ps( mask_f ), _mm256_broadcast_ss( &m_f ) ) );

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );
		a_00 = _mm256_load_ps( &A[8] );
		d_01 = _mm256_mul_ps( a_00, b_11 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );
		c_00 = _mm256_load_ps( &C[8] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_01 = _mm256_add_ps( c_00, d_01 );

		_mm256_maskstore_ps( &D[0], mask_i, d_00 );
		_mm256_maskstore_ps( &D[8], mask_i, d_01 );

		}
	
	}



// B is the diagonal of a matrix
void kernel_sgemm_diag_right_1_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	__m256
		alpha0, beta0,
		mask_f,
		sign,
		a_00,
		b_00,
		c_00,
		d_00;
	
	__m256i
		mask_i;
	
	alpha0 = _mm256_broadcast_ss( alpha );
	beta0  = _mm256_broadcast_ss( beta );
	
	b_00 = _mm256_broadcast_ss( &B[0] );
	b_00 = _mm256_mul_ps( b_00, alpha0 );
	
	for(k=0; k<kmax-7; k+=8)
		{

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );

		_mm256_store_ps( &D[0], d_00 );

		A += 8*sda;
		C += 8*sdc;
		D += 8*sdd;

		}
	if(k<kmax)
		{

		const float mask_f[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
		float m_f = kmax-k;

		mask_i = _mm256_castps_si256( _mm256_sub_ps( _mm256_loadu_ps( mask_f ), _mm256_broadcast_ss( &m_f ) ) );

		a_00 = _mm256_load_ps( &A[0] );
		d_00 = _mm256_mul_ps( a_00, b_00 );

		c_00 = _mm256_load_ps( &C[0] );
		c_00 = _mm256_mul_ps( c_00, beta0 );
		d_00 = _mm256_add_ps( c_00, d_00 );

		_mm256_maskstore_ps( &D[0], mask_i, d_00 );

		}
	
	}




