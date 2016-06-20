/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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

#include <math.h>

#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX
#endif



/* converts a column-major matrix into a panel-major matrix */
void d_cvt_mat2pmat(int row, int col, double *A, int lda, int offset, double *pA, int sda)
	{
	
	const int bs = 4;

	int 
		i, ii, j, jj, row0, row1, row2;
	
	double
		*B, *pB;
	
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		tmp;
#endif

	row0 = (bs-offset%bs)%bs;
	if(row0>row)
		row0 = row;
	row1 = row - row0;

	jj = 0;
	for( ; jj<col-3; jj+=4)
		{

		B  =  A + jj*lda;
		pB = pA + jj*bs;

		ii = 0;
		if(row0>0)
			{
			for( ; ii<row0; ii++)
				{
				pB[ii+bs*0] = B[ii+lda*0];
				pB[ii+bs*1] = B[ii+lda*1];
				pB[ii+bs*2] = B[ii+lda*2];
				pB[ii+bs*3] = B[ii+lda*3];
				}
			B  += row0;
			pB += row0 + bs*(sda-1);
			}
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		for( ; ii<row-3; ii+=4)
			{
			tmp = _mm256_loadu_pd( &B[0+lda*0] );
			_mm256_store_pd( &pB[0+bs*0], tmp );
			tmp = _mm256_loadu_pd( &B[0+lda*1] );
			_mm256_store_pd( &pB[0+bs*1], tmp );
			tmp = _mm256_loadu_pd( &B[0+lda*2] );
			_mm256_store_pd( &pB[0+bs*2], tmp );
			tmp = _mm256_loadu_pd( &B[0+lda*3] );
			_mm256_store_pd( &pB[0+bs*3], tmp );
			// update
			B  += 4;
			pB += bs*sda;
			}
#else
		for( ; ii<row-3; ii+=4)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			pB[1+bs*0] = B[1+lda*0];
			pB[2+bs*0] = B[2+lda*0];
			pB[3+bs*0] = B[3+lda*0];
			// col 1
			pB[0+bs*1] = B[0+lda*1];
			pB[1+bs*1] = B[1+lda*1];
			pB[2+bs*1] = B[2+lda*1];
			pB[3+bs*1] = B[3+lda*1];
			// col 2
			pB[0+bs*2] = B[0+lda*2];
			pB[1+bs*2] = B[1+lda*2];
			pB[2+bs*2] = B[2+lda*2];
			pB[3+bs*2] = B[3+lda*2];
			// col 3
			pB[0+bs*3] = B[0+lda*3];
			pB[1+bs*3] = B[1+lda*3];
			pB[2+bs*3] = B[2+lda*3];
			pB[3+bs*3] = B[3+lda*3];
			// update
			B  += 4;
			pB += bs*sda;
			}
#endif
		for( ; ii<row; ii++)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			// col 1
			pB[0+bs*1] = B[0+lda*1];
			// col 2
			pB[0+bs*2] = B[0+lda*2];
			// col 3
			pB[0+bs*3] = B[0+lda*3];
			// update
			B  += 1;
			pB += 1;
			}
		}
	for( ; jj<col; jj++)
		{

		B  =  A + jj*lda;
		pB = pA + jj*bs;

		ii = 0;
		if(row0>0)
			{
			for( ; ii<row0; ii++)
				{
				pB[ii+bs*0] = B[ii+lda*0];
				}
			B  += row0;
			pB += row0 + bs*(sda-1);
			}
		for( ; ii<row-3; ii+=4)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			pB[1+bs*0] = B[1+lda*0];
			pB[2+bs*0] = B[2+lda*0];
			pB[3+bs*0] = B[3+lda*0];
			// update
			B  += 4;
			pB += bs*sda;
			}
		for( ; ii<row; ii++)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			// update
			B  += 1;
			pB += 1;
			}
		}
	
	}



/* converts and transposes a column-major matrix into a panel-major matrix */
// row and col of the source matrix, offset in the destination matrix
void d_cvt_tran_mat2pmat(int row, int col, double *A, int lda, int offset, double *pA, int sda)
	{
	
	const int bs = 4;

	int i, ii, j, row0, row1, row2;
	
	double
		*B, *pB;
	
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	__m256d
		v0, v1, v2, v3,
		v4, v5, v6, v7;
#endif

	row0 = (bs-offset%bs)%bs;
	if(row0>col)
		row0 = col;
	row1 = col - row0;
	
	ii = 0;
	if(row0>0)
		{
		for(j=0; j<row; j++)
			{
			for(i=0; i<row0; i++)
				{
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
	
		A  += row0*lda;
		pA += row0 + bs*(sda-1);
		}
	
	ii = 0;
	for(; ii<row1-3; ii+=bs)
		{
		j=0;
		B  = A + ii*lda;
		pB = pA + ii*sda;
#if defined(TARGET_X64_INTEL_HASWELL) || defined(TARGET_X64_INTEL_SANDY_BRIDGE)
		for(; j<row-3; j+=4)
			{
			v0 = _mm256_loadu_pd( &B[0+0*lda] ); // 00 10 20 30
			v1 = _mm256_loadu_pd( &B[0+1*lda] ); // 01 11 21 31
			v4 = _mm256_unpacklo_pd( v0, v1 ); // 00 01 20 21
			v5 = _mm256_unpackhi_pd( v0, v1 ); // 10 11 30 31
			v2 = _mm256_loadu_pd( &B[0+2*lda] ); // 02 12 22 32
			v3 = _mm256_loadu_pd( &B[0+3*lda] ); // 03 13 23 33
			v6 = _mm256_unpacklo_pd( v2, v3 ); // 02 03 22 23
			v7 = _mm256_unpackhi_pd( v2, v3 ); // 12 13 32 33
			
			B += 4;

			v0 = _mm256_permute2f128_pd( v4, v6, 0x20 ); // 00 01 02 03
			_mm256_store_pd( &pB[0+bs*0], v0 );
			v2 = _mm256_permute2f128_pd( v4, v6, 0x31 ); // 20 21 22 23
			_mm256_store_pd( &pB[0+bs*2], v2 );
			v1 = _mm256_permute2f128_pd( v5, v7, 0x20 ); // 10 11 12 13
			_mm256_store_pd( &pB[0+bs*1], v1 );
			v3 = _mm256_permute2f128_pd( v5, v7, 0x31 ); // 30 31 32 33
			_mm256_store_pd( &pB[0+bs*3], v3 );

			pB += 4*bs;
			}
#else
		for(; j<row-3; j+=4)
			{
			// unroll 0
			pB[0+0*bs] = B[0+0*lda];
			pB[1+0*bs] = B[0+1*lda];
			pB[2+0*bs] = B[0+2*lda];
			pB[3+0*bs] = B[0+3*lda];
			// unroll 1
			pB[0+1*bs] = B[1+0*lda];
			pB[1+1*bs] = B[1+1*lda];
			pB[2+1*bs] = B[1+2*lda];
			pB[3+1*bs] = B[1+3*lda];
			// unroll 2
			pB[0+2*bs] = B[2+0*lda];
			pB[1+2*bs] = B[2+1*lda];
			pB[2+2*bs] = B[2+2*lda];
			pB[3+2*bs] = B[2+3*lda];
			// unroll 3
			pB[0+3*bs] = B[3+0*lda];
			pB[1+3*bs] = B[3+1*lda];
			pB[2+3*bs] = B[3+2*lda];
			pB[3+3*bs] = B[3+3*lda];
			B  += 4;
			pB += 4*bs;
			}
#endif
		for(; j<row; j++)
			{
			// unroll 0
			pB[0+0*bs] = B[0+0*lda];
			pB[1+0*bs] = B[0+1*lda];
			pB[2+0*bs] = B[0+2*lda];
			pB[3+0*bs] = B[0+3*lda];
			B  += 1;
			pB += 1*bs;
			}
		}
	if(ii<row1)
		{
		row2 = row1-ii;
		if(bs<row2) row2 = bs;
		for(j=0; j<row; j++)
			{
			for(i=0; i<row2; i++)
				{
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
		}
	
	}



/* converts a panel-major matrix into a column-major matrix */
void d_cvt_pmat2mat(int row, int col, int offset, double *pA, int sda, double *A, int lda)
	{
	
	const int bs = 4;

	int i, ii, jj;
	
	int row0 = (bs-offset%bs)%bs;
	
	double *ptr_pA;
	

	jj=0;
	for(; jj<col; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(row0>0)
			{
			for(; ii<row0; ii++)
				{
				A[ii+lda*jj] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row-bs+1; ii+=bs)
			{
			i=0;
			for(; i<bs; i++)
				{
				A[i+ii+lda*jj] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row; ii++)
			{
			A[ii+lda*jj] = ptr_pA[0];
			ptr_pA++;
			}
		}

	}



/* converts and transposes a panel-major matrix into a column-major matrix */
void d_cvt_tran_pmat2mat(int row, int col, int offset, double *pA, int sda, double *A, int lda)
	{
	
	const int bs = 4;

	int i, ii, jj;
	
	int row0 = (bs-offset%bs)%bs;
	
	double *ptr_pA;
	

	jj=0;
	for(; jj<col; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(row0>0)
			{
			for(; ii<row0; ii++)
				{
				A[jj+lda*ii] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row-bs+1; ii+=bs)
			{
			i=0;
			for(; i<bs; i++)
				{
				A[jj+lda*(i+ii)] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row; ii++)
			{
			A[jj+lda*ii] = ptr_pA[0];
			ptr_pA++;
			}
		}

	}


