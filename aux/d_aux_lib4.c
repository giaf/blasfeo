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
#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_d_kernel.h"



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



// copies a packed matrix into a packed matrix
void dgecp_lib(int m, int n, int offsetA, double *A, int sda, int offsetB, double *B, int sdb)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna, ii;

	int offA = offsetA%bs;
	int offB = offsetB%bs;

	// A at the beginning of the block
	A -= offA;

	// A at the beginning of the block
	B -= offB;

	// same alignment
	if(offA==offB)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(0, n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_0_lib4(0, n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_0_lib4(0, n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_0_lib4(0, n, A, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(0, n, A, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(0, n, A, B);
			else // if(m-ii==3)
				kernel_dgecp_3_0_lib4(0, n, A, B);
			}
		}
	// skip one element of A
	else if(offA==(offB+1)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
				//A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_3_lib4(0, n, A, sda, B+2);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_2_lib4(0, n, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_1_lib4(0, n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_1_lib4(0, n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(0, n, A+1, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(0, n, A+1, B);
			else // if(m-ii==3)
				kernel_dgecp_3_0_lib4(0, n, A+1, B);
			}
		}
	// skip 2 elements of A
	else if(offA==(offB+2)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgecp_2_3_lib4(0, n, A, sda, B+1);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(0, n, A+1, B+3);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(0, n, A, B+2);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_3_lib4(0, n, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_2_lib4(0, n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_2_lib4(0, n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(0, n, A+2, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(0, n, A+2, B);
			else // if(m-ii==3)
				kernel_dgecp_3_2_lib4(0, n, A, sda, B);
			}
		}
	// skip 3 elements of A
	else // if(offA==(offB+3)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(0, n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(0, n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(0, n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_0_lib4(0, n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_3_lib4(0, n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_3_lib4(0, n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(0, n, A+3, B);
			else if(m-ii==2)
				kernel_dgecp_2_3_lib4(0, n, A, sda, B);
			else // if(m-ii==3)
				kernel_dgecp_3_3_lib4(0, n, A, sda, B);
			}
		}

	}



// copies a lower triangular packed matrix into a lower triangular packed matrix
void dtrcp_l_lib(int m, int offsetA, double *A, int sda, int offsetB, double *B, int sdb)
	{

	if(m<=0)
		return;
	
	int n = m;

	const int bs = 4;

	int mna, ii;

	int offA = offsetA%bs;
	int offB = offsetB%bs;

	// A at the beginning of the block
	A -= offA;

	// A at the beginning of the block
	B -= offB;

	// same alignment
	if(offA==offB)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(1, ii, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_0_lib4(1, ii, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_0_lib4(1, ii, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_0_lib4(1, ii, A, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(1, ii, A, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(1, ii, A, B);
			else // if(m-ii==3)
				kernel_dgecp_3_0_lib4(1, ii, A, B);
			}
		}
	// skip one element of A
	else if(offA==(offB+1)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
				//A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_3_lib4(1, ii, A, sda, B+2);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_2_lib4(1, ii, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_1_lib4(1, ii, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_1_lib4(1, ii, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(1, ii, A+1, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(1, ii, A+1, B);
			else // if(m-ii==3)
				kernel_dgecp_3_0_lib4(1, ii, A+1, B);
			}
		}
	// skip 2 elements of A
	else if(offA==(offB+2)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgecp_2_3_lib4(1, ii, A, sda, B+1);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(1, ii, A+1, B+3);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(1, ii, A, B+2);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_3_lib4(1, ii, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_2_lib4(1, ii, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_2_lib4(1, ii, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(1, ii, A+2, B);
			else if(m-ii==2)
				kernel_dgecp_2_0_lib4(1, ii, A+2, B);
			else // if(m-ii==3)
				kernel_dgecp_3_2_lib4(1, ii, A, sda, B);
			}
		}
	// skip 3 elements of A
	else // if(offA==(offB+3)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgecp_2_0_lib4(1, ii, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgecp_1_0_lib4(1, ii, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgecp_2_0_lib4(1, ii, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgecp_3_0_lib4(1, ii, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgecp_8_3_lib4(1, ii, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgecp_4_3_lib4(1, ii, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgecp_1_0_lib4(1, ii, A+3, B);
			else if(m-ii==2)
				kernel_dgecp_2_3_lib4(1, ii, A, sda, B);
			else // if(m-ii==3)
				kernel_dgecp_3_3_lib4(1, ii, A, sda, B);
			}
		}

	}



// scaled and adds a packed matrix into a packed matrix: B = B + alpha*A
void dgead_lib(int m, int n, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna, ii;

	int offA = offsetA%bs;
	int offB = offsetB%bs;

	// A at the beginning of the block
	A -= offA;

	// A at the beginning of the block
	B -= offB;

	// same alignment
	if(offA==offB)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgead_2_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgead_2_0_lib4(n, alpha, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgead_3_0_lib4(n, alpha, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgead_8_0_lib4(n, alpha, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgead_4_0_lib4(n, alpha, A, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgead_1_0_lib4(n, alpha, A, B);
			else if(m-ii==2)
				kernel_dgead_2_0_lib4(n, alpha, A, B);
			else // if(m-ii==3)
				kernel_dgead_3_0_lib4(n, alpha, A, B);
			}
		}
	// skip one element of A
	else if(offA==(offB+1)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_dgead_2_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
				//A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgead_2_3_lib4(n, alpha, A, sda, B+2);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgead_3_2_lib4(n, alpha, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for( ; ii<m-7; ii+=8)
			{
			kernel_dgead_8_1_lib4(n, alpha, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for( ; ii<m-3; ii+=4)
			{
			kernel_dgead_4_1_lib4(n, alpha, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgead_1_0_lib4(n, alpha, A+1, B);
			else if(m-ii==2)
				kernel_dgead_2_0_lib4(n, alpha, A+1, B);
			else // if(m-ii==3)
				kernel_dgead_3_0_lib4(n, alpha, A+1, B);
			}
		}
	// skip 2 elements of A
	else if(offA==(offB+2)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgead_2_3_lib4(n, alpha, A, sda, B+1);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgead_1_0_lib4(n, alpha, A+1, B+3);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgead_2_0_lib4(n, alpha, A, B+2);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgead_3_3_lib4(n, alpha, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgead_8_2_lib4(n, alpha, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgead_4_2_lib4(n, alpha, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgead_1_0_lib4(n, alpha, A+2, B);
			else if(m-ii==2)
				kernel_dgead_2_0_lib4(n, alpha, A+2, B);
			else // if(m-ii==3)
				kernel_dgead_3_2_lib4(n, alpha, A, sda, B);
			}
		}
	// skip 3 elements of A
	else // if(offA==(offB+3)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_dgead_2_0_lib4(n, alpha, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_dgead_1_0_lib4(n, alpha, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_dgead_2_0_lib4(n, alpha, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_dgead_3_0_lib4(n, alpha, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_INTEL_SANDY_BRIDGE) || defined(TARGET_INTEL_HASWELL)
		for(; ii<m-7; ii+=8)
			{
			kernel_dgead_8_3_lib4(n, alpha, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_dgead_4_3_lib4(n, alpha, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_dgead_1_0_lib4(n, alpha, A+3, B);
			else if(m-ii==2)
				kernel_dgead_2_3_lib4(n, alpha, A, sda, B);
			else // if(m-ii==3)
				kernel_dgead_3_3_lib4(n, alpha, A, sda, B);
			}
		}

	}



// transpose general matrix; m and n are referred to the original matrix
void dgetr_lib(int m, int n, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc)
	{

/*

m = 5
n = 3
offsetA = 1
offsetC = 2

A = 
 x x x
 -
 x x x
 x x x
 x x x
 x x x

C =
 x x x x x
 x x x x x
 -
 x x x x x

*/

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna = (bs-offsetA%bs)%bs;
	mna = m<mna ? m : mna;
	int nna = (bs-offsetC%bs)%bs;
	nna = n<nna ? n : nna;
	
	int ii;

	ii = 0;

	if(mna>0)
		{
		if(mna==1)
			kernel_dgetr_1_lib4(0, n, nna, pA, pC, sdc);
		else if(mna==2)
			kernel_dgetr_2_lib4(0, n, nna, pA, pC, sdc);
		else //if(mna==3)
			kernel_dgetr_3_lib4(0, n, nna, pA, pC, sdc);
		ii += mna;
		pA += mna + bs*(sda-1);
		pC += mna*bs;
		}
#if defined(TARGET_INTEL_HASWELL)
	for( ; ii<m-7; ii+=8)
		{
		kernel_dgetr_8_lib4(0, n, nna, pA, sda, pC, sdc);
		pA += 2*bs*sda;
		pC += 2*bs*bs;
		}
#endif
	for( ; ii<m-3; ii+=4)
//	for( ; ii<m; ii+=4)
		{
		kernel_dgetr_4_lib4(0, n, nna, pA, pC, sdc);
		pA += bs*sda;
		pC += bs*bs;
		}

	// clean-up at the end using smaller kernels
	if(ii==m)
		return;
	
	if(m-ii==1)
		kernel_dgetr_1_lib4(0, n, nna, pA, pC, sdc);
	else if(m-ii==2)
		kernel_dgetr_2_lib4(0, n, nna, pA, pC, sdc);
	else if(m-ii==3)
		kernel_dgetr_3_lib4(0, n, nna, pA, pC, sdc);
		
	return;
	
	}	



// transpose lower triangular matrix
void dtrtr_l_lib(int m, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc)
	{

/*

A = 
 x
 x x
 x x x
 x x x x
  
 x x x x x
 x x x x x x
 x x x x x x x
 x x x x x x x x

C =
 x x x x x x x x
  
   x x x x x x x
     x x x x x x
	   x x x x x
	     x x x x

	       x x x
	         x x
	           x

*/

	int n = m;

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna = (bs-offsetA%bs)%bs;
	mna = m<mna ? m : mna;
	int nna = (bs-offsetC%bs)%bs;
	nna = n<nna ? n : nna;
	
	int ii;

	ii = 0;

	if(mna>0)
		{
		if(mna==1)
			{
			pC[0] = pA[0];
			}
		else if(mna==2)
			{
			if(nna==1)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[1+bs*(0+sdc)] = pA[1+bs*1];
				}
			else
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[1+bs*1] = pA[1+bs*1];
				}
			}
		else //if(mna==3)
			{
			if(nna==1)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[0+bs*2] = pA[2+bs*0];
				pC[1+bs*(0+sdc)] = pA[1+bs*1];
				pC[1+bs*(1+sdc)] = pA[2+bs*1];
				pC[2+bs*(1+sdc)] = pA[2+bs*2];
				}
			else if(nna==2)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[0+bs*2] = pA[2+bs*0];
				pC[1+bs*1] = pA[1+bs*1];
				pC[1+bs*2] = pA[2+bs*1];
				pC[2+bs*(1+sdc)] = pA[2+bs*2];
				}
			else
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[0+bs*2] = pA[2+bs*0];
				pC[1+bs*1] = pA[1+bs*1];
				pC[1+bs*2] = pA[2+bs*1];
				pC[2+bs*2] = pA[2+bs*2];
				}
			}
		ii += mna;
		pA += mna + bs*(sda-1);
		pC += mna*bs;
		}
#if 0 //defined(TARGET_INTEL_HASWELL)
	for( ; ii<m-7; ii+=8)
		{
		kernel_dgetr_8_lib4(1, n, nna, pA, sda, pC, sdc);
		pA += 2*bs*sda;
		pC += 2*bs*bs;
		}
#endif
	for( ; ii<m-3; ii+=4)
		{
		kernel_dgetr_4_lib4(1, ii, nna, pA, pC, sdc);
		pA += bs*sda;
		pC += bs*bs;
		}
	
	// clean-up at the end using smaller kernels
	if(ii==m)
		return;
	
	if(m-ii==1)
		kernel_dgetr_1_lib4(1, ii, nna, pA, pC, sdc);
	else if(m-ii==2)
		kernel_dgetr_2_lib4(1, ii, nna, pA, pC, sdc);
	else if(m-ii==3)
		kernel_dgetr_3_lib4(1, ii, nna, pA, pC, sdc);
		
	return;

	}



// transpose an aligned upper triangular matrix into an aligned lower triangular matrix
void dtrtr_u_lib(int m, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc)
	{

/*

A = 
 x x x x x x x x
   x x x x x x x

     x x x x x x
       x x x x x
         x x x x
           x x x
             x x
               x

C = 
 x

 x x
 x x x
 x x x x
 x x x x x
 x x x x x x
 x x x x x x x
 x x x x x x x x

*/

	int n = m;

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna = (bs-offsetA%bs)%bs;
	mna = m<mna ? m : mna;
	int nna = (bs-offsetC%bs)%bs;
	nna = n<nna ? n : nna;
	int tna = nna;
	
	int ii;

	ii = 0;

	if(mna>0)
		{
		if(mna==1)
			{
			kernel_dgetr_1_lib4(0, n, nna, pA, pC, sdc);
			if(nna!=1)
				{
//				pC[0+bs*0] = pA[0+bs*0];
				pA += 1*bs;
				pC += 1;
				tna = (bs-(offsetC+1)%bs)%bs;
				}
			else //if(nna==1)
				{
//				pC[0+bs*0] = pA[0+bs*0];
				pA += 1*bs;
				pC += 1 + (sdc-1)*bs;
				tna = 0; //(bs-(offsetC+1)%bs)%bs;
				}
//			kernel_dgetr_1_lib4(0, n-1, tna, pA, pC, sdc);
			}
		else if(mna==2)
			{
			if(nna==0 || nna==3)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pA += 2*bs;
				pC += 2;
				tna = (bs-(offsetC+2)%bs)%bs;
				kernel_dgetr_2_lib4(0, n-2, tna, pA, pC, sdc);
				}
			else if(nna==1)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pA += 1*bs;
				pC += 1 + (sdc-1)*bs;
//				pC[0+bs*0] = pA[0+bs*0];
//				pC[0+bs*1] = pA[1+bs*0];
				kernel_dgetr_2_lib4(0, n-1, 0, pA, pC, sdc);
				pA += 1*bs;
				pC += 1;
				tna = 3; //(bs-(offsetC+2)%bs)%bs;
//				kernel_dgetr_2_lib4(0, n-2, tna, pA, pC, sdc);
				}
			else if(nna==2)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pA += 2*bs;
				pC += 2 + (sdc-1)*bs;
				tna = 0; //(bs-(offsetC+2)%bs)%bs;
				kernel_dgetr_2_lib4(0, n-2, tna, pA, pC, sdc);
				}
			}
		else //if(mna==3)
			{
			if(nna==0)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pC[2+bs*0] = pA[0+bs*2];
				pC[2+bs*1] = pA[1+bs*2];
				pC[2+bs*2] = pA[2+bs*2];
				pA += 3*bs;
				pC += 3;
				tna = 1;
				kernel_dgetr_3_lib4(0, n-3, tna, pA, pC, sdc);
				}
			else if(nna==1)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pA += bs;
				pC += 1 + (sdc-1)*bs;
				pC[0+bs*0] = pA[0+bs*0];
				pC[0+bs*1] = pA[1+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pC[1+bs*2] = pA[2+bs*1];
				pA += 2*bs;
				pC += 2;
				tna = 2;
				kernel_dgetr_3_lib4(0, n-3, tna, pA, pC, sdc);
				}
			else if(nna==2)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pA += 2*bs;
				pC += 2 + (sdc-1)*bs;
//				pC[0+bs*0] = pA[0+bs*0];
//				pC[0+bs*1] = pA[1+bs*0];
//				pC[0+bs*2] = pA[2+bs*0];
				kernel_dgetr_3_lib4(0, n-2, 0, pA, pC, sdc);
				pA += 1*bs;
				pC += 1;
				tna = 3;
//				kernel_dgetr_3_lib4(0, n-3, tna, pA, pC, sdc);
				}
			else //if(nna==3)
				{
				pC[0+bs*0] = pA[0+bs*0];
				pC[1+bs*0] = pA[0+bs*1];
				pC[1+bs*1] = pA[1+bs*1];
				pC[2+bs*0] = pA[0+bs*2];
				pC[2+bs*1] = pA[1+bs*2];
				pC[2+bs*2] = pA[2+bs*2];
				pA += 3*bs;
				pC += 3 + (sdc-1)*bs;
				tna = 0;
				kernel_dgetr_3_lib4(0, n-3, tna, pA, pC, sdc);
				}
			}
		ii += mna;
		pA += mna + bs*(sda-1);
		pC += mna*bs;
		}
#if 0 //defined(TARGET_X64_AVX2)
	for( ; ii<m-7; ii+=8)
		{
		kernel_dgetr_8_lib4(0, n, nna, pA, sda, pC, sdc);
		pA += 2*bs*sda;
		pC += 2*bs*bs;
		}
#endif
	for( ; ii<m-3; ii+=4)
		{
		if(tna==0)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pC[2+bs*0] = pA[0+bs*2];
			pC[2+bs*1] = pA[1+bs*2];
			pC[2+bs*2] = pA[2+bs*2];
			pC[3+bs*0] = pA[0+bs*3];
			pC[3+bs*1] = pA[1+bs*3];
			pC[3+bs*2] = pA[2+bs*3];
			pC[3+bs*3] = pA[3+bs*3];
			pA += 4*bs;
			pC += sdc*bs;
			kernel_dgetr_4_lib4(0, n-ii-4, 0, pA, pC, sdc);
			}
		else if(tna==1)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pA += bs;
			pC += 1 + (sdc-1)*bs;
			pC[0+bs*0] = pA[0+bs*0];
			pC[0+bs*1] = pA[1+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pC[1+bs*2] = pA[2+bs*1];
			pC[2+bs*0] = pA[0+bs*2];
			pC[2+bs*1] = pA[1+bs*2];
			pC[2+bs*2] = pA[2+bs*2];
			pC[2+bs*3] = pA[3+bs*2];
			pA += 3*bs;
			pC += 3;
			kernel_dgetr_4_lib4(0, n-ii-4, 1, pA, pC, sdc);
			}
		else if(tna==2)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pA += 2*bs;
			pC += 2 + (sdc-1)*bs;
			pC[0+bs*0] = pA[0+bs*0];
			pC[0+bs*1] = pA[1+bs*0];
			pC[0+bs*2] = pA[2+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pC[1+bs*2] = pA[2+bs*1];
			pC[1+bs*3] = pA[3+bs*1];
			pA += 2*bs;
			pC += 2;
			kernel_dgetr_4_lib4(0, n-ii-4, 2, pA, pC, sdc);
			}
		else //if(tna==3)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pC[2+bs*0] = pA[0+bs*2];
			pC[2+bs*1] = pA[1+bs*2];
			pC[2+bs*2] = pA[2+bs*2];
			pA += 3*bs;
			pC += 3 + (sdc-1)*bs;
			kernel_dgetr_4_lib4(0, n-ii-3, 0, pA, pC, sdc);
//			pC[0+bs*0] = pA[0+bs*0];
//			pC[0+bs*1] = pA[1+bs*0];
//			pC[0+bs*2] = pA[2+bs*0];
//			pC[0+bs*3] = pA[3+bs*0];
			pA += bs;
			pC += 1;
//			kernel_dgetr_4_lib4(0, n-ii-4, tna, pA, pC, sdc);
			}
		pA += bs*sda;
		pC += bs*bs;
		}

	// clean-up at the end
	if(ii==m)
		return;
	
	if(m-ii==1)
		{
		pC[0+bs*0] = pA[0+bs*0];
		}
	else if(m-ii==2)
		{
		if(tna!=1)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			}
		else //if(tna==1)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pA += bs;
			pC += 1 + (sdc-1)*bs;
			pC[0+bs*0] = pA[0+bs*0];
			pC[0+bs*1] = pA[1+bs*0];
			}
		}
	else if(m-ii==3)
		{
		if(tna==0 || tna==3)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pC[2+bs*0] = pA[0+bs*2];
			pC[2+bs*1] = pA[1+bs*2];
			pC[2+bs*2] = pA[2+bs*2];
			}
		else if(tna==1)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pA += bs;
			pC += 1 + (sdc-1)*bs;
			pC[0+bs*0] = pA[0+bs*0];
			pC[0+bs*1] = pA[1+bs*0];
			pC[1+bs*0] = pA[0+bs*0];
			pC[1+bs*1] = pA[1+bs*1];
			pC[1+bs*2] = pA[2+bs*1];
			}
		else //if(tna==2)
			{
			pC[0+bs*0] = pA[0+bs*0];
			pC[1+bs*0] = pA[0+bs*1];
			pC[1+bs*1] = pA[1+bs*1];
			pA += 2*bs;
			pC += 2 + (sdc-1)*bs;
			pC[0+bs*0] = pA[0+bs*0];
			pC[0+bs*1] = pA[1+bs*0];
			pC[0+bs*2] = pA[2+bs*0];
			}
		}
		
	return;

	}



// regularize diagonal 
void ddiareg_lib(int kmax, double reg, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] += reg;
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+(jj+0)*bs+0] += reg;
		pD[jj*sdd+(jj+1)*bs+1] += reg;
		pD[jj*sdd+(jj+2)*bs+2] += reg;
		pD[jj*sdd+(jj+3)*bs+3] += reg;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] += reg;
		}
	
	}



// insert vector to diagonal 
void ddiain_lib(int kmax, double *x, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] = x[ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+(jj+0)*bs+0] = x[jj+0];
		pD[jj*sdd+(jj+1)*bs+1] = x[jj+1];
		pD[jj*sdd+(jj+2)*bs+2] = x[jj+2];
		pD[jj*sdd+(jj+3)*bs+3] = x[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] = x[jj+ll];
		}
	
	}



// insert sqrt of vector to diagonal 
void ddiain_sqrt_lib(int kmax, double *x, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] = sqrt(x[ll]);
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+(jj+0)*bs+0] = sqrt(x[jj+0]);
		pD[jj*sdd+(jj+1)*bs+1] = sqrt(x[jj+1]);
		pD[jj*sdd+(jj+2)*bs+2] = sqrt(x[jj+2]);
		pD[jj*sdd+(jj+3)*bs+3] = sqrt(x[jj+3]);
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] = sqrt(x[jj+ll]);
		}
	
	}



// extract diagonal to vector 
void ddiaex_lib(int kmax, int offset, double *pD, int sdd, double *x)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			x[ll] = pD[ll+bs*ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		x[jj+0] = pD[jj*sdd+(jj+0)*bs+0];
		x[jj+1] = pD[jj*sdd+(jj+1)*bs+1];
		x[jj+2] = pD[jj*sdd+(jj+2)*bs+2];
		x[jj+3] = pD[jj*sdd+(jj+3)*bs+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		x[jj+ll] = pD[jj*sdd+(jj+ll)*bs+ll];
		}
	
	}



// add scaled vector to diagonal 
void ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] += alpha * x[ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+(jj+0)*bs+0] += alpha * x[jj+0];
		pD[jj*sdd+(jj+1)*bs+1] += alpha * x[jj+1];
		pD[jj*sdd+(jj+2)*bs+2] += alpha * x[jj+2];
		pD[jj*sdd+(jj+3)*bs+3] += alpha * x[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] += alpha * x[jj+ll];
		}
	
	}



// insert vector to diagonal, sparse formulation 
void ddiain_libsp(int kmax, int *idx, double *x, double *pD, int sdd)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] = x[jj];
		}
	
	}



// add scaled vector to diagonal, sparse formulation 
void ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] += alpha * x[jj];
		}
	
	}



// add scaled vector to another vector and insert to diagonal, sparse formulation 
void ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] = y[jj] + alpha * x[jj];
		}
	
	}



// insert vector to row 
void drowin_lib(int kmax, double *x, double *pD)
	{
	
	const int bs = 4;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[(jj+0)*bs] = x[jj+0];
		pD[(jj+1)*bs] = x[jj+1];
		pD[(jj+2)*bs] = x[jj+2];
		pD[(jj+3)*bs] = x[jj+3];
		}
	for(; jj<kmax; jj++)
		{
		pD[(jj)*bs] = x[jj];
		}
	
	}



// extract row to vector
void drowex_lib(int kmax, double *pD, double *x)
	{
	
	const int bs = 4;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		x[jj+0] = pD[(jj+0)*bs];
		x[jj+1] = pD[(jj+1)*bs];
		x[jj+2] = pD[(jj+2)*bs];
		x[jj+3] = pD[(jj+3)*bs];
		}
	for(; jj<kmax; jj++)
		{
		x[jj] = pD[(jj)*bs];
		}
	
	}



// add scaled vector to row 
void drowad_lib(int kmax, double alpha, double *x, double *pD)
	{

	const int bs = 4;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[(jj+0)*bs] += alpha * x[jj+0];
		pD[(jj+1)*bs] += alpha * x[jj+1];
		pD[(jj+2)*bs] += alpha * x[jj+2];
		pD[(jj+3)*bs] += alpha * x[jj+3];
		}
	for(; jj<kmax; jj++)
		{
		pD[(jj)*bs] += alpha * x[jj];
		}
	
	}



// insert vector to row, sparse formulation 
void drowin_libsp(int kmax, int *idx, double *x, double *pD)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] = x[jj];
		}
	
	}



// add scaled vector to row, sparse formulation 
void drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] += alpha * x[jj];
		}
	
	}



// add scaled vector to another vector and insert to row, sparse formulation 
void drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] = y[jj] + alpha * x[jj];
		}
	
	}



// swap two rows
void drowsw_lib(int kmax, double *pA, double *pC)
	{

	const int bs = 4;

	int ii;
	double tmp;

	for(ii=0; ii<kmax-3; ii+=4)
		{
		tmp = pA[0+bs*0];
		pA[0+bs*0] = pC[0+bs*0];
		pC[0+bs*0] = tmp;
		tmp = pA[0+bs*1];
		pA[0+bs*1] = pC[0+bs*1];
		pC[0+bs*1] = tmp;
		tmp = pA[0+bs*2];
		pA[0+bs*2] = pC[0+bs*2];
		pC[0+bs*2] = tmp;
		tmp = pA[0+bs*3];
		pA[0+bs*3] = pC[0+bs*3];
		pC[0+bs*3] = tmp;
		pA += 4*bs;
		pC += 4*bs;
		}
	for( ; ii<kmax; ii++)
		{
		tmp = pA[0+bs*0];
		pA[0+bs*0] = pC[0+bs*0];
		pC[0+bs*0] = tmp;
		pA += 1*bs;
		pC += 1*bs;
		}
	
	}



// insert vector to column 
void dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll] = x[ll];
			}
		pD += kna + bs*(sdd-1);
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+0] = x[jj+0];
		pD[jj*sdd+1] = x[jj+1];
		pD[jj*sdd+2] = x[jj+2];
		pD[jj*sdd+3] = x[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+ll] = x[jj+ll];
		}
	
	}



// add scaled vector to column 
void dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd)
	{

	const int bs = 4;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll] += alpha * x[ll];
			}
		pD += kna + bs*(sdd-1);
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[jj*sdd+0] += alpha * x[jj+0];
		pD[jj*sdd+1] += alpha * x[jj+1];
		pD[jj*sdd+2] += alpha * x[jj+2];
		pD[jj*sdd+3] += alpha * x[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+ll] += alpha * x[jj+ll];
		}
	
	}



// insert vector to diagonal, sparse formulation 
void dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs] = x[jj];
		}
	
	}



// add scaled vector to diagonal, sparse formulation 
void dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs] += alpha * x[jj];
		}
	
	}



// swaps two cols
void dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc)
	{

	const int bs = 4;

	int ii;

	double tmp;

	if(offsetA==offsetC)
		{
		if(offsetA>0)
			{
			ii = 0;
			for(; ii<bs-offsetA; ii++)
				{
				tmp = pA[0+bs*0];
				pA[0+bs*0] = pC[0+bs*0];
				pC[0+bs*0] = tmp;
				pA += 1;
				pC += 1;
				}
			pA += bs*(sda-1);
			pC += bs*(sdc-1);
			kmax -= bs-offsetA;
			}
		ii = 0;
		for(; ii<kmax-3; ii+=4)
			{
			tmp = pA[0+bs*0];
			pA[0+bs*0] = pC[0+bs*0];
			pC[0+bs*0] = tmp;
			tmp = pA[1+bs*0];
			pA[1+bs*0] = pC[1+bs*0];
			pC[1+bs*0] = tmp;
			tmp = pA[2+bs*0];
			pA[2+bs*0] = pC[2+bs*0];
			pC[2+bs*0] = tmp;
			tmp = pA[3+bs*0];
			pA[3+bs*0] = pC[3+bs*0];
			pC[3+bs*0] = tmp;
			pA += bs*sda;
			pC += bs*sdc;
			}
		for(; ii<kmax; ii++)
			{
			tmp = pA[0+bs*0];
			pA[0+bs*0] = pC[0+bs*0];
			pC[0+bs*0] = tmp;
			pA += 1;
			pC += 1;
			}
		}
	else
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}

	return;

	}



// insert vector to vector, sparse formulation
void dvecin_libsp(int kmax, int *idx, double *x, double *y)
	{

	int jj;

	for(jj=0; jj<kmax; jj++)
		{
		y[idx[jj]] = x[jj];
		}
	
	}



// adds vector to vector, sparse formulation
void dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y)
	{

	int jj;

	for(jj=0; jj<kmax; jj++)
		{
		y[idx[jj]] += alpha * x[jj];
		}
	
	}



/****************************
* new interface
****************************/



#if defined(BLASFEO_LA)



// return memory size (in bytes) needed for a strmat
int d_size_strmat(int m, int n)
	{
	int bs = D_BS;
	int nc = D_NC;
	int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	int memory_size = (pm*cn+tmp)*sizeof(double);
	return memory_size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer (and update it)
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory)
	{
	int bs = D_BS;
	int nc = D_NC;
	int al = bs*nc;
	sA->bs = bs;
	sA->m = m;
	sA->n = n;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	double *ptr = (double *) memory;
	sA->pA = ptr;
	ptr += pm*cn;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->use_dA = 0;
	sA->memory_size = (pm*cn+tmp)*sizeof(double);
	return;
	}



// convert a matrix into a matrix structure
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_mat2pmat(m, n, A, lda, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn);
	return;
	}



// convert and transpose a matrix into a matrix structure
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_tran_mat2pmat(m, n, A, lda, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn);
	return;
	}



// convert a matrix structure into a matrix
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_pmat2mat(m, n, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn, A, lda);
	return;
	}



// convert and transpose a matrix structure into a matrix
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int bs = sA->bs;
	double *pA = sA->pA;
	int pm = sA->pm;
	int cn = sA->cn;
	d_cvt_tran_pmat2mat(m, n, ai, pA+ai/bs*bs*cn+ai%bs+aj*bs, cn, A, lda);
	return;
	}



// swap two rows of a matrix struct
void drowsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
	int bsA = sA->bs;
	int sda = sA->cn;
	double *pA = sA->pA + ai/bsA*bsA*sda + ai%bsA + aj*bsA;
	int bsC = sC->bs;
	int sdc = sC->cn;
	double *pC = sC->pA + ci/bsC*bsC*sdc + ci%bsC + cj*bsC;
	drowsw_lib(kmax, pA, pC);
	return;
	}



// permute the rows of a matrix struct
void drowpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		drowsw_libstr(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}



// swap two cols of a matrix struct
void dcolsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
	int bsA = sA->bs;
	int sda = sA->cn;
	double *pA = sA->pA + ai/bsA*bsA*sda + ai%bsA + aj*bsA;
	int bsC = sC->bs;
	int sdc = sC->cn;
	double *pC = sC->pA + ci/bsC*bsC*sdc + ci%bsC + cj*bsC;
	dcolsw_lib(kmax, ai%bsA, pA, sda, ci%bsC, pC, sdc);
	return;
	}



// permute the cols of a matrix struct
void dcolpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		dcolsw_libstr(sA->n, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// linear algebra provided by BLAS
#elif defined(BLAS_LA)



// return memory size (in bytes) needed for a strmat
int d_size_strmat(int m, int n)
	{
	int size = (m*n)*sizeof(double);
	return size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer (and update it)
void d_create_strmat(int m, int n, struct d_strmat *sA, void *memory)
	{
	sA->m = m;
	sA->n = n;
	double *ptr = (double *) memory;
	sA->pA = ptr;
	ptr += m * n;
	sA->memory_size = (m*n)*sizeof(double);
	return;
	}



// convert a matrix into a matrix structure
void d_cvt_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pA[ii+0+jj*lda2] = A[ii+0+jj*lda];
			pA[ii+1+jj*lda2] = A[ii+1+jj*lda];
			pA[ii+2+jj*lda2] = A[ii+2+jj*lda];
			pA[ii+3+jj*lda2] = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pA[ii+0+jj*lda2] = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert and transpose a matrix into a matrix structure
void d_cvt_tran_mat2strmat(int m, int n, double *A, int lda, struct d_strmat *sA, int ai, int aj)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			pA[jj+(ii+0)*lda2] = A[ii+0+jj*lda];
			pA[jj+(ii+1)*lda2] = A[ii+1+jj*lda];
			pA[jj+(ii+2)*lda2] = A[ii+2+jj*lda];
			pA[jj+(ii+3)*lda2] = A[ii+3+jj*lda];
			}
		for(; ii<m; ii++)
			{
			pA[jj+(ii+0)*lda2] = A[ii+0+jj*lda];
			}
		}
	return;
	}



// convert a matrix structure into a matrix 
void d_cvt_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			A[ii+0+jj*lda] = pA[ii+0+jj*lda2];
			A[ii+1+jj*lda] = pA[ii+1+jj*lda2];
			A[ii+2+jj*lda] = pA[ii+2+jj*lda2];
			A[ii+3+jj*lda] = pA[ii+3+jj*lda2];
			}
		for(; ii<m; ii++)
			{
			A[ii+0+jj*lda] = pA[ii+0+jj*lda2];
			}
		}
	return;
	}



// convert and transpose a matrix structure into a matrix 
void d_cvt_tran_strmat2mat(int m, int n, struct d_strmat *sA, int ai, int aj, double *A, int lda)
	{
	int ii, jj;
	int lda2 = sA->m;
	double *pA = sA->pA + ai + aj*lda2;
	for(jj=0; jj<n; jj++)
		{
		ii = 0;
		for(; ii<m-3; ii+=4)
			{
			A[ii+0+jj*lda] = pA[jj+(ii+0)*lda2];
			A[ii+1+jj*lda] = pA[jj+(ii+1)*lda2];
			A[ii+2+jj*lda] = pA[jj+(ii+2)*lda2];
			A[ii+3+jj*lda] = pA[jj+(ii+3)*lda2];
			}
		for(; ii<m; ii++)
			{
			A[ii+0+jj*lda] = pA[jj+(ii+0)*lda2];
			}
		}
	return;
	}



// swap two rows of a matrix struct
void drowsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*lda;
	int ii;
	double tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = pA[ii*lda];
		pA[ii*lda] = pC[ii*ldc];
		pC[ii*ldc] = tmp;
		}
	return;
	}



// permute the rows of a matrix struct
void drowpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		drowsw_libstr(sA->n, sA, ii, 0, sA, ipiv[ii]-1, 0);
		}
	return;
	}



// swap two cols of a matrix struct
void dcolsw_libstr(int kmax, struct d_strmat *sA, int ai, int aj, struct d_strmat *sC, int ci, int cj)
	{
	int lda = sA->m;
	double *pA = sA->pA + ai + aj*lda;
	int ldc = sC->m;
	double *pC = sC->pA + ci + cj*lda;
	int ii;
	double tmp;
	for(ii=0; ii<kmax; ii++)
		{
		tmp = pA[ii];
		pA[ii] = pC[ii];
		pC[ii] = tmp;
		}
	return;
	}



// permute the cols of a matrix struct
void dcolpe_libstr(int kmax, int *ipiv, struct d_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		dcolsw_libstr(sA->n, sA, 0, ii, sA, 0, ipiv[ii]-1);
		}
	return;
	}



#endif


