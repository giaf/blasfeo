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

#include <stdlib.h>
#include <stdio.h>
#if 0
#include <malloc.h>
#endif

#if ! defined(OS_WINDOWS)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif



/* creates a zero matrix aligned */
void s_zeros(float **pA, int row, int col)
	{
	void *temp = malloc((row*col)*sizeof(float));
	*pA = temp;
	float *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* creates a zero matrix aligned to a cache line */
void s_zeros_align(float **pA, int row, int col)
	{
#if defined(OS_WINDOWS)
	*pA = (float *) _aligned_malloc( (row*col)*sizeof(float), 64 );
#else
	void *temp;
	int err = posix_memalign(&temp, 64, (row*col)*sizeof(float));
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*pA = temp;
#endif
	float *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* frees matrix */
void s_free(float *pA)
	{
	free( pA );
	}



/* frees aligned matrix */
void d_free_align(float *pA)
	{
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
	}



/* prints a matrix in column-major format */
void s_print_mat(int row, int col, float *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			printf("%9.5f ", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}	



/* prints a matrix in column-major format (exponential notation) */
void s_print_mat_e(int row, int col, float *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			printf("%e\t", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}	



/* prints a matrix in panel-major format */
void s_print_pmat(int row, int col, float *pA, int sda)
	{

	const int bs = 4;

	int ii, i, j, row2;

	for(ii=0; ii<row-(bs-1); ii+=bs)
		{
		for(i=0; i<bs; i++)
			{
			for(j=0; j<col; j++)
				{
				printf("%9.5f ", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<row)
		{
		row2 = row-ii;
		for(i=0; i<row2; i++)
			{
			for(j=0; j<col; j++)
				{
				printf("%9.5f ", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");

	}	



/* prints a matrix in panel-major format (exponential notation) */
void s_print_pmat_e(int row, int col, float *pA, int sda)
	{

	const int bs = 4;

	int ii, i, j, row2;

	for(ii=0; ii<row-(bs-1); ii+=bs)
		{
		for(i=0; i<bs; i++)
			{
			for(j=0; j<col; j++)
				{
				printf("%e\t", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<row)
		{
		row2 = row-ii;
		for(i=0; i<row2; i++)
			{
			for(j=0; j<col; j++)
				{
				printf("%e\t", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");

	}	



