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

#include <stdlib.h>
#include <stdio.h>
#if 0
#include <malloc.h>
#endif

#if ! defined(OS_WINDOWS)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif



/* creates a zero matrix aligned */
void int_zeros(int **pA, int row, int col)
	{
	void *temp = malloc((row*col)*sizeof(int));
	*pA = temp;
	int *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0;
	}



/* creates a zero matrix aligned to a cache line */
void int_zeros_align(int **pA, int row, int col)
	{
#if defined(OS_WINDOWS)
	*pA = (int *) _aligned_malloc( (row*col)*sizeof(int), 64 );
#elif defined(__DSPACE__)
	*pA = malloc((row*col)*sizeof(int));
#else
	void *temp;
	int err = posix_memalign(&temp, 64, (row*col)*sizeof(int));
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*pA = temp;
#endif
	int *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* frees matrix */
void int_free(int *pA)
	{
	free( pA );
	}



/* frees aligned matrix */
void int_free_align(int *pA)
	{
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
	}



/* prints a matrix in column-major format */
void int_print_mat(int row, int col, int *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			printf("%d ", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}	



/* prints a matrix in column-major format */
void int_print_to_string_mat(char **buf_out, int row, int col, int *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			*buf_out += sprintf(*buf_out, "%d ", A[i+lda*j]);
			}
		*buf_out += sprintf(*buf_out, "\n");
		}
	*buf_out += sprintf(*buf_out, "\n");
	return;
	}	
