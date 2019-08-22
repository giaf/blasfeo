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

#include "../include/blasfeo_stdlib.h"

#if ! defined(OS_WINDOWS)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif



/* creates a zero matrix */
void ZEROS(REAL **pA, int row, int col)
	{
	*pA = malloc((row*col)*sizeof(REAL));
	REAL *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* creates a zero matrix aligned to a cache line */
void ZEROS_ALIGN(REAL **pA, int row, int col)
	{
    blasfeo_malloc_align((void **) pA, (row*col)*sizeof(REAL));
	REAL *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* frees matrix */
void FREE(REAL *pA)
	{
	free( pA );
	}



/* frees aligned matrix */
void FREE_ALIGN(REAL *pA)
	{
	blasfeo_free_align(pA);
	}



/* prints a matrix in column-major format */
void PRINT_MAT(int m, int n, REAL *A, int lda)
	{
	int i, j;
	for(i=0; i<m; i++)
		{
		for(j=0; j<n; j++)
			{
			printf("%9.5f ", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	return;
	}



/* prints the transposed of a matrix in column-major format */
void PRINT_TRAN_MAT(int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(j=0; j<col; j++)
		{
		for(i=0; i<row; i++)
			{
			printf("%9.5f ", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}



/* prints a matrix in column-major format */
void PRINT_TO_FILE_MAT(FILE *file, int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			fprintf(file, "%9.5f ", A[i+lda*j]);
			}
		fprintf(file, "\n");
		}
	fprintf(file, "\n");
	}

/* prints a matrix in column-major format */
void PRINT_TO_FILE_EXP_MAT(FILE *file, int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			fprintf(file, "%9.5e ", A[i+lda*j]);
			}
		fprintf(file, "\n");
		}
	fprintf(file, "\n");
	}


/* prints a matrix in column-major format */
void PRINT_TO_STRING_MAT(char **buf_out, int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			*buf_out += sprintf(*buf_out, "%9.5f ", A[i+lda*j]);
			}
		*buf_out += sprintf(*buf_out, "\n");
		}
	*buf_out += sprintf(*buf_out, "\n");
	return;
	}



/* prints the transposed of a matrix in column-major format */
void PRINT_TO_FILE_TRAN_MAT(FILE *file, int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(j=0; j<col; j++)
		{
		for(i=0; i<row; i++)
			{
			fprintf(file, "%9.5f ", A[i+lda*j]);
			}
		fprintf(file, "\n");
		}
	fprintf(file, "\n");
	}



/* prints a matrix in column-major format (exponential notation) */
void PRINT_EXP_MAT(int m, int n, REAL *A, int lda)
	{
	int i, j;
	for(i=0; i<m; i++)
		{
		for(j=0; j<n; j++)
			{
			printf("%e\t", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}



/* prints the transposed of a matrix in column-major format (exponential notation) */
void PRINT_EXP_TRAN_MAT(int row, int col, REAL *A, int lda)
	{
	int i, j;
	for(j=0; j<col; j++)
		{
		for(i=0; i<row; i++)
			{
			printf("%e\t", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
	}
