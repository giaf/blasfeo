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
#if defined(OS_WINDOWS)
	*pA = (REAL *) _aligned_malloc( (row*col)*sizeof(REAL), 64 );
#else
	void *temp;
	int err = posix_memalign(&temp, 64, (row*col)*sizeof(REAL));
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*pA = temp;
#endif
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
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
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
void PRINT_E_MAT(int m, int n, REAL *A, int lda)
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
void PRINT_E_TRAN_MAT(int row, int col, REAL *A, int lda)
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
