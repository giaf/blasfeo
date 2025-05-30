/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/



#ifdef EXT_DEP_MALLOC
/* creates a zero matrix */
void ZEROS(REAL **pA, int row, int col)
	{
	blasfeo_malloc((void **) pA, (row*col)*sizeof(REAL));
	REAL *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}
#endif



#ifdef EXT_DEP_MALLOC
/* creates a zero matrix aligned to a cache line */
void ZEROS_ALIGN(REAL **pA, int row, int col)
	{
    blasfeo_malloc_align((void **) pA, (row*col)*sizeof(REAL));
	REAL *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}
#endif



#ifdef EXT_DEP_MALLOC
/* frees matrix */
void FREE(REAL *pA)
	{
	blasfeo_free( pA );
	}
#endif



#ifdef EXT_DEP_MALLOC
/* frees aligned matrix */
void FREE_ALIGN(REAL *pA)
	{
	blasfeo_free_align(pA);
	}
#endif



/* prints a matrix in column-major format */
void PRINT_MAT(int m, int n, REAL *A, int lda)
	{
#ifdef EXT_DEP
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
#endif
	return;
	}



/* prints the transposed of a matrix in column-major format */
void PRINT_TRAN_MAT(int row, int col, REAL *A, int lda)
	{
#ifdef EXT_DEP
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
#endif
	}



#ifdef EXT_DEP
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
#endif

#ifdef EXT_DEP
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
#endif


#ifdef EXT_DEP
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
#endif



#ifdef EXT_DEP
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
#endif



/* prints a matrix in column-major format (exponential notation) */
void PRINT_EXP_MAT(int m, int n, REAL *A, int lda)
	{
#ifdef EXT_DEP
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
#endif
	}



/* prints the transposed of a matrix in column-major format (exponential notation) */
void PRINT_EXP_TRAN_MAT(int row, int col, REAL *A, int lda)
	{
#ifdef EXT_DEP
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
#endif
	}
