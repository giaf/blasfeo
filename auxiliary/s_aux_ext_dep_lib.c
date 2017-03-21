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

#include <stdlib.h>
#include <stdio.h>
#if 0
#include <malloc.h>
#endif

#include "../include/blasfeo_common.h"



#if ! defined(OS_WINDOWS)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif



/* creates a zero matrix */
void s_zeros(float **pA, int row, int col)
	{
	*pA = malloc((row*col)*sizeof(float));
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
void s_free_align(float *pA)
	{
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
	}



/* prints a matrix in column-major format */
void s_print_mat(int m, int n, float *A, int lda)
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
	}	



/* prints the transposed of a matrix in column-major format */
void s_print_tran_mat(int row, int col, float *A, int lda)
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
void s_print_to_file_mat(FILE *file, int row, int col, float *A, int lda)
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



/* prints the transposed of a matrix in column-major format */
void s_print_tran_to_file_mat(FILE *file, int row, int col, float *A, int lda)
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
void s_print_e_mat(int m, int n, float *A, int lda)
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
void s_print_e_tran_mat(int row, int col, float *A, int lda)
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



/****************************
* new interface
****************************/

#if defined(LA_HIGH_PERFORMANCE)



#include "../include/blasfeo_block_size.h"



// create a matrix structure for a matrix of size m*n by dynamically allocating the memory
void s_allocate_strmat(int m, int n, struct s_strmat *sA)
	{
	const int bs = S_BS;
	int nc = S_NC;
	int al = bs*nc;
	sA->m = m;
	sA->n = n;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	s_zeros_align(&(sA->pA), sA->pm, sA->cn);
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	s_zeros_align(&(sA->dA), tmp, 1);
	sA->use_dA = 0;
	sA->memory_size = (pm*cn+tmp)*sizeof(float);
	return;
	}



// free memory of a matrix structure
void s_free_strmat(struct s_strmat *sA)
	{
	s_free_align(sA->pA);
	s_free_align(sA->dA);
	return;
	}



// create a vector structure for a vector of size m by dynamically allocating the memory
void s_allocate_strvec(int m, struct s_strvec *sa)
	{
	const int bs = S_BS;
//	int nc = S_NC;
//	int al = bs*nc;
	sa->m = m;
	int pm = (m+bs-1)/bs*bs;
	sa->pm = pm;
	s_zeros_align(&(sa->pa), sa->pm, 1);
	sa->memory_size = pm*sizeof(float);
	return;
	}



// free memory of a matrix structure
void s_free_strvec(struct s_strvec *sa)
	{
	s_free_align(sa->pa);
	return;
	}



// print a matrix structure
void s_print_strmat(int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = S_BS;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%bs>0)
		{
		tmp = bs-ai%bs;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+bs*j]);
				}
			printf("\n");
			}
		pA += tmp + bs*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(bs-1); ii+=bs)
		{
		for(i=0; i<bs; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
	return;
	}



// print a vector structure
void s_print_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_mat(m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void s_print_tran_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_mat(1, m, pa, 1);
	return;
	}



// print a matrix structure
void s_print_to_file_strmat(FILE * file, int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = S_BS;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%bs>0)
		{
		tmp = bs-ai%bs;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+bs*j]);
				}
			fprintf(file, "\n");
			}
		pA += tmp + bs*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(bs-1); ii+=bs)
		{
		for(i=0; i<bs; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+bs*j+sda*ii]);
				}
			fprintf(file, "\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+bs*j+sda*ii]);
				}
			fprintf(file, "\n");
			}
		}
	fprintf(file, "\n");
	return;
	}



// print a vector structure
void s_print_to_file_strvec(FILE * file, int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_to_file_mat(file, m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void s_print_tran_to_file_strvec(FILE * file, int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_to_file_mat(file, 1, m, pa, 1);
	return;
	}



// print a matrix structure
void s_print_e_strmat(int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = S_BS;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%bs>0)
		{
		tmp = bs-ai%bs;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+bs*j]);
				}
			printf("\n");
			}
		pA += tmp + bs*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(bs-1); ii+=bs)
		{
		for(i=0; i<bs; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+bs*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
	return;
	}



// print a vector structure
void s_print_e_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_e_mat(m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void s_print_e_tran_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_e_mat(1, m, pa, 1);
	return;
	}



#elif defined(LA_BLAS) | defined(LA_REFERENCE)



// create a matrix structure for a matrix of size m*n
void s_allocate_strmat(int m, int n, struct s_strmat *sA)
	{
	sA->m = m;
	sA->n = n;
	s_zeros(&(sA->pA), sA->m, sA->n);
#if defined(LA_REFERENCE)
	int tmp = m<n ? m : n; // al(min(m,n)) // XXX max ???
	s_zeros(&(sA->dA), tmp, 1);
	sA->memory_size = (m*n+tmp)*sizeof(float);
#else
	sA->memory_size = (m*n)*sizeof(float);
#endif
	return;
	}



// free memory of a matrix structure
void s_free_strmat(struct s_strmat *sA)
	{
	free(sA->pA);
#if defined(LA_REFERENCE)
	free(sA->dA);
#endif
	return;
	}



// create a vector structure for a vector of size m
void s_allocate_strvec(int m, struct s_strvec *sa)
	{
	sa->m = m;
	s_zeros(&(sa->pa), sa->m, 1);
	sa->memory_size = m*sizeof(float);
	return;
	}



// free memory of a vector structure
void s_free_strvec(struct s_strvec *sa)
	{
	free(sa->pa);
	return;
	}



// print a matrix structure
void s_print_strmat(int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	float *pA = sA->pA + ai + aj*lda;
	s_print_mat(m, n, pA, lda);
	return;
	}



// print a vector structure
void s_print_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_mat(m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void s_print_tran_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_mat(1, m, pa, 1);
	return;
	}



// print a matrix structure
void s_print_to_file_strmat(FILE *file, int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	float *pA = sA->pA + ai + aj*lda;
	s_print_to_file_mat(file, m, n, pA, lda);
	return;
	}



// print a vector structure
void s_print_to_file_strvec(FILE *file, int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_to_file_mat(file, m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void s_print_to_file_tran_strvec(FILE *file, int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_to_file_mat(file, 1, m, pa, 1);
	return;
	}



// print a matrix structure
void s_print_e_strmat(int m, int n, struct s_strmat *sA, int ai, int aj)
	{
	int lda = sA->m;
	float *pA = sA->pA + ai + aj*lda;
	s_print_e_mat(m, n, pA, lda);
	return;
	}



// print a vector structure
void s_print_e_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_e_mat(m, 1, pa, m);
	return;
	}



// print and transpose a vector structure
void s_print_e_tran_strvec(int m, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	s_print_e_mat(1, m, pa, 1);
	return;
	}



#else

#error : wrong LA choice

#endif



