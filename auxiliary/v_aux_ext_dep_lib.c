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



/* creates a zero matrix given the size in bytes */
void v_zeros(void **ptrA, int size)
	{
	*ptrA = (void *) malloc(size);
	char *A = *ptrA;
	int i;
	for(i=0; i<size; i++) A[i] = 0;
	}


/* creates a zero matrix aligned to a cache line given the size in bytes */
void v_zeros_align(void **ptrA, int size)
	{
#if defined(OS_WINDOWS)
	*ptrA = _aligned_malloc( size, 64 );
#elif defined(__DSPACE__)
	*ptrA = malloc(size);
#else
	int err = posix_memalign(ptrA, 64, size);
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
#endif
	char *A = *ptrA;
	int i;
	for(i=0; i<size; i++) A[i] = 0;
	}


/* frees matrix */
void v_free(void *pA)
	{
	free( pA );
	}


/* frees aligned matrix */
void v_free_align(void *pA)
	{
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
	}


/* creates a zero matrix given the size in bytes */
void c_zeros(char **ptrA, int size)
	{
	*ptrA = malloc(size);
	char *A = *ptrA;
	int i;
	for(i=0; i<size; i++) A[i] = 0;
	}


/* creates a zero matrix aligned to a cache line given the size in bytes */
void c_zeros_align(char **ptrA, int size)
	{
#if defined(OS_WINDOWS)
	*ptrA = _aligned_malloc( size, 64 );
#elif defined(__DSPACE__)
	*ptrA = malloc(size);
#else
	void *temp;
	int err = posix_memalign(&temp, 64, size);
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*ptrA = temp;
#endif
	char *A = *ptrA;
	int i;
	for(i=0; i<size; i++) A[i] = 0;
	}


/* frees matrix */
void c_free(char *pA)
	{
	free( pA );
	}


/* frees aligned matrix */
void c_free_align(char *pA)
	{
#if defined(OS_WINDOWS)
	_aligned_free( pA );
#else
	free( pA );
#endif
	}
