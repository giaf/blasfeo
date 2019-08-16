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
#include "../include/platforms.h"
#if 0
#include <malloc.h>
#endif


/* creates a zero matrix given the size in bytes */
void v_zeros(void **ptrA, int size)
	{
	// allocate memory
	*ptrA = (void *) malloc(size);
	// zero memory
	int i;
	double *dA = (double *) *ptrA;
	for(i=0; i<size/8; i++) dA[i] = 0.0;
	char *cA = (char *) dA;
	i *= 8;
	for(; i<size; i++) cA[i] = 0;
	return;
	}


/* creates a zero matrix aligned to a cache line given the size in bytes */
void v_zeros_align(void **ptrA, int size)
	{
	// allocate memory
	MEMALIGN(ptrA,size);
	// zero allocated memory
	int i;
	double *dA = (double *) *ptrA;
	for(i=0; i<size/8; i++) dA[i] = 0.0;
	char *cA = (char *) dA;
	i *= 8;
	for(; i<size; i++) cA[i] = 0;
	return;
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
	// allocate memory
	*ptrA = malloc(size);
	// zero memory
	int i;
	double *dA = (double *) *ptrA;
	for(i=0; i<size/8; i++) dA[i] = 0.0;
	char *cA = (char *) dA;
	i *= 8;
	for(; i<size; i++) cA[i] = 0;
	return;
	}


/* creates a zero matrix aligned to a cache line given the size in bytes */
void c_zeros_align(char **ptrA, int size)
	{
	// allocate memory
	MEMALIGN(ptrA,size);
	// zero allocated memory
	int i;
	double *dA = (double *) *ptrA;
	for(i=0; i<size/8; i++) dA[i] = 0.0;
	char *cA = (char *) dA;
	i *= 8;
	for(; i<size; i++) cA[i] = 0;
	return;
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
