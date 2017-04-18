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


#ifdef __cplusplus
extern "C" {
#endif



#ifndef BLASFEO_COMMON
#define BLASFEO_COMMON



#if defined(LA_HIGH_PERFORMANCE)

#include "blasfeo_block_size.h"

// matrix structure
struct d_strmat
	{
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	double *pA; // pointer to a pm*pn array of doubles, the first is aligned to cache line size
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memory_size; // size of needed memory
	};

struct s_strmat
	{
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	float *pA; // pointer to a pm*pn array of floats, the first is aligned to cache line size
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int use_dA; // flag to tell if dA can be used
	int memory_size; // size of needed memory
	};

// vector structure
struct d_strvec
	{
	int m; // size
	int pm; // packed size
	double *pa; // pointer to a pm array of doubles, the first is aligned to cache line size
	int memory_size; // size of needed memory
	};

struct s_strvec
	{
	int m; // size
	int pm; // packed size
	float *pa; // pointer to a pm array of floats, the first is aligned to cache line size
	int memory_size; // size of needed memory
	};

#define DMATEL_LIBSTR(sA,ai,aj) ((sA)->pA[(ai-(ai&(D_PS-1)))*(sA)->cn+aj*D_PS+(ai&(D_PS-1))])
#define SMATEL_LIBSTR(sA,ai,aj) ((sA)->pA[(ai-(ai&(S_PS-1)))*(sA)->cn+aj*S_PS+(ai&(S_PS-1))])
#define DVECEL_LIBSTR(sa,ai) ((sa)->pa[ai])
#define SVECEL_LIBSTR(sa,ai) ((sa)->pa[ai])

#elif defined(LA_BLAS) | defined(LA_REFERENCE)

// matrix structure
struct d_strmat
	{
	int m; // rows
	int n; // cols
	double *pA; // pointer to a m*n array of doubles
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memory_size; // size of needed memory
	};

struct s_strmat
	{
	int m; // rows
	int n; // cols
	float *pA; // pointer to a m*n array of floats
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int use_dA; // flag to tell if dA can be used
	int memory_size; // size of needed memory
	};

// vector structure
struct d_strvec
	{
	int m; // size
	double *pa; // pointer to a m array of doubles, the first is aligned to cache line size
	int memory_size; // size of needed memory
	};

struct s_strvec
	{
	int m; // size
	float *pa; // pointer to a m array of floats, the first is aligned to cache line size
	int memory_size; // size of needed memory
	};

#define DMATEL_LIBSTR(sA,ai,aj) ((sA)->pA[ai+aj*(sA)->m])
#define SMATEL_LIBSTR(sA,ai,aj) ((sA)->pA[ai+aj*(sA)->m])
#define DVECEL_LIBSTR(sa,ai) ((sa)->pa[ai])
#define SVECEL_LIBSTR(sa,ai) ((sa)->pa[ai])

#else

#error : wrong LA choice

#endif

#endif  // BLASFEO_COMMON


#ifdef __cplusplus
}
#endif
