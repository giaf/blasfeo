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

#ifndef BLASFEO_COMMON_H_
#define BLASFEO_COMMON_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#if defined(LA_HIGH_PERFORMANCE)

#include "blasfeo_block_size.h"

// matrix structure
struct blasfeo_dmat
	{
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	double *pA; // pointer to a pm*pn array of doubles, the first is aligned to cache line size
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_smat
	{
	int m; // rows
	int n; // cols
	int pm; // packed number or rows
	int cn; // packed number or cols
	float *pA; // pointer to a pm*pn array of floats, the first is aligned to cache line size
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

// vector structure
struct blasfeo_dvec
	{
	int m; // size
	int pm; // packed size
	double *pa; // pointer to a pm array of doubles, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

struct blasfeo_svec
	{
	int m; // size
	int pm; // packed size
	float *pa; // pointer to a pm array of floats, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

#define BLASFEO_DMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(D_PS-1)))*(sA)->cn+(aj)*D_PS+((ai)&(D_PS-1))])
#define BLASFEO_SMATEL(sA,ai,aj) ((sA)->pA[((ai)-((ai)&(S_PS-1)))*(sA)->cn+(aj)*S_PS+((ai)&(S_PS-1))])
#define BLASFEO_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_SVECEL(sa,ai) ((sa)->pa[ai])

#elif defined(LA_BLAS_WRAPPER) | defined(LA_REFERENCE)

// matrix structure
struct blasfeo_dmat
	{
	int m; // rows
	int n; // cols
	double *pA; // pointer to a m*n array of doubles
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_smat
	{
	int m; // rows
	int n; // cols
	float *pA; // pointer to a m*n array of floats
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

// vector structure
struct blasfeo_dvec
	{
	int m; // size
	double *pa; // pointer to a m array of doubles, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

struct blasfeo_svec
	{
	int m; // size
	float *pa; // pointer to a m array of floats, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

#define BLASFEO_DMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_SMATEL(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define BLASFEO_DVECEL(sa,ai) ((sa)->pa[ai])
#define BLASFEO_SVECEL(sa,ai) ((sa)->pa[ai])

#else

#error : wrong LA choice

#endif



#if defined(TESTING_MODE)

// matrix structure
struct blasfeo_dmat_ref
	{
	int m; // rows
	int n; // cols
	double *pA; // pointer to a m*n array of doubles
	double *dA; // pointer to a min(m,n) (or max???) array of doubles
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

struct blasfeo_smat_ref
	{
	int m; // rows
	int n; // cols
	float *pA; // pointer to a m*n array of floats
	float *dA; // pointer to a min(m,n) (or max???) array of floats
	int use_dA; // flag to tell if dA can be used
	int memsize; // size of needed memory
	};

// vector structure
struct blasfeo_dvec_ref
	{
	int m; // size
	double *pa; // pointer to a m array of doubles, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

struct blasfeo_svec_ref
	{
	int m; // size
	float *pa; // pointer to a m array of floats, the first is aligned to cache line size
	int memsize; // size of needed memory
	};

#define MATEL_REF(sA,ai,aj) ((sA)->pA[(ai)+(aj)*(sA)->m])
#define VECEL_REF(sa,ai) ((sa)->pa[ai])

#endif // TESTING_MODE

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_COMMON_H_
