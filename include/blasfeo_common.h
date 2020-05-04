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

#ifndef BLASFEO_COMMON_H_
#define BLASFEO_COMMON_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#if defined(USING_INTEL_COMPILER) || defined(__GCC__) || defined(__clang__)
#define ALIGNED(VEC, BYTES) VEC __attribute__ ((aligned ( BYTES )))
#elif defined (_MSC_VER)
#define ALIGNED(VEC, BYTES) __declspec(align( BYTES )) VEC
#else
#define ALIGNED(VEC, BYTES) VEC
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

#elif defined(LA_EXTERNAL_BLAS_WRAPPER) | defined(LA_REFERENCE)

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
