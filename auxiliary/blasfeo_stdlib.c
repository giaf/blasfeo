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


#include <stdlib.h>
#include <stdio.h>

#ifdef __MABX2__
// dSPACE MicroAutoBox II (32-bit) does not provide stdint
typedef unsigned int uintptr_t;
#else
#include <stdint.h>
#endif

#include <blasfeo_stdlib.h>
#include <blasfeo_block_size.h>
#include <blasfeo_align.h>



#ifdef EXT_DEP_MALLOC
// needed in hp cm routines !!!
void blasfeo_malloc(void **ptr, size_t size)
	{
	*ptr = malloc(size);
	if(*ptr==NULL)
		{
#ifdef EXT_DEP
		printf("Memory allocation error");
#endif
		exit(1);
		}
	return;
	}
#endif



#ifdef EXT_DEP_MALLOC
// allocate memory aligned to typical cache line size (64 bytes)
void blasfeo_malloc_align(void **ptr, size_t size)
	{

#if 1

	void *ptr_raw = NULL;
	blasfeo_malloc(&ptr_raw, size+64);
	uintptr_t ptr_tmp0 = (uintptr_t) ptr_raw;
	uintptr_t ptr_tmp1 = ptr_tmp0 + 1; // make space for at least 1 byte, to store the offset as a char, that is more than enough to cover 64 bytes of alignment
	void *ptr_tmp = (void *) ptr_tmp1;
	blasfeo_align_64_byte(ptr_tmp, ptr);
	ptr_tmp1 = (uintptr_t) *ptr;
	char offset = (char) (ptr_tmp1-ptr_tmp0);
	char *offset_ptr = (char *) ptr_tmp1;
	offset_ptr[-1] = offset;

#else

#if defined(OS_WINDOWS)

	*ptr = _aligned_malloc( size, CACHE_LINE_SIZE );

#elif defined(__DSPACE__)

	// XXX fix this hack !!! (Andrea?)
	*ptr = malloc( size );

#elif(defined __XILINX_NONE_ELF__ || defined __XILINX_ULTRASCALE_NONE_ELF_JAILHOUSE__)

	*ptr = memalign( CACHE_LINE_SIZE, size );

#else

	int err = posix_memalign( ptr, CACHE_LINE_SIZE, size );
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}

#endif

#endif

	return;

	}
#endif



#ifdef EXT_DEP_MALLOC
// needed in hp cm routines !!!
void blasfeo_free(void *ptr)
	{
	free(ptr);
	return;
	}
#endif



#ifdef EXT_DEP_MALLOC
void blasfeo_free_align(void *ptr)
	{

#if 1

	char *offset_ptr = ptr;
	char offset = offset_ptr[-1];
	uintptr_t ptr_tmp0 = (uintptr_t) ptr;
	ptr_tmp0 -= offset;
	free( (void *) ptr_tmp0 );

#else

#if defined(OS_WINDOWS)

	_aligned_free( ptr );

#else

	free( ptr );

#endif

#endif

	return;

	}
#endif
