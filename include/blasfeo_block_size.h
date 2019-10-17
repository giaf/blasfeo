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

#ifndef BLASFEO_BLOCK_SIZE_H_
#define BLASFEO_BLOCK_SIZE_H_



#if defined( TARGET_X64_INTEL_HASWELL )

#define D_PS 4
#define S_PS 8
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_X64_INTEL_SANDY_BRIDGE )

#define D_PS 4
#define S_PS 8
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_X64_INTEL_CORE )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_X64_AMD_BULLDOZER )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_X86_AMD_JAGUAR )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2
#define S_NC 4 //2

#elif defined( TARGET_X86_AMD_BARCELONA )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2
#define S_NC 4 //2

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)

#define D_PS 4
#define S_PS 4
#define D_NC 4
#define S_NC 4

#elif defined( TARGET_ARMV7A_ARM_CORTEX_A15 )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_ARMV7A_ARM_CORTEX_A7 )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_ARMV7A_ARM_CORTEX_A9 )
// FIXME: these values are just hacked in to make it build
#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#elif defined( TARGET_GENERIC )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#else
#error "Unknown architecture"
#endif


#endif  // BLASFEO_BLOCK_SIZE_H_
