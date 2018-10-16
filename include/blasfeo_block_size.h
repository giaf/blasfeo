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

#elif defined( TARGET_GENERIC )

#define D_PS 4
#define S_PS 4
#define D_NC 4 // 2 // until the smaller kernel is 4x4
#define S_NC 4 //2

#else
#error "Unknown architecture"
#endif


#endif  // BLASFEO_BLOCK_SIZE_H_
