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

#ifndef PLATFORMS_H_
#define PLATFORMS_H_

#ifdef __cplusplus
extern "C" {
#endif

// header for embedded platforms

#if defined(OS_WINDOWS)
#define MEMALIGN(ptrA,size) _aligned_malloc( size, 64 )

#elif defined(__DSPACE__)
// XXX fix this hack !!! (Andrea?)
#define MEMALIGN(ptrA,size) *ptrA = malloc(size);
	
#elif defined(__XILINX_NONE_ELF__)
#define MEMALIGN(ptrA,size) *ptrA = memalign( 64, size )

#else
#define MEMALIGN(ptrA,size) posix_memalign(ptrA, 64, size )

#endif



#ifdef __cplusplus
}
#endif

#endif  //PLATFORMS_H_
