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

#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif



//             r0 r1 r2 r3 sp+0 sp+4 sp+8
// void kernel()

//	.p2align 4,,15
#if defined(OS_LINUX)
	.global	kernel
	.type	kernel, %function
kernel:
#elif defined(OS_MAC)
	.global	_kernel
_kernel:
#endif

	// prologue

	// save GP registers
	stmdb	sp!, {r4 - r10, fp, lr} // save registers
	add		fp, sp, #36 // fp to old sp position

	// save FP registers
	fstmfdd	sp!, {d8-d15}

	// TODO

	// epilogue

	// load FP registers
	fldmfdd	sp!, {d8-d15}

	// load GP registers and return
//	ldmia	sp!, {r4 - r10, fp, lr} // load registers
//	mov		pc, lr // return
	ldmia	sp!, {r4 - r10, fp, pc} // load registers and return

#if defined(OS_LINUX)
	.size	kernel, .-kernel
#endif



