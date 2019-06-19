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

// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NT_4X8_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x8_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	56(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x8_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NT_4X7_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x7_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	48(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x7_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NT_4X6_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x6_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	40(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x6_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NT_4X5_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x5_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	32(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x5_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NN_4X8_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x8_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+3*ldb
	movq	%r15, %rax
	addq	%r13, %rax
	addq	%r13, %rax
	addq	%r13, %rax // B+6*ldb

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	0(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	8(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	16(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$32, %rax
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	0(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	8(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	16(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$32, %rax
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	0(%rax, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$8, %rax
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x8_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NN_4X7_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x7_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+3*ldb
	movq	%r15, %rax
	addq	%r13, %rax
	addq	%r13, %rax
	addq	%r13, %rax // B+6*ldb

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$32, %rax
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$32, %rax
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	0(%rax), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$8, %rax
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x7_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NN_4X6_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x6_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+3*ldb

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	16(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	0(%r15, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x6_lib4c)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NN_4X5_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x5_lib4c)
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+3*ldb

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	8(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r12, %r13, 2), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x5_lib4c)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_4x8_lib)
#endif

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3
	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x8_lib)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc
// r14d   <- km
// r15d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X8_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_4x8_vs_lib)
#endif

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3
	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0
	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end


	vcvtsi2sd	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm13, %ymm13


	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	addq		%r13, %r12
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	addq		%r13, %r12
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	addq		%r13, %r12
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3
	addq		%r13, %r12
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	addq		%r13, %r12
	cmpl		$6, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	addq		%r13, %r12
	cmpl		$7, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	addq		%r13, %r12
	cmpl		$7, %r15d
	je			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x8_vs_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_4x8_lib)
#endif

	vmovupd		%ymm0, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm1, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm2, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm3, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm4, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm5, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm6, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm7, 0(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4x8_lib)
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10   <- D
// r11  <- ldd
// r12d   <- km
// r13d   <- kn
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X8_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_4x8_vs_lib)
#endif
	
	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	vmaskmovpd	%ymm0, %ymm15, 0(%r10)
	addq		%r11, %r10
	vmaskmovpd	%ymm1, %ymm15, 0(%r10)
	addq		%r11, %r10
	vmaskmovpd	%ymm2, %ymm15, 0(%r10)
	addq		%r11, %r10
	vmaskmovpd	%ymm3, %ymm15, 0(%r10)
	addq		%r11, %r10
	vmaskmovpd	%ymm4, %ymm15, 0(%r10)
	addq		%r11, %r10
	cmpl		$6, %r13d
	jl			0f // end
	vmaskmovpd	%ymm5, %ymm15, 0(%r10)
	addq		%r11, %r10
	cmpl		$7, %r13d
	jl			0f // end
	vmaskmovpd	%ymm6, %ymm15, 0(%r10)
	addq		%r11, %r10
	cmpl		$7, %r13d
	je			0f // end
	vmaskmovpd	%ymm7, %ymm15, 0(%r10)
//	addq	%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4x8_vs_lib)
#endif





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nn_4x8_lib4ccc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_4x8_lib4ccc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X8_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x8_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X8_LIB
#else
	CALL(inner_scale_ab_4x8_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X8_LIB
#else
	CALL(inner_store_4x8_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_4x8_lib4ccc)





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nn_4x8_vs_lib4ccc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_4x8_vs_lib4ccc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

	movq	ARG12, %r14  // n1
	cmpl	$5, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X5_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x5_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$6, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X6_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x6_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$7, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X7_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x7_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X8_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x8_lib4c)
#endif

103:


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X8_VS_LIB
#else
	CALL(inner_scale_ab_4x8_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X8_VS_LIB
#else
	CALL(inner_store_4x8_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_4x8_vs_lib4ccc)





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nt_4x8_lib4ccc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x8_lib4ccc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X8_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x8_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X8_LIB
#else
	CALL(inner_scale_ab_4x8_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X8_LIB
#else
	CALL(inner_store_4x8_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x8_lib4ccc)





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nt_4x8_vs_lib4ccc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x8_vs_lib4ccc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d


	movq	ARG12, %r14  // n1
	cmpl	$5, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X5_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x5_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$6, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X6_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x6_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$7, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X7_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x7_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X8_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x8_lib4c)
#endif

103:


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X8_VS_LIB
#else
	CALL(inner_scale_ab_4x8_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X8_VS_LIB
#else
	CALL(inner_store_4x8_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x8_vs_lib4ccc)






