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
	.macro INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5
	vmovapd			%ymm4, %ymm6
	vmovapd			%ymm4, %ymm7

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1
	vaddpd			%ymm6, %ymm2, %ymm2
	vaddpd			%ymm7, %ymm3, %ymm3

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x4_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5
	vmovapd			%ymm4, %ymm6

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1
	vaddpd			%ymm6, %ymm2, %ymm2

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x3_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x2_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_4x1_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nt_4x1_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5
	vmovapd			%ymm4, %ymm6
	vmovapd			%ymm4, %ymm7

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1
	vaddpd			%ymm6, %ymm2, %ymm2
	vaddpd			%ymm7, %ymm3, %ymm3

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x4_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5
	vmovapd			%ymm4, %ymm6

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1
	vaddpd			%ymm6, %ymm2, %ymm2

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x3_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4
	vmovapd			%ymm4, %ymm5

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0
	vaddpd			%ymm5, %ymm1, %ymm1

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x2_lib4c)
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
	.macro INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_4x1_lib4c)
#endif

	cmpl	$0, %r10d
	jle		5f // return

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// preload

	vxorpd			%ymm4, %ymm4, %ymm4

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

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
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

	vaddpd			%ymm4, %ymm0, %ymm0

5: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_4x1_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_rl_4x4_lib4c)
#endif
	
	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_rl_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_rl_4x4_vs_lib4c)
#endif
	
	cmpl	$0, %r10d
	jle		0f // end

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_rl_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_ONE_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_rl_one_4x4_lib4c)
#endif
	
	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vaddpd			%ymm1, %ymm13, %ymm1

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vaddpd			%ymm2, %ymm13, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	24(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vaddpd			%ymm3, %ymm13, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_rl_one_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_ONE_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_rl_one_4x4_vs_lib4c)
#endif
	
	cmpl	$0, %r10d
	jle		0f // end

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vaddpd			%ymm1, %ymm13, %ymm1

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vaddpd			%ymm2, %ymm13, %ymm2

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vaddpd			%ymm3, %ymm13, %ymm3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_rl_one_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RL_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_rl_4x4_lib4c)
#endif
	
	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_rl_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B lower triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// r14   <- n1
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RL_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_rl_4x4_vs_lib4c)
#endif
	
	cmpl	$0, %r14d
	jle		0f // end

	cmpl	$4, %r14d
	jl		1f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq	%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

	jmp		0f

1:

	cmpl	$3, %r14d
	jl		2f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq	%r13, %r12

	// unroll 3

	subl	$3, %r10d
	addq	$96, %r11

	jmp		0f

2:

	cmpl	$2, %r14d
	jl		3f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq	%r13, %r12

	// unroll 2

	// unroll 3

	subl	$2, %r10d
	addq	$64, %r11

	jmp		0f

3:

//	cmpl	$1, %r14d
//	jl		0f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq	%r13, %r12

	// unroll 1

	// unroll 2

	// unroll 3

	subl	$1, %r10d
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_rl_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RU_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_ru_4x4_lib4c)
#endif
	
	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	8(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	16(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_ru_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// r14   <- n1
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RU_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_ru_4x4_vs_lib4c)
#endif
	
	cmpl	$0, %r14d
	jle		0f // end

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	cmpl	$4, %r14d
	jl		1f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	8(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	16(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	jmp		0f

1:

	cmpl	$3, %r14d
	jl		2f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 3

	subl	$3, %r10d
	addq	$24, %r12
	addq	$24, %r15
	addq	$96, %r11

	jmp		0f

2:

	cmpl	$2, %r14d
	jl		3f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 2

	// unroll 3

	subl	$2, %r10d
	addq	$16, %r12
	addq	$16, %r15
	addq	$64, %r11

	jmp		0f

3:

//	cmpl	$1, %r14d
//	jl		0f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	// unroll 1

	// unroll 2

	// unroll 3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_ru_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RU_ONE_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_ru_one_4x4_lib4c)
#endif
	
	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vaddpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	8(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vaddpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vaddpd			%ymm3, %ymm13, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_ru_one_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- ldb
// r14   <- n1
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RU_ONE_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nn_ru_one_4x4_vs_lib4c)
#endif
	
	cmpl	$0, %r14d
	jle		0f // end

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	cmpl	$4, %r14d
	jl		1f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	0(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vaddpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	8(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vaddpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r15, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vaddpd			%ymm3, %ymm13, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

	jmp		0f

1:

	cmpl	$3, %r14d
	jl		2f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vaddpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r15), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vaddpd			%ymm2, %ymm13, %ymm2

	// unroll 3

	subl	$3, %r10d
	addq	$24, %r12
	addq	$24, %r15
	addq	$96, %r11

	jmp		0f

2:

	cmpl	$2, %r14d
	jl		3f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r12, %r13), %ymm12 // B
	vmulpd			%ymm13, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vaddpd			%ymm1, %ymm13, %ymm1

	// unroll 2

	// unroll 3

	subl	$2, %r10d
	addq	$16, %r12
	addq	$16, %r15
	addq	$64, %r11

	jmp		0f

3:

//	cmpl	$1, %r14d
//	jl		0f // end

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vaddpd			%ymm0, %ymm13, %ymm0

	// unroll 1

	// unroll 2

	// unroll 3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nn_ru_one_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- kmax
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]

//
// output arguments:


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_ru_4x4_lib4c)
#endif

	vmovapd			0(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			%r13, %r12

	vmovapd			32(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			%r13, %r12

	vmovapd			64(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			%r13, %r12

	vmovapd			96(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq			%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_ru_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
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


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_ru_4x4_vs_lib4c)
#endif

	cmpl	$0, %r10d
	jle		0f // end

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			%r13, %r12
	addq			$32, %r11

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			%r13, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			%r13, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq			%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_ru_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- kmax
// r11   <- A
// r12   <- B
// r13   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]

//
// output arguments:


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_ONE_4X4_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_ru_one_4x4_lib4c)
#endif

	vmovapd			0(%r11), %ymm8
	vaddpd			%ymm0, %ymm8, %ymm0
	addq			%r13, %r12

	vmovapd			32(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vaddpd			%ymm1, %ymm8, %ymm1
	addq			%r13, %r12

	vmovapd			64(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vaddpd			%ymm2, %ymm8, %ymm2
	addq			%r13, %r12

	vmovapd			96(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vaddpd			%ymm3, %ymm8, %ymm3
	addq			%r13, %r12

	subl	$4, %r10d
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_ru_one_4x4_lib4c)
#endif





// common inner routine with file scope
//
// edge for B upper triangular
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


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_ONE_4X4_VS_LIB4C
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrmm_nt_ru_one_4x4_vs_lib4c)
#endif

	cmpl	$0, %r10d
	jle		0f // end

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vaddpd			%ymm0, %ymm8, %ymm0
	addq			$32, %r11
	addq			%r13, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vaddpd			%ymm1, %ymm8, %ymm1
	addq			%r13, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vaddpd			%ymm2, %ymm8, %ymm2
	addq			%r13, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	addq			$32, %r11
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vaddpd			%ymm3, %ymm8, %ymm3
	addq			%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrmm_nt_ru_one_4x4_vs_lib4c)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = lower
// tran = not-transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_lln_one_4x4_lib)
#endif

	vxorpd		%ymm14, %ymm14, %ymm14

	vmovupd		0(%r10), %ymm12
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	add			%r11, %r10

	vmovupd		0(%r10), %ymm12
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	add			%r11, %r10

	vmovupd		0(%r10), %ymm12
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x11, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_lln_one_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = not-transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLN_INV_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rln_inv_4x4_lib)
#endif
	
	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rln_inv_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = not-transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// r13  <- n1
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLN_INV_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rln_inv_4x4_vs_lib)
#endif
	
	cmpl			$3, %r13d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$2, %r13d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$1, %r13d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rln_inv_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = not-transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLN_ONE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rln_one_4x4_lib)
#endif
	
	// 4th column
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 3rd column
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 2nd column
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rln_one_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = not-transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- n1
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLN_ONE_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rln_one_4x4_vs_lib)
#endif
	
	cmpl			$3, %r12d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$2, %r12d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$1, %r12d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rln_one_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rlt_inv_4x4_lib)
#endif
	
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rlt_inv_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// r13d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rlt_inv_4x4_vs_lib)
#endif
	
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	cmpl			$2, %r13d
	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	cmpl			$3, %r13d
	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	cmpl			$4, %r13d
	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rlt_inv_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rlt_one_4x4_lib)
#endif
	
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rlt_one_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rlt_one_4x4_vs_lib)
#endif
	
	cmpl			$2, %r12d
	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	cmpl			$3, %r12d
	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	cmpl			$4, %r12d
	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rlt_one_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = not-transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_INV_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_run_inv_4x4_lib)
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_run_inv_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = not-transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// r13d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_INV_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_run_inv_4x4_vs_lib)
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	cmpl			$2, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	cmpl			$3, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	cmpl			$4, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_run_inv_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = not-transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_ONE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_run_one_4x4_lib)
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_run_one_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = not-transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_ONE_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_run_one_4x4_vs_lib)
#endif
	
	addq	%r11, %r10

	cmpl			$2, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	cmpl			$3, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	cmpl			$4, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_run_one_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rut_inv_4x4_lib)
#endif
	
	addq	%r11, %r10

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rut_inv_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// r13  <- n1
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rut_inv_4x4_vs_lib)
#endif
	
	addq	%r11, %r10

	cmpl			$3, %r13d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$2, %r13d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$1, %r13d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rut_inv_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_ONE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rut_one_4x4_lib)
#endif
	
	addq	%r11, %r10

	// 4th column
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 3rd column
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 2nd column
	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rut_one_4x4_lib)
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- n1
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_ONE_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsm_rut_one_4x4_vs_lib)
#endif
	
	addq	%r11, %r10

	cmpl			$3, %r12d
	jle				0f

	// 4th column
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$2, %r12d
	jle				0f

	// 3rd column
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$1, %r12d
	jle				0f

	// 2nd column
	vbroadcastsd	0(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsm_rut_one_4x4_vs_lib)
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
	.macro INNER_BLEND_SCALE_AB_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_ab_4x4_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_ab_4x4_lib)
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
	.macro INNER_BLEND_SCALE_AB_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_ab_4x4_vs_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

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
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	cmpl		$2, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	cmpl		$3, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	cmpl		$3, %r15d
	je			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_ab_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and generic beta
//
// input arguments:
// r10   <- beta
// r11   <- C
// r12   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_M1B_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_m1b_4x4_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	// beta
	vbroadcastsd	0(%r10), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_m1b_4x4_lib)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- beta
// r11   <- C
// r12   <- ldc
// r13d   <- km
// r14d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_M1B_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_m1b_4x4_vs_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	// beta
	vbroadcastsd	0(%r10), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0
	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end


	vcvtsi2sd	%r13d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm13, %ymm13


	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r12, %r11
	cmpl		$2, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r12, %r11
	cmpl		$3, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r12, %r11
	cmpl		$3, %r14d
	je			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_m1b_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_M11_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_m11_4x4_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_m11_4x4_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// r12d   <- km
// r13d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_M11_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_blend_scale_m11_4x4_vs_lib)
#endif

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm13, %ymm13


	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r11, %r10
	cmpl		$2, %r13d
	jl			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r11, %r10
	cmpl		$3, %r13d
	jl			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r11, %r10
	cmpl		$3, %r13d
	je			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_blend_scale_m11_4x4_vs_lib)
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
	.macro INNER_SCALE_AB_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_4x4_lib)
#endif

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x4_lib)
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
	.macro INNER_SCALE_AB_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_4x4_vs_lib)
#endif

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

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
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	cmpl		$2, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	cmpl		$3, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	cmpl		$3, %r15d
	je			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// tran_scale for generic alpha and beta
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
	.macro INNER_TRAN_SCALE_AB_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_tran_scale_ab_4x4_lib)
#endif

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpckhpd	%ymm1, %ymm0, %ymm13
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm3, %ymm2, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm3

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_tran_scale_ab_4x4_lib)
#endif





// common inner routine with file scope
//
// tran scale for generic alpha and beta
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
	.macro INNER_TRAN_SCALE_AB_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_tran_scale_ab_4x4_vs_lib)
#endif

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpckhpd	%ymm1, %ymm0, %ymm13
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm3, %ymm2, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm3

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

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
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		%r13, %r12
	cmpl		$2, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		%r13, %r12
	cmpl		$3, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	addq		%r13, %r12
	cmpl		$3, %r15d
	je			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_tran_scale_ab_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and generic beta
//
// input arguments:
// r10   <- beta
// r11   <- C
// r12   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_m1b_4x4_lib)
#endif

	// beta
	vbroadcastsd	0(%r10), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_4x4_lib)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- beta
// r11   <- C
// r12   <- ldc
// r13d   <- km
// r14d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_m1b_4x4_vs_lib)
#endif

	// beta
	vbroadcastsd	0(%r10), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0
	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end


	vcvtsi2sd	%r13d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm13, %ymm13


	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r12, %r11
	cmpl		$2, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r12, %r11
	cmpl		$3, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r12, %r11
	cmpl		$3, %r14d
	je			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vmulpd		%ymm14, %ymm15, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_m11_4x4_lib)
#endif

	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_4x4_lib)
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// r12d   <- km
// r13d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_m11_4x4_vs_lib)
#endif

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm13, %ymm13


	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	addq		%r11, %r10
	cmpl		$2, %r13d
	jl			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	addq		%r11, %r10
	cmpl		$3, %r13d
	jl			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	addq		%r11, %r10
	cmpl		$3, %r13d
	je			0f // end
	vmaskmovpd	0(%r10), %ymm13, %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_4x4_vs_lib)
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
	.macro INNER_STORE_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_4x4_lib)
#endif

	vmovupd		%ymm0, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm1, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm2, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm3, 0(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4x4_lib)
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
	.macro INNER_STORE_L_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_l_4x4_lib)
#endif

	vmovupd		%ymm0, 0(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x1, %ymm15, %ymm1, %ymm1
	vmovupd		%ymm1, 0(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x3, %ymm15, %ymm2, %ymm2
	vmovupd		%ymm2, 0(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x7, %ymm15, %ymm3, %ymm3
	vmovupd		%ymm3, 0(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_4x4_lib)
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
	.macro INNER_STORE_U_4X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_u_4x4_lib)
#endif

	vmovupd		0(%r10), %ymm15
	vblendpd	$0x1, %ymm0, %ymm15, %ymm0
	vmovupd		%ymm0, 0(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x3, %ymm1, %ymm15, %ymm1
	vmovupd		%ymm1, 0(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x7, %ymm2, %ymm15, %ymm2
	vmovupd		%ymm2, 0(%r10)
	addq		%r11, %r10
	vmovupd		%ymm3, 0(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_u_4x4_lib)
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
	.macro INNER_STORE_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_4x4_vs_lib)
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
	cmpl		$2, %r13d
	jl			0f // end
	vmaskmovpd	%ymm1, %ymm15, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	jl			0f // end
	vmaskmovpd	%ymm2, %ymm15, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	je			0f // end
	vmaskmovpd	%ymm3, %ymm15, 0(%r10)
//	addq	%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// r12d   <- km
// r13d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_l_4x4_vs_lib)
#endif

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm14

	vmaskmovpd	%ymm0, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$2, %r13d
	jl			0f // end
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x1, %ymm15, %ymm1, %ymm1
	vmaskmovpd	%ymm1, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	jl			0f // end
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x3, %ymm15, %ymm2, %ymm2
	vmaskmovpd	%ymm2, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	je			0f // end
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x7, %ymm15, %ymm3, %ymm3
	vmaskmovpd	%ymm3, %ymm14, 0(%r10)
//	addq	%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_4x4_vs_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// r12d   <- km
// r13d   <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_U_4X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_u_4x4_vs_lib)
#endif

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm14

	vmovupd		0(%r10), %ymm15
	vblendpd	$0x1, %ymm0, %ymm15, %ymm0
	vmaskmovpd	%ymm0, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$2, %r13d
	jl			0f // end
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x3, %ymm1, %ymm15, %ymm1
	vmaskmovpd	%ymm1, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	jl			0f // end
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x7, %ymm2, %ymm15, %ymm2
	vmaskmovpd	%ymm2, %ymm14, 0(%r10)
	addq		%r11, %r10
	cmpl		$3, %r13d
	je			0f // end
	vmaskmovpd	%ymm3, %ymm14, 0(%r10)
//	addq	%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_u_4x4_vs_lib)
#endif





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nt_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib4cc)

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
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nt_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_lib4cc)

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
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_vs_lib4cc)





//                                 1      2              3          4          5             6          7        8          9
// void kernel_dgemm_nt_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB
#else
	CALL(inner_blend_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_lib44c)





//                                    1      2              3          4          5             6          7        8          9        10      11
// void kernel_dgemm_nt_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d
	movq	ARG10, %r14 // m1
	movq	ARG11, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_blend_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12 // m1
	movq	ARG11, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_vs_lib44c)





//                                 1      2              3          4        5          6             7          8        9          10
// void kernel_dgemm_nt_4x4_libc4c(int k, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_libc4c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG5, %r11  // B
	movq	ARG3, %r12  // A
	movq	ARG4, %r13  // lda
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_TRAN_SCALE_AB_4X4_LIB
#else
	CALL(inner_tran_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_libc4c)





//                                    1      2              3          4        5          6             7          8        9          10       11      12
// void kernel_dgemm_nt_4x4_vs_libc4c(int k, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_libc4c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG5, %r11  // B
	movq	ARG3, %r12  // A
	movq	ARG4, %r13  // lda
	sall	$3, %r13d

	// TODO kernel ?x4 ???

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_TRAN_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_tran_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_4x4_vs_libc4c)





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nn_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_4x4_lib4cc)

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
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_4x4_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nn_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_4x4_vs_lib4cc)

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
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_4x4_vs_lib4cc)





//                                   1      2              3          4          5             6          7        8          9
// void kernel_dsyrk_nt_l_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dsyrk_nt_l_4x4_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB
#else
	CALL(inner_blend_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB
#else
	CALL(inner_store_l_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dsyrk_nt_l_4x4_lib44c)





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dsyrk_nt_l_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dsyrk_nt_l_4x4_vs_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d
	movq	ARG10, %r14 // m1
	movq	ARG11, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_blend_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12 // m1
	movq	ARG11, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB
#else
	CALL(inner_store_l_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dsyrk_nt_l_4x4_vs_lib44c)





//                                   1      2              3          4          5             6          7        8          9
// void kernel_dsyrk_nt_u_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dsyrk_nt_u_4x4_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB
#else
	CALL(inner_blend_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_U_4X4_LIB
#else
	CALL(inner_store_u_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dsyrk_nt_u_4x4_lib44c)





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dsyrk_nt_u_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dsyrk_nt_u_4x4_vs_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d
	movq	ARG10, %r14 // m1
	movq	ARG11, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_blend_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12 // m1
	movq	ARG11, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_U_4X4_VS_LIB
#else
	CALL(inner_store_u_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dsyrk_nt_u_4x4_vs_lib44c)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nn_rl_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_rl_4x4_lib4cc)

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
	INNER_EDGE_DTRMM_NN_RL_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_rl_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_rl_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nn_rl_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_rl_4x4_vs_lib4cc)

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
	INNER_EDGE_DTRMM_NN_RL_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_rl_4x4_vs_lib4c)
#endif

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_rl_4x4_vs_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nn_rl_one_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_rl_one_4x4_lib4cc)

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
	INNER_EDGE_DTRMM_NN_RL_ONE_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_rl_one_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_rl_one_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nn_rl_4x4_one_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_rl_one_4x4_vs_lib4cc)

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
	INNER_EDGE_DTRMM_NN_RL_ONE_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_rl_one_4x4_vs_lib4c)
#endif

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_rl_one_4x4_vs_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nn_ru_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_ru_4x4_lib4cc)

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
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RU_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_ru_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_ru_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nn_ru_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_ru_4x4_vs_lib4cc)

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
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:

	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RU_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_ru_4x4_vs_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_ru_4x4_vs_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nn_ru_one_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_ru_one_4x4_lib4cc)

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
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RU_ONE_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_ru_one_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_ru_one_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nn_ru_one_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nn_ru_one_4x4_vs_lib4cc)

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
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:

	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RU_ONE_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nn_ru_one_4x4_vs_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nn_ru_one_4x4_vs_lib4cc)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nt_rl_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_rl_4x4_lib4cc)

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
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NT_RL_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_rl_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_rl_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nt_rl_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_rl_4x4_vs_lib4cc)

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
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:

	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NT_RL_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_rl_4x4_vs_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_rl_4x4_vs_lib4cc)





//                                    1      2              3          4          5             6          7        8          9
// void kernel_dtrmm_nt_ru_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_4x4_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	subl	$4, %r10d
	movq	ARG3, %r11 // A
	addq	$128, %r11
	movq	ARG4, %r12 // B
	addq	$128, %r12

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
//	CALL(inner_edge_dtrmm_nt_ru_4x4_lib4)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
	CALL(inner_blend_4x4_lib4)
#endif


	// initial triangle

	movq	ARG1, %r10
	movq	ARG3, %r11
	movq	ARG4, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
	CALL(inner_edge_dtrmm_nt_ru_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_4x4_lib44c)





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dtrmm_nt_ru_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_4x4_vs_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	subl	$4, %r10d
	movq	ARG3, %r11 // A
	addq	$128, %r11
	movq	ARG4, %r12 // B
	addq	$128, %r12

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
//	CALL(inner_edge_dtrmm_nt_ru_4x4_vs_lib4)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
	CALL(inner_blend_4x4_lib4)
#endif


	// initial triangle

	movq	ARG1, %r10
	movq	ARG3, %r11
	movq	ARG4, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
	CALL(inner_edge_dtrmm_nt_ru_4x4_vs_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d
	movq	ARG10, %r14   // m1
	movq	ARG11, %r15   // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12   // m1
	movq	ARG11, %r13   // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_4x4_vs_lib44c)





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nt_ru_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_4x4_lib4cc)

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
	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_ru_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_4x4_lib4cc)





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nt_ru_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_4x4_vs_lib4cc)

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
	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_ru_4x4_vs_lib4c)
#endif

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_4x4_vs_lib4cc)





//                                        1      2              3          4          5             6          7        8          9
// void kernel_dtrmm_nt_ru_one_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_one_4x4_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	subl	$4, %r10d
	movq	ARG3, %r11 // A
	addq	$128, %r11
	movq	ARG4, %r12 // B
	addq	$128, %r12

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_LIB4
#else
//	CALL(inner_edge_dtrmm_one_nt_ru_4x4_lib4)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
	CALL(inner_blend_4x4_lib4)
#endif


	// initial triangle

	movq	ARG1, %r10
	movq	ARG3, %r11
	movq	ARG4, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_LIB4
#else
	CALL(inner_edge_dtrmm_nt_ru_one_4x4_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_one_4x4_lib44c)





//                                           1      2              3          4          5             6          7        8          9        10      11
// void kernel_dtrmm_nt_ru_one_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_one_4x4_vs_lib44c)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	subl	$4, %r10d
	movq	ARG3, %r11 // A
	addq	$128, %r11
	movq	ARG4, %r12 // B
	addq	$128, %r12

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_VS_LIB4
#else
//	CALL(inner_edge_dtrmm_nt_ru_one_4x4_vs_lib4)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
	CALL(inner_blend_4x4_lib4)
#endif


	// initial triangle

	movq	ARG1, %r10
	movq	ARG3, %r11
	movq	ARG4, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_VS_LIB4
#else
	CALL(inner_edge_dtrmm_nt_ru_one_4x4_vs_lib4)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d
	movq	ARG10, %r14   // m1
	movq	ARG11, %r15   // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12   // m1
	movq	ARG11, %r13   // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_one_4x4_vs_lib44c)





//                                        1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nt_ru_one_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_one_4x4_lib4cc)

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
	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_ru_one_4x4_lib4c)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	CALL(inner_scale_ab_4x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_one_4x4_lib4cc)





//                                           1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nt_ru_one_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrmm_nt_ru_one_4x4_vs_lib4cc)

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
	INNER_EDGE_DTRMM_NT_RU_ONE_4X4_VS_LIB4C
#else
	CALL(inner_edge_dtrmm_nt_ru_one_4x4_vs_lib4c)
#endif

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
	CALL(inner_scale_ab_4x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrmm_nt_ru_one_4x4_vs_lib4cc)





//                                    1      2          3          4          5        6          7        8
// void kernel_dpotrf_nt_l_4x4_lib44c(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dpotrf_nt_l_4x4_lib44c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M11_4X4_LIB
#else
	CALL(inner_blend_scale_m11_4x4_lib)
#endif


	// factorization

	movq	ARG8, %r10  // inv_diag_D 
	movl	$4, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
	CALL(inner_edge_dpotrf_4x4_vs_lib4)
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB
#else
	CALL(inner_store_l_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dpotrf_nt_l_4x4_lib44c)





//                                       1      2          3          4          5        6          7        8                   9       10
// void kernel_dpotrf_nt_l_4x4_vs_lib44c(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dpotrf_nt_l_4x4_vs_lib44c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d
	movq	ARG9, %r12 // m1
	movq	ARG10, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M11_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m11_4x4_vs_lib)
#endif


	// factorization

	movq	ARG8, %r10  // inv_diag_D 
	movq	ARG10, %r11 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
	CALL(inner_edge_dpotrf_4x4_vs_lib4)
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d
	movq	ARG9, %r12 // m1
	movq	ARG10, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB
#else
	CALL(inner_store_l_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dpotrf_nt_l_4x4_vs_lib44c)





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nn_rl_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_inv_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rln_inv_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_inv_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 
	movq	ARG12, %r13  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rln_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nn_rl_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_inv_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rln_inv_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_inv_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d


	movq	ARG14, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG13, %r13 // m1
	movq	ARG14, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rln_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG13, %r12 // m1
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_vs_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc)





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nn_rl_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_one_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rln_one_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_one_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG11, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rln_one_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nn_rl_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_one_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rln_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_one_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nn_rl_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_rl_one_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d


	movq	ARG13, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG13, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG13, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG12, %r13 // m1
	movq	ARG13, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG13, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rln_one_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG12, %r12 // m1
	movq	ARG13, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_vs_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_rl_one_4x4_vs_lib4ccc)





//                                         1      2          3          4             5          6        7          8        9          10
// void kernel_dtrsm_nt_rl_inv_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_lib44c4)





//                                            1      2          3          4             5          6        7          8        9          10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG11, %r13 // m1
	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4)






//                                         1      2          3          4         5        6          7        8          9        10
// void kernel_dtrsm_nt_rl_inv_4x4_lib44cc(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_lib44cc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M11_4X4_LIB
#else
	CALL(inner_blend_scale_m11_4x4_lib)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_lib)
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_lib44cc)




//                                            1      2          3          4         5        6          7        8          9        10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M11_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m11_4x4_vs_lib)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc)





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nt_rl_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11 // m1
	movq	ARG12, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d


	movq	ARG14, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG13, %r13 // m1
	movq	ARG14, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rlt_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG13, %r12 // m1
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_vs_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc)





//                                         1      2          3          4             5          6        7          8        9
// void kernel_dtrsm_nt_rl_one_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_lib44c4)





//                                            1      2          3          4             5          6        7          8        9          10      11
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG11, %r13 // m1
	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12 // m1
	movq	ARG11, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4)






//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nt_rl_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG11, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11 // m1
	movq	ARG11, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nt_rl_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d


	movq	ARG13, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG13, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG13, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG12, %r13 // m1
	movq	ARG13, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG13, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rlt_one_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG13, %r12 // m1
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_vs_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc)





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nn_ru_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_inv_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_run_inv_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_inv_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 
	movq	ARG12, %r13  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_run_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nn_ru_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_inv_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_run_inv_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_inv_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_inv_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG14, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG13, %r13 // m1
	movq	ARG14, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_run_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG13, %r12 // m1
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_inv_4x4_vs_lib4ccc)





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nn_ru_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_one_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_run_one_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_one_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG11, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_run_one_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nn_ru_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_one_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_run_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_one_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nn_ru_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ru_one_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG13, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG13, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG13, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG12, %r13 // m1
	movq	ARG13, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG13, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_run_one_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG12, %r12 // m1
	movq	ARG13, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ru_one_4x4_vs_lib4ccc)





//                                         1      2          3          4             5          6        7          8        9          10
// void kernel_dtrsm_nt_ru_inv_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11 // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_lib44c4)





//                                            1      2          3          4             5          6        7          8        9          10                  11      12
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, in1 n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG11, %r13 // m1
	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11 // inv_diag_E 
	movq	ARG12, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB4
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_vs_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4)





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nt_ru_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 
	movq	ARG12, %r13  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c)




//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nt_ru_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG14, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG13, %r13 // m1
	movq	ARG14, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rut_inv_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG13, %r12 // m1
	movq	ARG14, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc)





//                                         1      2          3          4             5          6        7          8        9
// void kernel_dtrsm_nt_ru_one_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_LIB4
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_lib44c4)





//                                            1      2          3          4             5          6        7          8        9          10      11
// void kernel_dtrsm_nt_ru_one_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, in1 n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_vs_lib44c4)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4)
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG10, %r13 // m1
	movq	ARG11, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_blend_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG11, %r11 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_VS_LIB4
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_vs_lib4)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d
	movq	ARG10, %r12 // m1
	movq	ARG11, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_vs_lib44c4)





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nt_ru_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_lib)
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	CALL(inner_store_4x4_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_lib4c4c)





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG11, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
	CALL(inner_scale_m1b_4x4_lib4)
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_vs_lib)
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
	CALL(inner_store_4x4_vs_lib4)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c)





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nt_ru_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nt_ru_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nt_ru_one_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

	movq	ARG13, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG13, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG13, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nt_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG12, %r13 // m1
	movq	ARG13, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG13, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONEINV_4X4_VS_LIB
#else
	CALL(inner_edge_dtrsm_rut_one_4x4_vs_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG12, %r12 // m1
	movq	ARG13, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nt_ru_one_4x4_vs_lib4ccc)





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nn_ll_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ll_one_4x4_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d // ldb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	CALL(inner_scale_m1b_4x4_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_lln_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	CALL(inner_store_4x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ll_one_4x4_lib4ccc)





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nn_ll_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dtrsm_nn_ll_one_4x4_vs_lib4ccc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d


	movq	ARG14, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x1_lib4c)
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x2_lib4c)
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x3_lib4c)
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
	CALL(inner_kernel_dgemm_nn_4x4_lib4c)
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d
	movq	ARG12, %r13 // m1
	movq	ARG13, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	CALL(inner_scale_m1b_4x4_vs_lib)
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB
#else
	CALL(inner_edge_dtrsm_lln_one_4x4_lib)
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d
	movq	ARG12, %r12 // m1
	movq	ARG13, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	CALL(inner_store_4x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dtrsm_nn_ll_one_4x4_vs_lib4ccc)






