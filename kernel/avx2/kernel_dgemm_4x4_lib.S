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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nt_4x4_lib4c, @function
inner_kernel_dgemm_nt_4x4_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nt_4x4_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nt_4x4_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nt_4x4_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r12), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nt_4x4_lib4c, .-inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nt_4x3_lib4c, @function
inner_kernel_dgemm_nt_4x3_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nt_4x3_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nt_4x3_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nt_4x3_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	16(%r12), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nt_4x3_lib4c, .-inner_kernel_dgemm_nt_4x3_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nt_4x2_lib4c, @function
inner_kernel_dgemm_nt_4x2_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nt_4x2_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nt_4x2_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nt_4x2_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nt_4x2_lib4c, .-inner_kernel_dgemm_nt_4x2_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nt_4x1_lib4c, @function
inner_kernel_dgemm_nt_4x1_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nt_4x1_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nt_4x1_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nt_4x1_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
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
	addq	%r13, %r12

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	addq	%r13, %r12

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	addq	%r13, %r12

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nt_4x1_lib4c, .-inner_kernel_dgemm_nt_4x1_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nn_4x4_lib4c, @function
inner_kernel_dgemm_nn_4x4_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nn_4x4_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nn_4x4_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nn_4x4_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

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
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	8(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	16(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm7

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
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nn_4x4_lib4c, .-inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nn_4x3_lib4c, @function
inner_kernel_dgemm_nn_4x3_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nn_4x3_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nn_4x3_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nn_4x3_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

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
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm6

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
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nn_4x3_lib4c, .-inner_kernel_dgemm_nn_4x3_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nn_4x2_lib4c, @function
inner_kernel_dgemm_nn_4x2_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nn_4x2_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nn_4x2_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nn_4x2_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
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

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm5

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nn_4x2_lib4c, .-inner_kernel_dgemm_nn_4x2_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_nn_4x1_lib4c, @function
inner_kernel_dgemm_nn_4x1_lib4c:
#elif defined(OS_MAC)
_inner_kernel_dgemm_nn_4x1_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_nn_4x1_lib4c; .scl 2; .type 32; .endef
inner_kernel_dgemm_nn_4x1_lib4c:
#endif
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
	vfmadd231pd		%ymm13, %ymm12, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
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

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm4

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
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

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_nn_4x1_lib4c, .-inner_kernel_dgemm_nn_4x1_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nn_rl_4x4_lib4c, @function
inner_edge_dtrmm_nn_rl_4x4_lib4c:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nn_rl_4x4_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nn_rl_4x4_lib4c; .scl 2; .type 32; .endef
inner_edge_dtrmm_nn_rl_4x4_lib4c:
#endif
#endif
	
	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vbroadcastsd	8(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	8(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vbroadcastsd	16(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	16(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vbroadcastsd	24(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	24(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	24(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	subl	$4, %r10d
	addq	$32, %r12
	addq	$32, %r15
	addq	$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nn_rl_4x4_lib4c, .-inner_edge_dtrmm_nn_rl_4x4_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nn_rl_4x4_vs_lib4c, @function
inner_edge_dtrmm_nn_rl_4x4_vs_lib4c:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nn_rl_4x4_vs_lib4c:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nn_rl_4x4_vs_lib4c; .scl 2; .type 32; .endef
inner_edge_dtrmm_nn_rl_4x4_vs_lib4c:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		0f // end

	movq	%r12, %r15
	addq	%r13, %r15
	addq	%r13, %r15 // B+2*ldb

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jle		0f // end

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vbroadcastsd	0(%r12), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vbroadcastsd	0(%r12, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vbroadcastsd	0(%r15, %r13, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3

	subl	$1, %r10d
	addq	$8, %r12
	addq	$8, %r15
	addq	$32, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nn_rl_4x4_vs_lib4c, .-inner_edge_dtrmm_nn_rl_4x4_vs_lib4c
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rln_inv_4x4_lib, @function
inner_edge_dtrsm_rln_inv_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rln_inv_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rln_inv_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rln_inv_4x4_lib:
#endif
#endif
	
	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rln_inv_4x4_lib, .-inner_edge_dtrsm_rln_inv_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rln_inv_4x4_vs_lib, @function
inner_edge_dtrsm_rln_inv_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rln_inv_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rln_inv_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rln_inv_4x4_vs_lib:
#endif
#endif
	
	cmpl			$3, %r13d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

0:
	cmpl			$2, %r13d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

0:
	cmpl			$1, %r13d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

0:

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rln_inv_4x4_vs_lib, .-inner_edge_dtrsm_rln_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rln_one_4x4_lib, @function
inner_edge_dtrsm_rln_one_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rln_one_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rln_one_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rln_one_4x4_lib:
#endif
#endif
	
	// 4th column
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

	// 3rd column
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

	// 2nd column
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rln_one_4x4_lib, .-inner_edge_dtrsm_rln_one_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rln_one_4x4_vs_lib, @function
inner_edge_dtrsm_rln_one_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rln_one_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rln_one_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rln_one_4x4_vs_lib:
#endif
#endif
	
	cmpl			$3, %r12d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

0:
	cmpl			$2, %r12d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

0:
	cmpl			$1, %r12d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

0:

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rln_one_4x4_vs_lib, .-inner_edge_dtrsm_rln_one_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_lib, @function
inner_edge_dtrsm_rlt_inv_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_4x4_lib:
#endif
#endif
	
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3

	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3

	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_lib, .-inner_edge_dtrsm_rlt_inv_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_vs_lib, @function
inner_edge_dtrsm_rlt_inv_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_4x4_vs_lib:
#endif
#endif
	
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	cmpl			$2, %r13d
	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	cmpl			$3, %r13d
	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	cmpl			$4, %r13d
	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_vs_lib, .-inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_4x4_lib, @function
inner_edge_dtrsm_rlt_one_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_one_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_one_4x4_lib:
#endif
#endif
	
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3

	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3

	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_4x4_lib, .-inner_edge_dtrsm_rlt_one_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_4x4_vs_lib, @function
inner_edge_dtrsm_rlt_one_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_one_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_one_4x4_vs_lib:
#endif
#endif
	
	cmpl			$2, %r12d
	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1

	cmpl			$3, %r12d
	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2

	cmpl			$4, %r12d
	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	24(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_4x4_vs_lib, .-inner_edge_dtrsm_rlt_one_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_inv_4x4_lib, @function
inner_edge_dtrsm_run_inv_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_inv_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_run_inv_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_run_inv_4x4_lib:
#endif
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3

	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3

	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_inv_4x4_lib, .-inner_edge_dtrsm_run_inv_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_inv_4x4_vs_lib, @function
inner_edge_dtrsm_run_inv_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_inv_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_run_inv_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_run_inv_4x4_vs_lib:
#endif
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	cmpl			$2, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	cmpl			$3, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	cmpl			$4, %r13d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_inv_4x4_vs_lib, .-inner_edge_dtrsm_run_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_one_4x4_lib, @function
inner_edge_dtrsm_run_one_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_one_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_run_one_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_run_one_4x4_lib:
#endif
#endif
	
	addq	%r11, %r10

	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3

	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3

	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_one_4x4_lib, .-inner_edge_dtrsm_run_one_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_one_4x4_vs_lib, @function
inner_edge_dtrsm_run_one_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_one_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_run_one_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_run_one_4x4_vs_lib:
#endif
#endif
	
	addq	%r11, %r10

	cmpl			$2, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1

	cmpl			$3, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2

	cmpl			$4, %r12d
	jl				0f // ret

	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_one_4x4_vs_lib, .-inner_edge_dtrsm_run_one_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_4x4_lib, @function
inner_edge_dtrsm_rut_inv_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_inv_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_inv_4x4_lib:
#endif
#endif
	
	addq	%r11, %r10

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_4x4_lib, .-inner_edge_dtrsm_rut_inv_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_4x4_vs_lib, @function
inner_edge_dtrsm_rut_inv_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_inv_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_inv_4x4_vs_lib:
#endif
#endif
	
	addq	%r11, %r10

	cmpl			$3, %r13d
	jle				0f

	// 4th column
	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

0:
	cmpl			$2, %r13d
	jle				0f

	// 3rd column
	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

0:
	cmpl			$1, %r13d
	jle				0f

	// 2nd column
	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

0:

	// 1st column
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_4x4_vs_lib, .-inner_edge_dtrsm_rut_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_one_4x4_lib, @function
inner_edge_dtrsm_rut_one_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_one_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_one_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_one_4x4_lib:
#endif
#endif
	
	addq	%r11, %r10

	// 4th column
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

	// 3rd column
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

	// 2nd column
	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_one_4x4_lib, .-inner_edge_dtrsm_rut_one_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_one_4x4_vs_lib, @function
inner_edge_dtrsm_rut_one_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_one_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_one_4x4_vs_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_one_4x4_vs_lib:
#endif
#endif
	
	addq	%r11, %r10

	cmpl			$3, %r12d
	jle				0f

	// 4th column
	vbroadcastsd	16(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm2
	vbroadcastsd	8(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 2), %ymm13
	vfnmadd231pd	%ymm3, %ymm13, %ymm0

0:
	cmpl			$2, %r12d
	jle				0f

	// 3rd column
	vbroadcastsd	8(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm1
	vbroadcastsd	0(%r10, %r11, 1), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm0

0:
	cmpl			$1, %r12d
	jle				0f

	// 2nd column
	vbroadcastsd	0(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm0

0:

	// 1st column

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_one_4x4_vs_lib, .-inner_edge_dtrsm_rut_one_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_lib, @function
inner_scale_ab_4x4_lib:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_4x4_lib; .scl 2; .type 32; .endef
inner_scale_ab_4x4_lib:
#endif
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
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_lib, .-inner_scale_ab_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_vs_lib, @function
inner_scale_ab_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_4x4_vs_lib; .scl 2; .type 32; .endef
inner_scale_ab_4x4_vs_lib:
#endif
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
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	addq		%r13, %r12
	cmpl		$2, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	addq		%r13, %r12
	cmpl		$3, %r15d
	jl			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	addq		%r13, %r12
	cmpl		$3, %r15d
	je			0f // end
	vmaskmovpd	0(%r12), %ymm13, %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_vs_lib, .-inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_m1b_4x4_lib, @function
inner_scale_m1b_4x4_lib:
#elif defined(OS_MAC)
_inner_scale_m1b_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_m1b_4x4_lib; .scl 2; .type 32; .endef
inner_scale_m1b_4x4_lib:
#endif
#endif

	// beta
	vbroadcastsd	0(%r10), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r11), %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm0
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm1
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm2
	addq		%r12, %r11
	vmovupd		0(%r11), %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_m1b_4x4_lib, .-inner_scale_m1b_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_m1b_4x4_vs_lib, @function
inner_scale_m1b_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_scale_m1b_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_m1b_4x4_vs_lib; .scl 2; .type 32; .endef
inner_scale_m1b_4x4_vs_lib:
#endif
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
	vfmsub231pd	%ymm14, %ymm15, %ymm0
	addq		%r12, %r11
	cmpl		$2, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm1
	addq		%r12, %r11
	cmpl		$3, %r14d
	jl			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm2
	addq		%r12, %r11
	cmpl		$3, %r14d
	je			0f // end
	vmaskmovpd	0(%r11), %ymm13, %ymm15
	vfmsub231pd	%ymm14, %ymm15, %ymm3
//	addq		%r12, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_m1b_4x4_vs_lib, .-inner_scale_m1b_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_m11_4x4_lib, @function
inner_scale_m11_4x4_lib:
#elif defined(OS_MAC)
_inner_scale_m11_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_m11_4x4_lib; .scl 2; .type 32; .endef
inner_scale_m11_4x4_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_scale_m11_4x4_lib, .-inner_scale_m11_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_scale_m11_4x4_vs_lib, @function
inner_scale_m11_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_scale_m11_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_m11_4x4_vs_lib; .scl 2; .type 32; .endef
inner_scale_m11_4x4_vs_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_scale_m11_4x4_vs_lib, .-inner_scale_m11_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_4x4_lib, @function
inner_store_4x4_lib:
#elif defined(OS_MAC)
_inner_store_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_4x4_lib; .scl 2; .type 32; .endef
inner_store_4x4_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib, .-inner_store_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_l_4x4_lib, @function
inner_store_l_4x4_lib:
#elif defined(OS_MAC)
_inner_store_l_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_l_4x4_lib; .scl 2; .type 32; .endef
inner_store_l_4x4_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_lib, .-inner_store_l_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_u_4x4_lib, @function
inner_store_u_4x4_lib:
#elif defined(OS_MAC)
_inner_store_u_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_u_4x4_lib; .scl 2; .type 32; .endef
inner_store_u_4x4_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_u_4x4_lib, .-inner_store_u_4x4_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_4x4_vs_lib, @function
inner_store_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_store_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_4x4_vs_lib; .scl 2; .type 32; .endef
inner_store_4x4_vs_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_4x4_vs_lib, .-inner_store_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_l_4x4_vs_lib, @function
inner_store_l_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_store_l_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_l_4x4_vs_lib; .scl 2; .type 32; .endef
inner_store_l_4x4_vs_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_vs_lib, .-inner_store_l_4x4_vs_lib
#endif
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
#if defined(OS_LINUX)
	.type inner_store_u_4x4_vs_lib, @function
inner_store_u_4x4_vs_lib:
#elif defined(OS_MAC)
_inner_store_u_4x4_vs_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_u_4x4_vs_lib; .scl 2; .type 32; .endef
inner_store_u_4x4_vs_lib:
#endif
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

#if defined(OS_LINUX)
	.size	inner_store_u_4x4_vs_lib, .-inner_store_u_4x4_vs_lib
#endif
#endif





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nt_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_lib4cc
	.type kernel_dgemm_nt_4x4_lib4cc, @function
kernel_dgemm_nt_4x4_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_lib4cc
_kernel_dgemm_nt_4x4_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_4x4_lib4cc
	.def kernel_dgemm_nt_4x4_lib4cc; .scl 2; .type 32; .endef
kernel_dgemm_nt_4x4_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_lib4cc, .-kernel_dgemm_nt_4x4_lib4cc
#endif





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nt_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_vs_lib4cc
	.type kernel_dgemm_nt_4x4_vs_lib4cc, @function
kernel_dgemm_nt_4x4_vs_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_vs_lib4cc
_kernel_dgemm_nt_4x4_vs_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_4x4_vs_lib4cc
	.def kernel_dgemm_nt_4x4_vs_lib4cc; .scl 2; .type 32; .endef
kernel_dgemm_nt_4x4_vs_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_vs_lib4cc, .-kernel_dgemm_nt_4x4_vs_lib4cc
#endif





//                                 1      2              3          4          5             6          7        8          9
// void kernel_dgemm_nt_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_lib44c
	.type kernel_dgemm_nt_4x4_lib44c, @function
kernel_dgemm_nt_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_lib44c
_kernel_dgemm_nt_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_4x4_lib44c
	.def kernel_dgemm_nt_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dgemm_nt_4x4_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_lib44c, .-kernel_dgemm_nt_4x4_lib44c
#endif





//                                    1      2              3          4          5             6          7        8          9        10      11
// void kernel_dgemm_nt_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_vs_lib44c
	.type kernel_dgemm_nt_4x4_vs_lib44c, @function
kernel_dgemm_nt_4x4_vs_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_vs_lib44c
_kernel_dgemm_nt_4x4_vs_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_4x4_vs_lib44c
	.def kernel_dgemm_nt_4x4_vs_lib44c; .scl 2; .type 32; .endef
kernel_dgemm_nt_4x4_vs_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_vs_lib44c, .-kernel_dgemm_nt_4x4_vs_lib44c
#endif





//                                 1      2              3          4          5        6             7          8        9          10
// void kernel_dgemm_nn_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_4x4_lib4cc
	.type kernel_dgemm_nn_4x4_lib4cc, @function
kernel_dgemm_nn_4x4_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_4x4_lib4cc
_kernel_dgemm_nn_4x4_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_4x4_lib4cc
	.def kernel_dgemm_nn_4x4_lib4cc; .scl 2; .type 32; .endef
kernel_dgemm_nn_4x4_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_4x4_lib4cc, .-kernel_dgemm_nn_4x4_lib4cc
#endif





//                                    1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dgemm_nn_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_4x4_vs_lib4cc
	.type kernel_dgemm_nn_4x4_vs_lib4cc, @function
kernel_dgemm_nn_4x4_vs_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_4x4_vs_lib4cc
_kernel_dgemm_nn_4x4_vs_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_4x4_vs_lib4cc
	.def kernel_dgemm_nn_4x4_vs_lib4cc; .scl 2; .type 32; .endef
kernel_dgemm_nn_4x4_vs_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_4x4_vs_lib4cc, .-kernel_dgemm_nn_4x4_vs_lib4cc
#endif





//                                   1      2              3          4          5             6          7        8          9
// void kernel_dsyrk_nt_l_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_lib44c
	.type kernel_dsyrk_nt_l_4x4_lib44c, @function
kernel_dsyrk_nt_l_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_lib44c
_kernel_dsyrk_nt_l_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_l_4x4_lib44c
	.def kernel_dsyrk_nt_l_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dsyrk_nt_l_4x4_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_4x4_lib44c, .-kernel_dsyrk_nt_l_4x4_lib44c
#endif





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dsyrk_nt_l_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_vs_lib44c
	.type kernel_dsyrk_nt_l_4x4_vs_lib44c, @function
kernel_dsyrk_nt_l_4x4_vs_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_vs_lib44c
_kernel_dsyrk_nt_l_4x4_vs_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_l_4x4_vs_lib44c
	.def kernel_dsyrk_nt_l_4x4_vs_lib44c; .scl 2; .type 32; .endef
kernel_dsyrk_nt_l_4x4_vs_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_4x4_vs_lib44c, .-kernel_dsyrk_nt_l_4x4_vs_lib44c
#endif





//                                   1      2              3          4          5             6          7        8          9
// void kernel_dsyrk_nt_u_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_u_4x4_lib44c
	.type kernel_dsyrk_nt_u_4x4_lib44c, @function
kernel_dsyrk_nt_u_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_u_4x4_lib44c
_kernel_dsyrk_nt_u_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_u_4x4_lib44c
	.def kernel_dsyrk_nt_u_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dsyrk_nt_u_4x4_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_U_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_u_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_u_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_u_4x4_lib44c, .-kernel_dsyrk_nt_u_4x4_lib44c
#endif





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dsyrk_nt_u_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_u_4x4_vs_lib44c
	.type kernel_dsyrk_nt_u_4x4_vs_lib44c, @function
kernel_dsyrk_nt_u_4x4_vs_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_u_4x4_vs_lib44c
_kernel_dsyrk_nt_u_4x4_vs_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_u_4x4_vs_lib44c
	.def kernel_dsyrk_nt_u_4x4_vs_lib44c; .scl 2; .type 32; .endef
kernel_dsyrk_nt_u_4x4_vs_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
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
	INNER_SCALE_AB_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_u_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_u_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_u_4x4_vs_lib44c, .-kernel_dsyrk_nt_u_4x4_vs_lib44c
#endif





//                                    1      2              3          4          5        6             7          8        9          10
// void kernel_dtrmm_nn_rl_4x4_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nn_rl_4x4_lib4cc
	.type kernel_dtrmm_nn_rl_4x4_lib4cc, @function
kernel_dtrmm_nn_rl_4x4_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nn_rl_4x4_lib4cc
_kernel_dtrmm_nn_rl_4x4_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nn_rl_4x4_lib4cc
	.def kernel_dtrmm_nn_rl_4x4_lib4cc; .scl 2; .type 32; .endef
kernel_dtrmm_nn_rl_4x4_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RL_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nn_rl_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nn_rl_4x4_lib4c
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nn_rl_4x4_lib4cc, .-kernel_dtrmm_nn_rl_4x4_lib4cc
#endif





//                                       1      2              3          4          5        6             7          8        9          10       11      12
// void kernel_dtrmm_nn_rl_4x4_vs_lib4cc(int k, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nn_rl_4x4_vs_lib4cc
	.type kernel_dtrmm_nn_rl_4x4_vs_lib4cc, @function
kernel_dtrmm_nn_rl_4x4_vs_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nn_rl_4x4_vs_lib4cc
_kernel_dtrmm_nn_rl_4x4_vs_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nn_rl_4x4_vs_lib4cc
	.def kernel_dtrmm_nn_rl_4x4_vs_lib4cc; .scl 2; .type 32; .endef
kernel_dtrmm_nn_rl_4x4_vs_lib4cc:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG5, %r13  // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_EDGE_DTRMM_NN_RL_4X4_VS_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nn_rl_4x4_vs_lib4c
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nn_rl_4x4_vs_lib4c
#endif
#endif

	movq	ARG12, %r14  // n1
	cmpl	$1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X1_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nn_rl_4x4_vs_lib4cc, .-kernel_dtrmm_nn_rl_4x4_vs_lib4cc
#endif





//                                    1      2              3          4          5             6          7        8          9
// void kernel_dtrmm_nt_ru_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_lib44c
	.type kernel_dtrmm_nt_ru_4x4_lib44c, @function
kernel_dtrmm_nt_ru_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_lib44c
_kernel_dtrmm_nt_ru_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nt_ru_4x4_lib44c
	.def kernel_dtrmm_nt_ru_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dtrmm_nt_ru_4x4_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
//	subl	$4, %r10d
	movq	ARG3, %r11 // A
//	addq	$128, %r11
	movq	ARG4, %r12 // B
//	addq	$128, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nt_ru_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blend

#if MACRO_LEVEL>=1
//	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
//	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
//	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

//	movq	ARG3, %r10
//	movq	ARG4, %r11

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
//	call inner_edge_dtrmm_nt_ru_4x4_lib4
#elif defined(OS_MAC)
//	callq _inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_4x4_lib44c, .-kernel_dtrmm_nt_ru_4x4_lib44c
#endif





//                                       1      2              3          4          5             6          7        8          9        10      11
// void kernel_dtrmm_nt_ru_4x4_vs_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_vs_lib44c
	.type kernel_dtrmm_nt_ru_4x4_vs_lib44c, @function
kernel_dtrmm_nt_ru_4x4_vs_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_vs_lib44c
_kernel_dtrmm_nt_ru_4x4_vs_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nt_ru_4x4_vs_lib44c
	.def kernel_dtrmm_nt_ru_4x4_vs_lib44c; .scl 2; .type 32; .endef
kernel_dtrmm_nt_ru_4x4_vs_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
//	subl	$4, %r10d
	movq	ARG3, %r11 // A
//	addq	$128, %r11
	movq	ARG4, %r12 // B
//	addq	$128, %r12

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blend

#if MACRO_LEVEL>=1
//	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
//	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
//	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

//	movq	ARG1, %r10
//	movq	ARG3, %r12
//	movq	ARG4, %r13

#if MACRO_LEVEL>=1
//	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
//	call inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#elif defined(OS_MAC)
//	callq _inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_4x4_vs_lib44c, .-kernel_dtrmm_nt_ru_4x4_vs_lib44c
#endif





//                                    1      2          3          4          5        6          7        8
// void kernel_dpotrf_nt_l_4x4_lib44c(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_4x4_lib44c
	.type kernel_dpotrf_nt_l_4x4_lib44c, @function
kernel_dpotrf_nt_l_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_4x4_lib44c
_kernel_dpotrf_nt_l_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dpotrf_nt_l_4x4_lib44c
	.def kernel_dpotrf_nt_l_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dpotrf_nt_l_4x4_lib44c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m11_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m11_4x4_lib
#endif
#endif


	// factorization

	movq	ARG8, %r10  // inv_diag_D 
	movl	$4, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_4x4_lib44c, .-kernel_dpotrf_nt_l_4x4_lib44c
#endif





//                                       1      2          3          4          5        6          7        8                   9       10
// void kernel_dpotrf_nt_l_4x4_vs_lib44c(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_4x4_vs_lib44c
	.type kernel_dpotrf_nt_l_4x4_vs_lib44c, @function
kernel_dpotrf_nt_l_4x4_vs_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_4x4_vs_lib44c
_kernel_dpotrf_nt_l_4x4_vs_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dpotrf_nt_l_4x4_vs_lib44c
	.def kernel_dpotrf_nt_l_4x4_vs_lib44c; .scl 2; .type 32; .endef
kernel_dpotrf_nt_l_4x4_vs_lib44c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d
	movq	ARG9, %r12 // m1
	movq	ARG10, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m11_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m11_4x4_vs_lib
#endif
#endif


	// factorization

	movq	ARG8, %r10  // inv_diag_D 
	movq	ARG10, %r11 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_4x4_vs_lib44c, .-kernel_dpotrf_nt_l_4x4_vs_lib44c
#endif





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nn_rl_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_inv_4x4_lib4c4c
	.type kernel_dtrsm_nn_rl_inv_4x4_lib4c4c, @function
kernel_dtrsm_nn_rl_inv_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_inv_4x4_lib4c4c
_kernel_dtrsm_nn_rl_inv_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_inv_4x4_lib4c4c
	.def kernel_dtrsm_nn_rl_inv_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_inv_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_inv_4x4_lib4c4c, .-kernel_dtrsm_nn_rl_inv_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c
	.type kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c, @function
kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c
_kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c
	.def kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_inv_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c, .-kernel_dtrsm_nn_rl_inv_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nn_rl_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_inv_4x4_lib4ccc
	.type kernel_dtrsm_nn_rl_inv_4x4_lib4ccc, @function
kernel_dtrsm_nn_rl_inv_4x4_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_inv_4x4_lib4ccc
_kernel_dtrsm_nn_rl_inv_4x4_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_inv_4x4_lib4ccc
	.def kernel_dtrsm_nn_rl_inv_4x4_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_inv_4x4_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_inv_4x4_lib4ccc, .-kernel_dtrsm_nn_rl_inv_4x4_lib4ccc
#endif





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc
	.type kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc, @function
kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc
_kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc
	.def kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc, .-kernel_dtrsm_nn_rl_inv_4x4_vs_lib4ccc
#endif





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nn_rl_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_one_4x4_lib4c4c
	.type kernel_dtrsm_nn_rl_one_4x4_lib4c4c, @function
kernel_dtrsm_nn_rl_one_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_one_4x4_lib4c4c
_kernel_dtrsm_nn_rl_one_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_one_4x4_lib4c4c
	.def kernel_dtrsm_nn_rl_one_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_one_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_one_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_one_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_one_4x4_lib4c4c, .-kernel_dtrsm_nn_rl_one_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c
	.type kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c, @function
kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c
_kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c
	.def kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLN_ONE_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rln_one_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rln_one_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c, .-kernel_dtrsm_nn_rl_one_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4             5          6        7          8        9          10
// void kernel_dtrsm_nt_rl_inv_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44c4
	.type kernel_dtrsm_nt_rl_inv_4x4_lib44c4, @function
kernel_dtrsm_nt_rl_inv_4x4_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib44c4
_kernel_dtrsm_nt_rl_inv_4x4_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44c4
	.def kernel_dtrsm_nt_rl_inv_4x4_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib44c4, .-kernel_dtrsm_nt_rl_inv_4x4_lib44c4
#endif





//                                            1      2          3          4             5          6        7          8        9          10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4
	.def kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG11, %r13 // m1
	movq	ARG12, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4
#endif





//                                         1      2          3          4         5        6          7        8          9        10
// void kernel_dtrsm_nt_rl_inv_4x4_lib44cc(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44cc
	.type kernel_dtrsm_nt_rl_inv_4x4_lib44cc, @function
kernel_dtrsm_nt_rl_inv_4x4_lib44cc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib44cc
_kernel_dtrsm_nt_rl_inv_4x4_lib44cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44cc
	.def kernel_dtrsm_nt_rl_inv_4x4_lib44cc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_lib44cc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m11_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m11_4x4_lib
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib44cc, .-kernel_dtrsm_nt_rl_inv_4x4_lib44cc
#endif





//                                            1      2          3          4         5        6          7        8          9        10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc
	.def kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m11_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m11_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib44cc
#endif





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nt_rl_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4c4c
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4c4c, @function
kernel_dtrsm_nt_rl_inv_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib4c4c
_kernel_dtrsm_nt_rl_inv_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4c4c
	.def kernel_dtrsm_nt_rl_inv_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4c4c, .-kernel_dtrsm_nt_rl_inv_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c
	.def kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11 // m1
	movq	ARG12, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4ccc
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4ccc, @function
kernel_dtrsm_nt_rl_inv_4x4_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib4ccc
_kernel_dtrsm_nt_rl_inv_4x4_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4ccc
	.def kernel_dtrsm_nt_rl_inv_4x4_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4ccc, .-kernel_dtrsm_nt_rl_inv_4x4_lib4ccc
#endif





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc
	.def kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc
#endif





//                                         1      2          3          4             5          6        7          8        9
// void kernel_dtrsm_nt_rl_one_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib44c4
	.type kernel_dtrsm_nt_rl_one_4x4_lib44c4, @function
kernel_dtrsm_nt_rl_one_4x4_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_lib44c4
_kernel_dtrsm_nt_rl_one_4x4_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib44c4
	.def kernel_dtrsm_nt_rl_one_4x4_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_lib4
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_lib44c4, .-kernel_dtrsm_nt_rl_one_4x4_lib44c4
#endif





//                                            1      2          3          4             5          6        7          8        9          10      11
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4
	.type kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4, @function
kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4
_kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4
	.def kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d
	movq	ARG10, %r13 // m1
	movq	ARG11, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4, .-kernel_dtrsm_nt_rl_one_4x4_vs_lib44c4
#endif






//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nt_rl_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib4c4c
	.type kernel_dtrsm_nt_rl_one_4x4_lib4c4c, @function
kernel_dtrsm_nt_rl_one_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_lib4c4c
_kernel_dtrsm_nt_rl_one_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib4c4c
	.def kernel_dtrsm_nt_rl_one_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_lib4c4c, .-kernel_dtrsm_nt_rl_one_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c
	.type kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c, @function
kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c
_kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c
	.def kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11 // m1
	movq	ARG11, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c, .-kernel_dtrsm_nt_rl_one_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4        5             6          7        8          9        10         11
// void kernel_dtrsm_nt_rl_one_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib4ccc
	.type kernel_dtrsm_nt_rl_one_4x4_lib4ccc, @function
kernel_dtrsm_nt_rl_one_4x4_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_lib4ccc
_kernel_dtrsm_nt_rl_one_4x4_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib4ccc
	.def kernel_dtrsm_nt_rl_one_4x4_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_lib
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_lib4ccc, .-kernel_dtrsm_nt_rl_one_4x4_lib4ccc
#endif





//                                            1      2          3          4        5             6          7        8          9        10         11       12      13
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc
	.type kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc, @function
kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc
_kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc
	.def kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG13, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG13, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG13, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc, .-kernel_dtrsm_nt_rl_one_4x4_vs_lib4ccc
#endif





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nn_ru_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_4x4_lib4c4c
	.type kernel_dtrsm_nn_ru_inv_4x4_lib4c4c, @function
kernel_dtrsm_nn_ru_inv_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_4x4_lib4c4c
_kernel_dtrsm_nn_ru_inv_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_inv_4x4_lib4c4c
	.def kernel_dtrsm_nn_ru_inv_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_inv_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_inv_4x4_lib4c4c, .-kernel_dtrsm_nn_ru_inv_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c
	.type kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c, @function
kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c
_kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c
	.def kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c, .-kernel_dtrsm_nn_ru_inv_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nn_ru_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_one_4x4_lib4c4c
	.type kernel_dtrsm_nn_ru_one_4x4_lib4c4c, @function
kernel_dtrsm_nn_ru_one_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_one_4x4_lib4c4c
_kernel_dtrsm_nn_ru_one_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_one_4x4_lib4c4c
	.def kernel_dtrsm_nn_ru_one_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_one_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_one_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_one_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_one_4x4_lib4c4c, .-kernel_dtrsm_nn_ru_one_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c
	.type kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c, @function
kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c
_kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c
	.def kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nn_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nn_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_ONE_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_one_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_one_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c, .-kernel_dtrsm_nn_ru_one_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4             5          6        7          8        9          10
// void kernel_dtrsm_nt_ru_inv_4x4_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib44c4
	.type kernel_dtrsm_nt_ru_inv_4x4_lib44c4, @function
kernel_dtrsm_nt_ru_inv_4x4_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_lib44c4
_kernel_dtrsm_nt_ru_inv_4x4_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib44c4
	.def kernel_dtrsm_nt_ru_inv_4x4_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11 // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_lib4
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_lib44c4, .-kernel_dtrsm_nt_ru_inv_4x4_lib44c4
#endif





//                                            1      2          3          4             5          6        7          8        9          10                  11      12
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4(int k, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, double *inv_diag_E, int m1, in1 n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4
	.type kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4, @function
kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4
_kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4
	.def kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // beta
	movq	ARG5, %r11 // C
	movq	ARG6, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11 // inv_diag_E 
	movq	ARG12, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_vs_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4, .-kernel_dtrsm_nt_ru_inv_4x4_vs_lib44c4
#endif





//                                         1      2          3          4        5             6          7          8          9        10
// void kernel_dtrsm_nt_ru_inv_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib4c4c
	.type kernel_dtrsm_nt_ru_inv_4x4_lib4c4c, @function
kernel_dtrsm_nt_ru_inv_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_lib4c4c
_kernel_dtrsm_nt_ru_inv_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib4c4c
	.def kernel_dtrsm_nt_ru_inv_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_lib4c4c, .-kernel_dtrsm_nt_ru_inv_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10                  11      12
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c
	.type kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c, @function
kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c
_kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c
	.def kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG12, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG12, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG11, %r11  // m1
	movq	ARG12, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c, .-kernel_dtrsm_nt_ru_inv_4x4_vs_lib4c4c
#endif





//                                         1      2          3          4        5             6          7        8          9        10         11       12
// void kernel_dtrsm_nt_ru_inv_4x4_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib4ccc
	.type kernel_dtrsm_nt_ru_inv_4x4_lib4ccc, @function
kernel_dtrsm_nt_ru_inv_4x4_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_lib4ccc
_kernel_dtrsm_nt_ru_inv_4x4_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib4ccc
	.def kernel_dtrsm_nt_ru_inv_4x4_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C
	movq	ARG7, %r12 // ldc
	sall	$3, %r12d

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_lib4ccc, .-kernel_dtrsm_nt_ru_inv_4x4_lib4ccc
#endif





//                                            1      2          3          4        5             6          7        8          9        10         11       12                  13      14
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc(int k, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc
	.type kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc, @function
kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc
_kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc
	.def kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG14, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG14, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_vs_lib
#endif
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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc, .-kernel_dtrsm_nt_ru_inv_4x4_vs_lib4ccc
#endif





//                                         1      2          3          4        5             6          7          8          9
// void kernel_dtrsm_nt_ru_one_4x4_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_one_4x4_lib4c4c
	.type kernel_dtrsm_nt_ru_one_4x4_lib4c4c, @function
kernel_dtrsm_nt_ru_one_4x4_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_one_4x4_lib4c4c
_kernel_dtrsm_nt_ru_one_4x4_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_one_4x4_lib4c4c
	.def kernel_dtrsm_nt_ru_one_4x4_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_one_4x4_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // B
	movq	ARG4, %r13 // ldb
	sall	$3, %r13d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_one_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_one_4x4_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_one_4x4_lib4c4c, .-kernel_dtrsm_nt_ru_one_4x4_lib4c4c
#endif





//                                            1      2          3          4        5             6          7          8          9        10      11
// void kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c(int k, double *A, double *B, int ldb, double *beta, double *C, double *D, double *E, int lde, int m1, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c
	.type kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c, @function
kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c
_kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c
	.def kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x1_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x1_lib4c
#endif
#endif
	
	jmp		103f

100:

	movq	ARG11, %r14  // n1
	cmpl	$2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X2_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x2_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x2_lib4c
#endif
#endif

	jmp		103f

101:

	movq	ARG11, %r14  // n1
	cmpl	$3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X3_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x3_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x3_lib4c
#endif
#endif

	jmp		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4C
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4c
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4c
#endif
#endif

103:


	// call inner blender_loader nn

	movq	ARG5, %r10 // beta
	movq	ARG6, %r11 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_m1b_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_m1b_4x4_lib4
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_ONE_4X4_VS_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_one_4x4_vs_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_one_4x4_vs_lib
#endif
#endif


	// store

	movq	ARG7, %r10 // D
	movq	ARG10, %r11  // m1
	movq	ARG11, %r12  // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c, .-kernel_dtrsm_nt_ru_one_4x4_vs_lib4c4c
#endif





