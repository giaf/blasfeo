/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#define STACKSIZE 96

//.LSTR:
//	.string	"\n%d %d %d\n"

#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif

// common inner routine with file scope
//
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_4x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	movapd		0(%r11), %xmm8 // A[0]
	movapd 		16(%r11), %xmm9 // A[2]
	movapd		0(%r12), %xmm10 // B[0]

	xorpd		%xmm11, %xmm11
	movapd		%xmm11, %xmm12
	movapd		%xmm11, %xmm13
	movapd		%xmm11, %xmm14
	movapd		%xmm11, %xmm15


	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	addpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[4]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[4]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[6]


	// unroll 1
	addpd		%xmm14, %xmm3
	movapd		48(%r12), %xmm14 // B[6]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		64(%r12), %xmm10 // B[8]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		64(%r11), %xmm8 // A[8]
	mulpd		%xmm9, %xmm13
	movapd 		80(%r11), %xmm9 // A[10]


	// unroll 2
	addpd		%xmm14, %xmm3
	movapd		80(%r12), %xmm14 // B[10]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl		$4, %r10d

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		96(%r12), %xmm10 // B[12]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		96(%r11), %xmm8 // A[12]
	mulpd		%xmm9, %xmm13
	movapd 		112(%r11), %xmm9 // A[14]
	

	// unroll 3
	addpd		%xmm14, %xmm3
	movapd		112(%r12), %xmm14 // B[14]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	addq		$128, %r12 // B += 16

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq		$128, %r11 // A += 16

	addpd		%xmm10, %xmm1
	movapd		0(%r12), %xmm10 // B[0]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	cmpl		$4, %r10d

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		0(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
	movapd 		16(%r11), %xmm9 // A[2]


	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	addpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[4]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[4]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[6]


	// unroll 1
	addpd		%xmm14, %xmm3
	movapd		48(%r12), %xmm14 // B[6]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		64(%r12), %xmm10 // B[8]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		64(%r11), %xmm8 // A[8]
	mulpd		%xmm9, %xmm13
	movapd 		80(%r11), %xmm9 // A[10]


	// unroll 2
	addpd		%xmm14, %xmm3
	movapd		80(%r12), %xmm14 // B[10]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl		$4, %r10d

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	addpd		%xmm10, %xmm1
	movapd		96(%r12), %xmm10 // B[12]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		96(%r11), %xmm8 // A[12]
	mulpd		%xmm9, %xmm13
	movapd 		112(%r11), %xmm9 // A[14]
	

	// unroll 3
	addpd		%xmm14, %xmm3
	movapd		112(%r12), %xmm14 // B[14]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	addq		$128, %r12 // B += 16

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq		$128, %r11 // A += 16

	addpd		%xmm10, %xmm1
//	movapd		0(%r12), %xmm10 // B[0]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	cmpl		$4, %r10d

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
//	movapd 		0(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
//	movapd 		16(%r11), %xmm9 // A[2]


	// clean accumulators
	addpd		%xmm14, %xmm3
	addpd		%xmm11, %xmm7
	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	

	// unroll 0
	addpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	addpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl	$1, %r10d

	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq	$32, %r12

	addpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[0]
	addpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	addq	$32, %r11

	addpd		%xmm15, %xmm0
	addpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[2]

	cmpl	$0, %r10d

	jg		3b // clean up loop 


	// clean accumulators
	addpd		%xmm14, %xmm3
	addpd		%xmm11, %xmm7
	addpd		%xmm12, %xmm2
	addpd		%xmm13, %xmm6


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nt_4x4_lib4, @function
inner_kernel_dgemm_sub_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nt_4x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	movapd		0(%r11), %xmm8 // A[0]
	movapd 		16(%r11), %xmm9 // A[2]
	movapd		0(%r12), %xmm10 // B[0]

	xorpd		%xmm11, %xmm11
	movapd		%xmm11, %xmm12
	movapd		%xmm11, %xmm13
	movapd		%xmm11, %xmm14
	movapd		%xmm11, %xmm15

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	subpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[4]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[4]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[6]


	// unroll 1
	subpd		%xmm14, %xmm3
	movapd		48(%r12), %xmm14 // B[6]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		64(%r12), %xmm10 // B[8]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		64(%r11), %xmm8 // A[8]
	mulpd		%xmm9, %xmm13
	movapd 		80(%r11), %xmm9 // A[10]


	// unroll 2
	subpd		%xmm14, %xmm3
	movapd		80(%r12), %xmm14 // B[10]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl		$4, %r10d

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		96(%r12), %xmm10 // B[12]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		96(%r11), %xmm8 // A[12]
	mulpd		%xmm9, %xmm13
	movapd 		112(%r11), %xmm9 // A[14]
	

	// unroll 3
	subpd		%xmm14, %xmm3
	movapd		112(%r12), %xmm14 // B[14]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	addq		$128, %r12 // B += 16

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq		$128, %r11 // A += 16

	subpd		%xmm10, %xmm1
	movapd		0(%r12), %xmm10 // B[0]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	cmpl		$4, %r10d

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		0(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
	movapd 		16(%r11), %xmm9 // A[2]


	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	subpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[4]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[4]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[6]


	// unroll 1
	subpd		%xmm14, %xmm3
	movapd		48(%r12), %xmm14 // B[6]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		64(%r12), %xmm10 // B[8]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		64(%r11), %xmm8 // A[8]
	mulpd		%xmm9, %xmm13
	movapd 		80(%r11), %xmm9 // A[10]


	// unroll 2
	subpd		%xmm14, %xmm3
	movapd		80(%r12), %xmm14 // B[10]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl		$4, %r10d

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13

	subpd		%xmm10, %xmm1
	movapd		96(%r12), %xmm10 // B[12]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		96(%r11), %xmm8 // A[12]
	mulpd		%xmm9, %xmm13
	movapd 		112(%r11), %xmm9 // A[14]
	

	// unroll 3
	subpd		%xmm14, %xmm3
	movapd		112(%r12), %xmm14 // B[14]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	addq		$128, %r12 // B += 16

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq		$128, %r11 // A += 16

	subpd		%xmm10, %xmm1
//	movapd		0(%r12), %xmm10 // B[0]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	cmpl		$4, %r10d

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
//	movapd 		0(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
//	movapd 		16(%r11), %xmm9 // A[2]


	// update accumulators
	subpd		%xmm14, %xmm3
	subpd		%xmm11, %xmm7
	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	

	// unroll 0
	subpd		%xmm14, %xmm3
	movapd		16(%r12), %xmm14 // B[2]
	subpd		%xmm11, %xmm7
	movapd		%xmm10, %xmm11
	pshufd		$0x4e, %xmm10, %xmm15
	mulpd		%xmm8, %xmm10
	mulpd		%xmm9, %xmm11
	subl	$1, %r10d

	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6
	movapd		%xmm15, %xmm13
	mulpd		%xmm8, %xmm15
	mulpd		%xmm9, %xmm13
	addq	$32, %r12

	subpd		%xmm10, %xmm1
	movapd		32(%r12), %xmm10 // B[0]
	subpd		%xmm11, %xmm5
	movapd		%xmm14, %xmm11
	pshufd		$0x4e, %xmm14, %xmm12
	mulpd		%xmm8, %xmm14
	mulpd		%xmm9, %xmm11
	addq	$32, %r11

	subpd		%xmm15, %xmm0
	subpd		%xmm13, %xmm4
	movapd		%xmm12, %xmm13
	mulpd		%xmm8, %xmm12
	movapd 		32(%r11), %xmm8 // A[0]
	mulpd		%xmm9, %xmm13
	movapd 		48(%r11), %xmm9 // A[2]

	cmpl	$0, %r10d

	jg		3b // clean up loop 


	// update accumulators
	subpd		%xmm14, %xmm3
	subpd		%xmm11, %xmm7
	subpd		%xmm12, %xmm2
	subpd		%xmm13, %xmm6


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nt_4x4_lib4, .-inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_4x4_lib4, @function
inner_edge_dtrmm_nt_ru_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_4x4_lib4:
#endif
#endif
	
	movapd			0(%r10), %xmm8
	movapd			16(%r10), %xmm9
	movddup			0(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4

	movapd			32(%r10), %xmm8
	movapd			48(%r10), %xmm9
	movddup			32(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	movddup			40(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5

	movapd			64(%r10), %xmm8
	movapd			80(%r10), %xmm9
	movddup			64(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	movddup			72(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5
	movddup			80(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm2
	addpd			%xmm13, %xmm6

	movapd			96(%r10), %xmm8
	movapd			112(%r10), %xmm9
	movddup			96(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	movddup			104(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5
	movddup			112(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm2
	addpd			%xmm13, %xmm6
	movddup			120(%r11), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm3
	addpd			%xmm13, %xmm7

	addq			$128, %r10
	addq			$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_4x4_lib4, .-inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_4x4_vs_lib4, @function
inner_edge_dtrmm_nt_ru_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_4x4_vs_lib4:
#endif
#endif
	
	movapd			0(%r11), %xmm8
	movapd			16(%r11), %xmm9
	subl			$1, %r10d
	movddup			0(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	movapd			0(%r11), %xmm8
	movapd			16(%r11), %xmm9
	subl			$1, %r10d
	movddup			0(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	addq			$32, %r11
	movddup			8(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	movapd			0(%r11), %xmm8
	movapd			16(%r11), %xmm9
	subl			$1, %r10d
	movddup			0(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	movddup			8(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5
	addq			$32, %r11
	movddup			16(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm2
	addpd			%xmm13, %xmm6
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	movapd			0(%r11), %xmm8
	movapd			16(%r11), %xmm9
	subl			$1, %r10d
	movddup			0(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm0
	addpd			%xmm13, %xmm4
	movddup			8(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm1
	addpd			%xmm13, %xmm5
	movddup			16(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm2
	addpd			%xmm13, %xmm6
	addq			$32, %r11
	movddup			24(%r12), %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm8, %xmm12
	mulpd			%xmm9, %xmm13
	addpd			%xmm12, %xmm3
	addpd			%xmm13, %xmm7
	addq			$32, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_4x4_vs_lib4, .-inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// blend
//
// input arguments:
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

// output arguments:
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_4x4_lib4, @function
inner_blend_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_4x4_lib4:
#endif
#endif
	
	movapd	%xmm0, %xmm8
	movsd	%xmm1, %xmm0
	movsd	%xmm8, %xmm1

	movapd	%xmm2, %xmm8
	movsd	%xmm3, %xmm2
	movsd	%xmm8, %xmm3

	movapd	%xmm4, %xmm8
	movsd	%xmm5, %xmm4
	movsd	%xmm8, %xmm5

	movapd	%xmm6, %xmm8
	movsd	%xmm7, %xmm6
	movsd	%xmm8, %xmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_4x4_lib4, .-inner_blend_4x4_lib4
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
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_lib4, @function
inner_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_lib4:
#endif
#endif
	
	// alpha
	movddup	0(%r10), %xmm15

	mulpd	%xmm15, %xmm0
	mulpd	%xmm15, %xmm1
	mulpd	%xmm15, %xmm2
	mulpd	%xmm15, %xmm3
	mulpd	%xmm15, %xmm4
	mulpd	%xmm15, %xmm5
	mulpd	%xmm15, %xmm6
	mulpd	%xmm15, %xmm7


	// beta
	movddup	0(%r11), %xmm14

	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm0
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm4
	movapd		32(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm1
	movapd		48(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm5
	movapd		64(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm2
	movapd		80(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm6
	movapd		96(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm3
	movapd		112(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_lib4, .-inner_scale_ab_4x4_lib4
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
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_4x4_lib4, @function
inner_blend_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_4x4_lib4:
#endif
#endif
	
	movapd	%xmm0, %xmm8
	movsd	%xmm1, %xmm0
	movsd	%xmm8, %xmm1

	movapd	%xmm2, %xmm8
	movsd	%xmm3, %xmm2
	movsd	%xmm8, %xmm3

	movapd	%xmm4, %xmm8
	movsd	%xmm5, %xmm4
	movsd	%xmm8, %xmm5

	movapd	%xmm6, %xmm8
	movsd	%xmm7, %xmm6
	movsd	%xmm8, %xmm7

	// alpha
	movddup	0(%r10), %xmm15

	mulpd	%xmm15, %xmm0
	mulpd	%xmm15, %xmm1
	mulpd	%xmm15, %xmm2
	mulpd	%xmm15, %xmm3
	mulpd	%xmm15, %xmm4
	mulpd	%xmm15, %xmm5
	mulpd	%xmm15, %xmm6
	mulpd	%xmm15, %xmm7


	// beta
	movddup	0(%r11), %xmm14

	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm0
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm4
	movapd		32(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm1
	movapd		48(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm5
	movapd		64(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm2
	movapd		80(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm6
	movapd		96(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm3
	movapd		112(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_4x4_lib4, .-inner_blend_scale_ab_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blender for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

// output arguments:
// r10   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_4x4_lib4, @function
inner_blend_scale_11_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_11_4x4_lib4:
#endif
#endif	
	
	movapd	%xmm0, %xmm8
	movsd	%xmm1, %xmm0
	movsd	%xmm8, %xmm1

	movapd	%xmm2, %xmm8
	movsd	%xmm3, %xmm2
	movsd	%xmm8, %xmm3

	movapd	%xmm4, %xmm8
	movsd	%xmm5, %xmm4
	movsd	%xmm8, %xmm5

	movapd	%xmm6, %xmm8
	movsd	%xmm7, %xmm6
	movsd	%xmm8, %xmm7


	movapd		0(%r10), %xmm15
	addpd		%xmm15, %xmm0
	movapd		16(%r10), %xmm15
	addpd		%xmm15, %xmm4
	movapd		32(%r10), %xmm15
	addpd		%xmm15, %xmm1
	movapd		48(%r10), %xmm15
	addpd		%xmm15, %xmm5
	movapd		64(%r10), %xmm15
	addpd		%xmm15, %xmm2
	movapd		80(%r10), %xmm15
	addpd		%xmm15, %xmm6
	movapd		96(%r10), %xmm15
	addpd		%xmm15, %xmm3
	movapd		112(%r10), %xmm15
	addpd		%xmm15, %xmm7

	ret

#if MACRO_LEVEL>=1
	.endm
#else
#if defined(OS_LINUX)
	.size	inner_blend_scale_11_4x4_lib4, .-inner_blend_scale_11_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10  <- inv_diag_E
// r11d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// r11d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_4x4_vs_lib4, @function
inner_edge_dpotrf_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_4x4_vs_lib4:
#endif
#endif
	
	xorpd			%xmm15, %xmm15 // 0.0

	movsd			%xmm0, %xmm13
	ucomisd			%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe				1f
	sqrtsd			%xmm13, %xmm13
#if defined(OS_LINUX)
	movsd			.LC04(%rip), %xmm12 // 1.0
#elif defined(OS_MAC)
	movsd			LC04(%rip), %xmm12 // 1.0
#endif
	divsd			%xmm13, %xmm12
2:
	cmpl			$2, %r11d
	movsd			%xmm12, 0(%r10)
	movddup			%xmm12, %xmm12
	mulpd			%xmm12, %xmm0
	mulpd			%xmm12, %xmm4

	jl				0f // ret

	movapd			%xmm0, %xmm12
	shufpd			$0x3, %xmm12, %xmm12
	movapd			%xmm12, %xmm13
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm1
	subpd			%xmm13, %xmm5
	movapd			%xmm1, %xmm13
	shufpd			$0x3, %xmm13, %xmm13 // 0x1 ???
	ucomisd			%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe				3f
	sqrtsd			%xmm13, %xmm13
#if defined(OS_LINUX)
	movsd			.LC04(%rip), %xmm12 // 1.0
#elif defined(OS_MAC)
	movsd			LC04(%rip), %xmm12 // 1.0
#endif
	divsd			%xmm13, %xmm12
4:
	cmpl			$3, %r11d
	movsd			%xmm12, 8(%r10)
	movddup			%xmm12, %xmm12
	mulpd			%xmm12, %xmm1
	mulpd			%xmm12, %xmm5

	jl				0f // ret

	movddup			%xmm4, %xmm12
	movddup			%xmm5, %xmm13
	mulpd			%xmm4, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm6
	subpd			%xmm13, %xmm6
	movsd			%xmm6, %xmm13
	ucomisd			%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe				5f
	sqrtsd			%xmm13, %xmm13
#if defined(OS_LINUX)
	movsd			.LC04(%rip), %xmm12 // 1.0
#elif defined(OS_MAC)
	movsd			LC04(%rip), %xmm12 // 1.0
#endif
	divsd			%xmm13, %xmm12
6:
	cmpl			$4, %r11d
	movsd			%xmm12, 16(%r10)
	movddup			%xmm12, %xmm12
	mulpd			%xmm12, %xmm6

	jl				0f // ret

	movapd			%xmm4, %xmm12
	movapd			%xmm5, %xmm13
	movapd			%xmm6, %xmm14
	shufpd			$0x3, %xmm12, %xmm12
	shufpd			$0x3, %xmm13, %xmm13
	shufpd			$0x3, %xmm14, %xmm14
	mulpd			%xmm4, %xmm12
	mulpd			%xmm5, %xmm13
	mulpd			%xmm6, %xmm14
	subpd			%xmm12, %xmm7
	subpd			%xmm13, %xmm7
	subpd			%xmm14, %xmm7
	movapd			%xmm7, %xmm13
	shufpd			$0x3, %xmm13, %xmm13
	ucomisd			%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe				7f
	sqrtsd			%xmm13, %xmm13
#if defined(OS_LINUX)
	movsd			.LC04(%rip), %xmm12 // 1.0
#elif defined(OS_MAC)
	movsd			LC04(%rip), %xmm12 // 1.0
#endif
	divsd			%xmm13, %xmm12
8:
	movsd			%xmm12, 24(%r10)
	movddup			%xmm12, %xmm12
	mulpd			%xmm12, %xmm7

	jmp		0f
	
1:
	xorpd	%xmm12, %xmm12
	jmp		2b

3:
	xorpd	%xmm12, %xmm12
	jmp		4b

5:
	xorpd	%xmm12, %xmm12
	jmp		6b

7:
	xorpd	%xmm12, %xmm12
	jmp		8b

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_4x4_vs_lib4, .-inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_lib4:
#endif
#endif
	
	movddup			0(%r11), %xmm13
	mulpd			%xmm13, %xmm0
	mulpd			%xmm13, %xmm4

	movddup			8(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm13
	mulpd			%xmm4, %xmm12
	subpd			%xmm13, %xmm1
	subpd			%xmm12, %xmm5
	movddup			8(%r11), %xmm13
	mulpd			%xmm13, %xmm1
	mulpd			%xmm13, %xmm5

	movddup			16(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			48(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			16(%r11), %xmm13
	mulpd			%xmm13, %xmm2
	mulpd			%xmm13, %xmm6

	movddup			24(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			56(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			88(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm2, %xmm12
	mulpd			%xmm6, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			24(%r11), %xmm13
	mulpd			%xmm13, %xmm3
	mulpd			%xmm13, %xmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
#endif
#endif
	
	movddup			0(%r11), %xmm13
	cmpl			$2, %r12d
	mulpd			%xmm13, %xmm0
	mulpd			%xmm13, %xmm4

	jl				0f // ret

	movddup			8(%r10), %xmm13
	cmpl			$3, %r12d
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm13
	mulpd			%xmm4, %xmm12
	subpd			%xmm13, %xmm1
	subpd			%xmm12, %xmm5
	movddup			8(%r11), %xmm13
	mulpd			%xmm13, %xmm1
	mulpd			%xmm13, %xmm5

	jl				0f // ret

	movddup			16(%r10), %xmm13
	cmpl			$4, %r12d
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			48(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			16(%r11), %xmm13
	mulpd			%xmm13, %xmm2
	mulpd			%xmm13, %xmm6

	jl				0f // ret

	movddup			24(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			56(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			88(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm2, %xmm12
	mulpd			%xmm6, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	movddup			24(%r11), %xmm13
	mulpd			%xmm13, %xmm3
	mulpd			%xmm13, %xmm7

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d01 d11]
// xmm2  <- [d02 d12]
// xmm3  <- [d03 d13]
// xmm4  <- [d20 d30]
// xmm5  <- [d21 d31]
// xmm6  <- [d22 d32]
// xmm7  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_lib4, @function
inner_store_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_lib4:
#endif
#endif
	
	movapd %xmm0,   0(%r10)
	movapd %xmm4,  16(%r10)
	movapd %xmm1,  32(%r10)
	movapd %xmm5,  48(%r10)
	movapd %xmm2,  64(%r10)
	movapd %xmm6,  80(%r10)
	movapd %xmm3,  96(%r10)
	movapd %xmm7, 112(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_vs_lib4, @function
inner_store_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_vs_lib4:
#endif
#endif
	
	cmpl		$2, %r11d
	jg			1f
	je			0f

	// km==1
	cmpl		$2, %r12d
	movsd		%xmm0,  0(%r10)
	jl			4f // end
	cmpl		$3, %r12d
	movsd		%xmm1, 32(%r10)
	jl			4f // end
	movsd		%xmm2, 64(%r10)
	je			4f // end
	movsd		%xmm3, 96(%r10)

	jmp		4f

0:
	// km==2
	cmpl		$2, %r12d
	movapd		%xmm0,  0(%r10)
	jl			4f // end
	cmpl		$3, %r12d
	movapd		%xmm1, 32(%r10)
	jl			4f // end
	movapd		%xmm2, 64(%r10)
	je			4f // end
	movapd		%xmm3, 96(%r10)

	jmp		4f

1:
	cmpl		$3, %r11d
	jg			2f

	// km==3
	cmpl		$2, %r12d
	movapd		%xmm0,   0(%r10)
	movsd		%xmm4,  16(%r10)
	jl			4f // end
	cmpl		$3, %r12d
	movapd		%xmm1,  32(%r10)
	movsd		%xmm5,  48(%r10)
	jl			4f // end
	movapd		%xmm2,  64(%r10)
	movsd		%xmm6,  80(%r10)
	je			4f // end
	movapd		%xmm3,  96(%r10)
	movsd		%xmm7, 112(%r10)

	jmp		4f

2:
	// km==4
	cmpl		$2, %r12d
	movapd		%xmm0,   0(%r10)
	movapd		%xmm4,  16(%r10)
	jl			4f // end
	cmpl		$3, %r12d
	movapd		%xmm1,  32(%r10)
	movapd		%xmm5,  48(%r10)
	jl			4f // end
	movapd		%xmm2,  64(%r10)
	movapd		%xmm6,  80(%r10)
	je			4f // end
	movapd		%xmm3,  96(%r10)
	movapd		%xmm7, 112(%r10)

4:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_vs_lib4, .-inner_store_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// store n lower triangular
//
// input arguments:
// r10   <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10   <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_4x4_lib4, @function
inner_store_l_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_l_4x4_lib4:
#endif
#endif
	
	vmovapd		%xmm0,   0(%r10)
	vmovapd		%xmm1,  16(%r10)
	vmovsd		32(%r10), %xmm15
	vmovsd		%xmm15, %xmm2, %xmm2
	vmovapd		%xmm2,  32(%r10)
	vmovapd		%xmm3,  48(%r10)
//	vmovapd		%xmm4,  64(%r10)
	vmovapd		%xmm5,  80(%r10)
//	vmovapd		%xmm6,  96(%r10)
	vmovsd		112(%r10), %xmm15
	vmovsd		%xmm15, %xmm7, %xmm7
	vmovapd		%xmm7, 112(%r10)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_lib4, .-inner_store_l_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n vs lower triangular
//
// input arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_4x4_vs_lib4, @function
inner_store_l_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_l_4x4_vs_lib4:
#endif
#endif
	
	cmpl		$2, %r11d
	jg			1f
	je			0f

	// km==1
	movsd		%xmm0,  0(%r10)

	jmp		3f

0:
	// km==2
	cmpl		$2, %r12d
	movapd		%xmm0,  0(%r10)
	jl			3f // end
	movsd		32(%r10), %xmm15
	movsd		%xmm15, %xmm1
	movapd		%xmm1, 32(%r10)

	jmp		3f

1:
	cmpl		$3, %r11d
	jg			2f

	// km==3
	cmpl		$2, %r12d
	movapd		%xmm0,   0(%r10)
	movsd		%xmm4,  16(%r10)
	jl			3f // end
	cmpl		$3, %r12d
	movsd		32(%r10), %xmm15
	movsd		%xmm15, %xmm1
	movapd		%xmm1,  32(%r10)
	movsd		%xmm5,  48(%r10)
	jl			3f // end
//	movapd		%xmm2,  64(%r10)
	movsd		%xmm6,  80(%r10)

	jmp		3f

2:
	// km==3
	cmpl		$2, %r12d
	movapd		%xmm0,   0(%r10)
	movapd		%xmm4,  16(%r10)
	jl			3f // end
	cmpl		$3, %r12d
	movsd		32(%r10), %xmm15
	movsd		%xmm15, %xmm1
	movapd		%xmm1,  32(%r10)
	movapd		%xmm5,  48(%r10)
	jl			3f // end
//	movapd		%xmm2,  64(%r10)
	movapd		%xmm6,  80(%r10)
	je			3f // end
//	movapd		%xmm3,  96(%r10)
	movsd		112(%r10), %xmm15
	movsd		%xmm15, %xmm7
	movapd		%xmm7, 112(%r10)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_vs_lib4, .-inner_store_l_4x4_vs_lib4
#endif
#endif





//                               rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dgemm_nt_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_lib4
	.type kernel_dgemm_nt_4x4_lib4, @function
kernel_dgemm_nt_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_lib4
_kernel_dgemm_nt_4x4_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dgemm_nt_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_vs_lib4
	.type kernel_dgemm_nt_4x4_vs_lib4, @function
kernel_dgemm_nt_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_vs_lib4
_kernel_dgemm_nt_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_vs_lib4, .-kernel_dgemm_nt_4x4_vs_lib4
#endif





//                                 rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dsyrk_nt_l_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_lib4
	.type kernel_dsyrk_nt_l_4x4_lib4, @function
kernel_dsyrk_nt_l_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_lib4
_kernel_dsyrk_nt_l_4x4_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D


#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib4
#endif
#endif



	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_4x4_lib4, .-kernel_dsyrk_nt_l_4x4_lib4
#endif





//                                    rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dsyrk_nt_l_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_vs_lib4
_kernel_dsyrk_nt_l_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 


#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib4
#endif
#endif



	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_4x4_vs_lib4, .-kernel_dsyrk_nt_l_4x4_vs_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dtrmm_nt_ru_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_lib4
	.type kernel_dtrmm_nt_ru_4x4_lib4, @function
kernel_dtrmm_nt_ru_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_lib4
_kernel_dtrmm_nt_ru_4x4_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs


#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movq	%rdx, %r10
	movq	%rcx, %r11

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_4x4_lib4, .-kernel_dtrmm_nt_ru_4x4_lib4
#endif





//                                     rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dtrmm_nt_ru_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);
// it holds k==kn !!!!!

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_vs_lib4
	.type kernel_dtrmm_nt_ru_4x4_vs_lib4, @function
kernel_dtrmm_nt_ru_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_vs_lib4
_kernel_dtrmm_nt_ru_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movq	%rcx, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
#endif


	// call inner loader nn

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_4x4_vs_lib4, .-kernel_dtrmm_nt_ru_4x4_vs_lib4
#endif





//                                     edi    rsi        rdx        ecx        r8         r9                  rsp+8   rsp+16
// void kernel_dpotrf_nt_l_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dpotrf_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_4x4_vs_lib4
_kernel_dpotrf_nt_l_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	%r9, %r10  // inv_diag_D 
	movl	STACKSIZE + 16(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dpotrf_nt_l_4x4_vs_lib4
#endif





//                                           edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24              rsp+32  rsp+40
// void kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
_kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 24(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 40(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10  // D 
	movl	STACKSIZE + 32(%rsp), %r11d // km 
	movl	STACKSIZE + 40(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx        r8         r9         rsp+8  
// void kernel_dtrsm_nt_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib4
_kernel_dtrsm_nt_rl_inv_4x4_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_lib4
#endif





//                                            edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24     rsp+32
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10   // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx        r8       r9           rsp+8               rsp+16  rsp+24
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
#endif





//                                               edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)



	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10  // C 

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D 
	movl	STACKSIZE + 40(%rsp), %r11d // km 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
#endif
#endif


	// epilogue


	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#endif
#if 1
#if defined(OS_LINUX)
	.align 32
.LC00: // { -1 -1 -1 1 }
#elif defined(OS_MAC)
LC00: // { -1 -1 -1 1 }
	.align 5
#endif
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	1
#if defined(OS_LINUX)
	.align 32
.LC01: // { -1 -1 -1 -1 }
#elif defined(OS_MAC)
LC01: // { -1 -1 -1 -1 }
	.align 5
#endif
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	-1
#endif
#if defined(OS_LINUX)
	.align 32
.LC02: // { 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
LC02: // { 3.5 2.5 1.5 0.5 }
	.align 5
#endif
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
#if defined(OS_LINUX)
	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
#elif defined(OS_MAC)
LC03: // { 7.5 6.5 5.5 4.5 }
	.align 5
#endif
	.long	0
	.long	1074921472
	.long	0
	.long	1075183616
	.long	0
	.long	1075445760
	.long	0
	.long	1075707904
#if defined(OS_LINUX)
	.align 32
.LC04: // { 1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
LC04: // { 1.0 1.0 1.0 1.0 }
	.align 5
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

