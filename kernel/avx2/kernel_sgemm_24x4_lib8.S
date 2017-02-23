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
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
// ymm4  <- []
// ymm5  <- []
// ymm6  <- []
// ymm7  <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*k*sizeof(double)
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
// ymm4  <- []
// ymm5  <- []
// ymm6  <- []
// ymm7  <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_SGEMM_ADD_NT_24X4_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgemm_add_nt_24x4_lib8, @function
inner_kernel_sgemm_add_nt_24x4_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgemm_add_nt_24x4_lib8:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(float)
	movq	%r15, %rax // A2 <- A1
	addq	%r12, %rax // A2 <- A1 + 4*sda*sizeof(float)

	// preload
	vbroadcastf128	0(%r13), %ymm12 // B
	vmovaps			0(%r11), %ymm13 // A0
	vmovaps			0(%r15), %ymm14 // A1
	vmovaps			0(%rax), %ymm15 // A2

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			32(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			32(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	32(%r13), %ymm12 // B
	vmovaps			32(%rax), %ymm15 // A2


	// unroll 1
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			64(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			64(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	64(%r13), %ymm12 // B
	vmovaps			64(%rax), %ymm15 // A2


	// unroll 2
	subl	$4, %r10d
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			96(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			96(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	96(%r13), %ymm12 // B
	vmovaps			96(%rax), %ymm15 // A2


	// unroll 3
	addq	$128, %r13
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	addq	$128, %r11
	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	addq	$128, %r15
	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	addq	$128, %rax
	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			0(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			0(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	0(%r13), %ymm12 // B
	vmovaps			0(%rax), %ymm15 // A2


	cmpl	$4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			32(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			32(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	32(%r13), %ymm12 // B
	vmovaps			32(%rax), %ymm15 // A2


	// unroll 1
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			64(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			64(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	64(%r13), %ymm12 // B
	vmovaps			64(%rax), %ymm15 // A2


	// unroll 2
	subl	$4, %r10d
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vmovaps			96(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vmovaps			96(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	vbroadcastf128	96(%r13), %ymm12 // B
	vmovaps			96(%rax), %ymm15 // A2


	// unroll 3
	addq	$128, %r13
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	vpermilps		$0xb1, %ymm12, %ymm12

	addq	$128, %r11
	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	vpermilps		$0x4e, %ymm12, %ymm12

	addq	$128, %r15
	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	vpermilps		$0xb1, %ymm12, %ymm12

	addq	$128, %rax
	vfmadd231ps		%ymm13, %ymm12, %ymm3
//	vmovaps			0(%r11), %ymm13 // A0
	vfmadd231ps		%ymm14, %ymm12, %ymm7
//	vmovaps			0(%r15), %ymm14 // A1
	vfmadd231ps		%ymm15, %ymm12, %ymm11
//	vbroadcastf128	0(%r13), %ymm12 // B
//	vmovaps			0(%rax), %ymm15 // A2


//	cmpl	$4, %r10d
	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vbroadcastf128	32(%r13), %ymm12 // B
	vmovaps			32(%r11), %ymm13 // A0
	vmovaps			32(%r15), %ymm14 // A1
	vmovaps			32(%rax), %ymm15 // A2
	vfmadd231ps		%ymm13, %ymm12, %ymm0
	vfmadd231ps		%ymm14, %ymm12, %ymm4
	vfmadd231ps		%ymm15, %ymm12, %ymm8
	subl	$1, %r10d

	vpermilps		$0xb1, %ymm12, %ymm12
	vfmadd231ps		%ymm13, %ymm12, %ymm1
	vfmadd231ps		%ymm14, %ymm12, %ymm5
	vfmadd231ps		%ymm15, %ymm12, %ymm9
	addq	$32, %r11

	vpermilps		$0x4e, %ymm12, %ymm12
	vfmadd231ps		%ymm13, %ymm12, %ymm2
	vfmadd231ps		%ymm14, %ymm12, %ymm6
	vfmadd231ps		%ymm15, %ymm12, %ymm10
	addq	$32, %r13

	vpermilps		$0xb1, %ymm12, %ymm12
	vfmadd231ps		%ymm13, %ymm12, %ymm3
	vfmadd231ps		%ymm14, %ymm12, %ymm7
	vfmadd231ps		%ymm15, %ymm12, %ymm11
	addq	$32, %r15


	cmpl	$0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgemm_add_nt_24x4_lib8, .-inner_kernel_sgemm_add_nt_24x4_lib8
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- 4*sdc*sizeof(float)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- 4*sdc*sizeof(float)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_24X4_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_24x4_lib8, @function
inner_blend_scale_ab_24x4_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_ab_24x4_lib8:
#endif
#endif
	
	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm0, %ymm15, %ymm0
	vmulps		%ymm1, %ymm15, %ymm1
	vmulps		%ymm2, %ymm15, %ymm2
	vmulps		%ymm3, %ymm15, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm4, %ymm15, %ymm4
	vmulps		%ymm5, %ymm15, %ymm5
	vmulps		%ymm6, %ymm15, %ymm6
	vmulps		%ymm7, %ymm15, %ymm7

	vblendps	$0xaa, %ymm9, %ymm8, %ymm12
	vblendps	$0x55, %ymm9, %ymm8, %ymm13
	vblendps	$0xaa, %ymm11, %ymm10, %ymm14
	vblendps	$0x55, %ymm11, %ymm10, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm8
	vblendps	$0x33, %ymm15, %ymm12, %ymm10
	vblendps	$0xcc, %ymm14, %ymm13, %ymm9
	vblendps	$0x33, %ymm14, %ymm13, %ymm11

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm8, %ymm15, %ymm8
	vmulps		%ymm9, %ymm15, %ymm9
	vmulps		%ymm10, %ymm15, %ymm10
	vmulps		%ymm11, %ymm15, %ymm11

	// beta
	vbroadcastss	0(%r11), %ymm14

	vxorps		%ymm15, %ymm15, %ymm15 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	movq	%r12, %r15 // C1 <- C0
	addq	%r13, %r15 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r15, %rax // C2 <- C1
	addq	%r13, %rax // C2 <- C1 + 4*sdc*sizeof(double)

	vmovaps		0(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm0
	vmovaps		32(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm1
	vmovaps		64(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm2
	vmovaps		96(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm3

	vmovaps		0(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm4
	vmovaps		32(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm5
	vmovaps		64(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm6
	vmovaps		96(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm7

	vmovaps		0(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm8
	vmovaps		32(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm9
	vmovaps		64(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm10
	vmovaps		96(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_24x4_lib8, .-inner_blend_scale_ab_24x4_lib8
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_24X4_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_24x4_gen_lib8, @function
inner_blend_scale_ab_24x4_gen_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_ab_24x4_gen_lib8:
#endif
#endif
	
	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm0, %ymm15, %ymm0
	vmulps		%ymm1, %ymm15, %ymm1
	vmulps		%ymm2, %ymm15, %ymm2
	vmulps		%ymm3, %ymm15, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm4, %ymm15, %ymm4
	vmulps		%ymm5, %ymm15, %ymm5
	vmulps		%ymm6, %ymm15, %ymm6
	vmulps		%ymm7, %ymm15, %ymm7

	vblendps	$0xaa, %ymm9, %ymm8, %ymm12
	vblendps	$0x55, %ymm9, %ymm8, %ymm13
	vblendps	$0xaa, %ymm11, %ymm10, %ymm14
	vblendps	$0x55, %ymm11, %ymm10, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm8
	vblendps	$0x33, %ymm15, %ymm12, %ymm10
	vblendps	$0xcc, %ymm14, %ymm13, %ymm9
	vblendps	$0x33, %ymm14, %ymm13, %ymm11

	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm8, %ymm15, %ymm8
	vmulps		%ymm9, %ymm15, %ymm9
	vmulps		%ymm10, %ymm15, %ymm10
	vmulps		%ymm11, %ymm15, %ymm11

	// beta
	vbroadcastss	0(%r11), %ymm14

	vxorps		%ymm15, %ymm15, %ymm15 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			3f // end

	movq	%r13, %r15 // C1 <- C0
	addq	%r14, %r15 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r15, %rax // C2 <- C1
	addq	%r14, %rax // C2 <- C1 + 4*sdc*sizeof(double)

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovaps		0(%r13), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm0
	vmovaps		32(%r13), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm1
	vmovaps		64(%r13), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm2
	vmovaps		96(%r13), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm3

	vmovaps		0(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm4
	vmovaps		32(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm5
	vmovaps		64(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm6
	vmovaps		96(%r15), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm7

	vmovaps		0(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm8
	vmovaps		32(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm9
	vmovaps		64(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm10
	vmovaps		96(%rax), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm11

	jmp		7f

0:

	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%rax, %rbx // C1
	addq	%r14, %rbx // C2 <- C1 + 4*sdc*sizeof(double)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:


#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_24x4_gen_lib8, .-inner_blend_scale_ab_24x4_gen_lib8
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(float)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(float)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_24X4_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_24x4_lib8, @function
inner_store_24x4_lib8:
#elif defined(OS_MAC)
_inner_store_24x4_lib8:
#endif
#endif
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)
	movq	%r15, %rax // D2 <- D1
	addq	%r11, %rax // D2 <- D1 + 4*sdd*sizeof(double)

	vmovaps 	%ymm0,  0(%r10)
	vmovaps 	%ymm1, 32(%r10)
	vmovaps 	%ymm2, 64(%r10)
	vmovaps		%ymm3, 96(%r10)

	vmovaps 	%ymm4,  0(%r15)
	vmovaps 	%ymm5, 32(%r15)
	vmovaps 	%ymm6, 64(%r15)
	vmovaps 	%ymm7, 96(%r15)

	vmovaps 	%ymm8,  0(%rax)
	vmovaps 	%ymm9, 32(%rax)
	vmovaps 	%ymm10, 64(%rax)
	vmovaps 	%ymm11, 96(%rax)


#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_24x4_lib8, .-inner_store_24x4_lib8
#endif
#endif





// common inner routine with file scope
//
// store n generalized
//
// input arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n0 // col index: start from (inc)
// rax  <- n1 // col index: up to (exc)
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
//
// output arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n1-n0
// rax  <- n1-n0
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_24X4_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_24x4_gen_lib8, @function
inner_store_24x4_gen_lib8:
#elif defined(OS_MAC)
_inner_store_24x4_gen_lib8:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm14, %xmm14
	vcvtsi2ss	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovups		.LC00(%rip), %ymm12
	vmovups		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %ymm12
	vmovups		LC02(%rip), %ymm13
#endif
	vshufps		$0x00, %xmm14, %xmm14, %xmm14
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm14, %ymm14, %ymm14
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm12, %ymm14, %ymm14
	vsubps		%ymm15, %ymm13, %ymm15

	// shift D and sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm9, %ymm8
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm6, %ymm5
	vmovaps		%ymm10, %ymm9
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm7, %ymm6
	vmovaps		%ymm11, %ymm10
	addq		$32, %r11

	cmpl	$1, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm9, %ymm8
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm6, %ymm5
	vmovaps		%ymm10, %ymm9
	addq		$32, %r11

	cmpl	$2, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm9, %ymm8
	addq		$32, %r11

0:

	// compute D1
	movq	%r11, %rbx // D1
	addq	%r12, %rbx // D1 <- D0 + 4*sdd*sizeof(float)
	movq	%rbx, %rbp // D2
	addq	%r12, %rbp // D2 <- D1 + 4*sdd*sizeof(float)

	// compute number of cols
	cmpl	$4, %eax
	jle		0f
	movl	$4, %eax
0:
	subl	%r15d, %eax
	movl	%eax, %r15d

	cmpl	$0, %r10d
	jg		0f

	// offset==0
	cmpl		$2, %r15d
	vmaskmovps	%ymm0, %ymm14,  0(%r11)
	vmovaps		%ymm4, 0(%rbx)
	vmaskmovps	%ymm8, %ymm15,  0(%rbp)
	jl			7f // end
	cmpl		$3, %r15d
	vmaskmovps	%ymm1, %ymm14, 32(%r11)
	vmovaps		%ymm5, 32(%rbx)
	vmaskmovps	%ymm9, %ymm15, 32(%rbp)
	jl			7f // end
	vmaskmovps	%ymm2, %ymm14, 64(%r11)
	vmovaps		%ymm6, 64(%rbx)
	vmaskmovps	%ymm10, %ymm15, 64(%rbp)
	je			7f // end
	vmaskmovps	%ymm3, %ymm14, 96(%r11)
	vmovaps		%ymm7, 96(%rbx)
	vmaskmovps	%ymm11, %ymm15, 96(%rbp)
	//
	jmp		7f

0:
	// offset > 0
	// 1 2 3 4 5 6 7
	
//	movq	%r11, %rbp // D1
//	addq	%r12, %rbp // D2 <- D1 + 4*sdd*sizeof(float)
	addq	%rbp, %r12 // D3 <- D2 + 4*sdd*sizeof(float)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_24x4_gen_lib8, .-inner_store_24x4_gen_lib8
#endif
#endif





//                                rdi    rsi           rdx       rcx      r8        r9           rsp+8     rsp+16   rsp+24    rsp+32
// void kernel_sgemm_nt_24x4_lib8(int k, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nt_24x4_lib8
	.type kernel_sgemm_nt_24x4_lib8, @function
kernel_sgemm_nt_24x4_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nt_24x4_lib8
_kernel_sgemm_nt_24x4_lib8:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorps		%ymm0, %ymm0, %ymm0
	vmovaps		%ymm0, %ymm1
	vmovaps		%ymm0, %ymm2
	vmovaps		%ymm0, %ymm3
	vmovaps		%ymm0, %ymm4
	vmovaps		%ymm0, %ymm5
	vmovaps		%ymm0, %ymm6
	vmovaps		%ymm0, %ymm7
	vmovaps		%ymm0, %ymm8
	vmovaps		%ymm0, %ymm9
	vmovaps		%ymm0, %ymm10
	vmovaps		%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 8*sda*sizeof(float)
	movq	%r8, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_ADD_NT_24X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_add_nt_24x4_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_add_nt_24x4_lib8
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12   // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 8*sdc*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_24X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_24x4_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_24x4_lib8
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 8*sdd*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_STORE_24X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_24x4_lib8
#elif defined(OS_MAC)
	callq _inner_store_24x4_lib8
#endif
#endif


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_sgemm_nt_24x4_lib8, .-kernel_sgemm_nt_24x4_lib8
#endif





//                                    rdi    rsi           rdx       rcx      r8        r9           rsp+8        rsp+16    rsp+24   rsp+32       rsp+40    rsp+48   rsp+56  rsp+64  rsp+72  rsp+80
// void kernel_sgemm_nt_24x4_gen_lib8(int k, float *alpha, float *A, int sda, float *B, float *beta, int offsetC, float *C, int sdc, int offsetD, float *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nt_24x4_gen_lib8
	.type kernel_sgemm_nt_24x4_gen_lib8, @function
kernel_sgemm_nt_24x4_gen_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nt_24x4_gen_lib8
_kernel_sgemm_nt_24x4_gen_lib8:
#endif
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorps		%ymm0, %ymm0, %ymm0
	vmovaps		%ymm0, %ymm1
	vmovaps		%ymm0, %ymm2
	vmovaps		%ymm0, %ymm3
	vmovaps		%ymm0, %ymm4
	vmovaps		%ymm0, %ymm5
	vmovaps		%ymm0, %ymm6
	vmovaps		%ymm0, %ymm7
	vmovaps		%ymm0, %ymm8
	vmovaps		%ymm0, %ymm9
	vmovaps		%ymm0, %ymm10
	vmovaps		%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 8*sda*sizeof(float)
	movq	%r8, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_ADD_NT_24X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_add_nt_24x4_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_add_nt_24x4_lib8
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movl	STACKSIZE +   8(%rsp), %r12d // offsetC
	movq	STACKSIZE +  16(%rsp), %r13 // C
	movl	STACKSIZE +  24(%rsp), %r14d // sdc
	sall	$5, %r14d // 8*sdc*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_24X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_24x4_gen_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_24x4_gen_lib8
#endif
#endif


	// store n gen

	movq	STACKSIZE + 32(%rsp), %r10 // offsetD
	movq	STACKSIZE + 40(%rsp), %r11 // D
	movl	STACKSIZE + 48(%rsp), %r12d // sdd
	sall	$5, %r12d // 8*sdb*sizeof(float)
	movl	STACKSIZE + 56(%rsp), %r13d // m0
	movl	STACKSIZE + 64(%rsp), %r14d // m1
	movl	STACKSIZE + 72(%rsp), %r15d // n0
	movl	STACKSIZE + 80(%rsp), %eax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_24X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_24x4_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_24x4_gen_lib8
#endif
#endif


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

#if defined(OS_LINUX)
	.size	kernel_sgemm_nt_24x4_gen_lib8, .-kernel_sgemm_nt_24x4_gen_lib8
#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#endif

#if defined(OS_LINUX)
	.align 32
.LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
	.align 5
LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#endif
	.long	1056964608
	.long	1069547520
	.long	1075838976
	.long	1080033280
	.long	1083179008
	.long	1085276160
	.long	1087373312
	.long	1089470464

#if defined(OS_LINUX)
	.align 32
.LC01: // { 15.5 14.5 13.5 12.5 11.5 10.5 9.5 8.5 }
#elif defined(OS_MAC)
	.align 5
LC01: // { 15.5 14.5 13.5 12.5 11.5 10.5 9.5 8.5 }
#endif
	.long	1091043328
	.long	1092091904
	.long	1093140480
	.long	1094189056
	.long	1095237632
	.long	1096286208
	.long	1097334784
	.long	1098383360

#if defined(OS_LINUX)
	.align 32
.LC02: // { 23.5 22.5 21.5 20.5 19.5 18.5 17.5 16.5 }
#elif defined(OS_MAC)
	.align 5
LC02: // { 23.5 22.5 21.5 20.5 19.5 18.5 17.5 16.5 }
#endif
	.long	1099169792
	.long	1099694080
	.long	1100218368
	.long	1100742656
	.long	1101266944
	.long	1101791232
	.long	1102315520
	.long	1102839808

#if defined(OS_LINUX)
	.align 32
.LC03: // { -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC03: // { -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 }
#endif
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	3212836864
	.long	3212836864



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

