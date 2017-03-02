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





// subroutine
//
// input arguments:
// r10d   <- k
// r11    <- A
// r12    <- 8*sda*sizeof(float)
// r13    <- B

#if MACRO_LEVEL>=1
	.macro INNER_KERNEL_SGETR_8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgetr_8_lib8, @function
inner_kernel_sgetr_8_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgetr_8_lib8:
#endif
#endif
	
	cmpl	$7, %r10d
	jle		0f // consider clean-up

	// main loop
	.p2align 3
1: // main loop

	vmovaps		0(%r11), %ymm0
	vmovaps		32(%r11), %ymm1
	vunpcklps	%ymm1, %ymm0, %ymm8
	vunpckhps	%ymm1, %ymm0, %ymm9
	vmovaps		64(%r11), %ymm0
	vmovaps		96(%r11), %ymm1
	vunpcklps	%ymm1, %ymm0, %ymm10
	vunpckhps	%ymm1, %ymm0, %ymm11
	vmovaps		128(%r11), %ymm0
	vmovaps		160(%r11), %ymm1
	vunpcklps	%ymm1, %ymm0, %ymm12
	vunpckhps	%ymm1, %ymm0, %ymm13
	vmovaps		192(%r11), %ymm0
	vmovaps		224(%r11), %ymm1
	vunpcklps	%ymm1, %ymm0, %ymm14
	vunpckhps	%ymm1, %ymm0, %ymm15

	vshufps		$0x44, %ymm10, %ymm8, %ymm0
	vshufps		$0x44, %ymm14, %ymm12, %ymm1
	vperm2f128	$0x20, %ymm1, %ymm0, %ymm2
	vperm2f128	$0x31, %ymm1, %ymm0, %ymm3
	vmovaps		%ymm2, 0(%r13)
	vmovaps		%ymm3, 128(%r13)
	vshufps		$0xee, %ymm10, %ymm8, %ymm0
	vshufps		$0xee, %ymm14, %ymm12, %ymm1
	vperm2f128	$0x20, %ymm1, %ymm0, %ymm2
	vperm2f128	$0x31, %ymm1, %ymm0, %ymm3
	vmovaps		%ymm2, 32(%r13)
	vmovaps		%ymm3, 160(%r13)
	vshufps		$0x44, %ymm11, %ymm9, %ymm0
	vshufps		$0x44, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x20, %ymm1, %ymm0, %ymm2
	vperm2f128	$0x31, %ymm1, %ymm0, %ymm3
	vmovaps		%ymm2, 64(%r13)
	vmovaps		%ymm3, 192(%r13)
	vshufps		$0xee, %ymm11, %ymm9, %ymm0
	vshufps		$0xee, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x20, %ymm1, %ymm0, %ymm2
	vperm2f128	$0x31, %ymm1, %ymm0, %ymm3
	vmovaps		%ymm2, 96(%r13)
	vmovaps		%ymm3, 224(%r13)

	subl		$8, %r10d
	addq		%r12, %r11
	addq		$256, %r13

	cmpl		$7, %r10d
	jg			1b // main loop 

0: // consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

3: // clean-up loop

	subl		$1, %r10d
	addq		$4, %r11
	addq		$32, %r12

	cmpl		$0, %r10d
	jg			3b // clean-up loop 

2: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgetr_8_lib8, .-inner_kernel_sgetr_8_lib8
#endif
#endif





//                          rdi    rsi          rdx       rcx      r8
// void kernel_sgetr_8_lib8(int k, int offsetA, float *A, int sda, float *B);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgetr_8_lib8
	.type kernel_sgetr_8_lib8, @function
kernel_sgetr_8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgetr_8_lib8
_kernel_sgetr_8_lib8:
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

	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 8*sda*sizeof(float)
	movq	%r8, %r13  // B
	movl	%esi, %r14d  // offsetA

	// TODO edge

#if MACRO_LEVEL>=1
	INNER_KERNEL_SGETR_8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgetr_8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgetr_8_lib8
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
	.size	kernel_sgetr_8_lib8, .-kernel_sgetr_8_lib8
#endif






