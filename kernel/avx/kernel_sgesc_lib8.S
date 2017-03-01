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
// r11    <- alpha
// r12    <- A

#if MACRO_LEVEL>=1
	.macro INNER_KERNEL_SGESC_8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgesc_8_lib8, @function
inner_kernel_sgesc_8_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgesc_8_lib8:
#endif
#endif
	
	vbroadcastss	0(%r11), %ymm15

	cmpl	$3, %r10d
	jle		0f // consider clean-up

	// main loop
	.p2align 3
1: // main loop

	vmovaps		0(%r12), %ymm0
	vmulps		%ymm15, %ymm0, %ymm0
	vmovaps		%ymm0, 0(%r12)
	subl		$4, %r10d

	vmovaps		32(%r12), %ymm0
	vmulps		%ymm15, %ymm0, %ymm0
	vmovaps		%ymm0, 32(%r12)

	vmovaps		64(%r12), %ymm0
	vmulps		%ymm15, %ymm0, %ymm0
	vmovaps		%ymm0, 64(%r12)
	addq		$128, %r12

	vmovaps		-32(%r12), %ymm0
	vmulps		%ymm15, %ymm0, %ymm0
	vmovaps		%ymm0, -32(%r12)

	cmpl		$4, %r10d
	jg			1b // main loop 

0: // consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

3: // clean-up loop

	vmovaps		0(%r12), %ymm0
	vmulps		%ymm15, %ymm0, %ymm0
	vmovaps		%ymm0, 0(%r12)
	subl		$1, %r10d
	addq		$32, %r12

	cmpl		$0, %r10d
	jg			3b // clean-up loop 

2: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgesc_8_lib8, .-inner_kernel_sgesc_8_lib8
#endif
#endif





// subroutine
//
// input arguments:
// r10d   <- k
// r11    <- alpha
// r12    <- A
// r13d   <- m0
// r14d   <- m1

#if MACRO_LEVEL>=1
	.macro INNER_KERNEL_SGESC_8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgesc_8_gen_lib8, @function
inner_kernel_sgesc_8_gen_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgesc_8_gen_lib8:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm14, %xmm14
	vcvtsi2ss	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovups		.LC00(%rip), %ymm12
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %ymm12
#endif
	vshufps		$0x00, %xmm14, %xmm14, %xmm14
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm14, %ymm14, %ymm14
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm12, %ymm14, %ymm14
	vsubps		%ymm15, %ymm12, %ymm15
	vandps		%ymm14, %ymm15, %ymm15

	vbroadcastss	0(%r11), %ymm14

	cmpl	$3, %r10d
	jle		0f // consider clean-up

	// main loop
	.p2align 3
1: // main loop

	vmovaps		0(%r12), %ymm0
	vmulps		%ymm14, %ymm0, %ymm0
	vmaskmovps	%ymm0, %ymm15,  0(%r12)
	subl		$4, %r10d

	vmovaps		32(%r12), %ymm0
	vmulps		%ymm14, %ymm0, %ymm0
	vmaskmovps	%ymm0, %ymm15,  32(%r12)

	vmovaps		64(%r12), %ymm0
	vmulps		%ymm14, %ymm0, %ymm0
	vmaskmovps	%ymm0, %ymm15,  64(%r12)
	addq		$128, %r12

	vmovaps		-32(%r12), %ymm0
	vmulps		%ymm14, %ymm0, %ymm0
	vmaskmovps	%ymm0, %ymm15,  -32(%r12)

	cmpl		$4, %r10d
	jg			1b // main loop 

0: // consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

3: // clean-up loop

	vmovaps		0(%r12), %ymm0
	vmulps		%ymm14, %ymm0, %ymm0
	vmaskmovps	%ymm0, %ymm15,  0(%r12)
	subl		$1, %r10d
	addq		$32, %r12

	cmpl		$0, %r10d
	jg			3b // clean-up loop 

2: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgesc_8_lib8, .-inner_kernel_sgesc_8_lib8
#endif
#endif





//                          rdi    rsi           rdx
// void kernel_sgesc_8_lib8(int k, float *alpha, float *A);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgesc_8_lib8
	.type kernel_sgesc_8_lib8, @function
kernel_sgesc_8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgesc_8_lib8
_kernel_sgesc_8_lib8:
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
	movq	%rsi, %r11  // alpha
	movq	%rdx, %r12  // A

#if MACRO_LEVEL>=1
	INNER_KERNEL_SGESC_8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgesc_8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgesc_8_lib8
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
	.size	kernel_sgesc_8_lib8, .-kernel_sgesc_8_lib8
#endif





//                              rdi    rsi           rdx       rcx     r8
// void kernel_sgecp_8_gen_lib8(int k, float *alpha, float *A, int m0, int m1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgesc_8_gen_lib8
	.type kernel_sgesc_8_gen_lib8, @function
kernel_sgesc_8_gen_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgesc_8_gen_lib8
_kernel_sgesc_8_gen_lib8:
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
	movq	%rsi, %r11  // alpha
	movq	%rdx, %r12  // A
	movl	%ecx, %r14d // m0
	movl	%r8d, %r15d // m1

#if MACRO_LEVEL>=1
	INNER_KERNEL_SGESC_8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgesc_8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgesc_8_gen_lib8
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
	.size	kernel_sgesc_8_gen_lib8, .-kernel_sgesc_8_gen_lib8
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

