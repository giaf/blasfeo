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

#if defined(OS_LINUX) | defined(OS_MAC)

//#define STACKSIZE 96
#define STACKSIZE 64
#define ARG1  %rdi
#define ARG2  %rsi
#define ARG3  %rdx
#define ARG4  %rcx
#define ARG5  %r8
#define ARG6  %r9
#define ARG7  STACKSIZE +  8(%rsp)
#define ARG8  STACKSIZE + 16(%rsp)
#define ARG9  STACKSIZE + 24(%rsp)
#define ARG10 STACKSIZE + 32(%rsp)
#define ARG11 STACKSIZE + 40(%rsp)
#define ARG12 STACKSIZE + 48(%rsp)
#define ARG13 STACKSIZE + 56(%rsp)
#define ARG14 STACKSIZE + 64(%rsp)
#define ARG15 STACKSIZE + 72(%rsp)
#define ARG16 STACKSIZE + 80(%rsp)
#define ARG17 STACKSIZE + 88(%rsp)
#define ARG18 STACKSIZE + 96(%rsp)
#define PROLOGUE \
	subq	$STACKSIZE, %rsp; \
	movq	%rbx,   (%rsp); \
	movq	%rbp,  8(%rsp); \
	movq	%r12, 16(%rsp); \
	movq	%r13, 24(%rsp); \
	movq	%r14, 32(%rsp); \
	movq	%r15, 40(%rsp); \
	vzeroupper;
#define EPILOGUE \
	vzeroupper; \
	movq	  (%rsp), %rbx; \
	movq	 8(%rsp), %rbp; \
	movq	16(%rsp), %r12; \
	movq	24(%rsp), %r13; \
	movq	32(%rsp), %r14; \
	movq	40(%rsp), %r15; \
	addq	$STACKSIZE, %rsp;

#elif defined(OS_WINDOWS)

#define STACKSIZE 256
#define ARG1  %rcx
#define ARG2  %rdx
#define ARG3  %r8
#define ARG4  %r9
#define ARG5  STACKSIZE + 40(%rsp)
#define ARG6  STACKSIZE + 48(%rsp)
#define ARG7  STACKSIZE + 56(%rsp)
#define ARG8  STACKSIZE + 64(%rsp)
#define ARG9  STACKSIZE + 72(%rsp)
#define ARG10 STACKSIZE + 80(%rsp)
#define ARG11 STACKSIZE + 88(%rsp)
#define ARG12 STACKSIZE + 96(%rsp)
#define ARG13 STACKSIZE + 104(%rsp)
#define ARG14 STACKSIZE + 112(%rsp)
#define ARG15 STACKSIZE + 120(%rsp)
#define ARG16 STACKSIZE + 128(%rsp)
#define ARG17 STACKSIZE + 136(%rsp)
#define ARG18 STACKSIZE + 144(%rsp)
#define PROLOGUE \
	subq	$STACKSIZE, %rsp; \
	movq	%rbx,   (%rsp); \
	movq	%rbp,  8(%rsp); \
	movq	%r12, 16(%rsp); \
	movq	%r13, 24(%rsp); \
	movq	%r14, 32(%rsp); \
	movq	%r15, 40(%rsp); \
	movq	%rdi, 48(%rsp); \
	movq	%rsi, 56(%rsp); \
	vmovups	%xmm6, 64(%rsp); \
	vmovups	%xmm7, 80(%rsp); \
	vmovups	%xmm8, 96(%rsp); \
	vmovups	%xmm9, 112(%rsp); \
	vmovups	%xmm10, 128(%rsp); \
	vmovups	%xmm11, 144(%rsp); \
	vmovups	%xmm12, 160(%rsp); \
	vmovups	%xmm13, 176(%rsp); \
	vmovups	%xmm14, 192(%rsp); \
	vmovups	%xmm15, 208(%rsp); \
	vzeroupper;
#define EPILOGUE \
	vzeroupper; \
	movq	  (%rsp), %rbx; \
	movq	 8(%rsp), %rbp; \
	movq	16(%rsp), %r12; \
	movq	24(%rsp), %r13; \
	movq	32(%rsp), %r14; \
	movq	40(%rsp), %r15; \
	movq	48(%rsp), %rdi; \
	movq	56(%rsp), %rsi; \
	vmovups	64(%rsp), %xmm6; \
	vmovups	80(%rsp), %xmm7; \
	vmovups	96(%rsp), %xmm8; \
	vmovups	112(%rsp), %xmm9; \
	vmovups	128(%rsp), %xmm10; \
	vmovups	144(%rsp), %xmm11; \
	vmovups	160(%rsp), %xmm12; \
	vmovups	176(%rsp), %xmm13; \
	vmovups	192(%rsp), %xmm14; \
	vmovups	208(%rsp), %xmm15; \
	addq	$STACKSIZE, %rsp;

#else

#error wrong OS

#endif



#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- x
// xmm0 <- [z0 z1]_a
// xmm1 <- [z2 z3]_a
// xmm2 <- [z0 z1]_b
// xmm3 <- [z2 z3]_b

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- x+k*sizeof(double)
// xmm0 <- [z0 z1]_a
// xmm1 <- [z2 z3]_a
// xmm2 <- [z0 z1]_b
// xmm3 <- [z2 z3]_b

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemv_add_n_4_lib4, @function
inner_kernel_dgemv_add_n_4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemv_add_n_4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemv_add_n_4_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemv_add_n_4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	cmpl	$4, %r10d
	jl		0f // clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	movddup	0(%r12), %xmm12
	movapd	0(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm0
	movapd	16(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm1
	subl	$4, %r10d

	movddup	8(%r12), %xmm12
	movapd	32(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm2
	movapd	48(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm3
	
	movddup	16(%r12), %xmm12
	movapd	64(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm0
	movapd	80(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm1

	movddup	24(%r12), %xmm12
	movapd	96(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm2
	movapd	112(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm3
	
	addq	$128, %r11
	addq	$32, %r12
	
	cmpl	$3, %r10d

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

0: // clean-up
	
	movddup	0(%r12), %xmm12
	movapd	0(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm0
	movapd	16(%r11), %xmm8
	mulpd	%xmm12, %xmm8
	addpd	%xmm8, %xmm1

	addq	$32, %r11
	addq	$8, %r12
	
	subl	$1, %r10d
	cmpl	$0, %r10d

	jg		0b // clean

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemv_add_n_4_lib4, .-inner_kernel_dgemv_add_n_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for ta==n, scale for generic alpha and beta
//
// input arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// xmm0 <- [z0 z1]_a
// xmm1 <- [z2 z3]_a
// xmm2 <- [z0 z1]_b
// xmm3 <- [z2 z3]_b
//
// output arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// xmm0 <- [z0 z1]
// xmm1 <- [z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_n_scale_ab_4_lib4, @function
inner_blend_n_scale_ab_4_lib4:
#elif defined(OS_MAC)
_inner_blend_n_scale_ab_4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_n_scale_ab_4_lib4; .scl 2; .type 32; .endef
inner_blend_n_scale_ab_4_lib4:
#endif
#endif

	// reduction
	addpd	%xmm2, %xmm0
	addpd	%xmm3, %xmm1

	// alpha
	movddup	0(%r10), %xmm15
	mulpd	%xmm15, %xmm0
	mulpd	%xmm15, %xmm1

	// beta
	movddup	0(%r11), %xmm15
	movupd	0(%r12), %xmm14
	mulpd	%xmm15, %xmm14
	addpd	%xmm14, %xmm0
	movupd	16(%r12), %xmm14
	mulpd	%xmm15, %xmm14
	addpd	%xmm14, %xmm1

#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blend_n_scale_ab_4_lib4, .-inner_blend_n_scale_ab_4_lib4
#endif
#endif





// common inner routine with file scope
//
// store 
//
// input arguments:
// r10  <- z
// xmm0 <- [z0 z1]
// xmm1 <- [z2 z3]
//
// output arguments:
// r10  <- z
// xmm0 <- [z0 z1]
// xmm1 <- [z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4_lib4, @function
inner_store_4_lib4:
#elif defined(OS_MAC)
_inner_store_4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_4_lib4; .scl 2; .type 32; .endef
inner_store_4_lib4:
#endif
#endif
	
	movupd %xmm0,  0(%r10)
	movupd %xmm1, 16(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4_lib4, .-inner_store_4_lib4
#endif
#endif





//                            1      2              3          4          5             6          7
// void kernel_dgemv_n_4_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemv_n_4_lib4
	.type kernel_dgemv_n_4_lib4, @function
kernel_dgemv_n_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemv_n_4_lib4
_kernel_dgemv_n_4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemv_n_4_lib4
	.def kernel_dgemv_n_4_lib4; .scl 2; .type 32; .endef
kernel_dgemv_n_4_lib4:
#endif
	
	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner dgemv kernel n

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemv_add_n_4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif
#endif


	// call inner blend n scale ab

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11   // beta
	movq	ARG6, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_n_scale_ab_4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_n_scale_ab_4_lib4
#endif
#endif


	// store

	movq	ARG7, %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemv_n_4_lib4, .-kernel_dgemv_n_4_lib4
#endif






