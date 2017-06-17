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
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
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
// r12   <- x+k*sizeof(double)
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_ADD_N_8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemv_add_n_8_lib8, @function
inner_kernel_gemv_add_n_8_lib8:
#elif defined(OS_MAC)
_inner_kernel_gemv_add_n_8_lib8:
#elif defined(OS_WINDOWS)
	.def inner_kernel_gemv_add_n_8_lib8; .scl 2; .type 32; .endef
inner_kernel_gemv_add_n_8_lib8:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	cmpl	$4, %r10d
	jl		0f // clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	vmovaps			0(%r11), %ymm8
	vbroadcastss	0(%r12), %ymm12
	vmulps			%ymm8, %ymm12, %ymm15
	vaddps			%ymm0, %ymm15, %ymm0
	
	subl	$4, %r10d

	vmovaps			32(%r11), %ymm8
	vbroadcastss	8(%r12), %ymm12
	vmulps			%ymm8, %ymm12, %ymm15
	vaddps			%ymm1, %ymm15, %ymm1
	
	vmovaps			64(%r11), %ymm8
	vbroadcastss	16(%r12), %ymm12
	vmulps			%ymm8, %ymm12, %ymm15
	vaddps			%ymm2, %ymm15, %ymm2

	vmovaps			96(%r11), %ymm8
	vbroadcastss	24(%r12), %ymm12
	vmulps			%ymm8, %ymm12, %ymm15
	vaddps			%ymm3, %ymm15, %ymm3
	
	addq	$128, %r11
	addq	$32, %r12
	
	cmpl	$3, %r10d

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

0: // clean-up
	
	vmovaps			0(%r11), %ymm8
	vbroadcastss	0(%r12), %ymm12
	vmulps			%ymm8, %ymm12, %ymm15
	vaddps			%ymm0, %ymm15, %ymm0
	
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
	.size	inner_kernel_gemv_add_n_8_lib8, .-inner_kernel_gemv_add_n_8_lib8
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
// ymm0 <- [z0 z1 z2 z3]_a
// ymm1 <- [z0 z1 z2 z3]_b
// ymm2 <- [z0 z1 z2 z3]_c
// ymm3 <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// ymm0 <- [z0 z1 z2 z3]
// ymm1 <- dirty
// ymm2 <- dirty
// ymm3 <- dirty
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_n_scale_ab_8_lib8, @function
inner_blend_n_scale_ab_8_lib8:
#elif defined(OS_MAC)
_inner_blend_n_scale_ab_8_lib8:
#elif defined(OS_WINDOWS)
	.def inner_blend_n_scale_ab_8_lib8; .scl 2; .type 32; .endef
inner_blend_n_scale_ab_8_lib8:
#endif
#endif

	// reduction
	vaddps			%ymm0, %ymm1, %ymm0
	vaddps			%ymm2, %ymm3, %ymm2
	vaddps			%ymm0, %ymm2, %ymm0

	// alpha
	vbroadcastss	0(%r10), %ymm15
	vmulps			%ymm0, %ymm15, %ymm0

	// beta
	vbroadcastss	0(%r11), %ymm15
	vmovups			0(%r12), %ymm14
	vmulps			%ymm15, %ymm14, %ymm14
	vaddps			%ymm0, %ymm14, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blend_n_scale_ab_8_lib8, .-inner_blend_n_scale_ab_8_lib8
#endif
#endif





// common inner routine with file scope
//
// store 
//
// input arguments:
// r10  <- z
// ymm0 <- [z0 z1 z2 z3]
//
// output arguments:
// r10  <- z
// ymm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8_lib8, @function
inner_store_8_lib8:
#elif defined(OS_MAC)
_inner_store_8_lib8:
#elif defined(OS_WINDOWS)
	.def inner_store_8_lib8; .scl 2; .type 32; .endef
inner_store_8_lib8:
#endif
#endif
	
	vmovups %ymm0,  0(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8_lib8, .-inner_store_8_lib8
#endif
#endif





// common inner routine with file scope
//
// store vs
//
// input arguments:
// r10   <- D
// r11d   <- km
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8_VS_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8_vs_lib8, @function
inner_store_8_vs_lib8:
#elif defined(OS_MAC)
_inner_store_8_vs_lib8:
#elif defined(OS_WINDOWS)
	.def inner_store_8_vs_lib8; .scl 2; .type 32; .endef
inner_store_8_vs_lib8:
#endif
#endif
	
	vcvtsi2ss	%r11d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC00(%rip), %ymm14
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %ymm14
#endif
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm15, %ymm14, %ymm15

	vmaskmovps	%ymm0, %ymm15,  0(%r10)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8_vs_lib8, .-inner_store_8_vs_lib8
#endif
#endif





// common inner routine with file scope
//
// store gen
//
// input arguments:
// r10   <- D
// r11d  <- k0 : start form (inc)
// r12d  <- k1 : up to (exc)
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d  <- k0 : start form (inc)
// r12d  <- k1 : up to (exc)
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8_gen_lib8, @function
inner_store_8_gen_lib8:
#elif defined(OS_MAC)
_inner_store_8_gen_lib8:
#elif defined(OS_WINDOWS)
	.def inner_store_8_gen_lib8; .scl 2; .type 32; .endef
inner_store_8_gen_lib8:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r11d, %xmm14, %xmm14
	vcvtsi2ss	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
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

	vmaskmovps	%ymm0, %ymm15,  0(%r10)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8_gen_lib8, .-inner_store_8_gen_lib8
#endif
#endif





//                            1      2              3          4          5             6          7
// void kernel_sgemv_n_8_lib8(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemv_n_8_lib8
	.type kernel_sgemv_n_8_lib8, @function
kernel_sgemv_n_8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemv_n_8_lib8
_kernel_sgemv_n_8_lib8:
#elif defined(OS_WINDOWS)
	.globl kernel_sgemv_n_8_lib8
	.def kernel_sgemv_n_8_lib8; .scl 2; .type 32; .endef
kernel_sgemv_n_8_lib8:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner sgemv kernel n

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_N_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemv_add_n_8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_gemv_add_n_8_lib8
#endif
#endif


	// call inner blend n scale ab

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11   // beta
	movq	ARG6, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_n_scale_ab_8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_n_scale_ab_8_lib8
#endif
#endif


	// store

	movq	ARG7, %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8_lib8
#elif defined(OS_MAC)
	callq _inner_store_8_lib8
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_sgemv_n_8_lib8, .-kernel_sgemv_n_8_lib8
#endif





//                               1      2              3          4          5             6          7          8
// void kernel_sgemv_n_8_vs_lib8(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z, int k1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemv_n_8_vs_lib8
	.type kernel_sgemv_n_8_vs_lib8, @function
kernel_sgemv_n_8_vs_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemv_n_8_vs_lib8
_kernel_sgemv_n_8_vs_lib8:
#elif defined(OS_WINDOWS)
	.globl kernel_sgemv_n_8_vs_lib8
	.def kernel_sgemv_n_8_vs_lib8; .scl 2; .type 32; .endef
kernel_sgemv_n_8_vs_lib8:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner sgemv kernel n

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_N_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemv_add_n_8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_gemv_add_n_8_lib8
#endif
#endif


	// call inner blend n scale ab

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11   // beta
	movq	ARG6, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_n_scale_ab_8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_n_scale_ab_8_lib8
#endif
#endif


	// store

	movq	ARG7, %r10 // z 
	movq	ARG8, %r11 // k1 

#if MACRO_LEVEL>=1
	INNER_STORE_8_VS_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8_vs_lib8
#elif defined(OS_MAC)
	callq _inner_store_8_vs_lib8
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_sgemv_n_8_vs_lib8, .-kernel_sgemv_n_8_vs_lib8
#endif





//                                1      2              3          4          5             6          7          8       9
// void kernel_sgemv_n_8_gen_lib8(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z, int k0, int kq);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemv_n_8_gen_lib8
	.type kernel_sgemv_n_8_gen_lib8, @function
kernel_sgemv_n_8_gen_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemv_n_8_gen_lib8
_kernel_sgemv_n_8_gen_lib8:
#elif defined(OS_WINDOWS)
	.globl kernel_sgemv_n_8_gen_lib8
	.def kernel_sgemv_n_8_gen_lib8; .scl 2; .type 32; .endef
kernel_sgemv_n_8_gen_lib8:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner sgemv kernel n

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_N_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemv_add_n_8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_gemv_add_n_8_lib8
#endif
#endif


	// call inner blend n scale ab

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11   // beta
	movq	ARG6, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_8_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_n_scale_ab_8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_n_scale_ab_8_lib8
#endif
#endif


	// store

	movq	ARG7, %r10 // z 
	movq	ARG8, %r11 // k1 
	movq	ARG9, %r12 // k2 

#if MACRO_LEVEL>=1
	INNER_STORE_8_GEN_LIB8
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_8_gen_lib8
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_sgemv_n_8_gen_lib8, .-kernel_sgemv_n_8_gen_lib8
#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#elif defined(OS_WINDOWS)
	.section .rdata,"dr"
#endif

#if defined(OS_LINUX)
	.align 32
.LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
	.align 5
LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#endif
	.float	0.5
	.float	1.5
	.float	2.5
	.float	3.5
	.float	4.5
	.float	5.5
	.float	6.5
	.float	7.5




#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

