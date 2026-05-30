/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2021 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
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
#define ARG19 STACKSIZE + 104(%rsp)
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

#if defined(OS_LINUX)

#define GLOB_FUN_START(NAME) \
	.globl NAME; \
	.type NAME, @function; \
NAME:
#define FUN_START(NAME) \
	.type NAME, @function; \
NAME:
#define FUN_END(NAME) \
	.size	NAME, .-NAME
#define CALL(NAME) \
	call NAME
#define ZERO_ACC \
	vxorps	%zmm0, %zmm0, %zmm0; \
	vmovaps	%zmm0, %zmm1; \
	vmovaps	%zmm0, %zmm2; \
	vmovaps	%zmm0, %zmm3; \
	vmovaps	%zmm0, %zmm4; \
	vmovaps	%zmm0, %zmm5; \
	vmovaps	%zmm0, %zmm6; \
	vmovaps	%zmm0, %zmm7; \
	vmovaps	%zmm0, %zmm8; \
	vmovaps	%zmm0, %zmm9; \
	vmovaps	%zmm0, %zmm10; \
	vmovaps	%zmm0, %zmm11; \
	vmovaps	%zmm0, %zmm12; \
	vmovaps	%zmm0, %zmm13; \
	vmovaps	%zmm0, %zmm14; \
	vmovaps	%zmm0, %zmm15;
#define NEG_ACC \
	vmovaps		.LC03(%rip), %zmm24; \
	vxorps		%zmm24, %zmm0, %zmm0; \
	vxorps		%zmm24, %zmm1, %zmm1; \
	vxorps		%zmm24, %zmm2, %zmm2; \
	vxorps		%zmm24, %zmm3, %zmm3; \
	vxorps		%zmm24, %zmm4, %zmm4; \
	vxorps		%zmm24, %zmm5, %zmm5; \
	vxorps		%zmm24, %zmm6, %zmm6; \
	vxorps		%zmm24, %zmm7, %zmm7; \
	vxorps		%zmm24, %zmm8, %zmm8; \
	vxorps		%zmm24, %zmm9, %zmm9; \
	vxorps		%zmm24, %zmm10, %zmm10; \
	vxorps		%zmm24, %zmm11, %zmm11; \
	vxorps		%zmm24, %zmm12, %zmm12; \
	vxorps		%zmm24, %zmm13, %zmm13; \
	vxorps		%zmm24, %zmm14, %zmm14; \
	vxorps		%zmm24, %zmm15, %zmm15;

#else // defined(OS_MAC)

#define GLOB_FUN_START(NAME) \
	.globl _ ## NAME; \
_ ## NAME:
#define FUN_START(NAME) \
_ ## NAME:
#define FUN_END(NAME)
#define CALL(NAME) \
	callq _ ## NAME
#define ZERO_ACC \
	vxorps	%zmm0, %zmm0, %zmm0; \
	vmovaps	%zmm0, %zmm1; \
	vmovaps	%zmm0, %zmm2; \
	vmovaps	%zmm0, %zmm3; \
	vmovaps	%zmm0, %zmm4; \
	vmovaps	%zmm0, %zmm5; \
	vmovaps	%zmm0, %zmm6; \
	vmovaps	%zmm0, %zmm7; \
	vmovaps	%zmm0, %zmm8; \
	vmovaps	%zmm0, %zmm9; \
	vmovaps	%zmm0, %zmm10; \
	vmovaps	%zmm0, %zmm11; \
	vmovaps	%zmm0, %zmm12; \
	vmovaps	%zmm0, %zmm13; \
	vmovaps	%zmm0, %zmm14; \
	vmovaps	%zmm0, %zmm15;
#define NEG_ACC \
	vmovaps		LC03(%rip), %zmm24; \
	vxorps		%zmm24, %zmm0, %zmm0; \
	vxorps		%zmm24, %zmm1, %zmm1; \
	vxorps		%zmm24, %zmm2, %zmm2; \
	vxorps		%zmm24, %zmm3, %zmm3; \
	vxorps		%zmm24, %zmm4, %zmm4; \
	vxorps		%zmm24, %zmm5, %zmm5; \
	vxorps		%zmm24, %zmm6, %zmm6; \
	vxorps		%zmm24, %zmm7, %zmm7; \
	vxorps		%zmm24, %zmm8, %zmm8; \
	vxorps		%zmm24, %zmm9, %zmm9; \
	vxorps		%zmm24, %zmm10, %zmm10; \
	vxorps		%zmm24, %zmm11, %zmm11; \
	vxorps		%zmm24, %zmm12, %zmm12; \
	vxorps		%zmm24, %zmm13, %zmm13; \
	vxorps		%zmm24, %zmm14, %zmm14; \
	vxorps		%zmm24, %zmm15, %zmm15;

#endif

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
#define ARG19 STACKSIZE + 152(%rsp)
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

#define GLOB_FUN_START(NAME) \
	.globl NAME; \
	.def NAME; .scl 2; .type 32; .endef; \
NAME:
#define FUN_START(NAME) \
	.def NAME; .scl 2; .type 32; .endef; \
NAME:
#define FUN_END(NAME)
#define CALL(NAME) \
	call NAME
#define ZERO_ACC \
	vxorps	%zmm0, %zmm0, %zmm0; \
	vmovaps	%zmm0, %zmm1; \
	vmovaps	%zmm0, %zmm2; \
	vmovaps	%zmm0, %zmm3; \
	vmovaps	%zmm0, %zmm4; \
	vmovaps	%zmm0, %zmm5; \
	vmovaps	%zmm0, %zmm6; \
	vmovaps	%zmm0, %zmm7; \
	vmovaps	%zmm0, %zmm8; \
	vmovaps	%zmm0, %zmm9; \
	vmovaps	%zmm0, %zmm10; \
	vmovaps	%zmm0, %zmm11; \
	vmovaps	%zmm0, %zmm12; \
	vmovaps	%zmm0, %zmm13; \
	vmovaps	%zmm0, %zmm14; \
	vmovaps	%zmm0, %zmm15;
#define NEG_ACC \
	vmovaps		.LC03(%rip), %zmm24; \
	vxorps		%zmm24, %zmm0, %zmm0; \
	vxorps		%zmm24, %zmm1, %zmm1; \
	vxorps		%zmm24, %zmm2, %zmm2; \
	vxorps		%zmm24, %zmm3, %zmm3; \
	vxorps		%zmm24, %zmm4, %zmm4; \
	vxorps		%zmm24, %zmm5, %zmm5; \
	vxorps		%zmm24, %zmm6, %zmm6; \
	vxorps		%zmm24, %zmm7, %zmm7; \
	vxorps		%zmm24, %zmm8, %zmm8; \
	vxorps		%zmm24, %zmm9, %zmm9; \
	vxorps		%zmm24, %zmm10, %zmm10; \
	vxorps		%zmm24, %zmm11, %zmm11; \
	vxorps		%zmm24, %zmm12, %zmm12; \
	vxorps		%zmm24, %zmm13, %zmm13; \
	vxorps		%zmm24, %zmm14, %zmm14; \
	vxorps		%zmm24, %zmm15, %zmm15;

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
// r12   <- B
// zmm0  <- []
// ...
// zmm15 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+...
// r12   <- B+...
// zmm0  <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_16X16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_16x16_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch
	prefetcht0	0+0*64(%r12) // software prefetch
	prefetcht0	0+1*64(%r12) // software prefetch
	prefetcht0	0+2*64(%r12) // software prefetch
	prefetcht0	0+3*64(%r12) // software prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	prefetcht0	256+0*64(%r12) // software prefetch
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	prefetcht0	256+1*64(%r12) // software prefetch
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	prefetcht0	256+2*64(%r12) // software prefetch
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	prefetcht0	256+3*64(%r12) // software prefetch
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	//prefetcht0	256+0*64(%r12) // software prefetch
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	//prefetcht0	256+1*64(%r12) // software prefetch
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	//prefetcht0	256+2*64(%r12) // software prefetch
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	//prefetcht0	256+3*64(%r12) // software prefetch
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vmovaps			0(%r11), %zmm25 // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_16x16_lib16)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX1_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx1_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	subl	$ 4, %r10d

	// unroll 1
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A

	// unroll 2
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	addq	$ 256, %r11

	// unroll 3
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	subl	$ 4, %r10d

	// unroll 1
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A

	// unroll 2
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	addq	$ 256, %r11

	// unroll 3
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx1_lib16)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm1  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm1  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX2_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx2_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx2_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm2  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm2  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX3_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx3_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx3_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm3  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm3  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX4_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx4_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx4_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm4  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm4  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX5_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx5_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx5_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm5  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm5  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX6_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx6_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx6_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm6  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm6  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX7_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx7_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx7_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm7  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm7  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX8_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx8_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx8_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm8  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm8  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX9_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx9_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx9_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm9  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm9  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX10_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx10_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx10_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm10 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm10 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX11_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx11_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx11_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm11 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm11 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX12_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx12_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx12_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm12 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm12 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX13_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx13_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx13_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm13 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm13 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX14_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx14_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx14_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm14 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm14 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX15_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx15_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx15_lib16)
#endif




// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// zmm0  <- []
// ...
// zmm15 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// zmm0  <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_VX16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_vx16_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 {%k1}{z} // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastss	0+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			128(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 0
	vbroadcastss	0+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			192(%r11), %zmm26 {%k1}{z} // A
	vbroadcastss	4+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60+128(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	addq	$ 256, %r11

	// unroll 0
	vbroadcastss	0+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	4+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	8+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	12+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	16+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	20+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	24+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	28+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	32+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	36+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	40+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	44+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	48+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	52+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	56+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	60+192(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vmovaps			0(%r11), %zmm25 {%k1}{z} // A
	vbroadcastss	0(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	4(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	16(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	20(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	24(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	28(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	32(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	36(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	40(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	44(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	48(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	52(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	56(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	60(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	addq	$ 64, %r11
	addq	$ 64, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_vx16_lib16)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- m1
// r14   <- n1
// zmm0  <- []
// ...
// zmm15  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// r13   <- m1
// r14   <- n1
// zmm0  <- []
// ...
// zmm15  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NT_16X16_VS_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nt_16x16_vs_lib16)
#endif

	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm25, %xmm25
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC00(%rip), %zmm24
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %zmm24
#endif
	vbroadcastss	%xmm25, %zmm25
	vsubps		%zmm25, %zmm24, %zmm25
	vpmovd2m	%zmm25, %k1

	cmpl	$ 1, %r14d
	jg		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX1_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx1_lib16)
#endif
	
	jmp		115f

100:

	cmpl	$ 2, %r14d
	jg		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX2_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx2_lib16)
#endif
	
	jmp		115f

101:

	cmpl	$ 3, %r14d
	jg		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX3_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx3_lib16)
#endif
	
	jmp		115f

102:

	cmpl	$ 4, %r14d
	jg		103f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX4_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx4_lib16)
#endif
	
	jmp		115f

103:

	cmpl	$ 5, %r14d
	jg		104f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX5_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx5_lib16)
#endif
	
	jmp		115f

104:

	cmpl	$ 6, %r14d
	jg		105f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX6_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx6_lib16)
#endif
	
	jmp		115f

105:

	cmpl	$ 7, %r14d
	jg		106f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX7_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx7_lib16)
#endif
	
	jmp		115f

106:

	cmpl	$ 8, %r14d
	jg		107f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX8_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx8_lib16)
#endif
	
	jmp		115f

107:

	cmpl	$ 9, %r14d
	jg		108f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX9_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx9_lib16)
#endif
	
	jmp		115f

108:

	cmpl	$ 10, %r14d
	jg		109f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX10_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx10_lib16)
#endif
	
	jmp		115f

109:

	cmpl	$ 11, %r14d
	jg		110f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX11_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx11_lib16)
#endif
	
	jmp		115f

110:

	cmpl	$ 12, %r14d
	jg		111f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX12_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx12_lib16)
#endif
	
	jmp		115f

111:

	cmpl	$ 13, %r14d
	jg		112f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX13_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx13_lib16)
#endif
	
	jmp		115f

112:

	cmpl	$ 14, %r14d
	jg		113f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX14_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx14_lib16)
#endif
	
	jmp		115f

113:

	cmpl	$ 15, %r14d
	jg		114f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX15_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx15_lib16)
#endif
	
	jmp		115f

114:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_VX16_LIB16
#else
	CALL(inner_kernel_gemm_nt_vx16_lib16)
#endif

115:

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nt_16x16_vs_lib16)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- 16*sdb*sizeof(float)
// zmm0  <- []
// ...
// zmm15 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+...
// r12   <- B+...
// r13   <- 16*sdb*sizeof(float)
// zmm0  <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_NN_16X16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_kernel_gemm_nn_16x16_lib16)
#endif

	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch
	//prefetcht0	0+0*64(%r12) // software prefetch
	//prefetcht0	0+1*64(%r12) // software prefetch
	//prefetcht0	0+2*64(%r12) // software prefetch
	//prefetcht0	0+3*64(%r12) // software prefetch

	// preload
	vmovaps 		0(%r11), %zmm25 // A

	cmpl	$ 16, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastss	0*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 // A
	vbroadcastss	0*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	//prefetcht0	256+0*64(%r12) // software prefetch
	vbroadcastss	0*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	//prefetcht0	256+1*64(%r12) // software prefetch
	vbroadcastss	0*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	0*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	0*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	0*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	0*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	0*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	0*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	0*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	0*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	0*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	0*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	0*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	0*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	subl	$ 16, %r10d

#if 1

	// unroll 1 - 14
	movq			$ 0, %r14

5:

	addq	$ 128, %r11
	addq	$ 1, %r14

	// unroll 1
	vbroadcastss	-1*4+0*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0*64(%r11), %zmm25 // A
	vbroadcastss	-1*4+1*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	-1*4+2*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	-1*4+3*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	-1*4+4*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	-1*4+5*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	-1*4+6*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	-1*4+7*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	-1*4+8*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	-1*4+9*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	-1*4+10*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	-1*4+11*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	-1*4+12*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	-1*4+13*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	-1*4+14*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	-1*4+15*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 2
	vbroadcastss	0*4+0*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			1*64(%r11), %zmm26 // A
	vbroadcastss	0*4+1*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	0*4+2*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	0*4+3*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	0*4+4*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	0*4+5*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	0*4+6*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	0*4+7*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	0*4+8*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	0*4+9*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	0*4+10*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	0*4+11*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	0*4+12*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	0*4+13*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	0*4+14*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	0*4+15*64(%r12, %r14, 8), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	cmpq	$ 7, %r14
	jl		5b // main loop 

	addq	$ 128, %r11

#else

	// unroll 1
	vbroadcastss	1*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			2*64(%r11), %zmm25 // A
	vbroadcastss	1*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	1*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	1*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	1*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	1*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	1*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	1*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	1*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	1*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	1*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	1*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	1*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	1*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	1*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	1*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 2
	vbroadcastss	2*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			3*64(%r11), %zmm26 // A
	vbroadcastss	2*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	2*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	2*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	2*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	2*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	2*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	2*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	2*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	2*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	2*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	2*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	2*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	2*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	2*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	2*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 3
	vbroadcastss	3*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			4*64(%r11), %zmm25 // A
	vbroadcastss	3*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	3*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	3*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	3*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	3*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	3*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	3*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	3*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	3*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	3*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	3*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	3*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	3*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	3*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	3*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 4
	vbroadcastss	4*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			5*64(%r11), %zmm26 // A
	vbroadcastss	4*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	4*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	4*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	4*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	4*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	4*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	4*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	4*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	4*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	4*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	4*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	4*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	4*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	4*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	4*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 5
	vbroadcastss	5*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			6*64(%r11), %zmm25 // A
	vbroadcastss	5*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	5*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	5*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	5*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	5*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	5*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	5*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	5*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	5*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	5*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	5*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	5*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	5*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	5*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	5*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 6
	vbroadcastss	6*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			7*64(%r11), %zmm26 // A
	vbroadcastss	6*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	6*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	6*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	6*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	6*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	6*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	6*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	6*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	6*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	6*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	6*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	6*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	6*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	6*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	6*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 7
	vbroadcastss	7*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			8*64(%r11), %zmm25 // A
	vbroadcastss	7*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	7*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	7*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	7*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	7*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	7*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	7*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	7*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	7*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	7*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	7*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	7*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	7*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	7*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	7*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 8
	vbroadcastss	8*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			9*64(%r11), %zmm26 // A
	vbroadcastss	8*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	8*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	8*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	8*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	8*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	8*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	8*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	8*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	8*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	8*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	8*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	8*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	8*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	8*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 9
	vbroadcastss	9*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			10*64(%r11), %zmm25 // A
	vbroadcastss	9*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	9*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	9*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	9*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	9*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	9*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	9*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	9*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	9*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	9*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	9*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	9*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	9*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	9*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	9*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 10
	vbroadcastss	10*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			11*64(%r11), %zmm26 // A
	vbroadcastss	10*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	10*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	10*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	10*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	10*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	10*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	10*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	10*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	10*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	10*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	10*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	10*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	10*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	10*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	10*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 11
	vbroadcastss	11*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			12*64(%r11), %zmm25 // A
	vbroadcastss	11*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	11*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	11*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	11*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	11*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	11*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	11*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	11*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	11*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	11*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	11*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	11*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	11*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	11*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	11*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 12
	vbroadcastss	12*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			13*64(%r11), %zmm26 // A
	vbroadcastss	12*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	12*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	12*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	12*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	12*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	12*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	12*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	12*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	12*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	12*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	12*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	12*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	12*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	12*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 13
	vbroadcastss	13*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			14*64(%r11), %zmm25 // A
	vbroadcastss	13*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	13*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	13*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	13*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	13*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	13*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	13*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	13*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	13*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	13*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	13*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	13*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	13*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	13*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	13*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 14
	vbroadcastss	14*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			15*64(%r11), %zmm26 // A
	vbroadcastss	14*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	14*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	14*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	14*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	14*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	14*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	14*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	14*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	14*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	14*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	14*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	14*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	14*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	14*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	14*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

#endif

	addq	$ 1024, %r11

	// unroll 15
	vbroadcastss	15*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			0*64(%r11), %zmm25 // A
	vbroadcastss	15*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	15*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	15*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	15*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	15*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	15*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	15*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	15*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	15*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	15*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	15*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	15*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	15*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	15*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	15*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	%r13, %r12

	cmpl	$ 16, %r10d
	jg		1b // main loop 


0: // consider clean16-up
	
	cmpl	$ 15, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastss	0*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			64(%r11), %zmm26 // A
	vbroadcastss	0*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	0*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	0*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	0*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	0*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	0*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	0*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	0*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	0*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	0*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	0*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	0*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	0*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	0*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	0*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	subl	$ 16, %r10d

	// unroll 1
	vbroadcastss	1*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			2*64(%r11), %zmm25 // A
	vbroadcastss	1*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	1*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	1*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	1*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	1*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	1*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	1*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	1*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	1*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	1*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	1*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	1*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	1*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	1*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	1*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 2
	vbroadcastss	2*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			3*64(%r11), %zmm26 // A
	vbroadcastss	2*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	2*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	2*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	2*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	2*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	2*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	2*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	2*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	2*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	2*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	2*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	2*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	2*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	2*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	2*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 3
	vbroadcastss	3*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			4*64(%r11), %zmm25 // A
	vbroadcastss	3*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	3*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	3*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	3*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	3*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	3*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	3*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	3*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	3*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	3*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	3*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	3*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	3*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	3*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	3*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 4
	vbroadcastss	4*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			5*64(%r11), %zmm26 // A
	vbroadcastss	4*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	4*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	4*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	4*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	4*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	4*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	4*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	4*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	4*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	4*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	4*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	4*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	4*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	4*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	4*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 5
	vbroadcastss	5*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			6*64(%r11), %zmm25 // A
	vbroadcastss	5*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	5*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	5*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	5*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	5*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	5*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	5*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	5*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	5*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	5*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	5*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	5*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	5*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	5*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	5*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 6
	vbroadcastss	6*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			7*64(%r11), %zmm26 // A
	vbroadcastss	6*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	6*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	6*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	6*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	6*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	6*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	6*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	6*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	6*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	6*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	6*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	6*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	6*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	6*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	6*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 7
	vbroadcastss	7*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			8*64(%r11), %zmm25 // A
	vbroadcastss	7*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	7*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	7*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	7*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	7*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	7*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	7*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	7*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	7*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	7*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	7*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	7*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	7*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	7*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	7*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 8
	vbroadcastss	8*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			9*64(%r11), %zmm26 // A
	vbroadcastss	8*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	8*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	8*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	8*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	8*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	8*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	8*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	8*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	8*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	8*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	8*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	8*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	8*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	8*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	8*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 9
	vbroadcastss	9*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			10*64(%r11), %zmm25 // A
	vbroadcastss	9*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	9*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	9*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	9*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	9*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	9*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	9*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	9*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	9*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	9*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	9*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	9*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	9*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	9*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	9*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 10
	vbroadcastss	10*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			11*64(%r11), %zmm26 // A
	vbroadcastss	10*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	10*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	10*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	10*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	10*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	10*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	10*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	10*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	10*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	10*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	10*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	10*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	10*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	10*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	10*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 11
	vbroadcastss	11*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			12*64(%r11), %zmm25 // A
	vbroadcastss	11*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	11*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	11*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	11*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	11*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	11*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	11*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	11*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	11*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	11*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	11*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	11*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	11*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	11*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	11*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 12
	vbroadcastss	12*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			13*64(%r11), %zmm26 // A
	vbroadcastss	12*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	12*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	12*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	12*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	12*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	12*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	12*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	12*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	12*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	12*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	12*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	12*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	12*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	12*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	12*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	// unroll 13
	vbroadcastss	13*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	vmovaps			14*64(%r11), %zmm25 // A
	vbroadcastss	13*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	13*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	13*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	13*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	13*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	13*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	13*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	13*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	13*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	13*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	13*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	13*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	13*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	13*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	13*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15

	// unroll 14
	vbroadcastss	14*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vmovaps			15*64(%r11), %zmm26 // A
	vbroadcastss	14*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	14*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	14*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	14*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	14*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	14*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	14*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	14*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	14*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	14*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	14*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	14*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	14*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	14*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	14*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15
	addq	$ 1024, %r11

	// unroll 15
	vbroadcastss	15*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm0
	//vmovaps			0*64(%r11), %zmm25 // A
	vbroadcastss	15*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm1
	vbroadcastss	15*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm2
	vbroadcastss	15*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm3
	vbroadcastss	15*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm4
	vbroadcastss	15*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm5
	vbroadcastss	15*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm6
	vbroadcastss	15*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm7
	vbroadcastss	15*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm8
	vbroadcastss	15*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm9
	vbroadcastss	15*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm10
	vbroadcastss	15*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm11
	vbroadcastss	15*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm12
	vbroadcastss	15*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm13
	vbroadcastss	15*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm14
	vbroadcastss	15*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm26, %zmm24, %zmm15
	addq	%r13, %r12


	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vmovaps			0(%r11), %zmm25 // A
	vbroadcastss	0*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	0*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	0*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	0*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	0*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	0*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	0*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	0*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	0*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	0*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	0*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	0*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	0*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	0*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	0*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	0*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	addq	$ 64, %r11
	addq	$ 4, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_nn_16x16_lib16)
#endif





// common inner routine with file scope
//
// edge for B unaligned
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- 16*sdb*sizeof(double)
// r14   <- offB
// zmm0  <- []
// ...
// zmm15 <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+...
// r12   <- B+...
// r13   <- 16*sdb*sizeof(double)
// r14   <- offB
// zmm0  <- []
// ...
// zmm15 <- []


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_GEMM_NN_16X16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_edge_gemm_nn_16x16_lib16)
#endif
	
	cmpl			$ 0, %r14d // offset==0
	jle				2f // end

	cmpl			$ 0, %r10d // k==0
	jle				2f // end

	movl			$ 16, %r15d
	subl			%r14d, %r15d // 16-offsetB
	cmpl			%r10d, %r15d
//	jle				0f
//	movl			%r10d, %r15d // kend=min(k,16-offsetB)
//0:
	cmovgl			%r10d, %r15d // kend=min(k,16-offsetB)

	movl			%r14d, %eax
	sall			$ 2, %eax // offsetB*sizeof(float)
	addq			%rax, %r12 // B+offsetB*sizeof(double)

1:
	vmovaps			0(%r11), %zmm25 // A
	vbroadcastss	0*4+0*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm0
	vbroadcastss	0*4+1*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm1
	vbroadcastss	0*4+2*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm2
	vbroadcastss	0*4+3*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm3
	vbroadcastss	0*4+4*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm4
	vbroadcastss	0*4+5*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm5
	vbroadcastss	0*4+6*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm6
	vbroadcastss	0*4+7*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm7
	vbroadcastss	0*4+8*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm8
	vbroadcastss	0*4+9*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm9
	vbroadcastss	0*4+10*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm10
	vbroadcastss	0*4+11*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm11
	vbroadcastss	0*4+12*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm12
	vbroadcastss	0*4+13*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm13
	vbroadcastss	0*4+14*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm14
	vbroadcastss	0*4+15*64(%r12), %zmm24 // B
	vfmadd231ps		%zmm25, %zmm24, %zmm15

	subl			$ 1, %r10d // k-1
	subl			$ 1, %r15d // kend-1
	addq			$ 64, %r11 // A+1*bs*sizeof(float)
	addq			$ 4, %r12 // B+1*sizeof(float)

	cmpl			$ 0, %r15d
	jg				1b

	cmpl			$ 0, %r10d
	jle				2f // end

	addq			%r13, %r12
	subq			$ 64, %r12 // B+bs*(sdb-1)*sizeof(float)

2:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_gemm_nn_16x16_lib16)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r10   <- C
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_16X16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_16x16_lib16)
#endif
	
	// alpha
	vbroadcastss	0(%r10), %zmm25

	vmulps		%zmm0, %zmm25, %zmm0
	vmulps		%zmm1, %zmm25, %zmm1
	vmulps		%zmm2, %zmm25, %zmm2
	vmulps		%zmm3, %zmm25, %zmm3
	vmulps		%zmm4, %zmm25, %zmm4
	vmulps		%zmm5, %zmm25, %zmm5
	vmulps		%zmm6, %zmm25, %zmm6
	vmulps		%zmm7, %zmm25, %zmm7
	vmulps		%zmm8, %zmm25, %zmm8
	vmulps		%zmm9, %zmm25, %zmm9
	vmulps		%zmm10, %zmm25, %zmm10
	vmulps		%zmm11, %zmm25, %zmm11
	vmulps		%zmm12, %zmm25, %zmm12
	vmulps		%zmm13, %zmm25, %zmm13
	vmulps		%zmm14, %zmm25, %zmm14
	vmulps		%zmm15, %zmm25, %zmm15

	// beta
	vbroadcastss	0(%r11), %zmm24

	vxorps		%zmm25, %zmm25, %zmm25 // 0.0

	vucomiss	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	vmovaps		0(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm0
	vmovaps		64(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm1
	vmovaps		128(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm2
	vmovaps		192(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm3
	vmovaps		0+256(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm4
	vmovaps		64+256(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm5
	vmovaps		128+256(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm6
	vmovaps		192+256(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm7
	vmovaps		0+512(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm4
	vmovaps		64+512(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm5
	vmovaps		128+512(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm6
	vmovaps		192+512(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm7
	vmovaps		0+768(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm4
	vmovaps		64+768(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm5
	vmovaps		128+768(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm6
	vmovaps		192+768(%r12), %zmm25
	vfmadd231ps	%zmm24, %zmm25, %zmm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_16x16_lib16)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13d  <- m1
// r14d  <- n1
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13d  <- m1
// r14d  <- n1
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_16X16_VS_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_16x16_vs_lib16)
#endif
	
	// alpha
	vbroadcastss	0(%r10), %zmm25

	vmulps		%zmm0, %zmm25, %zmm0
	vmulps		%zmm1, %zmm25, %zmm1
	vmulps		%zmm2, %zmm25, %zmm2
	vmulps		%zmm3, %zmm25, %zmm3
	vmulps		%zmm4, %zmm25, %zmm4
	vmulps		%zmm5, %zmm25, %zmm5
	vmulps		%zmm6, %zmm25, %zmm6
	vmulps		%zmm7, %zmm25, %zmm7
	vmulps		%zmm8, %zmm25, %zmm8
	vmulps		%zmm9, %zmm25, %zmm9
	vmulps		%zmm10, %zmm25, %zmm10
	vmulps		%zmm11, %zmm25, %zmm11
	vmulps		%zmm12, %zmm25, %zmm12
	vmulps		%zmm13, %zmm25, %zmm13
	vmulps		%zmm14, %zmm25, %zmm14
	vmulps		%zmm15, %zmm25, %zmm15

	// beta
	vbroadcastss	0(%r11), %zmm24

	vxorps		%zmm25, %zmm25, %zmm25 // 0.0

	vucomiss	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm25, %xmm25
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC00(%rip), %zmm26
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %zmm26
#endif
	vbroadcastss	%xmm25, %zmm25
	vsubps		%zmm25, %zmm26, %zmm25
	vpmovd2m	%zmm25, %k1

	vmovaps		0(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm0 {%k1}
	cmpl	$ 2, %r14d
	jl		0f // end
	vmovaps		64(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm1 {%k1}
	cmpl	$ 3, %r14d
	jl		0f // end
	vmovaps		128(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm2 {%k1}
	cmpl	$ 4, %r14d
	jl		0f // end
	vmovaps		192(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm3 {%k1}
	cmpl	$ 5, %r14d
	jl		0f // end
	vmovaps		0+256(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm4 {%k1}
	cmpl	$ 6, %r14d
	jl		0f // end
	vmovaps		64+256(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm5 {%k1}
	cmpl	$ 7, %r14d
	jl		0f // end
	vmovaps		128+256(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm6 {%k1}
	cmpl	$ 8, %r14d
	jl		0f // end
	vmovaps		192+256(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm7 {%k1}
	cmpl	$ 9, %r14d
	jl		0f // end
	vmovaps		0+512(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm8 {%k1}
	cmpl	$ 10, %r14d
	jl		0f // end
	vmovaps		64+512(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm9 {%k1}
	cmpl	$ 11, %r14d
	jl		0f // end
	vmovaps		128+512(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm10 {%k1}
	cmpl	$ 12, %r14d
	jl		0f // end
	vmovaps		192+512(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm11 {%k1}
	cmpl	$ 13, %r14d
	jl		0f // end
	vmovaps		0+768(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm12 {%k1}
	cmpl	$ 14, %r14d
	jl		0f // end
	vmovaps		64+768(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm13 {%k1}
	cmpl	$ 15, %r14d
	jl		0f // end
	vmovaps		128+768(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm14 {%k1}
	je		0f // end {%k1}
	vmovaps		192+768(%r12), %zmm25 {%k1}{z}
	vfmadd231ps	%zmm24, %zmm25, %zmm15 {%k1}

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_16x16_vs_lib16)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10  <- D
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_16X16_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_store_16x16_lib16)
#endif
	
	vmovaps %zmm0,   0(%r10)
	vmovaps %zmm1,  64(%r10)
	vmovaps %zmm2, 128(%r10)
	vmovaps %zmm3, 192(%r10)
	vmovaps %zmm4,   0+256(%r10)
	vmovaps %zmm5,  64+256(%r10)
	vmovaps %zmm6, 128+256(%r10)
	vmovaps %zmm7, 192+256(%r10)
	vmovaps %zmm8,   0+512(%r10)
	vmovaps %zmm9,  64+512(%r10)
	vmovaps %zmm10, 128+512(%r10)
	vmovaps %zmm11, 192+512(%r10)
	vmovaps %zmm12,   0+768(%r10)
	vmovaps %zmm13,  64+768(%r10)
	vmovaps %zmm14, 128+768(%r10)
	vmovaps %zmm15, 192+768(%r10)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_16x16_lib16)
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10   <- D
// r11d  <- m1
// r12d  <- n1
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10   <- D
// r11d  <- m1
// r12d  <- n1
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_16X16_VS_LIB16
#else
	.p2align 4,,15
	FUN_START(inner_store_16x16_vs_lib16)
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r11d, %xmm25, %xmm25
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC00(%rip), %zmm24
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %zmm24
#endif
	vbroadcastss	%xmm25, %zmm25
	vsubps		%zmm25, %zmm24, %zmm25
	vpmovd2m	%zmm25, %k1

	vmovaps %zmm0,   0(%r10) {%k1}
	cmpl	$ 2, %r12d
	jl		0f // end
	vmovaps %zmm1,  64(%r10) {%k1}
	cmpl	$ 3, %r12d
	jl		0f // end
	vmovaps %zmm2, 128(%r10) {%k1}
	cmpl	$ 4, %r12d
	jl		0f // end
	vmovaps %zmm3, 192(%r10) {%k1}
	cmpl	$ 5, %r12d
	jl		0f // end
	vmovaps %zmm4,   0+256(%r10) {%k1}
	cmpl	$ 6, %r12d
	jl		0f // end
	vmovaps %zmm5,  64+256(%r10) {%k1}
	cmpl	$ 7, %r12d
	jl		0f // end
	vmovaps %zmm6, 128+256(%r10) {%k1}
	cmpl	$ 8, %r12d
	jl		0f // end
	vmovaps %zmm7, 192+256(%r10) {%k1}
	cmpl	$ 9, %r12d
	jl		0f // end
	vmovaps %zmm8, 0+512(%r10) {%k1}
	cmpl	$ 10, %r12d
	jl		0f // end
	vmovaps %zmm9, 64+512(%r10) {%k1}
	cmpl	$ 11, %r12d
	jl		0f // end
	vmovaps %zmm10, 128+512(%r10) {%k1}
	cmpl	$ 12, %r12d
	jl		0f // end
	vmovaps %zmm11, 192+512(%r10) {%k1}
	cmpl	$ 13, %r12d
	jl		0f // end
	vmovaps %zmm12, 0+768(%r10) {%k1}
	cmpl	$ 14, %r12d
	jl		0f // end
	vmovaps %zmm13, 64+768(%r10) {%k1}
	cmpl	$ 15, %r12d
	jl		0f // end
	vmovaps %zmm14, 128+768(%r10) {%k1}
	je		0f // end
	vmovaps %zmm15, 192+768(%r10) {%k1}

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_16x16_vs_lib16)
#endif





//                                  1      2             3         4         5            6         7
// void kernel_sgemm_nt_16x16_lib16(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_sgemm_nt_16x16_lib16)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_16X16_LIB16
#else
	CALL(inner_kernel_gemm_nt_16x16_lib16)
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X16_LIB16
#else
	CALL(inner_scale_ab_16x16_lib16)
#endif


	// store n

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_16X16_LIB16
#else
	CALL(inner_store_16x16_lib16)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_sgemm_nt_16x16_lib16)






//                                     1      2             3         4         5            6         7         8       9
// void kernel_sgemm_nt_16x16_vs_lib16(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D, int km, int kn);

	.p2align 4,,15
	GLOB_FUN_START(kernel_sgemm_nt_16x16_vs_lib16)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B
	movq	ARG8, %r13 // m1 
	movq	ARG9, %r14 // n1

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_16X16_VS_LIB16
#else
	CALL(inner_kernel_gemm_nt_16x16_vs_lib16)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG8, %r13 // m1 
	movq	ARG9, %r14 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X16_VS_LIB16
#else
	CALL(inner_scale_ab_16x16_vs_lib16)
#endif


	// store n

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // m1 
	movq	ARG9, %r12 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_16X16_VS_LIB16
#else
	CALL(inner_store_16x16_vs_lib16)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_sgemm_nt_16x16_vs_lib16)





//                                  1      2             3         4            5         6        7            8         9
// void kernel_sgemm_nn_16x16_lib16(int k, float *alpha, float *A, int offsetB, float *B, int sdb, float *beta, float *C, float *D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_sgemm_nn_16x16_lib16)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG5, %r12  // B
	movq	ARG6, %r13 // sdb
	sall	$ 6, %r13d // 16*sdb*sizeof(float)
	movq	ARG4, %r14 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_GEMM_NN_16X16_LIB16
#else
	CALL(inner_edge_gemm_nn_16x16_lib16)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NN_16X16_LIB16
#else
	CALL(inner_kernel_gemm_nn_16x16_lib16)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X16_LIB16
#else
	CALL(inner_scale_ab_16x16_lib16)
#endif


	// store n

	movq	ARG9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_16X16_LIB16
#else
	CALL(inner_store_16x16_lib16)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_sgemm_nn_16x16_lib16)





//#if defined(BLAS_API)
//#if ( defined(BLAS_API) | ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) ) )

//#include "kernel_sgemm_16x16_lib.S"

//#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#elif defined(OS_WINDOWS)
	.section .rdata,"dr"
#endif


#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 64
.LC00:
#elif defined(OS_MAC)
	.align 6
LC00:
#endif
	.float 0.5
	.float 1.5
	.float 2.5
	.float 3.5
	.float 4.5
	.float 5.5
	.float 6.5
	.float 7.5
	.float 8.5
	.float 9.5
	.float 10.5
	.float 11.5
	.float 12.5
	.float 13.5
	.float 14.5
	.float 15.5


#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 64
.LC03:
#elif defined(OS_MAC)
	.align 6
LC03:
#endif
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0
	.float -1.0


#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 64
.LC04:
#elif defined(OS_MAC)
	.align 6
LC04:
#endif
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0
	.float 1.0


#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 64
.LC05:
#elif defined(OS_MAC)
	.align 6
LC05:
#endif
	.long 0x0
	.long 0x1
	.long 0x2
	.long 0x3
	.long 0x4
	.long 0x5
	.long 0x6
	.long 0x7
	.long 0x8
	.long 0x9
	.long 0xa
	.long 0xb
	.long 0xc
	.long 0xd
	.long 0xe
	.long 0xf



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

