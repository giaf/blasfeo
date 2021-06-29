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
	vpxord	%zmm0, %zmm0, %zmm0; \
	vmovapd	%zmm0, %zmm1; \
	vmovapd	%zmm0, %zmm2; \
	vmovapd	%zmm0, %zmm3; \
	vmovapd	%zmm0, %zmm4; \
	vmovapd	%zmm0, %zmm5; \
	vmovapd	%zmm0, %zmm6; \
	vmovapd	%zmm0, %zmm7;
//#define NEG_ACC \
//	vmovapd		.LC11(%rip), %ymm15; \
//	vxorpd		%ymm15, %ymm0, %ymm0; \
//	vxorpd		%ymm15, %ymm1, %ymm1; \
//	vxorpd		%ymm15, %ymm2, %ymm2; \
//	vxorpd		%ymm15, %ymm3, %ymm3; \
//	vxorpd		%ymm15, %ymm4, %ymm4; \
//	vxorpd		%ymm15, %ymm5, %ymm5; \
//	vxorpd		%ymm15, %ymm6, %ymm6; \
//	vxorpd		%ymm15, %ymm7, %ymm7;

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
	vpxord	%zmm0, %zmm0, %zmm0; \
	vmovapd	%zmm0, %zmm1; \
	vmovapd	%zmm0, %zmm2; \
	vmovapd	%zmm0, %zmm3; \
	vmovapd	%zmm0, %zmm4; \
	vmovapd	%zmm0, %zmm5; \
	vmovapd	%zmm0, %zmm6; \
	vmovapd	%zmm0, %zmm7;
//#define NEG_ACC \
//	vmovapd		LC11(%rip), %ymm15; \
//	vxorpd		%ymm15, %ymm0, %ymm0; \
//	vxorpd		%ymm15, %ymm1, %ymm1; \
//	vxorpd		%ymm15, %ymm2, %ymm2; \
//	vxorpd		%ymm15, %ymm3, %ymm3; \
//	vxorpd		%ymm15, %ymm4, %ymm4; \
//	vxorpd		%ymm15, %ymm5, %ymm5; \
//	vxorpd		%ymm15, %ymm6, %ymm6; \
//	vxorpd		%ymm15, %ymm7, %ymm7;

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
	vpxord	%zmm0, %zmm0, %zmm0; \
	vmovapd	%zmm0, %zmm1; \
	vmovapd	%zmm0, %zmm2; \
	vmovapd	%zmm0, %zmm3; \
	vmovapd	%zmm0, %zmm4; \
	vmovapd	%zmm0, %zmm5; \
	vmovapd	%zmm0, %zmm6; \
	vmovapd	%zmm0, %zmm7;
//#define NEG_ACC \
//	vmovapd		.LC11(%rip), %ymm15; \
//	vxorpd		%ymm15, %ymm0, %ymm0; \
//	vxorpd		%ymm15, %ymm1, %ymm1; \
//	vxorpd		%ymm15, %ymm2, %ymm2; \
//	vxorpd		%ymm15, %ymm3, %ymm3; \
//	vxorpd		%ymm15, %ymm4, %ymm4; \
//	vxorpd		%ymm15, %ymm5, %ymm5; \
//	vxorpd		%ymm15, %ymm6, %ymm6; \
//	vxorpd		%ymm15, %ymm7, %ymm7;

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
	.macro INNER_KERNEL_DGEMM_NT_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nt_8x8_lib8)
#endif
	
	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch
//	prefetcht0	0(%r12) // software prefetch
//	prefetcht0	0+64(%r12) // software prefetch

	// preload
	vmovapd 		0(%r11), %zmm25 // A

	cmpl	$ 4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vbroadcastsd	0(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			64(%r11), %zmm26 // A
	vbroadcastsd	8(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
//	prefetcht0	128(%r12) // software prefetch
	vbroadcastsd	16(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
//	prefetcht0	128+64(%r12) // software prefetch
	vbroadcastsd	24(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	40(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	56(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastsd	0+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			128(%r11), %zmm25 // A
	vbroadcastsd	8+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	16+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	32+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	48+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 0
	vbroadcastsd	0+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			192(%r11), %zmm26 // A
	vbroadcastsd	8+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
//	prefetcht0	128(%r12) // software prefetch
	vbroadcastsd	16+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
//	prefetcht0	128+64(%r12) // software prefetch
	vbroadcastsd	24+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	40+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	56+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	addq	$ 256, %r11

	// unroll 0
	vbroadcastsd	0+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			0(%r11), %zmm25 // A
	vbroadcastsd	8+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	16+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	32+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	48+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7
	addq	$ 256, %r12

	cmpl	$ 4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$ 3, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastsd	0(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			64(%r11), %zmm26 // A
	vbroadcastsd	8(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
//	prefetcht0	128(%r12) // software prefetch
	vbroadcastsd	16(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
//	prefetcht0	128+64(%r12) // software prefetch
	vbroadcastsd	24(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	40(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	56(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	subl	$ 4, %r10d

	// unroll 0
	vbroadcastsd	0+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			128(%r11), %zmm25 // A
	vbroadcastsd	8+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	16+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	32+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	48+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 0
	vbroadcastsd	0+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			192(%r11), %zmm26 // A
	vbroadcastsd	8+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
//	prefetcht0	128(%r12) // software prefetch
	vbroadcastsd	16+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
//	prefetcht0	128+64(%r12) // software prefetch
	vbroadcastsd	24+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	40+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	56+128(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	addq	$ 256, %r11

	// unroll 0
	vbroadcastsd	0+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
//	vmovapd			0(%r11), %zmm25 // A
	vbroadcastsd	8+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	16+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	32+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	48+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+192(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7
	addq	$ 256, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vmovapd			0(%r11), %zmm25 // A
	vbroadcastsd	0(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vbroadcastsd	8(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	16(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	24(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	40(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	56(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

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

	FUN_END(inner_kernel_dgemm_nt_8x8_lib8)
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- 4*sdb*sizeof(double)
// zmm0  <- []
// ...
// zmm7  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// r13   <- 4*sdb*sizeof(double)
// zmm0  <- []
// ...
// zmm7  <- []

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dgemm_nn_8x8_lib8)
#endif
	
	cmpl	$ 0, %r10d
	jle		2f // return

	// prefetch
//	prefetcht0	0(%r12) // software prefetch
//	prefetcht0	0+64(%r12) // software prefetch

	// preload
	vmovapd 		0(%r11), %zmm25 // A

	cmpl	$ 8, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	128(%r12) // software prefetch
//	prefetcht0	128+64(%r12) // software prefetch

	// unroll 0
	vbroadcastsd	0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			1*64(%r11), %zmm26 // A
	vbroadcastsd	1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	subl	$ 8, %r10d

	// unroll 1
	vbroadcastsd	8+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			2*64(%r11), %zmm25 // A
	vbroadcastsd	8+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	8+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	8+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	8+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	8+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	8+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	8+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 2
	vbroadcastsd	16+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			3*64(%r11), %zmm26 // A
	vbroadcastsd	16+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	16+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	16+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	16+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	16+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	16+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	16+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	// unroll 3
	vbroadcastsd	24+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			4*64(%r11), %zmm25 // A
	vbroadcastsd	24+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	24+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	24+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	24+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	24+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	24+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 4
	vbroadcastsd	32+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			5*64(%r11), %zmm26 // A
	vbroadcastsd	32+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	32+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	32+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	32+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	32+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	32+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	// unroll 5
	vbroadcastsd	40+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			6*64(%r11), %zmm25 // A
	vbroadcastsd	40+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	40+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	40+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	40+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	40+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	40+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 6
	vbroadcastsd	48+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			7*64(%r11), %zmm26 // A
	vbroadcastsd	48+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	48+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	48+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	48+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	48+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	48+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	addq	$ 512, %r11

	// unroll 6
	vbroadcastsd	56+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			0*64(%r11), %zmm25 // A
	vbroadcastsd	56+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	56+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	56+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	56+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	56+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	56+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7
	addq	%r13, %r12

	cmpl	$ 8, %r10d
	jg		1b // main loop 


0: // consider clean8-up
	
	cmpl	$ 7, %r10d
	jle		4f // clean1

	// unroll 0
	vbroadcastsd	0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			1*64(%r11), %zmm26 // A
	vbroadcastsd	1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	subl	$ 8, %r10d

	// unroll 1
	vbroadcastsd	8+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			2*64(%r11), %zmm25 // A
	vbroadcastsd	8+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	8+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	8+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	8+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	8+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	8+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	8+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 2
	vbroadcastsd	16+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			3*64(%r11), %zmm26 // A
	vbroadcastsd	16+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	16+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	16+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	16+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	16+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	16+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	16+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	// unroll 3
	vbroadcastsd	24+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			4*64(%r11), %zmm25 // A
	vbroadcastsd	24+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	24+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	24+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	24+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	24+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	24+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	24+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 4
	vbroadcastsd	32+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			5*64(%r11), %zmm26 // A
	vbroadcastsd	32+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	32+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	32+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	32+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	32+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	32+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	32+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	// unroll 5
	vbroadcastsd	40+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
	vmovapd			6*64(%r11), %zmm25 // A
	vbroadcastsd	40+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	40+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	40+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	40+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	40+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	40+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	40+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7

	// unroll 6
	vbroadcastsd	48+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vmovapd			7*64(%r11), %zmm26 // A
	vbroadcastsd	48+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	48+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	48+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	48+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	48+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	48+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	48+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7
	addq	$ 512, %r11

	// unroll 6
	vbroadcastsd	56+0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm0
//	vmovapd			0*64(%r11), %zmm25 // A
	vbroadcastsd	56+1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm1
	vbroadcastsd	56+2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm2
	vbroadcastsd	56+3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm3
	vbroadcastsd	56+4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm4
	vbroadcastsd	56+5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm5
	vbroadcastsd	56+6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm6
	vbroadcastsd	56+7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm26, %zmm24, %zmm7
	addq	%r13, %r12

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$ 0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vmovapd			0(%r11), %zmm25 // A
	vbroadcastsd	0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vbroadcastsd	1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	addq	$ 64, %r11
	addq	$ 8, %r12
	subl	$ 1, %r10d

	cmpl	$ 0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_dgemm_nn_8x8_lib8)
#endif





// common inner routine with file scope
//
// edge for B unaligned
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- 4*sdb*sizeof(double)
// r14   <- offB
// zmm0  <- []
// ...
// zmm7  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+8*k*sizeof(double)
// r12   <- B+8*k*sizeof(double)
// r13   <- 4*sdb*sizeof(double)
// r14   <- offB
// zmm0  <- []
// ...
// zmm7  <- []


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_edge_dgemm_nn_8x8_lib8)
#endif
	
	cmpl			$ 0, %r14d // offset==0
	jle				2f // end

	cmpl			$ 0, %r10d // k==0
	jle				2f // end

	movl			$ 8, %r15d
	subl			%r14d, %r15d // 8-offsetB
	cmpl			%r10d, %r15d
//	jle				0f
//	movl			%r10d, %r15d // kend=min(k,4-offsetB)
//0:
	cmovgl			%r10d, %r15d // kend=min(k,4-offsetB)

	movl			%r14d, %eax
	sall			$ 3, %eax // offsetB*sizeof(double)
	addq			%rax, %r12 // B+offsetB*sizeof(double)

1:
	vmovapd			0(%r11), %zmm25 // A
	vbroadcastsd	0*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm0
	vbroadcastsd	1*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm1
	vbroadcastsd	2*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm2
	vbroadcastsd	3*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm3
	vbroadcastsd	4*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm4
	vbroadcastsd	5*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm5
	vbroadcastsd	6*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm6
	vbroadcastsd	7*64(%r12), %zmm24 // B
	vfmadd231pd		%zmm25, %zmm24, %zmm7

	subl			$ 1, %r10d // k-1
	subl			$ 1, %r15d // kend-1
	addq			$ 64, %r11 // A+1*bs*sizeof(double)
	addq			$ 8, %r12 // B+1*sizeof(double)

	cmpl			$ 0, %r15d
	jg				1b

	cmpl			$ 0, %r10d
	jle				2f // end

	addq			%r13, %r12
	subq			$ 64, %r12 // B+bs*(sdb-1)*sizeof(double)

2:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dgemm_nn_8x8_lib8)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r10   <- C
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_8x8_lib8)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm25

	vmulpd		%zmm0, %zmm25, %zmm0
	vmulpd		%zmm1, %zmm25, %zmm1
	vmulpd		%zmm2, %zmm25, %zmm2
	vmulpd		%zmm3, %zmm25, %zmm3
	vmulpd		%zmm4, %zmm25, %zmm4
	vmulpd		%zmm5, %zmm25, %zmm5
	vmulpd		%zmm6, %zmm25, %zmm6
	vmulpd		%zmm7, %zmm25, %zmm7

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	vmovapd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	vmovapd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	vmovapd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	vmovapd		192(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	vmovapd		0+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	vmovapd		64+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	vmovapd		128+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	vmovapd		192+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x8_lib8)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13d  <- n1
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13d  <- n1
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_VS_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_8x8_vs_lib8)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm25

	vmulpd		%zmm0, %zmm25, %zmm0
	vmulpd		%zmm1, %zmm25, %zmm1
	vmulpd		%zmm2, %zmm25, %zmm2
	vmulpd		%zmm3, %zmm25, %zmm3
	vmulpd		%zmm4, %zmm25, %zmm4
	vmulpd		%zmm5, %zmm25, %zmm5
	vmulpd		%zmm6, %zmm25, %zmm6
	vmulpd		%zmm7, %zmm25, %zmm7

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	vmovapd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	cmpl	$ 2, %r13d
	jl		0f // end
	vmovapd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	cmpl	$ 3, %r13d
	jl		0f // end
	vmovapd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	cmpl	$ 4, %r13d
	jl		0f // end
	vmovapd		192(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	cmpl	$ 5, %r13d
	jl		0f // end
	vmovapd		0+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	cmpl	$ 6, %r13d
	jl		0f // end
	vmovapd		64+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	cmpl	$ 7, %r13d
	jl		0f // end
	vmovapd		128+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	je		0f // end
	vmovapd		192+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x8_vs_lib8)
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 8*sdc*sizeof(double)
// r15   <- n0
// rax   <- n1
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 8*sdc*sizeof(double)
// r15   <- n0
// rax   <- n1
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_GEN_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_8x8_gen_lib8)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm24

	vmulpd		%zmm0, %zmm24, %zmm0
	vmulpd		%zmm1, %zmm24, %zmm1
	vmulpd		%zmm2, %zmm24, %zmm2
	vmulpd		%zmm3, %zmm24, %zmm3
	vmulpd		%zmm4, %zmm24, %zmm4
	vmulpd		%zmm5, %zmm24, %zmm5
	vmulpd		%zmm6, %zmm24, %zmm6
	vmulpd		%zmm7, %zmm24, %zmm7

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm24, %xmm25 // beta==0.0 ?
	je			8f // end

	cmpl	$ 0, %r12d
	jg		0f

	// offset==0

	cmpl	$ 0, %r15d
	jg		1f // next
	vmovapd		0(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	jmp		2f
1:
	cmpl	$ 1, %r15d
	jg		1f // next
2:
	cmpl	$ 2, %eax
	jl		8f // end
	vmovapd		64(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	jmp		2f
1:
	cmpl	$ 2, %r15d
	jg		1f // next
2:
	cmpl	$ 3, %eax
	jl		8f // end
	vmovapd		128(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	jmp		2f
1:
	cmpl	$ 3, %r15d
	jg		1f // next
2:
	cmpl	$ 4, %eax
	jl		8f // end
	vmovapd		192(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	jmp		2f
1:
	cmpl	$ 4, %r15d
	jg		1f // next
2:
	cmpl	$ 5, %eax
	jl		8f // end
	vmovapd		0+256(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	jmp		2f
1:
	cmpl	$ 5, %r15d
	jg		1f // next
2:
	cmpl	$ 6, %eax
	jl		8f // end
	vmovapd		64+256(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	jmp		2f
1:
	cmpl	$ 6, %r15d
	jg		1f // next
2:
	cmpl	$ 7, %eax
	jl		8f // end
	vmovapd		128+256(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	jmp		2f
1:
	cmpl	$ 7, %r15d
	jg		8f // end 1f // next
2:
	cmpl	$ 7, %eax
	je		8f // end
	vmovapd		192+256(%r13), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7
//1:

	jmp		8f

0:

	cmpl	$ 3, %r12d
	jg		14f

	cmpl	$ 1, %r12d
	jg		11f

	jmp		0f

11: // 2 3

	cmpl	$ 2, %r12d
	jg		2f

	jmp		1f

14: // 4 5 6 7

	cmpl	$ 5, %r12d
	jg		16f

	cmpl	$ 4, %r12d
	jg		4f

	jmp		3f

16: // 6 7

	cmpl	$ 6, %r12d
	jg		6f

	jmp		5f

0: 	// offset==1

	movl	$ 0x7f, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0x80, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	8(%r13), %rbx

	jmp		7f

1:  // offset==2

	movl	$ 0x3f, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xc0, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	16(%r13), %rbx

	jmp		7f

2:  // offset==3

	movl	$ 0x1f, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xe0, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	24(%r13), %rbx

	jmp		7f

3:  // offset==4

	movl	$ 0x0f, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xf0, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	32(%r13), %rbx

	jmp		7f

4:  // offset==5

	movl	$ 0x07, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xf8, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	40(%r13), %rbx

	jmp		7f

5:  // offset==6

	movl	$ 0x03, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xfc, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	48(%r13), %rbx

	jmp		7f

6:  // offset==7

	movl	$ 0x01, %ebx
	kmovd	%ebx, %k2
//	kandd	%k2, %k1, %k2
	movl	$ 0xfe, %ebx
	kmovd	%ebx, %k3
//	kandd	%k3, %k1, %k3

	leaq	56(%r13), %rbx

//	jmp		7f

7:

	cmpl	$ 0, %r15d
	jg		1f // next
	vmovupd 0(%rbx), %zmm25 {%k2}
	vmovupd 0-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	jmp		2f
1:
	cmpl	$ 1, %r15d
	jg		1f // next
2:
	cmpl	$ 2, %eax
	jl		8f // end
	vmovupd 64(%rbx), %zmm25 {%k2}
	vmovupd 64-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	jmp		2f
1:
	cmpl	$ 2, %r15d
	jg		1f // next
2:
	cmpl	$ 3, %eax
	jl		8f // end
	vmovupd 128(%rbx), %zmm25 {%k2}
	vmovupd 128-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	jmp		2f
1:
	cmpl	$ 3, %r15d
	jg		1f // next
2:
	cmpl	$ 4, %eax
	jl		8f // end
	vmovupd 192(%rbx), %zmm25 {%k2}
	vmovupd 192-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	jmp		2f
1:
	cmpl	$ 4, %r15d
	jg		1f // next
2:
	cmpl	$ 5, %eax
	jl		8f // end
	vmovupd 0+256(%rbx), %zmm25 {%k2}
	vmovupd 0+256-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	jmp		2f
1:
	cmpl	$ 5, %r15d
	jg		1f // next
2:
	cmpl	$ 6, %eax
	jl		8f // end
	vmovupd 64+256(%rbx), %zmm25 {%k2}
	vmovupd 64+256-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	jmp		2f
1:
	cmpl	$ 6, %r15d
	jg		1f // next
2:
	cmpl	$ 7, %eax
	jl		8f // end
	vmovupd 128+256(%rbx), %zmm25 {%k2}
	vmovupd 128+256-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	jmp		2f
1:
	cmpl	$ 7, %r15d
	jg		8f // end 1f // next
2:
	cmpl	$ 7, %eax
	je		8f // end
	vmovupd 192+256(%rbx), %zmm25 {%k2}
	vmovupd 192+256-64(%rbx, %r14), %zmm25 {%k3}
	vfmadd231pd	%zmm24, %zmm25, %zmm7
//1:

8:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x8_gen_lib8)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// zmm0 <- []
// ...
// zmm7 <- []
//
// output arguments:
// r10  <- D
// zmm0 <- []
// ...
// zmm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_store_8x8_lib8)
#endif
	
	vmovapd %zmm0,   0(%r10)
	vmovapd %zmm1,  64(%r10)
	vmovapd %zmm2, 128(%r10)
	vmovapd %zmm3, 192(%r10)
	vmovapd %zmm4,   0+256(%r10)
	vmovapd %zmm5,  64+256(%r10)
	vmovapd %zmm6, 128+256(%r10)
	vmovapd %zmm7, 192+256(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x8_lib8)
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
// zmm7 <- []
//
// output arguments:
// r10   <- D
// r11d  <- m1
// r12d  <- n1
// zmm0 <- []
// ...
// zmm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_VS_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_store_8x8_vs_lib8)
#endif
	
	movq	%rcx, %r14 // backup
	movl	$ 8, %ecx // saturate to ps=8
	cmpl	%r11d, %ecx
	cmovgl	%r11d, %ecx
	movl	$ 1, %r13d // shift base
	sall	%cl, %r13d // shift by cl
	subl	$ 1, %r13d
	kmovd	%r13d, %k1
	movq	%r14, %rcx // restore

	vmovapd %zmm0,   0(%r10) {%k1}
	cmpl	$ 2, %r12d
	jl		0f // end
	vmovapd %zmm1,  64(%r10) {%k1}
	cmpl	$ 3, %r12d
	jl		0f // end
	vmovapd %zmm2, 128(%r10) {%k1}
	cmpl	$ 4, %r12d
	jl		0f // end
	vmovapd %zmm3, 192(%r10) {%k1}
	cmpl	$ 5, %r12d
	jl		0f // end
	vmovapd %zmm4,   0+256(%r10) {%k1}
	cmpl	$ 6, %r12d
	jl		0f // end
	vmovapd %zmm5,  64+256(%r10) {%k1}
	cmpl	$ 7, %r12d
	jl		0f // end
	vmovapd %zmm6, 128+256(%r10) {%k1}
	je		0f // end
	vmovapd %zmm7, 192+256(%r10) {%k1}

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x8_vs_lib8)
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
// zmm0 <- []
// ...
// zmm7 <- []
//
// output arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n1-n0
// rax  <- n1-n0
// zmm0 <- []
// ...
// zmm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_GEN_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_store_8x8_gen_lib8)
#endif

#if 1

	// compute mask for rows
	vcvtsi2sd	%r13d, %xmm25, %xmm25
	vcvtsi2sd	%r14d, %xmm26, %xmm26
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC00(%rip), %zmm24
#elif defined(OS_MAC)
	vmovupd		LC00(%rip), %zmm24
#endif
	vbroadcastsd	%xmm25, %zmm25
	vbroadcastsd	%xmm26, %zmm26
	vsubpd		%zmm24, %zmm25, %zmm25
	vsubpd		%zmm26, %zmm24, %zmm26
	vandpd		%zmm25, %zmm26, %zmm26

	vpmovq2m	%zmm26, %k1

#else

	movq	%rcx, %rbp // backup

	movl	$ 0, %ecx // saturate to 0
	cmpl	%r13d, %ecx
	cmovgl	%ecx, %r13d

	movl	$ 8, %ecx // saturate to ps=8
	cmpl	%r13d, %ecx
	cmovgl	%r13d, %ecx

	movl	$ 1, %ebx // shift base
	sall	%cl, %ebx // shift by cl
	subl	$ 1, %ebx
	notl	%ebx
	kmovd	%ebx, %k1

	movl	$ 8, %ecx // saturate to ps=8
	cmpl	%r14d, %ecx
	cmovgl	%r14d, %ecx
	movl	$ 1, %ebx // shift base
	sall	%cl, %ebx // shift by cl
	subl	$ 1, %ebx
	kmovd	%ebx, %k2

	kandd	%k2, %k1, %k1

	movq	%rbp, %rcx // restore

#endif


	// return if no cols
	cmpl	$ 0, %eax
	jle		8f

	cmpl	$ 0, %r10d
	jg		0f

	// offset==0

	cmpl	$ 0, %r15d
	jg		1f // next
	vmovapd %zmm0,   0(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 1, %r15d
	jg		1f // next
2:
	cmpl	$ 2, %eax
	jl		8f // end
	vmovapd %zmm1,  64(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 2, %r15d
	jg		1f // next
2:
	cmpl	$ 3, %eax
	jl		8f // end
	vmovapd %zmm2, 128(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 3, %r15d
	jg		1f // next
2:
	cmpl	$ 4, %eax
	jl		8f // end
	vmovapd %zmm3, 192(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 4, %r15d
	jg		1f // next
2:
	cmpl	$ 5, %eax
	jl		8f // end
	vmovapd %zmm4,   0+256(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 5, %r15d
	jg		1f // next
2:
	cmpl	$ 6, %eax
	jl		8f // end
	vmovapd %zmm5,  64+256(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 6, %r15d
	jg		1f // next
2:
	cmpl	$ 7, %eax
	jl		8f // end
	vmovapd %zmm6, 128+256(%r11) {%k1}
	jmp		2f
1:
	cmpl	$ 7, %r15d
	jg		8f // end 1f // next
2:
	cmpl	$ 7, %eax
	je		8f // end
	vmovapd %zmm7, 192+256(%r11) {%k1}
//1:

	jmp		8f

0:
	
	cmpl	$ 3, %r10d
	jg		14f

	cmpl	$ 1, %r10d
	jg		11f

	jmp		0f

11: // 2 3

	cmpl	$ 2, %r10d
	jg		2f

	jmp		1f

14: // 4 5 6 7

	cmpl	$ 5, %r10d
	jg		16f

	cmpl	$ 4, %r10d
	jg		4f

	jmp		3f

16: // 6 7

	cmpl	$ 6, %r10d
	jg		6f

	jmp		5f

0: 	// offset==1

	movl	$ 0x7f, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0x80, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	8(%r11), %rbx

	jmp		7f

1:  // offset==2

	movl	$ 0x3f, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xc0, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	16(%r11), %rbx

	jmp		7f

2:  // offset==3

	movl	$ 0x1f, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xe0, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	24(%r11), %rbx

	jmp		7f

3:  // offset==4

	movl	$ 0x0f, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xf0, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	32(%r11), %rbx

	jmp		7f

4:  // offset==5

	movl	$ 0x07, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xf8, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	40(%r11), %rbx

	jmp		7f

5:  // offset==6

	movl	$ 0x03, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xfc, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	48(%r11), %rbx

	jmp		7f

6:  // offset==7

	movl	$ 0x01, %ebx
	kmovd	%ebx, %k2
	kandd	%k2, %k1, %k2
	movl	$ 0xfe, %ebx
	kmovd	%ebx, %k3
	kandd	%k3, %k1, %k3

	leaq	56(%r11), %rbx

//	jmp		7f

7:

	cmpl	$ 0, %r15d
	jg		1f // next
	vmovupd %zmm0,   0(%rbx) {%k2}
	vmovupd %zmm0,   0-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 1, %r15d
	jg		1f // next
2:
	cmpl	$ 2, %eax
	jl		8f // end
	vmovupd %zmm1,  64(%rbx) {%k2}
	vmovupd %zmm1,  64-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 2, %r15d
	jg		1f // next
2:
	cmpl	$ 3, %eax
	jl		8f // end
	vmovupd %zmm2, 128(%rbx) {%k2}
	vmovupd %zmm2, 128-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 3, %r15d
	jg		1f // next
2:
	cmpl	$ 4, %eax
	jl		8f // end
	vmovupd %zmm3, 192(%rbx) {%k2}
	vmovupd %zmm3, 192-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 4, %r15d
	jg		1f // next
2:
	cmpl	$ 5, %eax
	jl		8f // end
	vmovupd %zmm4,   0+256(%rbx) {%k2}
	vmovupd %zmm4,   0+256-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 5, %r15d
	jg		1f // next
2:
	cmpl	$ 6, %eax
	jl		8f // end
	vmovupd %zmm5,  64+256(%rbx) {%k2}
	vmovupd %zmm5,  64+256-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 6, %r15d
	jg		1f // next
2:
	cmpl	$ 7, %eax
	jl		8f // end
	vmovupd %zmm6, 128+256(%rbx) {%k2}
	vmovupd %zmm6, 128+256-64(%rbx, %r12) {%k3}
	jmp		2f
1:
	cmpl	$ 7, %r15d
	jg		8f // end 1f // next
2:
	cmpl	$ 7, %eax
	je		8f // end
	vmovupd %zmm7, 192+256(%rbx) {%k2}
	vmovupd %zmm7, 192+256-64(%rbx, %r12) {%k3}
//1:

8:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x8_gen_lib8)
#endif





// common inner routine with file scope
//
// transpose
//
// input arguments:
// zmm0  <- []
// ...
// zmm7  <- []
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_TRAN_8X8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_tran_8x8_lib8)
#endif

	vunpcklpd	%zmm1, %zmm0, %zmm24
	vunpckhpd	%zmm1, %zmm0, %zmm25
	vunpcklpd	%zmm3, %zmm2, %zmm26
	vunpckhpd	%zmm3, %zmm2, %zmm27
	vunpcklpd	%zmm5, %zmm4, %zmm28
	vunpckhpd	%zmm5, %zmm4, %zmm29
	vunpcklpd	%zmm7, %zmm6, %zmm30
	vunpckhpd	%zmm7, %zmm6, %zmm31

	movl	$ 0xcc, %r10d
	kmovd	%r10d, %k1
	movl	$ 0x33, %r10d
	kmovd	%r10d, %k2

	vmovapd		%zmm24, %zmm0
	vpermq		$ 0x40, %zmm26, %zmm24 {%k1}
	vpermq		$ 0x0e, %zmm0, %zmm26 {%k2}
	vmovapd		%zmm25, %zmm0
	vpermq		$ 0x40, %zmm27, %zmm25 {%k1}
	vpermq		$ 0x0e, %zmm0, %zmm27 {%k2}
	vmovapd		%zmm28, %zmm0
	vpermq		$ 0x40, %zmm30, %zmm28 {%k1}
	vpermq		$ 0x0e, %zmm0, %zmm30 {%k2}
	vmovapd		%zmm29, %zmm0
	vpermq		$ 0x40, %zmm31, %zmm29 {%k1}
	vpermq		$ 0x0e, %zmm0, %zmm31 {%k2}

	vshuff64x2	$ 0x44, %zmm28, %zmm24, %zmm0
	vshuff64x2	$ 0xee, %zmm28, %zmm24, %zmm4
	vshuff64x2	$ 0x44, %zmm29, %zmm25, %zmm1
	vshuff64x2	$ 0xee, %zmm29, %zmm25, %zmm5
	vshuff64x2	$ 0x44, %zmm30, %zmm26, %zmm2
	vshuff64x2	$ 0xee, %zmm30, %zmm26, %zmm6
	vshuff64x2	$ 0x44, %zmm31, %zmm27, %zmm3
	vshuff64x2	$ 0xee, %zmm31, %zmm27, %zmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_tran_8x8_lib8)
#endif





//                               1      2              3          4          5             6          7
// void kernel_dgemm_nt_8x8_lib8(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_8x8_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nt_8x8_lib8)
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB8
#else
	CALL(inner_scale_ab_8x8_lib8)
#endif


	// store n

	movq	ARG7, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB8
#else
	CALL(inner_store_8x8_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_8x8_lib8)





//                                  1      2              3          4          5             6          7          8       9
// void kernel_dgemm_nt_8x8_vs_lib8(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_8x8_vs_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nt_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG9, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_VS_LIB8
#else
	CALL(inner_scale_ab_8x8_vs_lib8)
#endif


	// store n

	movq	ARG7, %r10 // D
	movq	ARG8, %r11 // km 
	movq	ARG9, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_VS_LIB8
#else
	CALL(inner_store_8x8_vs_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_8x8_vs_lib8)





//                                   1      2              3          4          5             6            7          8        9            10         11       12      13      14      15
// void kernel_dgemm_nt_8x8_gen_lib8(int k, double *alpha, double *A, double *B, double *beta, int offsetC, double *C, int sdc, int offsetD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_8x8_gen_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nt_8x8_lib8)
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12 // offsetC
	movq	ARG7, %r13 // C
	movq	ARG8, %r14 // sdc
	sall	$ 6, %r14d // 4*sdc*sizeof(double)
	movq	ARG14, %r15 // n0
	movq	ARG15, %rax // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_GEN_LIB8
#else
	CALL(inner_scale_ab_8x8_gen_lib8)
#endif


	// store n gen

	movq	ARG9, %r10 // offsetD
	movq	ARG10, %r11 // D
	movq	ARG11, %r12 // sdd
	sall	$ 6, %r12d // 4*sdb*sizeof(double)
	movq	ARG12, %r13 // m0
	movq	ARG13, %r14 // m1
	movq	ARG14, %r15 // n0
	movq	ARG15, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_GEN_LIB8
#else
	CALL(inner_store_8x8_gen_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_8x8_gen_lib8)





//                               1      2              3          4            5          6        7             8          9
// void kernel_dgemm_nn_8x8_lib8(int k, double *alpha, double *A, int offsetB, double *B, int sdb, double *beta, double *C, double *D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_8x8_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG5, %r12  // B
	movq	ARG6, %r13 // sdb
	sall	$ 6, %r13d // 4*sdb*sizeof(double)
	movq	ARG4, %r14 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB8
#else
	CALL(inner_scale_ab_8x8_lib8)
#endif


	// store n

	movq	ARG9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB8
#else
	CALL(inner_store_8x8_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_8x8_lib8)





//                                  1      2              3          4            5          6        7             8          9          10      11
// void kernel_dgemm_nn_8x8_vs_lib8(int k, double *alpha, double *A, int offsetB, double *B, int sdb, double *beta, double *C, double *D, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_8x8_vs_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG5, %r12  // B
	movq	ARG6, %r13 // sdb
	sall	$ 6, %r13d // 4*sdb*sizeof(double)
	movq	ARG4, %r14 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C
	movq	ARG11, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_VS_LIB8
#else
	CALL(inner_scale_ab_8x8_vs_lib8)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // km 
	movq	ARG11, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_VS_LIB8
#else
	CALL(inner_store_8x8_vs_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_8x8_vs_lib8)





//                                   1      2              3          4            5          6        7             8         9          10       11        12         13       14      15      16      17
// void kernel_dgemm_nn_8x8_gen_lib8(int k, double *alpha, double *A, int offsetB, double *B, int sdb, double *beta, int offC, double *C, int sdc, int offD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nn_8x8_gen_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG5, %r12  // B
	movq	ARG6, %r13 // sdb
	sall	$ 6, %r13d // 4*sdb*sizeof(double)
	movq	ARG4, %r14 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12 // offsetC
	movq	ARG9, %r13 // C
	movq	ARG10, %r14 // sdc
	sall	$ 6, %r14d // 4*sdc*sizeof(double)
	movq	ARG16, %r15 // n0
	movq	ARG17, %rax // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_GEN_LIB8
#else
	CALL(inner_scale_ab_8x8_gen_lib8)
#endif


	// store n

	movq	ARG11, %r10 // offsetD
	movq	ARG12, %r11 // D
	movq	ARG13, %r12 // sdd
	sall	$ 6, %r12d // 4*sdb*sizeof(double)
	movq	ARG14, %r13 // m0
	movq	ARG15, %r14 // m1
	movq	ARG16, %r15 // n0
	movq	ARG17, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_GEN_LIB8
#else
	CALL(inner_store_8x8_gen_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nn_8x8_gen_lib8)





//                               1      2              3          4            5          6        7             8          9
// void kernel_dgemm_tt_8x8_lib8(int k, double *alpha, int offsetA, double *A, int sda, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_tt_8x8_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG6, %r11  // B
	movq	ARG4, %r12  // A
	movq	ARG5, %r13 // sda
	sall	$ 6, %r13d // 8*sda*sizeof(double)
	movq	ARG3, %r14 // offsetA

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=1
	INNER_TRAN_8X8_LIB8
#else
	CALL(inner_tran_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB8
#else
	CALL(inner_scale_ab_8x8_lib8)
#endif


	// store n

	movq	ARG9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB8
#else
	CALL(inner_store_8x8_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_tt_8x8_lib8)





//                                  1      2              3          4            5          6        7             8          9          10      11
// void kernel_dgemm_tt_8x8_vs_lib8(int k, double *alpha, int offsetA, double *A, int sda, double *B, double *beta, double *C, double *D, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_tt_8x8_vs_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG6, %r11  // B
	movq	ARG4, %r12  // A
	movq	ARG5, %r13 // sda
	sall	$ 6, %r13d // 8*sda*sizeof(double)
	movq	ARG3, %r14 // offsetA

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=1
	INNER_TRAN_8X8_LIB8
#else
	CALL(inner_tran_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG11, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_VS_LIB8
#else
	CALL(inner_scale_ab_8x8_vs_lib8)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // km 
	movq	ARG11, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_VS_LIB8
#else
	CALL(inner_store_8x8_vs_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_tt_8x8_vs_lib8)





//                                   1      2              3            4          5        6          7             8         9          10       11        12         13       14      15      16      17
// void kernel_dgemm_tt_8x8_gen_lib8(int k, double *alpha, int offsetA, double *A, int sda, double *B, double *beta, int offC, double *C, int sdc, int offD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_tt_8x8_gen_lib8)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG6, %r11  // B
	movq	ARG4, %r12  // A
	movq	ARG5, %r13 // sda
	sall	$ 6, %r13d // 8*sda*sizeof(double)
	movq	ARG3, %r14 // offsetA

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_edge_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NN_8X8_LIB8
#else
	CALL(inner_kernel_dgemm_nn_8x8_lib8)
#endif

#if MACRO_LEVEL>=1
	INNER_TRAN_8X8_LIB8
#else
	CALL(inner_tran_8x8_lib8)
#endif


	// call inner blend 

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12 // offsetC
	movq	ARG9, %r13 // C
	movq	ARG10, %r14 // sdc
	sall	$ 6, %r14d // 4*sdc*sizeof(double)
	movq	ARG16, %r15 // n0
	movq	ARG17, %rax // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_GEN_LIB8
#else
	CALL(inner_scale_ab_8x8_gen_lib8)
#endif


	// store n

	movq	ARG11, %r10 // offsetD
	movq	ARG12, %r11 // D
	movq	ARG13, %r12 // sdd
	sall	$ 6, %r12d // 4*sdb*sizeof(double)
	movq	ARG14, %r13 // m0
	movq	ARG15, %r14 // m1
	movq	ARG16, %r15 // n0
	movq	ARG17, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_GEN_LIB8
#else
	CALL(inner_store_8x8_gen_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_tt_8x8_gen_lib8)





//#if defined(BLAS_API)
#if ( defined(BLAS_API) | ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) ) )

//#include "kernel_dgemm_8x8_lib.S"

#endif





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
.LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
	.align 6
LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#endif
	.double 0.5
	.double 1.5
	.double 2.5
	.double 3.5
	.double 4.5
	.double 5.5
	.double 6.5
	.double 7.5



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

