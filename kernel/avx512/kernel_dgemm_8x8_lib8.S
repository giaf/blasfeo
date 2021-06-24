/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
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
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// zmm0  <- []
// ...
// zmm7  <- []

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*k*sizeof(double)
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
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
// r13d  <- m1
// r14d  <- n1
// zmm0 <- [d00 d11 d22 d33]
// ...
// ymm7 <- [d02 d13 d20 d31]
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13d  <- m1
// r14d  <- n1
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
	cmpl	$ 2, %r14d
	jl		0f // end
	vmovapd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	cmpl	$ 3, %r14d
	jl		0f // end
	vmovapd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	cmpl	$ 4, %r14d
	jl		0f // end
	vmovapd		192(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	cmpl	$ 5, %r14d
	jl		0f // end
	vmovapd		0+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	cmpl	$ 6, %r14d
	jl		0f // end
	vmovapd		64+256(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	cmpl	$ 7, %r14d
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
	
	movl	$ 1, %r13d // shift base
	movq	%rcx, %r14 // backup
	movl	$ 8, %ecx // saturate to ps=8
	cmpl	%r11d, %ecx
	cmovgl	%r11d, %ecx
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
	movq	ARG8, %r13 // km 
	movq	ARG9, %r14 // kn 

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

#if defined(OS_LINUX) | defined(OS_WINDOWS)
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

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC02: // { 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
LC02: // { 3.5 2.5 1.5 0.5 }
	.align 5
#endif
	.double 0.5
	.double 1.5
	.double 2.5
	.double 3.5

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
#elif defined(OS_MAC)
LC03: // { 7.5 6.5 5.5 4.5 }
	.align 5
#endif
	.double 4.5
	.double 5.5
	.double 6.5
	.double 7.5

#if defined(OS_LINUX) | defined(OS_WINDOWS)
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

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC05: // { 1.0 1.0 1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC05: // { 1.0 1.0 1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC06: // { 1.0 1.0 -1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC06: // { 1.0 1.0 -1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC07: // { 1.0 -1.0 -1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC07: // { 1.0 -1.0 -1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC08: // { -1.0 -1.0 -1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC08: // { -1.0 -1.0 -1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC09: // { -1.0 -1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC09: // { -1.0 -1.0 1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC10: // { -1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC10: // { -1.0 1.0 1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC11: // { -1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC11: // { -1.0 1.0 1.0 1.0 }
#endif
	.long	0x0
	.long	0x80000000
	.long	0x0
	.long	0x80000000
	.long	0x0
	.long	0x80000000
	.long	0x0
	.long	0x80000000
	.long	0x0
	.long	0x80000000



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

