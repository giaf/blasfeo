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
#define ARG2L  %esi
#define ARG6L  %r9d
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

#else // defined(OS_MAC)

#define GLOB_FUN_START(NAME) \
	.globl _ ## NAME; \
_ ## NAME:
#define FUN_START(NAME) \
_ ## NAME:
#define FUN_END(NAME)
#define CALL(NAME) \
	callq _ ## NAME

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
#define ARG2L  %edx
#define ARG6L  STACKSIZE + 48(%rsp)
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

#else

#error wrong OS

#endif



#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif





#if MACRO_LEVEL>=1
	.macro INNER_KERNEL_DPACP_TN_8_LIB8
#else
	.p2align 4,,15
	FUN_START(inner_kernel_dpacp_tn_8_lib8)
#endif

	cmpl	$ 0, %r10d
	jle		0f // return

	cmpl	$ 0, %r11d
	jle		3f // offset==0

	movl			$ 8, %r15d				// load 8
	subl			%r11d, %r15d			// 8-offsetA
	cmpl			%r10d, %r15d			// k > 8-offsetA
	cmovgl			%r10d, %r15d			// kend=min(k,8-offsetA)

	movl			%r11d, %eax				// load offsetA
	sall			$ 3, %eax				// offsetA*sizeof(double)
	addq			%rax, %r12				// B+offsetB*sizeof(double)

// TODO vector and mask instead ?

4:
	vmovsd		0*64(%r12), %xmm0
	vmovsd		%xmm0, 0(%r14)
	vmovsd		1*64(%r12), %xmm0
	vmovsd		%xmm0, 8(%r14)
	vmovsd		2*64(%r12), %xmm0
	vmovsd		%xmm0, 16(%r14)
	vmovsd		3*64(%r12), %xmm0
	vmovsd		%xmm0, 24(%r14)
	vmovsd		4*64(%r12), %xmm0
	vmovsd		%xmm0, 32(%r14)
	vmovsd		5*64(%r12), %xmm0
	vmovsd		%xmm0, 40(%r14)
	vmovsd		6*64(%r12), %xmm0
	vmovsd		%xmm0, 48(%r14)
	vmovsd		7*64(%r12), %xmm0
	vmovsd		%xmm0, 56(%r14)

	subl			$ 1, %r10d				// k=-1
	subl			$ 1, %r15d				// k_panel=-1
	addq			$ 8, %r12				// A=+1
	addq			$ 64, %r14				// C=+bs

	cmpl			$ 0, %r15d				// if k_panel=0
	jg				4b						// loop 1

	cmpl			$ 0, %r10d				// if k=0
	jle				0f						// end

	addq			%r13, %r12				// A=Boff+sdb*bs
	subq			$ 64, %r12				// A-=4*sizeof(double) (loop+offsetA)


3:
	cmpl	$ 7, %r10d
	jle		2f // clean-up loop

	movl	$ 0xcc, %r15d
	kmovd	%r15d, %k1
	movl	$ 0x33, %r15d
	kmovd	%r15d, %k2

	// main loop
	.p2align 3
1: // main loop
	
//	prefetcht0	0(%r12, %r13, 1) // software prefetch
//	prefetcht0	64(%r12, %r13, 1) // software prefetch

	vmovapd			0+0*64(%r12), %ymm0
	vmovapd			0+4*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm0, %zmm0
	vmovapd			0+1*64(%r12), %ymm1
	vmovapd			0+5*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm1, %zmm1
	vmovapd			0+2*64(%r12), %ymm2
	vmovapd			0+6*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm2, %zmm2
	vmovapd			0+3*64(%r12), %ymm3
	vmovapd			0+7*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm3, %zmm3
	vmovapd			32+0*64(%r12), %ymm4
	vmovapd			32+4*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm4, %zmm4
	vmovapd			32+1*64(%r12), %ymm5
	vmovapd			32+5*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm5, %zmm5
	vmovapd			32+2*64(%r12), %ymm6
	vmovapd			32+6*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm6, %zmm6
	vmovapd			32+3*64(%r12), %ymm7
	vmovapd			32+7*64(%r12), %ymm24
	vinsertf64x4	$ 0x1, %ymm24, %zmm7, %zmm7

	vmovapd		%zmm0, %zmm24
	vpermq		$ 0x40, %zmm2, %zmm0 {%k1}
	vpermq		$ 0x0e, %zmm24, %zmm2 {%k2}
	vmovapd		%zmm1, %zmm24
	vpermq		$ 0x40, %zmm3, %zmm1 {%k1}
	vpermq		$ 0x0e, %zmm24, %zmm3 {%k2}
	vmovapd		%zmm4, %zmm24
	vpermq		$ 0x40, %zmm6, %zmm4 {%k1}
	vpermq		$ 0x0e, %zmm24, %zmm6 {%k2}
	vmovapd		%zmm5, %zmm24
	vpermq		$ 0x40, %zmm7, %zmm5 {%k1}
	vpermq		$ 0x0e, %zmm24, %zmm7 {%k2}

	vunpcklpd	%zmm1, %zmm0, %zmm24
	vmovapd		%zmm24, 0*64(%r14)
	vunpckhpd	%zmm1, %zmm0, %zmm24
	vmovapd		%zmm24, 1*64(%r14)
	vunpcklpd	%zmm3, %zmm2, %zmm24
	vmovapd		%zmm24, 2*64(%r14)
	vunpckhpd	%zmm3, %zmm2, %zmm24
	vmovapd		%zmm24, 3*64(%r14)
	vunpcklpd	%zmm5, %zmm4, %zmm24
	vmovapd		%zmm24, 4*64(%r14)
	vunpckhpd	%zmm5, %zmm4, %zmm24
	vmovapd		%zmm24, 5*64(%r14)
	vunpcklpd	%zmm7, %zmm6, %zmm24
	vmovapd		%zmm24, 6*64(%r14)
	vunpckhpd	%zmm7, %zmm6, %zmm24
	vmovapd		%zmm24, 7*64(%r14)

	subl	$ 8, %r10d
	addq	%r13, %r12
	addq	$ 512, %r14

	cmpl	$ 7, %r10d
	jg		1b


	cmpl	$ 0, %r10d
	jle		0f // return


// TODO vector and mask instead ?

2: // clean-up loop
	
	vmovsd		0*64(%r12), %xmm0
	vmovsd		%xmm0, 0(%r14)
	vmovsd		1*64(%r12), %xmm0
	vmovsd		%xmm0, 8(%r14)
	vmovsd		2*64(%r12), %xmm0
	vmovsd		%xmm0, 16(%r14)
	vmovsd		3*64(%r12), %xmm0
	vmovsd		%xmm0, 24(%r14)
	vmovsd		4*64(%r12), %xmm0
	vmovsd		%xmm0, 32(%r14)
	vmovsd		5*64(%r12), %xmm0
	vmovsd		%xmm0, 40(%r14)
	vmovsd		6*64(%r12), %xmm0
	vmovsd		%xmm0, 48(%r14)
	vmovsd		7*64(%r12), %xmm0
	vmovsd		%xmm0, 56(%r14)

	subl	$ 1, %r10d
	addq	$ 8, %r12
	addq	$ 64, %r14

	cmpl	$ 0, %r10d
	jg		2b


0: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_kernel_dpacp_tn_8_lib8)
#endif





//                             1         2            3          4        5
// void kernel_dpacp_tn_8_lib8(int kmax, int offsetA, double *A, int sda, double *C)

	.p2align 4,,15
	GLOB_FUN_START(kernel_dpacp_tn_8_lib8)

	PROLOGUE


	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // offsetA
	movq	ARG3, %r12 // A
	movq	ARG4, %r13 // sda
	sall	$ 6, %r13d
	movq	ARG5, %r14 // C

#if MACRO_LEVEL>=1
	INNER_KERNEL_DPACP_TN_8_LIB8
#else
	CALL(inner_kernel_dpacp_tn_8_lib8)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dpacp_tn_8_lib8)





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#elif defined(OS_WINDOWS)
	.section .rdata,"dr"
#endif


#if defined(OS_LINUX) | defined(OS_WINDOWS)
//	.align 32
//.LC00:
#elif defined(OS_MAC)
//	.align 5
//LC00:
#endif
//	.double 0.5
//	.double 1.5
//	.double 2.5
//	.double 3.5


#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif


