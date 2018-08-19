/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
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



#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif



// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- lda
// r13   <- B
// r14   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nt_12x4_lib, @function
inner_kernel_gemm_add_nt_12x4_lib:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nt_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_kernel_gemm_add_nt_12x4_lib; .scl 2; .type 32; .endef
inner_kernel_gemm_add_nt_12x4_lib:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload


	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	subl	$4, %r10d

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	subl	$4, %r10d

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11
	addq	%r14, %r13

	subl	$1, %r10d

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nt_12x4_lib, .-inner_kernel_gemm_add_nt_12x4_lib
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4X
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nt_12x4_lib4x, @function
inner_kernel_gemm_add_nt_12x4_lib4x:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nt_12x4_lib4x:
#elif defined(OS_WINDOWS)
	.def inner_kernel_gemm_add_nt_12x4_lib4x; .scl 2; .type 32; .endef
inner_kernel_gemm_add_nt_12x4_lib4x:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload


	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nt_12x4_lib4x, .-inner_kernel_gemm_add_nt_12x4_lib4x
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4X
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nt_12x4_lib4x, @function
inner_kernel_dgemm_sub_nt_12x4_lib4x:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nt_12x4_lib4x:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_sub_nt_12x4_lib4x; .scl 2; .type 32; .endef
inner_kernel_dgemm_sub_nt_12x4_lib4x:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	// preload


	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$4, %r10d
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$4, %r10d
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm0
	vfnmadd231pd	%ymm14, %ymm12, %ymm4
	vfnmadd231pd	%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm1
	vfnmadd231pd	%ymm14, %ymm12, %ymm5
	vfnmadd231pd	%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm2
	vfnmadd231pd	%ymm14, %ymm12, %ymm6
	vfnmadd231pd	%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r13), %ymm12 // B
	vfnmadd231pd	%ymm13, %ymm12, %ymm3
	vfnmadd231pd	%ymm14, %ymm12, %ymm7
	vfnmadd231pd	%ymm15, %ymm12, %ymm11
	addq	%r14, %r13

	subl	$1, %r10d
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nt_12x4_lib4x, .-inner_kernel_dgemm_sub_nt_12x4_lib4x
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- lda
// r13   <- B
// r14   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nn_12x4_lib, @function
inner_kernel_gemm_add_nn_12x4_lib:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nn_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_kernel_gemm_add_nn_12x4_lib; .scl 2; .type 32; .endef
inner_kernel_gemm_add_nn_12x4_lib:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r13, %r15
	addq	%r14, %r15
	addq	%r14, %r15 // B+2*ldb

	// preload

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	8(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	16(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	16(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	24(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	subl	$4, %r10d
	addq	$32, %r13
	addq	$32, %r15

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 1
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	8(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 2
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	16(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	16(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	// unroll 3
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	24(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	subl	$4, %r10d
	addq	$32, %r13
	addq	$32, %r15

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			32(%r11), %ymm14 // A
	vmovupd			64(%r11), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq	%r12, %r11

	subl	$1, %r10d
	addq	$8, %r13
	addq	$8, %r15

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nn_12x4_lib, .-inner_kernel_gemm_add_nn_12x4_lib
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- ldb
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_12X4_LIB4X
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nn_12x4_lib4x, @function
inner_kernel_gemm_add_nn_12x4_lib4x:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nn_12x4_lib4x:
#elif defined(OS_WINDOWS)
	.def inner_kernel_gemm_add_nn_12x4_lib4x; .scl 2; .type 32; .endef
inner_kernel_gemm_add_nn_12x4_lib4x:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	movq	%r13, %r15
	addq	%r14, %r15
	addq	%r14, %r15 // B+2*ldb

	// preload


	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

//	prefetcht0	0(%r12, %r13, 2) // software prefetch
//	prefetcht0	64(%r12, %r13, 2) // software prefetch

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	8(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	16(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	16(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	24(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl	$4, %r10d
	addq	$32, %r13
	addq	$32, %r15
	addq	$128, %r11

	cmpl	$4, %r10d
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 1
	vmovupd			32(%r11), %ymm13 // A
	vmovupd			32(%r11, %r12, 1), %ymm14 // A
	vmovupd			32(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	8(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	8(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	8(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 2
	vmovupd			64(%r11), %ymm13 // A
	vmovupd			64(%r11, %r12, 1), %ymm14 // A
	vmovupd			64(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	16(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	16(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	16(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	// unroll 3
	vmovupd			96(%r11), %ymm13 // A
	vmovupd			96(%r11, %r12, 1), %ymm14 // A
	vmovupd			96(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	24(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	24(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	24(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	24(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl	$4, %r10d
	addq	$32, %r13
	addq	$32, %r15
	addq	$128, %r11

	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovupd			0(%r11), %ymm13 // A
	vmovupd			0(%r11, %r12, 1), %ymm14 // A
	vmovupd			0(%r11, %r12, 2), %ymm15 // A
	vbroadcastsd	0(%r13), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	0(%r13, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	0(%r15), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	0(%r15, %r14, 1), %ymm12 // B
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl	$1, %r10d
	addq	$8, %r13
	addq	$8, %r15
	addq	$32, %r11

	cmpl	$0, %r10d
	jg		3b // clean up loop


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nn_12x4_lib4x, .-inner_kernel_gemm_add_nn_12x4_lib4x
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- lde
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_12x4_lib, @function
inner_edge_dtrsm_rlt_inv_12x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_12x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_12x4_lib:
#endif
#endif
	
	vbroadcastsd	0(%r12), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	addq	%r11, %r10

	vbroadcastsd	8(%r12), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9
	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	addq	%r11, %r10

	vbroadcastsd	16(%r12), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10
	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	addq	%r11, %r10

	vbroadcastsd	24(%r12), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_12x4_lib, .-inner_edge_dtrsm_rlt_inv_12x4_lib
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
// r13   <- ldc
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_12x4_lib, @function
inner_scale_ab_12x4_lib:
#elif defined(OS_MAC)
_inner_scale_ab_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_12x4_lib; .scl 2; .type 32; .endef
inner_scale_ab_12x4_lib:
#endif
#endif

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3
	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7
	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	vmovupd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	vmovupd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm8
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	vmovupd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	vmovupd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm9
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	vmovupd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	vmovupd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	addq		%r13, %r12
	vmovupd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3
	vmovupd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7
	vmovupd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm11
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_12x4_lib, .-inner_scale_ab_12x4_lib
#endif
#endif





// common inner routine with file scope
//
// scale for alpha=1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_11_12x4_lib, @function
inner_scale_11_12x4_lib:
#elif defined(OS_MAC)
_inner_scale_11_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_11_12x4_lib; .scl 2; .type 32; .endef
inner_scale_11_12x4_lib:
#endif
#endif

	vmovupd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovupd		32(%r10), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovupd		64(%r10), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovupd		32(%r10), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovupd		64(%r10), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovupd		32(%r10), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovupd		64(%r10), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
	vmovupd		32(%r10), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7
	vmovupd		64(%r10), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_11_12x4_lib, .-inner_scale_11_12x4_lib
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [840 d90 da0 db0]
// ymm9  <- [841 d91 da1 db1]
// ymm10 <- [842 d92 da2 db2]
// ymm11 <- [843 d93 da3 db3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_12x4_lib, @function
inner_store_12x4_lib:
#elif defined(OS_MAC)
_inner_store_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_12x4_lib; .scl 2; .type 32; .endef
inner_store_12x4_lib:
#endif
#endif

	vmovupd		%ymm0, 0(%r10)
	vmovupd		%ymm4, 32(%r10)
	vmovupd		%ymm8, 64(%r10)
	addq		%r11, %r10
	vmovupd		%ymm1, 0(%r10)
	vmovupd		%ymm5, 32(%r10)
	vmovupd		%ymm9, 64(%r10)
	addq		%r11, %r10
	vmovupd		%ymm2, 0(%r10)
	vmovupd		%ymm6, 32(%r10)
	vmovupd		%ymm10, 64(%r10)
	addq		%r11, %r10
	vmovupd		%ymm3, 0(%r10)
	vmovupd		%ymm7, 32(%r10)
	vmovupd		%ymm11, 64(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_12x4_lib, .-inner_store_12x4_lib
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_12X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_12x4_lib, @function
inner_store_l_12x4_lib:
#elif defined(OS_MAC)
_inner_store_l_12x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_l_12x4_lib; .scl 2; .type 32; .endef
inner_store_l_12x4_lib:
#endif
#endif

	vmovupd		%ymm0, 0(%r10)
	vmovupd		%ymm4, 32(%r10)
	vmovupd		%ymm8, 64(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x1, %ymm15, %ymm1, %ymm1
	vmovupd		%ymm1, 0(%r10)
	vmovupd		%ymm5, 32(%r10)
	vmovupd		%ymm9, 64(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x3, %ymm15, %ymm2, %ymm2
	vmovupd		%ymm2, 0(%r10)
	vmovupd		%ymm6, 32(%r10)
	vmovupd		%ymm10, 64(%r10)
	addq		%r11, %r10
	vmovupd		0(%r10), %ymm15
	vblendpd	$0x7, %ymm15, %ymm3, %ymm3
	vmovupd		%ymm3, 0(%r10)
	vmovupd		%ymm7, 32(%r10)
	vmovupd		%ymm11, 64(%r10)
//	addq	%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_12x4_lib, .-inner_store_l_12x4_lib
#endif
#endif





//                               1      2              3          4        5          6        7             8          9        10         11
// void kernel_dgemm_nt_12x4_lib(int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_lib
	.type kernel_dgemm_nt_12x4_lib, @function
kernel_dgemm_nt_12x4_lib:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_lib
_kernel_dgemm_nt_12x4_lib:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_12x4_lib
	.def kernel_dgemm_nt_12x4_lib; .scl 2; .type 32; .endef
kernel_dgemm_nt_12x4_lib:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // lda
	sall	$3, %r12d
	movq	ARG5, %r13  // B
	movq	ARG6, %r14  // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemm_add_nt_12x4_lib
#elif defined(OS_MAC)
	callq _inner_kernel_gemm_add_nt_12x4_lib
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C
	movq	ARG9, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib
#endif
#endif


	// store n

	movq	ARG10, %r10 // D
	movq	ARG11, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_12x4_lib, .-kernel_dgemm_nt_12x4_lib
#endif





//                                 1      2              3          4        5          6        7             8          9        10         11
// void kernel_dgemm_nt_12x4_lib4x(int k, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_lib4x
	.type kernel_dgemm_nt_12x4_lib4x, @function
kernel_dgemm_nt_12x4_lib4x:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_lib4x
_kernel_dgemm_nt_12x4_lib4x:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_12x4_lib4x
	.def kernel_dgemm_nt_12x4_lib4x; .scl 2; .type 32; .endef
kernel_dgemm_nt_12x4_lib4x:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13  // B
	movq	ARG6, %r14  // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4X
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemm_add_nt_12x4_lib4x
#elif defined(OS_MAC)
	callq _inner_kernel_gemm_add_nt_12x4_lib4x
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C
	movq	ARG9, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib
#endif
#endif


	// store n

	movq	ARG10, %r10 // D
	movq	ARG11, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_12x4_lib4x, .-kernel_dgemm_nt_12x4_lib4x
#endif





//                                  1      2              3          4        5          6             7          8        9          10
// void kernel_dgemm_nt_12x4_lib44x(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_lib44x
	.type kernel_dgemm_nt_12x4_lib44x, @function
kernel_dgemm_nt_12x4_lib44x:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_lib44x
_kernel_dgemm_nt_12x4_lib44x:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_12x4_lib44x
	.def kernel_dgemm_nt_12x4_lib44x; .scl 2; .type 32; .endef
kernel_dgemm_nt_12x4_lib44x:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_12x4_lib44x, .-kernel_dgemm_nt_12x4_lib44x
#endif





//                               1      2              3          4        5          6        7             8          9        10         11
// void kernel_dgemm_nn_12x4_lib(int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_12x4_lib
	.type kernel_dgemm_nn_12x4_lib, @function
kernel_dgemm_nn_12x4_lib:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_12x4_lib
_kernel_dgemm_nn_12x4_lib:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_12x4_lib
	.def kernel_dgemm_nn_12x4_lib; .scl 2; .type 32; .endef
kernel_dgemm_nn_12x4_lib:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // lda
	sall	$3, %r12d
	movq	ARG5, %r13  // B
	movq	ARG6, %r14  // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemm_add_nn_12x4_lib
#elif defined(OS_MAC)
	callq _inner_kernel_gemm_add_nn_12x4_lib
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C
	movq	ARG9, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib
#endif
#endif


	// store n

	movq	ARG10, %r10 // D
	movq	ARG11, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_12x4_lib, .-kernel_dgemm_nn_12x4_lib
#endif





//                                 1      2              3          4        5          6        7             8          9        10         11
// void kernel_dgemm_nn_12x4_lib4x(int k, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_12x4_lib4x
	.type kernel_dgemm_nn_12x4_lib4x, @function
kernel_dgemm_nn_12x4_lib4x:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_12x4_lib4x
_kernel_dgemm_nn_12x4_lib4x:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_12x4_lib4x
	.def kernel_dgemm_nn_12x4_lib4x; .scl 2; .type 32; .endef
kernel_dgemm_nn_12x4_lib4x:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13  // B
	movq	ARG6, %r14  // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_12X4_LIB4X
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_gemm_add_nn_12x4_lib4x
#elif defined(OS_MAC)
	callq _inner_kernel_gemm_add_nn_12x4_lib4x
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12   // C
	movq	ARG9, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib
#endif
#endif


	// store n

	movq	ARG10, %r10 // D
	movq	ARG11, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_12x4_lib4x, .-kernel_dgemm_nn_12x4_lib4x
#endif





//                                     1      2          3        4          5        6          7        8          9        10
// void kernel_dpotrf_nt_l_12x4_lib4cc(int k, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_12x4_lib4cc
	.type kernel_dpotrf_nt_l_12x4_lib4cc, @function
kernel_dpotrf_nt_l_12x4_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_12x4_lib4cc
_kernel_dpotrf_nt_l_12x4_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dpotrf_nt_l_12x4_lib4cc
	.def kernel_dpotrf_nt_l_12x4_lib4cc; .scl 2; .type 32; .endef
kernel_dpotrf_nt_l_12x4_lib4cc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // kmax
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // ldb
	sall	$5, %r12d
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4X
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_12x4_lib4x
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4x
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib
#endif
#endif


	// factorization

	movq	ARG10, %r10  // inv_diag_D 
	movl	$4, %r11d

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_12X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_12x4_vs_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_12x4_lib4cc, .-kernel_dpotrf_nt_l_12x4_lib4cc
#endif





//                                       1      2          3        4          5        6          7        8          9        10         11       12
// void kernel_dtrsm_nt_rl_inv_12x4_lib4(int k, double *A, int sda, double *B, int ldb, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_12x4_lib4cc
	.type kernel_dtrsm_nt_rl_inv_12x4_lib4cc, @function
kernel_dtrsm_nt_rl_inv_12x4_lib4cc:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_12x4_lib4cc
_kernel_dtrsm_nt_rl_inv_12x4_lib4cc:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_12x4_lib4cc
	.def kernel_dtrsm_nt_rl_inv_12x4_lib4cc; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_12x4_lib4cc:
#endif
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d
	movq	ARG4, %r13
	movq	ARG5, %r14 // ldb
	sall	$3, %r14d

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4X
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_12x4_lib4x
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4x
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_12x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // lde
	sall	$3, %r11d
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_12x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_12x4_lib
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_12x4_lib
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_12x4_lib4cc, .-kernel_dtrsm_nt_rl_inv_12x4_lib4cc
#endif






