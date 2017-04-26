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
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_8x4_lib4, @function
inner_kernel_dgemm_add_nt_8x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_add_nt_8x4_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemm_add_nt_8x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

//	movq	%r11, %r15 // A1 <- A0
//	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
//	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmovapd 0(%r13), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 32(%r15), %ymm11 // A1[4]
	vmovapd 32(%r11, %r12, 1), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 64(%r15), %ymm9 // A1[8]
	vmovapd 64(%r11, %r12, 1), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 96(%r15), %ymm11 // A1[12]
	vmovapd 96(%r11, %r12, 1), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	addq	$128, %r11
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


	// unroll 3
	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	cmpl	$4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 32(%r15), %ymm11 // A1[4]
	vmovapd 32(%r11, %r12, 1), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 64(%r15), %ymm9 // A1[8]
	vmovapd 64(%r11, %r12, 1), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 96(%r15), %ymm11 // A1[12]
	vmovapd 96(%r11, %r12, 1), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
//	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


	// unroll 3
//	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

//	vmovapd 0(%r15), %ymm9 // A1[0]
//	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


//	cmpl	$3, %r10d
	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 0(%r13), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
//	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4
	addq	$32, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	addq	$32, %r13
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
//	addq	$32, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	subl	$1, %r10d
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	cmpl	$0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_8x4_lib4, .-inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nt_8x4_lib4, @function
inner_kernel_dgemm_sub_nt_8x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nt_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_sub_nt_8x4_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemm_sub_nt_8x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmovapd 0(%r13), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 32(%r11, %r12, 1), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 64(%r11, %r12, 1), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 96(%r11, %r12, 1), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	// unroll 3
	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5
	cmpl	$4, %r10d

	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 32(%r11, %r12, 1), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 64(%r11, %r12, 1), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 96(%r11, %r12, 1), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	// unroll 3
//	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5
//	cmpl	$3, %r10d

//	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 0(%r13), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r11, %r12, 1), %ymm9 // A1[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4
	addq	$32, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	addq	$32, %r13
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	subl	$1, %r10d
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nt_8x4_lib4, .-inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
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
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
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
// r10d  <- k
// r11   <- A+4*sda*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+(k/4)*sdb*sizeof(double)+(k%4)
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NN_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nn_8x4_lib4, @function
inner_kernel_dgemm_add_nn_8x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nn_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_add_nn_8x4_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemm_add_nn_8x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r13, %rax // B_next <- B
	addq	%r14, %rax // B_next <- B + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%rax) // software prefetch
	prefetcht0	64(%rax) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A0[0]
	vmovapd			32(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	8(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	72(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	104(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A0[0]
	vmovapd			64(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	16(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	112(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A0[0]
	vmovapd			96(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	24(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	88(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	120(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	addq	$128, %r11

	mov		%rax, %r13
	addq	%r14, %rax

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl	$1, %r10d

	addq	$32, %r11
	addq	$8, %r13

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nn_8x4_lib4, .-inner_kernel_dgemm_add_nn_8x4_lib4
#endif
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
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
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
// r10d  <- k
// r11   <- A+4*sda*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+(k/4)*sdb*sizeof(double)+(k%4)
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nn_8x4_lib4, @function
inner_kernel_dgemm_sub_nn_8x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nn_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_sub_nn_8x4_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemm_sub_nn_8x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r13, %rax // B_next <- B
	addq	%r14, %rax // B_next <- B + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%rax) // software prefetch
	prefetcht0	64(%rax) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A0[0]
	vmovapd			32(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	8(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	72(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	104(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A0[0]
	vmovapd			64(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	16(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	112(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A0[0]
	vmovapd			96(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	24(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	88(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	120(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	addq	$128, %r11

	mov		%rax, %r13
	addq	%r14, %rax

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r11, %r12, 1), %ymm14 // A1[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm14, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	subl	$1, %r10d

	addq	$32, %r11
	addq	$8, %r13

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nn_8x4_lib4, .-inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B unaligned
//
// input arguments:
// r10   <- k
// r11   <- A
// r12   <- bs*sda*sizeof(double)
// r13   <- B
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- k-(4-offB)
// r11   <- A+(4-offB)*bs*sizeof(double)
// r12   <- bs*sda*sizeof(double)
// r13   <- B-offB+bs*sdb*sizeof(double)
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGEMM_ADD_NN_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgemm_add_nn_8x4_lib4, @function
inner_edge_dgemm_add_nn_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgemm_add_nn_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dgemm_add_nn_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dgemm_add_nn_8x4_lib4:
#endif
#endif
	
	cmpl			$0, %r15d // offset==0
	jle				2f // end

	cmpl			$0, %r10d // k==0
	jle				2f // end

	movl			$4, %ebx
	subl			%r15d, %ebx // 4-offsetB
	cmpl			%r10d, %ebx
//	jle				0f
//	movl			%r10d, %ebx // kend=min(k,4-offsetB)
//0:
	cmovgl			%r10d, %ebx // kend=min(k,4-offsetB)

	movl			%r15d, %eax
	sall			$3, %eax // offsetB*sizeof(double)
	addq			%rax, %r13 // B+offsetB*sizeof(double)

1:
	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	subl			$1, %ebx // kend-1
	addq			$32, %r11 // A0+1*bs*sizeof(float)
	addq			$8, %r13 // B+1*sizeof(float)

	cmpl			$0, %ebx
	jg				1b

	cmpl			$0, %r10d
	jle				2f // end

	addq			%r14, %r13
	subq			$32, %r13 // B+bs*(sdb-1)*sizeof(double)

2:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgemm_add_nn_8x4_lib4, .-inner_edge_dgemm_add_nn_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- 4*sda*sizeof(double)
// r12   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- 4*sda*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_8x4_lib4, @function
inner_edge_dtrmm_nt_ru_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nt_ru_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrmm_nt_ru_8x4_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r12), %ymm12
	vmovapd			0(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r10, %r11, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

	vbroadcastsd	32(%r12), %ymm12
	vmovapd			32(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			32(%r10, %r11, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	40(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

	vbroadcastsd	64(%r12), %ymm12
	vmovapd			64(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			64(%r10, %r11, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	72(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	80(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

	vbroadcastsd	96(%r12), %ymm12
	vmovapd			96(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			96(%r10, %r11, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	104(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	112(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	120(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

	addq			$128, %r10
	addq			$128, %r12

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_8x4_lib4, .-inner_edge_dtrmm_nt_ru_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_8x4_vs_lib4, @function
inner_edge_dtrmm_nt_ru_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nt_ru_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dtrmm_nt_ru_8x4_vs_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	vmovapd			0(%r11, %r12, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r13

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r11, %r12, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r11
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r11, %r12, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r11
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r11, %r12, 1), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r11
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	vbroadcastsd	24(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_8x4_vs_lib4, .-inner_edge_dtrmm_nt_ru_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B lower triangular
//
// input arguments:
// r10   <- k
// r11   <- A0
// r12   <- bs*sda*sizeof(double)
// r13   <- B
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- k-(4-offB)
// r11   <- A+(4-offB)*bs*sizeof(double)
// r12   <- bs*sda*sizeof(double)
// r13   <- B-offB+bs*sdb*sizeof(double)
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nn_rl_8x4_lib4, @function
inner_edge_dtrmm_nn_rl_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nn_rl_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nn_rl_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrmm_nn_rl_8x4_lib4:
#endif
#endif
	
	cmpl	$0, %r15d
	jg		0f

	// offB==0

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	vmovapd			32(%r11), %ymm8
	vmovapd			32(%r11, %r12, 1), %ymm9
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	vmovapd			64(%r11), %ymm8
	vmovapd			64(%r11, %r12, 1), %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	vmovapd			96(%r11), %ymm8
	vmovapd			96(%r11, %r12, 1), %ymm9
	vbroadcastsd	24(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	88(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	120(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

	jmp		3f

0:
	cmpl	$1, %r15d
	jg		1f

	// offB==1

	addq			$8, %r13 // B+1*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	vmovapd			32(%r11), %ymm8
	vmovapd			32(%r11, %r12, 1), %ymm9
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	vmovapd			64(%r11), %ymm8
	vmovapd			64(%r11, %r12, 1), %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	subl			$3, %r10d // k-3
	addq			$96, %r11 // A0+3*bs*sizeof(double)
	addq			%r14, %r13
	subq			$8, %r13 // B+bs*sdb*sizeof(double)-1

	jmp		3f

1:
	cmpl	$2, %r15d
	jg		2f

	// offB==2

	addq			$16, %r13 // B+2*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	vmovapd			32(%r11), %ymm8
	vmovapd			32(%r11, %r12, 1), %ymm9
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	subl			$2, %r10d // k-2
	addq			$64, %r11 // A0+2*bs*sizeof(double)
	addq			%r14, %r13
	subq			$16, %r13 // B+bs*sdb*sizeof(double)-2

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	vmovapd			32(%r11), %ymm8
	vmovapd			32(%r11, %r12, 1), %ymm9
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	72(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	104(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	vmovapd			64(%r11), %ymm8
	vmovapd			64(%r11, %r12, 1), %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	112(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	vmovapd			96(%r11), %ymm8
	vmovapd			96(%r11, %r12, 1), %ymm9
	vbroadcastsd	24(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	88(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	120(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

	jmp		3f

2:
	// offB==3

	addq			$24, %r13 // B+3*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-3

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	vmovapd			32(%r11), %ymm8
	vmovapd			32(%r11, %r12, 1), %ymm9
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	72(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	vmovapd			64(%r11), %ymm8
	vmovapd			64(%r11, %r12, 1), %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	80(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	112(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	vmovapd			96(%r11), %ymm8
	vmovapd			96(%r11, %r12, 1), %ymm9
	vbroadcastsd	24(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	88(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	120(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nn_rl_8x4_lib4, .-inner_edge_dtrmm_nn_rl_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B lower triangular
//
// input arguments:
// r10   <- k
// r11   <- A0
// r12   <- bs*sda*sizeof(double)
// r13   <- B
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- k-(4-offB)
// r11   <- A+(4-offB)*bs*sizeof(double)
// r12   <- bs*sda*sizeof(double)
// r13   <- B-offB+bs*sdb*sizeof(double)
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_8X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nn_rl_8x4_gen_lib4, @function
inner_edge_dtrmm_nn_rl_8x4_gen_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nn_rl_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrmm_nn_rl_8x4_gen_lib4; .scl 2; .type 32; .endef
inner_edge_dtrmm_nn_rl_8x4_gen_lib4:
#endif
#endif
	
	cmpl			$0, %r10d
	jle				3f // end

	cmpl			$0, %r15d
	jg				0f // offB>0

	// offB==0

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

	jmp				3f // end

0:
	cmpl			$1, %r15d
	jg				1f // offB>1

	// offB==1

	addq			$8, %r13 // B+1*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

	jmp				3f // end

1:
	cmpl			$2, %r15d
	jg				2f // offB>2

	// offB==2

	addq			$16, %r13 // B+2*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	subl			$1, %r10d // k-2
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

	jmp				3f

2:
	// offB==3

	addq			$24, %r13 // B+3*sizeof(double)

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$8, %r13 // B+1*sizeof(double)

	cmpl			$0, %r10d
	jle				3f // end

	vmovapd			0(%r11), %ymm8
	vmovapd			0(%r11, %r12, 1), %ymm9
	vbroadcastsd	0(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	32(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	96(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm15, %ymm7, %ymm7

	subl			$1, %r10d // k-4
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-(bs-1)*sizeof(double)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nn_rl_8x4_gen_lib4, .-inner_edge_dtrmm_nn_rl_8x4_gen_lib4
#endif
#endif





// common inner routine with file scope
//
// blend
//
// input arguments:
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_8x4_lib4, @function
inner_blend_8x4_lib4:
#elif defined(OS_MAC)
_inner_blend_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_8x4_lib4; .scl 2; .type 32; .endef
inner_blend_8x4_lib4:
#endif
#endif
	

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_8x4_lib4, .-inner_blend_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_11_8x4_lib4, @function
inner_scale_11_8x4_lib4:
#elif defined(OS_MAC)
_inner_scale_11_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_scale_11_8x4_lib4; .scl 2; .type 32; .endef
inner_scale_11_8x4_lib4:
#endif
#endif
	

	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		0(%r10, %r11, 1), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r10, %r11, 1), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r10, %r11, 1), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r10, %r11, 1), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_11_8x4_lib4, .-inner_scale_11_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta=0
//
// input arguments:
// r10   <- alpha
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_A0_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_a0_8x4_lib4, @function
inner_scale_a0_8x4_lib4:
#elif defined(OS_MAC)
_inner_scale_a0_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_scale_a0_8x4_lib4; .scl 2; .type 32; .endef
inner_scale_a0_8x4_lib4:
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

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_a0_8x4_lib4, .-inner_scale_a0_8x4_lib4
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
// r13   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r13   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_8x4_lib4, @function
inner_scale_ab_8x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_8x4_lib4; .scl 2; .type 32; .endef
inner_scale_ab_8x4_lib4:
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

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	// alg==1
	vmovapd		0(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		0(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_8x4_lib4, .-inner_scale_ab_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- offset
// r13   <- C
// r14   <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- offset
// r13   <- C
// r14   <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_8x4_gen_lib4, @function
inner_scale_ab_8x4_gen_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_8x4_gen_lib4; .scl 2; .type 32; .endef
inner_scale_ab_8x4_gen_lib4:
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

	// beta
	vbroadcastsd	0(%r11), %ymm15

	vxorpd		%ymm14, %ymm14, %ymm14 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			3f // end

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovapd		0(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm0, %ymm14, %ymm0
	vmovapd		32(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm1, %ymm14, %ymm1
	vmovapd		64(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm2, %ymm14, %ymm2
	vmovapd		96(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm3, %ymm14, %ymm3

	vmovapd		0(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm4, %ymm14, %ymm4
	vmovapd		32(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm5, %ymm14, %ymm5
	vmovapd		64(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm6, %ymm14, %ymm6
	vmovapd		96(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm7, %ymm14, %ymm7

	jmp		3f

0:

	cmpl	$1, %r12d
	jg		1f

	// offset==1

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7

	jmp		3f

1:

	cmpl	$2, %r12d
	jg		2f

	// offset==2

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7


	jmp		3f

2:

	// offset==3

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7


3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_8x4_gen_lib4, .-inner_scale_ab_8x4_gen_lib4
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
// r13   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r13   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_8x4_lib4, @function
inner_blend_scale_ab_8x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_ab_8x4_lib4; .scl 2; .type 32; .endef
inner_blend_scale_ab_8x4_lib4:
#endif
#endif
	

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	// alg==1
	vmovapd		0(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		0(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r12, %r13, 1), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_8x4_lib4, .-inner_blend_scale_ab_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- offset
// r13   <- C
// r14   <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- offset
// r13   <- C
// r14   <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_8X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_8x4_gen_lib4, @function
inner_blend_scale_ab_8x4_gen_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_ab_8x4_gen_lib4; .scl 2; .type 32; .endef
inner_blend_scale_ab_8x4_gen_lib4:
#endif
#endif
	

	// alpha
	vbroadcastsd	0(%r10), %ymm15

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	// beta
	vbroadcastsd	0(%r11), %ymm15

	vxorpd		%ymm14, %ymm14, %ymm14 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			3f // end

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovapd		0(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm0, %ymm14, %ymm0
	vmovapd		32(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm1, %ymm14, %ymm1
	vmovapd		64(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm2, %ymm14, %ymm2
	vmovapd		96(%r13), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm3, %ymm14, %ymm3

	vmovapd		0(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm4, %ymm14, %ymm4
	vmovapd		32(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm5, %ymm14, %ymm5
	vmovapd		64(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm6, %ymm14, %ymm6
	vmovapd		96(%r13, %r14, 1), %ymm14
	vmulpd		%ymm14, %ymm15, %ymm14
	vaddpd		%ymm7, %ymm14, %ymm7

	jmp		3f

0:

	cmpl	$1, %r12d
	jg		1f

	// offset==1

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x1, %ymm13, %ymm12, %ymm12
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7

	jmp		3f

1:

	cmpl	$2, %r12d
	jg		2f

	// offset==2

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x3, %ymm13, %ymm12, %ymm12
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7


	jmp		3f

2:

	// offset==3

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r13, %r14, 1), %ymm13
	vmovapd		0(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm4, %ymm13, %ymm4

	vmovapd		32(%r13), %ymm12
	vmovapd		32(%r13, %r14, 1), %ymm13
	vmovapd		32(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm5, %ymm13, %ymm5

	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r13, %r14, 1), %ymm13
	vmovapd		64(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm6, %ymm13, %ymm6

	vmovapd		96(%r13), %ymm12
	vmovapd		96(%r13, %r14, 1), %ymm13
	vmovapd		96(%r13, %r14, 2), %ymm14
	vblendpd	$0x7, %ymm13, %ymm12, %ymm12
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm7, %ymm13, %ymm7


3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_8x4_gen_lib4, .-inner_blend_scale_ab_8x4_gen_lib4
#endif
#endif





// common inner routine with file scope
//
// blender for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_8x4_lib4, @function
inner_blend_scale_11_8x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_11_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_11_8x4_lib4; .scl 2; .type 32; .endef
inner_blend_scale_11_8x4_lib4:
#endif
#endif
	

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	// alg==1
	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		0(%r10, %r11, 1), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r10, %r11, 1), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r10, %r11, 1), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r10, %r11, 1), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_11_8x4_lib4, .-inner_blend_scale_11_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization
//
// input arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_8x4_lib4, @function
inner_edge_dpotrf_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dpotrf_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dpotrf_8x4_lib4:
#endif
#endif
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovsd	.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd	LC04(%rip), %xmm14 // 1.0
#endif

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			1f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
2:
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0
	vmulpd		%ymm4, %ymm13, %ymm4
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm11
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm1, %ymm1
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm5, %ymm5
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm11
	vpermilpd	$0x0, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7


	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			3f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
4:
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1
	vmulpd		%ymm5, %ymm13, %ymm5
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm11
	vpermilpd	$0x0, %ymm11, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7


	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			5f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
6:
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2
	vmulpd		%ymm6, %ymm13, %ymm6
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm11
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm6, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7

	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			7f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
8:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3
	vmulpd		%ymm7, %ymm13, %ymm7

	jmp				0f

1:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		2b

3:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		4b

5:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		6b

7:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		8b

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_8x4_lib4, .-inner_edge_dpotrf_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization vs
//
// input arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_8x4_vs_lib4, @function
inner_edge_dpotrf_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dpotrf_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dpotrf_8x4_vs_lib4:
#endif
#endif
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovsd	.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd	LC04(%rip), %xmm14 // 1.0
#endif

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			1f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
2:
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0
	vmulpd		%ymm4, %ymm13, %ymm4
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm11
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm1, %ymm1
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm5, %ymm5
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm11
	vpermilpd	$0x0, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7

	cmpl		$2, %r11d
	jl			0f // ret

	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			3f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
4:
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1
	vmulpd		%ymm5, %ymm13, %ymm5
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm11
	vpermilpd	$0x0, %ymm11, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7

	cmpl		$3, %r11d
	jl			0f // ret

	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			5f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
6:
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2
	vmulpd		%ymm6, %ymm13, %ymm6
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm11
	vpermilpd	$0xf, %ymm11, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm6, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7

	cmpl		$4, %r11d
	jl			0f // ret

	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			7f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
8:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3
	vmulpd		%ymm7, %ymm13, %ymm7

	jmp				0f

1:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		2b

3:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		4b

5:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		6b

7:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		8b

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_8x4_vs_lib4, .-inner_edge_dpotrf_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_8x4_lib4, @function
inner_edge_dtrsm_rlt_inv_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_8x4_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_8x4_lib4, .-inner_edge_dtrsm_rlt_inv_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_8x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_8x4_vs_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5
	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	cmpl			$2, %r12d
	jl				0f // ret

	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	cmpl			$3, %r12d
	jl				0f // ret

	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

	cmpl			$4, %r12d
	jl				0f // ret

	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_8x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_8x4_lib4, @function
inner_edge_dtrsm_rlt_one_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_one_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_one_8x4_lib4:
#endif
#endif
	
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_8x4_lib4, .-inner_edge_dtrsm_rlt_one_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- D
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- D
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_8x4_vs_lib4, @function
inner_edge_dtrsm_rlt_one_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_one_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_one_8x4_vs_lib4:
#endif
#endif
	
	cmpl			$2, %r11d

	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5

	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6

	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_8x4_vs_lib4, .-inner_edge_dtrsm_rlt_one_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_8x4_lib4, @function
inner_edge_dtrsm_rut_inv_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_inv_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_inv_8x4_lib4:
#endif
#endif
	
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm6, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm6, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm5, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_8x4_lib4, .-inner_edge_dtrsm_rut_inv_8x4_lib4
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
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_8x4_vs_lib4, @function
inner_edge_dtrsm_rut_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rut_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_rut_inv_8x4_vs_lib4:
#endif
#endif
	
	cmpl			$3, %r12d
	jle				0f

	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm7, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

0:
	cmpl			$2, %r12d
	jle				1f

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm6, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm6, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

1:
	cmpl			$1, %r12d
	jle				2f

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmulpd			%ymm5, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4

2:

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_8x4_vs_lib4, .-inner_edge_dtrsm_rut_inv_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = up
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_INV_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_inv_8x4_lib4, @function
inner_edge_dtrsm_run_inv_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_run_inv_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_run_inv_8x4_lib4:
#endif
#endif

	// first column
	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4

	// second column
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5

	// third column
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6

	// fourth column
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_inv_8x4_lib4, .-inner_edge_dtrsm_run_inv_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = lower
// tran = normal
// unit diagonal
//
// input arguments:
// r10  <- E0
// r11  <- 4*sde*sizeof(double)
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E0
// r11  <- 4*sde*sizeof(double)
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LLN_ONE_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lln_one_8x4_lib4, @function
inner_edge_dtrsm_lln_one_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lln_one_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_lln_one_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_lln_one_8x4_lib4:
#endif
#endif

	// solve top-left
	vxorpd		%ymm14, %ymm14, %ymm14

	vmovapd		0(%r10), %ymm12
	vxorpd		%ymm14, %ymm14, %ymm14
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vmovapd		0(%r10, %r11, 1), %ymm14
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	vmovapd		32(%r10), %ymm12
	vxorpd		%ymm14, %ymm14, %ymm14
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vmovapd		32(%r10, %r11, 1), %ymm14
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	vmovapd		64(%r10), %ymm12
	vxorpd		%ymm14, %ymm14, %ymm14
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vmovapd		64(%r10, %r11, 1), %ymm14
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x11, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	vmovapd		96(%r10, %r11, 1), %ymm14
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x11, %ymm3, %ymm3, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm14, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	addq		$128, %r10


	// solve top-left
	vxorpd		%ymm14, %ymm14, %ymm14

	vmovapd		0(%r10, %r11, 1), %ymm12
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm4, %ymm4, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x00, %ymm5, %ymm5, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x00, %ymm6, %ymm6, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x00, %ymm7, %ymm7, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	vmovapd		32(%r10, %r11, 1), %ymm12
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm4, %ymm4, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x00, %ymm5, %ymm5, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x00, %ymm6, %ymm6, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x00, %ymm7, %ymm7, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7

	vmovapd		64(%r10, %r11, 1), %ymm12
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x11, %ymm4, %ymm4, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm4, %ymm4
	vperm2f128	$0x11, %ymm5, %ymm5, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm5, %ymm5
	vperm2f128	$0x11, %ymm6, %ymm6, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm6, %ymm6
	vperm2f128	$0x11, %ymm7, %ymm7, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm7, %ymm7



#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lln_one_8x4_lib4, .-inner_edge_dtrsm_lln_one_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = upper
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_8x4_lib4, @function
inner_edge_dtrsm_lun_inv_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_lun_inv_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_lun_inv_8x4_lib4:
#endif
#endif
	
	// bottom-right

	vmovapd			224(%r10, %r11, 1), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	56(%r12), %ymm12
	vmovapd			224(%r10), %ymm11

	vperm2f128		$0x11, %ymm4, %ymm4, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm5, %ymm5, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm6, %ymm6, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm7, %ymm7, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			192(%r10, %r11, 1), %xmm13
	vbroadcastsd	48(%r12), %ymm12
	vmovapd			192(%r10), %ymm11

	vperm2f128		$0x11, %ymm4, %ymm4, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm5, %ymm5, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm6, %ymm6, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm7, %ymm7, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			160(%r10, %r11, 1), %xmm13
	vbroadcastsd	40(%r12), %ymm12
	vmovapd			160(%r10), %ymm11

	vperm2f128		$0x00, %ymm4, %ymm4, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x00, %ymm5, %ymm5, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x00, %ymm6, %ymm6, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x00, %ymm7, %ymm7, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vbroadcastsd	32(%r12), %ymm12
	vmovapd			128(%r10), %ymm11

	vperm2f128		$0x00, %ymm4, %ymm4, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x00, %ymm5, %ymm5, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x00, %ymm6, %ymm6, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x00, %ymm7, %ymm7, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	// top-left

	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r12), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r12), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r12), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vbroadcastsd	0(%r12), %ymm12

	vmulpd			%ymm0, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm0, %ymm0

	vmulpd			%ymm1, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm1, %ymm1

	vmulpd			%ymm2, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm2, %ymm2

	vmulpd			%ymm3, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lun_inv_8x4_lib4, .-inner_edge_dtrsm_lun_inv_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = upper
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// r13  <- km
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// r13  <- km
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_8x4_vs_lib4, @function
inner_edge_dtrsm_lun_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_lun_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_edge_dtrsm_lun_inv_8x4_vs_lib4:
#endif
#endif
	
	// bottom-right

	cmpl	$7, %r13d
	jle		0f

	vmovapd			224(%r10, %r11, 1), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	56(%r12), %ymm12
	vmovapd			224(%r10), %ymm11

	vperm2f128		$0x11, %ymm4, %ymm4, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm5, %ymm5, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm6, %ymm6, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm7, %ymm7, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

0:
	cmpl	$6, %r13d
	jle		1f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			192(%r10, %r11, 1), %xmm13
	vbroadcastsd	48(%r12), %ymm12
	vmovapd			192(%r10), %ymm11

	vperm2f128		$0x11, %ymm4, %ymm4, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm5, %ymm5, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm6, %ymm6, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm7, %ymm7, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

1:
	cmpl	$5, %r13d
	jle		2f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			160(%r10, %r11, 1), %xmm13
	vbroadcastsd	40(%r12), %ymm12
	vmovapd			160(%r10), %ymm11

	vperm2f128		$0x00, %ymm4, %ymm4, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x00, %ymm5, %ymm5, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x00, %ymm6, %ymm6, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x00, %ymm7, %ymm7, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

2:

	vbroadcastsd	32(%r12), %ymm12
	vmovapd			128(%r10), %ymm11

	vperm2f128		$0x00, %ymm4, %ymm4, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm4, %ymm4
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x00, %ymm5, %ymm5, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm5, %ymm5
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x00, %ymm6, %ymm6, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm6, %ymm6
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x00, %ymm7, %ymm7, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm7, %ymm7
	vmulpd			%ymm11, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	// top-left

	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r12), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r12), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r12), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vbroadcastsd	0(%r12), %ymm12

	vmulpd			%ymm0, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm0, %ymm0

	vmulpd			%ymm1, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm1, %ymm1

	vmulpd			%ymm2, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm2, %ymm2

	vmulpd			%ymm3, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lun_inv_8x4_vs_lib4, .-inner_edge_dtrsm_lun_inv_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// LU factorization without pivoting
// left kernel
//
// input arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGETRF_L_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgetrf_l_8x4_lib4, @function
inner_edge_dgetrf_l_8x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgetrf_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_edge_dgetrf_l_8x4_lib4; .scl 2; .type 32; .endef
inner_edge_dgetrf_l_8x4_lib4:
#endif
#endif
	
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovsd			.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC04(%rip), %xmm14 // 1.0
#endif
//	vmovddup		%xmm14, %xmm14

	// first column
//	vblendpd		$0x1, %ymm0, %ymm12, %ymm12
	vmovapd			%ymm0, %ymm12
	vdivsd			%xmm0, %xmm14, %xmm13
//	vpermpd			$0x00, %ymm13, %ymm13
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmovsd			%xmm13, 0(%r10)
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vblendpd		$0x1, %ymm12, %ymm0, %ymm0

	// second column
//	vpermpd			$0x00, %ymm1, %ymm13
	vmovddup		%xmm1, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vblendpd		$0x2, %ymm1, %ymm13, %ymm12

	vpermilpd		$0x3, %xmm1, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
//	vpermpd			$0x00, %ymm13, %ymm13
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmovsd			%xmm13, 8(%r10)
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vblendpd		$0x3, %ymm12, %ymm1, %ymm1

	// third column
//	vpermpd			$0x00, %ymm2, %ymm13
	vmovddup		%xmm2, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vblendpd		$0x2, %ymm2, %ymm13, %ymm12

//	vpermpd			$0x55, %ymm2, %ymm13
	vperm2f128		$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd		$0xf, %ymm13, %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vblendpd		$0x4, %ymm2, %ymm12, %ymm12

//	vpermpd			$0xaa, %ymm2, %ymm13
	vperm2f128		$0x11, %ymm2, %ymm2, %ymm13
	vpermilpd		$0x0, %ymm13, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
//	vpermpd			$0x00, %ymm13, %ymm13
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmovsd			%xmm13, 16(%r10)
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vblendpd		$0x7, %ymm12, %ymm2, %ymm2

	// fourth column
//	vpermpd			$0x00, %ymm3, %ymm13
	vmovddup		%xmm3, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm0, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vblendpd		$0x2, %ymm3, %ymm13, %ymm12

//	vpermpd			$0x55, %ymm3, %ymm13
	vperm2f128		$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd		$0xf, %ymm13, %ymm13
	vmulpd			%ymm1, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vblendpd		$0x4, %ymm3, %ymm12, %ymm12

//	vpermpd			$0xaa, %ymm3, %ymm13
	vperm2f128		$0x11, %ymm3, %ymm3, %ymm11
	vpermilpd		$0x0, %ymm11, %ymm13
	vmulpd			%ymm2, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vblendpd		$0x8, %ymm3, %ymm12, %ymm12
	
//	vpermpd			$0xff, %ymm3, %ymm13
//	vperm2f128		$0x11, %ymm3, %ymm3, %ymm11
	vpermilpd		$0xf, %ymm11, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
//	vpermpd			$0x00, %ymm13, %ymm13
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmovsd			%xmm13, 24(%r10)
//	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vblendpd		$0x7, %ymm12, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgetrf_l_8x4_lib4, .-inner_edge_dgetrf_l_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x4_lib4, @function
inner_store_8x4_lib4:
#elif defined(OS_MAC)
_inner_store_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_8x4_lib4; .scl 2; .type 32; .endef
inner_store_8x4_lib4:
#endif
#endif
	
	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4,  0(%r10, %r11, 1)
	vmovapd %ymm5, 32(%r10, %r11, 1)
	vmovapd %ymm6, 64(%r10, %r11, 1)
	vmovapd %ymm7, 96(%r10, %r11, 1)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x4_lib4, .-inner_store_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x4_vs_lib4, @function
inner_store_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_store_8x4_vs_lib4:
#endif
#endif
	
	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC03(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC03(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r10, %r11, 1)
	jl			0f // end
	cmpl		$3, %r13d
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r10, %r11, 1)
	jl			0f // end
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r10, %r11, 1)
	je			0f // end
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r10, %r11, 1)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x4_vs_lib4, .-inner_store_8x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// store lower n
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x4_lib4, @function
inner_store_l_8x4_lib4:
#elif defined(OS_MAC)
_inner_store_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_l_8x4_lib4; .scl 2; .type 32; .endef
inner_store_l_8x4_lib4:
#endif
#endif
	
	vmovapd		%ymm0,0(%r10)
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmovapd		%ymm1, 32(%r10)
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmovapd		%ymm2, 64(%r10)
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmovapd		%ymm3, 96(%r10)

	vmovapd		%ymm4, 0(%r10, %r11, 1)
	vmovapd		%ymm5, 32(%r10, %r11, 1)
	vmovapd		%ymm6, 64(%r10, %r11, 1)
	vmovapd		%ymm7, 96(%r10, %r11, 1)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_8x4_lib4, .-inner_store_l_8x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store lower n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x4_vs_lib4, @function
inner_store_l_8x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_l_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_l_8x4_vs_lib4; .scl 2; .type 32; .endef
inner_store_l_8x4_vs_lib4:
#endif
#endif
	
	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC03(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC03(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r10, %r11, 1)
	jl			0f // end
	cmpl		$3, %r13d
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r10, %r11, 1)
	jl			0f // end
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r10, %r11, 1)
	je			0f // end
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r10, %r11, 1)
0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_8x4_vs_lib4, .-inner_store_l_8x4_vs_lib4
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
// rbp  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
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
// rbp  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x4_gen_lib4, @function
inner_store_8x4_gen_lib4:
#elif defined(OS_MAC)
_inner_store_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_8x4_gen_lib4; .scl 2; .type 32; .endef
inner_store_8x4_gen_lib4:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2sd	%r13d, %xmm14, %xmm14
	vcvtsi2sd	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm12
	vmovupd		.LC03(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm12
	vmovupd		LC03(%rip), %ymm13
#endif
	vmovddup	%xmm14, %xmm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm14, %ymm14, %ymm14
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm12, %ymm14, %ymm14
	vsubpd		%ymm15, %ymm13, %ymm15

	// shift D and sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	vmovapd		%ymm2, %ymm1
	vmovapd		%ymm6, %ymm5
	vmovapd		%ymm3, %ymm2
	vmovapd		%ymm7, %ymm6
	addq		$32, %r11

	cmpl	$1, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	vmovapd		%ymm2, %ymm1
	vmovapd		%ymm6, %ymm5
	addq		$32, %r11

	cmpl	$2, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	addq		$32, %r11

0:

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
	vmaskmovpd	%ymm0, %ymm14,  0(%r11)
	vmaskmovpd	%ymm4, %ymm15,  0(%r11, %r12, 1)
	jl			4f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm14, 32(%r11)
	vmaskmovpd	%ymm5, %ymm15, 32(%r11, %r12, 1)
	jl			4f // end
	vmaskmovpd	%ymm2, %ymm14, 64(%r11)
	vmaskmovpd	%ymm6, %ymm15, 64(%r11, %r12, 1)
	je			4f // end
	vmaskmovpd	%ymm3, %ymm14, 96(%r11)
	vmaskmovpd	%ymm7, %ymm15, 96(%r11, %r12, 1)

	jmp		4f

0:
	
	cmpl	$1, %r10d
	jg		1f

	// offset==1

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x03, %ymm4, %ymm0, %ymm12
	vshufpd		$0x5, %ymm0, %ymm12, %ymm0
	vperm2f128	$0x03, %ymm13, %ymm4, %ymm12
	vshufpd		$0x5, %ymm4, %ymm12, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x03, %ymm5, %ymm1, %ymm12
	vshufpd		$0x5, %ymm1, %ymm12, %ymm1
	vperm2f128	$0x03, %ymm13, %ymm5, %ymm12
	vshufpd		$0x5, %ymm5, %ymm12, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x03, %ymm6, %ymm2, %ymm12
	vshufpd		$0x5, %ymm2, %ymm12, %ymm2
	vperm2f128	$0x03, %ymm13, %ymm6, %ymm12
	vshufpd		$0x5, %ymm6, %ymm12, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x03, %ymm7, %ymm3, %ymm12
	vshufpd		$0x5, %ymm3, %ymm12, %ymm3
	vperm2f128	$0x03, %ymm13, %ymm7, %ymm12
	vshufpd		$0x5, %ymm7, %ymm12, %ymm7

	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm15, %ymm12, %ymm15
	vperm2f128	$0x01, %ymm14, %ymm14, %ymm12
	vshufpd		$0x5, %ymm14, %ymm12, %ymm14

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC08(%rip), %ymm12
	vmovupd		.LC05(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC08(%rip), %ymm12
	vmovupd		LC05(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x1, %ymm14, %ymm15, %ymm14

	jmp		3f

1:

	cmpl	$2, %r10d
	jg		2f

	// offset==2

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x03, %ymm4, %ymm0, %ymm0
	vperm2f128	$0x03, %ymm13, %ymm4, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x03, %ymm5, %ymm1, %ymm1
	vperm2f128	$0x03, %ymm13, %ymm5, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x03, %ymm6, %ymm2, %ymm2
	vperm2f128	$0x03, %ymm13, %ymm6, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x03, %ymm7, %ymm3, %ymm3
	vperm2f128	$0x03, %ymm13, %ymm7, %ymm7

	vperm2f128	$0x01, %ymm14, %ymm14, %ymm14
	vperm2f128	$0x01, %ymm15, %ymm15, %ymm15

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC09(%rip), %ymm12
	vmovupd		.LC06(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC09(%rip), %ymm12
	vmovupd		LC06(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x3, %ymm14, %ymm15, %ymm14

	jmp		3f

2:

	// offset==3

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x21, %ymm0, %ymm4, %ymm12
	vshufpd		$0x5, %ymm12, %ymm4, %ymm0
	vperm2f128	$0x21, %ymm4, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x21, %ymm1, %ymm5, %ymm12
	vshufpd		$0x5, %ymm12, %ymm5, %ymm1
	vperm2f128	$0x21, %ymm5, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x21, %ymm2, %ymm6, %ymm12
	vshufpd		$0x5, %ymm12, %ymm6, %ymm2
	vperm2f128	$0x21, %ymm6, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x21, %ymm3, %ymm7, %ymm12
	vshufpd		$0x5, %ymm12, %ymm7, %ymm3
	vperm2f128	$0x21, %ymm7, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm7

	vperm2f128	$0x01, %ymm14, %ymm14, %ymm12
	vshufpd		$0x5, %ymm12, %ymm14, %ymm14
	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm12, %ymm15, %ymm15

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC10(%rip), %ymm12
	vmovupd		.LC07(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC10(%rip), %ymm12
	vmovupd		LC07(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x7, %ymm14, %ymm15, %ymm14

3:

	cmpl		$2, %r15d
	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm4, %ymm14, 0(%r11, %r12, 1)
	vmaskmovpd	%ymm0, %ymm13, 0(%r11, %r12, 2)
	jl			4f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm5, %ymm14, 32(%r11, %r12, 1)
	vmaskmovpd	%ymm1, %ymm13, 32(%r11, %r12, 2)
	jl			4f // end
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm6, %ymm14, 64(%r11, %r12, 1)
	vmaskmovpd	%ymm2, %ymm13, 64(%r11, %r12, 2)
	je			4f // end
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm7, %ymm14, 96(%r11, %r12, 1)
	vmaskmovpd	%ymm3, %ymm13, 96(%r11, %r12, 2)

4:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x4_gen_lib4, .-inner_store_8x4_gen_lib4
#endif
#endif





// common inner routine with file scope
//
// store l generalized
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
// rbp  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
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
// rbp  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x4_gen_lib4, @function
inner_store_l_8x4_gen_lib4:
#elif defined(OS_MAC)
_inner_store_l_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_l_8x4_gen_lib4; .scl 2; .type 32; .endef
inner_store_l_8x4_gen_lib4:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2sd	%r13d, %xmm14, %xmm14
	vcvtsi2sd	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %ymm12
	vmovupd		.LC03(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm12
	vmovupd		LC03(%rip), %ymm13
#endif
	vmovddup	%xmm14, %xmm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm14, %ymm14, %ymm14
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm12, %ymm14, %ymm14
	vsubpd		%ymm15, %ymm13, %ymm15

	// shift D and sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	vmovapd		%ymm2, %ymm1
	vmovapd		%ymm6, %ymm5
	vmovapd		%ymm3, %ymm2
	vmovapd		%ymm7, %ymm6
	addq		$32, %r11

	cmpl	$1, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	vmovapd		%ymm2, %ymm1
	vmovapd		%ymm6, %ymm5
	addq		$32, %r11

	cmpl	$2, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm5, %ymm4
	addq		$32, %r11

0:

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
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovapd		.LC04(%rip), %ymm13
#elif defined(OS_MAC)
	vmovapd		LC04(%rip), %ymm13
#endif

	vmaskmovpd	%ymm0, %ymm14,  0(%r11)
	vmaskmovpd	%ymm4, %ymm15,  0(%r11, %r12, 1)
	cmpl		$2, %r15d
	jl			3f // end
	vblendpd	$0x1, %ymm13, %ymm14, %ymm14
	vmaskmovpd	%ymm1, %ymm14, 32(%r11)
	vmaskmovpd	%ymm5, %ymm15, 32(%r11, %r12, 1)
	cmpl		$3, %r15d
	jl			3f // end
	vblendpd	$0x2, %ymm13, %ymm14, %ymm14
	vmaskmovpd	%ymm2, %ymm14, 64(%r11)
	vmaskmovpd	%ymm6, %ymm15, 64(%r11, %r12, 1)
	je			3f // end
	vblendpd	$0x4, %ymm13, %ymm14, %ymm14
	vmaskmovpd	%ymm3, %ymm14, 96(%r11)
	vmaskmovpd	%ymm7, %ymm15, 96(%r11, %r12, 1)

	jmp		3f

0:
	
	cmpl	$1, %r10d
	jg		1f

	// offset==1

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x03, %ymm4, %ymm0, %ymm12
	vshufpd		$0x5, %ymm0, %ymm12, %ymm0
	vperm2f128	$0x03, %ymm13, %ymm4, %ymm12
	vshufpd		$0x5, %ymm4, %ymm12, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x03, %ymm5, %ymm1, %ymm12
	vshufpd		$0x5, %ymm1, %ymm12, %ymm1
	vperm2f128	$0x03, %ymm13, %ymm5, %ymm12
	vshufpd		$0x5, %ymm5, %ymm12, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x03, %ymm6, %ymm2, %ymm12
	vshufpd		$0x5, %ymm2, %ymm12, %ymm2
	vperm2f128	$0x03, %ymm13, %ymm6, %ymm12
	vshufpd		$0x5, %ymm6, %ymm12, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x03, %ymm7, %ymm3, %ymm12
	vshufpd		$0x5, %ymm3, %ymm12, %ymm3
	vperm2f128	$0x03, %ymm13, %ymm7, %ymm12
	vshufpd		$0x5, %ymm7, %ymm12, %ymm7

	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm15, %ymm12, %ymm15
	vperm2f128	$0x01, %ymm14, %ymm14, %ymm12
	vshufpd		$0x5, %ymm14, %ymm12, %ymm14

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC08(%rip), %ymm12
	vmovupd		.LC05(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC08(%rip), %ymm12
	vmovupd		LC05(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x1, %ymm14, %ymm15, %ymm14

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovapd		.LC04(%rip), %ymm15
#elif defined(OS_MAC)
	vmovapd		LC04(%rip), %ymm15
#endif

	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm4, %ymm14, 0(%r11, %r12, 1)
	vmaskmovpd	%ymm0, %ymm13, 0(%r11, %r12, 2)
	cmpl		$2, %r15d
	jl			3f // end
	vblendpd	$0x2, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm5, %ymm14, 32(%r11, %r12, 1)
	vmaskmovpd	%ymm1, %ymm13, 32(%r11, %r12, 2)
	cmpl		$3, %r15d
	jl			3f // end
	vblendpd	$0x4, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm6, %ymm14, 64(%r11, %r12, 1)
	vmaskmovpd	%ymm2, %ymm13, 64(%r11, %r12, 2)
	je			3f // end
	vblendpd	$0x8, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm7, %ymm14, 96(%r11, %r12, 1)
	vmaskmovpd	%ymm3, %ymm13, 96(%r11, %r12, 2)

	jmp		3f

1:

	cmpl	$2, %r10d
	jg		2f

	// offset==2

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x03, %ymm4, %ymm0, %ymm0
	vperm2f128	$0x03, %ymm13, %ymm4, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x03, %ymm5, %ymm1, %ymm1
	vperm2f128	$0x03, %ymm13, %ymm5, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x03, %ymm6, %ymm2, %ymm2
	vperm2f128	$0x03, %ymm13, %ymm6, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x03, %ymm7, %ymm3, %ymm3
	vperm2f128	$0x03, %ymm13, %ymm7, %ymm7

	vperm2f128	$0x01, %ymm14, %ymm14, %ymm14
	vperm2f128	$0x01, %ymm15, %ymm15, %ymm15

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC09(%rip), %ymm12
	vmovupd		.LC06(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC09(%rip), %ymm12
	vmovupd		LC06(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x3, %ymm14, %ymm15, %ymm14

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovapd		.LC04(%rip), %ymm15
#elif defined(OS_MAC)
	vmovapd		LC04(%rip), %ymm15
#endif

	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm4, %ymm14, 0(%r11, %r12, 1)
	vmaskmovpd	%ymm0, %ymm13, 0(%r11, %r12, 2)
	cmpl		$2, %r15d
	jl			3f // end
	vblendpd	$0x4, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm5, %ymm14, 32(%r11, %r12, 1)
	vmaskmovpd	%ymm1, %ymm13, 32(%r11, %r12, 2)
	cmpl		$3, %r15d
	jl			3f // end
	vblendpd	$0x8, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm6, %ymm14, 64(%r11, %r12, 1)
	vmaskmovpd	%ymm2, %ymm13, 64(%r11, %r12, 2)
	je			3f // end
	vblendpd	$0x1, %ymm15, %ymm14, %ymm14
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm7, %ymm14, 96(%r11, %r12, 1)
	vmaskmovpd	%ymm3, %ymm13, 96(%r11, %r12, 2)

	jmp		3f

2:

	// offset==3

	vmovapd		%ymm0, %ymm13
	vperm2f128	$0x21, %ymm0, %ymm4, %ymm12
	vshufpd		$0x5, %ymm12, %ymm4, %ymm0
	vperm2f128	$0x21, %ymm4, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm4

	vmovapd		%ymm1, %ymm13
	vperm2f128	$0x21, %ymm1, %ymm5, %ymm12
	vshufpd		$0x5, %ymm12, %ymm5, %ymm1
	vperm2f128	$0x21, %ymm5, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm5

	vmovapd		%ymm2, %ymm13
	vperm2f128	$0x21, %ymm2, %ymm6, %ymm12
	vshufpd		$0x5, %ymm12, %ymm6, %ymm2
	vperm2f128	$0x21, %ymm6, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm6

	vmovapd		%ymm3, %ymm13
	vperm2f128	$0x21, %ymm3, %ymm7, %ymm12
	vshufpd		$0x5, %ymm12, %ymm7, %ymm3
	vperm2f128	$0x21, %ymm7, %ymm13, %ymm12
	vshufpd		$0x5, %ymm12, %ymm13, %ymm7

	vperm2f128	$0x01, %ymm14, %ymm14, %ymm12
	vshufpd		$0x5, %ymm12, %ymm14, %ymm14
	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm12, %ymm15, %ymm15

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC10(%rip), %ymm12
	vmovupd		.LC07(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC10(%rip), %ymm12
	vmovupd		LC07(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm14, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	vblendpd	$0x7, %ymm14, %ymm15, %ymm14

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovapd		.LC04(%rip), %ymm15
#elif defined(OS_MAC)
	vmovapd		LC04(%rip), %ymm15
#endif

	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm4, %ymm14, 0(%r11, %r12, 1)
	vmaskmovpd	%ymm0, %ymm13, 0(%r11, %r12, 2)
	cmpl		$2, %r15d
	jl			3f // end
	vblendpd	$0x8, %ymm15, %ymm12, %ymm12
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm5, %ymm14, 32(%r11, %r12, 1)
	vmaskmovpd	%ymm1, %ymm13, 32(%r11, %r12, 2)
	cmpl		$3, %r15d
	jl			3f // end
	vblendpd	$0x1, %ymm15, %ymm14, %ymm14
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm6, %ymm14, 64(%r11, %r12, 1)
	vmaskmovpd	%ymm2, %ymm13, 64(%r11, %r12, 2)
	je			3f // end
	vblendpd	$0x2, %ymm15, %ymm14, %ymm14
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm7, %ymm14, 96(%r11, %r12, 1)
	vmaskmovpd	%ymm3, %ymm13, 96(%r11, %r12, 2)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_8x4_gen_lib4, .-inner_store_l_8x4_gen_lib4
#endif
#endif





//                               rdi    rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dgemm_nt_8x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_8x4_lib4
	.type kernel_dgemm_nt_8x4_lib4, @function
kernel_dgemm_nt_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_8x4_lib4
_kernel_dgemm_nt_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_8x4_lib4
	.def kernel_dgemm_nt_8x4_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nt_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_8x4_lib4, .-kernel_dgemm_nt_8x4_lib4
#endif





//                                  rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32   rsp+40  rsp+48
// void kernel_dgemm_nt_8x4_vs_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_8x4_vs_lib4
	.type kernel_dgemm_nt_8x4_vs_lib4, @function
kernel_dgemm_nt_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_8x4_vs_lib4
_kernel_dgemm_nt_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_8x4_vs_lib4
	.def kernel_dgemm_nt_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nt_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG9, %r10 // store address D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG11, %r12 // km 
	movq	ARG12, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_8x4_vs_lib4, .-kernel_dgemm_nt_8x4_vs_lib4
#endif





//                                   rdi    rsi            rdx        rcx      r8         r9            rsp+8        rsp+16     rsp+24   rsp+32       rsp+40     rsp+48   rsp+56  rsp+64  rsp+72  rsp+80
// void kernel_dgemm_nt_8x4_gen_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, int offsetC, double *C, int sdc, int offsetD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_8x4_gen_lib4
	.type kernel_dgemm_nt_8x4_gen_lib4, @function
kernel_dgemm_nt_8x4_gen_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_8x4_gen_lib4
_kernel_dgemm_nt_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_8x4_gen_lib4
	.def kernel_dgemm_nt_8x4_gen_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nt_8x4_gen_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // offsetC
	movq	ARG8, %r13 // C
	movq	ARG9, %r14 // sdc
	sall	$5, %r14d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_gen_lib4
#endif
#endif


	// store n

	movq	ARG10, %r10 // offsetD
	movq	ARG11, %r11 // D
	movq	ARG12, %r12 // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG13, %r13 // m0
	movq	ARG14, %r14 // m1
	movq	ARG15, %r15 // n0
	movq	ARG16, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_gen_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_8x4_gen_lib4, .-kernel_dgemm_nt_8x4_gen_lib4
#endif





//                               rdi    rsi            rdx        rcx      r8           r9         rsp+8    rsp+16        rsp+24     rsp+32   rsp+40     rsp+48
// void kernel_dgemm_nn_8x4_lib4(int k, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_8x4_lib4
	.type kernel_dgemm_nn_8x4_lib4, @function
kernel_dgemm_nn_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_8x4_lib4
_kernel_dgemm_nn_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_8x4_lib4
	.def kernel_dgemm_nn_8x4_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nn_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG6, %r13 // B
	movq	ARG7, %r14 // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)
	movq	ARG5, %r15 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgemm_add_nn_8x4_lib4
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_8x4_lib4
#endif
#endif


	// call inner scale

	movq	ARG2, %r10 // alpha
	movq	ARG8, %r11 // beta
	movq	ARG9, %r12 // C
	movq	ARG10, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG11, %r10 // D
	movq	ARG12, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_8x4_lib4, .-kernel_dgemm_nn_8x4_lib4
#endif





//                                   rdi    rsi            rdx        rcx      r8        r9         rsp+8    rsp+16        rsp+24    rsp+32     rsp+40   rsp+48    rsp+56     rsp+64   rsp+72  rsp+80  rsp+88  rsp+96
// void kernel_dgemm_nn_8x4_gen_lib4(int k, double *alpha, double *A, int sda, int offB, double *B, int sdb, double *beta, int offC, double *C, int sdc, int offD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_8x4_gen_lib4
	.type kernel_dgemm_nn_8x4_gen_lib4, @function
kernel_dgemm_nn_8x4_gen_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_8x4_gen_lib4
_kernel_dgemm_nn_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nn_8x4_gen_lib4
	.def kernel_dgemm_nn_8x4_gen_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nn_8x4_gen_lib4:
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


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG6, %r13  // B
	movq	ARG7, %r14 // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)
	movq	ARG5, %r15 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgemm_add_nn_8x4_lib4
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG8, %r11 // beta
	movq	ARG9, %r12 // offsetC
	movq	ARG10, %r13 // C
	movq	ARG11, %r14 // sdc
	sall	$5, %r14d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x4_gen_lib4
#endif
#endif


	// store n gen

	movq	ARG12, %r10 // offsetD
	movq	ARG13, %r11 // D
	movq	ARG14, %r12 // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG15, %r13 // m0
	movq	ARG16, %r14 // m1
	movq	ARG17, %r15 // n0
	movq	ARG18, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_gen_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nn_8x4_gen_lib4, .-kernel_dgemm_nn_8x4_gen_lib4
#endif





//                                 rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dsyrk_nt_l_8x4_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_8x4_lib4
	.type kernel_dsyrk_nt_l_8x4_lib4, @function
kernel_dsyrk_nt_l_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_8x4_lib4
_kernel_dsyrk_nt_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_l_8x4_lib4
	.def kernel_dsyrk_nt_l_8x4_lib4; .scl 2; .type 32; .endef
kernel_dsyrk_nt_l_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_lib4
#endif
#endif

	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_8x4_lib4, .-kernel_dsyrk_nt_l_8x4_lib4
#endif





//                                    rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32   rsp+40  rsp+48
// void kernel_dsyrk_nt_l_8x4_vs_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_8x4_vs_lib4
	.type kernel_dsyrk_nt_l_8x4_vs_lib4, @function
kernel_dsyrk_nt_l_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_8x4_vs_lib4
_kernel_dsyrk_nt_l_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_l_8x4_vs_lib4
	.def kernel_dsyrk_nt_l_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dsyrk_nt_l_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG9, %r10 // store address D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG11, %r12 // km 
	movq	ARG12, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_vs_lib4
#endif
#endif

	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_8x4_vs_lib4, .-kernel_dsyrk_nt_l_8x4_vs_lib4
#endif





//                                     rdi    rsi            rdx        rcx      r8         r9            rsp+8        rsp+16     rsp+24   rsp+32       rsp+40     rsp+48   rsp+56  rsp+64  rsp+72  rsp+80
// void kernel_dsyrk_nt_l_8x4_gen_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, int offsetC, double *C, int sdc, int offsetD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_8x4_gen_lib4
	.type kernel_dsyrk_nt_l_8x4_gen_lib4, @function
kernel_dsyrk_nt_l_8x4_gen_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_8x4_gen_lib4
_kernel_dsyrk_nt_l_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_nt_l_8x4_gen_lib4
	.def kernel_dsyrk_nt_l_8x4_gen_lib4; .scl 2; .type 32; .endef
kernel_dsyrk_nt_l_8x4_gen_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // offsetC
	movq	ARG8, %r13 // C
	movq	ARG9, %r14 // sdc
	sall	$5, %r14d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x4_gen_lib4
#endif
#endif


	// store n

	movq	ARG10, %r10 // offsetD
	movq	ARG11, %r11 // D
	movq	ARG12, %r12 // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG13, %r13 // m0
	movq	ARG14, %r14 // m1
	movq	ARG15, %r15 // n0
	movq	ARG16, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_gen_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_nt_l_8x4_gen_lib4, .-kernel_dsyrk_nt_l_8x4_gen_lib4
#endif





//                                  rdi    rsi            rdx        rcx      r8           r9         rsp+8    rsp+16     rsp+24
// void kernel_dtrmm_nn_rl_8x4_lib4(int k, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nn_rl_8x4_lib4
	.type kernel_dtrmm_nn_rl_8x4_lib4, @function
kernel_dtrmm_nn_rl_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nn_rl_8x4_lib4
_kernel_dtrmm_nn_rl_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nn_rl_8x4_lib4
	.def kernel_dtrmm_nn_rl_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrmm_nn_rl_8x4_lib4:
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


	// initial triangle

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sdb
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG6, %r13 // B
	movq	ARG7, %r14 // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)
	movq	ARG5, %r15 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NN_RL_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nn_rl_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nn_rl_8x4_lib4
#endif
#endif

	// call inner dgemm kernel nt after initial triangle

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_8x4_lib4
#endif
#endif


	// call inner scale

	movq	ARG2, %r10 // alpha

#if MACRO_LEVEL>=1
	INNER_SCALE_A0_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_a0_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_a0_8x4_lib4
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nn_rl_8x4_lib4, .-kernel_dtrmm_nn_rl_8x4_lib4
#endif





//                                      rdi    rsi            rdx        rcx      r8           r9         rsp+8    rsp+16       rsp+24     rsp+32   rsp+40  rsp+48  rsp+56  rsp+64
// void kernel_dtrmm_nn_rl_8x4_gen_lib4(int k, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, int offsetD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nn_rl_8x4_gen_lib4
	.type kernel_dtrmm_nn_rl_8x4_gen_lib4, @function
kernel_dtrmm_nn_rl_8x4_gen_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nn_rl_8x4_gen_lib4
_kernel_dtrmm_nn_rl_8x4_gen_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nn_rl_8x4_gen_lib4
	.def kernel_dtrmm_nn_rl_8x4_gen_lib4; .scl 2; .type 32; .endef
kernel_dtrmm_nn_rl_8x4_gen_lib4:
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


	// initial triangle

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sdb
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG6, %r13 // B
	movq	ARG7, %r14 // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)
	movq	ARG5, %r15 // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NN_RL_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nn_rl_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nn_rl_8x4_gen_lib4
#endif
#endif

	// call inner dgemm kernel nt after initial triangle

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_8x4_lib4
#endif
#endif


	// call inner scale

	movq	ARG2, %r10 // alpha

#if MACRO_LEVEL>=1
	INNER_SCALE_A0_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_a0_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_a0_8x4_lib4
#endif
#endif


	// store n

	movq	ARG8, %r10 // offsetD
	movq	ARG9, %r11 // D
	movq	ARG10, %r12 // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	ARG11, %r13 // m0
	movq	ARG12, %r14 // m1
	movq	ARG13, %r15 // n0
	movq	ARG14, %rax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_GEN_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_gen_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nn_rl_8x4_gen_lib4, .-kernel_dtrmm_nn_rl_8x4_gen_lib4
#endif





//                                  rdi    rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dtrmm_nt_ru_8x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_8x4_lib4
	.type kernel_dtrmm_nt_ru_8x4_lib4, @function
kernel_dtrmm_nt_ru_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_8x4_lib4
_kernel_dtrmm_nt_ru_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nt_ru_8x4_lib4
	.def kernel_dtrmm_nt_ru_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrmm_nt_ru_8x4_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movq	ARG1, %r10 // k
	subl	$4, %r10d //k-4
	movq	ARG3, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B
	addq	$128, %r13 // B+4*bs


#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_8x4_lib4
#endif
#endif


	// initial triangle

	movq	ARG3, %r10 // A
	movq	ARG4, %r11 // sda
	sall	$5, %r11d // 4*sda*sizeof(double)
	movq	ARG5, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nt_ru_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_8x4_lib4
#endif
#endif


	// call inner scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x4_lib4
#endif
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_8x4_lib4, .-kernel_dtrmm_nt_ru_8x4_lib4
#endif





//                                 rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32   rsp+40  rsp+48
// void kernel_dtrmm_nt_ru_8x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_8x4_vs_lib4
	.type kernel_dtrmm_nt_ru_8x4_vs_lib4, @function
kernel_dtrmm_nt_ru_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_8x4_vs_lib4
_kernel_dtrmm_nt_ru_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrmm_nt_ru_8x4_vs_lib4
	.def kernel_dtrmm_nt_ru_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrmm_nt_ru_8x4_vs_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movq	ARG1, %r10 // k
	subl	$4, %r10d //k-4
	movq	ARG3, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B
	addq	$128, %r13 // B+4*bs


#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_8x4_lib4
#endif
#endif


	// initial triangle

	movq	ARG1, %r10
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrmm_nt_ru_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_8x4_vs_lib4
#endif
#endif


	// call inner scale

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12 // C
	movq	ARG8, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x4_lib4
#endif
#endif


// store n

	movq	ARG9, %r10 // store address D
	movq	ARG10, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG11, %r12 // km 
	movq	ARG12, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrmm_nt_ru_8x4_vs_lib4, .-kernel_dtrmm_nt_ru_8x4_vs_lib4
#endif





//                                  rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24
// void kernel_dpotrf_nt_l_8x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_8x4_lib4
	.type kernel_dpotrf_nt_l_8x4_lib4, @function
kernel_dpotrf_nt_l_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_8x4_lib4
_kernel_dpotrf_nt_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dpotrf_nt_l_8x4_lib4
	.def kernel_dpotrf_nt_l_8x4_lib4; .scl 2; .type 32; .endef
kernel_dpotrf_nt_l_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10 // C
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG9, %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_8x4_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_8x4_lib4, .-kernel_dpotrf_nt_l_8x4_lib4
#endif





//                                     rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24              rsp+32  rsp+40 
// void kernel_dpotrf_nt_l_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_8x4_vs_lib4
	.type kernel_dpotrf_nt_l_8x4_vs_lib4, @function
kernel_dpotrf_nt_l_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_8x4_vs_lib4
_kernel_dpotrf_nt_l_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dpotrf_nt_l_8x4_vs_lib4
	.def kernel_dpotrf_nt_l_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dpotrf_nt_l_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10 // C
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG9, %r10  // inv_diag_D 
	movq	ARG11, %r11 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG10, %r12 // km 
	movq	ARG11, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dpotrf_nt_l_8x4_vs_lib4, .-kernel_dpotrf_nt_l_8x4_vs_lib4
#endif





//                                        rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56
// void kernel_dsyrk_dpotrf_nt_l_8x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_8x4_lib4
	.type kernel_dsyrk_dpotrf_nt_l_8x4_lib4, @function
kernel_dsyrk_dpotrf_nt_l_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_8x4_lib4
_kernel_dsyrk_dpotrf_nt_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_dpotrf_nt_l_8x4_lib4
	.def kernel_dsyrk_dpotrf_nt_l_8x4_lib4; .scl 2; .type 32; .endef
kernel_dsyrk_dpotrf_nt_l_8x4_lib4:
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


	// call inner dgemm kernel nt add

	movq	ARG1, %r10 // kp
	movq	ARG2, %r11  // Ap
	movq	ARG3, %r12 // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movq	ARG4, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movq	ARG5, %r10                 // km
	movq	ARG6, %r11                   // Am
	movq	ARG7, %r12 // sdam
	sall	$5, %r12d                   // 4*sdam*sizeof(double)
	movq	ARG8, %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG9, %r10 // C
	movq	ARG10, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG13, %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_8x4_lib4
#endif
#endif


	// store n

	movq	ARG11, %r10 // store address D
	movq	ARG12, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_dpotrf_nt_l_8x4_lib4, .-kernel_dsyrk_dpotrf_nt_l_8x4_lib4
#endif





//                                           rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56              rsp+64  rsp+72
// void kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4
	.type kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4
_kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4
	.def kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4:
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


	// call inner dgemm kernel nt add

	movq	ARG1, %r10 // kp
	movq	ARG2, %r11  // Ap
	movq	ARG3, %r12 // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movq	ARG4, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movq	ARG5, %r10                 // km
	movq	ARG6, %r11                   // Am
	movq	ARG7, %r12 // sdam
	sall	$5, %r12d                   // 4*sdam*sizeof(double)
	movq	ARG8, %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG9, %r10 // C
	movq	ARG10, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG13, %r10  // inv_diag_D 
	movq	ARG15, %r11 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dpotrf_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG11, %r10 // store address D
	movq	ARG12, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG14, %r12 // km 
	movq	ARG15, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_l_8x4_vs_lib4
#endif





//                                         rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_8x4_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_8x4_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_8x4_vs_lib4
_kernel_dtrsm_nt_rl_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_8x4_vs_lib4
	.def kernel_dtrsm_nt_rl_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 
	movq	ARG12, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG11, %r12 // km 
	movq	ARG12, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_8x4_vs_lib4, .-kernel_dtrsm_nt_rl_inv_8x4_vs_lib4
#endif





//                                               rdi     rsi         rdx       rcx          r8     r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56     rsp+64              rsp+72  rsp+80
// void kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4
	.def kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4:
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


	// call inner dgemm kernel nt add

	movq	ARG1, %r10 // kp
	movq	ARG2, %r11  // Ap
	movq	ARG3, %r12 // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movq	ARG4, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movq	ARG5, %r10                 // km
	movq	ARG6, %r11                   // Am
	movq	ARG7, %r12 // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
	movq	ARG8, %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG9, %r10  // C
	movq	ARG10, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG13, %r10  // E 
	movq	ARG14, %r11  // inv_diag_E 
	movq	ARG16, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG11, %r10 // store address D
	movq	ARG12, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG15, %r12 // km 
	movq	ARG16, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_8x4_vs_lib4
#endif





//                                      rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32 
// void kernel_dtrsm_nt_rl_inv_8x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_8x4_lib4
	.type kernel_dtrsm_nt_rl_inv_8x4_lib4, @function
kernel_dtrsm_nt_rl_inv_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_8x4_lib4
_kernel_dtrsm_nt_rl_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_8x4_lib4
	.def kernel_dtrsm_nt_rl_inv_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_8x4_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_inv_8x4_lib4, .-kernel_dtrsm_nt_rl_inv_8x4_lib4
#endif





//                                            rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56     rsp+64
// void kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4
	.def kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4; .scl 2; .type 32; .endef
kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4:
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


	// call inner dgemm kernel nt add

	movq	ARG1, %r10 // kp
	movq	ARG2, %r11  // Ap
	movq	ARG3, %r12 // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movq	ARG4, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movq	ARG5, %r10                 // km
	movq	ARG6, %r11                   // Am
	movq	ARG7, %r12 // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
	movq	ARG8, %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG9, %r10  // C
	movq	ARG10, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG13, %r10  // E 
	movq	ARG14, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_8x4_lib4
#endif
#endif


	// store n

	movq	ARG11, %r10 // store address D
	movq	ARG12, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_8x4_lib4
#endif





//                                      rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24
// void kernel_dtrsm_nt_rl_one_8x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_8x4_lib4
	.type kernel_dtrsm_nt_rl_one_8x4_lib4, @function
kernel_dtrsm_nt_rl_one_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_8x4_lib4
_kernel_dtrsm_nt_rl_one_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_8x4_lib4
	.def kernel_dtrsm_nt_rl_one_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_8x4_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_8x4_lib4, .-kernel_dtrsm_nt_rl_one_8x4_lib4
#endif





//                                         rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32  rsp+40
// void kernel_dtrsm_nt_ru_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_8x4_vs_lib4
	.type kernel_dtrsm_nt_rl_one_8x4_vs_lib4, @function
kernel_dtrsm_nt_rl_one_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_8x4_vs_lib4
_kernel_dtrsm_nt_rl_one_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_one_8x4_vs_lib4
	.def kernel_dtrsm_nt_rl_one_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_one_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG11, %r11 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_one_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG10, %r12 // km 
	movq	ARG11, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_rl_one_8x4_vs_lib4, .-kernel_dtrsm_nt_rl_one_8x4_vs_lib4
#endif





//                                      rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32 
// void kernel_dtrsm_nt_ru_inv_8x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_8x4_lib4
	.type kernel_dtrsm_nt_ru_inv_8x4_lib4, @function
kernel_dtrsm_nt_ru_inv_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_8x4_lib4
_kernel_dtrsm_nt_ru_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_8x4_lib4
	.def kernel_dtrsm_nt_ru_inv_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_8x4_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_8x4_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_8x4_lib4, .-kernel_dtrsm_nt_ru_inv_8x4_lib4
#endif





//                                         rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dtrsm_nt_ru_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_8x4_vs_lib4
	.type kernel_dtrsm_nt_ru_inv_8x4_vs_lib4, @function
kernel_dtrsm_nt_ru_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_8x4_vs_lib4
_kernel_dtrsm_nt_ru_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_ru_inv_8x4_vs_lib4
	.def kernel_dtrsm_nt_ru_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nt_ru_inv_8x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nt_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_8x4_lib4
#endif
#endif


	// call inner blender nn

	movq	ARG5, %r10
	movq	ARG6, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG9, %r10  // E 
	movq	ARG10, %r11  // inv_diag_E 
	movq	ARG12, %r12 // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rut_inv_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_8x4_vs_lib4
#endif
#endif


	// store n

	movq	ARG7, %r10 // store address D
	movq	ARG8, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG11, %r12 // km 
	movq	ARG12, %r13 // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nt_ru_inv_8x4_vs_lib4, .-kernel_dtrsm_nt_ru_inv_8x4_vs_lib4
#endif





//                                      edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40
// void kernel_dtrsm_nn_ru_inv_8x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_8x4_lib4
	.type kernel_dtrsm_nn_ru_inv_8x4_lib4, @function
kernel_dtrsm_nn_ru_inv_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_8x4_lib4
_kernel_dtrsm_nn_ru_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_inv_8x4_lib4
	.def kernel_dtrsm_nn_ru_inv_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_inv_8x4_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_inv_8x4_lib4, .-kernel_dtrsm_nn_ru_inv_8x4_lib4
#endif





//                                         edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40              rsp+48  rsp+56
// void kernel_dtrsm_nn_ru_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_8x4_vs_lib4
	.type kernel_dtrsm_nn_ru_inv_8x4_vs_lib4, @function
kernel_dtrsm_nn_ru_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_8x4_vs_lib4
_kernel_dtrsm_nn_ru_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ru_inv_8x4_vs_lib4
	.def kernel_dtrsm_nn_ru_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ru_inv_8x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_run_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG12, %r12 // km
	movq	ARG13, %r13 // kn

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ru_inv_8x4_vs_lib4, .-kernel_dtrsm_nn_ru_inv_8x4_vs_lib4
#endif





//                                      edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40
// void kernel_dtrsm_nn_ll_one_8x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_8x4_lib4
	.type kernel_dtrsm_nn_ll_one_8x4_lib4, @function
kernel_dtrsm_nn_ll_one_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_8x4_lib4
_kernel_dtrsm_nn_ll_one_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ll_one_8x4_lib4
	.def kernel_dtrsm_nn_ll_one_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ll_one_8x4_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // sde
	sall	$5, %r11d // 4*sde*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_lln_one_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ll_one_8x4_lib4, .-kernel_dtrsm_nn_ll_one_8x4_lib4
#endif





//                                         edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48  tsp+56
// void kernel_dtrsm_nn_ll_one_8x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_8x4_vs_lib4
	.type kernel_dtrsm_nn_ll_one_8x4_vs_lib4, @function
kernel_dtrsm_nn_ll_one_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_8x4_vs_lib4
_kernel_dtrsm_nn_ll_one_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_ll_one_8x4_vs_lib4
	.def kernel_dtrsm_nn_ll_one_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_ll_one_8x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // sde
	sall	$5, %r11d // 4*sde*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_lln_one_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

	movq	ARG12, %r12 // km
	movq	ARG13, %r13 // kn

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_ll_one_8x4_vs_lib4, .-kernel_dtrsm_nn_ll_one_8x4_vs_lib4
#endif





//                                      edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48
// void kernel_dtrsm_nn_lu_inv_8x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_8x4_lib4
	.type kernel_dtrsm_nn_lu_inv_8x4_lib4, @function
kernel_dtrsm_nn_lu_inv_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_8x4_lib4
_kernel_dtrsm_nn_lu_inv_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_lu_inv_8x4_lib4
	.def kernel_dtrsm_nn_lu_inv_8x4_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_lu_inv_8x4_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
	movq	ARG12, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_lun_inv_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_lu_inv_8x4_lib4, .-kernel_dtrsm_nn_lu_inv_8x4_lib4
#endif





//                                         edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48              rsp+56  rsp+64
// void kernel_dtrsm_nn_lu_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_8x4_vs_lib4
	.type kernel_dtrsm_nn_lu_inv_8x4_vs_lib4, @function
kernel_dtrsm_nn_lu_inv_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_8x4_vs_lib4
_kernel_dtrsm_nn_lu_inv_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nn_lu_inv_8x4_vs_lib4
	.def kernel_dtrsm_nn_lu_inv_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dtrsm_nn_lu_inv_8x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// solve

	movq	ARG10, %r10  // E 
	movq	ARG11, %r11 // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
	movq	ARG12, %r12  // inv_diag_E 
	movq	ARG13, %r13  // km

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_lun_inv_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_8x4_vs_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	ARG13, %r12  // km
	movq	ARG14, %r13  // km

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dtrsm_nn_lu_inv_8x4_vs_lib4, .-kernel_dtrsm_nn_lu_inv_8x4_vs_lib4
#endif





//                                edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32
// void kernel_dgetrf_nn_8x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_l_8x4_lib4
	.type kernel_dgetrf_nn_l_8x4_lib4, @function
kernel_dgetrf_nn_l_8x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_l_8x4_lib4
_kernel_dgetrf_nn_l_8x4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgetrf_nn_l_8x4_lib4
	.def kernel_dgetrf_nn_l_8x4_lib4; .scl 2; .type 32; .endef
kernel_dgetrf_nn_l_8x4_lib4:
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


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG10, %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_L_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dgetrf_l_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_l_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_lib4
#endif
#endif


	// epilogue

	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgetrf_nn_l_8x4_lib4, .-kernel_dgetrf_nn_l_8x4_lib4
#endif





//                                   edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32              rsp+40  rsp+48
// void kernel_dgetrf_nn_8x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_l_8x4_vs_lib4
	.type kernel_dgetrf_nn_l_8x4_vs_lib4, @function
kernel_dgetrf_nn_l_8x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_l_8x4_vs_lib4
_kernel_dgetrf_nn_l_8x4_vs_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgetrf_nn_l_8x4_vs_lib4
	.def kernel_dgetrf_nn_l_8x4_vs_lib4; .scl 2; .type 32; .endef
kernel_dgetrf_nn_l_8x4_vs_lib4:
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


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_sub_nn_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_8x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG6, %r10 // C
	movq	ARG7, %r11 // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_11_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_8x4_lib4
#endif
#endif


	// factorization

	movq	ARG10, %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_L_8X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dgetrf_l_8x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_l_8x4_lib4
#endif
#endif


	// store

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	ARG11, %r12  // km
	movq	ARG12, %r13  // km

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_8x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_8x4_vs_lib4
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgetrf_nn_l_8x4_vs_lib4, .-kernel_dgetrf_nn_l_8x4_vs_lib4
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
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
#elif defined(OS_MAC)
LC03: // { 7.5 6.5 5.5 4.5 }
	.align 5
#endif
	.long	0
	.long	1074921472
	.long	0
	.long	1075183616
	.long	0
	.long	1075445760
	.long	0
	.long	1075707904

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




#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

