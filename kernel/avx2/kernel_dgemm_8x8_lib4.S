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
// r14   <- 4*sdb*sizeof(double)
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
// r14   <- 4*sdb*sizeof(double)
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
	.macro INNER_KERNEL_DGEMM_ADD_NT_8X8_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_8x8_lib4, @function
inner_kernel_dgemm_add_nt_8x8_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_8x8_lib4:
#elif defined(OS_WINDOWS)
	.def inner_kernel_dgemm_add_nt_8x8_lib4; .scl 2; .type 32; .endef
inner_kernel_dgemm_add_nt_8x8_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	vmovapd 0(%r11), %ymm12          // A0[0]
	vmovapd 0(%r11, %r12, 1), %ymm13 // A1[0]
	vmovapd 0(%r13), %ymm14          // B0[0]
	vmovapd 0(%r13, %r14, 1), %ymm15 // B1[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	subl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		32(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		32(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		32(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		32(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 1
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		64(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		64(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		64(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		64(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 2
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		96(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		96(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		96(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		96(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 3
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	addq		$128, %r11
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	addq		$128, %r13
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		0(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		0(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		0(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		0(%r13, %r14, 1), %ymm15 // A1[4]


	cmpl	$4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	subl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		32(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		32(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		32(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		32(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 1
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		64(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		64(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		64(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		64(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 2
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	vmovapd		96(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vmovapd		96(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
	vmovapd		96(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		96(%r13, %r14, 1), %ymm15 // A1[4]

	// unroll 3
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	addq		$128, %r11
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	addq		$128, %r13
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
//	vmovapd		0(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm14, %ymm6
//	vmovapd		0(%r13), %ymm14 // B0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm10
//	vmovapd		0(%r11, %r12, 1), %ymm13 // A1[4]
//	vmovapd		0(%r13, %r14, 1), %ymm15 // A1[4]


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd		0(%r11), %ymm12 // A0[4]
	vmovapd		0(%r13), %ymm14 // B0[4]
	vmovapd		0(%r11, %r12, 1), %ymm13 // A1[4]
	vmovapd		0(%r13, %r14, 1), %ymm15 // A1[4]
	vfmadd231pd	%ymm12, %ymm14, %ymm0
	vfmadd231pd	%ymm13, %ymm14, %ymm4
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm8
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm1
	addq		$32, %r11
	vfmadd231pd	%ymm13, %ymm14, %ymm5
	vperm2f128	$0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm9
	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm3
	addq		$32, %r13
	vfmadd231pd	%ymm13, %ymm14, %ymm7
	vshufpd		$0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm15, %ymm11
	vshufpd		$0x5, %ymm15, %ymm15, %ymm15

	vfmadd231pd	%ymm12, %ymm14, %ymm2
	subl		$1, %r10d
	vfmadd231pd	%ymm13, %ymm14, %ymm6
	vfmadd231pd	%ymm13, %ymm15, %ymm10


	cmpl		$0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_8x8_lib4, .-inner_kernel_dgemm_add_nt_8x8_lib4
#endif
#endif





// common inner routine with file scope
//
// blend and scale for generic alpha and beta
//
// input arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_8X8_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_8x8_lib4, @function
inner_blend_scale_ab_8x8_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_8x8_lib4:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_ab_8x8_lib4; .scl 2; .type 32; .endef
inner_blend_scale_ab_8x8_lib4:
#endif
#endif
		

	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

	vbroadcastsd 0(%r11), %ymm14 // beta

	vxorpd		%ymm15, %ymm15, %ymm15 // 0.0

	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovapd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3

	vmovapd		0(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	vmovapd		32(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	vmovapd		64(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	vmovapd		96(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7

	vmovapd		128(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm8
	vmovapd		160(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm9
	vmovapd		192(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd		224(%r12, %r13, 1), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_8x8_lib4, .-inner_blend_scale_ab_8x8_lib4
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
//
// output arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x8_lib4, @function
inner_store_8x8_lib4:
#elif defined(OS_MAC)
_inner_store_8x8_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_8x8_lib4; .scl 2; .type 32; .endef
inner_store_8x8_lib4:
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

	vmovapd %ymm8,  128(%r10, %r11, 1)
	vmovapd %ymm9,  160(%r10, %r11, 1)
	vmovapd %ymm10, 192(%r10, %r11, 1)
	vmovapd %ymm11, 224(%r10, %r11, 1)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x8_lib4, .-inner_store_8x8_lib4
#endif
#endif





// common inner routine with file scope
//
// store lower n
//
// input arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d50 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X8_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x8_lib4, @function
inner_store_l_8x8_lib4:
#elif defined(OS_MAC)
_inner_store_l_8x8_lib4:
#elif defined(OS_WINDOWS)
	.def inner_store_l_8x8_lib4; .scl 2; .type 32; .endef
inner_store_l_8x8_lib4:
#endif
#endif
	
	vmovapd		%ymm0, 0(%r10)
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

	vmovapd		%ymm8, 128(%r10, %r11, 1)
	vmovapd		160(%r10, %r11, 1), %ymm14
	vblendpd	$0x1, %ymm14, %ymm9, %ymm9
	vmovapd		%ymm9, 160(%r10, %r11, 1)
	vmovapd		192(%r10, %r11, 1), %ymm14
	vblendpd	$0x3, %ymm14, %ymm10, %ymm10
	vmovapd		%ymm10, 192(%r10, %r11, 1)
	vmovapd		224(%r10, %r11, 1), %ymm14
	vblendpd	$0x7, %ymm14, %ymm11, %ymm11
	vmovapd		%ymm11, 224(%r10, %r11, 1)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_8x8_lib4, .-inner_store_l_8x8_lib4
#endif
#endif





//                               1      2              3          4        5          6        7             8          9        10         11
// void kernel_dgemm_nt_8x8_lib4(int k, double *alpha, double *A, int sda, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_8x8_lib4
	.type kernel_dgemm_nt_8x8_lib4, @function
kernel_dgemm_nt_8x8_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_8x8_lib4
_kernel_dgemm_nt_8x8_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_8x8_lib4
	.def kernel_dgemm_nt_8x8_lib4; .scl 2; .type 32; .endef
kernel_dgemm_nt_8x8_lib4:
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

	movq	ARG1, %r10 // k
	movq	ARG3, %r11 // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG5, %r13 // B
	movq	ARG6, %r14 // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X8_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_8x8_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_8x8_lib4
#endif
#endif


	// call inner blend scale

	movq	ARG2, %r10 // alpha
	movq	ARG7, %r11 // beta
	movq	ARG8, %r12 // C
	movq	ARG9, %r13 // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X8_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_8x8_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x8_lib4
#endif
#endif


	// store n

	movq	ARG10, %r10 // D
	movq	ARG11, %r11 // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X8_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_l_8x8_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_8x8_lib4
#endif
#endif


	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_8x8_lib4, .-kernel_dgemm_nt_8x8_lib4
#endif






