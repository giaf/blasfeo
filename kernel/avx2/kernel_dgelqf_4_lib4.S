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





//                                  1      2           3           4
// void kernel_dgelqf_dlarft_4_lib4(int n, double *pD, double *dD, double *pT, double *beta)

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgelqf_dlarft_4_lib4
	.type kernel_dgelqf_dlarft_4_lib4, @function
kernel_dgelqf_dlarft_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgelqf_dlarft_4_lib4
_kernel_dgelqf_dlarft_4_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dgelqf_dlarft_4_lib4
	.def kernel_dgelqf_dlarft_4_lib4; .scl 2; .type 32; .endef
kernel_dgelqf_dlarft_4_lib4:
#endif
	
	PROLOGUE

	// zero T

	movq	ARG4, %r10 // T

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			%ymm15, 0(%r10)
	vmovapd			%ymm15, 32(%r10)
	vmovapd			%ymm15, 64(%r10)
	vmovapd			%ymm15, 96(%r10)

	// first column

	movq	ARG2, %r11 // D
	movq	ARG3, %r12 // dD
	movq	ARG4, %r13 // T

	vxorpd			%xmm15, %xmm15, %xmm15
	movq	ARG1, %r10 // n
	subl	$1, %r10d
	addq	$32, %r11
100:
	vmovsd			0(%r11), %xmm14
	vfmadd231sd		%xmm14, %xmm14, %xmm15
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		100b

	vxorpd			%xmm14, %xmm14, %xmm14
	vucomisd		%xmm14, %xmm15
	jne		101f
//	jp		101f
	vmovsd			%xmm14, 0(%r12)
	jmp		102f

101:
	movq	ARG2, %r11 // D
	vmovsd			0(%r11), %xmm14 // alpha
	vfmadd231sd		%xmm14, %xmm14, %xmm15 // beta
	vsqrtsd			%xmm15, %xmm15, %xmm15 // beta
	vmovsd			.LC00(%rip), %xmm13 // mask
	vandpd			%xmm13, %xmm14, %xmm12
	vxorpd			%xmm13, %xmm12, %xmm12
	vxorpd			%xmm12, %xmm15, %xmm15 // beta
	vmovsd			%xmm15, 0(%r11) // pD[0+ps*0]
	vsubsd			%xmm14, %xmm15, %xmm14 // beta-alpha
	vmovapd			.LC01(%rip), %xmm12
	vmovsd			%xmm14, %xmm12, %xmm12
	vmovddup		%xmm14, %xmm14
	vmovsd			%xmm15, %xmm14, %xmm14
	vdivpd			%xmm14, %xmm12, %xmm14
	vmovsd			%xmm14, 0(%r12) // dD[0]
	vxorpd			%xmm13, %xmm14, %xmm12
	vmovsd			%xmm12, 0(%r13) // pT[0+ps*0]

	vpermpd			$0x55, %ymm14, %ymm15 // tmp

	vmovapd			0(%r11), %ymm0
	vbroadcastsd	32(%r11), %ymm8
	vbroadcastsd	64(%r11), %ymm9
	vbroadcastsd	96(%r11), %ymm10
	vmulpd			%ymm15, %ymm8, %ymm8
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vfmadd231pd		32(%r11), %ymm8, %ymm0
	vfmadd231pd		64(%r11), %ymm9, %ymm0
	vfmadd231pd		96(%r11), %ymm10, %ymm0
	vmovsd			%xmm8, 32(%r11)
	vmovsd			%xmm9, 64(%r11)
	vmovsd			%xmm10, 96(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		109f
103:
	vbroadcastsd	0(%r11), %ymm8
	vbroadcastsd	32(%r11), %ymm9
	vbroadcastsd	64(%r11), %ymm10
	vbroadcastsd	96(%r11), %ymm11
	vmulpd			%ymm15, %ymm8, %ymm8
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vmulpd			%ymm15, %ymm11, %ymm11
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vfmadd231pd		32(%r11), %ymm9, %ymm0
	vfmadd231pd		64(%r11), %ymm10, %ymm0
	vfmadd231pd		96(%r11), %ymm11, %ymm0
	vmovsd			%xmm8, 0(%r11)
	vmovsd			%xmm9, 32(%r11)
	vmovsd			%xmm10, 64(%r11)
	vmovsd			%xmm11, 96(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		103b
109:
	cmpl	$0, %r10d
	jle		104f
105:
	vbroadcastsd	0(%r11), %ymm8
	vmulpd			%ymm15, %ymm8, %ymm8
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vmovsd			%xmm8, 0(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		105b
104:

	vbroadcastsd	0(%r13), %ymm15
	vmulpd			%ymm15, %ymm0, %ymm0

	vxorpd			%ymm15, %ymm15, %ymm15
	vblendpd		$0x1, %ymm15, %ymm0, %ymm0

	movq	ARG2, %r11 // D
	vmovapd			0(%r11), %ymm8
	vmovapd			32(%r11), %ymm9
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vaddpd			%ymm0, %ymm8, %ymm8
	vbroadcastsd	32(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm9
	vbroadcastsd	64(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm10
	vbroadcastsd	96(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vmulpd			%ymm10, %ymm10, %ymm15
	vfmadd231pd		%ymm11, %ymm11, %ymm15
	vmovapd			%ymm8, 0(%r11)
	vmovapd			%ymm9, 32(%r11)
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		110f
106:
	vmovapd			0(%r11), %ymm8
	vmovapd			32(%r11), %ymm9
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vbroadcastsd	0(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vbroadcastsd	32(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm9
	vbroadcastsd	64(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm10
	vbroadcastsd	96(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vfmadd231pd		%ymm9, %ymm9, %ymm15
	vfmadd231pd		%ymm10, %ymm10, %ymm15
	vfmadd231pd		%ymm11, %ymm11, %ymm15
	vmovapd			%ymm8, 0(%r11)
	vmovapd			%ymm9, 32(%r11)
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		106b
110:
	cmpl	$0, %r10d
	jle		107f
108:
	vmovapd			0(%r11), %ymm8
	vbroadcastsd	0(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vmovapd			%ymm8, 0(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		108b
107:
	vpermpd	$0x55, %ymm15, %ymm15  // beta

	// second column
102:
	vxorpd			%xmm14, %xmm14, %xmm14
	vucomisd		%xmm14, %xmm15
	jne		101f
//	jp		111f
	vmovsd			%xmm14, 8(%r12)
	jmp		102f

101:
	movq	ARG2, %r11 // D
	vmovsd			40(%r11), %xmm14 // alpha
	vfmadd231sd		%xmm14, %xmm14, %xmm15 // beta
	vsqrtsd			%xmm15, %xmm15, %xmm15 // beta
	vmovsd			.LC00(%rip), %xmm13 // mask
	vandpd			%xmm13, %xmm14, %xmm12
	vxorpd			%xmm13, %xmm12, %xmm12
	vxorpd			%xmm12, %xmm15, %xmm15 // beta
	vmovsd			%xmm15, 40(%r11) // pD[0+ps*0]
	vsubsd			%xmm14, %xmm15, %xmm14 // beta-alpha
	vmovapd			.LC01(%rip), %xmm12
	vmovsd			%xmm14, %xmm12, %xmm12
	vmovddup		%xmm14, %xmm14
	vmovsd			%xmm15, %xmm14, %xmm14
	vdivpd			%xmm14, %xmm12, %xmm14
	vmovsd			%xmm14, 8(%r12) // dD[0]
	vxorpd			%xmm13, %xmm14, %xmm12
	vmovsd			%xmm12, 40(%r13) // pT[0+ps*0]

	vpermpd			$0x55, %ymm14, %ymm15 // tmp

	vmovapd			32(%r11), %ymm0
	vbroadcastsd	72(%r11), %ymm9
	vbroadcastsd	104(%r11), %ymm10
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vfmadd231pd		64(%r11), %ymm9, %ymm0
	vfmadd231pd		96(%r11), %ymm10, %ymm0
	vmovsd			%xmm9, 72(%r11)
	vmovsd			%xmm10, 104(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		109f
103:
	vbroadcastsd	8(%r11), %ymm8
	vbroadcastsd	40(%r11), %ymm9
	vbroadcastsd	72(%r11), %ymm10
	vbroadcastsd	104(%r11), %ymm11
	vmulpd			%ymm15, %ymm8, %ymm8
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vmulpd			%ymm15, %ymm11, %ymm11
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vfmadd231pd		32(%r11), %ymm9, %ymm0
	vfmadd231pd		64(%r11), %ymm10, %ymm0
	vfmadd231pd		96(%r11), %ymm11, %ymm0
	vmovsd			%xmm8, 8(%r11)
	vmovsd			%xmm9, 40(%r11)
	vmovsd			%xmm10, 72(%r11)
	vmovsd			%xmm11, 104(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		103b
109:
	cmpl	$0, %r10d
	jle		104f
105:
	vbroadcastsd	8(%r11), %ymm8
	vmulpd			%ymm15, %ymm8, %ymm8
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vmovsd			%xmm8, 8(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		105b
104:

	vmovapd			.LC02(%rip), %ymm12
	vmovapd			0(%r13), %ymm14
	vblendpd		$0x1, %ymm14, %ymm12, %ymm14
	vmulpd			%ymm14, %ymm0, %ymm0
	vbroadcastsd	40(%r13), %ymm15
	vmulpd			%ymm15, %ymm0, %ymm0
	vmovsd			%xmm0, 32(%r13)

	vxorpd			%ymm12, %ymm12, %ymm12
	vblendpd		$0x3, %ymm12, %ymm0, %ymm0

	movq	ARG2, %r11 // D
	vmovapd			32(%r11), %ymm9
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vaddpd			%ymm0, %ymm9, %ymm9
	vbroadcastsd	72(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm10
	vbroadcastsd	104(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vmulpd			%ymm11, %ymm11, %ymm15
	vmovapd			%ymm9, 32(%r11)
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		110f
106:
	vmovapd			0(%r11), %ymm8
	vmovapd			32(%r11), %ymm9
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vbroadcastsd	8(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vbroadcastsd	40(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm9
	vbroadcastsd	72(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm10
	vbroadcastsd	104(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vfmadd231pd		%ymm9, %ymm9, %ymm15
	vfmadd231pd		%ymm10, %ymm10, %ymm15
	vfmadd231pd		%ymm11, %ymm11, %ymm15
	vmovapd			%ymm8, 0(%r11)
	vmovapd			%ymm9, 32(%r11)
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		106b
110:
	cmpl	$0, %r10d
	jle		107f
108:
	vmovapd			0(%r11), %ymm8
	vbroadcastsd	8(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vmovapd			%ymm8, 0(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		108b
107:
	vpermpd	$0xaa, %ymm15, %ymm15  // beta

	// third column
102:
	vxorpd			%xmm14, %xmm14, %xmm14
	vucomisd		%xmm14, %xmm15
	jne		101f
//	jp		111f
	vmovsd			%xmm14, 16(%r12)
	jmp		102f

101:
	movq	ARG2, %r11 // D
	vmovsd			80(%r11), %xmm14 // alpha
	vfmadd231sd		%xmm14, %xmm14, %xmm15 // beta
	vsqrtsd			%xmm15, %xmm15, %xmm15 // beta
	vmovsd			.LC00(%rip), %xmm13 // mask
	vandpd			%xmm13, %xmm14, %xmm12
	vxorpd			%xmm13, %xmm12, %xmm12
	vxorpd			%xmm12, %xmm15, %xmm15 // beta
	vmovsd			%xmm15, 80(%r11) // pD[0+ps*0]
	vsubsd			%xmm14, %xmm15, %xmm14 // beta-alpha
	vmovapd			.LC01(%rip), %xmm12
	vmovsd			%xmm14, %xmm12, %xmm12
	vmovddup		%xmm14, %xmm14
	vmovsd			%xmm15, %xmm14, %xmm14
	vdivpd			%xmm14, %xmm12, %xmm14
	vmovsd			%xmm14, 16(%r12) // dD[0]
	vxorpd			%xmm13, %xmm14, %xmm12
	vmovsd			%xmm12, 80(%r13) // pT[0+ps*0]

	vpermpd			$0x55, %ymm14, %ymm15 // tmp

	vmovapd			64(%r11), %ymm0
	vbroadcastsd	112(%r11), %ymm10
	vmulpd			%ymm15, %ymm10, %ymm10
	vfmadd231pd		96(%r11), %ymm10, %ymm0
	vmovsd			%xmm10, 112(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		109f
103:
	vbroadcastsd	16(%r11), %ymm8
	vbroadcastsd	48(%r11), %ymm9
	vbroadcastsd	80(%r11), %ymm10
	vbroadcastsd	112(%r11), %ymm11
	vmulpd			%ymm15, %ymm8, %ymm8
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vmulpd			%ymm15, %ymm11, %ymm11
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vfmadd231pd		32(%r11), %ymm9, %ymm0
	vfmadd231pd		64(%r11), %ymm10, %ymm0
	vfmadd231pd		96(%r11), %ymm11, %ymm0
	vmovsd			%xmm8, 16(%r11)
	vmovsd			%xmm9, 48(%r11)
	vmovsd			%xmm10, 80(%r11)
	vmovsd			%xmm11, 112(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		103b
109:
	cmpl	$0, %r10d
	jle		104f
105:
	vbroadcastsd	16(%r11), %ymm8
	vmulpd			%ymm15, %ymm8, %ymm8
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vmovsd			%xmm8, 16(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		105b
104:

	vxorpd			%xmm12, %xmm12, %xmm12
	vmovapd			0(%r13), %ymm14
	vblendpd		$0x1, %ymm14, %ymm12, %ymm14
	vmulpd			%ymm14, %ymm0, %ymm1
	vmovapd			32(%r13), %ymm14
	vblendpd		$0x3, %ymm14, %ymm12, %ymm14
	vpermpd			$0x55, %ymm0, %ymm13
	vfmadd231pd		%ymm14, %ymm13, %ymm1
	vblendpd		$0x7, %ymm1, %ymm0, %ymm0
	vbroadcastsd	80(%r13), %ymm15
	vmulpd			%ymm15, %ymm0, %ymm0
	vmovapd			%xmm0, 64(%r13)

	vxorpd			%ymm12, %ymm12, %ymm12
	vblendpd		$0x7, %ymm12, %ymm0, %ymm0

	movq	ARG2, %r11 // D
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vaddpd			%ymm0, %ymm10, %ymm10
	vbroadcastsd	112(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		110f
106:
	vmovapd			0(%r11), %ymm8
	vmovapd			32(%r11), %ymm9
	vmovapd			64(%r11), %ymm10
	vmovapd			96(%r11), %ymm11
	vbroadcastsd	16(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vbroadcastsd	48(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm9
	vbroadcastsd	80(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm10
	vbroadcastsd	112(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm11
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vfmadd231pd		%ymm9, %ymm9, %ymm15
	vfmadd231pd		%ymm10, %ymm10, %ymm15
	vfmadd231pd		%ymm11, %ymm11, %ymm15
	vmovapd			%ymm8, 0(%r11)
	vmovapd			%ymm9, 32(%r11)
	vmovapd			%ymm10, 64(%r11)
	vmovapd			%ymm11, 96(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		106b
110:
	cmpl	$0, %r10d
	jle		107f
108:
	vmovapd			0(%r11), %ymm8
	vbroadcastsd	16(%r11), %ymm14
	vfmadd231pd		%ymm0, %ymm14, %ymm8
	vfmadd231pd		%ymm8, %ymm8, %ymm15
	vmovapd			%ymm8, 0(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		108b
107:
	vpermpd	$0xff, %ymm15, %ymm15  // beta

102:
	vxorpd			%xmm14, %xmm14, %xmm14
	vucomisd		%xmm14, %xmm15
	jne		101f
//	jp		111f
	vmovsd			%xmm14, 24(%r12)
	jmp		102f

101:
	movq	ARG2, %r11 // D
	vmovsd			120(%r11), %xmm14 // alpha
	vfmadd231sd		%xmm14, %xmm14, %xmm15 // beta
	vsqrtsd			%xmm15, %xmm15, %xmm15 // beta
	vmovsd			.LC00(%rip), %xmm13 // mask
	vandpd			%xmm13, %xmm14, %xmm12
	vxorpd			%xmm13, %xmm12, %xmm12
	vxorpd			%xmm12, %xmm15, %xmm15 // beta
	vmovsd			%xmm15, 120(%r11) // pD[0+ps*0]
	vsubsd			%xmm14, %xmm15, %xmm14 // beta-alpha
	vmovapd			.LC01(%rip), %xmm12
	vmovsd			%xmm14, %xmm12, %xmm12
	vmovddup		%xmm14, %xmm14
	vmovsd			%xmm15, %xmm14, %xmm14
	vdivpd			%xmm14, %xmm12, %xmm14
	vmovsd			%xmm14, 24(%r12) // dD[0]
	vxorpd			%xmm13, %xmm14, %xmm12
	vmovsd			%xmm12, 120(%r13) // pT[0+ps*0]

	vpermpd			$0x55, %ymm14, %ymm15 // tmp

	vmovapd			96(%r11), %ymm0
	movq	ARG1, %r10 // n
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jle		109f
103:
	vbroadcastsd	24(%r11), %ymm8
	vbroadcastsd	56(%r11), %ymm9
	vbroadcastsd	88(%r11), %ymm10
	vbroadcastsd	120(%r11), %ymm11
	vmulpd			%ymm15, %ymm8, %ymm8
	vmulpd			%ymm15, %ymm9, %ymm9
	vmulpd			%ymm15, %ymm10, %ymm10
	vmulpd			%ymm15, %ymm11, %ymm11
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vfmadd231pd		32(%r11), %ymm9, %ymm0
	vfmadd231pd		64(%r11), %ymm10, %ymm0
	vfmadd231pd		96(%r11), %ymm11, %ymm0
	vmovsd			%xmm8, 24(%r11)
	vmovsd			%xmm9, 56(%r11)
	vmovsd			%xmm10, 88(%r11)
	vmovsd			%xmm11, 120(%r11)
	subl	$4, %r10d
	addq	$128, %r11
	cmpl	$3, %r10d
	jg		103b
109:
	cmpl	$0, %r10d
	jle		104f
105:
	vbroadcastsd	24(%r11), %ymm8
	vmulpd			%ymm15, %ymm8, %ymm8
	vfmadd231pd		0(%r11), %ymm8, %ymm0
	vmovsd			%xmm8, 24(%r11)
	subl	$1, %r10d
	addq	$32, %r11
	cmpl	$0, %r10d
	jg		105b
104:

	vxorpd			%xmm12, %xmm12, %xmm12

	vmovapd			0(%r13), %ymm14
	vblendpd		$0x1, %ymm14, %ymm12, %ymm14
	vmulpd			%ymm14, %ymm0, %ymm1

	vmovapd			32(%r13), %ymm14
	vblendpd		$0x3, %ymm14, %ymm12, %ymm14
	vpermpd			$0x55, %ymm0, %ymm13
	vfmadd231pd		%ymm14, %ymm13, %ymm1

	vmovapd			64(%r13), %ymm14
	vblendpd		$0x7, %ymm14, %ymm12, %ymm14
	vpermpd			$0xaa, %ymm0, %ymm13
	vfmadd231pd		%ymm14, %ymm13, %ymm1

	vbroadcastsd	120(%r13), %ymm15
	vmulpd			%ymm15, %ymm1, %ymm1
	vmovapd			96(%r13), %ymm0
	vblendpd		$0x7, %ymm1, %ymm0, %ymm0
	vmovapd			%ymm0, 96(%r13)

102:

	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dgelqf_dlarft_4_lib4, .-kernel_dgelqf_dlarft_4_lib4
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
.LC00: // { 100...0 100...0 100...0 100...0 }
#elif defined(OS_MAC)
LC00: // { 100...0 100...0 100...0 100...0 }
	.align 5
#endif
	.long	0x00000000
	.long	0x80000000
	.long	0x00000000
	.long	0x80000000
	.long	0x00000000
	.long	0x80000000
	.long	0x00000000
	.long	0x80000000

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC01:
#elif defined(OS_MAC)
LC01:
	.align 5
#endif
	.double	-1.0
	.double	-1.0
	.double	-1.0
	.double	-1.0

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC02:
#elif defined(OS_MAC)
LC02:
	.align 5
#endif
	.double	1.0
	.double	1.0
	.double	1.0
	.double	1.0





#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

