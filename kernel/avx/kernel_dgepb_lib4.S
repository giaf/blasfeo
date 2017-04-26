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





//                               1      2          3        4          5          6
// void kernel_dger4_sub_8c_lib4(int m, double *A, int sda, double *B, double *C, int sdc)

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dger4_sub_8c_lib4
	.type kernel_dger4_sub_8c_lib4, @function
kernel_dger4_sub_8c_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dger4_sub_8c_lib4
_kernel_dger4_sub_8c_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dger4_sub_8c_lib4
	.def kernel_dger4_sub_8c_lib4; .scl 2; .type 32; .endef
kernel_dger4_sub_8c_lib4:
#endif
	
	PROLOGUE

	movq	ARG1, %r10 // m
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // C
	movq	ARG6, %r15 // sdc
	sall	$5, %r15d // 4*sdc*sizeof(double)

	cmpl	$0, %r10d
	jle		0f // return

	cmpl	$3, %r10d
	jle		2f // cleanup loop

	// main loop
	.p2align 3
1:
	vmovapd			0(%r11), %ymm12
	vbroadcastsd	0(%r13), %ymm13
	subl	$4, %r10d
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			0(%r14), %ymm0
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			32(%r14), %ymm1
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			64(%r14), %ymm2
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			96(%r14), %ymm3
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	128(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			128(%r14), %ymm4
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	160(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			160(%r14), %ymm5
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	192(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			192(%r14), %ymm6
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	224(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			224(%r14), %ymm7
	vsubpd			%ymm15, %ymm7, %ymm7
	prefetcht0		0(%r11, %r12, 1)

	vmovapd			32(%r11), %ymm12
	vbroadcastsd	8(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	40(%r13), %ymm13
	prefetcht0		0(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	72(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r13), %ymm13
	prefetcht0		64(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	136(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	168(%r13), %ymm13
	prefetcht0		128(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	200(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	232(%r13), %ymm13
	prefetcht0		192(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	vmovapd			64(%r11), %ymm12
	vbroadcastsd	16(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	48(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	80(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	112(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	144(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	176(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	208(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	240(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	prefetcht0		64(%r11, %r12, 1)

	vmovapd			96(%r11), %ymm12
	vbroadcastsd	24(%r13), %ymm13
	addq	%r12, %r11
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmovapd			%ymm0, 0(%r14)
	vbroadcastsd	56(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmovapd			%ymm1, 32(%r14)
	vbroadcastsd	88(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmovapd			%ymm2, 64(%r14)
	vbroadcastsd	120(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmovapd			%ymm3, 96(%r14)
	vbroadcastsd	152(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmovapd			%ymm4, 128(%r14)
	vbroadcastsd	184(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmovapd			%ymm5, 160(%r14)
	vbroadcastsd	216(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmovapd			%ymm6, 192(%r14)
	vbroadcastsd	248(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmovapd			%ymm7, 224(%r14)
	addq	%r15, %r14
	
	cmpl	$3, %r10d
	jg		1b // main loop

	cmpl	$0, %r10d
	jle		0f // return

	// cleanup loop
2:
	vcvtsi2sd	%r10d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC00(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC00(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm14

	vmovapd			0(%r11), %ymm12
	vbroadcastsd	0(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			0(%r14), %ymm0
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			32(%r14), %ymm1
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			64(%r14), %ymm2
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			96(%r14), %ymm3
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	128(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			128(%r14), %ymm4
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	160(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			160(%r14), %ymm5
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	192(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			192(%r14), %ymm6
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	224(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			224(%r14), %ymm7
	vsubpd			%ymm15, %ymm7, %ymm7

	vmovapd			32(%r11), %ymm12
	vbroadcastsd	8(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	40(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	72(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	136(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	168(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	200(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	232(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	vmovapd			64(%r11), %ymm12
	vbroadcastsd	16(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	48(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	80(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	112(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	144(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vbroadcastsd	176(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vbroadcastsd	208(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vbroadcastsd	240(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7

	vmovapd			96(%r11), %ymm12
	vbroadcastsd	24(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmaskmovpd		%ymm0, %ymm14, 0(%r14)
	vbroadcastsd	56(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmaskmovpd		%ymm1, %ymm14, 32(%r14)
	vbroadcastsd	88(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmaskmovpd		%ymm2, %ymm14, 64(%r14)
	vbroadcastsd	120(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmaskmovpd		%ymm3, %ymm14, 96(%r14)
	vbroadcastsd	152(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm4, %ymm4
	vmaskmovpd		%ymm4, %ymm14, 128(%r14)
	vbroadcastsd	184(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm5, %ymm5
	vmaskmovpd		%ymm5, %ymm14, 160(%r14)
	vbroadcastsd	216(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm6, %ymm6
	vmaskmovpd		%ymm6, %ymm14, 192(%r14)
	vbroadcastsd	248(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm7, %ymm7
	vmaskmovpd		%ymm7, %ymm14, 224(%r14)
	
	// return
0:

	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dger4_sub_4c_lib4, .-kernel_dger4_sub_4c_lib4
#endif





//                               1      2          3        4          5          6
// void kernel_dger4_sub_4c_lib4(int m, double *A, int sda, double *B, double *C, int sdc)

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dger4_sub_4c_lib4
	.type kernel_dger4_sub_4c_lib4, @function
kernel_dger4_sub_4c_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dger4_sub_4c_lib4
_kernel_dger4_sub_4c_lib4:
#elif defined(OS_WINDOWS)
	.globl kernel_dger4_sub_4c_lib4
	.def kernel_dger4_sub_4c_lib4; .scl 2; .type 32; .endef
kernel_dger4_sub_4c_lib4:
#endif
	
	PROLOGUE

	movq	ARG1, %r10 // m
	movq	ARG2, %r11 // A
	movq	ARG3, %r12 // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movq	ARG4, %r13 // B
	movq	ARG5, %r14 // C
	movq	ARG6, %r15 // sdc
	sall	$5, %r15d // 4*sdc*sizeof(double)

	cmpl	$0, %r10d
	jle		0f // return

	cmpl	$3, %r10d
	jle		2f // cleanup loop

	// main loop
	.p2align 3
1:
	vmovapd			0(%r11), %ymm12
	vbroadcastsd	0(%r13), %ymm13
	subl	$4, %r10d
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			0(%r14), %ymm0
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			32(%r14), %ymm1
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			64(%r14), %ymm2
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vmovapd			96(%r14), %ymm3
	vsubpd			%ymm15, %ymm3, %ymm3
	prefetcht0		0(%r11, %r12, 1)

	vmovapd			32(%r11), %ymm12
	vbroadcastsd	8(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	40(%r13), %ymm13
	prefetcht0		0(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	72(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r13), %ymm13
	prefetcht0		32(%r14, %r15, 1)
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	vmovapd			64(%r11), %ymm12
	vbroadcastsd	16(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	48(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	80(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	112(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	prefetcht0		32(%r11, %r12, 1)

	vmovapd			96(%r11), %ymm12
	vbroadcastsd	24(%r13), %ymm13
	addq	%r12, %r11
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vmovapd			%ymm0, 0(%r14)
	vbroadcastsd	56(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vmovapd			%ymm1, 32(%r14)
	vbroadcastsd	88(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vmovapd			%ymm2, 64(%r14)
	vbroadcastsd	120(%r13), %ymm13
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vmovapd			%ymm3, 96(%r14)
	addq	%r15, %r14
	
	cmpl	$3, %r10d
	jg		1b // main loop

	cmpl	$0, %r10d
	jle		0f // return

	// cleanup loop
2:
	vcvtsi2sd	%r10d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC00(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC00(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm13

	vmovapd			0(%r11), %ymm0
	vmovapd			32(%r11), %ymm1
	vmovapd			64(%r11), %ymm2
	vmovapd			96(%r11), %ymm3

	vmovapd			0(%r14), %ymm4
	vbroadcastsd	0(%r13), %ymm15
	vmulpd			%ymm0, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	8(%r13), %ymm15
	vmulpd			%ymm1, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	16(%r13), %ymm15
	vmulpd			%ymm2, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	24(%r13), %ymm15
	vmulpd			%ymm3, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vmaskmovpd		%ymm4, %ymm13, 0(%r14)

	vmovapd			32(%r14), %ymm4
	vbroadcastsd	32(%r13), %ymm15
	vmulpd			%ymm0, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	40(%r13), %ymm15
	vmulpd			%ymm1, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	48(%r13), %ymm15
	vmulpd			%ymm2, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	56(%r13), %ymm15
	vmulpd			%ymm3, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vmaskmovpd		%ymm4, %ymm13, 32(%r14)

	vmovapd			64(%r14), %ymm4
	vbroadcastsd	64(%r13), %ymm15
	vmulpd			%ymm0, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	72(%r13), %ymm15
	vmulpd			%ymm1, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	80(%r13), %ymm15
	vmulpd			%ymm2, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	88(%r13), %ymm15
	vmulpd			%ymm3, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vmaskmovpd		%ymm4, %ymm13, 64(%r14)

	vmovapd			96(%r14), %ymm4
	vbroadcastsd	96(%r13), %ymm15
	vmulpd			%ymm0, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	104(%r13), %ymm15
	vmulpd			%ymm1, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	112(%r13), %ymm15
	vmulpd			%ymm2, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vbroadcastsd	120(%r13), %ymm15
	vmulpd			%ymm3, %ymm15, %ymm14
	vsubpd			%ymm14, %ymm4, %ymm4
	vmaskmovpd		%ymm4, %ymm13, 96(%r14)

	// return
0:

	EPILOGUE
	
	ret

#if defined(OS_LINUX)
	.size	kernel_dger4_sub_4c_lib4, .-kernel_dger4_sub_4c_lib4
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
.LC00:
#elif defined(OS_MAC)
LC00:
	.align 5
#endif
	.double 0.5
	.double 1.5
	.double 2.5
	.double 3.5

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC01:
#elif defined(OS_MAC)
LC01:
	.align 5
#endif
	.double 4.5
	.double 5.5
	.double 6.5
	.double 7.5

#if defined(OS_LINUX) | defined(OS_WINDOWS)
	.align 32
.LC02:
#elif defined(OS_MAC)
LC02:
	.align 5
#endif
	.double 8.5
	.double 9.5
	.double 10.5
	.double 11.5





#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif


