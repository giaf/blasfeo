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

#if defined(OS_LINUX)

#define STACKSIZE 16
#define ARG1  STACKSIZE +  4(%esp)
#define ARG2  STACKSIZE +  8(%esp)
#define ARG3  STACKSIZE + 12(%esp)
#define ARG4  STACKSIZE + 16(%esp)
#define ARG5  STACKSIZE + 20(%esp)
#define ARG6  STACKSIZE + 24(%esp)
#define ARG7  STACKSIZE + 28(%esp)
#define ARG8  STACKSIZE + 32(%esp)
#define ARG9  STACKSIZE + 36(%esp)
#define ARG10 STACKSIZE + 40(%esp)
#define PROLOGUE \
	pushl	%ebp; \
	pushl	%edi; \
	pushl	%esi; \
	pushl	%ebx;
#define EPILOGUE \
	popl	%ebx; \
	popl	%esi; \
	popl	%edi; \
	popl	%ebp;

#else

#error wrong OS

#endif



#if defined(OS_LINUX)
	.text
#endif

// common inner routine with file scope
//
// input arguments:
// eax   <- k
// ebx   <- A
// ecx   <- B
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
	.align 16
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
#endif

	cmpl	$0, %eax
	jle		2f // return

	// preload

	cmpl	$4, %eax
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop

	// unroll 0
	vmovapd 		0(%ebx), %ymm4 // A
	vbroadcastsd	0(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	8(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	16(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	24(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 1
	vmovapd 		32(%ebx), %ymm4 // A
	vbroadcastsd	32(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	40(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	48(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	56(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 2
	vmovapd 		64(%ebx), %ymm4 // A
	vbroadcastsd	64(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	72(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	80(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	88(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 3
	vmovapd 		96(%ebx), %ymm4 // A
	vbroadcastsd	96(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	104(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	112(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	120(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	subl	$4, %eax
	addl	$128, %ecx
	addl	$128, %ebx

	cmpl	$4, %eax
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %eax
	jle		4f // clean1

	// unroll 0
	vmovapd 		0(%ebx), %ymm4 // A
	vbroadcastsd	0(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	8(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	16(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	24(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 1
	vmovapd 		32(%ebx), %ymm4 // A
	vbroadcastsd	32(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	40(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	48(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	56(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 2
	vmovapd 		64(%ebx), %ymm4 // A
	vbroadcastsd	64(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	72(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	80(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	88(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	// unroll 3
	vmovapd 		96(%ebx), %ymm4 // A
	vbroadcastsd	96(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	104(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	112(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	120(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	subl	$4, %eax
	addl	$128, %ecx
	addl	$128, %ebx

//	cmpl	$3, %eax
	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %eax
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	vmovapd 		0(%ebx), %ymm4 // A
	vbroadcastsd	0(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm0, %ymm7, %ymm0
	vbroadcastsd	8(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm1, %ymm7, %ymm1
	vbroadcastsd	16(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm2, %ymm7, %ymm2
	vbroadcastsd	24(%ecx), %ymm5 // B
	vmulpd			%ymm4, %ymm5, %ymm7
	vaddpd			%ymm3, %ymm7, %ymm3

	subl	$1, %eax
	addl	$32, %ecx
	addl	$32, %ebx

	cmpl	$0, %eax
	jg		3b // clean up loop

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// eax   <- alpha
// ebx   <- beta
// ecx   <- C
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_4X4_LIB4
#else
	.p2align 4,,15
	.type inner_blend_scale_ab_4x4_lib4, @function
inner_blend_scale_ab_4x4_lib4:
#endif

	// alpha
	vbroadcastsd	0(%eax), %ymm7

	vmulpd		%ymm0, %ymm7, %ymm0
	vmulpd		%ymm1, %ymm7, %ymm1
	vmulpd		%ymm2, %ymm7, %ymm2
	vmulpd		%ymm3, %ymm7, %ymm3

	// beta
	vbroadcastsd	0(%ebx), %ymm6

	vxorpd		%ymm7, %ymm7, %ymm7 // 0.0

	vucomisd	%xmm7, %xmm6 // beta==0.0 ?
	je			0f // end

	vmovapd		0(%ecx), %ymm7
	vmulpd		%ymm7, %ymm6, %ymm7
	vaddpd		%ymm0, %ymm7, %ymm0
	vmovapd		32(%ecx), %ymm7
	vmulpd		%ymm7, %ymm6, %ymm7
	vaddpd		%ymm1, %ymm7, %ymm1
	vmovapd		64(%ecx), %ymm7
	vmulpd		%ymm7, %ymm6, %ymm7
	vaddpd		%ymm2, %ymm7, %ymm2
	vmovapd		96(%ecx), %ymm7
	vmulpd		%ymm7, %ymm6, %ymm7
	vaddpd		%ymm3, %ymm7, %ymm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_blend_scale_ab_4x4_lib4, .-inner_blend_scale_ab_4x4_lib4
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// eax  <- D
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_LIB4
#else
	.p2align 4,,15
	.type inner_store_4x4_lib4, @function
inner_store_4x4_lib4:
#endif

	vmovapd	%ymm0,  0(%eax)
	vmovapd %ymm1, 32(%eax)
	vmovapd %ymm2, 64(%eax)
	vmovapd %ymm3, 96(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif





//                               1      2              3          4          5             6          7
// void kernel_dgemm_nt_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.align 16
	.globl kernel_dgemm_nt_4x4_lib4
	.type kernel_dgemm_nt_4x4_lib4, @function
kernel_dgemm_nt_4x4_lib4:

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx // beta
	movl	ARG6, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
	call inner_blend_scale_ab_4x4_lib4
#endif


	// store n

	movl	ARG7, %eax // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	call inner_store_4x4_lib4
#endif

	EPILOGUE

	ret

	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4





	// read-only data
	.section	.rodata.cst32,"aM",@progbits,32

	.align 32
.LC00: // { -1 -1 -1 1 }
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	1

	.align 32
.LC01: // { -1 -1 -1 -1 }
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	-1

	.align 32
.LC02: // { 3.5 2.5 1.5 0.5 }
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256

	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
	.long	0
	.long	1074921472
	.long	0
	.long	1075183616
	.long	0
	.long	1075445760
	.long	0
	.long	1075707904

	.align 32
.LC04: // { 1.0 1.0 1.0 1.0 }
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248

	.align 32
.LC05: // { 1.0 1.0 1.0 -1.0 }
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248

	.align 32
.LC06: // { 1.0 1.0 -1.0 -1.0 }
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248

	.align 32
.LC07: // { 1.0 -1.0 -1.0 -1.0 }
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248

	.align 32
.LC08: // { -1.0 -1.0 -1.0 1.0 }
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400

	.align 32
.LC09: // { -1.0 -1.0 1.0 1.0 }
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400

	.align 32
.LC10: // { -1.0 1.0 1.0 1.0 }
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400



	.section	.note.GNU-stack,"",@progbits

