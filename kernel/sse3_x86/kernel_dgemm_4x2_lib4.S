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
#define ARG11 STACKSIZE + 44(%esp)

#if 1

#define PROLOGUE \
	subl	$16, %esp; \
	movl	%ebx, 0(%esp); \
	movl	%esi, 4(%esp); \
	movl	%edi, 8(%esp); \
	movl	%ebp, 12(%esp);
#define EPILOGUE \
	movl	0(%esp), %ebx; \
	movl	4(%esp), %esi; \
	movl	8(%esp), %edi; \
	movl	12(%esp), %ebp; \
	addl	$16, %esp;

#else

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

#endif

#else

#error wrong OS

#endif



	.text



// common inner routine with file scope
//
// input arguments:
// eax   <- k
// ebx   <- A
// ecx   <- B
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4
#else
	.align 16
	.type inner_kernel_gemm_add_nt_4x2_lib4, @function
inner_kernel_gemm_add_nt_4x2_lib4:
#endif

	cmpl	$0, %eax
	jle		2f // return

	// preload

	cmpl	$4, %eax
	jle		0f // consider clean-up loop

	// main loop
	.align 8
1: // main loop

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 1
	movddup		32(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 2
	movddup		64(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		72(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 3
	movddup		96(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		104(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2


	subl	$4, %eax
	addl	$128, %ecx
	addl	$128, %ebx

	cmpl	$4, %eax
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %eax
	jle		4f // clean1

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 1
	movddup		32(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 2
	movddup		64(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		72(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 3
	movddup		96(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		104(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

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
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

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

	.size	inner_kernel_gemm_add_nt_4x2_lib4, .-inner_kernel_gemm_add_nt_4x2_lib4
#endif





// common inner routine with file scope
//
// input arguments:
// eax  <- k
// ebx   <- A
// ecx   <- B
// edx   <- 4*sdb*sizeof(double)
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4
#else
	.p2align 4,,15
	.type inner_kernel_gemm_add_nn_4x2_lib4, @function
inner_kernel_gemm_add_nn_4x2_lib4:
#endif

	cmpl	$0, %eax
	jle		2f // return

	// preload

	cmpl	$4, %eax
	jle		0f // consider clean-up loop

	// main loop
	.align 8
1: // main loop

	prefetcht0	0(%ecx, %edx, 1) // software prefetch

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		32(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 1
	movddup		8(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 2
	movddup		16(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		48(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 3
	movddup		24(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		56(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	subl	$4, %eax
	addl	$128, %ebx
	addl	%edx, %ecx

	cmpl	$4, %eax
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %eax
	jle		4f // clean1

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		32(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 1
	movddup		8(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 2
	movddup		16(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		48(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	// unroll 3
	movddup		24(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		56(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	subl	$4, %eax
	addl	$128, %ebx
	addl	%edx, %ecx

//	cmpl	$3, %eax
	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %eax
	jle		2f // return

	// clean-up loop
3: // clean up loop

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		32(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	subl	$1, %eax
	addl	$32, %ebx
	addl	$8, %ecx

	cmpl	$0, %eax
	jg		3b // clean up loop

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nn_4x2_lib4, .-inner_kernel_gemm_add_nn_4x2_lib4
#endif





// common inner routine with file scope
//
// input arguments:
// eax   <- k
// ebx   <- A
// ecx   <- B
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_SUB_NT_4X2_LIB4
#else
	.align 16
	.type inner_kernel_gemm_sub_nt_4x2_lib4, @function
inner_kernel_gemm_sub_nt_4x2_lib4:
#endif

	cmpl	$0, %eax
	jle		2f // return

	// preload

	cmpl	$4, %eax
	jle		0f // consider clean-up loop

	// main loop
	.align 8
1: // main loop

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 1
	movddup		32(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 2
	movddup		64(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		72(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 3
	movddup		96(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		104(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2


	subl	$4, %eax
	addl	$128, %ecx
	addl	$128, %ebx

	cmpl	$4, %eax
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %eax
	jle		4f // clean1

	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 1
	movddup		32(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		32(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		40(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		48(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 2
	movddup		64(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		64(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		72(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		80(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

	// unroll 3
	movddup		96(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		96(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		104(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		112(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

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
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm0
	movddup		8(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	subpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	subpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	subpd		%xmm4, %xmm2

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

	.size	inner_kernel_gemm_sub_nt_4x2_lib4, .-inner_kernel_gemm_sub_nt_4x2_lib4
#endif





// common inner routine with file scope
//
// edge for B unaligned, A aligned
//
// input arguments:
// eax   <- k
// ebx   <- A
// ecx   <- B
// edx   <- bs*sdb*sizeof(double)
// esi   <- offB
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]

//
// output arguments:


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_GEMM_ADD_NN_4X2_LIB4
#else
	.p2align 4,,15
	.type inner_edge_gemm_add_nn_4x2_lib4, @function
inner_edge_gemm_add_nn_4x2_lib4:
#endif

	cmpl			$0, %esi				// offset==0
	jle				2f						// end

	cmpl			$0, %eax				// k==0
	jle				2f						// end

	movl			%esi, %edi				// load offsetB
	sall			$3, %edi				// offsetB*sizeof(double)
	addl			%edi, %ecx				// B+offsetB*sizeof(double)

	movl			$4, %edi				// load 4
	subl			%esi, %edi				// 4-offsetB
	cmpl			%eax, %edi				// k > 4-offsetB
	cmovgl			%eax, %edi				// kend=min(k,4-offsetB)

1:
	// unroll 0
	movddup		0(%ecx), %xmm5 // B0
	movapd		%xmm5, %xmm6 // B0
	movapd		0(%ebx), %xmm4 // A0
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0
	movddup		32(%ecx), %xmm5 // B1
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1
	movapd		16(%ebx), %xmm4 // A2
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3
	mulpd		%xmm6, %xmm4
	addpd		%xmm4, %xmm2

	subl			$1, %eax				// k=-1
	subl			$1, %edi				// k_panel=-1
	addl			$32, %ebx				// A=+bs
	addl			$8, %ecx				// B=+1

	cmpl			$0, %edi				// if k_panel=0
	jg				1b						// loop 1

	cmpl			$0, %eax				// if k=0
	jle				2f						// end

	addl			%edx, %ecx				// B=Boff+sdb*bs
	subl			$32, %ecx				// B-=4*sizeof(double) (loop+offsetB)

2:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_edge_gemm_add_nn_4x2_lib4, .-inner_edge_gemm_add_nn_4x2_lib4
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// eax  <- E
// ebx  <- inv_diag_E
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_RLT_INV_4X2_LIB4
#else
	.align 16
	.type inner_edge_trsm_rlt_inv_4x2_lib4, @function
inner_edge_trsm_rlt_inv_4x2_lib4:
#endif
	
	movddup			0(%ebx), %xmm7
	mulpd			%xmm7, %xmm0
	mulpd			%xmm7, %xmm2

	movddup			8(%eax), %xmm7
	movapd			%xmm7, %xmm6
	mulpd			%xmm0, %xmm7
	mulpd			%xmm2, %xmm6
	subpd			%xmm7, %xmm1
	subpd			%xmm6, %xmm3
	movddup			8(%ebx), %xmm7
	mulpd			%xmm7, %xmm1
	mulpd			%xmm7, %xmm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_edge_trsm_rlt_inv_4x2_lib4, .-inner_edge_trsm_rlt_inv_4x2_lib4
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// eax   <- alpha
// ebx   <- beta
// ecx   <- C
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X2_LIB4
#else
	.align 16
	.type inner_scale_ab_4x2_lib4, @function
inner_scale_ab_4x2_lib4:
#endif

	// alpha
	movddup		0(%eax), %xmm7

	mulpd		%xmm7, %xmm0
	mulpd		%xmm7, %xmm1
	mulpd		%xmm7, %xmm2
	mulpd		%xmm7, %xmm3

	// beta
	movddup		0(%ebx), %xmm6

	xorpd		%xmm7, %xmm7 // 0.0

	ucomisd		%xmm7, %xmm6 // beta==0.0 ?
	je			0f // end

	movapd		0(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm0
	movapd		16(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm2
	movapd		32(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm1
	movapd		48(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_ab_4x2_lib4, .-inner_scale_ab_4x2_lib4
#endif





// common inner routine with file scope
//
// blend for generic alpha=1 and beta=1
//
// input arguments:
// eax   <- C
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_4X2_LIB4
#else
	.align 16
	.type inner_scale_11_4x2_lib4, @function
inner_scale_11_4x2_lib4:
#endif

	movapd		0(%eax), %xmm7
	addpd		%xmm7, %xmm0
	movapd		16(%eax), %xmm7
	addpd		%xmm7, %xmm2
	movapd		32(%eax), %xmm7
	addpd		%xmm7, %xmm1
	movapd		48(%eax), %xmm7
	addpd		%xmm7, %xmm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_11_4x2_lib4, .-inner_scale_11_4x2_lib4
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// eax  <- D
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X2_LIB4
#else
	.align 16
	.type inner_store_4x2_lib4, @function
inner_store_4x2_lib4:
#endif

	movapd	%xmm0,  0(%eax)
	movapd	%xmm2, 16(%eax)
	movapd	%xmm1, 32(%eax)
	movapd	%xmm3, 48(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4x2_lib4, .-inner_store_4x2_lib4
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// eax  <- D
// ebx  <- m1
// ecx  <- n1
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X2_VS_LIB4
#else
	.align 16
	.type inner_store_4x2_vs_lib4, @function
inner_store_4x2_vs_lib4:
#endif

	cmpl		$2, %ebx
	jg			1f
	je			0f

	// km==1
	movsd		%xmm0,  0(%eax)
	cmpl		$2, %ecx
	jl			4f // end
	movsd		%xmm1, 32(%eax)
	je			4f // end

	jmp		4f

0:
	// km==2
	movapd		%xmm0,  0(%eax)
	cmpl		$2, %ecx
	jl			4f // end
	movapd		%xmm1, 32(%eax)
	je			4f // end

	jmp		4f

1:
	cmpl		$3, %ebx
	jg			2f

	// km==3
	movapd		%xmm0,   0(%eax)
	movsd		%xmm2,  16(%eax)
	cmpl		$2, %ecx
	jl			4f // end
	movapd		%xmm1,  32(%eax)
	movsd		%xmm3,  48(%eax)
	je			4f // end

	jmp		4f

2:
	// km==4
	movapd		%xmm0,   0(%eax)
	movapd		%xmm2,  16(%eax)
	cmpl		$2, %ecx
	jl			4f // end
	movapd		%xmm1,  32(%eax)
	movapd		%xmm3,  48(%eax)
	je			4f // end

4:



#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4x2_vs_lib4, .-inner_store_4x2_vs_lib4
#endif





//                               1      2              3          4          5             6          7
// void kernel_dgemm_nt_4x2_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.align 16
	.globl kernel_dgemm_nt_4x2_lib4
	.type kernel_dgemm_nt_4x2_lib4, @function
kernel_dgemm_nt_4x2_lib4:

	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner gemm kernel nt

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4
#else
	call inner_kernel_gemm_add_nt_4x2_lib4
#endif


	// call inner blend scale

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx // beta
	movl	ARG6, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X2_LIB4
#else
	call inner_scale_ab_4x2_lib4
#endif


	// store n

	movl	ARG7, %eax // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_LIB4
#else
	call inner_store_4x2_lib4
#endif

	EPILOGUE

	ret

	.size	kernel_dgemm_nt_4x2_lib4, .-kernel_dgemm_nt_4x2_lib4





//                                  1      2              3          4          5             6          7          8       9
// void kernel_dgemm_nt_4x2_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int m1, int n1);

	.align 16
	.globl kernel_dgemm_nt_4x2_vs_lib4
	.type kernel_dgemm_nt_4x2_vs_lib4, @function
kernel_dgemm_nt_4x2_vs_lib4:

	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner gemm kernel nt

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4
#else
	call inner_kernel_gemm_add_nt_4x2_lib4
#endif


	// call inner blend scale

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx // beta
	movl	ARG6, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X2_LIB4
#else
	call inner_scale_ab_4x2_lib4
#endif


	// store n

	movl	ARG7, %eax // D
	movl	ARG8, %ebx // m1
	movl	ARG9, %ecx // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_VS_LIB4
#else
	call inner_store_4x2_vs_lib4
#endif

	EPILOGUE

	ret

	.size	kernel_dgemm_nt_4x2_vs_lib4, .-kernel_dgemm_nt_4x2_vs_lib4





//                               1      2              3          4            5          6        7             8          9
// void kernel_dgemm_nn_4x2_lib4(int k, double *alpha, double *A, int offsetB, double *B, int sdb, double *beta, double *C, double *D);

	.align 16
	.globl kernel_dgemm_nn_4x2_lib4
	.type kernel_dgemm_nn_4x2_lib4, @function
kernel_dgemm_nn_4x2_lib4:

	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner gemm kernel nn

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG5, %ecx  // B
	movl	ARG6, %edx // sdb
	sall	$5, %edx // 4*sdb*sizeof(double)
	movl	ARG4, %esi // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_GEMM_ADD_NN_4X2_LIB4
#else
	call inner_edge_gemm_add_nn_4x2_lib4
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4
#else
	call inner_kernel_gemm_add_nn_4x2_lib4
#endif


	// call inner blend

	movl	ARG2, %eax // alpha
	movl	ARG7, %ebx // beta
	movl	ARG8, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X2_LIB4
#else
	call inner_scale_ab_4x2_lib4
#endif


	// store n

	movl	ARG9, %eax // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_LIB4
#else
	call inner_store_4x2_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemm_nn_4x2_lib4, .-kernel_dgemm_nn_4x2_lib4





//                                  1      2              3          4            5          6        7             8          9          10      11
// void kernel_dgemm_nn_4x2_vs_lib4(int k, double *alpha, double *A, int offsetB, double *B, int sdb, double *beta, double *C, double *D, int m1, int n1);

	.align 16
	.globl kernel_dgemm_nn_4x2_vs_lib4
	.type kernel_dgemm_nn_4x2_vs_lib4, @function
kernel_dgemm_nn_4x2_vs_lib4:

	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner gemm kernel nn

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG5, %ecx  // B
	movl	ARG6, %edx // sdb
	sall	$5, %edx // 4*sdb*sizeof(double)
	movl	ARG4, %esi // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_GEMM_ADD_NN_4X2_LIB4
#else
	call inner_edge_gemm_add_nn_4x2_lib4
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4
#else
	call inner_kernel_gemm_add_nn_4x2_lib4
#endif


	// call inner blend

	movl	ARG2, %eax // alpha
	movl	ARG7, %ebx // beta
	movl	ARG8, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X2_LIB4
#else
	call inner_scale_ab_4x2_lib4
#endif


	// store n

	movl	ARG9, %eax // D
	movl	ARG10, %ebx // m1
	movl	ARG11, %ecx // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_VS_LIB4
#else
	call inner_store_4x2_vs_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemm_nn_4x2_vs_lib4, .-kernel_dgemm_nn_4x2_vs_lib4





//                                      1      2          3          4          5          6          7
// void kernel_dtrsm_nt_rl_inv_4x2_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.align 16
	.globl kernel_dtrsm_nt_rl_inv_4x2_lib4
	.type kernel_dtrsm_nt_rl_inv_4x2_lib4, @function
kernel_dtrsm_nt_rl_inv_4x2_lib4:
	
	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner dgemm kernel nt 

	movl	ARG1, %eax
	movl	ARG2, %ebx
	movl	ARG3, %ecx

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_4X2_LIB4
#else
	call inner_kernel_gemm_sub_nt_4x2_lib4
#endif


	// call inner blender_loader nn

	movl	ARG4, %eax

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X2_LIB4
#else
	call inner_scale_11_4x2_lib4
#endif


	// solve

	movl	ARG6, %eax  // E 
	movl	ARG7, %ebx  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X2_LIB4
#else
	call inner_edge_trsm_rlt_inv_4x2_lib4
#endif


	// store

	movl	ARG5, %eax // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_LIB4
#else
	call inner_store_4x2_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dtrsm_nt_rl_inv_4x2_lib4, .-kernel_dtrsm_nt_rl_inv_4x2_lib4





//                                         1      2          3          4          5          6          7                   8       9
// void kernel_dtrsm_nt_rl_inv_4x2_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int m1, int n1);

	.align 16
	.globl kernel_dtrsm_nt_rl_inv_4x2_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_4x2_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_4x2_vs_lib4:
	
	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner dgemm kernel nt 

	movl	ARG1, %eax
	movl	ARG2, %ebx
	movl	ARG3, %ecx

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_4X2_LIB4
#else
	call inner_kernel_gemm_sub_nt_4x2_lib4
#endif


	// call inner blender_loader nn

	movl	ARG4, %eax

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X2_LIB4
#else
	call inner_scale_11_4x2_lib4
#endif


	// solve

	movl	ARG6, %eax  // E 
	movl	ARG7, %ebx  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X2_LIB4
#else
	call inner_edge_trsm_rlt_inv_4x2_lib4
#endif


	// store

	movl	ARG5, %eax // D
	movl	ARG8, %ebx // m1
	movl	ARG9, %ecx // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_VS_LIB4
#else
	call inner_store_4x2_vs_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dtrsm_nt_rl_inv_4x2_vs_lib4, .-kernel_dtrsm_nt_rl_inv_4x2_vs_lib4





	.section	.note.GNU-stack,"",@progbits

