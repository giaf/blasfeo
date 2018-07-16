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
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_8X4_LIB8
#else
	.align 16
	.type inner_kernel_gemm_add_nt_8x4_lib8, @function
inner_kernel_gemm_add_nt_8x4_lib8:
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
	vmovaps 		0(%ebx), %ymm4 // A
	vbroadcastss	0(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	4(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	8(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	12(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 1
	vmovaps 		32(%ebx), %ymm4 // A
	vbroadcastss	32(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	36(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	40(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	44(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 2
	vmovaps 		64(%ebx), %ymm4 // A
	vbroadcastss	64(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	68(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	72(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	76(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 3
	vmovaps 		96(%ebx), %ymm4 // A
	vbroadcastss	96(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	100(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	104(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	108(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	subl	$4, %eax
	addl	$128, %ecx
	addl	$128, %ebx

	cmpl	$4, %eax
	jg		1b // main loop


0: // consider clean4-up

	cmpl	$3, %eax
	jle		4f // clean1

	// unroll 0
	vmovaps 		0(%ebx), %ymm4 // A
	vbroadcastss	0(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	4(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	8(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	12(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 1
	vmovaps 		32(%ebx), %ymm4 // A
	vbroadcastss	32(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	36(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	40(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	44(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 2
	vmovaps 		64(%ebx), %ymm4 // A
	vbroadcastss	64(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	68(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	72(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	76(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

	// unroll 3
	vmovaps 		96(%ebx), %ymm4 // A
	vbroadcastss	96(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	100(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	104(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	108(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

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
	vmovaps 		0(%ebx), %ymm4 // A
	vbroadcastss	0(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm0, %ymm7, %ymm0
	vbroadcastss	4(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm1, %ymm7, %ymm1
	vbroadcastss	8(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm2, %ymm7, %ymm2
	vbroadcastss	12(%ecx), %ymm5 // B
	vmulps			%ymm4, %ymm5, %ymm7
	vaddps			%ymm3, %ymm7, %ymm3

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

	.size	inner_kernel_gemm_add_nt_8x4_lib8, .-inner_kernel_gemm_add_nt_8x4_lib8
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// eax   <- alpha
// ebx   <- beta
// ecx   <- C
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_LIB8
#else
	.align 16
	.type inner_scale_ab_8x4_lib8, @function
inner_scale_ab_8x4_lib8:
#endif

	// alpha
	vbroadcastss	0(%eax), %ymm7

	vmulps		%ymm0, %ymm7, %ymm0
	vmulps		%ymm1, %ymm7, %ymm1
	vmulps		%ymm2, %ymm7, %ymm2
	vmulps		%ymm3, %ymm7, %ymm3

	// beta
	vbroadcastss	0(%ebx), %ymm6

	vxorps		%ymm7, %ymm7, %ymm7 // 0.0

	vucomiss	%xmm7, %xmm6 // beta==0.0 ?
	je			0f // end

	vmovaps		0(%ecx), %ymm7
	vmulps		%ymm7, %ymm6, %ymm7
	vaddps		%ymm0, %ymm7, %ymm0
	vmovaps		32(%ecx), %ymm7
	vmulps		%ymm7, %ymm6, %ymm7
	vaddps		%ymm1, %ymm7, %ymm1
	vmovaps		64(%ecx), %ymm7
	vmulps		%ymm7, %ymm6, %ymm7
	vaddps		%ymm2, %ymm7, %ymm2
	vmovaps		96(%ecx), %ymm7
	vmulps		%ymm7, %ymm6, %ymm7
	vaddps		%ymm3, %ymm7, %ymm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_ab_8x4_lib8, .-inner_scale_ab_8x4_lib8
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// eax  <- D
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_LIB8
#else
	.align 16
	.type inner_store_8x4_lib8, @function
inner_store_8x4_lib8:
#endif

	vmovaps	%ymm0,  0(%eax)
	vmovaps %ymm1, 32(%eax)
	vmovaps %ymm2, 64(%eax)
	vmovaps %ymm3, 96(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_8x4_lib4, .-inner_store_8x4_lib8
#endif





//                               1      2              3          4          5             6          7
// void kernel_sgemm_nt_8x4_lib8(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.align 16
	.globl kernel_sgemm_nt_8x4_lib8
	.type kernel_sgemm_nt_8x4_lib8, @function
kernel_sgemm_nt_8x4_lib8:

	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner gemm kernel nt

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB8
#else
	call inner_kernel_gemm_add_nt_8x4_lib8
#endif


	// call inner blend scale

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx // beta
	movl	ARG6, %ecx   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB8
#else
	call inner_scale_ab_8x4_lib8
#endif


	// store n

	movl	ARG7, %eax // D

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB8
#else
	call inner_store_8x4_lib8
#endif

	EPILOGUE

	ret

	.size	kernel_sgemm_nt_8x4_lib8, .-kernel_sgemm_nt_8x4_lib8






