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





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// eax   <- alpha
// ebx   <- beta
// ecx   <- C
// edx   <- ldc
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X2_LIB
#else
	.align 16
	.type inner_scale_ab_4x2_lib, @function
inner_scale_ab_4x2_lib:
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

	movupd		0(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm0
	movupd		16(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm2
	addl		%edx, %ecx
	movupd		0(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm1
	movupd		16(%ecx), %xmm7
	mulpd		%xmm6, %xmm7
	addpd		%xmm7, %xmm3

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_ab_4x2_lib, .-inner_scale_ab_4x2_lib
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// eax  <- D
// ebx  <- ldd
// xmm0 <- [d00 d10]
// xmm1 <- [d01 d11]
// xmm2 <- [d20 d30]
// xmm3 <- [d21 d31]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X2_LIB
#else
	.align 16
	.type inner_store_4x2_lib, @function
inner_store_4x2_lib:
#endif

	movupd	%xmm0,  0(%eax)
	movupd	%xmm2, 16(%eax)
	addl	%ebx, %eax
	movupd	%xmm1,  0(%eax)
	movupd	%xmm3, 16(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4x2_lib, .-inner_store_4x2_lib
#endif





//                                 1      2              3          4          5             6          7        8          9
// void kernel_dgemm_nt_4x2_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.align 16
	.globl kernel_dgemm_nt_4x2_lib44c
	.type kernel_dgemm_nt_4x2_lib44c, @function
kernel_dgemm_nt_4x2_lib44c:

	PROLOGUE

	// zero accumulation registers

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


	// call inner dgemm kernel nn

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_NT_4X2_LIB4
#else
	call inner_kernel_gemm_nt_4x2_lib4
#endif


	// call inner blend

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx // beta
	movl	ARG6, %ecx   // C
	movl	ARG7, %edx   // ldc
	sall	$3, %edx

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X2_LIB
#else
	call inner_scale_ab_4x2_lib
#endif


	// store n

	movl	ARG8, %eax // D
	movl	ARG9, %ebx // ldd
	sall	$3, %ebx

#if MACRO_LEVEL>=1
	INNER_STORE_4X2_LIB
#else
	call inner_store_4x2_lib
#endif


	EPILOGUE

	ret

	.size	kernel_dgemm_nt_4x2_lib44c, .-kernel_dgemm_nt_4x2_lib44c






