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
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_lib, @function
inner_scale_ab_4x4_lib:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_scale_ab_4x4_lib; .scl 2; .type 32; .endef
inner_scale_ab_4x4_lib:
#endif
#endif
	
	// alpha
	vmovddup	0(%r10), %xmm15

	vmulpd	%xmm15, %xmm0, %xmm0
	vmulpd	%xmm15, %xmm1, %xmm1
	vmulpd	%xmm15, %xmm2, %xmm2
	vmulpd	%xmm15, %xmm3, %xmm3


	// beta
	vmovddup	0(%r11), %xmm14


	vmovapd		0(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm0
	vmovapd		16(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm1
	addq		%r13, %r12
	vmovapd		0(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm2
	vmovapd		16(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm3
	addq		%r13, %r12
	vmovapd		0(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm4
	vmovapd		16(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm5
	addq		%r13, %r12
	vmovapd		0(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm6
	vmovapd		16(%r12), %xmm15
	vfmadd231pd	%xmm14, %xmm15, %xmm7
//	addq		%r13, %r12

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_lib, .-inner_scale_ab_4x4_lib
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_lib, @function
inner_store_4x4_lib:
#elif defined(OS_MAC)
_inner_store_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_store_4x4_lib; .scl 2; .type 32; .endef
inner_store_4x4_lib:
#endif
#endif
	
	vmovapd %xmm0,  0(%r10)
	vmovapd %xmm1, 16(%r10)
	add		%r11, %r10
	vmovapd %xmm2,  0(%r10)
	vmovapd %xmm3, 16(%r10)
	add		%r11, %r10
	vmovapd %xmm4,  0(%r10)
	vmovapd %xmm5, 16(%r10)
	add		%r11, %r10
	vmovapd %xmm6,  0(%r10)
	vmovapd %xmm7, 16(%r10)
//	add		%r11, %r10
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib, .-inner_store_4x4_lib
#endif
#endif





//                                 1      2              3          4          5             6          7        8          9
// void kernel_dgemm_nt_4x4_lib44c(int k, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_lib44c
	.type kernel_dgemm_nt_4x4_lib44c, @function
kernel_dgemm_nt_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_lib44c
_kernel_dgemm_nt_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dgemm_nt_4x4_lib44c
	.def kernel_dgemm_nt_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dgemm_nt_4x4_lib44c:
#endif

	PROLOGUE

	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG5, %r11 // beta
	movq	ARG6, %r12   // C
	movq	ARG7, %r13   // ldc
	sall	$3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib
#endif
#endif


	// store n

	movq	ARG8, %r10 // D
	movq	ARG9, %r11 // ldd
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_store_4x4_lib
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib
#endif
#endif


	EPILOGUE

	ret

#if defined(OS_LINUX)
	.size	kernel_dgemm_nt_4x4_lib44c, .-kernel_dgemm_nt_4x4_lib44c
#endif






