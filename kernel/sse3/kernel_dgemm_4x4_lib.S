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
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_lib, @function
inner_edge_dtrsm_rlt_inv_4x4_lib:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_edge_dtrsm_rlt_inv_4x4_lib; .scl 2; .type 32; .endef
inner_edge_dtrsm_rlt_inv_4x4_lib:
#endif
#endif
	
	movddup			0(%r12), %xmm13
	mulpd			%xmm13, %xmm0
	mulpd			%xmm13, %xmm4
	movddup			8(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm13
	mulpd			%xmm4, %xmm12
	subpd			%xmm13, %xmm1
	subpd			%xmm12, %xmm5
	movddup			16(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			24(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm0, %xmm12
	mulpd			%xmm4, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	addq	%r11, %r10

	movddup			8(%r12), %xmm13
	mulpd			%xmm13, %xmm1
	mulpd			%xmm13, %xmm5
	movddup			16(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm2
	subpd			%xmm13, %xmm6
	movddup			24(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm1, %xmm12
	mulpd			%xmm5, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	addq	%r11, %r10

	movddup			16(%r12), %xmm13
	mulpd			%xmm13, %xmm2
	mulpd			%xmm13, %xmm6
	movddup			24(%r10), %xmm13
	movapd			%xmm13, %xmm12
	mulpd			%xmm2, %xmm12
	mulpd			%xmm6, %xmm13
	subpd			%xmm12, %xmm3
	subpd			%xmm13, %xmm7
	addq	%r11, %r10

	movddup			24(%r12), %xmm13
	mulpd			%xmm13, %xmm3
	mulpd			%xmm13, %xmm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_lib, .-inner_edge_dtrsm_rlt_inv_4x4_lib
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
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_4X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_4x4_lib, @function
inner_blend_scale_ab_4x4_lib:
#elif defined(OS_MAC)
_inner_blend_scale_ab_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_ab_4x4_lib; .scl 2; .type 32; .endef
inner_blend_scale_ab_4x4_lib:
#endif
#endif

	movapd	%xmm0, %xmm8
	movsd	%xmm1, %xmm0
	movsd	%xmm8, %xmm1

	movapd	%xmm2, %xmm8
	movsd	%xmm3, %xmm2
	movsd	%xmm8, %xmm3

	movapd	%xmm4, %xmm8
	movsd	%xmm5, %xmm4
	movsd	%xmm8, %xmm5

	movapd	%xmm6, %xmm8
	movsd	%xmm7, %xmm6
	movsd	%xmm8, %xmm7

	// alpha
	movddup	0(%r10), %xmm15

	mulpd	%xmm15, %xmm0
	mulpd	%xmm15, %xmm1
	mulpd	%xmm15, %xmm2
	mulpd	%xmm15, %xmm3
	mulpd	%xmm15, %xmm4
	mulpd	%xmm15, %xmm5
	mulpd	%xmm15, %xmm6
	mulpd	%xmm15, %xmm7

	// beta
	movddup	0(%r11), %xmm14

	xorpd		%xmm15, %xmm15 // 0.0
	ucomisd		%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm0
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm4
	addq		%r13, %r12
	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm1
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm5
	addq		%r13, %r12
	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm2
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm6
	addq		%r13, %r12
	movapd		0(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm3
	movapd		16(%r12), %xmm15
	mulpd		%xmm14, %xmm15
	addpd		%xmm15, %xmm7
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_4x4_lib, .-inner_blend_scale_ab_4x4_lib
#endif
#endif





// common inner routine with file scope
//
// scale for alpha=-1 and beta=1
//
// input arguments:
// r10   <- C
// r11   <- ldc
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_M11_4X4_LIB
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_m11_4x4_lib, @function
inner_blend_scale_m11_4x4_lib:
#elif defined(OS_MAC)
_inner_blend_scale_m11_4x4_lib:
#elif defined(OS_WINDOWS)
	.def inner_blend_scale_m11_4x4_lib; .scl 2; .type 32; .endef
inner_blend_scale_m11_4x4_lib:
#endif
#endif

movapd	%xmm0, %xmm8
	movsd	%xmm1, %xmm0
	movsd	%xmm8, %xmm1

	movapd	%xmm2, %xmm8
	movsd	%xmm3, %xmm2
	movsd	%xmm8, %xmm3

	movapd	%xmm4, %xmm8
	movsd	%xmm5, %xmm4
	movsd	%xmm8, %xmm5

	movapd	%xmm6, %xmm8
	movsd	%xmm7, %xmm6
	movsd	%xmm8, %xmm7

	movapd		0(%r10), %xmm15
	subpd		%xmm0, %xmm15
	movapd		%xmm15, %xmm0
	movapd		16(%r10), %xmm15
	subpd		%xmm4, %xmm15
	movapd		%xmm15, %xmm4
	addq		%r11, %r10
	movapd		0(%r10), %xmm15
	subpd		%xmm1, %xmm15
	movapd		%xmm15, %xmm1
	movapd		16(%r10), %xmm15
	subpd		%xmm5, %xmm15
	movapd		%xmm15, %xmm5
	addq		%r11, %r10
	movapd		0(%r10), %xmm15
	subpd		%xmm2, %xmm15
	movapd		%xmm15, %xmm2
	movapd		16(%r10), %xmm15
	subpd		%xmm6, %xmm15
	movapd		%xmm15, %xmm6
	addq		%r11, %r10
	movapd		0(%r10), %xmm15
	subpd		%xmm3, %xmm15
	movapd		%xmm15, %xmm3
	movapd		16(%r10), %xmm15
	subpd		%xmm7, %xmm15
	movapd		%xmm15, %xmm7
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_m11_4x4_lib, .-inner_blend_scale_m11_4x4_lib
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
//
// output arguments:

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

	movapd		%xmm0, 0(%r10)
	movapd		%xmm4, 16(%r10)
	addq		%r11, %r10
	movapd		%xmm1, 0(%r10)
	movapd		%xmm5, 16(%r10)
	addq		%r11, %r10
	movapd		%xmm2, 0(%r10)
	movapd		%xmm6, 16(%r10)
	addq		%r11, %r10
	movapd		%xmm3, 0(%r10)
	movapd		%xmm7, 16(%r10)
//	addq	%r11, %r10

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

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


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
	INNER_BLEND_SCALE_AB_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_ab_4x4_lib
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib
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





//                                         1      2          3          4         5        6          7        8          9        10
// void kernel_dtrsm_nt_rl_inv_4x4_lib44c(int k, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44c
	.type kernel_dtrsm_nt_rl_inv_4x4_lib44c, @function
kernel_dtrsm_nt_rl_inv_4x4_lib44c:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib44c
_kernel_dtrsm_nt_rl_inv_4x4_lib44c:
#elif defined(OS_WINDOWS)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib44c
	.def kernel_dtrsm_nt_rl_inv_4x4_lib44c; .scl 2; .type 32; .endef
kernel_dtrsm_nt_rl_inv_4x4_lib44c:
#endif
	
	PROLOGUE

	// zero accumulation registers


	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3
	movapd	%xmm0, %xmm4
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm6
	movapd	%xmm0, %xmm7


	// call inner dgemm kernel nt 

	movq	ARG1, %r10
	movq	ARG2, %r11
	movq	ARG3, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_4X4_LIB4
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_kernel_dgemm_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	ARG4, %r10 // C
	movq	ARG5, %r11 // ldc
	sall	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_M11_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_blend_scale_m11_4x4_lib
#elif defined(OS_MAC)
	callq _inner_blend_scale_m11_4x4_lib
#endif
#endif


	// solve

	movq	ARG8, %r10  // E 
	movq	ARG9, %r11 // lde
	sall	$3, %r11d
	movq	ARG10, %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB
#else
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	call inner_edge_dtrsm_rlt_inv_4x4_lib
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib
#endif
#endif


	// store

	movq	ARG6, %r10 // D
	movq	ARG7, %r11 // ldd
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
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib44c, .-kernel_dtrsm_nt_rl_inv_4x4_lib44c
#endif





