/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
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
// zmm0 <- []
// ...
// zmm7 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r10   <- C
// r13   <- ldc
// zmm0 <- []
// ...
// zmm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_16X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_16x8_lib)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm25

	vmulpd		%zmm0, %zmm25, %zmm0
	vmulpd		%zmm1, %zmm25, %zmm1
	vmulpd		%zmm2, %zmm25, %zmm2
	vmulpd		%zmm3, %zmm25, %zmm3
	vmulpd		%zmm4, %zmm25, %zmm4
	vmulpd		%zmm5, %zmm25, %zmm5
	vmulpd		%zmm6, %zmm25, %zmm6
	vmulpd		%zmm7, %zmm25, %zmm7
	vmulpd		%zmm8, %zmm25, %zmm8
	vmulpd		%zmm9, %zmm25, %zmm9
	vmulpd		%zmm10, %zmm25, %zmm10
	vmulpd		%zmm11, %zmm25, %zmm11
	vmulpd		%zmm12, %zmm25, %zmm12
	vmulpd		%zmm13, %zmm25, %zmm13
	vmulpd		%zmm14, %zmm25, %zmm14
	vmulpd		%zmm15, %zmm25, %zmm15

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm8
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm9
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm10
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm11
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm12
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm13
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm14
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm15
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_16x8_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// zmm0 <- []
// ...
// zmm7 <- []
//
// output arguments:
// r10  <- D
// r11  <- ldd
// zmm0 <- []
// ...
// zmm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_16X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_16x8_lib)
#endif
	
	vmovupd %zmm0, 0(%r10)
	vmovupd %zmm8, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm1, 0(%r10)
	vmovupd %zmm9, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm2, 0(%r10)
	vmovupd %zmm10, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm3, 0(%r10)
	vmovupd %zmm11, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm4, 0(%r10)
	vmovupd %zmm12, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm5, 0(%r10)
	vmovupd %zmm13, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm6, 0(%r10)
	vmovupd %zmm14, 64(%r10)
	addq		%r11, %r10
	vmovupd %zmm7, 0(%r10)
	vmovupd %zmm15, 64(%r10)
//	addq		%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_16x8_lib)
#endif





//                                 1      2              3          4        5          6             7          8        9          10
// void kernel_dgemm_nt_16x8_lib88cc(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_16x8_lib88cc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// prefetch C

//	movq		ARG6, %r10 // beta
//	vmovsd		0(%r10), %xmm14
//	vxorpd		%xmm15, %xmm15, %xmm15 // 0.0
//	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
//	je			100f // end

//	movq	ARG7, %r10 // C
//	movq	ARG8, %r11 // ldc
//	sall	$ 3, %r11d

#if MACRO_LEVEL>=1
//	INNER_PREFETCH0_8X4_LIB
#else
//	CALL(inner_prefetch0_8x4_lib)
#endif

//100:


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$ 6, %r12d // 8*sda*sizeof(double)
	movq	ARG5, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_16X8_LIB8
#else
	CALL(inner_kernel_dgemm_nt_16x8_lib8)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$ 3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X8_LIB
#else
	CALL(inner_scale_ab_16x8_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$ 3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_16X8_LIB
#else
	CALL(inner_store_16x8_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_16x8_lib88cc)






