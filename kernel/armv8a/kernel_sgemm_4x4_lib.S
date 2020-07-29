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



// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- ldc*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_4x4_lib)
#endif

	ld1		{v28.4s}, [x8]

	ld1		{v29.4s}, [x9]

	fmul	v0.4s, v0.4s, v28.4s[0]
	fmul	v1.4s, v1.4s, v28.4s[0]
	fmul	v2.4s, v2.4s, v28.4s[0]
	fmul	v3.4s, v3.4s, v28.4s[0]

	fcmpe	s29, #0
	beq		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	ldr		q25, [x10, #0]
	add		x10, x10, x11
	ldr		q26, [x10, #0]
	add		x10, x10, x11
	ldr		q27, [x10, #0]
	add		x10, x10, x11
	fmla	v0.4s, v24.4s, v29.4s[0]
	fmla	v1.4s, v25.4s, v29.4s[0]
	fmla	v2.4s, v26.4s, v29.4s[0]
	fmla	v3.4s, v27.4s, v29.4s[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_LIB
#else
	.align 4
	FUN_START(inner_store_4x4_lib)
#endif

	str		q0, [x8, #0]
	add		x8, x8, x9
	str		q1, [x8, #0]
	add		x8, x8, x9
	str		q2, [x8, #0]
	add		x8, x8, x9
	str		q3, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_PREFETCH_4X4_LIB
#else
	.align 4
	FUN_START(inner_prefetch_4x4_lib)
#endif

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_prefetch_4x4_lib)
#endif





//                                 w0        x1             x2        x3        x4           x5        w6       x7        sp+0
// void kernel_sgemm_nt_4x4_lib44cc(int kmax, float *alpha, float *A, float *B, float *beta, float *C, int ldc, float *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_sgemm_nt_4x4_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// prefetch
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_4X4_LIB
#else
	bl inner_prefetch_4x4_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x4 // beta
	mov		x10, x5 // C
	mov		w11, w6 // ldc
	lsl		w11, w11, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	bl inner_scale_ab_4x4_lib
#endif



	// store n
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_sgemm_nt_4x4_lib44cc)






