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
// x11  <- ldc*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_8x8_lib)
#endif

	ld1		{v28.4s}, [x8]

	ld1		{v29.4s}, [x9]

	fmul	v0.4s, v0.4s, v28.s[0]
	fmul	v1.4s, v1.4s, v28.s[0]
	fmul	v2.4s, v2.4s, v28.s[0]
	fmul	v3.4s, v3.4s, v28.s[0]
	fmul	v4.4s, v4.4s, v28.s[0]
	fmul	v5.4s, v5.4s, v28.s[0]
	fmul	v6.4s, v6.4s, v28.s[0]
	fmul	v7.4s, v7.4s, v28.s[0]
	fmul	v8.4s, v8.4s, v28.s[0]
	fmul	v9.4s, v9.4s, v28.s[0]
	fmul	v10.4s, v10.4s, v28.s[0]
	fmul	v11.4s, v11.4s, v28.s[0]
	fmul	v12.4s, v12.4s, v28.s[0]
	fmul	v13.4s, v13.4s, v28.s[0]
	fmul	v14.4s, v14.4s, v28.s[0]
	fmul	v15.4s, v15.4s, v28.s[0]

	fcmpe	s29, #0.0
	beq		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]
	fmla	v1.4s, v26.4s, v29.s[0]
	fmla	v5.4s, v27.4s, v29.s[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]
	fmla	v3.4s, v26.4s, v29.s[0]
	fmla	v7.4s, v27.4s, v29.s[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v8.4s, v24.4s, v29.s[0]
	fmla	v12.4s, v25.4s, v29.s[0]
	fmla	v9.4s, v26.4s, v29.s[0]
	fmla	v13.4s, v27.4s, v29.s[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v10.4s, v24.4s, v29.s[0]
	fmla	v14.4s, v25.4s, v29.s[0]
	fmla	v11.4s, v26.4s, v29.s[0]
	fmla	v15.4s, v27.4s, v29.s[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x8_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_LIB
#else
	.align 4
	FUN_START(inner_store_8x8_lib)
#endif

	str		q0, [x8, #0]
	str		q4, [x8, #16]
	add		x8, x8, x9

	str		q1, [x8, #0]
	str		q5, [x8, #16]
	add		x8, x8, x9

	str		q2, [x8, #0]
	str		q6, [x8, #16]
	add		x8, x8, x9

	str		q3, [x8, #0]
	str		q7, [x8, #16]
	add		x8, x8, x9

	str		q8, [x8, #0]
	str		q12, [x8, #16]
	add		x8, x8, x9

	str		q9, [x8, #0]
	str		q13, [x8, #16]
	add		x8, x8, x9

	str		q10, [x8, #0]
	str		q14, [x8, #16]
	add		x8, x8, x9

	str		q11, [x8, #0]
	str		q15, [x8, #16]
//	add		x8, x8, x9

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x8_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_PREFETCH_8X8_LIB
#else
	.align 4
	FUN_START(inner_prefetch_8x8_lib)
#endif

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9

	prfm	PLDL1KEEP, [x8, #0]
//	add		x8, x8, x9

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_prefetch_8x8_lib)
#endif





//                                 w0        x1             x2        w3        x4       w5       x6           x7        sp+0     sp+8      sp+16
// void kernel_sgemm_nt_8x8_lib44cc(int kmax, float *alpha, float *A, int sda, float *B, int sdb, float *beta, float *C, int ldc, float *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_sgemm_nt_8x8_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// prefetch C
//	mov		x8, x6 // beta
	ld1		{v29.4s}, [x6]
	fcmpe	s29, #0.0
	beq		100f

	mov		x8, x7 // C
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldc
	lsl		w9, w9, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_PREFETCH_8X8_LIB
#else
	bl inner_prefetch_8x8_lib
#endif

100:


	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x4 // B
	mov		w12, w5 // sdb
	lsl		w12, w12, #4 // 16*sdb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X8_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x8_lib4
#endif



	// prefetch
	ldr		x8, [sp, #(STACKSIZE + 8)] // D
	ldr		w9, [sp, #(STACKSIZE + 16)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_8X8_LIB
#else
	bl inner_prefetch_8x8_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x6 // beta
	mov		x10, x7 // C
	ldr		w11, [sp, #(STACKSIZE + 0)] // ldc
	lsl		w11, w11, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB
#else
	bl inner_scale_ab_8x8_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 8)] // ldd
	ldr		w9, [sp, #(STACKSIZE + 16)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB
#else
	bl inner_store_8x8_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_sgemm_nt_8x8_lib44cc)







