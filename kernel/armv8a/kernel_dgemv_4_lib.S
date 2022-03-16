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
// w8   <- k
// x9   <- A
// w10  <- lda*sizeof(double)
// x11  <- z
// v0   <- [x_0, x_1]
// v1   <- [x_2, x_3]
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_N_4_LIBC
#else
	.align	4
	FUN_START(inner_kernel_gemv_n_4_libc)
#endif

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10
	add		x13, x9, x10, lsl #1
	add		x14, x12, x10, lsl #1

	// prefetch
//	prfm	PLDL1KEEP, [x9, #0]
//	prfm	PLDL1KEEP, [x9, #64]
//	prfm	PLDL1KEEP, [x11, #0]

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
//	prfm	PLDL1KEEP, [x9, x10]
//	prfm	PLDL1KEEP, [x9, x12]
//	prfm	PLDL1KEEP, [x11, #32]

	// main loop
1:
	
	ldp		q24, q25, [x11] //, #32

	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x12], #32
	ldp		q20, q21, [x13], #32
	ldp		q22, q23, [x14], #32

	// unroll x4
	fmla	v24.2d, v16.2d, v0.d[0]
	fmla	v25.2d, v17.2d, v0.d[0]
//	prfm	PLDL1KEEP, [x9, #96]

	fmla	v24.2d, v18.2d, v0.d[1]
	fmla	v25.2d, v19.2d, v0.d[1]
//	prfm	PLDL1KEEP, [x12, #96]

	fmla	v24.2d, v20.2d, v1.d[0]
	fmla	v25.2d, v21.2d, v1.d[0]
//	prfm	PLDL1KEEP, [x13, #96]
	sub		w8, w8, #4

	fmla	v24.2d, v22.2d, v1.d[1]
	cmp		w8, #4
	fmla	v25.2d, v23.2d, v1.d[1]
//	prfm	PLDL1KEEP, [x14, #96]

	stp		q24, q25, [x11], #32
//	add		x11, x11, #32

	bgt		1b

0:

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldr		d24, [x11] //, #8
	ldr		d16, [x9], #8
	ldr		d18, [x12], #8
	ldr		d20, [x13], #8
	ldr		d22, [x14], #8
	fmla	v24.2d, v16.2d, v0.d[0]
	fmla	v24.2d, v18.2d, v0.d[1]
	fmla	v24.2d, v20.2d, v1.d[0]
	fmla	v24.2d, v22.2d, v1.d[1]
	str		d24, [x11], #8

	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemv_n_4_libc)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// w10  <- lda*sizeof(double)
// x11  <- x
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_T_4_LIBC
#else
	.align	4
	FUN_START(inner_kernel_gemv_t_4_libc)
#endif

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10
	add		x13, x9, x10, lsl #1
	add		x14, x12, x10, lsl #1

	// prefetch
//	prfm	PLDL1KEEP, [x9, #0]
//	prfm	PLDL1KEEP, [x9, #64]
//	prfm	PLDL1KEEP, [x11, #0]

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
//	prfm	PLDL1KEEP, [x9, x10]
//	prfm	PLDL1KEEP, [x9, x12]
//	prfm	PLDL1KEEP, [x11, #32]

	// main loop
1:
	
	ldp		q24, q25, [x11], #32

	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x12], #32
	ldp		q20, q21, [x13], #32
	ldp		q22, q23, [x14], #32

	// unroll x4
	fmla	v0.2d, v16.2d, v24.2d
	fmla	v1.2d, v18.2d, v24.2d
//	prfm	PLDL1KEEP, [x9, #96]

	fmla	v2.2d, v20.2d, v24.2d
	fmla	v3.2d, v22.2d, v24.2d
//	prfm	PLDL1KEEP, [x12, #96]

	fmla	v0.2d, v17.2d, v25.2d
	fmla	v1.2d, v19.2d, v25.2d
//	prfm	PLDL1KEEP, [x13, #96]
	sub		w8, w8, #4

	fmla	v2.2d, v21.2d, v25.2d
	cmp		w8, #4
	fmla	v3.2d, v23.2d, v25.2d
//	prfm	PLDL1KEEP, [x14, #96]

	bgt		1b

0:

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldr		d24, [x11], #8
	ldr		d16, [x9], #8
	ldr		d18, [x12], #8
	ldr		d20, [x13], #8
	ldr		d22, [x14], #8
	fmla	v0.2d, v16.2d, v24.2d
	fmla	v1.2d, v18.2d, v24.2d
	fmla	v2.2d, v20.2d, v24.2d
	fmla	v3.2d, v22.2d, v24.2d

	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemv_t_4_libc)
#endif





//                            w0        x1             x2         w3       x4         x5
// void kernel_dgemv_n_4_libc(int kmax, double *alpha, double *A, int lda, double *x, double *z)

	.align	4
	GLOB(kernel_dgemv_n_4_libc)
	FUN_START(kernel_dgemv_n_4_libc)
	

	PROLOGUE


	// initialize n
	ldr		d16, [x1]
	ldp		q0, q1, [x4]
	fmul	v0.2d, v0.2d, v16.d[0]
	fmul	v1.2d, v1.2d, v16.d[0]


	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // lda
	lsl		w10, w10, #3 // 8*lda
	mov		x11, x5 // z

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_N_4_LIBC
#else
	CALL(inner_kernel_gemv_n_4_libc)
#endif


	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemv_n_4_libc)





//                            w0        x1             x2         w3       x4         x5            x6         x7
// void kernel_dgemv_t_4_libc(int kmax, double *alpha, double *A, int lda, double *x, double *beta, double *y, double *z)

	.align	4
	GLOB(kernel_dgemv_t_4_libc)
	FUN_START(kernel_dgemv_t_4_libc)
	

	PROLOGUE


	ZERO_ACC_T


	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // lda
	lsl		w10, w10, #3 // 8*lda
	mov		x11, x4 // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_T_4_LIBC
#else
	CALL(inner_kernel_gemv_t_4_libc)
#endif


	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // y

#if MACRO_LEVEL>=1
	INNER_BLEND_T_4_LIB4
	INNER_SCALE_AB_4_LIB4
#else
	CALL(inner_blend_t_4_lib4)
	CALL(inner_scale_ab_4_lib4)
#endif


	// store n
	mov		x8, x7 // z

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	CALL(inner_store_4_lib4)
#endif


	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemv_t_4_libc)






