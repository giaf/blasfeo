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



#define STACKSIZE 11*16
#define PROLOGUE \
	add sp, sp, #-(11 * 16); \
	stp d8, d9, [sp, #(0 * 16)]; \
	stp d10, d11, [sp, #(1 * 16)]; \
	stp d12, d13, [sp, #(2 * 16)]; \
	stp d14, d15, [sp, #(3 * 16)]; \
	stp x18, x19, [sp, #(4 * 16)]; \
	stp x20, x21, [sp, #(5 * 16)]; \
	stp x22, x23, [sp, #(6 * 16)]; \
	stp x24, x25, [sp, #(7 * 16)]; \
	stp x26, x27, [sp, #(8 * 16)]; \
	stp x28, x29, [sp, #(9 * 16)]; \
	str x30, [sp, #(10 * 16)];
#define EPILOGUE \
	ldp d8, d9, [sp, #(0 * 16)]; \
	ldp d10, d11, [sp, #(1 * 16)]; \
	ldp d12, d13, [sp, #(2 * 16)]; \
	ldp d14, d15, [sp, #(3 * 16)]; \
	ldp x18, x19, [sp, #(4 * 16)]; \
	ldp x20, x21, [sp, #(5 * 16)]; \
	ldp x22, x23, [sp, #(6 * 16)]; \
	ldp x24, x25, [sp, #(7 * 16)]; \
	ldp x26, x27, [sp, #(8 * 16)]; \
	ldp x28, x29, [sp, #(9 * 16)]; \
	ldr x30, [sp, #(10 * 16)]; \
	add sp, sp, #(11 * 16);
#define GLOB_FUN_START(NAME) \
	.global	NAME; \
	.type NAME, %function; \
NAME:
#define FUN_START(NAME) \
	.type NAME, %function; \
NAME:
#define FUN_END(NAME) \
	.size	NAME, .-NAME
#define ZERO_ACC \
	fmov	d0, xzr; \
	fmov    d1, d0





	.text





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- x
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_N_4_LIB4
#else
	.align	4
	FUN_START(inner_kernel_gemv_n_4_lib4)
#endif

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8

	// main loop
1:
	
	ldp		q24, q25, [x10, #0]

	ldp		q16, q17, [x9, #(0*32)]
	ldp		q18, q19, [x9, #(1*32)]
	ldp		q20, q21, [x9, #(2*32)]
	ldp		q22, q23, [x9, #(3*32)]

	// unroll 0
//	ldr		q20, [x9, #(0*16)]
	fmla	v0.2d, v16.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #(2*128+0)]
//	ldr		q21, [x9, #(1*16)]
	fmla	v1.2d, v17.2d, v24.2d[0]
	add		x10, x10, #32

	// unroll 1
//	ldr		q22, [x9, #(2*16)]
	fmla	v8.2d, v18.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #(2*128+64)]
//	ldr		q23, [x9, #(3*16)]
	fmla	v9.2d, v19.2d, v24.2d[1]
	sub		w8, w8, #4

	// unroll 2
//	ldr		q16, [x9, #(4*16)]
	fmla	v0.2d, v20.2d, v25.2d[0]
//	ldr		q17, [x9, #(5*16)]
	fmla	v1.2d, v21.2d, v25.2d[0]
	cmp		w8, #4

	// unroll 3
//	ldr		q18, [x9, #(6*16)]
	fmla	v8.2d, v22.2d, v25.2d[1]
//	ldr		q19, [x9, #(7*16)]
	fmla	v9.2d, v23.2d, v25.2d[1]
	add		x9, x9, #128

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d

0:

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.1d}, [x10], #8
	ld1		{v16.2d, v17.2d}, [x9], #32
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]

	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemv_n_4_lib4)
#endif





// subroutine
//
// triangular substitution with vector RHS
//
// input arguments:
// x8  <- E
// x9  <- inv_diag_E
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LN_INV_4_LIB4
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsv_ln_inv_4_lib4)
#endif
	
	ldp		q24, q25, [x8, #0]
	ldr		d16, [x9, #0]
	fmul	v17.2d, v0.2d, v16.2d[0]
	fmls	v0.2d, v24.2d, v17.2d[0]
	fmls	v1.2d, v25.2d, v17.2d[0]
	ins		v0.d[0], v17.d[0]

	ldr		q25, [x8, #(32+16)]
	ldr		d16, [x9, #8]
	fmul	v18.2d, v0.2d, v16.2d[0]
	fmls	v1.2d, v25.2d, v18.2d[1]
	ins		v0.d[1], v18.d[1]

	ldr		q25, [x8, #(64+16)]
	ldr		d16, [x9, #16]
	fmul	v19.2d, v1.2d, v16.2d[0]
	fmls	v1.2d, v25.2d, v19.2d[0]
	ins		v1.d[0], v19.d[0]

	ldr		d16, [x9, #24]
	fmul	v20.2d, v1.2d, v16.2d[0]
	ins		v1.d[1], v20.d[1]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsv_ln_inv_4_lib4)
#endif





// subroutine
//
// triangular substitution with vector RHS
//
// input arguments:
// x8  <- E
// x9  <- inv_diag_E
// w10 <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LN_INV_4_VS_LIB4
#else
	.p2align 4,,15
	FUN_START(inner_edge_dtrsv_ln_inv_4_vs_lib4)
#endif
	
	ldp		q24, q25, [x8, #0]
	ldr		d16, [x9, #0]
	fmul	v17.2d, v0.2d, v16.2d
	fmls	v0.2d, v24.2d, v17.2d[0]
	fmls	v1.2d, v25.2d, v17.2d[0]
	ins		v0.d[0], v17.d[0]

	cmp		w10, #2
	blt		0f // return

	ldr		q25, [x8, #(32+16)]
	ldr		d16, [x9, #8]
	fmul	v18.2d, v0.2d, v16.2d[0]
	fmls	v1.2d, v25.2d, v18.2d[1]
	ins		v0.d[1], v18.d[1]

	cmp		w10, #3
	blt		0f // return

	ldr		q25, [x8, #(64+16)]
	ldr		d16, [x9, #16]
	fmul	v19.2d, v1.2d, v16.2d[0]
	fmls	v1.2d, v25.2d, v19.2d[0]
	ins		v1.d[0], v19.d[0]

	cmp		w10, #4
	blt		0f // return

	ldr		d16, [x9, #24]
	fmul	v20.2d, v1.2d, v16.2d[0]
	ins		v1.d[1], v20.d[1]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_dtrsv_ln_inv_4_vs_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- y
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4_LIB4
#else
	.align	4
	FUN_START(inner_scale_ab_4_lib4)
#endif

	ldr		d28, [x8]

	ldr		d29, [x9]

	fmul	v0.2d, v0.2d, v28.2d[0]
	fmul	v1.2d, v1.2d, v28.2d[0]

	fcmpe	d29, #0
	beq		0f

	ldp		q24, q25, [x10, #0]
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4_lib4)
#endif





// subroutine
//
// input arguments:
// x8  <- y
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_4_LIB4
#else
	.align	4
	FUN_START(inner_scale_m11_4_lib4)
#endif

	ldp		q24, q25, [x8, #0]
	fsub	v0.2d, v24.2d, v0.2d
	fsub	v1.2d, v25.2d, v1.2d

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_4_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- z
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.align 4
	FUN_START(inner_store_4_lib4)
#endif

	stp		q0, q1, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- z
// x9   <- m1
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_VS_LIB4
#else
	.align 4
	FUN_START(inner_store_4_vs_lib4)
#endif

	cmp		w9, #4
	bge		1f
	cmp		w9, #2
	bgt		2f
	beq		3f

	// 1
	str		d0, [x8, #0]
	b		0f

3:
	// 2
	str		q0, [x8, #0]
	b		0f

2:
	// 3
	str		q0, [x8, #0]
	str		d1, [x8, #16]
	b		0f

1:
	// 4
	stp		q0, q1, [x8, #0]
	b		0f

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_4_vs_lib4)
#endif





//                            w0        x1             x2         x3         x4            x5         x6
// void kernel_dgemv_n_4_lib4(int kmax, double *alpha, double *A, double *x, double *beta, double *y, double *z)

	.align	4
	GLOB_FUN_START(kernel_dgemv_n_4_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_N_4_LIB4
#else
	bl	inner_kernel_gemv_n_4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x4 // beta
	mov		x10, x5 // y

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4_LIB4
#else
	bl inner_scale_ab_4_lib4
#endif



	// store n
	mov		x8, x6 // z

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	bl inner_store_4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemv_n_4_lib4)





//                               w0        x1             x2         x3         x4            x5         x6         w7
// void kernel_dgemv_n_4_vs_lib4(int kmax, double *alpha, double *A, double *x, double *beta, double *y, double *z, int m1)

	.align	4
	GLOB_FUN_START(kernel_dgemv_n_4_vs_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_N_4_LIB4
#else
	bl	inner_kernel_gemv_n_4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x4 // beta
	mov		x10, x5 // y

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4_LIB4
#else
	bl inner_scale_ab_4_lib4
#endif



	// store n
	mov		x8, x6 // z
	mov		w9, w7 // m1

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	bl inner_store_4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemv_n_4_vs_lib4)





//                                 w0     x1         x2                  x3         x4         x5
// void kernel_dtrsv_ln_inv_4_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4
	GLOB_FUN_START(kernel_dtrsv_ln_inv_4_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner dgemv kernel n
	mov		w8, w0 // k
	mov		x9, x1  // A
	mov		x10, x3  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_N_4_LIB4
#else
	bl	inner_kernel_gemv_n_4_lib4
#endif



	// call inner blender n
	mov		x8, x4   // y

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4_LIB4
#else
	bl	inner_scale_m11_4_lib4
#endif


	// solution
	mov		x8, x1 // A+k*sizeof(double)
	mov		w9, w0 // k
	lsl		w0, w0, #5 // k*sizeof(double)
	add		x8, x8, x9
	mov		x9, x2 // inv_diag_A

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LN_INV_4_LIB4
#else
	bl	inner_edge_dtrsv_ln_inv_4_lib4
#endif


	// store
	mov		x8, x5 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	bl	inner_store_4_lib4
#endif


	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsv_ln_inv_4_lib4)






//                                    w0     x1         x2                  x3         x4         x5         w6      w7
// void kernel_dtrsv_ln_inv_4_vs_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z, int m1, int n1);

	.p2align 4
	GLOB_FUN_START(kernel_dtrsv_ln_inv_4_vs_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner dgemv kernel n
	mov		w8, w0 // k
	mov		x9, x1  // A
	mov		x10, x3  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_N_4_LIB4
#else
	bl	inner_kernel_gemv_n_4_lib4
#endif



	// call inner blender n
	mov		x8, x4   // y

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4_LIB4
#else
	bl	inner_scale_m11_4_lib4
#endif


	// solution
	mov		x8, x1 // A+k*sizeof(double)
	mov		w9, w0 // k
	lsl		w9, w9, #5 // k*sizeof(double)
	add		x8, x8, x9
	mov		x9, x2 // inv_diag_A
	mov		w10, w7 // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LN_INV_4_VS_LIB4
#else
	bl	inner_edge_dtrsv_ln_inv_4_vs_lib4
#endif


	// store
	mov		x8, x5 // z 
	mov		w9, w6 // m1

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	bl	inner_store_4_vs_lib4
#endif


	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsv_ln_inv_4_vs_lib4)






