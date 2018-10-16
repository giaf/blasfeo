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



// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- ldc*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_4X4_LIB
#else
	.align	4
	.type inner_scale_ab_4x4_lib, %function
inner_scale_ab_4x4_lib:
#endif

	ld1		{v28.2d}, [x8]

	fmul	v0.2d, v0.2d, v28.2d[0]
	fmul	v1.2d, v1.2d, v28.2d[0]
	fmul	v2.2d, v2.2d, v28.2d[0]
	fmul	v3.2d, v3.2d, v28.2d[0]
	fmul	v4.2d, v4.2d, v28.2d[0]
	fmul	v5.2d, v5.2d, v28.2d[0]
	fmul	v6.2d, v6.2d, v28.2d[0]
	fmul	v7.2d, v7.2d, v28.2d[0]

	ld1		{v28.2d}, [x9]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v26.2d, v28.2d[0]
	fmla	v3.2d, v27.2d, v28.2d[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v6.2d, v26.2d, v28.2d[0]
	fmla	v7.2d, v27.2d, v28.2d[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_4x4_lib, .-inner_scale_ab_4x4_lib
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- ldc*sizeof(double)
// x12  <- km
// x13  <- kn
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_4X4_VS_LIB
#else
	.align	4
	.type inner_scale_ab_4x4_vs_lib, %function
inner_scale_ab_4x4_vs_lib:
#endif

	ld1		{v28.2d}, [x8]

	fmul	v0.2d, v0.2d, v28.2d[0]
	fmul	v1.2d, v1.2d, v28.2d[0]
	fmul	v2.2d, v2.2d, v28.2d[0]
	fmul	v3.2d, v3.2d, v28.2d[0]
	fmul	v4.2d, v4.2d, v28.2d[0]
	fmul	v5.2d, v5.2d, v28.2d[0]
	fmul	v6.2d, v6.2d, v28.2d[0]
	fmul	v7.2d, v7.2d, v28.2d[0]

	ld1		{v28.2d}, [x9]

	cmp		w12, #4
	blt		1f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]

	b 0f

1:
	cmp		w12, #3
	blt		2f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]

	b 0f

2:
	cmp		w12, #2
	blt		3f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]

	b 0f

3:
	cmp		w12, #1
	blt		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]

0:

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_4x4_vs_lib, .-inner_scale_ab_4x4_vs_lib
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_4X4_LIB
#else
	.align 4
	.type inner_store_4x4_lib, %function
inner_store_4x4_lib:
#endif

	stp		q0, q1, [x8, #0]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	add		x8, x8, x9
	stp		q4, q5, [x8, #0]
	add		x8, x8, x9
	stp		q6, q7, [x8, #0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_4x4_lib, .-inner_store_4x4_lib
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
// x10  <- km
// x11  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_VS_LIB
#else
	.align 4
	.type inner_store_4x4_vs_lib, %function
inner_store_4x4_vs_lib:
#endif

	cmp		w10, #4
	bge		1f

	mov		x12, x8

	ldp		q24, q25, [x12, #0]
	add		x12, x12, x9
	ldp		q26, q27, [x12, #0]
	add		x12, x12, x9
	ldp		q28, q29, [x12, #0]
	add		x12, x12, x9
	ldp		q30, q31, [x12, #0]

	// 4th row
	ins		v1.d[1], v25.d[1]
	ins		v3.d[1], v27.d[1]
	ins		v5.d[1], v29.d[1]
	ins		v7.d[1], v31.d[1]
	cmp		w10, #3
	bge		1f
	// 3th row
	ins		v1.d[0], v25.d[0]
	ins		v3.d[0], v27.d[0]
	ins		v5.d[0], v29.d[0]
	ins		v7.d[0], v31.d[0]
	cmp		w10, #2
	bge		1f
	// 2nd row
	ins		v0.d[1], v24.d[1]
	ins		v2.d[1], v26.d[1]
	ins		v4.d[1], v28.d[1]
	ins		v6.d[1], v30.d[1]
	cmp		w10, #1
	bge		1f
	// 1st row
	ins		v0.d[0], v24.d[0]
	ins		v2.d[0], v26.d[0]
	ins		v4.d[0], v28.d[0]
	ins		v6.d[0], v30.d[0]

1:
	// 1st col
	stp		q0, q1, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	stp		q4, q5, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #3
	beq		0f
	// 4th col
	stp		q6, q7, [x8, #0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4x4_vs_lib, .-inner_store_4x4_vs_lib
#endif





//                                 w0        x1             x2         x3         x4            x5         w6       x7         sp+0
// void kernel_dgemm_nt_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	.global	kernel_dgemm_nt_4x4_lib44c
	.type	kernel_dgemm_nt_4x4_lib44c, %function
kernel_dgemm_nt_4x4_lib44c:
	


	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov	d0, xzr
	fmov    d1, d0
	fmov    d2, d0
	fmov    d3, d0
	fmov    d4, d0
	fmov    d5, d0
	fmov    d6, d0
	fmov    d7, d0



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x4 // beta
	mov		x10, x5 // C
	mov		w11, w6 // ldc
	lsl		w11, w11, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	bl inner_scale_ab_4x4_lib
#endif



	// store n
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dgemm_nt_4x4_lib44c, .-kernel_dgemm_nt_4x4_lib44c





//                                    w0        x1             x2         x3         x4            x5         w6       x7         sp+0     sp+8    sp+16
// void kernel_dgemm_nt_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	.global	kernel_dgemm_nt_4x4_vs_lib44c
	.type	kernel_dgemm_nt_4x4_vs_lib44c, %function
kernel_dgemm_nt_4x4_vs_lib44c:
	


	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov	d0, xzr
	fmov    d1, d0
	fmov    d2, d0
	fmov    d3, d0
	fmov    d4, d0
	fmov    d5, d0
	fmov    d6, d0
	fmov    d7, d0



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x4 // beta
	mov		x10, x5 // C
	mov		w11, w6 // ldc
	lsl		w11, w11, #3 // 8*ldc
	ldr		w12, [sp, #(STACKSIZE + 8)] // m1
	ldr		w13, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	bl inner_scale_ab_4x4_vs_lib
#endif



	// store n
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 8)] // m1
	ldr		w11, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dgemm_nt_4x4_vs_lib44c, .-kernel_dgemm_nt_4x4_vs_lib44c






