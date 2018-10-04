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
	.macro INNER_SCALE_AB_12X4_LIB
#else
	.align	4
	.type inner_scale_ab_12x4_lib, %function
inner_scale_ab_12x4_lib:
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
	fmul	v8.2d, v8.2d, v28.2d[0]
	fmul	v9.2d, v9.2d, v28.2d[0]
	fmul	v10.2d, v10.2d, v28.2d[0]
	fmul	v11.2d, v11.2d, v28.2d[0]
	fmul	v12.2d, v12.2d, v28.2d[0]
	fmul	v13.2d, v13.2d, v28.2d[0]
	fmul	v14.2d, v14.2d, v28.2d[0]
	fmul	v15.2d, v15.2d, v28.2d[0]
	fmul	v16.2d, v16.2d, v28.2d[0]
	fmul	v17.2d, v17.2d, v28.2d[0]
	fmul	v18.2d, v18.2d, v28.2d[0]
	fmul	v19.2d, v19.2d, v28.2d[0]
	fmul	v20.2d, v20.2d, v28.2d[0]
	fmul	v21.2d, v21.2d, v28.2d[0]
	fmul	v22.2d, v22.2d, v28.2d[0]
	fmul	v23.2d, v23.2d, v28.2d[0]

	ld1		{v30.2d}, [x9]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	ldp		q28, q29, [x10, #64]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v30.2d[0]
	fmla	v1.2d, v25.2d, v30.2d[0]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v9.2d, v27.2d, v30.2d[0]
	fmla	v16.2d, v28.2d, v30.2d[0]
	fmla	v17.2d, v29.2d, v30.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	ldp		q28, q29, [x10, #64]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v30.2d[0]
	fmla	v3.2d, v25.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[0]
	fmla	v11.2d, v27.2d, v30.2d[0]
	fmla	v18.2d, v28.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	ldp		q28, q29, [x10, #64]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v12.2d, v26.2d, v30.2d[0]
	fmla	v13.2d, v27.2d, v30.2d[0]
	fmla	v20.2d, v28.2d, v30.2d[0]
	fmla	v21.2d, v29.2d, v30.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	ldp		q28, q29, [x10, #64]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v30.2d[0]
	fmla	v7.2d, v25.2d, v30.2d[0]
	fmla	v14.2d, v26.2d, v30.2d[0]
	fmla	v15.2d, v27.2d, v30.2d[0]
	fmla	v22.2d, v28.2d, v30.2d[0]
	fmla	v23.2d, v29.2d, v30.2d[0]


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_12x4_lib, .-inner_scale_ab_12x4_lib
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_12X4_LIB
#else
	.align 4
	.type inner_store_12x4_lib, %function
inner_store_12x4_lib:
#endif

	stp		q0, q1, [x8, #0]
	stp		q8, q9, [x8, #32]
	stp		q16, q17, [x8, #64]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	stp		q10, q11, [x8, #32]
	stp		q18, q19, [x8, #64]
	add		x8, x8, x9
	stp		q4, q5, [x8, #0]
	stp		q12, q13, [x8, #32]
	stp		q20, q21, [x8, #64]
	add		x8, x8, x9
	stp		q6, q7, [x8, #0]
	stp		q14, q15, [x8, #32]
	stp		q22, q23, [x8, #64]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_12x4_lib, .-inner_store_12x4_lib
#endif





//                                  w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dgemm_nt_12x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	.global	kernel_dgemm_nt_12x4_lib44c
	.type	kernel_dgemm_nt_12x4_lib44c, %function
kernel_dgemm_nt_12x4_lib44c:
	


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
	fmov    d8, d0
	fmov    d9, d0
	fmov    d10, d0
	fmov    d11, d0
	fmov    d12, d0
	fmov    d13, d0
	fmov    d14, d0
	fmov    d15, d0
	fmov    d16, d0
	fmov    d17, d0
	fmov    d18, d0
	fmov    d19, d0
	fmov    d20, d0
	fmov    d21, d0
	fmov    d22, d0
	fmov    d23, d0



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_12x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB
#else
	bl inner_scale_ab_12x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB
#else
	bl inner_store_12x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dgemm_nt_12x4_lib44c, .-kernel_dgemm_nt_12x4_lib44c






