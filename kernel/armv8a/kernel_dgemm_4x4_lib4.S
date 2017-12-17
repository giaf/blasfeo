/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
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





	.text





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_4x4_lib4, %function
inner_kernel_gemm_add_nt_4x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1   {v24.2d, v25.2d}, [x9], #32
	ld1   {v28.2d, v29.2d}, [x10], #32

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	
	// unroll 0
	ldp		q26, q27, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ldp		q24, q25, [x9], #32
	fmla	v0.2d, v26.2d, v30.2d[0]
	fmla	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmla	v2.2d, v26.2d, v30.2d[1]
	fmla	v3.2d, v27.2d, v30.2d[1]
	fmla	v4.2d, v26.2d, v31.2d[0]
	fmla	v5.2d, v27.2d, v31.2d[0]
	sub		w8, w8, #4
	fmla	v6.2d, v26.2d, v31.2d[1]
	fmla	v7.2d, v27.2d, v31.2d[1]

	// unroll 2
	ldp		q26, q27, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ldp		q24, q25, [x9], #32
	fmla	v0.2d, v26.2d, v30.2d[0]
	fmla	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmla	v2.2d, v26.2d, v30.2d[1]
	fmla	v3.2d, v27.2d, v30.2d[1]
	cmp		w8, #4
	fmla	v4.2d, v26.2d, v31.2d[0]
	fmla	v5.2d, v27.2d, v31.2d[0]
	fmla	v6.2d, v26.2d, v31.2d[1]
	fmla	v7.2d, v27.2d, v31.2d[1]

	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nt_4x4_lib4, .-inner_kernel_gemm_add_nt_4x4_lib4
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_SUB_NT_4X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_sub_nt_4x4_lib4, %function
inner_kernel_gemm_sub_nt_4x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1   {v24.2d, v25.2d}, [x9], #32
	ld1   {v28.2d, v29.2d}, [x10], #32

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	
	// unroll 0
	ldp		q26, q27, [x9], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ldp		q24, q25, [x9], #32
	fmls	v0.2d, v26.2d, v30.2d[0]
	fmls	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmls	v2.2d, v26.2d, v30.2d[1]
	fmls	v3.2d, v27.2d, v30.2d[1]
	fmls	v4.2d, v26.2d, v31.2d[0]
	fmls	v5.2d, v27.2d, v31.2d[0]
	sub		w8, w8, #4
	fmls	v6.2d, v26.2d, v31.2d[1]
	fmls	v7.2d, v27.2d, v31.2d[1]

	// unroll 2
	ldp		q26, q27, [x9], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ldp		q24, q25, [x9], #32
	fmls	v0.2d, v26.2d, v30.2d[0]
	fmls	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmls	v2.2d, v26.2d, v30.2d[1]
	fmls	v3.2d, v27.2d, v30.2d[1]
	cmp		w8, #4
	fmls	v4.2d, v26.2d, v31.2d[0]
	fmls	v5.2d, v27.2d, v31.2d[0]
	fmls	v6.2d, v26.2d, v31.2d[1]
	fmls	v7.2d, v27.2d, v31.2d[1]

	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_sub_nt_4x4_lib4, .-inner_kernel_gemm_sub_nt_4x4_lib4
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_SYRK_L_SUB_NT_4X4_LIB4
#else
	.align	4
	.type inner_kernel_syrk_l_sub_nt_4x4_lib4, %function
inner_kernel_syrk_l_sub_nt_4x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1   {v24.2d, v25.2d}, [x9], #32
	ld1   {v28.2d, v29.2d}, [x10], #32

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	
	// unroll 0
	ldp		q26, q27, [x9], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ldp		q24, q25, [x9], #32
	fmls	v0.2d, v26.2d, v30.2d[0]
	fmls	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmls	v2.2d, v26.2d, v30.2d[1]
	fmls	v3.2d, v27.2d, v30.2d[1]
//	fmls	v4.2d, v26.2d, v31.2d[0]
	fmls	v5.2d, v27.2d, v31.2d[0]
	sub		w8, w8, #4
//	fmls	v6.2d, v26.2d, v31.2d[1]
	fmls	v7.2d, v27.2d, v31.2d[1]

	// unroll 2
	ldp		q26, q27, [x9], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10], #32
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #64]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ldp		q24, q25, [x9], #32
	fmls	v0.2d, v26.2d, v30.2d[0]
	fmls	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10], #32
	fmls	v2.2d, v26.2d, v30.2d[1]
	fmls	v3.2d, v27.2d, v30.2d[1]
	cmp		w8, #4
//	fmls	v4.2d, v26.2d, v31.2d[0]
	fmls	v5.2d, v27.2d, v31.2d[0]
//	fmls	v6.2d, v26.2d, v31.2d[1]
	fmls	v7.2d, v27.2d, v31.2d[1]

	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.2d, v24.2d, v28.2d[0]
	fmls	v1.2d, v25.2d, v28.2d[0]
	fmls	v2.2d, v24.2d, v28.2d[1]
	fmls	v3.2d, v25.2d, v28.2d[1]
//	fmls	v4.2d, v24.2d, v29.2d[0]
	fmls	v5.2d, v25.2d, v29.2d[0]
//	fmls	v6.2d, v24.2d, v29.2d[1]
	fmls	v7.2d, v25.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_syrk_l_sub_nt_4x4_lib4, .-inner_kernel_syrk_l_sub_nt_4x4_lib4
#endif





// subroutine
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// x8   <- E
// x9   <- inv_diag_E
//
// output arguments:
// x8   <- E
// x9   <- inv_diag_E

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_RLT_INV_4X4_LIB4
#else
	.align 4
	.type inner_edge_trsm_rlt_inv_4x4_lib4, %function
inner_edge_trsm_rlt_inv_4x4_lib4:
#endif
	
	// first column
	ldr			d16, [x9, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]

	// second column
	ldr			d16, [x8, #8] // E[1+4*0]
	fmls		v2.2d, v0.2d, v16.2d[0]
	fmls		v3.2d, v1.2d, v16.2d[0]
	ldr			d16, [x9, #8] // E_inv[1]
	fmul		v2.2d, v2.2d, v16.2d[0]
	fmul		v3.2d, v3.2d, v16.2d[0]

	// third column
	ldr			d16, [x8, #16] // E[2+4*0]
	fmls		v4.2d, v0.2d, v16.2d[0]
	fmls		v5.2d, v1.2d, v16.2d[0]
	ldr			d16, [x8, #48] // E[2+4*1]
	fmls		v4.2d, v2.2d, v16.2d[0]
	fmls		v5.2d, v3.2d, v16.2d[0]
	ldr			d16, [x9, #16] // E_inv[2]
	fmul		v4.2d, v4.2d, v16.2d[0]
	fmul		v5.2d, v5.2d, v16.2d[0]

	// forth column
	ldr			d16, [x8, #24] // E[3+4*0]
	fmls		v6.2d, v0.2d, v16.2d[0]
	fmls		v7.2d, v1.2d, v16.2d[0]
	ldr			d16, [x8, #56] // E[3+4*1]
	fmls		v6.2d, v2.2d, v16.2d[0]
	fmls		v7.2d, v3.2d, v16.2d[0]
	ldr			d16, [x8, #88] // E[3+4*1]
	fmls		v6.2d, v4.2d, v16.2d[0]
	fmls		v7.2d, v5.2d, v16.2d[0]
	ldr			d16, [x9, #24] // E_inv[2]
	fmul		v6.2d, v6.2d, v16.2d[0]
	fmul		v7.2d, v7.2d, v16.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_trsm_rlt_inv_4x4_lib4, .-inner_edge_trsm_rlt_inv_4x4_lib4
#endif
#endif





// subroutine
//
// cholesky factorization 
//
// input arguments:
// x8   <- inv_diag_D
//
// output arguments:
// x8   <- inv_diag_D

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_POTRF_4X4_LIB4
#else
	.p2align 4
	.type inner_edge_potrf_4x4_lib4, %function
inner_edge_potrf_4x4_lib4:
#endif
	
	fmov		d16, 1.0e+0 // 1.0

	// first column
	ins			v17.d[0], v0.d[0]
	fcmpe		d17, #0
	ble			1f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
	str			d18, [x8, #0]
2:
	fmul		v0.2d, v0.2d, v18.2d[0]
	fmul		v1.2d, v1.2d, v18.2d[0]

	// second column
	fmls		v2.2d, v0.2d, v0.2d[1]
	fmls		v3.2d, v1.2d, v0.2d[1]
	ins			v17.d[0], v2.d[1]
	fcmpe		d17, #0
	ble			3f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
	str			d18, [x8, #8]
4:
	fmul		v2.2d, v2.2d, v18.2d[0]
	fmul		v3.2d, v3.2d, v18.2d[0]

	// third column
	fmls		v5.2d, v1.2d, v1.2d[0]
	fmls		v5.2d, v3.2d, v3.2d[0]
	ins			v17.d[0], v5.d[0]
	fcmpe		d17, #0
	ble			5f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
	str			d18, [x8, #16]
6:
	fmul		v5.2d, v5.2d, v18.2d[0]

	// fourth column
	fmls		v7.2d, v1.2d, v1.2d[1]
	fmls		v7.2d, v3.2d, v3.2d[1]
	fmls		v7.2d, v5.2d, v5.2d[1]
	ins			v17.d[0], v7.d[1]
	fcmpe		d17, #0
	ble			7f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
	str			d18, [x8, #24]
8:
	fmul		v7.2d, v7.2d, v18.2d[0]

	b			0f

1:
	fmov		d0, xzr
	b			2b

3:
	fmov		d5, xzr
	b			4b

5:
	fmov		d10, xzr
	b			6b

7:
	fmov		d15, xzr

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_edge_potrf_4x4_lib4, .-inner_edge_potrf_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_4X4_LIB4
#else
	.align	4
	.type inner_scale_ab_4x4_lib4, %function
inner_scale_ab_4x4_lib4:
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

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v26.2d, v28.2d[0]
	fmla	v3.2d, v27.2d, v28.2d[0]

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v6.2d, v26.2d, v28.2d[0]
	fmla	v7.2d, v27.2d, v28.2d[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_4x4_lib4, .-inner_scale_ab_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8  <- C
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_11_4X4_LIB4
#else
	.align	4
	.type inner_scale_11_4x4_lib4, %function
inner_scale_11_4x4_lib4:
#endif

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	fadd	v0.2d, v24.2d, v0.2d
	fadd	v1.2d, v25.2d, v1.2d
	fadd	v2.2d, v26.2d, v2.2d
	fadd	v3.2d, v27.2d, v3.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	fadd	v4.2d, v24.2d, v4.2d
	fadd	v5.2d, v25.2d, v5.2d
	fadd	v6.2d, v26.2d, v6.2d
	fadd	v7.2d, v27.2d, v7.2d

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_11_4x4_lib4, .-inner_scale_11_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_4X4_LIB4
#else
	.align 4
	.type inner_store_4x4_lib4, %function
inner_store_4x4_lib4:
#endif

	stp		q0, q1, [x8, #0]
	stp		q2, q3, [x8, #32]
	stp		q4, q5, [x8, #64]
	stp		q6, q7, [x8, #96]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_L_4X4_LIB4
#else
	.align 4
	.type inner_store_l_4x4_lib4, %function
inner_store_l_4x4_lib4:
#endif

	ldr		q16, [x8, #32]
	ldr		q17, [x8, #112]

	ins		v2.d[0], v16.d[0]
	ins		v7.d[0], v17.d[0]

	stp		q0, q1, [x8, #0]
	stp		q2, q3, [x8, #32]
	str		q5, [x8, #80]
	str		q7, [x8, #112]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_l_4x4_lib4, .-inner_store_l_4x4_lib4
#endif





//                               w0        x1             x2         x3         x4            x5         x6
// void kernel_dgemm_nt_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)

	.align	4
	.global	kernel_dgemm_nt_4x4_lib4
	.type	kernel_dgemm_nt_4x4_lib4, %function
kernel_dgemm_nt_4x4_lib4:
	


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

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
	bl inner_scale_ab_4x4_lib4
#endif



	// store n
	mov		x8, x6

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	bl inner_store_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4





//                                      w0        x1         x2         x3         x4         x5         x6
// void kernel_dtrsm_nt_rl_inv_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.align	4
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4, %function
kernel_dtrsm_nt_rl_inv_4x4_lib4:



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
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
	bl inner_scale_11_4x4_lib4
#endif



	// solution
	mov		x8, x5 // E
	mov		x9, x6 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X4_LIB4
#else
	bl inner_edge_trsm_rlt_inv_4x4_lib4
#endif



	// store
	mov		x8, x4

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	bl inner_store_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_lib4





//                                  w0        x1         x2         x3         x4         x5
// void kernel_dpotrf_nt_l_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, double *inv_diag_D);

	.align	4
	.globl kernel_dpotrf_nt_l_4x4_lib4
	.type kernel_dpotrf_nt_l_4x4_lib4, %function
kernel_dpotrf_nt_l_4x4_lib4:



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



	// call inner kernel syrk l nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_SUB_NT_4X4_LIB4
#else
	bl	inner_kernel_syrk_l_sub_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
	bl inner_scale_11_4x4_lib4
#endif



	// factorization
	mov		x8, x5 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_4X4_LIB4
#else
	bl inner_edge_potrf_4x4_lib4
#endif



	// store l
	mov		x8, x4

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
	bl inner_store_l_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dpotrf_nt_l_4x4_lib4, .-kernel_dpotrf_nt_l_4x4_lib4





