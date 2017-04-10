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
	sub sp, sp, #(11 * 16); \
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
// x10  <- sda
// x11  <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
	.align	4
	.type inner_kernel_dgemm_add_nt_8x4_lib4, %function
inner_kernel_dgemm_add_nt_8x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x12, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1   {v24.2d, v25.2d}, [x9], #32
	ld1   {v28.2d, v29.2d}, [x11], #32

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x12, #32]

	// main loop
1:
	
	// unroll 0
	fmla	v0.2d, v24.2d, v28.2d[0]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v1.2d, v25.2d, v28.2d[0]
	ld1		{v30.2d, v31.2d}, [x11], #32
	fmla	v2.2d, v24.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #96]
	fmla	v4.2d, v24.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x12, #96]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	fmla	v8.2d, v26.2d, v28.2d[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]
	sub		w8, w8, #4

	// unroll 1
	fmla	v0.2d, v24.2d, v30.2d[0]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v1.2d, v25.2d, v30.2d[0]
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	fmla	v5.2d, v25.2d, v31.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v9.2d, v27.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v13.2d, v27.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]
	fmla	v15.2d, v27.2d, v31.2d[1]

	// unroll 2
	fmla	v0.2d, v24.2d, v28.2d[0]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v1.2d, v25.2d, v28.2d[0]
	ld1		{v30.2d, v31.2d}, [x11], #32
	fmla	v2.2d, v24.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #96]
	fmla	v4.2d, v24.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x12, #96]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	fmla	v8.2d, v26.2d, v28.2d[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	// unroll 3

	fmla	v0.2d, v24.2d, v30.2d[0]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v1.2d, v25.2d, v30.2d[0]
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	fmla	v5.2d, v25.2d, v31.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v9.2d, v27.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v13.2d, v27.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]
	fmla	v15.2d, v27.2d, v31.2d[1]

	cmp		w8, #4
	bgt		1b

	sub		x9, x9, #32
	sub		x11, x11, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v28.2d, v29.2d}, [x11], #32
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	// unroll 1
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	// unroll 3
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]
	ld1		{v26.2d, v27.2d}, [x12], #32
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[1]
	fmla	v11.2d, v27.2d, v28.2d[1]
	fmla	v12.2d, v26.2d, v29.2d[0]
	fmla	v13.2d, v27.2d, v29.2d[0]
	fmla	v14.2d, v26.2d, v29.2d[1]
	fmla	v15.2d, v27.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_dgemm_add_nt_8x4_lib4, .-inner_kernel_dgemm_add_nt_8x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- sdc
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_8X4_LIB4
#else
	.align	4
	.type inner_scale_ab_8x4_lib4, %function
inner_scale_ab_8x4_lib4:
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

	ld1		{v28.2d}, [x9]

	add		x12, x10, x11

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

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	fmla	v8.2d, v24.2d, v28.2d[0]
	fmla	v9.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	fmla	v12.2d, v24.2d, v28.2d[0]
	fmla	v13.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_8x4_lib4, .-inner_scale_ab_8x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_8X4_LIB4
#else
	.align 4
	.type inner_store_8x4_lib4, %function
inner_store_8x4_lib4:
#endif

	add		x10, x8, x9

	st1		{v0.2d, v1.2d, v2.2d, v3.2d}, [x8], #64
	st1		{v4.2d, v5.2d, v6.2d, v7.2d}, [x8], #64
	st1		{v8.2d, v9.2d, v10.2d, v11.2d}, [x10], #64
	st1		{v12.2d, v13.2d, v14.2d, v15.2d}, [x10], #64

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_8x4_lib4, .-inner_store_8x4_lib4
#endif





//                               w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dgemm_nt_8x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)

	.align	4
	.global	kernel_dgemm_nt_8x4_lib4
	.type	kernel_dgemm_nt_8x4_lib4, %function
kernel_dgemm_nt_8x4_lib4:
	


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



	// call inner kernel dgemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_dgemm_add_nt_8x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // C
	lsl		w11, w11, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
	bl inner_scale_ab_8x4_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // sdd
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
	bl inner_store_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret


