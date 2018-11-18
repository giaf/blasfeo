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
	fmov    d1, d0; \
	fmov    d2, d0; \
	fmov    d3, d0; \
	fmov    d4, d0; \
	fmov    d5, d0; \
	fmov    d6, d0; \
	fmov    d7, d0; \
	fmov    d8, d0; \
	fmov    d9, d0; \
	fmov    d10, d0; \
	fmov    d11, d0; \
	fmov    d12, d0; \
	fmov    d13, d0; \
	fmov    d14, d0; \
	fmov    d15, d0





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
	.macro INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_8x4_lib4)
#endif

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x12, #0]

	// preload
	ldp		d24, d25, [x11], #16
	ldp		d26, d27, [x11], #16
	ldp		q16, q17, [x9], #32
	ldp		q20, q21, [x12], #32

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x12, #32]

	// main loop
1:
	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
//	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
//	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
//	ldr		q16, [x9], #16
	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
//	ldr		q17, [x9], #16
	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
//	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
//	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
//	cmp		w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x9, x9, #32
	sub		x11, x11, #32
	sub		x12, x12, #32

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[0]
	fmla	v13.2d, v23.2d, v29.2d[0]
	fmla	v14.2d, v22.2d, v29.2d[1]
	fmla	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]
	add		x12, x9, x10
	prfm	PLDL1KEEP, [x12, #0]

	// preload
	ldr		d16, [x9, #(0*8+0*32)] // A0
	ldr		x16, [x9, #(1*8+0*32)] // A0
	ldr		d24, [x11, #(0*8+0*32)] // B
	ldr		x24, [x11, #(1*8+0*32)] // B
	ldr		d17, [x9, #(2*8+0*32)] // A0
	ins		v16.d[1], x16
	ldr		x17, [x9, #(3*8+0*32)] // A0
	ldr		d25, [x11, #(2*8+0*32)] // B
	ins		v24.d[1], x24
	ldr		x25, [x11, #(3*8+0*32)] // B
	ldr		d20, [x12, #(0*8+0*32)] // A1
	ins		v17.d[1], x17
	ldr		x20, [x12, #(1*8+0*32)] // A1
	ldr		d21, [x12, #(2*8+0*32)] // A1
	ins		v25.d[1], x25
	ldr		x21, [x12, #(3*8+0*32)] // A1

	// prefetch
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x11, #64]
	prfm	PLDL1KEEP, [x12, #64]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// main loop
1:
	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+1*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11, #128]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x12, #128]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+1*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+1*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+2*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+2*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+2*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+2*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+2*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+3*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11, #192]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x12, #192]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+3*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+3*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	add		x9, x9, #128
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+4*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+4*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	add		x11, x11, #128
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+0*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	add		x12, x12, #128
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+0*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+0*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	cmp		w8, #4
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+0*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+0*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+0*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+0*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+1*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #128]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x12, #128]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+1*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+1*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+2*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+2*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+2*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+2*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+2*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+3*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #192]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x12, #192]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+3*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+3*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
//	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
//	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	add		x9, x9, #128
	fmla	v1.2d, v19.2d, v26.2d[0]
//	ldr		d24, [x11, #(0*8+4*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
//	ldr		x24, [x11, #(1*8+4*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	add		x11, x11, #128
	fmla	v6.2d, v18.2d, v27.2d[1]
//	ldr		d17, [x9, #(2*8+0*32)] // A0
//	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
//	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	add		x12, x12, #128
	fmla	v8.2d, v22.2d, v26.2d[0]
//	ldr		d25, [x11, #(2*8+0*32)] // B
//	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
//	ldr		x25, [x11, #(3*8+0*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
//	cmp		w8, #4
	fmla	v11.2d, v23.2d, v26.2d[1]
//	ldr		d20, [x12, #(0*8+0*32)] // A1
//	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
//	ldr		x20, [x12, #(1*8+0*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
//	ldr		x21, [x12, #(3*8+0*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
//	ldr		d21, [x12, #(2*8+0*32)] // A1
//	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x12, x12, #32

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[0]
	fmla	v13.2d, v23.2d, v29.2d[0]
	fmla	v14.2d, v22.2d, v29.2d[1]
	fmla	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11   <- B
// x12   <- 32*sdb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_8X4_LIB4
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_8x4_lib4)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x11, #64]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
	ldp		q24, q25, [x11, #0]
	ldp		q26, q27, [x11, #32]
	ldp		q28, q29, [x11, #64]
	ldp		q30, q31, [x11, #96]
	ldp		q16, q17, [x9, #0]
	ldp		q20, q21, [x13, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop


	// prefetch
	add		x14, x12, #32

//	prfm	PLDL1KEEP, [x11, x12]
//	prfm	PLDL1KEEP, [x11, x14]


	// main loop
1:
	
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #32]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #32]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v10.2d, v20.2d, v26.2d[0]
	fmla	v11.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v12.2d, v20.2d, v28.2d[0]
	fmla	v13.2d, v21.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]
	prfm	PLDL1KEEP, [x11, x12]
	fmla	v14.2d, v20.2d, v30.2d[0]
	fmla	v15.2d, v21.2d, v30.2d[0]
	prfm	PLDL1KEEP, [x11, x14]

	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #64]
	fmla	v8.2d, v22.2d, v24.2d[1]
	fmla	v9.2d, v23.2d, v24.2d[1]
	ldp		q20, q21, [x13, #64]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v28.2d[1]
	fmla	v13.2d, v23.2d, v28.2d[1]
	add		x11, x11, x12
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4
	fmla	v14.2d, v22.2d, v30.2d[1]
	fmla	v15.2d, v23.2d, v30.2d[1]

	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	ldp		q18, q19, [x9, #96]
	fmla	v8.2d, v20.2d, v25.2d[0]
	fmla	v9.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x13, #96]
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	fmla	v10.2d, v20.2d, v27.2d[0]
	fmla	v11.2d, v21.2d, v27.2d[0]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	ldp		q16, q17, [x9, #0]
	fmla	v14.2d, v20.2d, v31.2d[0]
	fmla	v15.2d, v21.2d, v31.2d[0]
	ldp		q20, q21, [x13, #0]

	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
	fmla	v8.2d, v22.2d, v25.2d[1]
	fmla	v9.2d, v23.2d, v25.2d[1]
	ldp		q24, q25, [x11, #0]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
	fmla	v10.2d, v22.2d, v27.2d[1]
	fmla	v11.2d, v23.2d, v27.2d[1]
	ldp		q26, q27, [x11, #32]
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[1]
	fmla	v13.2d, v23.2d, v29.2d[1]
	ldp		q28, q29, [x11, #64]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]
	ldp		q30, q31, [x11, #96]

	cmp		w8, #4
	bgt		1b

//	sub		x9, x9, #32
//	sub		x11, x11, #32

0:

	cmp		w8, #3
	ble		4f

	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #32]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #32]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x11, x12]
	fmla	v10.2d, v20.2d, v26.2d[0]
	fmla	v11.2d, v21.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x11, x14]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v12.2d, v20.2d, v28.2d[0]
	fmla	v13.2d, v21.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x13, #128]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v14.2d, v20.2d, v30.2d[0]
	fmla	v15.2d, v21.2d, v30.2d[0]
//	prfm	PLDL1KEEP, [x13, #192]

	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #64]
	fmla	v8.2d, v22.2d, v24.2d[1]
	fmla	v9.2d, v23.2d, v24.2d[1]
	ldp		q20, q21, [x13, #64]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v28.2d[1]
	fmla	v13.2d, v23.2d, v28.2d[1]
	add		x11, x11, x12
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4
	fmla	v14.2d, v22.2d, v30.2d[1]
	fmla	v15.2d, v23.2d, v30.2d[1]

	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	ldp		q18, q19, [x9, #96]
	fmla	v8.2d, v20.2d, v25.2d[0]
	fmla	v9.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x13, #96]
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	fmla	v10.2d, v20.2d, v27.2d[0]
	fmla	v11.2d, v21.2d, v27.2d[0]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
//	ldp		q16, q17, [x9, #0]
	fmla	v14.2d, v20.2d, v31.2d[0]
	fmla	v15.2d, v21.2d, v31.2d[0]
//	ldp		q20, q21, [x13, #0]

	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
	fmla	v8.2d, v22.2d, v25.2d[1]
	fmla	v9.2d, v23.2d, v25.2d[1]
//	ldp		q24, q25, [x11, #0]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
	fmla	v10.2d, v22.2d, v27.2d[1]
	fmla	v11.2d, v23.2d, v27.2d[1]
//	ldp		q26, q27, [x11, #32]
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[1]
	fmla	v13.2d, v23.2d, v29.2d[1]
//	ldp		q28, q29, [x11, #64]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #96]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldp		q26, q27, [x13, #0]
	ldr		d28, [x11, #0]
	ldr		d29, [x11, #32]
	ldr		d30, [x11, #64]
	ldr		d31, [x11, #96]

	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	fmla	v10.2d, v26.2d, v29.2d[0]
	fmla	v11.2d, v27.2d, v29.2d[0]
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v12.2d, v26.2d, v30.2d[0]
	fmla	v13.2d, v27.2d, v30.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[0]
	fmla	v15.2d, v27.2d, v31.2d[0]

	add		x9, x9, #32
	add		x13, x13, #32
	add		x11, x11, #8
	sub		w8, w8, #1

	cmp		w8, #0
	bgt		3b

2: // return



#else // cortex a53 TODO !!!!!



#if 1



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]
	add		x13, x9, x10
	prfm	PLDL1KEEP, [x13, #0]

	// preload
	ldr		d16, [x9, #(0*8+0*32)] // A0
	ldr		x16, [x9, #(1*8+0*32)] // A0
	ldr		d24, [x11, #(0*8+0*32)] // B
	ldr		x24, [x11, #(0*8+1*32)] // B
	ldr		d17, [x9, #(2*8+0*32)] // A0
	ins		v16.d[1], x16
	ldr		x17, [x9, #(3*8+0*32)] // A0
	ldr		d25, [x11, #(0*8+2*32)] // B
	ins		v24.d[1], x24
	ldr		x25, [x11, #(0*8+3*32)] // B
	ldr		d20, [x13, #(0*8+0*32)] // A1
	ins		v17.d[1], x17
	ldr		x20, [x13, #(1*8+0*32)] // A1
	ldr		d21, [x13, #(2*8+0*32)] // A1
	ins		v25.d[1], x25
	ldr		x21, [x13, #(3*8+0*32)] // A1

	// prefetch
	prfm	PLDL1KEEP, [x9, #64]
//	prfm	PLDL1KEEP, [x11, #64]
	prfm	PLDL1KEEP, [x13, #64]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// main loop
1:
	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(1*8+0*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #128]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(1*8+2*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(1*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x13, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x13, #(1*8+1*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x13, #(3*8+1*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x13, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(2*8+0*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(2*8+1*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(2*8+3*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x13, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x13, #(1*8+2*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x13, #(3*8+2*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x13, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(3*8+0*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(3*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #192]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(3*8+2*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x13, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x13, #(1*8+3*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x13, #(3*8+3*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x13, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
//	add		x9, x9, #128
	add		x11, x11, x12
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+0*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(0*8+1*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
//	add		x11, x11, #128
	add		x9, x9, #128
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+0*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	add		x13, x13, #128
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(0*8+2*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(0*8+3*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	cmp		w8, #4
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x13, #(0*8+0*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x13, #(1*8+0*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x13, #(3*8+0*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x13, #(2*8+0*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(1*8+0*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #128]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x13, #128]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(1*8+2*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(1*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x13, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x13, #(1*8+1*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x13, #(3*8+1*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x13, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(2*8+0*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(2*8+1*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	fmla	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(2*8+3*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
	fmla	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x13, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x13, #(1*8+2*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x13, #(3*8+2*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x13, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmla	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(3*8+0*32)] // B
	ins		v21.d[1], x21
	fmla	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(3*8+1*32)] // B
	fmla	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #192]
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmla	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x13, #192]
	fmla	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(3*8+2*32)] // B
	ins		v26.d[1], x26
	fmla	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmla	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmla	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x13, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmla	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x13, #(1*8+3*32)] // A1
	fmla	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x13, #(3*8+3*32)] // A1
	fmla	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x13, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
//	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmla	v0.2d, v18.2d, v26.2d[0]
//	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmla	v2.2d, v18.2d, v26.2d[1]
//	add		x9, x9, #128
	add		x11, x11, x12
	fmla	v1.2d, v19.2d, v26.2d[0]
//	ldr		d24, [x11, #(0*8+0*32)] // B
	ins		v23.d[1], x23
	fmla	v3.2d, v19.2d, v26.2d[1]
//	ldr		x24, [x11, #(0*8+1*32)] // B
	fmla	v4.2d, v18.2d, v27.2d[0]
//	add		x11, x11, #128
	add		x9, x9, #128
	fmla	v6.2d, v18.2d, v27.2d[1]
//	ldr		d17, [x9, #(2*8+0*32)] // A0
//	ins		v16.d[1], x16
	fmla	v5.2d, v19.2d, v27.2d[0]
//	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmla	v7.2d, v19.2d, v27.2d[1]
	add		x13, x13, #128
	fmla	v8.2d, v22.2d, v26.2d[0]
//	ldr		d25, [x11, #(0*8+2*32)] // B
//	ins		v24.d[1], x24
	fmla	v10.2d, v22.2d, v26.2d[1]
//	ldr		x25, [x11, #(0*8+3*32)] // B
	fmla	v9.2d, v23.2d, v26.2d[0]
//	cmp		w8, #4
	fmla	v11.2d, v23.2d, v26.2d[1]
//	ldr		d20, [x13, #(0*8+0*32)] // A1
//	ins		v17.d[1], x17
	fmla	v12.2d, v22.2d, v27.2d[0]
//	ldr		x20, [x13, #(1*8+0*32)] // A1
	fmla	v14.2d, v22.2d, v27.2d[1]
//	ldr		x21, [x13, #(3*8+0*32)] // A1
	fmla	v13.2d, v23.2d, v27.2d[0]
//	ldr		d21, [x13, #(2*8+0*32)] // A1
//	ins		v25.d[1], x25
	fmla	v15.2d, v23.2d, v27.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x13, x13, #32

3: // clean1-up loop

	// unroll 0
	ldr		d28, [x11, #(0*8+0*32)]
	ldr		d29, [x11, #(0*8+1*32)]
	ldr		d30, [x11, #(0*8+2*32)]
	ldr		d31, [x11, #(0*8+3*32)]
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v22.2d, v23.2d}, [x13], #32
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v29.2d[0]
	fmla	v3.2d, v21.2d, v29.2d[0]
	fmla	v4.2d, v20.2d, v30.2d[0]
	fmla	v5.2d, v21.2d, v30.2d[0]
	fmla	v6.2d, v20.2d, v31.2d[0]
	fmla	v7.2d, v21.2d, v31.2d[0]
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	add		x11, x11, #8

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return






#else



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	// prefetch
//	prfm	PLDL1KEEP, [x9, #0]
//	prfm	PLDL1KEEP, [x11, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload

	// prefetch
//	prfm	PLDL1KEEP, [x9, #32]
//	prfm	PLDL1KEEP, [x11, #32]

	add		x14, x12, #32

	// main loop
1:

	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x13, #128]
	prfm	PLDL1KEEP, [x11, x12]
	prfm	PLDL1KEEP, [x11, x14]

	// unroll 0 & 1
	ldr		q28, [x11, #0]
	ldr		q29, [x11, #32]
	ldr		q30, [x11, #64]
	ldr		q31, [x11, #96]

	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v18.2d, v19.2d}, [x13], #32
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v22.2d, v23.2d}, [x13], #32

	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	fmla	v8.2d, v18.2d, v28.2d[0]
	fmla	v9.2d, v19.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v29.2d[0]
	fmla	v3.2d, v17.2d, v29.2d[0]
	fmla	v10.2d, v18.2d, v29.2d[0]
	fmla	v11.2d, v19.2d, v29.2d[0]
	fmla	v4.2d, v16.2d, v30.2d[0]
	fmla	v5.2d, v17.2d, v30.2d[0]
	fmla	v12.2d, v18.2d, v30.2d[0]
	fmla	v13.2d, v19.2d, v30.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	fmla	v14.2d, v18.2d, v31.2d[0]
	fmla	v15.2d, v19.2d, v31.2d[0]

	fmla	v0.2d, v20.2d, v28.2d[1]
	fmla	v1.2d, v21.2d, v28.2d[1]
	fmla	v8.2d, v22.2d, v28.2d[1]
	fmla	v9.2d, v23.2d, v28.2d[1]
	fmla	v2.2d, v20.2d, v29.2d[1]
	fmla	v3.2d, v21.2d, v29.2d[1]
	fmla	v10.2d, v22.2d, v29.2d[1]
	fmla	v11.2d, v23.2d, v29.2d[1]
	fmla	v4.2d, v20.2d, v30.2d[1]
	fmla	v5.2d, v21.2d, v30.2d[1]
	fmla	v12.2d, v22.2d, v30.2d[1]
	fmla	v13.2d, v23.2d, v30.2d[1]
	fmla	v6.2d, v20.2d, v31.2d[1]
	fmla	v7.2d, v21.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]

	prfm	PLDL1KEEP, [x9, #192]
	prfm	PLDL1KEEP, [x13, #192]

	// unroll 2 & 3
	ldr		q28, [x11, #16]
	ldr		q29, [x11, #48]
	ldr		q30, [x11, #80]
	ldr		q31, [x11, #112]

	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v18.2d, v19.2d}, [x13], #32
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v22.2d, v23.2d}, [x13], #32

	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	fmla	v8.2d, v18.2d, v28.2d[0]
	fmla	v9.2d, v19.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v29.2d[0]
	fmla	v3.2d, v17.2d, v29.2d[0]
	fmla	v10.2d, v18.2d, v29.2d[0]
	fmla	v11.2d, v19.2d, v29.2d[0]
	fmla	v4.2d, v16.2d, v30.2d[0]
	fmla	v5.2d, v17.2d, v30.2d[0]
	fmla	v12.2d, v18.2d, v30.2d[0]
	fmla	v13.2d, v19.2d, v30.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	fmla	v14.2d, v18.2d, v31.2d[0]
	fmla	v15.2d, v19.2d, v31.2d[0]

	fmla	v0.2d, v20.2d, v28.2d[1]
	fmla	v1.2d, v21.2d, v28.2d[1]
	fmla	v8.2d, v22.2d, v28.2d[1]
	fmla	v9.2d, v23.2d, v28.2d[1]
	fmla	v2.2d, v20.2d, v29.2d[1]
	fmla	v3.2d, v21.2d, v29.2d[1]
	fmla	v10.2d, v22.2d, v29.2d[1]
	fmla	v11.2d, v23.2d, v29.2d[1]
	fmla	v4.2d, v20.2d, v30.2d[1]
	fmla	v5.2d, v21.2d, v30.2d[1]
	fmla	v12.2d, v22.2d, v30.2d[1]
	fmla	v13.2d, v23.2d, v30.2d[1]
	fmla	v6.2d, v20.2d, v31.2d[1]
	fmla	v7.2d, v21.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]

	add		x11, x11, x12
	sub		w8, w8, #4

	cmp		w8, #4
	bgt		1b

//	sub		x9, x9, #32
//	sub		x11, x11, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0 & 1
	ldr		q28, [x11, #0]
	ldr		q29, [x11, #32]
	ldr		q30, [x11, #64]
	ldr		q31, [x11, #96]

	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v18.2d, v19.2d}, [x13], #32
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v22.2d, v23.2d}, [x13], #32

	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	fmla	v8.2d, v18.2d, v28.2d[0]
	fmla	v9.2d, v19.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v29.2d[0]
	fmla	v3.2d, v17.2d, v29.2d[0]
	fmla	v10.2d, v18.2d, v29.2d[0]
	fmla	v11.2d, v19.2d, v29.2d[0]
	fmla	v4.2d, v16.2d, v30.2d[0]
	fmla	v5.2d, v17.2d, v30.2d[0]
	fmla	v12.2d, v18.2d, v30.2d[0]
	fmla	v13.2d, v19.2d, v30.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	fmla	v14.2d, v18.2d, v31.2d[0]
	fmla	v15.2d, v19.2d, v31.2d[0]

	fmla	v0.2d, v20.2d, v28.2d[1]
	fmla	v1.2d, v21.2d, v28.2d[1]
	fmla	v8.2d, v22.2d, v28.2d[1]
	fmla	v9.2d, v23.2d, v28.2d[1]
	fmla	v2.2d, v20.2d, v29.2d[1]
	fmla	v3.2d, v21.2d, v29.2d[1]
	fmla	v10.2d, v22.2d, v29.2d[1]
	fmla	v11.2d, v23.2d, v29.2d[1]
	fmla	v4.2d, v20.2d, v30.2d[1]
	fmla	v5.2d, v21.2d, v30.2d[1]
	fmla	v12.2d, v22.2d, v30.2d[1]
	fmla	v13.2d, v23.2d, v30.2d[1]
	fmla	v6.2d, v20.2d, v31.2d[1]
	fmla	v7.2d, v21.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]

	// unroll 2 & 3
	ldr		q28, [x11, #16]
	ldr		q29, [x11, #48]
	ldr		q30, [x11, #80]
	ldr		q31, [x11, #112]

	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v18.2d, v19.2d}, [x13], #32
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v22.2d, v23.2d}, [x13], #32

	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	fmla	v8.2d, v18.2d, v28.2d[0]
	fmla	v9.2d, v19.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v29.2d[0]
	fmla	v3.2d, v17.2d, v29.2d[0]
	fmla	v10.2d, v18.2d, v29.2d[0]
	fmla	v11.2d, v19.2d, v29.2d[0]
	fmla	v4.2d, v16.2d, v30.2d[0]
	fmla	v5.2d, v17.2d, v30.2d[0]
	fmla	v12.2d, v18.2d, v30.2d[0]
	fmla	v13.2d, v19.2d, v30.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	fmla	v14.2d, v18.2d, v31.2d[0]
	fmla	v15.2d, v19.2d, v31.2d[0]

	fmla	v0.2d, v20.2d, v28.2d[1]
	fmla	v1.2d, v21.2d, v28.2d[1]
	fmla	v8.2d, v22.2d, v28.2d[1]
	fmla	v9.2d, v23.2d, v28.2d[1]
	fmla	v2.2d, v20.2d, v29.2d[1]
	fmla	v3.2d, v21.2d, v29.2d[1]
	fmla	v10.2d, v22.2d, v29.2d[1]
	fmla	v11.2d, v23.2d, v29.2d[1]
	fmla	v4.2d, v20.2d, v30.2d[1]
	fmla	v5.2d, v21.2d, v30.2d[1]
	fmla	v12.2d, v22.2d, v30.2d[1]
	fmla	v13.2d, v23.2d, v30.2d[1]
	fmla	v6.2d, v20.2d, v31.2d[1]
	fmla	v7.2d, v21.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]

	add		x11, x11, x12
	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldr		d28, [x11, #0]
	ldr		d29, [x11, #32]
	ldr		d30, [x11, #64]
	ldr		d31, [x11, #96]

	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v26.2d, v27.2d}, [x13], #32

	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	fmla	v10.2d, v26.2d, v29.2d[0]
	fmla	v11.2d, v27.2d, v29.2d[0]
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v12.2d, v26.2d, v30.2d[0]
	fmla	v13.2d, v27.2d, v30.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[0]
	fmla	v15.2d, v27.2d, v31.2d[0]

	add		x11, x11, #8
	sub		w8, w8, #1

	cmp		w8, #0
	bgt		3b

2: // return



#endif



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_8x4_lib4)
#endif





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
	.macro INNER_KERNEL_SYRK_L_ADD_NT_8X4_LIB4
#else
	.align	4
	FUN_START(inner_kernel_syrk_l_add_nt_8x4_lib4)
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

	// preload
	ldp		d24, d25, [x11], #16
	ldp		d26, d27, [x11], #16
	ldp		q16, q17, [x9], #32
	ldp		q20, q21, [x12], #32

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x12, #32]

	// main loop
1:
	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmla	v2.2d, v16.2d, v25.2d[0]
	fmla	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmla	v4.2d, v16.2d, v26.2d[0]
	fmla	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmla	v10.2d, v20.2d, v25.2d[0]
	fmla	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmla	v12.2d, v20.2d, v26.2d[0]
	fmla	v13.2d, v21.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmla	v14.2d, v20.2d, v27.2d[0]
	fmla	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
//	ldp		d24, d25, [x11], #16
	fmla	v0.2d, v18.2d, v28.2d[0]
	fmla	v1.2d, v19.2d, v28.2d[0]
//	ldp		d26, d27, [x11], #16
	fmla	v2.2d, v18.2d, v29.2d[0]
	fmla	v3.2d, v19.2d, v29.2d[0]
//	ldr		q16, [x9], #16
//	fmla	v4.2d, v18.2d, v30.2d[0]
	fmla	v5.2d, v19.2d, v30.2d[0]
//	ldr		q17, [x9], #16
//	fmla	v6.2d, v18.2d, v31.2d[0]
	fmla	v7.2d, v19.2d, v31.2d[0]
//	ldr		q20, [x12], #16
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
//	ldr		q21, [x12], #16
	fmla	v10.2d, v22.2d, v29.2d[0]
	fmla	v11.2d, v23.2d, v29.2d[0]
//	cmp		w8, #4
	fmla	v12.2d, v22.2d, v30.2d[0]
	fmla	v13.2d, v23.2d, v30.2d[0]
	fmla	v14.2d, v22.2d, v31.2d[0]
	fmla	v15.2d, v23.2d, v31.2d[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x9, x9, #32
	sub		x11, x11, #32
	sub		x12, x12, #32

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11], #32
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
//	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[0]
	fmla	v13.2d, v23.2d, v29.2d[0]
	fmla	v14.2d, v22.2d, v29.2d[1]
	fmla	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_syrk_l_add_nt_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11   <- B
// x12   <- 32*sdb
// w13   <- offsetB
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_GEMM_ADD_NN_8X4_LIB4
#else
	.align	4
	FUN_START(inner_edge_gemm_add_nn_8x4_lib4)
#endif

	cmp		w13, #0
	ble		2f // return

	cmp		w8, #0
	ble		2f // return

	mov		w14, #4
	sub		w15, w14, w13 // 4-offsetB
	cmp		w15, w8
	ble		0f
	mov		w15, w8 // kend=min(k,4-offsetB(
0:
//	movgt	w15, w8 // kend=min(k,4-offsetB(
	
	add		x11, x11, x13, LSL #3 // B + offsetB*sizeof(double)

	add		x14, x9, x10

1:
	ldp		q24, q25, [x9, #0]
	ldp		q26, q27, [x14, #0]
	ldr		d28, [x11, #0]
	ldr		d29, [x11, #32]
	ldr		d30, [x11, #64]
	ldr		d31, [x11, #96]

	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	fmla	v10.2d, v26.2d, v29.2d[0]
	fmla	v11.2d, v27.2d, v29.2d[0]
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v12.2d, v26.2d, v30.2d[0]
	fmla	v13.2d, v27.2d, v30.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[0]
	fmla	v15.2d, v27.2d, v31.2d[0]

	add		x9, x9, #32
	add		x14, x14, #32
	add		x11, x11, #8
	sub		w8, w8, #1

	sub		w15, w15, #1

	cmp		w15, #0
	bgt		1b

	cmp		w8, #0
	ble		2f // return

	add		x11, x11, x12
	sub		x11, x11, #32

2: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_gemm_add_nn_8x4_lib4)
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
	.macro INNER_EDGE_TRSM_RLT_INV_8X4_LIB4
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_8x4_lib4)
#endif
	
	// first column
	ldr			d16, [x9, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]
	fmul		v8.2d, v8.2d, v16.2d[0]
	fmul		v9.2d, v9.2d, v16.2d[0]
	ldr			d16, [x8, #8] // E[1+4*0]
	fmls		v2.2d, v0.2d, v16.2d[0]
	fmls		v3.2d, v1.2d, v16.2d[0]
	fmls		v10.2d, v8.2d, v16.2d[0]
	fmls		v11.2d, v9.2d, v16.2d[0]
	ldr			d16, [x8, #16] // E[2+4*0]
	fmls		v4.2d, v0.2d, v16.2d[0]
	fmls		v5.2d, v1.2d, v16.2d[0]
	fmls		v12.2d, v8.2d, v16.2d[0]
	fmls		v13.2d, v9.2d, v16.2d[0]
	ldr			d16, [x8, #24] // E[3+4*0]
	fmls		v6.2d, v0.2d, v16.2d[0]
	fmls		v7.2d, v1.2d, v16.2d[0]
	fmls		v14.2d, v8.2d, v16.2d[0]
	fmls		v15.2d, v9.2d, v16.2d[0]

	ldr			d16, [x9, #8] // E_inv[1]
	fmul		v2.2d, v2.2d, v16.2d[0]
	fmul		v3.2d, v3.2d, v16.2d[0]
	fmul		v10.2d, v10.2d, v16.2d[0]
	fmul		v11.2d, v11.2d, v16.2d[0]
	ldr			d16, [x8, #48] // E[2+4*1]
	fmls		v4.2d, v2.2d, v16.2d[0]
	fmls		v5.2d, v3.2d, v16.2d[0]
	fmls		v12.2d, v10.2d, v16.2d[0]
	fmls		v13.2d, v11.2d, v16.2d[0]
	ldr			d16, [x8, #56] // E[3+4*1]
	fmls		v6.2d, v2.2d, v16.2d[0]
	fmls		v7.2d, v3.2d, v16.2d[0]
	fmls		v14.2d, v10.2d, v16.2d[0]
	fmls		v15.2d, v11.2d, v16.2d[0]

	ldr			d16, [x9, #16] // E_inv[2]
	fmul		v4.2d, v4.2d, v16.2d[0]
	fmul		v5.2d, v5.2d, v16.2d[0]
	fmul		v12.2d, v12.2d, v16.2d[0]
	fmul		v13.2d, v13.2d, v16.2d[0]
	ldr			d16, [x8, #88] // E[3+4*1]
	fmls		v6.2d, v4.2d, v16.2d[0]
	fmls		v7.2d, v5.2d, v16.2d[0]
	fmls		v14.2d, v12.2d, v16.2d[0]
	fmls		v15.2d, v13.2d, v16.2d[0]

	ldr			d16, [x9, #24] // E_inv[2]
	fmul		v6.2d, v6.2d, v16.2d[0]
	fmul		v7.2d, v7.2d, v16.2d[0]
	fmul		v14.2d, v14.2d, v16.2d[0]
	fmul		v15.2d, v15.2d, v16.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_8x4_lib4)
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
// w10  <- n1
//
// output arguments:
// x8   <- E
// x9   <- inv_diag_E
// w10  <- n1

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_RLT_INV_8X4_VS_LIB4
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_8x4_vs_lib4)
#endif
	
	// first column
	ldr			d16, [x9, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]
	fmul		v8.2d, v8.2d, v16.2d[0]
	fmul		v9.2d, v9.2d, v16.2d[0]
	cmp		w10, #2
	blt		0f // return

	// second column
	ldr			d16, [x8, #8] // E[1+4*0]
	fmls		v2.2d, v0.2d, v16.2d[0]
	fmls		v3.2d, v1.2d, v16.2d[0]
	fmls		v10.2d, v8.2d, v16.2d[0]
	fmls		v11.2d, v9.2d, v16.2d[0]
	ldr			d16, [x9, #8] // E_inv[1]
	fmul		v2.2d, v2.2d, v16.2d[0]
	fmul		v3.2d, v3.2d, v16.2d[0]
	fmul		v10.2d, v10.2d, v16.2d[0]
	fmul		v11.2d, v11.2d, v16.2d[0]
	cmp		w10, #3
	blt		0f // return

	// third column
	ldr			d16, [x8, #16] // E[2+4*0]
	fmls		v4.2d, v0.2d, v16.2d[0]
	fmls		v5.2d, v1.2d, v16.2d[0]
	fmls		v12.2d, v8.2d, v16.2d[0]
	fmls		v13.2d, v9.2d, v16.2d[0]
	ldr			d16, [x8, #48] // E[2+4*1]
	fmls		v4.2d, v2.2d, v16.2d[0]
	fmls		v5.2d, v3.2d, v16.2d[0]
	fmls		v12.2d, v10.2d, v16.2d[0]
	fmls		v13.2d, v11.2d, v16.2d[0]
	ldr			d16, [x9, #16] // E_inv[2]
	fmul		v4.2d, v4.2d, v16.2d[0]
	fmul		v5.2d, v5.2d, v16.2d[0]
	fmul		v12.2d, v12.2d, v16.2d[0]
	fmul		v13.2d, v13.2d, v16.2d[0]
	cmp		w10, #4
	blt		0f // return

	// forth column
	ldr			d16, [x8, #24] // E[3+4*0]
	fmls		v6.2d, v0.2d, v16.2d[0]
	fmls		v7.2d, v1.2d, v16.2d[0]
	fmls		v14.2d, v8.2d, v16.2d[0]
	fmls		v15.2d, v9.2d, v16.2d[0]
	ldr			d16, [x8, #56] // E[3+4*1]
	fmls		v6.2d, v2.2d, v16.2d[0]
	fmls		v7.2d, v3.2d, v16.2d[0]
	fmls		v14.2d, v10.2d, v16.2d[0]
	fmls		v15.2d, v11.2d, v16.2d[0]
	ldr			d16, [x8, #88] // E[3+4*1]
	fmls		v6.2d, v4.2d, v16.2d[0]
	fmls		v7.2d, v5.2d, v16.2d[0]
	fmls		v14.2d, v12.2d, v16.2d[0]
	fmls		v15.2d, v13.2d, v16.2d[0]
	ldr			d16, [x9, #24] // E_inv[2]
	fmul		v6.2d, v6.2d, v16.2d[0]
	fmul		v7.2d, v7.2d, v16.2d[0]
	fmul		v14.2d, v14.2d, v16.2d[0]
	fmul		v15.2d, v15.2d, v16.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_8x4_vs_lib4)
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
	.macro INNER_EDGE_POTRF_8X4_LIB4
#else
	.p2align 4
	FUN_START(inner_edge_potrf_8x4_lib4)
#endif
	
	fmov		d16, 1.0e+0 // 1.0

	// first column
	ins			v17.d[0], v0.d[0]
	fcmpe		d17, #0
	ble			1f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
2:
	str			d18, [x8, #0]
	fmul		v0.2d, v0.2d, v18.2d[0]
	fmul		v1.2d, v1.2d, v18.2d[0]
	fmul		v8.2d, v8.2d, v18.2d[0]
	fmul		v9.2d, v9.2d, v18.2d[0]
	fmls		v2.2d, v0.2d, v0.2d[1]
	fmls		v3.2d, v1.2d, v0.2d[1]
	fmls		v10.2d, v8.2d, v0.2d[1]
	fmls		v11.2d, v9.2d, v0.2d[1]
	fmls		v5.2d, v1.2d, v1.2d[0]
	fmls		v12.2d, v8.2d, v1.2d[0]
	fmls		v13.2d, v9.2d, v1.2d[0]
	fmls		v7.2d, v1.2d, v1.2d[1]
	fmls		v14.2d, v8.2d, v1.2d[1]
	fmls		v15.2d, v9.2d, v1.2d[1]

	// second column
	ins			v17.d[0], v2.d[1]
	fcmpe		d17, #0
	ble			3f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
4:
	str			d18, [x8, #8]
	fmul		v2.2d, v2.2d, v18.2d[0]
	fmul		v3.2d, v3.2d, v18.2d[0]
	fmul		v10.2d, v10.2d, v18.2d[0]
	fmul		v11.2d, v11.2d, v18.2d[0]
	fmls		v5.2d, v3.2d, v3.2d[0]
	fmls		v12.2d, v10.2d, v3.2d[0]
	fmls		v13.2d, v11.2d, v3.2d[0]
	fmls		v7.2d, v3.2d, v3.2d[1]
	fmls		v14.2d, v10.2d, v3.2d[1]
	fmls		v15.2d, v11.2d, v3.2d[1]

	// third column
	ins			v17.d[0], v5.d[0]
	fcmpe		d17, #0
	ble			5f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
6:
	str			d18, [x8, #16]
	fmul		v5.2d, v5.2d, v18.2d[0]
	fmul		v12.2d, v12.2d, v18.2d[0]
	fmul		v13.2d, v13.2d, v18.2d[0]
	fmls		v7.2d, v5.2d, v5.2d[1]
	fmls		v14.2d, v12.2d, v5.2d[1]
	fmls		v15.2d, v13.2d, v5.2d[1]

	// fourth column
	ins			v17.d[0], v7.d[1]
	fcmpe		d17, #0
	ble			7f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
8:
	str			d18, [x8, #24]
	fmul		v7.2d, v7.2d, v18.2d[0]
	fmul		v14.2d, v14.2d, v18.2d[0]
	fmul		v15.2d, v15.2d, v18.2d[0]

	b			0f

1:
	fmov		d18, xzr
	b			2b

3:
	fmov		d18, xzr
	b			4b

5:
	fmov		d18, xzr
	b			6b

7:
	fmov		d18, xzr

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_potrf_8x4_lib4)
#endif





// subroutine
//
// cholesky factorization 
//
// input arguments:
// x8   <- inv_diag_D
// x9   <- n1
//
// output arguments:
// x8   <- inv_diag_D
// x9   <- n1

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_POTRF_8X4_VS_LIB4
#else
	.p2align 4
	FUN_START(inner_edge_potrf_8x4_vs_lib4)
#endif
	
	fmov		d16, 1.0e+0 // 1.0

	// first column
	ins			v17.d[0], v0.d[0]
	fcmpe		d17, #0
	ble			1f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
2:
	str			d18, [x8, #0]
	fmul		v0.2d, v0.2d, v18.2d[0]
	fmul		v1.2d, v1.2d, v18.2d[0]
	fmul		v8.2d, v8.2d, v18.2d[0]
	fmul		v9.2d, v9.2d, v18.2d[0]
	cmp		w9, #2
	blt		0f // return

	// second column
	fmls		v2.2d, v0.2d, v0.2d[1]
	fmls		v3.2d, v1.2d, v0.2d[1]
	fmls		v10.2d, v8.2d, v0.2d[1]
	fmls		v11.2d, v9.2d, v0.2d[1]
	ins			v17.d[0], v2.d[1]
	fcmpe		d17, #0
	ble			3f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
4:
	str			d18, [x8, #8]
	fmul		v2.2d, v2.2d, v18.2d[0]
	fmul		v3.2d, v3.2d, v18.2d[0]
	fmul		v10.2d, v10.2d, v18.2d[0]
	fmul		v11.2d, v11.2d, v18.2d[0]
	cmp		w9, #3
	blt		0f // return

	// third column
	fmls		v5.2d, v1.2d, v1.2d[0]
	fmls		v12.2d, v8.2d, v1.2d[0]
	fmls		v13.2d, v9.2d, v1.2d[0]
	fmls		v5.2d, v3.2d, v3.2d[0]
	fmls		v12.2d, v10.2d, v3.2d[0]
	fmls		v13.2d, v11.2d, v3.2d[0]
	ins			v17.d[0], v5.d[0]
	fcmpe		d17, #0
	ble			5f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
6:
	str			d18, [x8, #16]
	fmul		v5.2d, v5.2d, v18.2d[0]
	fmul		v12.2d, v12.2d, v18.2d[0]
	fmul		v13.2d, v13.2d, v18.2d[0]
	cmp		w9, #4
	blt		0f // return

	// fourth column
	fmls		v7.2d, v1.2d, v1.2d[1]
	fmls		v14.2d, v8.2d, v1.2d[1]
	fmls		v15.2d, v9.2d, v1.2d[1]
	fmls		v7.2d, v3.2d, v3.2d[1]
	fmls		v14.2d, v10.2d, v3.2d[1]
	fmls		v15.2d, v11.2d, v3.2d[1]
	fmls		v7.2d, v5.2d, v5.2d[1]
	fmls		v14.2d, v12.2d, v5.2d[1]
	fmls		v15.2d, v13.2d, v5.2d[1]
	ins			v17.d[0], v7.d[1]
	fcmpe		d17, #0
	ble			7f
	fsqrt		d17, d17
	fdiv		d18, d16, d17
8:
	str			d18, [x8, #24]
	fmul		v7.2d, v7.2d, v18.2d[0]
	fmul		v14.2d, v14.2d, v18.2d[0]
	fmul		v15.2d, v15.2d, v18.2d[0]

	b			0f

1:
	fmov		d18, xzr
	b			2b

3:
	fmov		d18, xzr
	b			4b

5:
	fmov		d18, xzr
	b			6b

7:
	fmov		d18, xzr

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_potrf_8x4_vs_lib4)
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

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_LIB4
#else
	.align	4
	FUN_START(inner_scale_ab_8x4_lib4)
#endif

	ld1		{v28.2d}, [x8]
	// TODO check on beta

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

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// x8  <- beta
// x9  <- C
// x10  <- sdc
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_8X4_LIB4
#else
	.align	4
	FUN_START(inner_scale_m1b_8x4_lib4)
#endif

	add		x11, x9, x10

	ld1		{v28.2d}, [x8]
	// TODO check on beta

	fneg	v0.2d, v0.2d
	fneg	v1.2d, v1.2d
	fneg	v2.2d, v2.2d
	fneg	v3.2d, v3.2d

	fneg	v4.2d, v4.2d
	fneg	v5.2d, v5.2d
	fneg	v6.2d, v6.2d
	fneg	v7.2d, v7.2d

	fneg	v8.2d, v8.2d
	fneg	v9.2d, v9.2d
	fneg	v10.2d, v10.2d
	fneg	v11.2d, v11.2d

	fneg	v12.2d, v12.2d
	fneg	v13.2d, v13.2d
	fneg	v14.2d, v14.2d
	fneg	v15.2d, v15.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x9], #64
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v26.2d, v28.2d[0]
	fmla	v3.2d, v27.2d, v28.2d[0]

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x9], #64
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v6.2d, v26.2d, v28.2d[0]
	fmla	v7.2d, v27.2d, v28.2d[0]

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x11], #64
	fmla	v8.2d, v24.2d, v28.2d[0]
	fmla	v9.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x11], #64
	fmla	v12.2d, v24.2d, v28.2d[0]
	fmla	v13.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- sdc
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_8X4_LIB4
#else
	.align	4
	FUN_START(inner_scale_m11_8x4_lib4)
#endif

	add		x10, x8, x9

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	fsub	v0.2d, v24.2d, v0.2d
	fsub	v1.2d, v25.2d, v1.2d
	fsub	v2.2d, v26.2d, v2.2d
	fsub	v3.2d, v27.2d, v3.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	fsub	v4.2d, v24.2d, v4.2d
	fsub	v5.2d, v25.2d, v5.2d
	fsub	v6.2d, v26.2d, v6.2d
	fsub	v7.2d, v27.2d, v7.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	fsub	v8.2d, v24.2d, v8.2d
	fsub	v9.2d, v25.2d, v9.2d
	fsub	v10.2d, v26.2d, v10.2d
	fsub	v11.2d, v27.2d, v11.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	fsub	v12.2d, v24.2d, v12.2d
	fsub	v13.2d, v25.2d, v13.2d
	fsub	v14.2d, v26.2d, v14.2d
	fsub	v15.2d, v27.2d, v15.2d

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_LIB4
#else
	.align 4
	FUN_START(inner_store_8x4_lib4)
#endif

	add		x10, x8, x9

	st1		{v0.2d, v1.2d, v2.2d, v3.2d}, [x8], #64
	st1		{v4.2d, v5.2d, v6.2d, v7.2d}, [x8], #64
	st1		{v8.2d, v9.2d, v10.2d, v11.2d}, [x10], #64
	st1		{v12.2d, v13.2d, v14.2d, v15.2d}, [x10], #64

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
// x10  <- km
// x11  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_VS_LIB4
#else
	.align 4
	FUN_START(inner_store_8x4_vs_lib4)
#endif

	add		x12, x8, x9

	cmp		w10, #8
	bge		1f

	ldp		q24, q25, [x12, #(0*8+0*32)]
	ldp		q26, q27, [x12, #(0*8+1*32)]
	ldp		q28, q29, [x12, #(0*8+2*32)]
	ldp		q30, q31, [x12, #(0*8+3*32)]

	// 4th row
	ins		v9.d[1], v25.d[1]
	ins		v11.d[1], v27.d[1]
	ins		v13.d[1], v29.d[1]
	ins		v15.d[1], v31.d[1]
	cmp		w10, #7
	bge		1f
	// 3th row
	ins		v9.d[0], v25.d[0]
	ins		v11.d[0], v27.d[0]
	ins		v13.d[0], v29.d[0]
	ins		v15.d[0], v31.d[0]
	cmp		w10, #6
	bge		1f
	// 2nd row
	ins		v8.d[1], v24.d[1]
	ins		v10.d[1], v26.d[1]
	ins		v12.d[1], v28.d[1]
	ins		v14.d[1], v30.d[1]
	cmp		w10, #5
	bge		1f
	// 1st row
	ins		v8.d[0], v24.d[0]
	ins		v10.d[0], v26.d[0]
	ins		v12.d[0], v28.d[0]
	ins		v14.d[0], v30.d[0]

1:
	// 1st col
	stp		q0, q1, [x8, #(0*8+0*32)]
	stp		q8, q9, [x12, #(0*8+0*32)]
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #(0*8+1*32)]
	stp		q10, q11, [x12, #(0*8+1*32)]
	cmp		w11, #3
	blt		0f
	// 3rd col
	stp		q4, q5, [x8, #(0*8+2*32)]
	stp		q12, q13, [x12, #(0*8+2*32)]
	beq		0f
	// 4th col
	stp		q6, q7, [x8, #(0*8+3*32)]
	stp		q14, q15, [x12, #(0*8+3*32)]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x4_vs_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_LIB4
#else
	.align 4
	FUN_START(inner_store_l_8x4_lib4)
#endif

	ldr		q16, [x8, #32]
	ldr		q17, [x8, #112]

	ins		v2.d[0], v16.d[0]
	ins		v7.d[0], v17.d[0]

	add		x10, x8, x9

	stp		q0, q1, [x8, #0]
	stp		q2, q3, [x8, #32]
	str		q5, [x8, #80]
	str		q7, [x8, #112]

	st1		{v8.2d, v9.2d, v10.2d, v11.2d}, [x10], #64
	st1		{v12.2d, v13.2d, v14.2d, v15.2d}, [x10], #64

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_8x4_lib4)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
// x10  <- km
// x11  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_VS_LIB4
#else
	.align 4
	FUN_START(inner_store_l_8x4_vs_lib4)
#endif

	add		x12, x8, x9

	cmp		w10, #8
	bge		1f

	ldp		q24, q25, [x12, #(0*8+0*32)]
	ldp		q26, q27, [x12, #(0*8+1*32)]
	ldp		q28, q29, [x12, #(0*8+2*32)]
	ldp		q30, q31, [x12, #(0*8+3*32)]

	// 4th row
	ins		v9.d[1], v25.d[1]
	ins		v11.d[1], v27.d[1]
	ins		v13.d[1], v29.d[1]
	ins		v15.d[1], v31.d[1]
	cmp		w10, #7
	bge		1f
	// 3th row
	ins		v9.d[0], v25.d[0]
	ins		v11.d[0], v27.d[0]
	ins		v13.d[0], v29.d[0]
	ins		v15.d[0], v31.d[0]
	cmp		w10, #6
	bge		1f
	// 2nd row
	ins		v8.d[1], v24.d[1]
	ins		v10.d[1], v26.d[1]
	ins		v12.d[1], v28.d[1]
	ins		v14.d[1], v30.d[1]
	cmp		w10, #5
	bge		1f
	// 1st row
	ins		v8.d[0], v24.d[0]
	ins		v10.d[0], v26.d[0]
	ins		v12.d[0], v28.d[0]
	ins		v14.d[0], v30.d[0]

1:
	ldr		q16, [x8, #32]
	ldr		q17, [x8, #112]

	ins		v2.d[0], v16.d[0]
	ins		v7.d[0], v17.d[0]

	// 1st col
	stp		q0, q1, [x8, #(0*8+0*32)]
	stp		q8, q9, [x12, #(0*8+0*32)]
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #(0*8+1*32)]
	stp		q10, q11, [x12, #(0*8+1*32)]
	cmp		w11, #3
	blt		0f
	// 3rd col
	str		q5, [x8, #80]
	stp		q12, q13, [x12, #(0*8+2*32)]
	beq		0f
	// 4th col
	str		q7, [x8, #112]
	stp		q14, q15, [x12, #(0*8+3*32)]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_8x4_vs_lib4)
#endif





//                               w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dgemm_nt_8x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
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

	FUN_END(kernel_dgemm_nt_8x4_lib4)





//                                  w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nt_8x4_vs_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_vs_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
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
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
	bl inner_store_8x4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_8x4_vs_lib4)





//                               w0        x1             x2         w3       w4           x5         w6       x7            sp+0       sp+8     sp+16      sp+24
// void kernel_dgemm_nn_8x4_lib4(int kmax, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_8x4_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x5 // B
	mov		w12, w6 // sdb
	lsl		w12, w12, #5 // 32*sdb
	mov		w13, w4 // offsetB

#if MACRO_LEVEL>=2
	INNER_EDGE_GEMM_ADD_NN_8X4_LIB4
#else
	bl	inner_edge_gemm_add_nn_8x4_lib4
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nn_8x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x7 // beta
	ldr		x10, [sp, #(STACKSIZE + 0)] // C
	ldr		w11, [sp, #(STACKSIZE + 8)] // sdc
	lsl		w11, w11, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
	bl inner_scale_ab_8x4_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 16)] // D
	ldr		w9, [sp, #(STACKSIZE + 24)] // sdd
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
	bl inner_store_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nn_8x4_lib4)





//                                  w0        x1             x2         w3       w4           x5         w6       x7            sp+0       sp+8     sp+16      sp+24    sp+32   sp+40
// void kernel_dgemm_nn_8x4_vs_lib4(int kmax, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_8x4_vs_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x5 // B
	mov		w12, w6 // sdb
	lsl		w12, w12, #5 // 32*sdb
	mov		w13, w4 // offsetB

#if MACRO_LEVEL>=2
	INNER_EDGE_GEMM_ADD_NN_8X4_LIB4
#else
	bl	inner_edge_gemm_add_nn_8x4_lib4
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nn_8x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x7 // beta
	ldr		x10, [sp, #(STACKSIZE + 0)] // C
	ldr		w11, [sp, #(STACKSIZE + 8)] // sdc
	lsl		w11, w11, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB4
#else
	bl inner_scale_ab_8x4_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 16)] // D
	ldr		w9, [sp, #(STACKSIZE + 24)] // sdd
	lsl		w9, w9, #5 // 32*sdd
	ldr		w10, [sp, #(STACKSIZE + 32)] // m1
	ldr		w11, [sp, #(STACKSIZE + 40)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
	bl inner_store_8x4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nn_8x4_vs_lib4)





//                                 w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dsyrk_nt_l_8x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_8x4_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_syrk_l_add_nt_8x4_lib4
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
	INNER_STORE_L_8X4_LIB4
#else
	bl inner_store_l_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_8x4_lib4)





//                                    w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dsyrk_nt_l_8x4_vs_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_8x4_vs_lib4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_syrk_l_add_nt_8x4_lib4
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
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB4
#else
	bl inner_store_l_8x4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_8x4_vs_lib4)





//                                      w0        x1         w2        x3        x4            x5         w6       x7         sp+0     sp+8       sp+16
// void kernel_dtrsm_nt_rl_inv_8x4_lib4(int kmax, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_8x4_lib4)



	PROLOGUE



	ZERO_ACC



	// call inner kernel dgemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // sdc
	lsl		w10, w10, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_8X4_LIB4
#else
	bl inner_scale_m1b_8x4_lib4
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		x9, [sp, #(STACKSIZE + 16)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_8X4_LIB4
#else
	bl inner_edge_trsm_rlt_inv_8x4_lib4
#endif



	// store
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // sdc
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
	bl inner_store_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_8x4_lib4)





//                                         w0        x1         w2        x3        x4            x5         w6       x7         sp+0     sp+8       sp+16               sp+24   sp+32
// void kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(int kmax, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int m1, int n1);

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_8x4_vs_lib4)



	PROLOGUE



	ZERO_ACC



	// call inner kernel dgemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // sdc
	lsl		w10, w10, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_8X4_LIB4
#else
	bl inner_scale_m1b_8x4_lib4
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		x9, [sp, #(STACKSIZE + 16)] // inv_diag_E
	ldr		w10, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_8X4_VS_LIB4
#else
	bl inner_edge_trsm_rlt_inv_8x4_vs_lib4
#endif



	// store
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // sdd
	lsl		w9, w9, #5 // 32*sdd
	ldr		w10, [sp, #(STACKSIZE + 24)] // m1
	ldr		w11, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB4
#else
	bl inner_store_8x4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_8x4_vs_lib4)





//                                  w0        x1         w2        x3        x4         w5       x6         w7       sp+0
// void kernel_dpotrf_nt_l_8x4_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.align	4
	GLOB_FUN_START(kernel_dpotrf_nt_l_8x4_lib4)



	PROLOGUE



	ZERO_ACC



	// call inner kernel dsyrk l nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_syrk_l_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_8X4_LIB4
#else
	bl inner_scale_m11_8x4_lib4
#endif



	// factorization
	ldr		x8, [sp, #(STACKSIZE + 0)] // inv_diag_D

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_8X4_LIB4
#else
	bl inner_edge_potrf_8x4_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // sdd
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB4
#else
	bl inner_store_l_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dpotrf_nt_l_8x4_lib4)




//                                     w0        x1         w2        x3        x4         w5       x6         w7       sp+0                sp+8    sp+16
// void kernel_dpotrf_nt_l_8x4_vs_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int m1, int n1);

	.align	4
	GLOB_FUN_START(kernel_dpotrf_nt_l_8x4_vs_lib4)



	PROLOGUE



	ZERO_ACC



	// call inner kernel dsyrk l nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_syrk_l_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_8X4_LIB4
#else
	bl inner_scale_m11_8x4_lib4
#endif



	// factorization
	ldr		x8, [sp, #(STACKSIZE + 0)] // inv_diag_D
	ldr		w9, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_8X4_VS_LIB4
#else
	bl inner_edge_potrf_8x4_vs_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // sdd
	lsl		w9, w9, #5 // 32*sdd
	ldr		w10, [sp, #(STACKSIZE + 8)] // m1
	ldr		w11, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB4
#else
	bl inner_store_l_8x4_vs_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dpotrf_nt_l_8x4_vs_lib4)





#if defined(BLAS_API)

#include "kernel_dgemm_8x4_lib.S"

#endif

