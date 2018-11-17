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
// w8   <- k
// x9   <- A
// x10  <- B
// x11  <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_4x4_lib4c)
#endif

#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) TODO



// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
//	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	add		x10, x10, x11

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
//	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	
	// unroll 0
	ldp		q26, q27, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10]
	fmla	v2.2d, v24.2d, v28.2d[1]
	add		x10, x10, x11
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
//	prfm	PLDL1KEEP, [x10, #64]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ldp		q24, q25, [x9], #32
	fmla	v0.2d, v26.2d, v30.2d[0]
	fmla	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10]
	fmla	v2.2d, v26.2d, v30.2d[1]
	fmla	v3.2d, v27.2d, v30.2d[1]
	add		x10, x10, x11
	fmla	v4.2d, v26.2d, v31.2d[0]
	fmla	v5.2d, v27.2d, v31.2d[0]
	sub		w8, w8, #4
	fmla	v6.2d, v26.2d, v31.2d[1]
	fmla	v7.2d, v27.2d, v31.2d[1]

	// unroll 2
	ldp		q26, q27, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	ldp		q30, q31, [x10]
	fmla	v2.2d, v24.2d, v28.2d[1]
	add		x10, x10, x11
	fmla	v3.2d, v25.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
//	prfm	PLDL1KEEP, [x10, #64]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ldp		q24, q25, [x9], #32
	fmla	v0.2d, v26.2d, v30.2d[0]
	fmla	v1.2d, v27.2d, v30.2d[0]
	ldp		q28, q29, [x10]
	fmla	v2.2d, v26.2d, v30.2d[1]
	fmla	v3.2d, v27.2d, v30.2d[1]
	add		x10, x10, x11
	fmla	v4.2d, v26.2d, v31.2d[0]
	fmla	v5.2d, v27.2d, v31.2d[0]
	cmp		w8, #4
	fmla	v6.2d, v26.2d, v31.2d[1]
	fmla	v7.2d, v27.2d, v31.2d[1]

	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, x11

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 1
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	// unroll 3
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
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
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
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

	

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53) // TODO



	// early return
	cmp		w8, #0
	ble		2f // return

	// preload
	ldr		d16, [x9, #(0*8+0*32)] // A
	ldr		x16, [x9, #(1*8+0*32)] // A
	ldr		d24, [x10, #(0*8+0*32)] // B
	ldr		x22, [x10, #(1*8+0*32)] // B
	ldr		d17, [x9, #(2*8+0*32)] // A
	ldr		x17, [x9, #(3*8+0*32)] // A
	ldr		d25, [x10, #(2*8+0*32)] // B
	ldr		x23, [x10, #(3*8+0*32)] // B

	ldr		d18, [x9, #(0*8+1*32)] // A
	ldr		x12, [x9, #(1*8+1*32)] // A
	ldr		d26, [x10, #(0*8+1*32)] // B
	ins		v16.d[1], x16
	ldr		x14, [x10, #(1*8+1*32)] // B
	ldr		d19, [x9, #(2*8+1*32)] // A
	ins		v24.d[1], x22
	ldr		x13, [x9, #(3*8+1*32)] // A
	ldr		d27, [x10, #(2*8+1*32)] // B
	ins		v17.d[1], x17
	ldr		x15, [x10, #(3*8+1*32)] // B

	// prefetch
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x10, #64]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// main loop
1:
	
	// pre-load
	ldr		d20, [x9, #(0*8+2*32)]
	ins		v25.d[1], x23
	ldr		d28, [x10, #(0*8+2*32)]
	ins		v18.d[1], x12
	ldr		d21, [x9, #(2*8+2*32)]
	ins		v26.d[1], x14
	ldr		d29, [x10, #(2*8+2*32)]
	ins		v19.d[1], x13

	// unroll 0
	ldr		d22, [x9, #(0*8+3*32)] // A
	ins		v27.d[1], x15
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+2*32)] // A
	fmla	v2.2d, v16.2d, v24.2d[1]
	ldr		x24, [x10, #(1*8+2*32)] // B
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d30, [x10, #(0*8+3*32)] // B
	ins		v20.d[1], x18
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v4.2d, v16.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+2*32)] // A
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d23, [x9, #(2*8+3*32)] // A
	ins		v28.d[1], x24
	fmla	v5.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x10, #128]
	fmla	v7.2d, v17.2d, v25.2d[1]
	ldr		x25, [x10, #(3*8+2*32)] // B

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		d31, [x10, #(2*8+3*32)] // B
	ins		v21.d[1], x19
	fmla	v2.2d, v18.2d, v26.2d[1]
	ldr		x20, [x9, #(1*8+3*32)] // A
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		x26, [x10, #(1*8+3*32)] // B
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4
	ldr		d16, [x9, #(0*8+4*32)] // A
	ins		v29.d[1], x25
	fmla	v4.2d, v18.2d, v27.2d[0]
	ldr		x21, [x9, #(3*8+3*32)] // A
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		x27, [x10, #(3*8+3*32)] // B
	fmla	v5.2d, v19.2d, v27.2d[0]
	ldr		d24, [x10, #(0*8+4*32)] // B
	ins		v22.d[1], x20
	fmla	v7.2d, v19.2d, v27.2d[1]
	ldr		x16, [x9, #(1*8+4*32)] // A

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	ldr		x22, [x10, #(1*8+4*32)] // B
	fmla	v2.2d, v20.2d, v28.2d[1]
	ldr		d17, [x9, #(2*8+4*32)] // A
	ins		v30.d[1], x26
	fmla	v1.2d, v21.2d, v28.2d[0]
	ldr		x17, [x9, #(3*8+4*32)] // A
	fmla	v3.2d, v21.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v20.2d, v29.2d[0]
	add		x9, x9, #128
	ldr		d25, [x10, #(2*8+4*32)] // B
	ins		v23.d[1], x21
	fmla	v6.2d, v20.2d, v29.2d[1]
	ldr		x23, [x10, #(3*8+4*32)] // B
	fmla	v5.2d, v21.2d, v29.2d[0]
	prfm	PLDL1KEEP, [x10, #192]
	fmla	v7.2d, v21.2d, v29.2d[1]
	add		x10, x10, #128

	// unroll 3
	ldr		d18, [x9, #(0*8+1*32)]
	ins		v31.d[1], x27
	fmla	v0.2d, v22.2d, v30.2d[0]
	ldr		x12, [x9, #(1*8+1*32)]
	fmla	v2.2d, v22.2d, v30.2d[1]
	cmp		w8, #4
	fmla	v1.2d, v23.2d, v30.2d[0]
	ldr		d26, [x10, #(0*8+1*32)]
	ins		v16.d[1], x16
	fmla	v3.2d, v23.2d, v30.2d[1]
	ldr		x14, [x10, #(1*8+1*32)]
	fmla	v4.2d, v22.2d, v31.2d[0]
	ldr		x13, [x9, #(3*8+1*32)]
	fmla	v5.2d, v23.2d, v31.2d[0]
	ldr		d19, [x9, #(2*8+1*32)]
	ins		v24.d[1], x22
	fmla	v6.2d, v22.2d, v31.2d[1]
	ldr		x15, [x10, #(3*8+1*32)]
	ldr		d27, [x10, #(2*8+1*32)]
	ins		v17.d[1], x17
	fmla	v7.2d, v23.2d, v31.2d[1]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	// pre-load
	ldr		d20, [x9, #(0*8+2*32)]
	ins		v25.d[1], x23
	ldr		d28, [x10, #(0*8+2*32)]
	ins		v18.d[1], x12
	ldr		d21, [x9, #(2*8+2*32)]
	ins		v26.d[1], x14
	ldr		d29, [x10, #(2*8+2*32)]
	ins		v19.d[1], x13

	// unroll 0
	ldr		d22, [x9, #(0*8+3*32)] // A
	ins		v27.d[1], x15
	fmla	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+2*32)] // A
	fmla	v2.2d, v16.2d, v24.2d[1]
	ldr		x24, [x10, #(1*8+2*32)] // B
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldr		d30, [x10, #(0*8+3*32)] // B
	ins		v20.d[1], x18
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v4.2d, v16.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+2*32)] // A
	fmla	v6.2d, v16.2d, v25.2d[1]
	ldr		d23, [x9, #(2*8+3*32)] // A
	ins		v28.d[1], x24
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10, #128]
	fmla	v7.2d, v17.2d, v25.2d[1]
	ldr		x25, [x10, #(3*8+2*32)] // B

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	ldr		d31, [x10, #(2*8+3*32)] // B
	ins		v21.d[1], x19
	fmla	v2.2d, v18.2d, v26.2d[1]
	ldr		x20, [x9, #(1*8+3*32)] // A
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldr		x26, [x10, #(1*8+3*32)] // B
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4
//	ldr		d16, [x9, #(0*8+4*32)] // A
	ins		v29.d[1], x25
	fmla	v4.2d, v18.2d, v27.2d[0]
	ldr		x21, [x9, #(3*8+3*32)] // A
	fmla	v6.2d, v18.2d, v27.2d[1]
	ldr		x27, [x10, #(3*8+3*32)] // B
	fmla	v5.2d, v19.2d, v27.2d[0]
//	ldr		d24, [x10, #(0*8+4*32)] // B
	ins		v22.d[1], x20
	fmla	v7.2d, v19.2d, v27.2d[1]
//	ldr		x16, [x9, #(1*8+4*32)] // A

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
//	ldr		x22, [x10, #(1*8+4*32)] // B
	fmla	v2.2d, v20.2d, v28.2d[1]
//	ldr		d17, [x9, #(2*8+4*32)] // A
	ins		v30.d[1], x26
	fmla	v1.2d, v21.2d, v28.2d[0]
//	ldr		x17, [x9, #(3*8+4*32)] // A
	fmla	v3.2d, v21.2d, v28.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v20.2d, v29.2d[0]
	add		x9, x9, #128
//	ldr		d25, [x10, #(2*8+4*32)] // B
	ins		v23.d[1], x21
	fmla	v6.2d, v20.2d, v29.2d[1]
//	ldr		x23, [x10, #(3*8+4*32)] // B
	fmla	v5.2d, v21.2d, v29.2d[0]
//	prfm	PLDL1KEEP, [x10, #192]
	fmla	v7.2d, v21.2d, v29.2d[1]
	add		x10, x10, #128

	// unroll 3
//	ldr		d18, [x9, #(0*8+1*32)]
	ins		v31.d[1], x27
	fmla	v0.2d, v22.2d, v30.2d[0]
//	ldr		x12, [x9, #(1*8+1*32)]
	fmla	v2.2d, v22.2d, v30.2d[1]
//	cmp		w8, #4
	fmla	v1.2d, v23.2d, v30.2d[0]
//	ldr		d26, [x10, #(0*8+1*32)]
//	ins		v16.d[1], x16
	fmla	v3.2d, v23.2d, v30.2d[1]
//	ldr		x14, [x10, #(1*8+1*32)]
	fmla	v4.2d, v22.2d, v31.2d[0]
//	ldr		x13, [x9, #(3*8+1*32)]
	fmla	v5.2d, v23.2d, v31.2d[0]
//	ldr		d19, [x9, #(2*8+1*32)]
//	ins		v24.d[1], x22
	fmla	v6.2d, v22.2d, v31.2d[1]
//	ldr		x15, [x10, #(3*8+1*32)]
//	ldr		d27, [x10, #(2*8+1*32)]
//	ins		v17.d[1], x17
	fmla	v7.2d, v23.2d, v31.2d[1]

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

	

#endif // cortex a53 TODO



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_4x4_lib4)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10   <- B
// x11   <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_4x4_lib4c)
#endif



#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11
	add		x13, x12, x11
	add		x14, x13, x11

	// prefetch
//	prfm	PLDL1KEEP, [x10, #0]
//	prfm	PLDL1KEEP, [x10, #32]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #32]

	// preload
	ldp		q24, q25, [x10]
	add		x10, x10, #32
	ldp		q26, q27, [x12]
	add		x12, x12, #32
	ldp		q28, q29, [x13]
	add		x13, x13, #32
	ldp		q30, q31, [x14]
	add		x14, x14, #32
	ldp		q16, q17, [x9, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop


	// prefetch
//	prfm	PLDL1KEEP, [x9, #32]
//	prfm	PLDL1KEEP, [x10, #32]

//	add		x13, x11, #32

	// main loop
1:
	
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #32]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]
//	prfm	PLDL1KEEP, [x10, x13]

	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #64]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4

	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	ldp		q18, q19, [x9, #96]
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
	ldp		q16, q17, [x9, #0]

	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
	ldp		q24, q25, [x10]
	fmla	v2.2d, v18.2d, v27.2d[1]
	add		x10, x10, #32
	fmla	v3.2d, v19.2d, v27.2d[1]
	ldp		q26, q27, [x12]
	fmla	v4.2d, v18.2d, v29.2d[1]
	add		x12, x12, #32
	fmla	v5.2d, v19.2d, v29.2d[1]
	ldp		q28, q29, [x13]
	fmla	v6.2d, v18.2d, v31.2d[1]
	add		x13, x13, #32
	fmla	v7.2d, v19.2d, v31.2d[1]
	ldp		q30, q31, [x14]
	add		x14, x14, #32

//b 2f
	cmp		w8, #4
	bgt		1b

//	sub		x9, x9, #32
//	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #32]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x10, x13]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]

	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #64]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4

	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	ldp		q18, q19, [x9, #96]
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]
//	ldp		q16, q17, [x9, #0]

	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
//	ldp		q24, q25, [x10]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
//	ldp		q26, q27, [x10]
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
//	ldp		q28, q29, [x10]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
//	ldp		q30, q31, [x10]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, #32
	sub		x12, x12, #32
	sub		x13, x13, #32
	sub		x14, x14, #32

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10]
	add		x10, x10, #8
	ldr		d29, [x12]
	add		x12, x12, #8
	ldr		d30, [x13]
	add		x13, x13, #8
	ldr		d31, [x14]
	add		x14, x14, #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]

	add		x9, x9, #32
	sub		w8, w8, #1

	cmp		w8, #0
	bgt		3b

2: // return



#else // cortex a53 TODO !!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
//	prfm	PLDL1KEEP, [x9, #0]
//	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload

	// prefetch
//	prfm	PLDL1KEEP, [x9, #32]
//	prfm	PLDL1KEEP, [x10, #32]

	add		x13, x11, #32

	// main loop
1:

	ldr		q24, [x10, #0]
	ldr		q25, [x10, #32]
	ldr		q26, [x10, #64]
	ldr		q27, [x10, #96]

	ldr		q28, [x10, #16]
	ldr		q29, [x10, #48]
	ldr		q30, [x10, #80]
	ldr		q31, [x10, #112]

	ldr		q16, [x9, #0]
	ldr		q17, [x9, #16]
	ldr		q18, [x9, #32]
	ldr		q19, [x9, #48]
	ldr		q20, [x9, #64]
	ldr		q21, [x9, #80]
	ldr		q22, [x9, #96]
	ldr		q23, [x9, #112]

	//
	fmla	v0.2d, v16.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v2.2d, v16.2d, v25.2d[0]

	fmla	v3.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x10, x11]
	fmla	v4.2d, v16.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x10, x13]
	fmla	v5.2d, v17.2d, v26.2d[0]

	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]

	//
	fmla	v0.2d, v18.2d, v24.2d[1]

	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v25.2d[1]
	fmla	v3.2d, v19.2d, v25.2d[1]

	fmla	v4.2d, v18.2d, v26.2d[1]
	fmla	v5.2d, v19.2d, v26.2d[1]
	fmla	v6.2d, v18.2d, v27.2d[1]

	fmla	v7.2d, v19.2d, v27.2d[1]

	//
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]

	fmla	v2.2d, v20.2d, v29.2d[0]
	fmla	v3.2d, v21.2d, v29.2d[0]
	fmla	v4.2d, v20.2d, v30.2d[0]

	fmla	v5.2d, v21.2d, v30.2d[0]
	fmla	v6.2d, v20.2d, v31.2d[0]
	fmla	v7.2d, v21.2d, v31.2d[0]

	//
	fmla	v0.2d, v22.2d, v28.2d[1]
	sub		w8, w8, #4
	fmla	v1.2d, v23.2d, v28.2d[1]
	add		x9, x9, #128 // XXX !!!!!!!!!!!!!!!!!!!
	fmla	v2.2d, v22.2d, v29.2d[1]

	fmla	v3.2d, v23.2d, v29.2d[1]
	add		x10, x10, x11 // XXX !!!!!!!!!!!!!!!!!!!
	fmla	v4.2d, v22.2d, v30.2d[1]
	fmla	v5.2d, v23.2d, v30.2d[1]

	fmla	v6.2d, v22.2d, v31.2d[1]
	cmp		w8, #4
	fmla	v7.2d, v23.2d, v31.2d[1]


	bgt		1b

//	sub		x9, x9, #32
//	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	ldr		q24, [x10, #0]
	ldr		q25, [x10, #32]
	ldr		q26, [x10, #64]
	ldr		q27, [x10, #96]

	ldr		q28, [x10, #16]
	ldr		q29, [x10, #48]
	ldr		q30, [x10, #80]
	ldr		q31, [x10, #112]

	ldr		q16, [x9, #0]
	ldr		q17, [x9, #16]
	ldr		q18, [x9, #32]
	ldr		q19, [x9, #48]
	ldr		q20, [x9, #64]
	ldr		q21, [x9, #80]
	ldr		q22, [x9, #96]
	ldr		q23, [x9, #112]

	//
	fmla	v0.2d, v16.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v2.2d, v16.2d, v25.2d[0]

	fmla	v3.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	fmla	v4.2d, v16.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x13]
	fmla	v5.2d, v17.2d, v26.2d[0]

	fmla	v6.2d, v16.2d, v27.2d[0]
	fmla	v7.2d, v17.2d, v27.2d[0]

	//
	fmla	v0.2d, v18.2d, v24.2d[1]

	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v25.2d[1]
	fmla	v3.2d, v19.2d, v25.2d[1]

	fmla	v4.2d, v18.2d, v26.2d[1]
	fmla	v5.2d, v19.2d, v26.2d[1]
	fmla	v6.2d, v18.2d, v27.2d[1]

	fmla	v7.2d, v19.2d, v27.2d[1]

	//
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]

	fmla	v2.2d, v20.2d, v29.2d[0]
	fmla	v3.2d, v21.2d, v29.2d[0]
	fmla	v4.2d, v20.2d, v30.2d[0]

	fmla	v5.2d, v21.2d, v30.2d[0]
	fmla	v6.2d, v20.2d, v31.2d[0]
	fmla	v7.2d, v21.2d, v31.2d[0]

	//
	fmla	v0.2d, v22.2d, v28.2d[1]
	sub		w8, w8, #4
	fmla	v1.2d, v23.2d, v28.2d[1]
	add		x9, x9, #128 // XXX !!!!!!!!!!!!!!!!!!!
	fmla	v2.2d, v22.2d, v29.2d[1]

	fmla	v3.2d, v23.2d, v29.2d[1]
	add		x10, x10, x11 // XXX !!!!!!!!!!!!!!!!!!!
	fmla	v4.2d, v22.2d, v30.2d[1]
	fmla	v5.2d, v23.2d, v30.2d[1]

	fmla	v6.2d, v22.2d, v31.2d[1]
	fmla	v7.2d, v23.2d, v31.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldr		d28, [x10, #0]
	ldr		d29, [x10, #32]
	ldr		d30, [x10, #64]
	ldr		d31, [x10, #96]

	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]

	add		x10, x10, #8
	sub		w8, w8, #1

	cmp		w8, #0
	bgt		3b

2: // return



#endif // TODO
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_4x4_lib4c)
#endif





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
	FUN_START(inner_scale_ab_4x4_lib)
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

	FUN_END(inner_scale_ab_4x4_lib)
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
	FUN_START(inner_scale_ab_4x4_vs_lib)
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

	FUN_END(inner_scale_ab_4x4_vs_lib)
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
	FUN_START(inner_store_4x4_lib)
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

	FUN_END(inner_store_4x4_lib)
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
	FUN_START(inner_store_4x4_vs_lib)
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

	FUN_END(inner_store_4x4_vs_lib)
#endif





//                                 w0        x1             x2         x3         x4            x5         w6       x7         sp+0
// void kernel_dgemm_nt_4x4_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib44c)
	


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

	FUN_END(kernel_dgemm_nt_4x4_lib44c)





//                                    w0        x1             x2         x3         x4            x5         w6       x7         sp+0     sp+8    sp+16
// void kernel_dgemm_nt_4x4_vs_lib44c(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_lib44c)
	


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

	FUN_END(kernel_dgemm_nt_4x4_vs_lib44c)





//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_nt_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B
	mov		w11, w4 // ldb
	lsl		w11, w11, #3 // 8*ldb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4c
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	bl inner_scale_ab_4x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_4x4_lib4cc)






//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_nn_4x4_lib4cc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_4x4_lib4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B
	mov		w11, w4 // ldb
	lsl		w11, w11, #3 // 8*ldb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x4_lib4c
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB
#else
	bl inner_scale_ab_4x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nn_4x4_lib4cc)






