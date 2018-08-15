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

	.text




// subroutine INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11  <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_12x4_lib4, %function
inner_kernel_gemm_add_nt_12x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return


	// prefetch
	// B
	prfm	PLDL1KEEP, [x11, #0]
	// A 1st panel
	prfm	PLDL1KEEP, [x9, #0]
	// A 2nd panel
	add		x12, x9, x10
	prfm	PLDL1KEEP, [x12, #0]
	// A 3rd panel
	add		x13, x12, x10
	prfm	PLDL1KEEP, [x13, #0]


	// preload
	// 1st loop
	// A
	ldr		d24, [x9, #(0*8+0*32)] // A0[0]
	ldr		x20, [x9, #(1*8+0*32)] // A0[1]
	ldr		d25, [x9, #(2*8+0*32)] // A1[0]
	ldr		x21, [x9, #(3*8+0*32)] // A1[1]
	ldr		d26, [x12, #(0*8+0*32)] // A2[0]
	ldr		x22, [x12, #(1*8+0*32)] // A2[1]
	ldr		d27, [x12, #(2*8+0*32)] // A3[0]
	ldr		x23, [x12, #(3*8+0*32)] // A3[1]
	//                              // A4[0]
	ldr		x24, [x13, #(1*8+0*32)] // A4[1]
	//                              // A5[0]
	ldr		x25, [x13, #(3*8+0*32)] // A5[1]
	// B
	ldr		d30, [x11, #(0*8+0*32)] // B0[0]
	ldr		x26, [x11, #(1*8+0*32)] // B0[1]
	//                              // B1[0]
	ldr		x27, [x11, #(3*8+0*32)] // B1[1]

	prfm	PLDL1KEEP, [x11, #64]
	ins		v24.d[1], x20 // A0[1]
	prfm	PLDL1KEEP, [x9, #64]
	ins		v25.d[1], x21 // A1[1]
	prfm	PLDL1KEEP, [x12, #64]
	ins		v30.d[1], x26 // B0[1]
	prfm	PLDL1KEEP, [x13, #64]

	// 2nd loop
	// A
	ldr		x20, [x9, #(1*8+1*32)] // A0[1]
	ldr		x21, [x9, #(3*8+1*32)] // A1[1]
	// B
	ldr		x26, [x11, #(1*8+1*32)] // B0[1]


	cmp		w8, #4
	ble		0f // consider clean up loop

	// main loop
1:

//	ldr		d30, [x11], #8 // B0[0]
//	ldr		x26, [x11], #8 // B0[1]
//	ins		v30.d[1], x26 // B0[1]
//	ld1		{v30.d}[1], [x11], #8 // B0[1]


	// unroll 0

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+0*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	prfm	PLDL1KEEP, [x11, #128]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+0*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+0*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+1*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	prfm	PLDL1KEEP, [x12, #128]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+1*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+1*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+1*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+1*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+1*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+2*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+1*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+2*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+1*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+2*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 1

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+1*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+2*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+1*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+2*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+1*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+2*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+2*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+2*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+2*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+2*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+2*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+3*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+2*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+3*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+2*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 2

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+2*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	prfm	PLDL1KEEP, [x11, #192]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+2*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+2*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+3*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	prfm	PLDL1KEEP, [x12, #192]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+3*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+3*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+3*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+3*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+3*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+4*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+3*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+4*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+3*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+4*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 3

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+3*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+4*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	add		x9, x9, #(4*32) // A_0
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+3*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+4*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+3*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+4*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	sub		w8, w8, #4
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+0*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+4*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	add		x12, x12, #(4*32) // A_1
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+0*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+4*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	add		x11, x11, #(4*32) // B
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+0*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+1*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	add		x13, x13, #(4*32) // A_2
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+0*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+1*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	cmp		w8, #4
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+0*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	bgt		1b

0:

//	b 2f // XXX

	cmp		w8, #3
	ble		4f


	// unroll 0

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+0*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+0*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+0*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+1*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+1*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+1*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+1*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+1*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+1*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+2*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+1*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+2*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+1*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+2*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 1

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+1*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+2*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+1*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+2*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+1*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+2*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+2*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+2*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+2*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+2*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+2*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
	ldr		x20, [x9, #(1*8+3*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+2*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
	ldr		x21, [x9, #(3*8+3*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+2*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 2

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+2*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+2*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+2*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
	ldr		x23, [x12, #(3*8+3*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
	ldr		d24, [x9, #(0*8+3*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	ldr		x24, [x13, #(1*8+3*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
	ldr		d25, [x9, #(2*8+3*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
	ldr		x25, [x13, #(3*8+3*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
	ldr		d26, [x12, #(0*8+3*32)] // A2[0]
	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
//	ldr		x20, [x9, #(1*8+4*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
	ldr		d30, [x11, #(0*8+3*32)] // B0[0]
	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
//	ldr		x21, [x9, #(3*8+4*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
	ldr		d27, [x12, #(2*8+3*32)] // A3[0]
	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
//	ldr		x26, [x11, #(1*8+4*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]


	// unroll 3

	// load 31 ins 26
	ldr		d31, [x11, #(2*8+3*32)] // B1[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v0.2d, v24.2d, v30.2d[0]
//	ldr		x22, [x12, #(1*8+4*32)] // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	add		x9, x9, #(4*32) // A_0
	fmla	v1.2d, v25.2d, v30.2d[0]

	// load 28 ins 31
	ldr		d28, [x13, #(0*8+3*32)] // A4[0]
	ins		v31.d[1], x27 // B1[1]
	fmla	v3.2d, v25.2d, v30.2d[1]
//	ldr		x27, [x11, #(3*8+4*32)] // B1[1]
	fmla	v8.2d, v26.2d, v30.2d[0]
	fmla	v10.2d, v26.2d, v30.2d[1]

	// load 29 ins 27
	ldr		d29, [x13, #(2*8+3*32)] // A5[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v4.2d, v24.2d, v31.2d[0]
//	ldr		x23, [x12, #(3*8+4*32)] // A3[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	sub		w8, w8, #4
	fmla	v5.2d, v25.2d, v31.2d[0]

	// load 24 ins 28
//	ldr		d24, [x9, #(0*8+0*32)] // A0[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
//	ldr		x24, [x13, #(1*8+4*32)] // A4[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	add		x12, x12, #(4*32) // A_1
	fmla	v14.2d, v26.2d, v31.2d[1]

	// load 25 ins 29
//	ldr		d25, [x9, #(2*8+0*32)] // A1[0]
	ins		v29.d[1], x25 // A5[1]
	fmla	v9.2d, v27.2d, v30.2d[0]
//	ldr		x25, [x13, #(3*8+4*32)] // A5[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	add		x11, x11, #(4*32) // B
	fmla	v16.2d, v28.2d, v30.2d[0]

	// load 26 ins 24
//	ldr		d26, [x12, #(0*8+0*32)] // A2[0]
//	ins		v24.d[1], x20 // A0[1]
	fmla	v18.2d, v28.2d, v30.2d[1]
//	ldr		x20, [x9, #(1*8+1*32)] // A0[1]
	fmla	v17.2d, v29.2d, v30.2d[0]
	add		x13, x13, #(4*32) // A_2
	fmla	v19.2d, v29.2d, v30.2d[1]

	// load 30 ins 25
//	ldr		d30, [x11, #(0*8+0*32)] // B0[0]
//	ins		v25.d[1], x21 // A1[1]
	fmla	v13.2d, v27.2d, v31.2d[0]
//	ldr		x21, [x9, #(3*8+1*32)] // A1[1]
	fmla	v15.2d, v27.2d, v31.2d[1]
//	cmp		w8, #4
	fmla	v20.2d, v28.2d, v31.2d[0]

	// load 27 ins 30
//	ldr		d27, [x12, #(2*8+0*32)] // A3[0]
//	ins		v30.d[1], x26 // B0[1]
	fmla	v22.2d, v28.2d, v31.2d[1]
//	ldr		x26, [x11, #(1*8+1*32)] // B0[1]
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x12, x12, #32

3: // clean1-up loop

	// unroll 0
	ldr		q24, [x9], #16 // A0
	ldr		q25, [x9], #16 // A1
	ldr		q30, [x11], #16 // B0
	ldr		q31, [x11], #16 // B1

	ldr		d26, [x12], #8 // A2[0]
	fmla	v0.2d, v24.2d, v30.2d[0]
	ldr		x22, [x12], #8 // A2[1]
	fmla	v2.2d, v24.2d, v30.2d[1]
	fmla	v4.2d, v24.2d, v31.2d[0]

	ldr		d27, [x12], #8 // A3[0]
	ins		v26.d[1], x22 // A2[1]
	fmla	v6.2d, v24.2d, v31.2d[1]
	ldr		x23, [x12], #8 // A3[1]
	fmla	v1.2d, v25.2d, v30.2d[0]
	fmla	v3.2d, v25.2d, v30.2d[1]

	ldr		d28, [x13], #8 // A4[0]
	ins		v27.d[1], x23 // A3[1]
	fmla	v5.2d, v25.2d, v31.2d[0]
	ldr		x24, [x13], #8 // A4[1]
	fmla	v7.2d, v25.2d, v31.2d[1]
	fmla	v8.2d, v26.2d, v30.2d[0]

	ldr		d29, [x13], #8 // A5[0]
	ins		v28.d[1], x24 // A4[1]
	fmla	v10.2d, v26.2d, v30.2d[1]
	ldr		x25, [x13], #8 // A5[1]
	fmla	v12.2d, v26.2d, v31.2d[0]
	fmla	v9.2d, v27.2d, v30.2d[0]

	ins		v29.d[1], x25 // A5[1]
	fmla	v14.2d, v26.2d, v31.2d[1]
	fmla	v11.2d, v27.2d, v30.2d[1]
	fmla	v13.2d, v27.2d, v31.2d[0]

	fmla	v15.2d, v27.2d, v31.2d[1]
	fmla	v16.2d, v28.2d, v30.2d[0]
	fmla	v18.2d, v28.2d, v30.2d[1]

	fmla	v20.2d, v28.2d, v31.2d[0]
	sub		w8, w8, #1
	fmla	v22.2d, v28.2d, v31.2d[1]
	fmla	v17.2d, v29.2d, v30.2d[0]

	fmla	v19.2d, v29.2d, v30.2d[1]
	cmp		w8, #0
	fmla	v21.2d, v29.2d, v31.2d[0]
	fmla	v23.2d, v29.2d, v31.2d[1]

	bgt		3b

2: // return


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nt_12x4_lib4, .-inner_kernel_gemm_add_nt_12x4_lib4
#endif
// end




// subroutine INNER_SCALE_AB_12X4_LIB4
//
// input arguments:
// v00..v23 <- D
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- sdc
//
// output arguments:
// x12  <- dirty
// x12  <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_12X4_LIB4
#else
	.align	4
	.type inner_scale_ab_12x4_lib4, %function
inner_scale_ab_12x4_lib4:
#endif

	// load alpha
	ld1		{v28.2d}, [x8]

	// A = A*alpha
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

	// load beta
	ld1		{v28.2d}, [x9]

	// get index for second panel of C
	add		x12, x10, x11

	// load first two column of first panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	// D=D+C*beta
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v2.2d, v26.2d, v28.2d[0]
	fmla	v3.2d, v27.2d, v28.2d[0]

	// load second two column of first panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x10], #64
	// D=D+C*beta
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v6.2d, v26.2d, v28.2d[0]
	fmla	v7.2d, v27.2d, v28.2d[0]

	// get index for third panel of C
	add		x13, x12, x11

	// load first two column of second panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	// D=D+C*beta
	fmla	v8.2d, v24.2d, v28.2d[0]
	fmla	v9.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	// load second two column of second panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	// D=D+C*beta
	fmla	v12.2d, v24.2d, v28.2d[0]
	fmla	v13.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]

	// load first two column of third panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x13], #64
	// D=D+C*beta
	fmla	v16.2d, v24.2d, v28.2d[0]
	fmla	v17.2d, v25.2d, v28.2d[0]
	fmla	v18.2d, v26.2d, v28.2d[0]
	fmla	v19.2d, v27.2d, v28.2d[0]

	// load second two column of third  panel of C
	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x13], #64
	// D=D+C*beta
	fmla	v20.2d, v24.2d, v28.2d[0]
	fmla	v21.2d, v25.2d, v28.2d[0]
	fmla	v22.2d, v26.2d, v28.2d[0]
	fmla	v23.2d, v27.2d, v28.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_ab_12x4_lib4, .-inner_scale_ab_12x4_lib4
#endif
// end




// subroutine INNER_SCALE_11_12X4_LIB4
//
// input arguments:
// x8  <- C
// x9  <- sdc
//
// output arguments:
// doc:
// Unit value for coefficients alpha and beta

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_12X4_LIB4
#else
	.align	4
	.type inner_scale_11_12x4_lib4, %function
inner_scale_11_12x4_lib4:
#endif

	add		x12, x8, x9
	add		x13, x10, x9

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	// D=D+C (D=A*B)
	fadd	v0.2d, v24.2d, v0.2d
	fadd	v1.2d, v25.2d, v1.2d
	fadd	v2.2d, v26.2d, v2.2d
	fadd	v3.2d, v27.2d, v3.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x8], #64
	// D=D+C (D=A*B)
	fadd	v4.2d, v24.2d, v4.2d
	fadd	v5.2d, v25.2d, v5.2d
	fadd	v6.2d, v26.2d, v6.2d
	fadd	v7.2d, v27.2d, v7.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	// D=D+C (D=A*B)
	fadd	v8.2d, v24.2d, v8.2d
	fadd	v9.2d, v25.2d, v9.2d
	fadd	v10.2d, v26.2d, v10.2d
	fadd	v11.2d, v27.2d, v11.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	// D=D+C (D=A*B)
	fadd	v12.2d, v24.2d, v12.2d
	fadd	v13.2d, v25.2d, v13.2d
	fadd	v14.2d, v26.2d, v14.2d
	fadd	v15.2d, v27.2d, v15.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x13], #64
	// D=D+C (D=A*B)
	fadd	v16.2d, v24.2d, v8.2d
	fadd	v17.2d, v25.2d, v9.2d
	fadd	v18.2d, v26.2d, v10.2d
	fadd	v19.2d, v27.2d, v11.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x13], #64
	// D=D+C (D=A*B)
	fadd	v20.2d, v24.2d, v12.2d
	fadd	v21.2d, v25.2d, v13.2d
	fadd	v22.2d, v26.2d, v14.2d
	fadd	v23.2d, v27.2d, v15.2d

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_11_12x4_lib4, .-inner_scale_11_12x4_lib4
#endif
// end




// subroutine INNER_STORE_12X4_LIB4
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:
// x10 -> dirty
// x11 -> dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_12X4_LIB4
#else
	.align 4
	.type inner_store_12x4_lib4, %function
inner_store_12x4_lib4:
#endif

	// 2nd panel
	add		x10, x8, x9
	// 3rd panel
	add		x11, x10, x9

	st1		{v0.2d, v1.2d, v2.2d, v3.2d}, [x8], #64
	st1		{v4.2d, v5.2d, v6.2d, v7.2d}, [x8], #64

	st1		{v8.2d, v9.2d, v10.2d, v11.2d}, [x10], #64
	st1		{v12.2d, v13.2d, v14.2d, v15.2d}, [x10], #64

	st1		{v16.2d, v17.2d, v18.2d, v19.2d}, [x11], #64
	st1		{v20.2d, v21.2d, v22.2d, v23.2d}, [x11], #64

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_12x4_lib4, .-inner_store_12x4_lib4
#endif
// end




// void kernel_dgemm_nt_12x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)
//                                w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8

	.align	4
	.global	kernel_dgemm_nt_12x4_lib4
	.type	kernel_dgemm_nt_12x4_lib4, %function
kernel_dgemm_nt_12x4_lib4:



	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov    d0, xzr
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
	mov		w8, w0       // kmax
	mov		x9, x2       // A
	mov		w10, w3      // sda
	lsl		w10, w10, #5 // sda_bit = 32*sda
	mov		x11, x4      // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_12x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // sdc
	lsl		w11, w11, #5 // sdc_bit = 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
	bl inner_scale_ab_12x4_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // sdd
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
	bl inner_store_12x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dgemm_nt_12x4_lib4, .-kernel_dgemm_nt_12x4_lib4
// end
