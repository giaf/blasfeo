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
	.macro INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_8x4_lib4, %function
inner_kernel_gemm_add_nt_8x4_lib4:
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

	// prefetch

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

	.size	inner_kernel_gemm_add_nt_8x4_lib4, .-inner_kernel_gemm_add_nt_8x4_lib4
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
	.macro INNER_KERNEL_GEMM_SUB_NT_8X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_sub_nt_8x4_lib4, %function
inner_kernel_gemm_sub_nt_8x4_lib4:
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
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	cmp		w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
//	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
//	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
//	ldr		q16, [x9], #16
	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
//	ldr		q17, [x9], #16
	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
//	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
//	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
//	cmp		w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

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
	fmls	v0.2d, v20.2d, v28.2d[0]
	fmls	v1.2d, v21.2d, v28.2d[0]
	fmls	v2.2d, v20.2d, v28.2d[1]
	fmls	v3.2d, v21.2d, v28.2d[1]
	fmls	v4.2d, v20.2d, v29.2d[0]
	fmls	v5.2d, v21.2d, v29.2d[0]
	fmls	v6.2d, v20.2d, v29.2d[1]
	fmls	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	fmls	v10.2d, v22.2d, v28.2d[1]
	fmls	v11.2d, v23.2d, v28.2d[1]
	fmls	v12.2d, v22.2d, v29.2d[0]
	fmls	v13.2d, v23.2d, v29.2d[0]
	fmls	v14.2d, v22.2d, v29.2d[1]
	fmls	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10

	// prefetch

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


	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// main loop
1:
	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmls	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmls	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #128]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+1*32)] // B
	ins		v21.d[1], x21
	fmls	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmls	v4.2d, v16.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11, #128]
	fmls	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmls	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmls	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x12, #128]
	fmls	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+1*32)] // B
	ins		v26.d[1], x26
	fmls	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B
	fmls	v9.2d, v21.2d, v24.2d[0]
	fmls	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmls	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A1
	fmls	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+1*32)] // A1
	fmls	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmls	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmls	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmls	v2.2d, v18.2d, v26.2d[1]
	fmls	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+2*32)] // B
	ins		v23.d[1], x23
	fmls	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+2*32)] // B
	fmls	v4.2d, v18.2d, v27.2d[0]
	fmls	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmls	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmls	v7.2d, v19.2d, v27.2d[1]
	fmls	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmls	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+2*32)] // B
	fmls	v9.2d, v23.2d, v26.2d[0]
	fmls	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmls	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+2*32)] // A1
	fmls	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+2*32)] // A1
	fmls	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmls	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmls	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmls	v2.2d, v16.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+3*32)] // B
	ins		v21.d[1], x21
	fmls	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B
	fmls	v4.2d, v16.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11, #192]
	fmls	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmls	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmls	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x12, #192]
	fmls	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+3*32)] // B
	ins		v26.d[1], x26
	fmls	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmls	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmls	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmls	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A1
	fmls	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+3*32)] // A1
	fmls	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmls	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmls	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmls	v2.2d, v18.2d, v26.2d[1]
	add		x9, x9, #128
	fmls	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+4*32)] // B
	ins		v23.d[1], x23
	fmls	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+4*32)] // B
	fmls	v4.2d, v18.2d, v27.2d[0]
	add		x11, x11, #128
	fmls	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+0*32)] // A0
	ins		v16.d[1], x16
	fmls	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmls	v7.2d, v19.2d, v27.2d[1]
	add		x12, x12, #128
	fmls	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+0*32)] // B
	ins		v24.d[1], x24
	fmls	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+0*32)] // B
	fmls	v9.2d, v23.2d, v26.2d[0]
	cmp		w8, #4
	fmls	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+0*32)] // A1
	ins		v17.d[1], x17
	fmls	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+0*32)] // A1
	fmls	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+0*32)] // A1
	fmls	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+0*32)] // A1
	ins		v25.d[1], x25
	fmls	v15.2d, v23.2d, v27.2d[1]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldr		d18, [x9, #(0*8+1*32)] // A0
	ins		v20.d[1], x20
	fmls	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+1*32)] // A0
	fmls	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+1*32)] // B
	ins		v21.d[1], x21
	fmls	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+1*32)] // B
	fmls	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #128]
	fmls	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+1*32)] // A0
	ins		v18.d[1], x18
	fmls	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+1*32)] // A0
	fmls	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x12, #128]
	fmls	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+1*32)] // B
	ins		v26.d[1], x26
	fmls	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+1*32)] // B
	fmls	v9.2d, v21.2d, v24.2d[0]
	fmls	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+1*32)] // A1
	ins		v19.d[1], x19
	fmls	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+1*32)] // A1
	fmls	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+1*32)] // A1
	fmls	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+1*32)] // A1
	ins		v27.d[1], x27
	fmls	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldr		d16, [x9, #(0*8+2*32)] // A0
	ins		v22.d[1], x22
	fmls	v0.2d, v18.2d, v26.2d[0]
	ldr		x16, [x9, #(1*8+2*32)] // A0
	fmls	v2.2d, v18.2d, v26.2d[1]
	fmls	v1.2d, v19.2d, v26.2d[0]
	ldr		d24, [x11, #(0*8+2*32)] // B
	ins		v23.d[1], x23
	fmls	v3.2d, v19.2d, v26.2d[1]
	ldr		x24, [x11, #(1*8+2*32)] // B
	fmls	v4.2d, v18.2d, v27.2d[0]
	fmls	v6.2d, v18.2d, v27.2d[1]
	ldr		d17, [x9, #(2*8+2*32)] // A0
	ins		v16.d[1], x16
	fmls	v5.2d, v19.2d, v27.2d[0]
	ldr		x17, [x9, #(3*8+2*32)] // A0
	fmls	v7.2d, v19.2d, v27.2d[1]
	fmls	v8.2d, v22.2d, v26.2d[0]
	ldr		d25, [x11, #(2*8+2*32)] // B
	ins		v24.d[1], x24
	fmls	v10.2d, v22.2d, v26.2d[1]
	ldr		x25, [x11, #(3*8+2*32)] // B
	fmls	v9.2d, v23.2d, v26.2d[0]
	fmls	v11.2d, v23.2d, v26.2d[1]
	ldr		d20, [x12, #(0*8+2*32)] // A1
	ins		v17.d[1], x17
	fmls	v12.2d, v22.2d, v27.2d[0]
	ldr		x20, [x12, #(1*8+2*32)] // A1
	fmls	v14.2d, v22.2d, v27.2d[1]
	ldr		x21, [x12, #(3*8+2*32)] // A1
	fmls	v13.2d, v23.2d, v27.2d[0]
	ldr		d21, [x12, #(2*8+2*32)] // A1
	ins		v25.d[1], x25
	fmls	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldr		d18, [x9, #(0*8+3*32)] // A0
	ins		v20.d[1], x20
	fmls	v0.2d, v16.2d, v24.2d[0]
	ldr		x18, [x9, #(1*8+3*32)] // A0
	fmls	v2.2d, v16.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldr		d26, [x11, #(0*8+3*32)] // B
	ins		v21.d[1], x21
	fmls	v3.2d, v17.2d, v24.2d[1]
	ldr		x26, [x11, #(1*8+3*32)] // B
	fmls	v4.2d, v16.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x11, #192]
	fmls	v6.2d, v16.2d, v25.2d[1]
	ldr		d19, [x9, #(2*8+3*32)] // A0
	ins		v18.d[1], x18
	fmls	v5.2d, v17.2d, v25.2d[0]
	ldr		x19, [x9, #(3*8+3*32)] // A0
	fmls	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x12, #192]
	fmls	v8.2d, v20.2d, v24.2d[0]
	ldr		d27, [x11, #(2*8+3*32)] // B
	ins		v26.d[1], x26
	fmls	v10.2d, v20.2d, v24.2d[1]
	ldr		x27, [x11, #(3*8+3*32)] // B
	fmls	v9.2d, v21.2d, v24.2d[0]
	sub		w8, w8, #4
	fmls	v11.2d, v21.2d, v24.2d[1]
	ldr		d22, [x12, #(0*8+3*32)] // A1
	ins		v19.d[1], x19
	fmls	v12.2d, v20.2d, v25.2d[0]
	ldr		x22, [x12, #(1*8+3*32)] // A1
	fmls	v14.2d, v20.2d, v25.2d[1]
	ldr		x23, [x12, #(3*8+3*32)] // A1
	fmls	v13.2d, v21.2d, v25.2d[0]
	ldr		d23, [x12, #(2*8+3*32)] // A1
	ins		v27.d[1], x27
	fmls	v15.2d, v21.2d, v25.2d[1]

	// unroll 3
//	ldr		d16, [x9, #(0*8+4*32)] // A0
	ins		v22.d[1], x22
	fmls	v0.2d, v18.2d, v26.2d[0]
//	ldr		x16, [x9, #(1*8+4*32)] // A0
	fmls	v2.2d, v18.2d, v26.2d[1]
	add		x9, x9, #128
	fmls	v1.2d, v19.2d, v26.2d[0]
//	ldr		d24, [x11, #(0*8+4*32)] // B
	ins		v23.d[1], x23
	fmls	v3.2d, v19.2d, v26.2d[1]
//	ldr		x24, [x11, #(1*8+4*32)] // B
	fmls	v4.2d, v18.2d, v27.2d[0]
	add		x11, x11, #128
	fmls	v6.2d, v18.2d, v27.2d[1]
//	ldr		d17, [x9, #(2*8+0*32)] // A0
//	ins		v16.d[1], x16
	fmls	v5.2d, v19.2d, v27.2d[0]
//	ldr		x17, [x9, #(3*8+0*32)] // A0
	fmls	v7.2d, v19.2d, v27.2d[1]
	add		x12, x12, #128
	fmls	v8.2d, v22.2d, v26.2d[0]
//	ldr		d25, [x11, #(2*8+0*32)] // B
//	ins		v24.d[1], x24
	fmls	v10.2d, v22.2d, v26.2d[1]
//	ldr		x25, [x11, #(3*8+0*32)] // B
	fmls	v9.2d, v23.2d, v26.2d[0]
//	cmp		w8, #4
	fmls	v11.2d, v23.2d, v26.2d[1]
//	ldr		d20, [x12, #(0*8+0*32)] // A1
//	ins		v17.d[1], x17
	fmls	v12.2d, v22.2d, v27.2d[0]
//	ldr		x20, [x12, #(1*8+0*32)] // A1
	fmls	v14.2d, v22.2d, v27.2d[1]
//	ldr		x21, [x12, #(3*8+0*32)] // A1
	fmls	v13.2d, v23.2d, v27.2d[0]
//	ldr		d21, [x12, #(2*8+0*32)] // A1
//	ins		v25.d[1], x25
	fmls	v15.2d, v23.2d, v27.2d[1]

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
	fmls	v0.2d, v20.2d, v28.2d[0]
	fmls	v1.2d, v21.2d, v28.2d[0]
	fmls	v2.2d, v20.2d, v28.2d[1]
	fmls	v3.2d, v21.2d, v28.2d[1]
	fmls	v4.2d, v20.2d, v29.2d[0]
	fmls	v5.2d, v21.2d, v29.2d[0]
	fmls	v6.2d, v20.2d, v29.2d[1]
	fmls	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	fmls	v10.2d, v22.2d, v28.2d[1]
	fmls	v11.2d, v23.2d, v28.2d[1]
	fmls	v12.2d, v22.2d, v29.2d[0]
	fmls	v13.2d, v23.2d, v29.2d[0]
	fmls	v14.2d, v22.2d, v29.2d[1]
	fmls	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return



#endif // cortex a53



	#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_syrk_l_sub_nt_8x4_lib4, .-inner_kernel_syrk_l_sub_nt_8x4_lib4
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
	.macro INNER_KERNEL_SYRK_L_SUB_NT_8X4_LIB4
#else
	.align	4
	.type inner_kernel_syrk_l_sub_nt_8x4_lib4, %function
inner_kernel_syrk_l_sub_nt_8x4_lib4:
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
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	cmp		w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 1
	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
	ldr		q16, [x9], #16
//	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
	ldr		q17, [x9], #16
//	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
	sub		w8, w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

	// unroll 2
	ldp		d28, d29, [x11], #16
	fmls	v0.2d, v16.2d, v24.2d[0]
	fmls	v1.2d, v17.2d, v24.2d[0]
	ldp		d30, d31, [x11], #16
	fmls	v2.2d, v16.2d, v25.2d[0]
	fmls	v3.2d, v17.2d, v25.2d[0]
	ldr		q18, [x9], #16
//	fmls	v4.2d, v16.2d, v26.2d[0]
	fmls	v5.2d, v17.2d, v26.2d[0]
	ldr		q19, [x9], #16
//	fmls	v6.2d, v16.2d, v27.2d[0]
	fmls	v7.2d, v17.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmls	v8.2d, v20.2d, v24.2d[0]
	fmls	v9.2d, v21.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmls	v10.2d, v20.2d, v25.2d[0]
	fmls	v11.2d, v21.2d, v25.2d[0]
	ldp		q22, q23, [x12], #32
	fmls	v12.2d, v20.2d, v26.2d[0]
	fmls	v13.2d, v21.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmls	v14.2d, v20.2d, v27.2d[0]
	fmls	v15.2d, v21.2d, v27.2d[0]

	// unroll 3
//	ldp		d24, d25, [x11], #16
	fmls	v0.2d, v18.2d, v28.2d[0]
	fmls	v1.2d, v19.2d, v28.2d[0]
//	ldp		d26, d27, [x11], #16
	fmls	v2.2d, v18.2d, v29.2d[0]
	fmls	v3.2d, v19.2d, v29.2d[0]
//	ldr		q16, [x9], #16
//	fmls	v4.2d, v18.2d, v30.2d[0]
	fmls	v5.2d, v19.2d, v30.2d[0]
//	ldr		q17, [x9], #16
//	fmls	v6.2d, v18.2d, v31.2d[0]
	fmls	v7.2d, v19.2d, v31.2d[0]
//	ldr		q20, [x12], #16
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
//	ldr		q21, [x12], #16
	fmls	v10.2d, v22.2d, v29.2d[0]
	fmls	v11.2d, v23.2d, v29.2d[0]
//	cmp		w8, #4
	fmls	v12.2d, v22.2d, v30.2d[0]
	fmls	v13.2d, v23.2d, v30.2d[0]
	fmls	v14.2d, v22.2d, v31.2d[0]
	fmls	v15.2d, v23.2d, v31.2d[0]

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
	fmls	v0.2d, v20.2d, v28.2d[0]
	fmls	v1.2d, v21.2d, v28.2d[0]
	fmls	v2.2d, v20.2d, v28.2d[1]
	fmls	v3.2d, v21.2d, v28.2d[1]
//	fmls	v4.2d, v20.2d, v29.2d[0]
	fmls	v5.2d, v21.2d, v29.2d[0]
//	fmls	v6.2d, v20.2d, v29.2d[1]
	fmls	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x12], #32
	fmls	v8.2d, v22.2d, v28.2d[0]
	fmls	v9.2d, v23.2d, v28.2d[0]
	fmls	v10.2d, v22.2d, v28.2d[1]
	fmls	v11.2d, v23.2d, v28.2d[1]
	fmls	v12.2d, v22.2d, v29.2d[0]
	fmls	v13.2d, v23.2d, v29.2d[0]
	fmls	v14.2d, v22.2d, v29.2d[1]
	fmls	v15.2d, v23.2d, v29.2d[1]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_syrk_l_sub_nt_8x4_lib4, .-inner_kernel_syrk_l_sub_nt_8x4_lib4
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
	.type inner_edge_trsm_rlt_inv_8x4_lib4, %function
inner_edge_trsm_rlt_inv_8x4_lib4:
#endif
	
	// first column
	ldr			d16, [x9, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]
	fmul		v8.2d, v8.2d, v16.2d[0]
	fmul		v9.2d, v9.2d, v16.2d[0]

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

#if defined(OS_LINUX)
	.size	inner_edge_trsm_rlt_inv_8x4_lib4, .-inner_edge_trsm_rlt_inv_8x4_lib4
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
	.macro INNER_EDGE_POTRF_8X4_LIB4
#else
	.p2align 4
	.type inner_edge_potrf_8x4_lib4, %function
inner_edge_potrf_8x4_lib4:
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

	.size	inner_edge_potrf_8x4_lib4, .-inner_edge_potrf_8x4_lib4
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

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_ab_8x4_lib4, .-inner_scale_ab_8x4_lib4
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- sdc
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_8X4_LIB4
#else
	.align	4
	.type inner_scale_11_8x4_lib4, %function
inner_scale_11_8x4_lib4:
#endif

	add		x12, x8, x9

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

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	fadd	v8.2d, v24.2d, v8.2d
	fadd	v9.2d, v25.2d, v9.2d
	fadd	v10.2d, v26.2d, v10.2d
	fadd	v11.2d, v27.2d, v11.2d

	ld1		{v24.2d, v25.2d, v26.2d, v27.2d}, [x12], #64
	fadd	v12.2d, v24.2d, v12.2d
	fadd	v13.2d, v25.2d, v13.2d
	fadd	v14.2d, v26.2d, v14.2d
	fadd	v15.2d, v27.2d, v15.2d

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_scale_11_8x4_lib4, .-inner_scale_11_8x4_lib4
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
	.type inner_store_8x4_lib4, %function
inner_store_8x4_lib4:
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

	.size	inner_store_8x4_lib4, .-inner_store_8x4_lib4
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
	.type inner_store_8x4_vs_lib4, %function
inner_store_8x4_vs_lib4:
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

	.size	inner_store_8x4_vs_lib4, .-inner_store_8x4_vs_lib4
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
	.type inner_store_l_8x4_lib4, %function
inner_store_l_8x4_lib4:
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

	.size	inner_store_l_8x4_lib4, .-inner_store_l_8x4_lib4
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

	.size	kernel_dgemm_nt_8x4_lib4, .-kernel_dgemm_nt_8x4_lib4





//                                  w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nt_8x4_vs_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int m1, int n1)

	.align	4
	.global	kernel_dgemm_nt_8x4_vs_lib4
	.type	kernel_dgemm_nt_8x4_vs_lib4, %function
kernel_dgemm_nt_8x4_vs_lib4:
	


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

	.size	kernel_dgemm_nt_8x4_vs_lib4, .-kernel_dgemm_nt_8x4_vs_lib4





//                                      w0        x1         w2        x3        x4         w5       x6         w7       sp+0       sp+8
// void kernel_dtrsm_nt_rl_inv_8x4_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.align	4
	.globl kernel_dtrsm_nt_rl_inv_8x4_lib4
	.type kernel_dtrsm_nt_rl_inv_8x4_lib4, %function
kernel_dtrsm_nt_rl_inv_8x4_lib4:



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
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
	bl inner_scale_11_8x4_lib4
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 0)] // E
	ldr		x9, [sp, #(STACKSIZE + 8)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_8X4_LIB4
#else
	bl inner_edge_trsm_rlt_inv_8x4_lib4
#endif



	// store
	mov		x8, x6 // D
	mov		w9, w7 // sdd
	lsl		w9, w9, #5 // 32*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB4
#else
	bl inner_store_8x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dtrsm_nt_rl_inv_8x4_lib4, .-kernel_dtrsm_nt_rl_inv_8x4_lib4





//                                  w0        x1         w2        x3        x4         w5       x6         w7       sp+0
// void kernel_dpotrf_nt_l_8x4_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.align	4
	.globl kernel_dpotrf_nt_l_8x4_lib4
	.type kernel_dpotrf_nt_l_8x4_lib4, %function
kernel_dpotrf_nt_l_8x4_lib4:



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



	// call inner kernel dsyrk l nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SYRK_L_SUB_NT_8X4_LIB4
#else
	bl	inner_kernel_syrk_l_sub_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #5 // 32*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_11_8X4_LIB4
#else
	bl inner_scale_11_8x4_lib4
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

	.size	kernel_dpotrf_nt_l_8x4_lib4, .-kernel_dpotrf_nt_l_8x4_lib4





