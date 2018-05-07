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
	.macro INNER_KERNEL_GEMM_ADD_NT_16X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_16x4_lib4, %function
inner_kernel_gemm_add_nt_16x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10
	add		x13, x12, x10
	add		x14, x13, x10

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x12, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x14, #0]

	// preload
	ldp		s24, s25, [x11], #8
	ldp		s26, s27, [x11], #8
	ldr		q16, [x9], #16
	ldr		q17, [x12], #16
	ldr		q18, [x13], #16
	ldr		q19, [x14], #16

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x12, #32]
	prfm	PLDL1KEEP, [x13, #32]
	prfm	PLDL1KEEP, [x14, #32]

	// main loop
1:
	
	// unroll 0
	ldp		s28, s29, [x11], #8
	fmla	v0.4s, v16.4s, v24.4s[0]
	fmla	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmla	v2.4s, v16.4s, v26.4s[0]
	fmla	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmla	v4.4s, v17.4s, v24.4s[0]
	fmla	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmla	v6.4s, v17.4s, v26.4s[0]
	fmla	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmla	v8.4s, v18.4s, v24.4s[0]
	fmla	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmla	v10.4s, v18.4s, v26.4s[0]
	fmla	v11.4s, v18.4s, v27.4s[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v12.4s, v19.4s, v24.4s[0]
	fmla	v13.4s, v19.4s, v25.4s[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v14.4s, v19.4s, v26.4s[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmla	v15.4s, v19.4s, v27.4s[0]


	// unroll 1
	ldp		s24, s25, [x11], #8
	fmla	v0.4s, v20.4s, v28.4s[0]
	fmla	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmla	v2.4s, v20.4s, v30.4s[0]
	fmla	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmla	v4.4s, v21.4s, v28.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmla	v6.4s, v21.4s, v30.4s[0]
	fmla	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmla	v8.4s, v22.4s, v28.4s[0]
	fmla	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmla	v10.4s, v22.4s, v30.4s[0]
	fmla	v11.4s, v22.4s, v31.4s[0]
	prfm	PLDL1KEEP, [x13, #32]
	fmla	v12.4s, v23.4s, v28.4s[0]
	fmla	v13.4s, v23.4s, v29.4s[0]
	prfm	PLDL1KEEP, [x14, #32]
	fmla	v14.4s, v23.4s, v30.4s[0]
	fmla	v15.4s, v23.4s, v31.4s[0]

	// unroll 2
	ldp		s28, s29, [x11], #8
	fmla	v0.4s, v16.4s, v24.4s[0]
	fmla	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmla	v2.4s, v16.4s, v26.4s[0]
	fmla	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmla	v4.4s, v17.4s, v24.4s[0]
	fmla	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmla	v6.4s, v17.4s, v26.4s[0]
	fmla	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmla	v8.4s, v18.4s, v24.4s[0]
	fmla	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmla	v10.4s, v18.4s, v26.4s[0]
	fmla	v11.4s, v18.4s, v27.4s[0]
	fmla	v12.4s, v19.4s, v24.4s[0]
	fmla	v13.4s, v19.4s, v25.4s[0]
	fmla	v14.4s, v19.4s, v26.4s[0]
	fmla	v15.4s, v19.4s, v27.4s[0]


	// unroll 3
	ldp		s24, s25, [x11], #8
	fmla	v0.4s, v20.4s, v28.4s[0]
	fmla	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmla	v2.4s, v20.4s, v30.4s[0]
	fmla	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmla	v4.4s, v21.4s, v28.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmla	v6.4s, v21.4s, v30.4s[0]
	fmla	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmla	v8.4s, v22.4s, v28.4s[0]
	fmla	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmla	v10.4s, v22.4s, v30.4s[0]
	fmla	v11.4s, v22.4s, v31.4s[0]
	sub		w8, w8, #4
	fmla	v12.4s, v23.4s, v28.4s[0]
	fmla	v13.4s, v23.4s, v29.4s[0]
	cmp		w8, #4
	fmla	v14.4s, v23.4s, v30.4s[0]
	fmla	v15.4s, v23.4s, v31.4s[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		s28, s29, [x11], #8
	fmla	v0.4s, v16.4s, v24.4s[0]
	fmla	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmla	v2.4s, v16.4s, v26.4s[0]
	fmla	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmla	v4.4s, v17.4s, v24.4s[0]
	fmla	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmla	v6.4s, v17.4s, v26.4s[0]
	fmla	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmla	v8.4s, v18.4s, v24.4s[0]
	fmla	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmla	v10.4s, v18.4s, v26.4s[0]
	fmla	v11.4s, v18.4s, v27.4s[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmla	v12.4s, v19.4s, v24.4s[0]
	fmla	v13.4s, v19.4s, v25.4s[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmla	v14.4s, v19.4s, v26.4s[0]
	fmla	v15.4s, v19.4s, v27.4s[0]


	// unroll 1
	ldp		s24, s25, [x11], #8
	fmla	v0.4s, v20.4s, v28.4s[0]
	fmla	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmla	v2.4s, v20.4s, v30.4s[0]
	fmla	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmla	v4.4s, v21.4s, v28.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmla	v6.4s, v21.4s, v30.4s[0]
	fmla	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmla	v8.4s, v22.4s, v28.4s[0]
	fmla	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmla	v10.4s, v22.4s, v30.4s[0]
	fmla	v11.4s, v22.4s, v31.4s[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmla	v12.4s, v23.4s, v28.4s[0]
	fmla	v13.4s, v23.4s, v29.4s[0]
//	prfm	PLDL1KEEP, [x13, #64]
	fmla	v14.4s, v23.4s, v30.4s[0]
	fmla	v15.4s, v23.4s, v31.4s[0]

	// unroll 2
	ldp		s28, s29, [x11], #8
	fmla	v0.4s, v16.4s, v24.4s[0]
	fmla	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmla	v2.4s, v16.4s, v26.4s[0]
	fmla	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmla	v4.4s, v17.4s, v24.4s[0]
	fmla	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmla	v6.4s, v17.4s, v26.4s[0]
	fmla	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmla	v8.4s, v18.4s, v24.4s[0]
	fmla	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmla	v10.4s, v18.4s, v26.4s[0]
	fmla	v11.4s, v18.4s, v27.4s[0]
//	prfm	PLDL1KEEP, [x14, #64]
	fmla	v12.4s, v19.4s, v24.4s[0]
	fmla	v13.4s, v19.4s, v25.4s[0]
	fmla	v14.4s, v19.4s, v26.4s[0]
	fmla	v15.4s, v19.4s, v27.4s[0]


	// unroll 3
	ldp		s24, s25, [x11], #8
	fmla	v0.4s, v20.4s, v28.4s[0]
	fmla	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmla	v2.4s, v20.4s, v30.4s[0]
	fmla	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmla	v4.4s, v21.4s, v28.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmla	v6.4s, v21.4s, v30.4s[0]
	fmla	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmla	v8.4s, v22.4s, v28.4s[0]
	fmla	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmla	v10.4s, v22.4s, v30.4s[0]
	fmla	v11.4s, v22.4s, v31.4s[0]
//	sub		w8, w8, #4
	fmla	v12.4s, v23.4s, v28.4s[0]
	fmla	v13.4s, v23.4s, v29.4s[0]
//	cmp		w8, #4
	fmla	v14.4s, v23.4s, v30.4s[0]
	fmla	v15.4s, v23.4s, v31.4s[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x9, x9, #16
	sub		x11, x11, #16
	sub		x12, x12, #16
	sub		x13, x13, #16
	sub		x14, x14, #16

3: // clean1-up loop

	// unroll 0
	// TODO
	ldp		s24, s25, [x11], #8
	ldr		q16, [x9], #16
	fmla	v0.4s, v16.4s, v24.4s[0]
	fmla	v1.4s, v16.4s, v25.4s[0]
	ldp		s26, s27, [x11], #8
	fmla	v2.4s, v16.4s, v26.4s[0]
	fmla	v3.4s, v16.4s, v27.4s[0]
	ldr		q17, [x12], #16
	fmla	v4.4s, v17.4s, v24.4s[0]
	fmla	v5.4s, v17.4s, v25.4s[0]
	fmla	v6.4s, v17.4s, v26.4s[0]
	fmla	v7.4s, v17.4s, v27.4s[0]
	ldr		q18, [x13], #16
	fmla	v8.4s, v18.4s, v24.4s[0]
	fmla	v9.4s, v18.4s, v25.4s[0]
	fmla	v10.4s, v18.4s, v26.4s[0]
	fmla	v11.4s, v18.4s, v27.4s[0]
	ldr		q19, [x14], #16
	fmla	v12.4s, v19.4s, v24.4s[0]
	fmla	v13.4s, v19.4s, v25.4s[0]
	fmla	v14.4s, v19.4s, v26.4s[0]
	fmla	v15.4s, v19.4s, v27.4s[0]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nt_16x4_lib4, .-inner_kernel_gemm_add_nt_16x4_lib4
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
	.macro INNER_KERNEL_GEMM_SUB_NT_16X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_sub_nt_16x4_lib4, %function
inner_kernel_gemm_sub_nt_16x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x9, x10
	add		x13, x12, x10
	add		x14, x13, x10

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x12, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x14, #0]

	// preload
	ldp		s24, s25, [x11], #8
	ldp		s26, s27, [x11], #8
	ldr		q16, [x9], #16
	ldr		q17, [x12], #16
	ldr		q18, [x13], #16
	ldr		q19, [x14], #16

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x12, #32]
	prfm	PLDL1KEEP, [x13, #32]
	prfm	PLDL1KEEP, [x14, #32]

	// main loop
1:
	
	// unroll 0
	ldp		s28, s29, [x11], #8
	fmls	v0.4s, v16.4s, v24.4s[0]
	fmls	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmls	v2.4s, v16.4s, v26.4s[0]
	fmls	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmls	v4.4s, v17.4s, v24.4s[0]
	fmls	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmls	v6.4s, v17.4s, v26.4s[0]
	fmls	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmls	v8.4s, v18.4s, v24.4s[0]
	fmls	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmls	v10.4s, v18.4s, v26.4s[0]
	fmls	v11.4s, v18.4s, v27.4s[0]
	prfm	PLDL1KEEP, [x11, #64]
	fmls	v12.4s, v19.4s, v24.4s[0]
	fmls	v13.4s, v19.4s, v25.4s[0]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v14.4s, v19.4s, v26.4s[0]
	prfm	PLDL1KEEP, [x12, #64]
	fmls	v15.4s, v19.4s, v27.4s[0]


	// unroll 1
	ldp		s24, s25, [x11], #8
	fmls	v0.4s, v20.4s, v28.4s[0]
	fmls	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmls	v2.4s, v20.4s, v30.4s[0]
	fmls	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmls	v4.4s, v21.4s, v28.4s[0]
	fmls	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmls	v6.4s, v21.4s, v30.4s[0]
	fmls	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmls	v8.4s, v22.4s, v28.4s[0]
	fmls	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmls	v10.4s, v22.4s, v30.4s[0]
	fmls	v11.4s, v22.4s, v31.4s[0]
	prfm	PLDL1KEEP, [x13, #32]
	fmls	v12.4s, v23.4s, v28.4s[0]
	fmls	v13.4s, v23.4s, v29.4s[0]
	prfm	PLDL1KEEP, [x14, #32]
	fmls	v14.4s, v23.4s, v30.4s[0]
	fmls	v15.4s, v23.4s, v31.4s[0]

	// unroll 2
	ldp		s28, s29, [x11], #8
	fmls	v0.4s, v16.4s, v24.4s[0]
	fmls	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmls	v2.4s, v16.4s, v26.4s[0]
	fmls	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmls	v4.4s, v17.4s, v24.4s[0]
	fmls	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmls	v6.4s, v17.4s, v26.4s[0]
	fmls	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmls	v8.4s, v18.4s, v24.4s[0]
	fmls	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmls	v10.4s, v18.4s, v26.4s[0]
	fmls	v11.4s, v18.4s, v27.4s[0]
	fmls	v12.4s, v19.4s, v24.4s[0]
	fmls	v13.4s, v19.4s, v25.4s[0]
	fmls	v14.4s, v19.4s, v26.4s[0]
	fmls	v15.4s, v19.4s, v27.4s[0]


	// unroll 3
	ldp		s24, s25, [x11], #8
	fmls	v0.4s, v20.4s, v28.4s[0]
	fmls	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmls	v2.4s, v20.4s, v30.4s[0]
	fmls	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmls	v4.4s, v21.4s, v28.4s[0]
	fmls	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmls	v6.4s, v21.4s, v30.4s[0]
	fmls	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmls	v8.4s, v22.4s, v28.4s[0]
	fmls	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmls	v10.4s, v22.4s, v30.4s[0]
	fmls	v11.4s, v22.4s, v31.4s[0]
	sub		w8, w8, #4
	fmls	v12.4s, v23.4s, v28.4s[0]
	fmls	v13.4s, v23.4s, v29.4s[0]
	cmp		w8, #4
	fmls	v14.4s, v23.4s, v30.4s[0]
	fmls	v15.4s, v23.4s, v31.4s[0]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	ldp		s28, s29, [x11], #8
	fmls	v0.4s, v16.4s, v24.4s[0]
	fmls	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmls	v2.4s, v16.4s, v26.4s[0]
	fmls	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmls	v4.4s, v17.4s, v24.4s[0]
	fmls	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmls	v6.4s, v17.4s, v26.4s[0]
	fmls	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmls	v8.4s, v18.4s, v24.4s[0]
	fmls	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmls	v10.4s, v18.4s, v26.4s[0]
	fmls	v11.4s, v18.4s, v27.4s[0]
//	prfm	PLDL1KEEP, [x11, #64]
	fmls	v12.4s, v19.4s, v24.4s[0]
	fmls	v13.4s, v19.4s, v25.4s[0]
//	prfm	PLDL1KEEP, [x9, #64]
	fmls	v14.4s, v19.4s, v26.4s[0]
	fmls	v15.4s, v19.4s, v27.4s[0]


	// unroll 1
	ldp		s24, s25, [x11], #8
	fmls	v0.4s, v20.4s, v28.4s[0]
	fmls	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmls	v2.4s, v20.4s, v30.4s[0]
	fmls	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmls	v4.4s, v21.4s, v28.4s[0]
	fmls	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmls	v6.4s, v21.4s, v30.4s[0]
	fmls	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmls	v8.4s, v22.4s, v28.4s[0]
	fmls	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmls	v10.4s, v22.4s, v30.4s[0]
	fmls	v11.4s, v22.4s, v31.4s[0]
//	prfm	PLDL1KEEP, [x12, #64]
	fmls	v12.4s, v23.4s, v28.4s[0]
	fmls	v13.4s, v23.4s, v29.4s[0]
//	prfm	PLDL1KEEP, [x13, #64]
	fmls	v14.4s, v23.4s, v30.4s[0]
	fmls	v15.4s, v23.4s, v31.4s[0]

	// unroll 2
	ldp		s28, s29, [x11], #8
	fmls	v0.4s, v16.4s, v24.4s[0]
	fmls	v1.4s, v16.4s, v25.4s[0]
	ldp		s30, s31, [x11], #8
	fmls	v2.4s, v16.4s, v26.4s[0]
	fmls	v3.4s, v16.4s, v27.4s[0]
	ldr		q20, [x9], #16
	fmls	v4.4s, v17.4s, v24.4s[0]
	fmls	v5.4s, v17.4s, v25.4s[0]
	ldr		q21, [x12], #16
	fmls	v6.4s, v17.4s, v26.4s[0]
	fmls	v7.4s, v17.4s, v27.4s[0]
	ldr		q22, [x13], #16
	fmls	v8.4s, v18.4s, v24.4s[0]
	fmls	v9.4s, v18.4s, v25.4s[0]
	ldr		q23, [x14], #16
	fmls	v10.4s, v18.4s, v26.4s[0]
	fmls	v11.4s, v18.4s, v27.4s[0]
//	prfm	PLDL1KEEP, [x14, #64]
	fmls	v12.4s, v19.4s, v24.4s[0]
	fmls	v13.4s, v19.4s, v25.4s[0]
	fmls	v14.4s, v19.4s, v26.4s[0]
	fmls	v15.4s, v19.4s, v27.4s[0]


	// unroll 3
	ldp		s24, s25, [x11], #8
	fmls	v0.4s, v20.4s, v28.4s[0]
	fmls	v1.4s, v20.4s, v29.4s[0]
	ldp		s26, s27, [x11], #8
	fmls	v2.4s, v20.4s, v30.4s[0]
	fmls	v3.4s, v20.4s, v31.4s[0]
	ldr		q16, [x9], #16
	fmls	v4.4s, v21.4s, v28.4s[0]
	fmls	v5.4s, v21.4s, v29.4s[0]
	ldr		q17, [x12], #16
	fmls	v6.4s, v21.4s, v30.4s[0]
	fmls	v7.4s, v21.4s, v31.4s[0]
	ldr		q18, [x13], #16
	fmls	v8.4s, v22.4s, v28.4s[0]
	fmls	v9.4s, v22.4s, v29.4s[0]
	ldr		q19, [x14], #16
	fmls	v10.4s, v22.4s, v30.4s[0]
	fmls	v11.4s, v22.4s, v31.4s[0]
//	sub		w8, w8, #4
	fmls	v12.4s, v23.4s, v28.4s[0]
	fmls	v13.4s, v23.4s, v29.4s[0]
//	cmp		w8, #4
	fmls	v14.4s, v23.4s, v30.4s[0]
	fmls	v15.4s, v23.4s, v31.4s[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x9, x9, #16
	sub		x11, x11, #16
	sub		x12, x12, #16
	sub		x13, x13, #16
	sub		x14, x14, #16

3: // clean1-up loop

	// unroll 0
	// TODO
	ldp		s24, s25, [x11], #8
	ldr		q16, [x9], #16
	fmls	v0.4s, v16.4s, v24.4s[0]
	fmls	v1.4s, v16.4s, v25.4s[0]
	ldp		s26, s27, [x11], #8
	fmls	v2.4s, v16.4s, v26.4s[0]
	fmls	v3.4s, v16.4s, v27.4s[0]
	ldr		q17, [x12], #16
	fmls	v4.4s, v17.4s, v24.4s[0]
	fmls	v5.4s, v17.4s, v25.4s[0]
	fmls	v6.4s, v17.4s, v26.4s[0]
	fmls	v7.4s, v17.4s, v27.4s[0]
	ldr		q18, [x13], #16
	fmls	v8.4s, v18.4s, v24.4s[0]
	fmls	v9.4s, v18.4s, v25.4s[0]
	fmls	v10.4s, v18.4s, v26.4s[0]
	fmls	v11.4s, v18.4s, v27.4s[0]
	ldr		q19, [x14], #16
	fmls	v12.4s, v19.4s, v24.4s[0]
	fmls	v13.4s, v19.4s, v25.4s[0]
	fmls	v14.4s, v19.4s, v26.4s[0]
	fmls	v15.4s, v19.4s, v27.4s[0]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_sub_nt_16x4_lib4, .-inner_kernel_gemm_sub_nt_16x4_lib4
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
	.macro INNER_EDGE_TRSM_RLT_INV_16X4_LIB4
#else
	.align 4
	.type inner_edge_trsm_rlt_inv_16x4_lib4, %function
inner_edge_trsm_rlt_inv_16x4_lib4:
#endif
	
	// first column
	ldr			s16, [x9, #0] // E_inv[0]
	fmul		v0.4s, v0.4s, v16.4s[0]
	fmul		v4.4s, v4.4s, v16.4s[0]
	fmul		v8.4s, v8.4s, v16.4s[0]
	fmul		v12.4s, v12.4s, v16.4s[0]

	// second column
	ldr			s16, [x8, #4] // E[1+4*0]
	fmls		v1.4s, v0.4s, v16.4s[0]
	fmls		v5.4s, v4.4s, v16.4s[0]
	fmls		v9.4s, v8.4s, v16.4s[0]
	fmls		v13.4s, v12.4s, v16.4s[0]
	ldr			s16, [x9, #4] // E_inv[1]
	fmul		v1.4s, v1.4s, v16.4s[0]
	fmul		v5.4s, v5.4s, v16.4s[0]
	fmul		v9.4s, v9.4s, v16.4s[0]
	fmul		v13.4s, v13.4s, v16.4s[0]

	// third column
	ldr			s16, [x8, #8] // E[2+4*0]
	fmls		v2.4s, v0.4s, v16.4s[0]
	fmls		v6.4s, v4.4s, v16.4s[0]
	fmls		v10.4s, v8.4s, v16.4s[0]
	fmls		v14.4s, v12.4s, v16.4s[0]
	ldr			s16, [x8, #24] // E[2+4*1]
	fmls		v2.4s, v1.4s, v16.4s[0]
	fmls		v6.4s, v5.4s, v16.4s[0]
	fmls		v10.4s, v9.4s, v16.4s[0]
	fmls		v14.4s, v13.4s, v16.4s[0]
	ldr			s16, [x9, #8] // E_inv[2]
	fmul		v2.4s, v2.4s, v16.4s[0]
	fmul		v6.4s, v6.4s, v16.4s[0]
	fmul		v10.4s, v10.4s, v16.4s[0]
	fmul		v14.4s, v14.4s, v16.4s[0]

	// forth column
	ldr			s16, [x8, #12] // E[3+4*0]
	fmls		v3.4s, v0.4s, v16.4s[0]
	fmls		v7.4s, v4.4s, v16.4s[0]
	fmls		v11.4s, v8.4s, v16.4s[0]
	fmls		v15.4s, v12.4s, v16.4s[0]
	ldr			s16, [x8, #28] // E[3+4*1]
	fmls		v3.4s, v1.4s, v16.4s[0]
	fmls		v7.4s, v5.4s, v16.4s[0]
	fmls		v11.4s, v9.4s, v16.4s[0]
	fmls		v15.4s, v13.4s, v16.4s[0]
	ldr			s16, [x8, #44] // E[3+4*1]
	fmls		v3.4s, v2.4s, v16.4s[0]
	fmls		v7.4s, v6.4s, v16.4s[0]
	fmls		v11.4s, v10.4s, v16.4s[0]
	fmls		v15.4s, v14.4s, v16.4s[0]
	ldr			s16, [x9, #12] // E_inv[2]
	fmul		v3.4s, v3.4s, v16.4s[0]
	fmul		v7.4s, v7.4s, v16.4s[0]
	fmul		v11.4s, v11.4s, v16.4s[0]
	fmul		v15.4s, v15.4s, v16.4s[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_trsm_rlt_inv_16x4_lib4, .-inner_edge_trsm_rlt_inv_16x4_lib4
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
	.macro INNER_EDGE_POTRF_16X4_LIB4
#else
	.p2align 4
	.type inner_edge_potrf_16x4_lib4, %function
inner_edge_potrf_16x4_lib4:
#endif
	
	fmov		s16, 1.0e+0 // 1.0

	// first column
	ins			v17.s[0], v0.s[0]
	fcmpe		s17, #0
	ble			1f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
2:
	str			s18, [x8, #0]
	fmul		v0.4s, v0.4s, v18.4s[0]
	fmul		v4.4s, v4.4s, v18.4s[0]
	fmul		v8.4s, v8.4s, v18.4s[0]
	fmul		v12.4s, v12.4s, v18.4s[0]

	// second column
	fmls		v1.4s, v0.4s, v0.4s[1]
	fmls		v5.4s, v4.4s, v0.4s[1]
	fmls		v9.4s, v8.4s, v0.4s[1]
	fmls		v13.4s, v12.4s, v0.4s[1]
	ins			v17.s[0], v1.s[1]
	fcmpe		s17, #0
	ble			3f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
4:
	str			s18, [x8, #4]
	fmul		v1.4s, v1.4s, v18.4s[0]
	fmul		v5.4s, v5.4s, v18.4s[0]
	fmul		v9.4s, v9.4s, v18.4s[0]
	fmul		v13.4s, v13.4s, v18.4s[0]

	// third column
	fmls		v2.4s, v0.4s, v0.4s[2]
	fmls		v6.4s, v4.4s, v0.4s[2]
	fmls		v10.4s, v8.4s, v0.4s[2]
	fmls		v14.4s, v12.4s, v0.4s[2]
	fmls		v2.4s, v1.4s, v1.4s[2]
	fmls		v6.4s, v5.4s, v1.4s[2]
	fmls		v10.4s, v9.4s, v1.4s[2]
	fmls		v14.4s, v13.4s, v1.4s[2]
	ins			v17.s[0], v2.s[2]
	fcmpe		s17, #0
	ble			5f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
6:
	str			s18, [x8, #8]
	fmul		v2.4s, v2.4s, v18.4s[0]
	fmul		v6.4s, v6.4s, v18.4s[0]
	fmul		v10.4s, v10.4s, v18.4s[0]
	fmul		v14.4s, v14.4s, v18.4s[0]

	// fourth column
	fmls		v3.4s, v0.4s, v0.4s[3]
	fmls		v7.4s, v4.4s, v0.4s[3]
	fmls		v11.4s, v8.4s, v0.4s[3]
	fmls		v15.4s, v12.4s, v0.4s[3]
	fmls		v3.4s, v1.4s, v1.4s[3]
	fmls		v7.4s, v5.4s, v1.4s[3]
	fmls		v11.4s, v9.4s, v1.4s[3]
	fmls		v15.4s, v13.4s, v1.4s[3]
	fmls		v3.4s, v2.4s, v2.4s[3]
	fmls		v7.4s, v6.4s, v2.4s[3]
	fmls		v11.4s, v10.4s, v2.4s[3]
	fmls		v15.4s, v14.4s, v2.4s[3]
	ins			v17.s[0], v3.s[3]
	fcmpe		s17, #0
	ble			7f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
8:
	str			s18, [x8, #12]
	fmul		v3.4s, v3.4s, v18.4s[0]
	fmul		v7.4s, v7.4s, v18.4s[0]
	fmul		v11.4s, v11.4s, v18.4s[0]
	fmul		v15.4s, v15.4s, v18.4s[0]

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

	.size	inner_edge_potrf_16x4_lib4, .-inner_edge_potrf_16x4_lib4
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
	.macro INNER_SCALE_AB_16X4_LIB4
#else
	.align	4
	.type inner_scale_ab_16x4_lib4, %function
inner_scale_ab_16x4_lib4:
#endif

	ld1		{v28.4s}, [x8]

	fmul	v0.4s, v0.4s, v28.4s[0]
	fmul	v1.4s, v1.4s, v28.4s[0]
	fmul	v2.4s, v2.4s, v28.4s[0]
	fmul	v3.4s, v3.4s, v28.4s[0]
	fmul	v4.4s, v4.4s, v28.4s[0]
	fmul	v5.4s, v5.4s, v28.4s[0]
	fmul	v6.4s, v6.4s, v28.4s[0]
	fmul	v7.4s, v7.4s, v28.4s[0]
	fmul	v8.4s, v8.4s, v28.4s[0]
	fmul	v9.4s, v9.4s, v28.4s[0]
	fmul	v10.4s, v10.4s, v28.4s[0]
	fmul	v11.4s, v11.4s, v28.4s[0]
	fmul	v12.4s, v12.4s, v28.4s[0]
	fmul	v13.4s, v13.4s, v28.4s[0]
	fmul	v14.4s, v14.4s, v28.4s[0]
	fmul	v15.4s, v15.4s, v28.4s[0]

	ld1		{v28.4s}, [x9]

	add		x12, x10, x11
	add		x13, x12, x11
	add		x14, x13, x11

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v25.4s, v28.4s[0]
	fmla	v2.4s, v26.4s, v28.4s[0]
	fmla	v3.4s, v27.4s, v28.4s[0]

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x12], #64
	fmla	v4.4s, v24.4s, v28.4s[0]
	fmla	v5.4s, v25.4s, v28.4s[0]
	fmla	v6.4s, v26.4s, v28.4s[0]
	fmla	v7.4s, v27.4s, v28.4s[0]

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x13], #64
	fmla	v8.4s, v24.4s, v28.4s[0]
	fmla	v9.4s, v25.4s, v28.4s[0]
	fmla	v10.4s, v26.4s, v28.4s[0]
	fmla	v11.4s, v27.4s, v28.4s[0]

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x14], #64
	fmla	v12.4s, v24.4s, v28.4s[0]
	fmla	v13.4s, v25.4s, v28.4s[0]
	fmla	v14.4s, v26.4s, v28.4s[0]
	fmla	v15.4s, v27.4s, v28.4s[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_16x4_lib4, .-inner_scale_ab_16x4_lib4
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- sdc
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_11_16X4_LIB4
#else
	.align	4
	.type inner_scale_11_16x4_lib4, %function
inner_scale_11_16x4_lib4:
#endif

	add		x12, x8, x9
	add		x13, x12, x9
	add		x14, x13, x9

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x8], #64
	fadd	v0.4s, v24.4s, v0.4s
	fadd	v1.4s, v25.4s, v1.4s
	fadd	v2.4s, v26.4s, v2.4s
	fadd	v3.4s, v27.4s, v3.4s

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x12], #64
	fadd	v4.4s, v24.4s, v4.4s
	fadd	v5.4s, v25.4s, v5.4s
	fadd	v6.4s, v26.4s, v6.4s
	fadd	v7.4s, v27.4s, v7.4s

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x13], #64
	fadd	v8.4s, v24.4s, v8.4s
	fadd	v9.4s, v25.4s, v9.4s
	fadd	v10.4s, v26.4s, v10.4s
	fadd	v11.4s, v27.4s, v11.4s

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x14], #64
	fadd	v12.4s, v24.4s, v12.4s
	fadd	v13.4s, v25.4s, v13.4s
	fadd	v14.4s, v26.4s, v14.4s
	fadd	v15.4s, v27.4s, v15.4s

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_11_16x4_lib4, .-inner_scale_11_16x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_16X4_LIB4
#else
	.align 4
	.type inner_store_16x4_lib4, %function
inner_store_16x4_lib4:
#endif

	add		x10, x8, x9
	add		x11, x10, x9
	add		x12, x11, x9

	st1		{v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
	st1		{v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
	st1		{v8.4s, v9.4s, v10.4s, v11.4s}, [x11], #64
	st1		{v12.4s, v13.4s, v14.4s, v15.4s}, [x12], #64

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_16x4_lib4, .-inner_store_16x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_L_16X4_LIB4
#else
	.align 4
	.type inner_store_l_16x4_lib4, %function
inner_store_l_16x4_lib4:
#endif

	ldr		q16, [x8, #16]
	ldr		q17, [x8, #32]
	ldr		q18, [x8, #48]

	ins		v1.s[0], v16.s[0]
	ins		v2.d[0], v17.d[0]
	ins		v3.d[0], v18.d[0]
	ins		v3.s[2], v18.s[2]

	add		x10, x8, x9
	add		x11, x10, x9
	add		x12, x11, x9

	st1		{v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
	st1		{v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
	st1		{v8.4s, v9.4s, v10.4s, v11.4s}, [x11], #64
	st1		{v12.4s, v13.4s, v14.4s, v15.4s}, [x12], #64

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_l_16x4_lib4, .-inner_store_l_16x4_lib4
#endif





//                                w0        x1            x2        w3       x4        x5           x6        w7       sp+0      sp+8
// void kernel_sgemm_nt_16x4_lib4(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int sdc, float *D, int sdd)

	.align	4
	.global	kernel_sgemm_nt_16x4_lib4
	.type	kernel_sgemm_nt_16x4_lib4, %function
kernel_sgemm_nt_16x4_lib4:
	


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
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_16X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_16x4_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // C
	lsl		w11, w11, #4 // 16*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X4_LIB4
#else
	bl inner_scale_ab_16x4_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // sdd
	lsl		w9, w9, #4 // 16*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_16X4_LIB4
#else
	bl inner_store_16x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_sgemm_nt_16x4_lib4, .-kernel_sgemm_nt_16x4_lib4





//                                      w0        x1         w2        x3        x4         w5       x6         w7       sp+0       sp+8
// void kernel_strsm_nt_rl_inv_16x4_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.align	4
	.globl kernel_strsm_nt_rl_inv_16x4_lib4
	.type kernel_strsm_nt_rl_inv_16x4_lib4, %function
kernel_strsm_nt_rl_inv_16x4_lib4:



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
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_16X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_16x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #4 // 16*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_11_16X4_LIB4
#else
	bl inner_scale_11_16x4_lib4
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 0)] // E
	ldr		x9, [sp, #(STACKSIZE + 8)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_16X4_LIB4
#else
	bl inner_edge_trsm_rlt_inv_16x4_lib4
#endif



	// store
	mov		x8, x6 // D
	mov		w9, w7 // sdd
	lsl		w9, w9, #4 // 16*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_16X4_LIB4
#else
	bl inner_store_16x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_strsm_nt_rl_inv_16x4_lib4, .-kernel_strsm_nt_rl_inv_16x4_lib4





//                                  w0        x1         w2        x3        x4         w5       x6         w7       sp+0
// void kernel_spotrf_nt_l_16x4_lib4(int kmax, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.align	4
	.globl kernel_spotrf_nt_l_16x4_lib4
	.type kernel_spotrf_nt_l_16x4_lib4, %function
kernel_spotrf_nt_l_16x4_lib4:



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
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_16X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_16x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // sdc
	lsl		w9, w9, #4 // 16*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_11_16X4_LIB4
#else
	bl inner_scale_11_16x4_lib4
#endif



	// factorization
	ldr		x8, [sp, #(STACKSIZE + 0)] // inv_diag_D

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_16X4_LIB4
#else
	bl inner_edge_potrf_16x4_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // sdd
	lsl		w9, w9, #4 // 16*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_L_16X4_LIB4
#else
	bl inner_store_l_16x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_spotrf_nt_l_16x4_lib4, .-kernel_spotrf_nt_l_16x4_lib4





