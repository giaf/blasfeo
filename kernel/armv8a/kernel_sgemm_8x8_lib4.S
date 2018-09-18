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
	.macro INNER_KERNEL_GEMM_ADD_NT_8X8_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_8x8_lib4, %function
inner_kernel_gemm_add_nt_8x8_lib4:
#endif

	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10
	add		x14, x11, x12

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x14, #0]

	// preload
	ld1		{v24.4s, v25.4s}, [x9], #32
	ld1		{v28.4s, v29.4s}, [x11], #32
	ld1		{v20.4s, v21.4s}, [x13], #32
	ld1		{v16.4s, v17.4s}, [x14], #32

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x11, #32]
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x13, #32]
	prfm	PLDL1KEEP, [x14, #32]

	// main loop
1:

	// unroll 0
	ld1		{v26.4s}, [x9], #16
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	ld1		{v27.4s}, [x9], #16
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]
	ld1		{v30.4s}, [x11], #16
	fmla	v4.4s, v20.4s, v28.4s[0]
	fmla	v5.4s, v20.4s, v28.4s[1]
	ld1		{v31.4s}, [x11], #16
	fmla	v6.4s, v20.4s, v28.4s[2]
	fmla	v7.4s, v20.4s, v28.4s[3]
	ld1		{v22.4s}, [x13], #16
	fmla	v8.4s, v24.4s, v16.4s[0]
	fmla	v9.4s, v24.4s, v16.4s[1]
	ld1		{v23.4s}, [x13], #16
	fmla	v10.4s, v24.4s, v16.4s[2]
	fmla	v11.4s, v24.4s, v16.4s[3]
	ld1		{v18.4s}, [x14], #16
	fmla	v12.4s, v20.4s, v16.4s[0]
	fmla	v13.4s, v20.4s, v16.4s[1]
	ld1		{v19.4s}, [x14], #16
	fmla	v14.4s, v20.4s, v16.4s[2]
	fmla	v15.4s, v20.4s, v16.4s[3]

	// unroll 1
	prfm	PLDL1KEEP, [x11, #64]
	fmla	v0.4s, v25.4s, v29.4s[0]
	fmla	v1.4s, v25.4s, v29.4s[1]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v2.4s, v25.4s, v29.4s[2]
	fmla	v3.4s, v25.4s, v29.4s[3]
	prfm	PLDL1KEEP, [x13, #64]
	fmla	v4.4s, v21.4s, v29.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[1]
	prfm	PLDL1KEEP, [x14, #64]
	fmla	v6.4s, v21.4s, v29.4s[2]
	fmla	v7.4s, v21.4s, v29.4s[3]
	sub		w8, w8, #4
	fmla	v8.4s, v25.4s, v17.4s[0]
	fmla	v9.4s, v25.4s, v17.4s[1]
	fmla	v10.4s, v25.4s, v17.4s[2]
	fmla	v11.4s, v25.4s, v17.4s[3]
	fmla	v12.4s, v21.4s, v17.4s[0]
	fmla	v13.4s, v21.4s, v17.4s[1]
	cmp		w8, #4
	fmla	v14.4s, v21.4s, v17.4s[2]
	fmla	v15.4s, v21.4s, v17.4s[3]

	// unroll 2
	ld1		{v24.4s}, [x9], #16
	fmla	v0.4s, v26.4s, v30.4s[0]
	fmla	v1.4s, v26.4s, v30.4s[1]
	ld1		{v25.4s}, [x9], #16
	fmla	v2.4s, v26.4s, v30.4s[2]
	fmla	v3.4s, v26.4s, v30.4s[3]
	ld1		{v28.4s}, [x11], #16
	fmla	v4.4s, v22.4s, v30.4s[0]
	fmla	v5.4s, v22.4s, v30.4s[1]
	ld1		{v29.4s}, [x11], #16
	fmla	v6.4s, v22.4s, v30.4s[2]
	fmla	v7.4s, v22.4s, v30.4s[3]
	ld1		{v20.4s}, [x13], #16
	fmla	v8.4s, v26.4s, v18.4s[0]
	fmla	v9.4s, v26.4s, v18.4s[1]
	ld1		{v21.4s}, [x13], #16
	fmla	v10.4s, v26.4s, v18.4s[2]
	fmla	v11.4s, v26.4s, v18.4s[3]
	ld1		{v16.4s}, [x14], #16
	fmla	v12.4s, v22.4s, v18.4s[0]
	fmla	v13.4s, v22.4s, v18.4s[1]
	ld1		{v17.4s}, [x14], #16
	fmla	v14.4s, v22.4s, v18.4s[2]
	fmla	v15.4s, v22.4s, v18.4s[3]

	// unroll 3
	fmla	v0.4s, v27.4s, v31.4s[0]
	fmla	v1.4s, v27.4s, v31.4s[1]
	fmla	v2.4s, v27.4s, v31.4s[2]
	fmla	v3.4s, v27.4s, v31.4s[3]
	fmla	v4.4s, v23.4s, v31.4s[0]
	fmla	v5.4s, v23.4s, v31.4s[1]
	fmla	v6.4s, v23.4s, v31.4s[2]
	fmla	v7.4s, v23.4s, v31.4s[3]
	fmla	v8.4s, v27.4s, v19.4s[0]
	fmla	v9.4s, v27.4s, v19.4s[1]
	fmla	v10.4s, v27.4s, v19.4s[2]
	fmla	v11.4s, v27.4s, v19.4s[3]
	fmla	v12.4s, v23.4s, v19.4s[0]
	fmla	v13.4s, v23.4s, v19.4s[1]
	fmla	v14.4s, v23.4s, v19.4s[2]
	fmla	v15.4s, v23.4s, v19.4s[3]

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v26.4s}, [x9], #16
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	ld1		{v27.4s}, [x9], #16
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]
	ld1		{v30.4s}, [x11], #16
	fmla	v4.4s, v20.4s, v28.4s[0]
	fmla	v5.4s, v20.4s, v28.4s[1]
	ld1		{v31.4s}, [x11], #16
	fmla	v6.4s, v20.4s, v28.4s[2]
	fmla	v7.4s, v20.4s, v28.4s[3]
	ld1		{v22.4s}, [x13], #16
	fmla	v8.4s, v24.4s, v16.4s[0]
	fmla	v9.4s, v24.4s, v16.4s[1]
	ld1		{v23.4s}, [x13], #16
	fmla	v10.4s, v24.4s, v16.4s[2]
	fmla	v11.4s, v24.4s, v16.4s[3]
	ld1		{v18.4s}, [x14], #16
	fmla	v12.4s, v20.4s, v16.4s[0]
	fmla	v13.4s, v20.4s, v16.4s[1]
	ld1		{v19.4s}, [x14], #16
	fmla	v14.4s, v20.4s, v16.4s[2]
	fmla	v15.4s, v20.4s, v16.4s[3]

	// unroll 1
//	prfm	PLDL1KEEP, [x11, #64]
	fmla	v0.4s, v25.4s, v29.4s[0]
	fmla	v1.4s, v25.4s, v29.4s[1]
//	prfm	PLDL1KEEP, [x9, #64]
	fmla	v2.4s, v25.4s, v29.4s[2]
	fmla	v3.4s, v25.4s, v29.4s[3]
//	prfm	PLDL1KEEP, [x13, #64]
	fmla	v4.4s, v21.4s, v29.4s[0]
	fmla	v5.4s, v21.4s, v29.4s[1]
//	prfm	PLDL1KEEP, [x14, #64]
	fmla	v6.4s, v21.4s, v29.4s[2]
	fmla	v7.4s, v21.4s, v29.4s[3]
	sub		w8, w8, #4
	fmla	v8.4s, v25.4s, v17.4s[0]
	fmla	v9.4s, v25.4s, v17.4s[1]
	fmla	v10.4s, v25.4s, v17.4s[2]
	fmla	v11.4s, v25.4s, v17.4s[3]
	fmla	v12.4s, v21.4s, v17.4s[0]
	fmla	v13.4s, v21.4s, v17.4s[1]
	cmp		w8, #4
	fmla	v14.4s, v21.4s, v17.4s[2]
	fmla	v15.4s, v21.4s, v17.4s[3]

	// unroll 2
//	ld1		{v24.4s}, [x9], #16
	fmla	v0.4s, v26.4s, v30.4s[0]
	fmla	v1.4s, v26.4s, v30.4s[1]
//	ld1		{v25.4s}, [x9], #16
	fmla	v2.4s, v26.4s, v30.4s[2]
	fmla	v3.4s, v26.4s, v30.4s[3]
//	ld1		{v28.4s}, [x11], #16
	fmla	v4.4s, v22.4s, v30.4s[0]
	fmla	v5.4s, v22.4s, v30.4s[1]
//	ld1		{v29.4s}, [x11], #16
	fmla	v6.4s, v22.4s, v30.4s[2]
	fmla	v7.4s, v22.4s, v30.4s[3]
//	ld1		{v20.4s}, [x13], #16
	fmla	v8.4s, v26.4s, v18.4s[0]
	fmla	v9.4s, v26.4s, v18.4s[1]
//	ld1		{v21.4s}, [x13], #16
	fmla	v10.4s, v26.4s, v18.4s[2]
	fmla	v11.4s, v26.4s, v18.4s[3]
//	ld1		{v16.4s}, [x14], #16
	fmla	v12.4s, v22.4s, v18.4s[0]
	fmla	v13.4s, v22.4s, v18.4s[1]
//	ld1		{v17.4s}, [x14], #16
	fmla	v14.4s, v22.4s, v18.4s[2]
	fmla	v15.4s, v22.4s, v18.4s[3]

	// unroll 3
	fmla	v0.4s, v27.4s, v31.4s[0]
	fmla	v1.4s, v27.4s, v31.4s[1]
	fmla	v2.4s, v27.4s, v31.4s[2]
	fmla	v3.4s, v27.4s, v31.4s[3]
	fmla	v4.4s, v23.4s, v31.4s[0]
	fmla	v5.4s, v23.4s, v31.4s[1]
	fmla	v6.4s, v23.4s, v31.4s[2]
	fmla	v7.4s, v23.4s, v31.4s[3]
	fmla	v8.4s, v27.4s, v19.4s[0]
	fmla	v9.4s, v27.4s, v19.4s[1]
	fmla	v10.4s, v27.4s, v19.4s[2]
	fmla	v11.4s, v27.4s, v19.4s[3]
	fmla	v12.4s, v23.4s, v19.4s[0]
	fmla	v13.4s, v23.4s, v19.4s[1]
	fmla	v14.4s, v23.4s, v19.4s[2]
	fmla	v15.4s, v23.4s, v19.4s[3]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x9, x9, #32
	sub		x13, x13, #32
	sub		x11, x11, #32
	sub		x14, x14, #32

3: // clean1-up loop

	// unroll 0

	ld1		{v28.4s}, [x11], #16
	ld1		{v24.4s}, [x9], #16
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]
	ld1		{v20.4s}, [x13], #16
	fmla	v4.4s, v20.4s, v28.4s[0]
	fmla	v5.4s, v20.4s, v28.4s[1]
	fmla	v6.4s, v20.4s, v28.4s[2]
	fmla	v7.4s, v20.4s, v28.4s[3]
	ld1		{v16.4s}, [x14], #16
	fmla	v8.4s, v24.4s, v16.4s[0]
	fmla	v9.4s, v24.4s, v16.4s[1]
	fmla	v10.4s, v24.4s, v16.4s[2]
	fmla	v11.4s, v24.4s, v16.4s[3]
	fmla	v12.4s, v20.4s, v16.4s[0]
	fmla	v13.4s, v20.4s, v16.4s[1]
	fmla	v14.4s, v20.4s, v16.4s[2]
	fmla	v15.4s, v20.4s, v16.4s[3]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return
	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nt_8x8_lib4, .-inner_kernel_gemm_add_nt_8x8_lib4
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
	.macro INNER_SCALE_AB_8X8_LIB4
#else
	.align	4
	.type inner_scale_ab_8x8_lib4, %function
inner_scale_ab_8x8_lib4:
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

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
	fmla	v8.4s, v24.4s, v28.4s[0]
	fmla	v9.4s, v25.4s, v28.4s[0]
	fmla	v10.4s, v26.4s, v28.4s[0]
	fmla	v11.4s, v27.4s, v28.4s[0]

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x12], #64
	fmla	v12.4s, v24.4s, v28.4s[0]
	fmla	v13.4s, v25.4s, v28.4s[0]
	fmla	v14.4s, v26.4s, v28.4s[0]
	fmla	v15.4s, v27.4s, v28.4s[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_8x8_lib4, .-inner_scale_ab_8x8_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_8X8_LIB4
#else
	.align 4
	.type inner_store_8x8_lib4, %function
inner_store_8x8_lib4:
#endif

	add		x10, x8, x9

	st1		{v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
	st1		{v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
	st1		{v8.4s, v9.4s, v10.4s, v11.4s}, [x8], #64
	st1		{v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_8x8_lib4, .-inner_store_8x8_lib4
#endif





//                               w0        x1             x2         w3       x4         w5       x6             x7        sp+0     sp+8       sp+16
// void kernel_sgemm_nt_8x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd)

	.align	4
	.global	kernel_sgemm_nt_8x8_lib4
	.type	kernel_sgemm_nt_8x8_lib4, %function
kernel_sgemm_nt_8x8_lib4:
	


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
	mov		w12, w5 // sdb
	lsl		w12, w12, #4 // 16*sdb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X8_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x8_lib4
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x6 // beta
	mov		x10, x7 // C
	ldr		w11, [sp, #(STACKSIZE + 0)] // D
	lsl		w11, w11, #4 // 16*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB4
#else
	bl inner_scale_ab_8x8_lib4
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 8)] // D
	ldr		w9, [sp, #(STACKSIZE + 16)] // sdd
	lsl		w9, w9, #4 // 16*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB4
#else
	bl inner_store_8x8_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret




