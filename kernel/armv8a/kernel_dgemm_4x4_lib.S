/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
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



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x11, x11
	add		x13, x12, x11
	add		x14, x12, x12
	add		x15, x13, x12
	add		x16, x13, x13
	add		x17, x14, x13

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldp		q24, q25, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q28, q29, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q30, q31, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8
	fmov    d12, d8
	fmov    d13, d8
	fmov    d14, d8
	fmov    d15, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x14]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v18.2d, v26.2d[0]
	fmla	v9.2d, v19.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x15]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]
//	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x16]
	fmla	v12.2d, v18.2d, v27.2d[0]
	fmla	v13.2d, v19.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x10, x17]
	fmla	v14.2d, v18.2d, v27.2d[1]
	fmla	v15.2d, v19.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	sub		w8, w8, #4
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v6.2d, v16.2d, v29.2d[1]
	fmla	v7.2d, v17.2d, v29.2d[1]

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v30.2d[0]
	fmla	v9.2d, v19.2d, v30.2d[0]
	ldp		q24, q25, [x10, #(0*8)]
	fmla	v10.2d, v18.2d, v30.2d[1]
	add		x10, x10, x11
	fmla	v11.2d, v19.2d, v30.2d[1]
	ldp		q26, q27, [x10, #(0*8)]
	fmla	v12.2d, v18.2d, v31.2d[0]
	add		x10, x10, x11
	fmla	v13.2d, v19.2d, v31.2d[0]
	ldp		q28, q29, [x10, #(0*8)]
	fmla	v14.2d, v18.2d, v31.2d[1]
	add		x10, x10, x11
	fmla	v15.2d, v19.2d, v31.2d[1]
	ldp		q30, q31, [x10, #(0*8)]
	add		x10, x10, x11

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d
	fadd	v4.2d, v4.2d, v12.2d
	fadd	v5.2d, v5.2d, v13.2d
	fadd	v6.2d, v6.2d, v14.2d
	fadd	v7.2d, v7.2d, v15.2d

//	sub		x9, x9, #32
//	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10, #128]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #192]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	sub		w8, w8, #4
	fmla	v6.2d, v18.2d, v27.2d[1]
	fmla	v7.2d, v19.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
//	cmp		w8, #4
	fmla	v6.2d, v16.2d, v29.2d[1]
	fmla	v7.2d, v17.2d, v29.2d[1]

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v30.2d[1]
//	add		x10, x10, x11
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v31.2d[0]
//	add		x10, x10, x11
	fmla	v5.2d, v19.2d, v31.2d[0]
//	ldp		q28, q29, [x10, #(0*8+2*32)]
	fmla	v6.2d, v18.2d, v31.2d[1]
//	add		x10, x10, x11
	fmla	v7.2d, v19.2d, v31.2d[1]
//	ldp		q30, q31, [x10, #(0*8+3*32)]
//	add		x10, x10, x11

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	cmp		w8, #0
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v24.2d, v25.2d}, [x10]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ld1		{v26.2d, v27.2d}, [x10]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ld1		{v30.2d, v31.2d}, [x10]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]
	fmla	v4.2d, v22.2d, v31.2d[0]
	fmla	v5.2d, v23.2d, v31.2d[0]
	fmla	v6.2d, v22.2d, v31.2d[1]
	fmla	v7.2d, v23.2d, v31.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ld1		{v24.2d, v25.2d}, [x10]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ld1		{v26.2d, v27.2d}, [x10]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ld1		{v30.2d, v31.2d}, [x10]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]
	fmla	v4.2d, v22.2d, v31.2d[0]
	fmla	v5.2d, v23.2d, v31.2d[0]
	fmla	v6.2d, v22.2d, v31.2d[1]
	fmla	v7.2d, v23.2d, v31.2d[1]

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
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	cmp		w8, #0
	fmla	v6.2d, v24.2d, v29.2d[1]
	fmla	v7.2d, v25.2d, v29.2d[1]

	bgt		3b

2: // return



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_4x4_lib4c)
#endif





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
	.macro INNER_KERNEL_GEMM_ADD_NT_4X3_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_4x3_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x11, x11
	add		x13, x12, x11
	add		x14, x12, x12
	add		x15, x13, x12
	add		x16, x13, x13
	add		x17, x14, x13

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldr		q24, [x10, #(0*8)]
	ldr		d25, [x10, #(2*8)]
	add		x10, x10, x11
	ldr		q26, [x10, #(0*8)]
	ldr		d27, [x10, #(2*8)]
	add		x10, x10, x11
	ldr		q28, [x10, #(0*8)]
	ldr		d29, [x10, #(2*8)]
	add		x10, x10, x11
	ldr		q30, [x10, #(0*8)]
	ldr		d31, [x10, #(2*8)]
	add		x10, x10, x11
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8
	fmov    d12, d8
	fmov    d13, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x14]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v18.2d, v26.2d[0]
	fmla	v9.2d, v19.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x15]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]
//	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x16]
	fmla	v12.2d, v18.2d, v27.2d[0]
	fmla	v13.2d, v19.2d, v27.2d[0]
//	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x10, x17]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	sub		w8, w8, #4
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v30.2d[0]
	fmla	v9.2d, v19.2d, v30.2d[0]
	ldr		q24, [x10, #(0*8)]
	ldr		d25, [x10, #(2*8)]
	fmla	v10.2d, v18.2d, v30.2d[1]
	add		x10, x10, x11
	fmla	v11.2d, v19.2d, v30.2d[1]
	ldr		q26, [x10, #(0*8)]
	ldr		d27, [x10, #(2*8)]
	fmla	v12.2d, v18.2d, v31.2d[0]
	add		x10, x10, x11
	fmla	v13.2d, v19.2d, v31.2d[0]
	ldr		q28, [x10, #(0*8)]
	ldr		d29, [x10, #(2*8)]
	add		x10, x10, x11
	ldr		q30, [x10, #(0*8)]
	ldr		d31, [x10, #(2*8)]
	add		x10, x10, x11

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d
	fadd	v4.2d, v4.2d, v12.2d
	fadd	v5.2d, v5.2d, v13.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x10, #128]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #192]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
//	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v30.2d[1]
//	add		x10, x10, x11
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v31.2d[0]
//	add		x10, x10, x11
	fmla	v5.2d, v19.2d, v31.2d[0]
//	ldp		q28, q29, [x10, #(0*8+2*32)]
//	add		x10, x10, x11
//	ldp		q30, q31, [x10, #(0*8+3*32)]
//	add		x10, x10, x11

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		q28, [x10, #(0*8)]
	ldr		d29, [x10, #(2*8)]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	cmp		w8, #0

	bgt		3b

2: // return

	

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		q26, [x10, #0]
	ldr		d27, [x10, #16]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		q28, [x10, #0]
	ldr		d29, [x10, #16]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		q30, [x10, #0]
	ldr		d31, [x10, #16]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]
	fmla	v4.2d, v22.2d, v31.2d[0]
	fmla	v5.2d, v23.2d, v31.2d[0]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		q26, [x10, #0]
	ldr		d27, [x10, #16]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		q28, [x10, #0]
	ldr		d29, [x10, #16]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		q30, [x10, #0]
	ldr		d31, [x10, #16]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]
	fmla	v4.2d, v22.2d, v31.2d[0]
	fmla	v5.2d, v23.2d, v31.2d[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		q28, [x10, #0]
	ldr		d29, [x10, #16]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	cmp		w8, #0

	bgt		3b

2: // return



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_4x3_lib4c)
#endif





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
	.macro INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_4x2_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x11, x11
	add		x13, x12, x11
	add		x14, x12, x12
	add		x15, x13, x12
	add		x16, x13, x13
	add		x17, x14, x13

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldr		q24, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		q26, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		q28, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		q30, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x9, #192]
//	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x14]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v18.2d, v26.2d[0]
	fmla	v9.2d, v19.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x15]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]
//	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x16]
//	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x10, x17]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	sub		w8, w8, #4
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v30.2d[0]
	fmla	v9.2d, v19.2d, v30.2d[0]
	ldr		q24, [x10, #(0*8)]
	fmla	v10.2d, v18.2d, v30.2d[1]
	add		x10, x10, x11
	fmla	v11.2d, v19.2d, v30.2d[1]
	ldr		q26, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		q28, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		q30, [x10, #(0*8)]
	add		x10, x10, x11

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #192]
//	prfm	PLDL1KEEP, [x10, #128]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #192]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
//	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v30.2d[1]
//	add		x10, x10, x11
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]
//	add		x10, x10, x11
//	ldp		q28, q29, [x10, #(0*8+2*32)]
//	add		x10, x10, x11
//	ldp		q30, q31, [x10, #(0*8+3*32)]
//	add		x10, x10, x11

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		q28, [x10, #(0*8)]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return

	

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		q24, [x10, #0]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		q26, [x10, #0]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		q28, [x10, #0]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		q30, [x10, #0]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		q24, [x10, #0]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		q26, [x10, #0]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		q28, [x10, #0]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		q30, [x10, #0]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v22.2d, v30.2d[1]
	fmla	v3.2d, v23.2d, v30.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		q28, [x10, #0]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[1]
	fmla	v3.2d, v25.2d, v28.2d[1]
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_4x2_lib4c)
#endif





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
	.macro INNER_KERNEL_GEMM_ADD_NT_4X1_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_4x1_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x11, x11
	add		x13, x12, x11
	add		x14, x12, x12
	add		x15, x13, x12
	add		x16, x13, x13
	add		x17, x14, x13

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldr		d24, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d26, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d28, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d30, [x10, #(0*8)]
	add		x10, x10, x11
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x9, #192]
//	prfm	PLDL1KEEP, [x10]
	prfm	PLDL1KEEP, [x10, x14]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v18.2d, v26.2d[0]
	fmla	v9.2d, v19.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x10, x11]
	prfm	PLDL1KEEP, [x10, x15]
//	prfm	PLDL1KEEP, [x10, x12]
	prfm	PLDL1KEEP, [x10, x16]
//	prfm	PLDL1KEEP, [x10, x13]
	prfm	PLDL1KEEP, [x10, x17]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
	sub		w8, w8, #4
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v30.2d[0]
	fmla	v9.2d, v19.2d, v30.2d[0]
	ldr		d24, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d26, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d28, [x10, #(0*8)]
	add		x10, x10, x11
	ldr		d30, [x10, #(0*8)]
	add		x10, x10, x11

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
//	prfm	PLDL1KEEP, [x9, #192]
//	prfm	PLDL1KEEP, [x10, #128]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #192]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	add		x9, x9, #128
//	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
//	add		x10, x10, x11
//	ldp		q26, q27, [x10, #(0*8+1*32)]
//	add		x10, x10, x11
//	ldp		q28, q29, [x10, #(0*8+2*32)]
//	add		x10, x10, x11
//	ldp		q30, q31, [x10, #(0*8+3*32)]
//	add		x10, x10, x11

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11
	sub		x10, x10, x11

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		d28, [x10, #(0*8)]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return

	

#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		d24, [x10, #0]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		d26, [x10, #0]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		d28, [x10, #0]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		d30, [x10, #0]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ld1		{v16.2d, v17.2d}, [x9], #32
	ldr		d24, [x10, #0]
	add		x10, x10, x11
	ld1		{v18.2d, v19.2d}, [x9], #32
	ldr		d26, [x10, #0]
	add		x10, x10, x11
	ld1		{v20.2d, v21.2d}, [x9], #32
	ldr		d28, [x10, #0]
	add		x10, x10, x11
	ld1		{v22.2d, v23.2d}, [x9], #32
	ldr		d30, [x10, #0]
	add		x10, x10, x11

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]

	// unroll  1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]

	// unroll 3
	fmla	v0.2d, v22.2d, v30.2d[0]
	fmla	v1.2d, v23.2d, v30.2d[0]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ldr		d28, [x10, #0]
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x10, x10, x11
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_4x1_lib4c)
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



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11
	add		x13, x12, x11
	add		x14, x13, x11

	// prefetch
	prfm	PLDL1KEEP, [x10, #0]
	prfm	PLDL1KEEP, [x12, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x14, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q30, q31, [x14], #32
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
//	prfm	PLDL1KEEP, [x10, #0]
//	prfm	PLDL1KEEP, [x12, #0]
//	prfm	PLDL1KEEP, [x13, #0]
//	prfm	PLDL1KEEP, [x14, #0]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8
	fmov    d12, d8
	fmov    d13, d8
	fmov    d14, d8
	fmov    d15, d8

//	add		x12, x11, #64
//	add		x12, x11, x11
//	add		x13, x12, #64

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	prfm	PLDL1KEEP, [x9, #320]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x10, #32]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]

	// unroll 1
	prfm	PLDL1KEEP, [x12, #32]
	fmla	v8.2d, v18.2d, v24.2d[1]
	fmla	v9.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x13, #32]
	fmla	v12.2d, v18.2d, v28.2d[1]
	fmla	v13.2d, v19.2d, v28.2d[1]
	prfm	PLDL1KEEP, [x14, #32]
	fmla	v14.2d, v18.2d, v30.2d[1]
	fmla	v15.2d, v19.2d, v30.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	sub		w8, w8, #4
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v25.2d[1]
	fmla	v9.2d, v19.2d, v25.2d[1]
	ldp		q24, q25, [x10], #32
	fmla	v10.2d, v18.2d, v27.2d[1]
	fmla	v11.2d, v19.2d, v27.2d[1]
	ldp		q26, q27, [x12], #32
	fmla	v12.2d, v18.2d, v29.2d[1]
	fmla	v13.2d, v19.2d, v29.2d[1]
	ldp		q28, q29, [x13], #32
	fmla	v14.2d, v18.2d, v31.2d[1]
	fmla	v15.2d, v19.2d, v31.2d[1]
	ldp		q30, q31, [x14], #32

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d
	fadd	v4.2d, v4.2d, v12.2d
	fadd	v5.2d, v5.2d, v13.2d
	fadd	v6.2d, v6.2d, v14.2d
	fadd	v7.2d, v7.2d, v15.2d

//	sub		x9, x9, #32
//	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #320]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x10, #256]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #320]
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
//	add		x10, x10, x11
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	sub		w8, w8, #4
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v6.2d, v16.2d, v31.2d[0]
	fmla	v7.2d, v17.2d, v31.2d[0]

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
//	ldp		q28, q29, [x10, #(0*8+2*32)]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
//	ldp		q30, q31, [x10, #(0*8+3*32)]

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
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	ldr		d30, [x13], #8
	ldr		d31, [x14], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	cmp		w8, #0
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]

	bgt		3b

2: // return



#else // cortex a53



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11
	add		x13, x12, x11
	add		x14, x13, x11

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q30, q31, [x14], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v31.2d[0]
	fmla	v7.2d, v21.2d, v31.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]
	fmla	v4.2d, v22.2d, v29.2d[1]
	fmla	v5.2d, v23.2d, v29.2d[1]
	fmla	v6.2d, v22.2d, v31.2d[1]
	fmla	v7.2d, v23.2d, v31.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q30, q31, [x14], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v6.2d, v18.2d, v30.2d[1]
	fmla	v7.2d, v19.2d, v30.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v31.2d[0]
	fmla	v7.2d, v21.2d, v31.2d[0]
//	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]
	fmla	v4.2d, v22.2d, v29.2d[1]
	fmla	v5.2d, v23.2d, v29.2d[1]
	fmla	v6.2d, v22.2d, v31.2d[1]
	fmla	v7.2d, v23.2d, v31.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	ldr		d30, [x13], #8
	ldr		d31, [x14], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	cmp		w8, #0
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]

	bgt		3b

2: // return



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_4x4_lib4c)
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
	.macro INNER_KERNEL_GEMM_ADD_NN_4X3_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_4x3_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11
	add		x13, x12, x11

	// prefetch
	prfm	PLDL1KEEP, [x10, #0]
	prfm	PLDL1KEEP, [x12, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
//	prfm	PLDL1KEEP, [x10, #0]
//	prfm	PLDL1KEEP, [x12, #0]
//	prfm	PLDL1KEEP, [x13, #0]
//	prfm	PLDL1KEEP, [x14, #0]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8
	fmov    d12, d8
	fmov    d13, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	prfm	PLDL1KEEP, [x9, #320]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x10, #32]

	// unroll 1
	prfm	PLDL1KEEP, [x12, #32]
	fmla	v8.2d, v18.2d, v24.2d[1]
	fmla	v9.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x13, #32]
	fmla	v12.2d, v18.2d, v28.2d[1]
	fmla	v13.2d, v19.2d, v28.2d[1]
//	prfm	PLDL1KEEP, [x14, #32]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	sub		w8, w8, #4
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v25.2d[1]
	fmla	v9.2d, v19.2d, v25.2d[1]
	ldp		q24, q25, [x10], #32
	fmla	v10.2d, v18.2d, v27.2d[1]
	fmla	v11.2d, v19.2d, v27.2d[1]
	ldp		q26, q27, [x12], #32
	fmla	v12.2d, v18.2d, v29.2d[1]
	fmla	v13.2d, v19.2d, v29.2d[1]
	ldp		q28, q29, [x13], #32

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d
	fadd	v4.2d, v4.2d, v12.2d
	fadd	v5.2d, v5.2d, v13.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #320]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
//	prfm	PLDL1KEEP, [x10, #256]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #320]
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
//	add		x10, x10, x11
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
//	ldp		q28, q29, [x10, #(0*8+2*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, #32
	sub		x12, x12, #32
	sub		x13, x13, #32

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	ldr		d30, [x13], #8
//	ldr		d31, [x14], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	cmp		w8, #0

	bgt		3b

2: // return



#else // cortex a53



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11
	add		x13, x12, x11

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]
	fmla	v4.2d, v22.2d, v29.2d[1]
	fmla	v5.2d, v23.2d, v29.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q28, q29, [x13], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
//	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]
	fmla	v4.2d, v22.2d, v29.2d[1]
	fmla	v5.2d, v23.2d, v29.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	ldr		d30, [x13], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
	cmp		w8, #0

	bgt		3b

2: // return



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_4x3_lib4c)
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
	.macro INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_4x2_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11

	// prefetch
	prfm	PLDL1KEEP, [x10, #0]
	prfm	PLDL1KEEP, [x12, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
//	prfm	PLDL1KEEP, [x10, #0]
//	prfm	PLDL1KEEP, [x12, #0]
//	prfm	PLDL1KEEP, [x13, #0]
//	prfm	PLDL1KEEP, [x14, #0]
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8
	fmov    d10, d8
	fmov    d11, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	prfm	PLDL1KEEP, [x9, #320]
	prfm	PLDL1KEEP, [x10, #32]

	// unroll 1
	prfm	PLDL1KEEP, [x12, #32]
	fmla	v8.2d, v18.2d, v24.2d[1]
	fmla	v9.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v10.2d, v18.2d, v26.2d[1]
	fmla	v11.2d, v19.2d, v26.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	sub		w8, w8, #4
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v25.2d[1]
	fmla	v9.2d, v19.2d, v25.2d[1]
	ldp		q24, q25, [x10], #32
	fmla	v10.2d, v18.2d, v27.2d[1]
	fmla	v11.2d, v19.2d, v27.2d[1]
	ldp		q26, q27, [x12], #32

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d
	fadd	v2.2d, v2.2d, v10.2d
	fadd	v3.2d, v3.2d, v11.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #256]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
//	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x10, #256]

	// unroll 1
//	prfm	PLDL1KEEP, [x10, #320]
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
//	add		x10, x10, x11
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	fmla	v2.2d, v16.2d, v27.2d[0]
	fmla	v3.2d, v17.2d, v27.2d[0]
	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
//	ldp		q24, q25, [x10, #(0*8+0*32)]
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
//	ldp		q26, q27, [x10, #(0*8+1*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, #32
	sub		x12, x12, #32

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#else // cortex a53



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x12, x10, x11

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q26, q27, [x12], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	fmla	v2.2d, v20.2d, v27.2d[0]
	fmla	v3.2d, v21.2d, v27.2d[0]
//	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]
	fmla	v2.2d, v22.2d, v27.2d[1]
	fmla	v3.2d, v23.2d, v27.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	ldr		d29, [x12], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_4x2_lib4c)
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
	.macro INNER_KERNEL_GEMM_ADD_NN_4X1_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_4x1_lib4c)
#endif



#if defined(TARGET_ARMV8A_ARM_CORTEX_A57)



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x10, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x9, #64]

	// preload
	ldp		q24, q25, [x10], #32
	ldp		q16, q17, [x9, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch
	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #192]

	// zero tmp acc
	fmov	d8, xzr
	fmov    d9, d8

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #128]
	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #192]
	prfm	PLDL1KEEP, [x9, #320]
	prfm	PLDL1KEEP, [x10, #32]

	// unroll 1
	fmla	v8.2d, v18.2d, v24.2d[1]
	fmla	v9.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	sub		w8, w8, #4
	cmp		w8, #4

	// unroll 3
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v8.2d, v18.2d, v25.2d[1]
	fmla	v9.2d, v19.2d, v25.2d[1]
	ldp		q24, q25, [x10], #32

	bgt		1b


	// reduce
	fadd	v0.2d, v0.2d, v8.2d
	fadd	v1.2d, v1.2d, v9.2d

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
//	prfm	PLDL1KEEP, [x9, #256]
//	prfm	PLDL1KEEP, [x9, #320]
//	prfm	PLDL1KEEP, [x10, #256]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q16, q17, [x9, #(0*8+2*32)]
//	add		x10, x10, x11
	sub		w8, w8, #4

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v25.2d[0]
	fmla	v1.2d, v17.2d, v25.2d[0]
	add		x9, x9, #128
	cmp		w8, #4

	// unroll 3
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v0.2d, v18.2d, v25.2d[1]
	fmla	v1.2d, v19.2d, v25.2d[1]
//	ldp		q24, q25, [x10, #(0*8+0*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

	sub		x10, x10, #32

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#else // cortex a53



	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch

	// preload

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// zero tmp acc

	// main loop
1:
	
	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]

	bgt		1b


	// reduce

0:

	cmp		w8, #3
	ble		4f

	// load 0 & 1 & 2 & 3
	ldp		q24, q25, [x10], #32
	ldp		q16, q17, [x9], #32
	ldp		q18, q19, [x9], #32
	ldp		q20, q21, [x9], #32
	ldp		q22, q23, [x9], #32

	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]

	// unroll 1
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	sub		w8, w8, #4

	// unroll 2
	fmla	v0.2d, v20.2d, v25.2d[0]
	fmla	v1.2d, v21.2d, v25.2d[0]
//	cmp		w8, #4

	// unroll 3
	fmla	v0.2d, v22.2d, v25.2d[1]
	fmla	v1.2d, v23.2d, v25.2d[1]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldr		d28, [x10], #8
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	add		x9, x9, #32
	sub		w8, w8, #1
	cmp		w8, #0

	bgt		3b

2: // return



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_4x1_lib4c)
#endif





// subroutine
//
// triangular substitution:
// side = left
// uplo = lower
// tran = not-transposed
// unit diagonal
//
// input arguments:
// x8   <- E
// x9   <- lde
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_LLN_ONE_4X4_LIB
#else
	.align 4
	FUN_START(inner_edge_trsm_lln_one_4x4_lib)
#endif

	ldp		q24, q25, [x8, #0] // E[0+4*0]
	add		x8, x8, x9
	ins		v24.d[0], xzr
	fmls	v0.2d, v24.2d, v0.2d[0]
	fmls	v1.2d, v25.2d, v0.2d[0]
	fmls	v2.2d, v24.2d, v2.2d[0]
	fmls	v3.2d, v25.2d, v2.2d[0]
	fmls	v4.2d, v24.2d, v4.2d[0]
	fmls	v5.2d, v25.2d, v4.2d[0]
	fmls	v6.2d, v24.2d, v6.2d[0]
	fmls	v7.2d, v25.2d, v6.2d[0]

	ldr		q25, [x8, #16] // E[2+4*1]
	add		x8, x8, x9
	fmls	v1.2d, v25.2d, v0.2d[1]
	fmls	v3.2d, v25.2d, v2.2d[1]
	fmls	v5.2d, v25.2d, v4.2d[1]
	fmls	v7.2d, v25.2d, v6.2d[1]

	ldr		q25, [x8, #16] // E[2+4*2]
//	add		x8, x8, x9
	ins		v25.d[0], xzr
	fmls	v1.2d, v25.2d, v1.2d[0]
	fmls	v3.2d, v25.2d, v3.2d[0]
	fmls	v5.2d, v25.2d, v5.2d[0]
	fmls	v7.2d, v25.2d, v7.2d[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_lln_one_4x4_lib)
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
// x9   <- lde
// x10  <- inv_diag_E
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_RLT_INV_4X4_LIB
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_4x4_lib)
#endif
	
	ldr			d16, [x10, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]
	ldr			d16, [x8, #8] // E[1+4*0]
	fmls		v2.2d, v0.2d, v16.2d[0]
	fmls		v3.2d, v1.2d, v16.2d[0]
	ldr			d16, [x8, #16] // E[2+4*0]
	fmls		v4.2d, v0.2d, v16.2d[0]
	fmls		v5.2d, v1.2d, v16.2d[0]
	ldr			d16, [x8, #24] // E[3+4*0]
	fmls		v6.2d, v0.2d, v16.2d[0]
	fmls		v7.2d, v1.2d, v16.2d[0]
	add			x8, x8, x9

	ldr			d16, [x10, #8] // E_inv[1]
	fmul		v2.2d, v2.2d, v16.2d[0]
	fmul		v3.2d, v3.2d, v16.2d[0]
	ldr			d16, [x8, #16] // E[2+4*1]
	fmls		v4.2d, v2.2d, v16.2d[0]
	fmls		v5.2d, v3.2d, v16.2d[0]
	ldr			d16, [x8, #24] // E[3+4*1]
	fmls		v6.2d, v2.2d, v16.2d[0]
	fmls		v7.2d, v3.2d, v16.2d[0]
	add			x8, x8, x9

	ldr			d16, [x10, #16] // E_inv[2]
	fmul		v4.2d, v4.2d, v16.2d[0]
	fmul		v5.2d, v5.2d, v16.2d[0]
	ldr			d16, [x8, #24] // E[3+4*1]
	fmls		v6.2d, v4.2d, v16.2d[0]
	fmls		v7.2d, v5.2d, v16.2d[0]
//	add			x8, x8, x9

	ldr			d16, [x10, #24] // E_inv[2]
	fmul		v6.2d, v6.2d, v16.2d[0]
	fmul		v7.2d, v7.2d, v16.2d[0]
//	add			x8, x8, x9

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_4x4_lib)
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
// w9   <- lde
// x10  <- inv_diag_E
// w11  <- n1
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_TRSM_RLT_INV_4X4_VS_LIB
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_4x4_vs_lib)
#endif
	
	// first column
	ldr			d16, [x10, #0] // E_inv[0]
	fmul		v0.2d, v0.2d, v16.2d[0]
	fmul		v1.2d, v1.2d, v16.2d[0]
	cmp			w11, #2
	blt			0f // return

	// second column
	ldr			d16, [x8, #8] // E[1+4*0]
	fmls		v2.2d, v0.2d, v16.2d[0]
	fmls		v3.2d, v1.2d, v16.2d[0]
	ldr			d16, [x10, #8] // E_inv[1]
	fmul		v2.2d, v2.2d, v16.2d[0]
	fmul		v3.2d, v3.2d, v16.2d[0]
	cmp			w11, #3
	blt			0f // return

	// third column
	add			x12, x8, x9
	ldr			d16, [x8, #16] // E[2+4*0]
	fmls		v4.2d, v0.2d, v16.2d[0]
	fmls		v5.2d, v1.2d, v16.2d[0]
	ldr			d16, [x12, #16] // E[2+4*1]
	fmls		v4.2d, v2.2d, v16.2d[0]
	fmls		v5.2d, v3.2d, v16.2d[0]
	ldr			d16, [x10, #16] // E_inv[2]
	fmul		v4.2d, v4.2d, v16.2d[0]
	fmul		v5.2d, v5.2d, v16.2d[0]
	cmp			w11, #4
	blt			0f // return

	// forth column
	add			x13, x12, x9
	ldr			d16, [x8, #24] // E[3+4*0]
	fmls		v6.2d, v0.2d, v16.2d[0]
	fmls		v7.2d, v1.2d, v16.2d[0]
	ldr			d16, [x12, #24] // E[3+4*1]
	fmls		v6.2d, v2.2d, v16.2d[0]
	fmls		v7.2d, v3.2d, v16.2d[0]
	ldr			d16, [x13, #24] // E[3+4*2]
	fmls		v6.2d, v4.2d, v16.2d[0]
	fmls		v7.2d, v5.2d, v16.2d[0]
	ldr			d16, [x10, #24] // E_inv[3]
	fmul		v6.2d, v6.2d, v16.2d[0]
	fmul		v7.2d, v7.2d, v16.2d[0]

0:
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_TRAN_4X4_LIB
#else
	.align	4
	FUN_START(inner_tran_4x4_lib)
#endif

	trn1	v24.2d, v0.2d, v2.2d
	trn2	v2.2d, v0.2d, v2.2d
	trn1	v25.2d, v5.2d, v7.2d
	trn2	v7.2d, v5.2d, v7.2d
	trn1	v26.2d, v1.2d, v3.2d
	trn2	v27.2d, v1.2d, v3.2d
	trn1	v1.2d, v4.2d, v6.2d
	trn2	v3.2d, v4.2d, v6.2d
	mov		v0.16b, v24.16b
	mov		v5.16b, v25.16b
	mov		v4.16b, v26.16b
	mov		v6.16b, v27.16b

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_tran_4x4_lib)
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

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_4x4_lib)
#endif

	ld1		{v28.2d}, [x8]

	ld1		{v29.2d}, [x9]

	fmul	v0.2d, v0.2d, v28.2d[0]
	fmul	v1.2d, v1.2d, v28.2d[0]
	fmul	v2.2d, v2.2d, v28.2d[0]
	fmul	v3.2d, v3.2d, v28.2d[0]
	fmul	v4.2d, v4.2d, v28.2d[0]
	fmul	v5.2d, v5.2d, v28.2d[0]
	fmul	v6.2d, v6.2d, v28.2d[0]
	fmul	v7.2d, v7.2d, v28.2d[0]

	fcmpe	d29, #0
	beq		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]
	fmla	v2.2d, v26.2d, v29.2d[0]
	fmla	v3.2d, v27.2d, v29.2d[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v26.2d, v29.2d[0]
	fmla	v7.2d, v27.2d, v29.2d[0]

0:

#if MACRO_LEVEL>=1
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

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_4x4_vs_lib)
#endif

	ld1		{v28.2d}, [x8]

	ld1		{v29.2d}, [x9]

	fmul	v0.2d, v0.2d, v28.2d[0]
	fmul	v1.2d, v1.2d, v28.2d[0]
	fmul	v2.2d, v2.2d, v28.2d[0]
	fmul	v3.2d, v3.2d, v28.2d[0]
	fmul	v4.2d, v4.2d, v28.2d[0]
	fmul	v5.2d, v5.2d, v28.2d[0]
	fmul	v6.2d, v6.2d, v28.2d[0]
	fmul	v7.2d, v7.2d, v28.2d[0]

	fcmpe	d29, #0
	beq		0f

	cmp		w12, #4
	blt		1f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v29.2d[0]
	fmla	v7.2d, v25.2d, v29.2d[0]

	b 0f

1:
	cmp		w12, #3
	blt		2f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v29.2d[0]
	fmla	v7.2d, v25.2d, v29.2d[0]

	b 0f

2:
	cmp		w12, #2
	blt		3f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v29.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v29.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v29.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v29.2d[0]

	b 0f

3:
	cmp		w12, #1
	blt		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v29.2d[0]

	cmp		w13, #1
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v29.2d[0]

	cmp		w13, #2
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v29.2d[0]

	cmp		w13, #3
	ble		0f

	ldr		d24, [x10, #0]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v29.2d[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8  <- beta
// x9  <- C
// x10 <- ldc
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_4X4_LIB
#else
	.align	4
	FUN_START(inner_scale_m1b_4x4_lib)
#endif

	ld1		{v29.2d}, [x8]

	fneg	v0.2d, v0.2d
	fneg	v1.2d, v1.2d
	fneg	v2.2d, v2.2d
	fneg	v3.2d, v3.2d

	fneg	v4.2d, v4.2d
	fneg	v5.2d, v5.2d
	fneg	v6.2d, v6.2d
	fneg	v7.2d, v7.2d

	fcmpe	d29, #0
	beq		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	ldp		q26, q27, [x9, #0]
	add		x9, x9, x10
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]
	fmla	v2.2d, v26.2d, v29.2d[0]
	fmla	v3.2d, v27.2d, v29.2d[0]

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	ldp		q26, q27, [x9, #0]
	add		x9, x9, x10
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]
	fmla	v6.2d, v26.2d, v29.2d[0]
	fmla	v7.2d, v27.2d, v29.2d[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_4x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- beta
// x9  <- C
// x19  <- ldc*sizeof(double)
// x11  <- km
// x12  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_4X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_m1b_4x4_vs_lib)
#endif

	ld1		{v29.2d}, [x8]

	fneg	v0.2d, v0.2d
	fneg	v1.2d, v1.2d
	fneg	v2.2d, v2.2d
	fneg	v3.2d, v3.2d

	fneg	v4.2d, v4.2d
	fneg	v5.2d, v5.2d
	fneg	v6.2d, v6.2d
	fneg	v7.2d, v7.2d

	fcmpe	d29, #0
	beq		0f

	cmp		w11, #4
	blt		1f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]

	cmp		w12, #1
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]

	cmp		w12, #2
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]

	cmp		w12, #3
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v6.2d, v24.2d, v29.2d[0]
	fmla	v7.2d, v25.2d, v29.2d[0]

	b 0f

1:
	cmp		w11, #3
	blt		2f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v0.2d, v24.2d, v29.2d[0]
	fmla	v1.2d, v25.2d, v29.2d[0]

	cmp		w12, #1
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]

	cmp		w12, #2
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v4.2d, v24.2d, v29.2d[0]
	fmla	v5.2d, v25.2d, v29.2d[0]

	cmp		w12, #3
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v6.2d, v24.2d, v29.2d[0]
	fmla	v7.2d, v25.2d, v29.2d[0]

	b 0f

2:
	cmp		w11, #2
	blt		3f

	ldr		q24, [x9, #0]
	add		x9, x9, x10
	fmla	v0.2d, v24.2d, v29.2d[0]

	cmp		w12, #1
	ble		0f

	ldr		q24, [x9, #0]
	add		x9, x9, x10
	fmla	v2.2d, v24.2d, v29.2d[0]

	cmp		w12, #2
	ble		0f

	ldr		q24, [x9, #0]
	add		x9, x9, x10
	fmla	v4.2d, v24.2d, v29.2d[0]

	cmp		w12, #3
	ble		0f

	ldr		q24, [x9, #0]
	add		x9, x9, x10
	fmla	v6.2d, v24.2d, v29.2d[0]

	b 0f

3:
	cmp		w11, #1
	blt		0f

	ldr		d24, [x9, #0]
	add		x9, x9, x10
	fmla	v0.2d, v24.2d, v29.2d[0]

	cmp		w12, #1
	ble		0f

	ldr		d24, [x9, #0]
	add		x9, x9, x10
	fmla	v2.2d, v24.2d, v29.2d[0]

	cmp		w12, #2
	ble		0f

	ldr		d24, [x9, #0]
	add		x9, x9, x10
	fmla	v4.2d, v24.2d, v29.2d[0]

	cmp		w12, #3
	ble		0f

	ldr		d24, [x9, #0]
	add		x9, x9, x10
	fmla	v6.2d, v24.2d, v29.2d[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- ldc*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_4X4_LIB
#else
	.align	4
	FUN_START(inner_scale_m11_4x4_lib)
#endif

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	ldp		q26, q27, [x8, #0]
	add		x8, x8, x9
	fsub	v0.2d, v24.2d, v0.2d
	fsub	v1.2d, v25.2d, v1.2d
	fsub	v2.2d, v26.2d, v2.2d
	fsub	v3.2d, v27.2d, v3.2d

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	ldp		q26, q27, [x8, #0]
	add		x8, x8, x9
	fsub	v4.2d, v24.2d, v4.2d
	fsub	v5.2d, v25.2d, v5.2d
	fsub	v6.2d, v26.2d, v6.2d
	fsub	v7.2d, v27.2d, v7.2d

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_4x4_lib)
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- ldc*sizeof(double)
// x10  <- km
// x11  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_4X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_m11_4x4_vs_lib)
#endif

	cmp		w10, #4
	blt		1f

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	fsub	v0.2d, v24.2d, v0.2d
	fsub	v1.2d, v25.2d, v1.2d

	cmp		w11, #1
	ble		0f

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	fsub	v2.2d, v24.2d, v2.2d
	fsub	v3.2d, v25.2d, v3.2d

	cmp		w11, #2
	ble		0f

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	fsub	v4.2d, v24.2d, v4.2d
	fsub	v5.2d, v25.2d, v5.2d

	cmp		w11, #3
	ble		0f

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	fsub	v6.2d, v24.2d, v6.2d
	fsub	v7.2d, v25.2d, v7.2d

	b 0f

1:
	cmp		w10, #3
	blt		2f

	ldr		q24, [x8, #0]
	ldr		d25, [x8, #16]
	add		x8, x8, x9
	fsub	v0.2d, v24.2d, v0.2d
	fsub	v1.2d, v25.2d, v1.2d

	cmp		w11, #1
	ble		0f

	ldr		q24, [x8, #0]
	ldr		d25, [x8, #16]
	add		x8, x8, x9
	fsub	v2.2d, v24.2d, v2.2d
	fsub	v3.2d, v25.2d, v3.2d

	cmp		w11, #2
	ble		0f

	ldr		q24, [x8, #0]
	ldr		d25, [x8, #16]
	add		x8, x8, x9
	fsub	v4.2d, v24.2d, v4.2d
	fsub	v5.2d, v25.2d, v5.2d

	cmp		w11, #3
	ble		0f

	ldr		q24, [x8, #0]
	ldr		d25, [x8, #16]
	add		x8, x8, x9
	fsub	v6.2d, v24.2d, v6.2d
	fsub	v7.2d, v25.2d, v7.2d

	b 0f

2:
	cmp		w10, #2
	blt		3f

	ldr		q24, [x8, #0]
	add		x8, x8, x9
	fsub	v0.2d, v24.2d, v0.2d

	cmp		w11, #1
	ble		0f

	ldr		q24, [x8, #0]
	add		x8, x8, x9
	fsub	v2.2d, v24.2d, v2.2d

	cmp		w11, #2
	ble		0f

	ldr		q24, [x8, #0]
	add		x8, x8, x9
	fsub	v4.2d, v24.2d, v4.2d

	cmp		w11, #3
	ble		0f

	ldr		q24, [x8, #0]
	add		x8, x8, x9
	fsub	v6.2d, v24.2d, v6.2d

	b 0f

3:
	cmp		w10, #1
	blt		0f

	ldr		d24, [x8, #0]
	add		x8, x8, x9
	fsub	v0.2d, v24.2d, v0.2d

	cmp		w11, #1
	ble		0f

	ldr		d24, [x8, #0]
	add		x8, x8, x9
	fsub	v2.2d, v24.2d, v2.2d

	cmp		w11, #2
	ble		0f

	ldr		d24, [x8, #0]
	add		x8, x8, x9
	fsub	v4.2d, v24.2d, v4.2d

	cmp		w11, #3
	ble		0f

	ldr		d24, [x8, #0]
	add		x8, x8, x9
	fsub	v6.2d, v24.2d, v6.2d

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
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

#if MACRO_LEVEL>=1
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





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_LIB
#else
	.align 4
	FUN_START(inner_store_l_4x4_lib)
#endif

	mov		x12, x8

	add		x12, x12, x9
	ldr		q16, [x12, #0]
	add		x12, x12, x9
	add		x12, x12, x9
	ldr		q17, [x12, #16]

	ins		v2.d[0], v16.d[0]
	ins		v7.d[0], v17.d[0]

	stp		q0, q1, [x8, #0]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	add		x8, x8, x9
	str		q5, [x8, #16]
	add		x8, x8, x9
	str		q7, [x8, #16]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_4x4_lib)
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
	.macro INNER_STORE_L_4X4_VS_LIB
#else
	.align 4
	FUN_START(inner_store_l_4x4_vs_lib)
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
	mov		x12, x8

	add		x12, x12, x9
	ldr		q16, [x12, #0]
	add		x12, x12, x9
	add		x12, x12, x9
	ldr		q17, [x12, #16]

	ins		v2.d[0], v16.d[0]
	ins		v7.d[0], v17.d[0]

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
	str		q5, [x8, #16]
	add		x8, x8, x9
	beq		0f
	// 4th col
	str		q7, [x8, #16]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_U_4X4_LIB
#else
	.align 4
	FUN_START(inner_store_u_4x4_lib)
#endif

	str		d0, [x8, #0]
	add		x8, x8, x9
	str		q2, [x8, #0]
	add		x8, x8, x9
	str		q4, [x8, #0]
	str		d5, [x8, #16]
	add		x8, x8, x9
	stp		q6, q7, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_u_4x4_lib)
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
	.macro INNER_STORE_U_4X4_VS_LIB
#else
	.align 4
	FUN_START(inner_store_u_4x4_vs_lib)
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
	str		d0, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	str		q2, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	str		q4, [x8, #0]
	str		d5, [x8, #16]
	add		x8, x8, x9
	beq		0f
	// 4th col
	stp		q6, q7, [x8, #0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_u_4x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_PREFETCH_4X4_LIB
#else
	.align 4
	FUN_START(inner_prefetch_4x4_lib)
#endif

	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]
	add		x8, x8, x9
	prfm	PLDL1KEEP, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_prefetch_4x4_lib)
#endif





//                                 w0        x1             x2         x3         x4            x5         w6       x7         sp+0
// void kernel_dgemm_nt_4x4_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib44cc)
	


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



	// prefetch
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_4X4_LIB
#else
	bl inner_prefetch_4x4_lib
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

	FUN_END(kernel_dgemm_nt_4x4_lib44cc)





//                                    w0        x1             x2         x3         x4            x5         w6       x7         sp+0     sp+8    sp+16
// void kernel_dgemm_nt_4x4_vs_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_lib44cc)
	


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

	FUN_END(kernel_dgemm_nt_4x4_vs_lib44cc)





//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_nt_4x4_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_lib4ccc)
	


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



	// prefetch
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_4X4_LIB
#else
	bl inner_prefetch_4x4_lib
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

	FUN_END(kernel_dgemm_nt_4x4_lib4ccc)






//                                    w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nt_4x4_vs_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_lib4ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B
	mov		w11, w4 // ldb
	lsl		w11, w11, #3 // 8*ldb

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x1_lib4c
#endif
	
	b		103f

100:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x2_lib4c
#endif
	
	b		103f

101:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4c
#endif

103:



	// prefetch
	// TODO prefethc vs
//	ldr		x8, [sp, #(STACKSIZE + 0)] // D
//	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
//	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
//	INNER_PREFETCH_4X4_LIB
#else
//	bl inner_prefetch_4x4_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	bl inner_scale_ab_4x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_4x4_vs_lib4ccc)






//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_nn_4x4_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_4x4_lib4ccc)
	


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



	// prefetch
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_4X4_LIB
#else
	bl inner_prefetch_4x4_lib
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

	FUN_END(kernel_dgemm_nn_4x4_lib4ccc)





//                                    w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nn_4x4_vs_lib4ccc(int kmax, double *alpha, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_4x4_vs_lib4ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		x10, x3 // B
	mov		w11, w4 // ldb
	lsl		w11, w11, #3 // 8*ldb

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x1_lib4c
#endif

	b		103f

100:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x2_lib4c
#endif
	
	b		103f

101:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x4_lib4c
#endif

103:



	// prefetch
	// TODO prefethc vs
//	ldr		x8, [sp, #(STACKSIZE + 0)] // D
//	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
//	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
//	INNER_PREFETCH_4X4_LIB
#else
//	bl inner_prefetch_4x4_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	bl inner_scale_ab_4x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nn_4x4_vs_lib4ccc)





//                                   w0        x1             x2         x3         x4            x5         w6       x7         sp+0
// void kernel_dsyrk_nt_l_4x4_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_4x4_lib44cc)
	


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
	INNER_STORE_L_4X4_LIB
#else
	bl inner_store_l_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_4x4_lib44cc)





//                                      w0        x1             x2         x3         x4            x5         w6       x7         sp+0     sp+8    sp+16
// void kernel_dsyrk_nt_l_4x4_vs_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_4x4_vs_lib44cc)
	


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
	INNER_STORE_L_4X4_VS_LIB
#else
	bl inner_store_l_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_4x4_vs_lib44cc)






//                                   w0        x1             x2         x3         x4            x5         w6       x7         sp+0
// void kernel_dsyrk_nt_u_4x4_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_u_4x4_lib44cc)
	


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
	INNER_STORE_U_4X4_LIB
#else
	bl inner_store_u_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_u_4x4_lib44cc)





//                                      w0        x1             x2         x3         x4            x5         w6       x7         sp+0     sp+8    sp+16
// void kernel_dsyrk_nt_u_4x4_vs_lib44cc(int kmax, double *alpha, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_u_4x4_vs_lib44cc)
	


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
	INNER_STORE_U_4X4_VS_LIB
#else
	bl inner_store_u_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_u_4x4_vs_lib44cc)





//                                         w0        x1         x2         x3         w4       x5         w6       x7         sp+0     sp+8
// void kernel_dtrsm_nt_rl_inv_4x4_lib44ccc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_lib44ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C
	mov		w9, w4 // ldc
	lsl		w9, w9, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_LIB
#else
	bl inner_scale_m11_4x4_lib
#endif



	// solution
	mov		x8, x7 // E
	ldr		w9, [sp, #(STACKSIZE + 0)] // sde
	lsl		w9, w9, #3 // 8*ldc
	ldr		x10, [sp, #(STACKSIZE + 8)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X4_LIB
#else
	bl inner_edge_trsm_rlt_inv_4x4_lib
#endif



	// store l
	mov		x8, x5 // D
	mov		w9, w6 // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_lib44ccc)





//                                            w0        x1         x2         x3         w4       x5         w6       x7         sp+0     sp+8                sp+16   sp+24
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib44ccc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *E, int lde, double *inv_diag_E, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C
	mov		w9, w4 // ldc
	lsl		w9, w9, #3 // 8*ldc
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_VS_LIB
#else
	bl inner_scale_m11_4x4_vs_lib
#endif



	// solution
	mov		x8, x7 // E
	ldr		w9, [sp, #(STACKSIZE + 0)] // sde
	lsl		w9, w9, #3 // 8*ldc
	ldr		x10, [sp, #(STACKSIZE + 8)] // inv_diag_E
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X4_VS_LIB
#else
	bl inner_edge_trsm_rlt_inv_4x4_vs_lib
#endif



	// store l
	mov		x8, x5 // D
	mov		w9, w6 // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nt_rl_inv_4x4_vs_lib44ccc)





//                                         w0        x1         x2         x3            x4         w5       x6         w7       sp+0
// void kernel_dtrsm_nt_rl_one_4x4_lib44cc4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_lib44cc4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0
	mov		x8, x3 // beta
	mov		x9, x4 // C
	mov		w10, w5 // ldc
	lsl		w10, w10, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	bl inner_scale_m1b_4x4_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 0)] // E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_ONE_4X4_LIB4
#else
	bl inner_edge_trsm_rlt_one_4x4_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_lib44cc4)





//                                            w0        x1         x2         x3            x4         w5       x6         w7       sp+0       sp+8    sp+16
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib44cc4(int kmax, double *A, double *B, double *beta, double *C, int ldc, double *D, int ldd, double *E, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nt_rl_one_4x4_vs_lib44cc4)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0
	mov		x8, x3 // beta
	mov		x9, x4 // C
	mov		w10, w5 // ldc
	lsl		w10, w10, #3 // 8*ldc
	ldr		w11, [sp, #(STACKSIZE + 8)] // m1
	ldr		w12, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	bl inner_scale_m1b_4x4_vs_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 0)] // E
	ldr		w9, [sp, #(STACKSIZE + 16)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_ONE_4X4_VS_LIB4
#else
	bl inner_edge_trsm_rlt_one_4x4_vs_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // ldd
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

	FUN_END(kernel_dtrsm_nt_rl_one_4x4_vs_lib44cc4)





//                                    w0        x1         x2         x3         w4       x5         w6       x7
// void kernel_dpotrf_nt_l_4x4_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D)

	.align	4
	GLOB_FUN_START(kernel_dpotrf_nt_l_4x4_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C
	mov		w9, w4 // ldc
	lsl		w9, w9, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_LIB
#else
	bl inner_scale_m11_4x4_lib
#endif



	// factorization
	mov		x8, x7 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_4X4_LIB4
#else
	bl inner_edge_potrf_4x4_lib4
#endif



	// store l
	mov		x8, x5 // D
	mov		w9, w6 // ldd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB
#else
	bl inner_store_l_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dpotrf_nt_l_4x4_lib44cc)





//                                       w0        x1         x2         x3         w4       x5         w6       x7                  sp+0    sp+1
// void kernel_dpotrf_nt_l_4x4_vs_lib44cc(int kmax, double *A, double *B, double *C, int ldc, double *D, int ldd, double *inv_diag_D, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dpotrf_nt_l_4x4_vs_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C
	mov		w9, w4 // ldc
	lsl		w9, w9, #3 // 8*ldc
	ldr		w10, [sp, #(STACKSIZE + 0)] // m1
	ldr		w11, [sp, #(STACKSIZE + 8)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_4X4_VS_LIB
#else
	bl inner_scale_m11_4x4_vs_lib
#endif



	// factorization
	mov		x8, x7 // inv_diag_E
	ldr		w9, [sp, #(STACKSIZE + 8)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_4X4_VS_LIB4
#else
	bl inner_edge_potrf_4x4_vs_lib4
#endif



	// store l
	mov		x8, x5 // D
	mov		w9, w6 // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 0)] // m1
	ldr		w11, [sp, #(STACKSIZE + 8)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB
#else
	bl inner_store_l_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dpotrf_nt_l_4x4_vs_lib44cc)





//                                         w0        x1         x2         x3       x4            x5         w6       x7         sp+0     sp+8       sp+16
// void kernel_dtrsm_nn_ll_one_4x4_lib4cccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nn_ll_one_4x4_lib4cccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B
	mov		w11, w3 // ldb
	lsl		w11, w11, #3 // 8*ldb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x4_lib4c
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // ldc
	lsl		w10, w10, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_LIB
#else
	bl inner_scale_m1b_4x4_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		w9, [sp, #(STACKSIZE + 16)] // sde
	lsl		w9, w9, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_LLN_ONE_4X4_LIB
#else
	bl inner_edge_trsm_lln_one_4x4_lib
#endif



	// store l
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // sdd
	lsl		w9, w9, #3 // 8*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB
#else
	bl inner_store_4x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nn_ll_one_4x4_lib4cccc)






//                                            w0        x1         x2         x3       x4            x5         w6       x7         sp+0     sp+8       sp+16    sp+24   sp+32
// void kernel_dtrsm_nn_ll_one_4x4_vs_lib4cccc(int kmax, double *A, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, double *E, int lde, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dtrsm_nn_ll_one_4x4_vs_lib4cccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B
	mov		w11, w3 // ldb
	lsl		w11, w11, #3 // 8*ldb

	ldr		w12, [sp, #(STACKSIZE + 32)] // n1
	cmp		w12, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x1_lib4c
#endif

	b		103f

100:

	ldr		w12, [sp, #(STACKSIZE + 32)] // n1
	cmp		w12, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x2_lib4c
#endif
	
	b		103f

101:

	ldr		w12, [sp, #(STACKSIZE + 32)] // n1
	cmp		w12, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x4_lib4c
#endif

103:



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // ldc
	lsl		w10, w10, #3 // 8*ldc
	ldr		w11, [sp, #(STACKSIZE + 24)] // m1
	ldr		w12, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_4X4_VS_LIB
#else
	bl inner_scale_m1b_4x4_vs_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		w9, [sp, #(STACKSIZE + 16)] // sde
	lsl		w9, w9, #3 // 8*ldc

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_LLN_ONE_4X4_LIB
#else
	bl inner_edge_trsm_lln_one_4x4_lib
#endif



	// store l
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // sdd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 24)] // m1
	ldr		w11, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dtrsm_nn_ll_one_4x4_vs_lib4cccc)





//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_nt_4x4_libc4cc(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_libc4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x4 // B
	mov		x10, x2 // A
	mov		w11, w3 // lda
	lsl		w11, w11, #3 // 8*lda

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
	INNER_TRAN_4X4_LIB
#else
	bl inner_tran_4x4_lib
#endif


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

	FUN_END(kernel_dgemm_nt_4x4_libc4cc)





//                                    w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nt_4x4_vs_libc4cc(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_4x4_vs_libc4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x4 // B
	mov		x10, x2 // A
	mov		w11, w3 // lda
	lsl		w11, w11, #3 // 8*ldb

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x1_lib4c
#endif
	
	b		103f

100:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x2_lib4c
#endif
	
	b		103f

101:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_4x4_lib4c
#endif

103:



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_TRAN_4X4_LIB
#else
	bl inner_tran_4x4_lib
#endif


#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	bl inner_scale_ab_4x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_4x4_vs_libc4cc)





//                                 w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8
// void kernel_dgemm_tt_4x4_libc4cc(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_tt_4x4_libc4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x4 // B
	mov		x10, x2 // A
	mov		w11, w3 // lda
	lsl		w11, w11, #3 // 8*lda

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
	INNER_TRAN_4X4_LIB
#else
	bl inner_tran_4x4_lib
#endif


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

	FUN_END(kernel_dgemm_tt_4x4_libc4cc)





//                                    w0        x1             x2         x3         x4       x5            w6         x7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_tt_4x4_vs_libc4cc(int kmax, double *alpha, double *A, int lda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_tt_4x4_vs_libc4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x4 // B
	mov		x10, x2 // A
	mov		w11, w3 // lda
	lsl		w11, w11, #3 // 8*ldb

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x1_lib4c
#endif
	
	b		103f

100:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x2_lib4c
#endif
	
	b		103f

101:

	ldr		w12, [sp, #(STACKSIZE + 24)] // n1
	cmp		w12, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_4X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_4x4_lib4c
#endif

103:



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*ldc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_TRAN_4X4_LIB
#else
	bl inner_tran_4x4_lib
#endif


#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_VS_LIB
#else
	bl inner_scale_ab_4x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*ldd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB
#else
	bl inner_store_4x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_tt_4x4_vs_libc4cc)







