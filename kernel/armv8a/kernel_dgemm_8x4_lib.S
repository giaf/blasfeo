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
// x10  <- sda
// x11  <- B
// x12  <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_8x4_lib4c)
#endif

#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO cortex a53 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	add		x14, x12, x12
	add		x15, x14, x12

	// prefetch
	prfm	PLDL1KEEP, [x11]
	prfm	PLDL1KEEP, [x11, x12]
	prfm	PLDL1KEEP, [x11, x14]
	prfm	PLDL1KEEP, [x11, x15]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
	ldp		q24, q25, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q26, q27, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q28, q29, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q30, q31, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q16, q17, [x9, #(0*8+0*32)]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v12.2d, v20.2d, v25.2d[0]
	fmla	v13.2d, v21.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x11, x12]
	fmla	v14.2d, v20.2d, v25.2d[1]
	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x11, x14]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x11, x15]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v12.2d, v22.2d, v27.2d[0]
	fmla	v13.2d, v23.2d, v27.2d[0]
	fmla	v6.2d, v18.2d, v27.2d[1]
	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
	fmla	v14.2d, v22.2d, v27.2d[1]
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
	cmp		w8, #4
	fmla	v6.2d, v16.2d, v29.2d[1]
	fmla	v7.2d, v17.2d, v29.2d[1]
	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v14.2d, v20.2d, v29.2d[1]
	fmla	v15.2d, v21.2d, v29.2d[1]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v9.2d, v23.2d, v30.2d[0]
	ldp		q24, q25, [x11, #(0*8)]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v3.2d, v19.2d, v30.2d[1]
	add		x11, x11, x12
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
	ldp		q26, q27, [x11, #(0*8)]
	fmla	v4.2d, v18.2d, v31.2d[0]
	fmla	v5.2d, v19.2d, v31.2d[0]
	add		x11, x11, x12
	fmla	v12.2d, v22.2d, v31.2d[0]
	fmla	v13.2d, v23.2d, v31.2d[0]
	ldp		q28, q29, [x11, #(0*8)]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
	add		x11, x11, x12
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]
	ldp		q30, q31, [x11, #(0*8)]
	add		x11, x11, x12

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x13, #128]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v12.2d, v20.2d, v25.2d[0]
	fmla	v13.2d, v21.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x13, #192]
	fmla	v6.2d, v16.2d, v25.2d[1]
	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #128]
	fmla	v14.2d, v20.2d, v25.2d[1]
	fmla	v15.2d, v21.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #192]

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v12.2d, v22.2d, v27.2d[0]
	fmla	v13.2d, v23.2d, v27.2d[0]
//	add		x11, x11, #128
	fmla	v6.2d, v18.2d, v27.2d[1]
	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
	fmla	v14.2d, v22.2d, v27.2d[1]
	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v16.2d, v29.2d[1]
	fmla	v7.2d, v17.2d, v29.2d[1]
//	ldp		q16, q17, [x9, #(0*8+0*32)]
	fmla	v14.2d, v20.2d, v29.2d[1]
	fmla	v15.2d, v21.2d, v29.2d[1]
//	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q24, q25, [x11, #(0*8+0*32)]
	fmla	v9.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v31.2d[0]
	fmla	v5.2d, v19.2d, v31.2d[0]
	fmla	v12.2d, v22.2d, v31.2d[0]
	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8+2*32)]
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8+3*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x13, x13, #32
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11]
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	add		x11, x11, x12
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
	fmla	v6.2d, v20.2d, v29.2d[1]
	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x13], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	sub		w8, w8, #1
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	cmp		w8, #0
	fmla	v12.2d, v22.2d, v29.2d[0]
	fmla	v13.2d, v23.2d, v29.2d[0]
	fmla	v14.2d, v22.2d, v29.2d[1]
	fmla	v15.2d, v23.2d, v29.2d[1]

	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// TODO



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_8x4_lib4c)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11  <- B
// x12  <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_8X3_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_8x3_lib4c)
#endif

#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO cortex a53 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	add		x14, x12, x12
	add		x15, x14, x12

	// prefetch
	prfm	PLDL1KEEP, [x11]
	prfm	PLDL1KEEP, [x11, x12]
	prfm	PLDL1KEEP, [x11, x14]
	prfm	PLDL1KEEP, [x11, x15]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
	ldp		q24, q25, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q26, q27, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q28, q29, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q30, q31, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q16, q17, [x9, #(0*8+0*32)]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v12.2d, v20.2d, v25.2d[0]
	fmla	v13.2d, v21.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x11, x12]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x11, x14]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x11, x15]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v12.2d, v22.2d, v27.2d[0]
	fmla	v13.2d, v23.2d, v27.2d[0]
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
	cmp		w8, #4
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v9.2d, v23.2d, v30.2d[0]
	ldp		q24, q25, [x11, #(0*8)]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v3.2d, v19.2d, v30.2d[1]
	add		x11, x11, x12
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
	ldp		q26, q27, [x11, #(0*8)]
	fmla	v4.2d, v18.2d, v31.2d[0]
	fmla	v5.2d, v19.2d, v31.2d[0]
	add		x11, x11, x12
	fmla	v12.2d, v22.2d, v31.2d[0]
	fmla	v13.2d, v23.2d, v31.2d[0]
	ldp		q28, q29, [x11, #(0*8)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
	add		x11, x11, x12
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
	ldp		q30, q31, [x11, #(0*8)]
	add		x11, x11, x12

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x13, #128]
	fmla	v4.2d, v16.2d, v25.2d[0]
	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
	fmla	v12.2d, v20.2d, v25.2d[0]
	fmla	v13.2d, v21.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x13, #192]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #128]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #192]

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v27.2d[0]
	fmla	v5.2d, v19.2d, v27.2d[0]
	fmla	v12.2d, v22.2d, v27.2d[0]
	fmla	v13.2d, v23.2d, v27.2d[0]
//	add		x11, x11, #128
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
	fmla	v4.2d, v16.2d, v29.2d[0]
	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
	fmla	v12.2d, v20.2d, v29.2d[0]
	fmla	v13.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
//	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
//	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q24, q25, [x11, #(0*8+0*32)]
	fmla	v9.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8+1*32)]
	fmla	v4.2d, v18.2d, v31.2d[0]
	fmla	v5.2d, v19.2d, v31.2d[0]
	fmla	v12.2d, v22.2d, v31.2d[0]
	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8+2*32)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8+3*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x13, x13, #32
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x11]
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	add		x11, x11, x12
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
	fmla	v4.2d, v20.2d, v29.2d[0]
	fmla	v5.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v20.2d, v29.2d[1]
//	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x13], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	sub		w8, w8, #1
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	cmp		w8, #0
	fmla	v12.2d, v22.2d, v29.2d[0]
	fmla	v13.2d, v23.2d, v29.2d[0]
//	fmla	v14.2d, v22.2d, v29.2d[1]
//	fmla	v15.2d, v23.2d, v29.2d[1]

	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// TODO



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_8x3_lib4c)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11  <- B
// x12  <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_8X2_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_8x2_lib4c)
#endif

#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO cortex a53 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	add		x14, x12, x12
	add		x15, x14, x12

	// prefetch
	prfm	PLDL1KEEP, [x11]
	prfm	PLDL1KEEP, [x11, x12]
	prfm	PLDL1KEEP, [x11, x14]
	prfm	PLDL1KEEP, [x11, x15]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
//	ldp		q24, q25, [x11, #(0*8)]
	ldr		q24, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q26, q27, [x11, #(0*8)]
	ldr		q26, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q28, q29, [x11, #(0*8)]
	ldr		q28, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q30, q31, [x11, #(0*8)]
	ldr		q30, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q16, q17, [x9, #(0*8+0*32)]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
//	fmla	v4.2d, v16.2d, v25.2d[0]
//	fmla	v5.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
//	fmla	v12.2d, v20.2d, v25.2d[0]
//	fmla	v13.2d, v21.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x11, x12]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x11, x14]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x11, x15]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
//	fmla	v4.2d, v18.2d, v27.2d[0]
//	fmla	v5.2d, v19.2d, v27.2d[0]
//	fmla	v12.2d, v22.2d, v27.2d[0]
//	fmla	v13.2d, v23.2d, v27.2d[0]
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
//	fmla	v4.2d, v16.2d, v29.2d[0]
//	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
//	fmla	v12.2d, v20.2d, v29.2d[0]
//	fmla	v13.2d, v21.2d, v29.2d[0]
	cmp		w8, #4
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v9.2d, v23.2d, v30.2d[0]
//	ldp		q24, q25, [x11, #(0*8)]
	ldr		q24, [x11, #(0*8)]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v3.2d, v19.2d, v30.2d[1]
	add		x11, x11, x12
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8)]
	ldr		q26, [x11, #(0*8)]
//	fmla	v4.2d, v18.2d, v31.2d[0]
//	fmla	v5.2d, v19.2d, v31.2d[0]
	add		x11, x11, x12
//	fmla	v12.2d, v22.2d, v31.2d[0]
//	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8)]
	ldr		q28, [x11, #(0*8)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
	add		x11, x11, x12
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8)]
	ldr		q30, [x11, #(0*8)]
	add		x11, x11, x12

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v2.2d, v16.2d, v24.2d[1]
	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
	fmla	v10.2d, v20.2d, v24.2d[1]
	fmla	v11.2d, v21.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x13, #128]
//	fmla	v4.2d, v16.2d, v25.2d[0]
//	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
//	fmla	v12.2d, v20.2d, v25.2d[0]
//	fmla	v13.2d, v21.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x13, #192]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #128]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #192]

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
//	fmla	v4.2d, v18.2d, v27.2d[0]
//	fmla	v5.2d, v19.2d, v27.2d[0]
//	fmla	v12.2d, v22.2d, v27.2d[0]
//	fmla	v13.2d, v23.2d, v27.2d[0]
//	add		x11, x11, #128
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v2.2d, v16.2d, v28.2d[1]
	fmla	v3.2d, v17.2d, v28.2d[1]
	fmla	v10.2d, v20.2d, v28.2d[1]
	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
//	fmla	v4.2d, v16.2d, v29.2d[0]
//	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
//	fmla	v12.2d, v20.2d, v29.2d[0]
//	fmla	v13.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
//	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
//	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q24, q25, [x11, #(0*8+0*32)]
	fmla	v9.2d, v23.2d, v30.2d[0]
	fmla	v2.2d, v18.2d, v30.2d[1]
	fmla	v10.2d, v22.2d, v30.2d[1]
	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8+1*32)]
//	fmla	v4.2d, v18.2d, v31.2d[0]
//	fmla	v5.2d, v19.2d, v31.2d[0]
//	fmla	v12.2d, v22.2d, v31.2d[0]
//	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8+2*32)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8+3*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x13, x13, #32
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
//	ld1		{v28.2d, v29.2d}, [x11]
	ld1		{v28.2d}, [x11]
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	add		x11, x11, x12
	fmla	v2.2d, v20.2d, v28.2d[1]
	fmla	v3.2d, v21.2d, v28.2d[1]
//	fmla	v4.2d, v20.2d, v29.2d[0]
//	fmla	v5.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v20.2d, v29.2d[1]
//	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x13], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	sub		w8, w8, #1
	fmla	v10.2d, v22.2d, v28.2d[1]
	fmla	v11.2d, v23.2d, v28.2d[1]
	cmp		w8, #0
//	fmla	v12.2d, v22.2d, v29.2d[0]
//	fmla	v13.2d, v23.2d, v29.2d[0]
//	fmla	v14.2d, v22.2d, v29.2d[1]
//	fmla	v15.2d, v23.2d, v29.2d[1]

	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// TODO



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_8x2_lib4c)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11  <- B
// x12  <- ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_8X1_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nt_8x1_lib4c)
#endif

#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO cortex a53 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	add		x14, x12, x12
	add		x15, x14, x12

	// prefetch
	prfm	PLDL1KEEP, [x11]
	prfm	PLDL1KEEP, [x11, x12]
	prfm	PLDL1KEEP, [x11, x14]
	prfm	PLDL1KEEP, [x11, x15]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
//	ldp		q24, q25, [x11, #(0*8)]
	ldr		q24, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q26, q27, [x11, #(0*8)]
	ldr		q26, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q28, q29, [x11, #(0*8)]
	ldr		q28, [x11, #(0*8)]
	add		x11, x11, x12
//	ldp		q30, q31, [x11, #(0*8)]
	ldr		q30, [x11, #(0*8)]
	add		x11, x11, x12
	ldp		q16, q17, [x9, #(0*8+0*32)]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// prefetch

	// main loop
1:
	
	// unroll 0
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
//	fmla	v2.2d, v16.2d, v24.2d[1]
//	fmla	v3.2d, v17.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x13, #128]
//	fmla	v10.2d, v20.2d, v24.2d[1]
//	fmla	v11.2d, v21.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x9, #192]
//	fmla	v4.2d, v16.2d, v25.2d[0]
//	fmla	v5.2d, v17.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
//	fmla	v12.2d, v20.2d, v25.2d[0]
//	fmla	v13.2d, v21.2d, v25.2d[0]
	prfm	PLDL1KEEP, [x11]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
	prfm	PLDL1KEEP, [x11, x12]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]

	// unroll 1
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x11, x14]
//	fmla	v2.2d, v18.2d, v26.2d[1]
//	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x11, x15]
//	fmla	v10.2d, v22.2d, v26.2d[1]
//	fmla	v11.2d, v23.2d, v26.2d[1]
//	fmla	v4.2d, v18.2d, v27.2d[0]
//	fmla	v5.2d, v19.2d, v27.2d[0]
//	fmla	v12.2d, v22.2d, v27.2d[0]
//	fmla	v13.2d, v23.2d, v27.2d[0]
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
//	fmla	v2.2d, v16.2d, v28.2d[1]
//	fmla	v3.2d, v17.2d, v28.2d[1]
//	fmla	v10.2d, v20.2d, v28.2d[1]
//	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
//	fmla	v4.2d, v16.2d, v29.2d[0]
//	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
//	fmla	v12.2d, v20.2d, v29.2d[0]
//	fmla	v13.2d, v21.2d, v29.2d[0]
	cmp		w8, #4
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v9.2d, v23.2d, v30.2d[0]
//	ldp		q24, q25, [x11, #(0*8)]
	ldr		q24, [x11, #(0*8)]
//	fmla	v2.2d, v18.2d, v30.2d[1]
//	fmla	v3.2d, v19.2d, v30.2d[1]
	add		x11, x11, x12
//	fmla	v10.2d, v22.2d, v30.2d[1]
//	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8)]
	ldr		q26, [x11, #(0*8)]
//	fmla	v4.2d, v18.2d, v31.2d[0]
//	fmla	v5.2d, v19.2d, v31.2d[0]
	add		x11, x11, x12
//	fmla	v12.2d, v22.2d, v31.2d[0]
//	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8)]
	ldr		q28, [x11, #(0*8)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
	add		x11, x11, x12
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8)]
	ldr		q30, [x11, #(0*8)]
	add		x11, x11, x12

	bgt		1b

0:

	cmp		w8, #3
	ble		4f

	
	// unroll 0
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q18, q19, [x9, #(0*8+1*32)]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	ldp		q22, q23, [x13, #(0*8+1*32)]
//	fmla	v2.2d, v16.2d, v24.2d[1]
//	fmla	v3.2d, v17.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x9, #128]
//	fmla	v10.2d, v20.2d, v24.2d[1]
//	fmla	v11.2d, v21.2d, v24.2d[1]
//	prfm	PLDL1KEEP, [x13, #128]
//	fmla	v4.2d, v16.2d, v25.2d[0]
//	fmla	v5.2d, v17.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x9, #192]
//	fmla	v12.2d, v20.2d, v25.2d[0]
//	fmla	v13.2d, v21.2d, v25.2d[0]
//	prfm	PLDL1KEEP, [x13, #192]
//	fmla	v6.2d, v16.2d, v25.2d[1]
//	fmla	v7.2d, v17.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #128]
//	fmla	v14.2d, v20.2d, v25.2d[1]
//	fmla	v15.2d, v21.2d, v25.2d[1]
//	prfm	PLDL1KEEP, [x11, #192]

	// unroll 1
	fmla	v0.2d, v18.2d, v26.2d[0]
	fmla	v1.2d, v19.2d, v26.2d[0]
	ldp		q16, q17, [x9, #(0*8+2*32)]
	fmla	v8.2d, v22.2d, v26.2d[0]
	fmla	v9.2d, v23.2d, v26.2d[0]
	ldp		q20, q21, [x13, #(0*8+2*32)]
//	fmla	v2.2d, v18.2d, v26.2d[1]
//	fmla	v3.2d, v19.2d, v26.2d[1]
//	fmla	v10.2d, v22.2d, v26.2d[1]
//	fmla	v11.2d, v23.2d, v26.2d[1]
//	fmla	v4.2d, v18.2d, v27.2d[0]
//	fmla	v5.2d, v19.2d, v27.2d[0]
//	fmla	v12.2d, v22.2d, v27.2d[0]
//	fmla	v13.2d, v23.2d, v27.2d[0]
//	add		x11, x11, #128
//	fmla	v6.2d, v18.2d, v27.2d[1]
//	fmla	v7.2d, v19.2d, v27.2d[1]
	sub		w8, w8, #4
//	fmla	v14.2d, v22.2d, v27.2d[1]
//	fmla	v15.2d, v23.2d, v27.2d[1]

	// unroll 2
	fmla	v0.2d, v16.2d, v28.2d[0]
	fmla	v1.2d, v17.2d, v28.2d[0]
	ldp		q18, q19, [x9, #(0*8+3*32)]
	fmla	v8.2d, v20.2d, v28.2d[0]
	fmla	v9.2d, v21.2d, v28.2d[0]
	ldp		q22, q23, [x13, #(0*8+3*32)]
//	fmla	v2.2d, v16.2d, v28.2d[1]
//	fmla	v3.2d, v17.2d, v28.2d[1]
//	fmla	v10.2d, v20.2d, v28.2d[1]
//	fmla	v11.2d, v21.2d, v28.2d[1]
	add		x9, x9, #128
//	fmla	v4.2d, v16.2d, v29.2d[0]
//	fmla	v5.2d, v17.2d, v29.2d[0]
	add		x13, x13, #128
//	fmla	v12.2d, v20.2d, v29.2d[0]
//	fmla	v13.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v16.2d, v29.2d[1]
//	fmla	v7.2d, v17.2d, v29.2d[1]
//	ldp		q16, q17, [x9, #(0*8+0*32)]
//	fmla	v14.2d, v20.2d, v29.2d[1]
//	fmla	v15.2d, v21.2d, v29.2d[1]
//	ldp		q20, q21, [x13, #(0*8+0*32)]

	// unroll 3
	fmla	v0.2d, v18.2d, v30.2d[0]
	fmla	v1.2d, v19.2d, v30.2d[0]
	fmla	v8.2d, v22.2d, v30.2d[0]
	fmla	v3.2d, v19.2d, v30.2d[1]
//	ldp		q24, q25, [x11, #(0*8+0*32)]
//	fmla	v9.2d, v23.2d, v30.2d[0]
//	fmla	v2.2d, v18.2d, v30.2d[1]
//	fmla	v10.2d, v22.2d, v30.2d[1]
//	fmla	v11.2d, v23.2d, v30.2d[1]
//	ldp		q26, q27, [x11, #(0*8+1*32)]
//	fmla	v4.2d, v18.2d, v31.2d[0]
//	fmla	v5.2d, v19.2d, v31.2d[0]
//	fmla	v12.2d, v22.2d, v31.2d[0]
//	fmla	v13.2d, v23.2d, v31.2d[0]
//	ldp		q28, q29, [x11, #(0*8+2*32)]
//	fmla	v6.2d, v18.2d, v31.2d[1]
//	fmla	v7.2d, v19.2d, v31.2d[1]
//	fmla	v14.2d, v22.2d, v31.2d[1]
//	fmla	v15.2d, v23.2d, v31.2d[1]
//	ldp		q30, q31, [x11, #(0*8+3*32)]

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

//	sub		x9, x9, #32
//	sub		x11, x11, #32
//	sub		x13, x13, #32
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12
	sub		x11, x11, x12

3: // clean1-up loop

	// unroll 0
	ld1		{v20.2d, v21.2d}, [x9], #32
//	ld1		{v28.2d, v29.2d}, [x11]
	ld1		{v28.2d}, [x11]
	fmla	v0.2d, v20.2d, v28.2d[0]
	fmla	v1.2d, v21.2d, v28.2d[0]
	add		x11, x11, x12
//	fmla	v2.2d, v20.2d, v28.2d[1]
//	fmla	v3.2d, v21.2d, v28.2d[1]
//	fmla	v4.2d, v20.2d, v29.2d[0]
//	fmla	v5.2d, v21.2d, v29.2d[0]
//	fmla	v6.2d, v20.2d, v29.2d[1]
//	fmla	v7.2d, v21.2d, v29.2d[1]
	ld1		{v22.2d, v23.2d}, [x13], #32
	fmla	v8.2d, v22.2d, v28.2d[0]
	fmla	v9.2d, v23.2d, v28.2d[0]
	sub		w8, w8, #1
//	fmla	v10.2d, v22.2d, v28.2d[1]
//	fmla	v11.2d, v23.2d, v28.2d[1]
	cmp		w8, #0
//	fmla	v12.2d, v22.2d, v29.2d[0]
//	fmla	v13.2d, v23.2d, v29.2d[0]
//	fmla	v14.2d, v22.2d, v29.2d[1]
//	fmla	v15.2d, v23.2d, v29.2d[1]

	bgt		3b

2: // return



#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)



	// TODO



#endif // cortex a53



#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nt_8x1_lib4c)
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10  <- sda
// x11   <- B
// x12   <- 8*ldb
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NN_8X4_LIB4C
#else
	.align	4
	FUN_START(inner_kernel_gemm_add_nn_8x4_lib4c)
#endif



#if 1 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) // TODO cortex A53 !!!!!!!!!!!!!!!!!!!!!!!!!



	// early return
	cmp		w8, #0
	ble		2f // return

	add		x13, x9, x10

	add		x14, x11, x12
	add		x15, x14, x12
	add		x16, x15, x12

	// prefetch
	prfm	PLDL1KEEP, [x11, #0]
	prfm	PLDL1KEEP, [x14, #0]
	prfm	PLDL1KEEP, [x15, #0]
	prfm	PLDL1KEEP, [x16, #0]
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x13, #0]
	prfm	PLDL1KEEP, [x9, #64]
	prfm	PLDL1KEEP, [x13, #64]

	// preload
	ldp		q24, q25, [x11], #32
	ldp		q26, q27, [x14], #32
	ldp		q28, q29, [x15], #32
	ldp		q30, q31, [x16], #32
	ldp		q16, q17, [x9, #0]
	ldp		q20, q21, [x13, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop


	// prefetch
//	add		x14, x12, #64

//	prfm	PLDL1KEEP, [x11, x12]
//	prfm	PLDL1KEEP, [x11, x14]


	// main loop
1:
	
	ldp		q18, q19, [x9, #32]
	fmla	v0.2d, v16.2d, v24.2d[0]
	fmla	v1.2d, v17.2d, v24.2d[0]
	ldp		q22, q23, [x13, #32]
	fmla	v8.2d, v20.2d, v24.2d[0]
	fmla	v9.2d, v21.2d, v24.2d[0]
	prfm	PLDL1KEEP, [x9, #128]
	fmla	v2.2d, v16.2d, v26.2d[0]
	fmla	v3.2d, v17.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x13, #128]
	fmla	v10.2d, v20.2d, v26.2d[0]
	fmla	v11.2d, v21.2d, v26.2d[0]
	prfm	PLDL1KEEP, [x9, #192]
	fmla	v4.2d, v16.2d, v28.2d[0]
	fmla	v5.2d, v17.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x13, #192]
	fmla	v12.2d, v20.2d, v28.2d[0]
	fmla	v13.2d, v21.2d, v28.2d[0]
	prfm	PLDL1KEEP, [x11, #32]
	fmla	v6.2d, v16.2d, v30.2d[0]
	fmla	v7.2d, v17.2d, v30.2d[0]
	prfm	PLDL1KEEP, [x14, #32]
	fmla	v14.2d, v20.2d, v30.2d[0]
	fmla	v15.2d, v21.2d, v30.2d[0]

	ldp		q16, q17, [x9, #64]
	fmla	v0.2d, v18.2d, v24.2d[1]
	fmla	v1.2d, v19.2d, v24.2d[1]
	ldp		q20, q21, [x13, #64]
	fmla	v8.2d, v22.2d, v24.2d[1]
	fmla	v9.2d, v23.2d, v24.2d[1]
	prfm	PLDL1KEEP, [x15, #32]
	fmla	v2.2d, v18.2d, v26.2d[1]
	fmla	v3.2d, v19.2d, v26.2d[1]
	prfm	PLDL1KEEP, [x16, #32]
	fmla	v10.2d, v22.2d, v26.2d[1]
	fmla	v11.2d, v23.2d, v26.2d[1]
	fmla	v4.2d, v18.2d, v28.2d[1]
	fmla	v5.2d, v19.2d, v28.2d[1]
	fmla	v12.2d, v22.2d, v28.2d[1]
	fmla	v13.2d, v23.2d, v28.2d[1]
//	add		x11, x11, x12
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
	cmp		w8, #4
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
	ldp		q24, q25, [x11], #32
	fmla	v2.2d, v18.2d, v27.2d[1]
	fmla	v3.2d, v19.2d, v27.2d[1]
	fmla	v10.2d, v22.2d, v27.2d[1]
	fmla	v11.2d, v23.2d, v27.2d[1]
	ldp		q26, q27, [x14], #32
	fmla	v4.2d, v18.2d, v29.2d[1]
	fmla	v5.2d, v19.2d, v29.2d[1]
	fmla	v12.2d, v22.2d, v29.2d[1]
	fmla	v13.2d, v23.2d, v29.2d[1]
	ldp		q28, q29, [x15], #32
	fmla	v6.2d, v18.2d, v31.2d[1]
	fmla	v7.2d, v19.2d, v31.2d[1]
	fmla	v14.2d, v22.2d, v31.2d[1]
	fmla	v15.2d, v23.2d, v31.2d[1]
	ldp		q30, q31, [x16], #32

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
//	add		x11, x11, x12
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

	sub		x11, x11, #32
	sub		x14, x14, #32
	sub		x15, x15, #32
	sub		x16, x16, #32

3: // clean1-up loop

	// unroll 0
	ldp		q24, q25, [x9, #0]
	ldp		q26, q27, [x13, #0]
	ldr		d28, [x11], #8
	ldr		d29, [x14], #8
	ldr		d30, [x15], #8
	ldr		d31, [x16], #8

	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]
	fmla	v2.2d, v24.2d, v29.2d[0]
	fmla	v3.2d, v25.2d, v29.2d[0]
	add		x9, x9, #32
	fmla	v10.2d, v26.2d, v29.2d[0]
	fmla	v11.2d, v27.2d, v29.2d[0]
	add		x13, x13, #32
	fmla	v4.2d, v24.2d, v30.2d[0]
	fmla	v5.2d, v25.2d, v30.2d[0]
//	add		x11, x11, #8
	fmla	v12.2d, v26.2d, v30.2d[0]
	fmla	v13.2d, v27.2d, v30.2d[0]
	sub		w8, w8, #1
	fmla	v6.2d, v24.2d, v31.2d[0]
	fmla	v7.2d, v25.2d, v31.2d[0]
	cmp		w8, #0
	fmla	v14.2d, v26.2d, v31.2d[0]
	fmla	v15.2d, v27.2d, v31.2d[0]

	bgt		3b

2: // return



#else // cortex a53



	// TODO



#endif
	


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_kernel_gemm_add_nn_8x4_lib4c)
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
	.macro INNER_SCALE_AB_8X4_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_8x4_lib)
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

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v12.2d, v26.2d, v28.2d[0]
	fmla	v13.2d, v27.2d, v28.2d[0]

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]


#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x4_lib)
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
	.macro INNER_SCALE_AB_8X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_8x4_vs_lib)
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

	cmp		w12, #4
	blt		1f

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v12.2d, v26.2d, v28.2d[0]
	fmla	v13.2d, v27.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldp		q26, q27, [x10, #32]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]

	b 0f

1:
	cmp		w12, #3
	blt		2f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	ldr		d27, [x10, #48]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]
	fmla	v9.2d, v27.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	ldr		d27, [x10, #48]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]
	fmla	v11.2d, v27.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	ldr		d27, [x10, #48]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v12.2d, v26.2d, v28.2d[0]
	fmla	v13.2d, v27.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	ldr		d27, [x10, #48]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]
	fmla	v15.2d, v27.2d, v28.2d[0]

	b 0f

2:
	cmp		w12, #2
	blt		3f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v12.2d, v26.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]

	b 0f

3:
	cmp		w12, #1
	blt		0f

	ldp		q24, q25, [x10, #0]
	ldr		d26, [x10, #32]
	add		x10, x10, x11
	fmla	v0.2d, v24.2d, v28.2d[0]
	fmla	v1.2d, v25.2d, v28.2d[0]
	fmla	v8.2d, v26.2d, v28.2d[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		d26, [x10, #32]
	add		x10, x10, x11
	fmla	v2.2d, v24.2d, v28.2d[0]
	fmla	v3.2d, v25.2d, v28.2d[0]
	fmla	v10.2d, v26.2d, v28.2d[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		d26, [x10, #32]
	add		x10, x10, x11
	fmla	v4.2d, v24.2d, v28.2d[0]
	fmla	v5.2d, v25.2d, v28.2d[0]
	fmla	v12.2d, v26.2d, v28.2d[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	ldr		q26, [x10, #32]
	add		x10, x10, x11
	fmla	v6.2d, v24.2d, v28.2d[0]
	fmla	v7.2d, v25.2d, v28.2d[0]
	fmla	v14.2d, v26.2d, v28.2d[0]

0:

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_8X4_LIB
#else
	.align 4
	FUN_START(inner_store_8x4_lib)
#endif

	stp		q0, q1, [x8, #0]
	stp		q8, q9, [x8, #32]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	stp		q10, q11, [x8, #32]
	add		x8, x8, x9
	stp		q4, q5, [x8, #0]
	stp		q12, q13, [x8, #32]
	add		x8, x8, x9
	stp		q6, q7, [x8, #0]
	stp		q14, q15, [x8, #32]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_store_8x4_lib)
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
	.macro INNER_STORE_8X4_VS_LIB
#else
	.align 4
	FUN_START(inner_store_8x4_vs_lib)
#endif

	cmp		w10, #8
	bge		1f

	mov		x12, x8

	ldp		q24, q25, [x12, #32]
	add		x12, x12, x9
	ldp		q26, q27, [x12, #32]
	add		x12, x12, x9
	ldp		q28, q29, [x12, #32]
	add		x12, x12, x9
	ldp		q30, q31, [x12, #32]

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
	stp		q0, q1, [x8, #0]
	stp		q8, q9, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #0]
	stp		q10, q11, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	stp		q4, q5, [x8, #0]
	stp		q12, q13, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #3
	beq		0f
	// 4th col
	stp		q6, q7, [x8, #0]
	stp		q14, q15, [x8, #32]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_L_8X4_LIB
#else
	.align 4
	FUN_START(inner_store_l_8x4_lib)
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
	stp		q8, q9, [x8, #32]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	stp		q10, q11, [x8, #32]
	add		x8, x8, x9
	str		q5, [x8, #16]
	stp		q12, q13, [x8, #32]
	add		x8, x8, x9
	str		q7, [x8, #16]
	stp		q14, q15, [x8, #32]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_store_l_8x4_lib)
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
	.macro INNER_STORE_L_8X4_VS_LIB
#else
	.align 4
	FUN_START(inner_store_l_8x4_vs_lib)
#endif

	cmp		w10, #8
	bge		1f

	mov		x12, x8

	ldp		q24, q25, [x12, #32]
	add		x12, x12, x9
	ldp		q26, q27, [x12, #32]
	add		x12, x12, x9
	ldp		q28, q29, [x12, #32]
	add		x12, x12, x9
	ldp		q30, q31, [x12, #32]

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
	stp		q8, q9, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #0]
	stp		q10, q11, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	str		q5, [x8, #16]
	stp		q12, q13, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #3
	beq		0f
	// 4th col
	str		q7, [x8, #16]
	stp		q14, q15, [x8, #32]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(double)
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_U_8X4_LIB
#else
	.align 4
	FUN_START(inner_store_u_8x4_lib)
#endif

	stp		q0, q1, [x8, #0]
	str		d8, [x8, #32]
	add		x8, x8, x9
	stp		q2, q3, [x8, #0]
	str		q10, [x8, #32]
	add		x8, x8, x9
	stp		q4, q5, [x8, #0]
	str		q12, [x8, #32]
	str		d13, [x8, #48]
	add		x8, x8, x9
	stp		q6, q7, [x8, #0]
	stp		q14, q15, [x8, #32]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	FUN_END(inner_store_u_8x4_lib)
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
	.macro INNER_STORE_U_8X4_VS_LIB
#else
	.align 4
	FUN_START(inner_store_u_8x4_vs_lib)
#endif

	cmp		w10, #8
	bge		1f

	mov		x12, x8

	ldp		q24, q25, [x12, #32]
	add		x12, x12, x9
	ldp		q26, q27, [x12, #32]
	add		x12, x12, x9
	ldp		q28, q29, [x12, #32]
	add		x12, x12, x9
	ldp		q30, q31, [x12, #32]

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
	stp		q0, q1, [x8, #0]
	str		d8, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q2, q3, [x8, #0]
	str		q10, [x8, #32]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	stp		q4, q5, [x8, #0]
	str		q12, [x8, #32]
	str		d13, [x8, #48]
	add		x8, x8, x9
	cmp		w11, #3
	beq		0f
	// 4th col
	stp		q6, q7, [x8, #0]
	stp		q14, q15, [x8, #32]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_u_8x4_vs_lib)
#endif





//                                 w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dgemm_nt_8x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB
#else
	bl inner_store_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_8x4_lib44c)





//                                    w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dgemm_nt_8x4_vs_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_vs_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_VS_LIB
#else
	bl inner_scale_ab_8x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB
#else
	bl inner_store_8x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_8x4_vs_lib44c)






//                                 w0        x1             x2         w3       x4         w5       x6            x7         sp+0     sp+8       sp+16
// void kernel_dgemm_nt_8x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_lib4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B
	mov		w12, w5 // sda
	lsl		w12, w12, #3 // 8*ldb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4c
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x6 // beta
	mov		x10, x7 // C
	ldr		w11, [sp, #(STACKSIZE + 0)] // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 8)] // D
	ldr		w9, [sp, #(STACKSIZE + 16)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB
#else
	bl inner_store_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_8x4_lib4cc)





//                                    w0        x1             x2         w3       x4         w5       x6            x7         sp+0     sp+8       sp+16    sp+24   sp+32
// void kernel_dgemm_nt_8x4_vs_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nt_8x4_vs_lib4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B
	mov		w12, w5 // sda
	lsl		w12, w12, #3 // 8*ldb

	ldr		w13, [sp, #(STACKSIZE + 32)] // n1
	cmp		w13, #1
	bgt		100f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X1_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_8x1_lib4c
#endif

	b		103f

100:

	ldr		w13, [sp, #(STACKSIZE + 32)] // n1
	cmp		w13, #2
	bgt		101f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X2_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_8x2_lib4c
#endif
	
	b		103f

101:

	ldr		w13, [sp, #(STACKSIZE + 32)] // n1
	cmp		w13, #3
	bgt		102f

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X3_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_8x3_lib4c
#endif
	
	b		103f

102:

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4c
#endif

103:



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x6 // beta
	mov		x10, x7 // C
	ldr		w11, [sp, #(STACKSIZE + 0)] // ldc
	lsl		w11, w11, #3 // 8*sdc
	ldr		w12, [sp, #(STACKSIZE + 24)] // m1
	ldr		w13, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_VS_LIB
#else
	bl inner_scale_ab_8x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 8)] // D
	ldr		w9, [sp, #(STACKSIZE + 16)] // ldd
	lsl		w9, w9, #3 // 8*sdd
	ldr		w10, [sp, #(STACKSIZE + 24)] // m1
	ldr		w11, [sp, #(STACKSIZE + 32)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB
#else
	bl inner_store_8x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nt_8x4_vs_lib4cc)





//                                 w0        x1             x2         w3       x4         w5       x6            x7         sp+0     sp+8       sp+16
// void kernel_dgemm_nn_8x4_lib4cc(int kmax, double *alpha, double *A, int sda, double *B, int ldb, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dgemm_nn_8x4_lib4cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #5 // 32*sda
	mov		x11, x4 // B
	mov		w12, w5 // ldb
	lsl		w12, w12, #3 // 8*ldb

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NN_8X4_LIB4C
#else
	bl	inner_kernel_gemm_add_nn_8x4_lib4c
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x6 // beta
	mov		x10, x7 // C
	ldr		w11, [sp, #(STACKSIZE + 0)] // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 8)] // D
	ldr		w9, [sp, #(STACKSIZE + 16)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB
#else
	bl inner_store_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dgemm_nn_8x4_lib4cc)





//                                   w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dsyrk_nt_l_8x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_8x4_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB
#else
	bl inner_store_l_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_8x4_lib44c)





//                                      w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dsyrk_nt_l_8x4_vs_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_l_8x4_vs_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_VS_LIB
#else
	bl inner_scale_ab_8x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_VS_LIB
#else
	bl inner_store_l_8x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_l_8x4_vs_lib44c)





//                                   w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8
// void kernel_dsyrk_nt_u_8x4_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_u_8x4_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_U_8X4_LIB
#else
	bl inner_store_u_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_u_8x4_lib44c)





//                                      w0        x1             x2         w3       x4         x5            x6         w7       sp+0       sp+8     sp+16   sp+24
// void kernel_dsyrk_nt_u_8x4_vs_lib44c(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_dsyrk_nt_u_8x4_vs_lib44c)
	


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
	mov		w11, w7 // ldc
	lsl		w11, w11, #3 // 8*sdc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_VS_LIB
#else
	bl inner_scale_ab_8x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // D
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #3 // 8*sdd
	ldr		w10, [sp, #(STACKSIZE + 16)] // m1
	ldr		w11, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_U_8X4_VS_LIB
#else
	bl inner_store_u_8x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_dsyrk_nt_u_8x4_vs_lib44c)





