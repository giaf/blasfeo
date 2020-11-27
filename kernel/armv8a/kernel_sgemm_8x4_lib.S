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
	.macro INNER_EDGE_TRSM_RLT_INV_8X4_LIB
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_8x4_lib)
#endif

	ldr			s16, [x10, #0] // E_inv[0]
	fmul		v0.4s, v0.4s, v16.s[0]
	fmul		v4.4s, v4.4s, v16.s[0]
	ldr			s16, [x8, #4] // E[1+4*0]
	fmls		v1.4s, v0.4s, v16.s[0]
	fmls		v5.4s, v4.4s, v16.s[0]
	ldr			s16, [x8, #8] // E[2+4*0]
	fmls		v2.4s, v0.4s, v16.s[0]
	fmls		v6.4s, v4.4s, v16.s[0]
	ldr			s16, [x8, #12] // E[3+4*0]
	fmls		v3.4s, v0.4s, v16.s[0]
	fmls		v7.4s, v4.4s, v16.s[0]
	add			x8, x8, x9

	ldr			s16, [x10, #4] // E_inv[1]
	fmul		v1.4s, v1.4s, v16.s[0]
	fmul		v5.4s, v5.4s, v16.s[0]
	ldr			s16, [x8, #8] // E[2+4*1]
	fmls		v2.4s, v1.4s, v16.s[0]
	fmls		v6.4s, v5.4s, v16.s[0]
	ldr			s16, [x8, #12] // E[3+4*1]
	fmls		v3.4s, v1.4s, v16.s[0]
	fmls		v7.4s, v5.4s, v16.s[0]
	add			x8, x8, x9

	ldr			s16, [x10, #8] // E_inv[2]
	fmul		v2.4s, v2.4s, v16.s[0]
	fmul		v6.4s, v6.4s, v16.s[0]
	ldr			s16, [x8, #12] // E[3+4*2]
	fmls		v3.4s, v2.4s, v16.s[0]
	fmls		v7.4s, v6.4s, v16.s[0]
//	add			x8, x8, x9

	ldr			s16, [x10, #12] // E_inv[3]
	fmul		v3.4s, v3.4s, v16.s[0]
	fmul		v7.4s, v7.4s, v16.s[0]
//	add			x8, x8, x9

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_8x4_lib)
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
	.macro INNER_EDGE_TRSM_RLT_INV_8X4_VS_LIB
#else
	.align 4
	FUN_START(inner_edge_trsm_rlt_inv_8x4_vs_lib)
#endif
	
	// first column
	ldr			s16, [x10, #0] // E_inv[0]
	fmul		v0.4s, v0.4s, v16.s[0]
	fmul		v4.4s, v4.4s, v16.s[0]
	cmp			w11, #2
	blt			0f // return

	// second column
	ldr			s16, [x8, #4] // E[1+4*0]
	fmls		v1.4s, v0.4s, v16.s[0]
	fmls		v5.4s, v4.4s, v16.s[0]
	ldr			s16, [x10, #4] // E_inv[1]
	fmul		v1.4s, v1.4s, v16.s[0]
	fmul		v5.4s, v5.4s, v16.s[0]
	cmp			w11, #3
	blt			0f // return

	// third column
	add			x12, x8, x9
	ldr			s16, [x8, #8] // E[2+4*0]
	fmls		v2.4s, v0.4s, v16.s[0]
	fmls		v6.4s, v4.4s, v16.s[0]
	ldr			s16, [x12, #8] // E[2+4*1]
	fmls		v2.4s, v1.4s, v16.s[0]
	fmls		v6.4s, v5.4s, v16.s[0]
	ldr			s16, [x10, #8] // E_inv[2]
	fmul		v2.4s, v2.4s, v16.s[0]
	fmul		v6.4s, v6.4s, v16.s[0]
	cmp			w11, #4
	blt			0f // return

	// forth column
	add			x13, x12, x9
	ldr			s16, [x8, #12] // E[3+4*0]
	fmls		v3.4s, v0.4s, v16.s[0]
	fmls		v7.4s, v4.4s, v16.s[0]
	ldr			s16, [x12, #12] // E[3+4*1]
	fmls		v3.4s, v1.4s, v16.s[0]
	fmls		v7.4s, v5.4s, v16.s[0]
	ldr			s16, [x13, #12] // E[3+4*2]
	fmls		v3.4s, v2.4s, v16.s[0]
	fmls		v7.4s, v6.4s, v16.s[0]
	ldr			s16, [x10, #12] // E_inv[3]
	fmul		v3.4s, v3.4s, v16.s[0]
	fmul		v7.4s, v7.4s, v16.s[0]

0:
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_edge_trsm_rlt_inv_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
// x11  <- ldc*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_8x4_lib)
#endif

	ld1		{v28.4s}, [x8]

	ld1		{v29.4s}, [x9]

	fmul	v0.4s, v0.4s, v28.s[0]
	fmul	v1.4s, v1.4s, v28.s[0]
	fmul	v2.4s, v2.4s, v28.s[0]
	fmul	v3.4s, v3.4s, v28.s[0]
	fmul	v4.4s, v4.4s, v28.s[0]
	fmul	v5.4s, v5.4s, v28.s[0]
	fmul	v6.4s, v6.4s, v28.s[0]
	fmul	v7.4s, v7.4s, v28.s[0]

	fcmpe	s29, #0.0
	beq		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]
	fmla	v1.4s, v26.4s, v29.s[0]
	fmla	v5.4s, v27.4s, v29.s[0]

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	ldp		q26, q27, [x10, #0]
	add		x10, x10, x11

	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]
	fmla	v3.4s, v26.4s, v29.s[0]
	fmla	v7.4s, v27.4s, v29.s[0]

0:

#if MACRO_LEVEL>=1
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
// x11  <- ldc*sizeof(float)
// x12  <- km
// x13  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_ab_8x4_vs_lib)
#endif

	ld1		{v28.4s}, [x8]

	ld1		{v29.4s}, [x9]

	fmul	v0.4s, v0.4s, v28.s[0]
	fmul	v1.4s, v1.4s, v28.s[0]
	fmul	v2.4s, v2.4s, v28.s[0]
	fmul	v3.4s, v3.4s, v28.s[0]
	fmul	v4.4s, v4.4s, v28.s[0]
	fmul	v5.4s, v5.4s, v28.s[0]
	fmul	v6.4s, v6.4s, v28.s[0]
	fmul	v7.4s, v7.4s, v28.s[0]

	fcmpe	d29, #0.0
	beq		0f

	cmp		w12, #8
	blt		1f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w13, #1
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w13, #2
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w13, #3
	ble		0f

	ldp		q24, q25, [x10, #0]
	add		x10, x10, x11
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

1:
	cmp		w12, #7
	blt		2f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	ldr		s26, [x10, #24]
	ins		v25.s[2], v26.s[0]
	add		x10, x10, x11
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	ldr		s26, [x10, #24]
	ins		v25.s[2], v26.s[0]
	add		x10, x10, x11
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	ldr		s26, [x10, #24]
	ins		v25.s[2], v26.s[0]
	add		x10, x10, x11
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	ldr		s26, [x10, #24]
	ins		v25.s[2], v26.s[0]
	add		x10, x10, x11
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

2:
	cmp		w12, #6
	blt		3f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	ldr		d25, [x10, #16]
	add		x10, x10, x11
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

3:
	cmp		w12, #5
	blt		0f

	ldr		q24, [x10, #0]
	ldr		s25, [x10, #16]
	add		x10, x10, x11
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w13, #1
	ble		0f

	ldr		q24, [x10, #0]
	ldr		s25, [x10, #16]
	add		x10, x10, x11
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w13, #2
	ble		0f

	ldr		q24, [x10, #0]
	ldr		s25, [x10, #16]
	add		x10, x10, x11
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w13, #3
	ble		0f

	ldr		q24, [x10, #0]
	ldr		s25, [x10, #16]
	add		x10, x10, x11
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- beta
// x9  <- C
// x11  <- ldc*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_8X4_LIB
#else
	.align	4
	FUN_START(inner_scale_m1b_8x4_lib)
#endif

	ld1		{v29.4s}, [x8]

	fneg	v0.4s, v0.4s
	fneg	v1.4s, v1.4s
	fneg	v2.4s, v2.4s
	fneg	v3.4s, v3.4s
	fneg	v4.4s, v4.4s
	fneg	v5.4s, v5.4s
	fneg	v6.4s, v6.4s
	fneg	v7.4s, v7.4s

	fcmpe	s29, #0.0
	beq		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	ldp		q26, q27, [x9, #0]
	add		x9, x9, x10

	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]
	fmla	v1.4s, v26.4s, v29.s[0]
	fmla	v5.4s, v27.4s, v29.s[0]

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	ldp		q26, q27, [x9, #0]
	add		x9, x9, x10

	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]
	fmla	v3.4s, v26.4s, v29.s[0]
	fmla	v7.4s, v27.4s, v29.s[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_8x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- beta
// x9  <- C
// x10  <- ldc*sizeof(float)
// x11  <- km
// x12  <- kn
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M1B_8X4_VS_LIB
#else
	.align	4
	FUN_START(inner_scale_m1b_8x4_vs_lib)
#endif

	ld1		{v29.4s}, [x8]

	fneg	v0.4s, v0.4s
	fneg	v1.4s, v1.4s
	fneg	v2.4s, v2.4s
	fneg	v3.4s, v3.4s
	fneg	v4.4s, v4.4s
	fneg	v5.4s, v5.4s
	fneg	v6.4s, v6.4s
	fneg	v7.4s, v7.4s

	fcmpe	d29, #0.0
	beq		0f

	cmp		w11, #8
	blt		1f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w12, #1
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w12, #2
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w12, #3
	ble		0f

	ldp		q24, q25, [x9, #0]
	add		x9, x9, x10
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

1:
	cmp		w11, #7
	blt		2f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	ldr		s26, [x9, #24]
	ins		v25.s[2], v26.s[0]
	add		x9, x9, x10
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w12, #1
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	ldr		s26, [x9, #24]
	ins		v25.s[2], v26.s[0]
	add		x9, x9, x10
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w12, #2
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	ldr		s26, [x9, #24]
	ins		v25.s[2], v26.s[0]
	add		x9, x9, x10
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w12, #3
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	ldr		s26, [x9, #24]
	ins		v25.s[2], v26.s[0]
	add		x9, x9, x10
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

2:
	cmp		w11, #6
	blt		3f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w12, #1
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w12, #2
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w12, #3
	ble		0f

	ldr		q24, [x9, #0]
	ldr		d25, [x9, #16]
	add		x9, x9, x10
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

	b 0f

3:
	cmp		w11, #5
	blt		0f

	ldr		q24, [x9, #0]
	ldr		s25, [x9, #16]
	add		x9, x9, x10
	fmla	v0.4s, v24.4s, v29.s[0]
	fmla	v4.4s, v25.4s, v29.s[0]

	cmp		w12, #1
	ble		0f

	ldr		q24, [x9, #0]
	ldr		s25, [x9, #16]
	add		x9, x9, x10
	fmla	v1.4s, v24.4s, v29.s[0]
	fmla	v5.4s, v25.4s, v29.s[0]

	cmp		w12, #2
	ble		0f

	ldr		q24, [x9, #0]
	ldr		s25, [x9, #16]
	add		x9, x9, x10
	fmla	v2.4s, v24.4s, v29.s[0]
	fmla	v6.4s, v25.4s, v29.s[0]

	cmp		w12, #3
	ble		0f

	ldr		q24, [x9, #0]
	ldr		s25, [x9, #16]
	add		x9, x9, x10
	fmla	v3.4s, v24.4s, v29.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m1b_8x4_vs_lib)
#endif





// subroutine
//
// input arguments:
// x8  <- C
// x9  <- ldc*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_M11_8X4_LIB
#else
	.align	4
	FUN_START(inner_scale_m11_8x4_lib)
#endif

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	ldp		q26, q27, [x8, #0]
	add		x8, x8, x9

	fsub	v0.4s, v24.4s, v0.4s
	fsub	v4.4s, v25.4s, v4.4s
	fsub	v1.4s, v26.4s, v1.4s
	fsub	v5.4s, v27.4s, v5.4s

	ldp		q24, q25, [x8, #0]
	add		x8, x8, x9
	ldp		q26, q27, [x8, #0]
	add		x8, x8, x9

	fsub	v2.4s, v24.4s, v2.4s
	fsub	v6.4s, v25.4s, v6.4s
	fsub	v3.4s, v26.4s, v3.4s
	fsub	v7.4s, v27.4s, v7.4s

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_m11_8x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X4_LIB
#else
	.align 4
	FUN_START(inner_store_8x4_lib)
#endif

	stp		q0, q4, [x8, #0]
	add		x8, x8, x9
	stp		q1, q5, [x8, #0]
	add		x8, x8, x9
	stp		q2, q6, [x8, #0]
	add		x8, x8, x9
	stp		q3, q7, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_8x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(float)
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

	ldr		q16, [x12, #16]
	add		x12, x12, x9
	ldr		q17, [x12, #16]
	add		x12, x12, x9
	ldr		q18, [x12, #16]
	add		x12, x12, x9
	ldr		q19, [x12, #16]

	// 4th row
	ins		v4.s[3], v16.s[3]
	ins		v5.s[3], v17.s[3]
	ins		v6.s[3], v18.s[3]
	ins		v7.s[3], v19.s[3]
	cmp		w10, #7
	bge		1f
	// 3th row
	ins		v4.s[2], v16.s[2]
	ins		v5.s[2], v17.s[2]
	ins		v6.s[2], v18.s[2]
	ins		v7.s[2], v19.s[2]
	cmp		w10, #6
	bge		1f
	// 2nd row
	ins		v4.s[1], v16.s[1]
	ins		v5.s[1], v17.s[1]
	ins		v6.s[1], v18.s[1]
	ins		v7.s[1], v19.s[1]
	cmp		w10, #5
	bge		1f
	// 1st row
	ins		v4.s[0], v16.s[0]
	ins		v5.s[0], v17.s[0]
	ins		v6.s[0], v18.s[0]
	ins		v7.s[0], v19.s[0]

1:
	// 1st col
	stp		q0, q4, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #2
	blt		0f
	// 2nd col
	stp		q1, q5, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #3
	blt		0f
	// 3rd col
	stp		q2, q6, [x8, #0]
	add		x8, x8, x9
	cmp		w11, #3
	beq		0f
	// 4th col
	stp		q3, q7, [x8, #0]

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
// x9   <- ldd*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X4_LIB
#else
	.align 4
	FUN_START(inner_store_l_8x4_lib)
#endif

	mov		x12, x8

	add		x12, x12, x9
	ldr		q16, [x12, #0]
	add		x12, x12, x9
	ldr		q17, [x12, #0]
	add		x12, x12, x9
	ldr		q18, [x12, #0]

	ins		v1.s[0], v16.s[0]
	ins		v2.d[0], v17.d[0]
	ins		v3.d[0], v18.d[0]
	ins		v3.s[2], v18.s[2]

	stp		q0, q4, [x8, #0]
	add		x8, x8, x9
	stp		q1, q5, [x8, #0]
	add		x8, x8, x9
	stp		q2, q6, [x8, #0]
	add		x8, x8, x9
	stp		q3, q7, [x8, #0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_l_8x4_lib)
#endif





// subroutine
//
// input arguments:
// x8   <- D
// x9   <- ldd*sizeof(float)
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_PREFETCH_8X4_LIB
#else
	.align 4
	FUN_START(inner_prefetch_8x4_lib)
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

	FUN_END(inner_prefetch_8x4_lib)
#endif





//                                 w0        x1             x2        w3        x4       x5           x6        w7       sp+0      sp+8
// void kernel_sgemm_nt_8x4_lib44cc(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd)

	.align	4
	GLOB_FUN_START(kernel_sgemm_nt_8x4_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// prefetch
	ldr		x8, [sp, #(STACKSIZE + 0)] // ldd
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_PREFETCH_8X4_LIB
#else
	bl inner_prefetch_8x4_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_LIB
#else
	bl inner_scale_ab_8x4_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // ldd
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB
#else
	bl inner_store_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_sgemm_nt_8x4_lib44cc)





//                                    w0        x1             x2        w3        x4       x5           x6        w7       sp+0      sp+8     sp+16   sp+24
// void kernel_sgemm_nt_8x4_vs_lib44cc(int kmax, float *alpha, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_sgemm_nt_8x4_vs_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x2 // A
	mov		w10, w3 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x4 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// prefetch
//	ldr		x8, [sp, #(STACKSIZE + 0)] // ldd
//	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
//	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
//	INNER_PREFETCH_8X4_LIB
#else
//	bl inner_prefetch_8x4_lib
#endif



	// call inner blend for generic alpha and beta
	mov		x8, x1 // alpha
	mov		x9, x5 // beta
	mov		x10, x6 // C
	mov		w11, w7 // ldc
	lsl		w11, w11, #2 // 4*ldc
	ldr		w12, [sp, #(STACKSIZE + 16)] // m1
	ldr		w13, [sp, #(STACKSIZE + 24)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X4_VS_LIB
#else
	bl inner_scale_ab_8x4_vs_lib
#endif



	// store n
	ldr		x8, [sp, #(STACKSIZE + 0)] // ldd
	ldr		w9, [sp, #(STACKSIZE + 8)] // ldd
	lsl		w9, w9, #2 // 4*ldd
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

	FUN_END(kernel_sgemm_nt_8x4_vs_lib44cc)





//                                          w0        x1        x2        x3           x4        w5       x6        w7       sp+0      sp+8     sp+16    sp+24
// void kernel_strsm_nt_rl_inv_8x4_lib44ccc(int kmax, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd, float *E, int lde, float *inv_diag_E)

	.align	4
	GLOB_FUN_START(kernel_strsm_nt_rl_inv_8x4_lib44ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // ldc
	lsl		w10, w10, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_8X4_LIB
#else
	bl inner_scale_m1b_8x4_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		w9, [sp, #(STACKSIZE + 16)] // sde
	lsl		w9, w9, #2 // 4*ldc
	ldr		x10, [sp, #(STACKSIZE + 24)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_8X4_LIB
#else
	bl inner_edge_trsm_rlt_inv_8x4_lib
#endif



	// store l
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_LIB
#else
	bl inner_store_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_strsm_nt_rl_inv_8x4_lib44ccc)





//                                             w0        x1        x2        x3           x4        w5       x6        w7       sp+0      sp+8     sp+16    sp+24              sp+32   sp+40
// void kernel_strsm_nt_rl_inv_8x4_vs_lib44ccc(int kmax, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd, float *E, int lde, float *inv_diag_E, int m1, int n1)

	.align	4
	GLOB_FUN_START(kernel_strsm_nt_rl_inv_8x4_vs_lib44ccc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // beta
	mov		x9, x5 // C
	mov		w10, w6 // ldc
	lsl		w10, w10, #2 // 4*ldc
	ldr		w11, [sp, #(STACKSIZE + 32)] // m1
	ldr		w12, [sp, #(STACKSIZE + 40)] // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_M1B_8X4_VS_LIB
#else
	bl inner_scale_m1b_8x4_vs_lib
#endif



	// solution
	ldr		x8, [sp, #(STACKSIZE + 8)] // E
	ldr		w9, [sp, #(STACKSIZE + 16)] // sde
	lsl		w9, w9, #2 // 4*ldc
	ldr		x10, [sp, #(STACKSIZE + 24)] // inv_diag_E
	ldr		w11, [sp, #(STACKSIZE + 40)] // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_8X4_VS_LIB
#else
	bl inner_edge_trsm_rlt_inv_8x4_vs_lib
#endif



	// store l
	mov		x8, x7 // D
	ldr		w9, [sp, #(STACKSIZE + 0)] // ldd
	lsl		w9, w9, #2 // 4*ldd
	ldr		w10, [sp, #(STACKSIZE + 32)] // m1
	ldr		w11, [sp, #(STACKSIZE + 40)] // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_VS_LIB
#else
	bl inner_store_8x4_vs_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_strsm_nt_rl_inv_8x4_vs_lib44ccc)





//                                     w0        x1        x2        x3        w4       x5        w6       x7       sp+0
// void kernel_spotrf_nt_l_8x4_lib44cc(int kmax, float *A, int sda, float *B, float *C, int ldc, float *D, int ldd, float *inv_diag_D)

	.align	4
	GLOB_FUN_START(kernel_spotrf_nt_l_8x4_lib44cc)
	


	PROLOGUE



	ZERO_ACC



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // sda
	lsl		w10, w10, #4 // 16*sda
	mov		x11, x3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_8X4_LIB4
#else
	bl	inner_kernel_gemm_add_nt_8x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x4 // C
	mov		w9, w5 // ldc
	lsl		w9, w9, #2 // 4*ldc

#if MACRO_LEVEL>=1
	INNER_SCALE_M11_8X4_LIB
#else
	bl inner_scale_m11_8x4_lib
#endif



	// factorization
	ldr		x8, [sp, #(STACKSIZE + 0)] // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_8X4_LIB4
#else
	bl inner_edge_potrf_8x4_lib4
#endif



	// store l
	mov		x8, x6 // D
	mov		w9, w7 // ldd
	lsl		w9, w9, #2 // 4*ldd

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_LIB
#else
	bl inner_store_l_8x4_lib
#endif



	EPILOGUE

	mov	x0, #0

	ret

	FUN_END(kernel_spotrf_nt_l_8x4_lib44cc)






