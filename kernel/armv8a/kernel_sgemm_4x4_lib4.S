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
	add sp, sp, #-(11 * 16); \
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
// x10   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_add_nt_4x4_lib4, %function
inner_kernel_gemm_add_nt_4x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	

	// unroll 0
	fmla	v0.4s, v24.4s, v28.4s[0]
	ld1		{v26.2d, v27.2d}, [x9], #32
	fmla	v1.4s, v24.4s, v28.4s[1]
	ld1		{v30.2d, v31.2d}, [x10], #32
	fmla	v2.4s, v24.4s, v28.4s[2]
	prfm	PLDL1KEEP, [x9, #64]
	fmla	v3.4s, v24.4s, v28.4s[3]
	prfm	PLDL1KEEP, [x10, #64]

	// unroll 1
	fmla	v0.4s, v25.4s, v29.4s[0]
	sub		w8, w8, #4
	fmla	v1.4s, v25.4s, v29.4s[1]
	fmla	v2.4s, v25.4s, v29.4s[2]
	fmla	v3.4s, v25.4s, v29.4s[3]

	// unroll 2
	fmla	v0.4s, v26.4s, v30.4s[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmla	v1.4s, v26.4s, v30.4s[1]
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v2.4s, v26.4s, v30.4s[2]
	fmla	v3.4s, v26.4s, v30.4s[3]

	// unroll 3
	fmla	v0.4s, v27.4s, v31.4s[0]
	fmla	v1.4s, v27.4s, v31.4s[1]
	fmla	v2.4s, v27.4s, v31.4s[2]
	fmla	v3.4s, v27.4s, v31.4s[3]

	cmp		w8, #4
	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]

	// unroll 1
	fmla	v0.4s, v25.4s, v29.4s[0]
	fmla	v1.4s, v25.4s, v29.4s[1]
	fmla	v2.4s, v25.4s, v29.4s[2]
	fmla	v3.4s, v25.4s, v29.4s[3]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]

	// unroll 3
	fmla	v0.4s, v25.4s, v29.4s[0]
	fmla	v1.4s, v25.4s, v29.4s[1]
	fmla	v2.4s, v25.4s, v29.4s[2]
	fmla	v3.4s, v25.4s, v29.4s[3]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d}, [x9], #16
	ld1		{v28.2d}, [x10], #16
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v24.4s, v28.4s[1]
	fmla	v2.4s, v24.4s, v28.4s[2]
	fmla	v3.4s, v24.4s, v28.4s[3]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_add_nt_4x4_lib4, .-inner_kernel_gemm_add_nt_4x4_lib4
#endif





// subroutine
//
// input arguments:
// w8   <- k
// x9   <- A
// x10   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_SUB_NT_4X4_LIB4
#else
	.align	4
	.type inner_kernel_gemm_sub_nt_4x4_lib4, %function
inner_kernel_gemm_sub_nt_4x4_lib4:
#endif

// TODO more aggressive preload of A !!!

	// early return
	cmp		w8, #0
	ble		2f // return

	// prefetch
	prfm	PLDL1KEEP, [x9, #0]
	prfm	PLDL1KEEP, [x10, #0]

	cmp		w8, #4
	ble		0f // consider clean up loop

	// preload
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32

	// prefetch
	prfm	PLDL1KEEP, [x9, #32]
	prfm	PLDL1KEEP, [x10, #32]

	// main loop
1:
	

	// unroll 0
	fmls	v0.4s, v24.4s, v28.4s[0]
	ld1		{v26.2d, v27.2d}, [x9], #32
	fmls	v1.4s, v24.4s, v28.4s[1]
	ld1		{v30.2d, v31.2d}, [x10], #32
	fmls	v2.4s, v24.4s, v28.4s[2]
	prfm	PLDL1KEEP, [x9, #64]
	fmls	v3.4s, v24.4s, v28.4s[3]
	prfm	PLDL1KEEP, [x10, #64]

	// unroll 1
	fmls	v0.4s, v25.4s, v29.4s[0]
	sub		w8, w8, #4
	fmls	v1.4s, v25.4s, v29.4s[1]
	fmls	v2.4s, v25.4s, v29.4s[2]
	fmls	v3.4s, v25.4s, v29.4s[3]

	// unroll 2
	fmls	v0.4s, v26.4s, v30.4s[0]
	ld1		{v24.2d, v25.2d}, [x9], #32
	fmls	v1.4s, v26.4s, v30.4s[1]
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v2.4s, v26.4s, v30.4s[2]
	fmls	v3.4s, v26.4s, v30.4s[3]

	// unroll 3
	fmls	v0.4s, v27.4s, v31.4s[0]
	fmls	v1.4s, v27.4s, v31.4s[1]
	fmls	v2.4s, v27.4s, v31.4s[2]
	fmls	v3.4s, v27.4s, v31.4s[3]

	cmp		w8, #4
	bgt		1b

	sub		x9, x9, #32
	sub		x10, x10, #32

0:

	cmp		w8, #3
	ble		4f

	// unroll 0
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.4s, v24.4s, v28.4s[0]
	fmls	v1.4s, v24.4s, v28.4s[1]
	fmls	v2.4s, v24.4s, v28.4s[2]
	fmls	v3.4s, v24.4s, v28.4s[3]

	// unroll 1
	fmls	v0.4s, v25.4s, v29.4s[0]
	fmls	v1.4s, v25.4s, v29.4s[1]
	fmls	v2.4s, v25.4s, v29.4s[2]
	fmls	v3.4s, v25.4s, v29.4s[3]

	// unroll 2
	ld1		{v24.2d, v25.2d}, [x9], #32
	ld1		{v28.2d, v29.2d}, [x10], #32
	fmls	v0.4s, v24.4s, v28.4s[0]
	fmls	v1.4s, v24.4s, v28.4s[1]
	fmls	v2.4s, v24.4s, v28.4s[2]
	fmls	v3.4s, v24.4s, v28.4s[3]

	// unroll 3
	fmls	v0.4s, v25.4s, v29.4s[0]
	fmls	v1.4s, v25.4s, v29.4s[1]
	fmls	v2.4s, v25.4s, v29.4s[2]
	fmls	v3.4s, v25.4s, v29.4s[3]

	sub		w8, w8, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		w8, #0
	ble		2f // return

3: // clean1-up loop

	// unroll 0
	ld1		{v24.2d}, [x9], #16
	ld1		{v28.2d}, [x10], #16
	fmls	v0.4s, v24.4s, v28.4s[0]
	fmls	v1.4s, v24.4s, v28.4s[1]
	fmls	v2.4s, v24.4s, v28.4s[2]
	fmls	v3.4s, v24.4s, v28.4s[3]

	sub		w8, w8, #1
	cmp		w8, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemm_sub_nt_4x4_lib4, .-inner_kernel_gemm_sub_nt_4x4_lib4
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
	.macro INNER_EDGE_TRSM_RLT_INV_4X4_LIB4
#else
	.align 4
	.type inner_edge_trsm_rlt_inv_4x4_lib4, %function
inner_edge_trsm_rlt_inv_4x4_lib4:
#endif
	
	// first column
	ldr			s16, [x9, #0] // E_inv[0]
	fmul		v0.4s, v0.4s, v16.4s[0]

	// second column
	ldr			s16, [x8, #4] // E[1+4*0]
	fmls		v1.4s, v0.4s, v16.4s[0]
	ldr			s16, [x9, #4] // E_inv[1]
	fmul		v1.4s, v1.4s, v16.4s[0]

	// third column
	ldr			s16, [x8, #8] // E[2+4*0]
	fmls		v2.4s, v0.4s, v16.4s[0]
	ldr			s16, [x8, #24] // E[2+4*1]
	fmls		v2.4s, v1.4s, v16.4s[0]
	ldr			s16, [x9, #8] // E_inv[2]
	fmul		v2.4s, v2.4s, v16.4s[0]

	// forth column
	ldr			s16, [x8, #12] // E[3+4*0]
	fmls		v3.4s, v0.4s, v16.4s[0]
	ldr			s16, [x8, #28] // E[3+4*1]
	fmls		v3.4s, v1.4s, v16.4s[0]
	ldr			s16, [x8, #44] // E[3+4*1]
	fmls		v3.4s, v2.4s, v16.4s[0]
	ldr			s16, [x9, #12] // E_inv[2]
	fmul		v3.4s, v3.4s, v16.4s[0]

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_trsm_rlt_inv_4x4_lib4, .-inner_edge_trsm_rlt_inv_4x4_lib4
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
	.macro INNER_EDGE_POTRF_4X4_LIB4
#else
	.p2align 4
	.type inner_edge_potrf_4x4_lib4, %function
inner_edge_potrf_4x4_lib4:
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

	// second column
	fmls		v1.4s, v0.4s, v0.4s[1]
	ins			v17.s[0], v1.s[1]
	fcmpe		s17, #0
	ble			3f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
4:
	str			s18, [x8, #4]
	fmul		v1.4s, v1.4s, v18.4s[0]

	// third column
	fmls		v2.4s, v0.4s, v0.4s[2]
	fmls		v2.4s, v1.4s, v1.4s[2]
	ins			v17.s[0], v2.s[2]
	fcmpe		s17, #0
	ble			5f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
6:
	str			s18, [x8, #8]
	fmul		v2.4s, v2.4s, v18.4s[0]

	// fourth column
	fmls		v3.4s, v0.4s, v0.4s[3]
	fmls		v3.4s, v1.4s, v1.4s[3]
	fmls		v3.4s, v2.4s, v2.4s[3]
	ins			v17.s[0], v3.s[3]
	fcmpe		s17, #0
	ble			7f
	fsqrt		s17, s17
	fdiv		s18, s16, s17
8:
	str			s18, [x8, #12]
	fmul		v3.4s, v3.4s, v18.4s[0]

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

	.size	inner_edge_potrf_4x4_lib4, .-inner_edge_potrf_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- alpha
// x9   <- beta
// x10  <- C
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_4X4_LIB4
#else
	.align	4
	.type inner_scale_ab_4x4_lib4, %function
inner_scale_ab_4x4_lib4:
#endif

	ld1		{v28.2d}, [x8]

	fmul	v0.4s, v0.4s, v28.4s[0]
	fmul	v1.4s, v1.4s, v28.4s[0]
	fmul	v2.4s, v2.4s, v28.4s[0]
	fmul	v3.4s, v3.4s, v28.4s[0]

	ld1		{v28.2d}, [x9]

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
	fmla	v0.4s, v24.4s, v28.4s[0]
	fmla	v1.4s, v25.4s, v28.4s[0]
	fmla	v2.4s, v26.4s, v28.4s[0]
	fmla	v3.4s, v27.4s, v28.4s[0]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_ab_4x4_lib4, .-inner_scale_ab_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8  <- C
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_11_4X4_LIB4
#else
	.align	4
	.type inner_scale_11_4x4_lib4, %function
inner_scale_11_4x4_lib4:
#endif

	ld1		{v24.4s, v25.4s, v26.4s, v27.4s}, [x8], #64
	fadd	v0.4s, v24.4s, v0.4s
	fadd	v1.4s, v25.4s, v1.4s
	fadd	v2.4s, v26.4s, v2.4s
	fadd	v3.4s, v27.4s, v3.4s

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_scale_11_4x4_lib4, .-inner_scale_11_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_4X4_LIB4
#else
	.align 4
	.type inner_store_4x4_lib4, %function
inner_store_4x4_lib4:
#endif

	st1		{v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif





// subroutine
//
// input arguments:
// x8   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_L_4X4_LIB4
#else
	.align 4
	.type inner_store_l_4x4_lib4, %function
inner_store_l_4x4_lib4:
#endif

	ldr		q16, [x8, #16]
	ldr		q17, [x8, #32]
	ldr		q18, [x8, #48]

	ins		v1.s[0], v16.s[0]
	ins		v2.d[0], v17.d[0]
	ins		v3.d[0], v18.d[0]
	ins		v3.s[2], v18.s[2]

	stp		q0, q1, [x8, #0]
	stp		q2, q3, [x8, #32]

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_store_l_4x4_lib4, .-inner_store_l_4x4_lib4
#endif




//                               w0        x1             x2         x3         x4            x5         x6
// void kernel_sgemm_nt_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)

	.align	4
	.global	kernel_sgemm_nt_4x4_lib4
	.type	kernel_sgemm_nt_4x4_lib4, %function
kernel_sgemm_nt_4x4_lib4:
	


	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov	d0, xzr
	fmov    d1, d0
	fmov    d2, d0
	fmov    d3, d0



	// call inner kernel dgemm nt
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

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
	bl inner_scale_ab_4x4_lib4
#endif



	// store n
	mov		x8, x6

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	bl inner_store_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_sgemm_nt_4x4_lib4, .-kernel_sgemm_nt_4x4_lib4





//                                      w0        x1         x2         x3         x4         x5         x6
// void kernel_strsm_nt_rl_inv_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.align	4
	.globl kernel_strsm_nt_rl_inv_4x4_lib4
	.type kernel_strsm_nt_rl_inv_4x4_lib4, %function
kernel_strsm_nt_rl_inv_4x4_lib4:



	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov	d0, xzr
	fmov    d1, d0
	fmov    d2, d0
	fmov    d3, d0



	// call inner kernel gemm nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
	bl inner_scale_11_4x4_lib4
#endif



	// solution
	mov		x8, x5 // E
	mov		x9, x6 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_TRSM_RLT_INV_4X4_LIB4
#else
	bl inner_edge_trsm_rlt_inv_4x4_lib4
#endif



	// store
	mov		x8, x4

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
	bl inner_store_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_strsm_nt_rl_inv_4x4_lib4, .-kernel_strsm_nt_rl_inv_4x4_lib4





//                                  w0        x1         x2         x3         x4         x5
// void kernel_spotrf_nt_l_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, double *inv_diag_D);

	.align	4
	.globl kernel_spotrf_nt_l_4x4_lib4
	.type kernel_spotrf_nt_l_4x4_lib4, %function
kernel_spotrf_nt_l_4x4_lib4:



	PROLOGUE



	// TODO zero the entire 128-bit register ???
	fmov	d0, xzr
	fmov    d1, d0
	fmov    d2, d0
	fmov    d3, d0



	// call inner kernel syrk l nt
	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		x10, x2 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_SUB_NT_4X4_LIB4
#else
	bl	inner_kernel_gemm_sub_nt_4x4_lib4
#endif



	// call inner blend for alpha=1.0 and beta=1.0
	mov		x8, x3 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
	bl inner_scale_11_4x4_lib4
#endif



	// factorization
	mov		x8, x5 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_POTRF_4X4_LIB4
#else
	bl inner_edge_potrf_4x4_lib4
#endif



	// store l
	mov		x8, x4

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
	bl inner_store_l_4x4_lib4
#endif



	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_spotrf_nt_l_4x4_lib4, .-kernel_spotrf_nt_l_4x4_lib4





