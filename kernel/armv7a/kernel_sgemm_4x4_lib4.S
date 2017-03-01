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

#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif



// subroutine
//
// input arguments:
// r4   <- k
// r5   <- A
// r6   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nt_4x4_lib4, %function
inner_kernel_gemm_add_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nt_4x4_lib4:
#endif
#endif

	// early return
	cmp		r4, #0
	ble		2f // return

	// prefetch
	pld		[r5, #0]
	pld		[r6, #0]

	// preload A
	vld1.64		{d0, d1}, [r6:128]! // B
	vld1.64		{d2, d3}, [r5:128]! // A

	cmp		r4, #4
	ble		0f // consider clean up loop

	// main loop
1:
	
	// prefetch
	pld		[r5, #64]
	pld		[r6, #64]

	// unroll 0
	vmla.f32	q4, q1, d0[0]
	vld1.64		{d4, d5}, [r6:128]! // B
	vmla.f32	q5, q1, d0[1]
	vld1.64		{d6, d7}, [r5:128]! // A
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]

	// unroll 1
	vmla.f32	q4, q3, d4[0]
	vld1.64		{d0, d1}, [r6:128]! // B
	vmla.f32	q5, q3, d4[1]
	vld1.64		{d2, d3}, [r5:128]! // A
	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]

	// unroll 2
	vmla.f32	q4, q1, d0[0]
	vld1.64		{d4, d5}, [r6:128]! // B
	vmla.f32	q5, q1, d0[1]
	vld1.64		{d6, d7}, [r5:128]! // A
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]

	// unroll 3
	vmla.f32	q4, q3, d4[0]
	vld1.64		{d0, d1}, [r6:128]! // B
	vmla.f32	q5, q3, d4[1]
	vld1.64		{d2, d3}, [r5:128]! // A
	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]

	sub		r4, r4, #4

	cmp		r4, #4
	bgt		1b

0:

	cmp		r4, #3
	ble		4f

	// unroll 0
	vmla.f32	q4, q1, d0[0]
	vld1.64		{d4, d5}, [r6:128]! // B
	vmla.f32	q5, q1, d0[1]
	vld1.64		{d6, d7}, [r5:128]! // A
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]

	// unroll 1
	vmla.f32	q4, q3, d4[0]
	vld1.64		{d0, d1}, [r6:128]! // B
	vmla.f32	q5, q3, d4[1]
	vld1.64		{d2, d3}, [r5:128]! // A
	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]

	// unroll 2
	vmla.f32	q4, q1, d0[0]
	vld1.64		{d4, d5}, [r6:128]! // B
	vmla.f32	q5, q1, d0[1]
	vld1.64		{d6, d7}, [r5:128]! // A
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]

	// unroll 3
	vmla.f32	q4, q3, d4[0]
	vmla.f32	q5, q3, d4[1]
	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]

	sub		r4, r4, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		r4, #0
	ble		2f // return

	sub		r5, r5, #16
	sub		r6, r6, #16

3: // clean1-up loop

	// unroll 0
	vld1.64		{d0, d1}, [r6:128]! // B
	vld1.64		{d2, d3}, [r5:128]! // A
	vmla.f32	q4, q1, d0[0]
	vmla.f32	q5, q1, d0[1]
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]

	sub		r4, r4, #1
	cmp		r4, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nt_4x4_lib4, .-inner_kernel_gemm_add_nt_4x4_lib4
#endif
#endif





// subroutine
//
// input arguments:
// r4   <- alpha
// r5   <- beta
// r6   <- C
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_lib4, %function
inner_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_lib4:
#endif
#endif

	flds		s8, [r4, #0] // alpha

	vmul.f32	q4, q4, d4[0]
	flds		s9, [r5, #0] // beta
	vmul.f32	q5, q5, d4[0]
	flds		s10, .LC00 // 0.0
	vmul.f32	q6, q6, d4[0]
	fcmpes		s9, s10
	vmul.f32	q7, q7, d4[0]
	fmstat

	beq		0f // end

	vld1.64		{d0, d1, d2, d3}, [r6:128]!
	vmla.f32	q4, q0, d4[1]
	vmla.f32	q5, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r6:128]!
	vmla.f32	q6, q0, d4[1]
	vmla.f32	q7, q1, d4[1]

0:

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_lib4, .-inner_scale_ab_4x4_lib4
#endif
#endif





// subroutine
//
// input arguments:
// r4   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_lib4, %function
inner_store_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_lib4:
#endif
#endif

	vst1.64		{d8, d9, d10, d11}, [r4:128]!
	vst1.64		{d12, d13, d14, d15}, [r4:128]!

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif
#endif





// zero double word
	.align 3
.LC00: // { 0 }
	.word 0
	.word 0
	.word 0
	.word 0

//                               r0        r1             r2         r3         sp+0          sp+4       sp+8
// void kernel_sgemm_nt_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)

//	.p2align 4,,15
#if defined(OS_LINUX)
	.global	kernel_sgemm_nt_4x4_lib4
	.type	kernel_sgemm_nt_4x4_lib4, %function
kernel_sgemm_nt_4x4_lib4:
#elif defined(OS_MAC)
	.global	kernel_sgemm_nt_4x4_lib4
_kernel_sgemm_nt_4x4_lib4:
#endif

	// prologue

	// save GP registers
	stmdb	sp!, {r4 - r10, fp, lr} // save registers
	add		fp, sp, #36 // fp to old sp position

	// save FP registers
	fstmfdd	sp!, {d8-d15}



	// zero accumulation registers
	vldr	d8, .LC00
	vldr	d9, .LC00+8
	vmov	q5, q4
	vmov	q6, q4
	vmov	q7, q4



	// call inner kernel dgemm nt
	mov		r4, r0 // kmax
	mov		r5, r2 // A
	mov		r6, r3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl	inner_kernel_gemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	bl	_inner_kernel_gemm_add_nt_4x4_lib4
#endif
#endif



	// call inner blend for generic alpha and beta
	mov		r4, r1 // alpha
	ldr		r5, [fp, #0] // beta
	ldr		r6, [fp, #4] // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	bl _inner_scale_ab_4x4_lib4
#endif
#endif



	// store n
	ldr		r4, [fp, #8] // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_store_4x4_lib4
#elif defined(OS_MAC)
	bl _inner_store_4x4_lib4
#endif
#endif



	// epilogue

	// load FP registers
	fldmfdd	sp!, {d8-d15}

	// load GP registers and return
//	ldmia	sp!, {r4 - r10, fp, lr} // load registers
//	mov		pc, lr // return
	ldmia	sp!, {r4 - r10, fp, pc} // load registers and return

#if defined(OS_LINUX)
	.size	kernel_sgemm_nt_4x4_lib4, .-kernel_sgemm_nt_4x4_lib4
#endif






