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
// r6   <- sda
// r7   <- B
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_gemm_add_nt_12x4_lib4, %function
inner_kernel_gemm_add_nt_12x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_gemm_add_nt_12x4_lib4:
#endif
#endif

	// early return
	cmp		r4, #0
	ble		2f // return

	add		r8, r5, r6 // A1
	add		r9, r8, r6 // A2

	// prefetch
	pld			[r5, #0] // A0
	pld			[r7, #0] // B
	pld			[r8, #0] // A1
	pld			[r9, #0] // A2

	// preload
	vld1.64		{d0, d1}, [r7:128] // B
	vld1.64		{d2, d3}, [r5:128] // A0
	vld1.64		{d4, d5}, [r8:128] // A1
//	vld1.64		{d6, d7}, [r9:128] // A2

	cmp		r4, #4
	ble		0f // consider clean up loop

	// prefetch
//	pld			[r5, #64] // A0
//	pld			[r7, #64] // B
//	pld			[r8, #64] // A1
//	pld			[r9, #64] // A2

	// main loop
1:
	
	// unroll 0
	vmla.f32	q4, q1, d0[0]
	vldr		d6, [r9, #0] // A2
	vmla.f32	q5, q1, d0[1]
	vldr		d7, [r9, #8] // A2
	vmla.f32	q6, q1, d1[0]
	pld			[r7, #64]
	vmla.f32	q7, q1, d1[1]
	vldr		d2, [r5, #16] // A0
	vmla.f32	q8, q2, d0[0]
	vldr		d3, [r5, #24] // A0
	vmla.f32	q9, q2, d0[1]
	pld			[r5, #64]
	vmla.f32	q10, q2, d1[0]
	pld			[r8, #64]
	vmla.f32	q11, q2, d1[1]
	vldr		d4, [r7, #16] // B
	vmla.f32	q12, q3, d0[0]
	vldr		d5, [r7, #24] // B
	vmla.f32	q13, q3, d0[1]
	vldr		d0, [r8, #16] // A1
	vmla.f32	q14, q3, d1[0]
	pld			[r9, #64]
	vmla.f32	q15, q3, d1[1]
	vldr		d1, [r8, #24] // A1

	// unroll 1
	vmla.f32	q4, q1, d4[0]
	vldr		d6, [r9, #16] // A2
	vmla.f32	q5, q1, d4[1]
	vldr		d7, [r9, #24] // A2
	vmla.f32	q6, q1, d5[0]
	sub		r4, r4, #4
	vmla.f32	q7, q1, d5[1]
	vldr		d2, [r5, #32] // A0
	vmla.f32	q8, q0, d4[0]
	vldr		d3, [r5, #40] // A0
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10, q0, d5[0]
	vmla.f32	q11, q0, d5[1]
	vldr		d0, [r7, #32] // B
	vmla.f32	q12, q3, d4[0]
	vldr		d1, [r7, #40] // B
	vmla.f32	q13, q3, d4[1]
	vldr		d4, [r8, #32] // A1
	vmla.f32	q14, q3, d5[0]
	vmla.f32	q15, q3, d5[1]
	vldr		d5, [r8, #40] // A1

	// unroll 2
	vmla.f32	q4, q1, d0[0]
	vldr		d6, [r9, #32] // A2
	vmla.f32	q5, q1, d0[1]
	vldr		d7, [r9, #40] // A2
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]
	vldr		d2, [r5, #48] // A0
	vmla.f32	q8, q2, d0[0]
	vldr		d3, [r5, #56] // A0
	vmla.f32	q9, q2, d0[1]
	vmla.f32	q10, q2, d1[0]
	add		r5, r5, #64
	vmla.f32	q11, q2, d1[1]
	vldr		d4, [r7, #48] // B
	vmla.f32	q12, q3, d0[0]
	vldr		d5, [r7, #56] // B
	vmla.f32	q13, q3, d0[1]
	vldr		d0, [r8, #48] // A1
	vmla.f32	q14, q3, d1[0]
	add		r7, r7, #64
	vmla.f32	q15, q3, d1[1]
	vldr		d1, [r8, #56] // A1

	// unroll 3
	vmla.f32	q4, q1, d4[0]
	vldr		d6, [r9, #48] // A2
	vmla.f32	q5, q1, d4[1]
	vldr		d7, [r9, #56] // A2
	vmla.f32	q6, q1, d5[0]
	add		r8, r8, #64
	vmla.f32	q7, q1, d5[1]
	vldr		d2, [r5, #0] // A0
	vmla.f32	q8, q0, d4[0]
	vldr		d3, [r5, #8] // A0
	vmla.f32	q9, q0, d4[1]
	add		r9, r9, #64
	vmla.f32	q10, q0, d5[0]
	cmp		r4, #4
	vmla.f32	q11, q0, d5[1]
	vldr		d0, [r7, #0] // B
	vmla.f32	q12, q3, d4[0]
	vldr		d1, [r7, #8] // B
	vmla.f32	q13, q3, d4[1]
	vldr		d4, [r8, #0] // A1
	vmla.f32	q14, q3, d5[0]
	vmla.f32	q15, q3, d5[1]
	vldr		d5, [r8, #8] // A1


	bgt		1b

0:

	cmp		r4, #3
	ble		4f


	// unroll 0
	vmla.f32	q4, q1, d0[0]
	vldr		d6, [r9, #0] // A2
	vmla.f32	q5, q1, d0[1]
	vldr		d7, [r9, #8] // A2
	vmla.f32	q6, q1, d1[0]
	pld			[r7, #64]
	vmla.f32	q7, q1, d1[1]
	vldr		d2, [r5, #16] // A0
	vmla.f32	q8, q2, d0[0]
	vldr		d3, [r5, #24] // A0
	vmla.f32	q9, q2, d0[1]
	pld			[r5, #64]
	vmla.f32	q10, q2, d1[0]
	pld			[r8, #64]
	vmla.f32	q11, q2, d1[1]
	vldr		d4, [r7, #16] // B
	vmla.f32	q12, q3, d0[0]
	vldr		d5, [r7, #24] // B
	vmla.f32	q13, q3, d0[1]
	vldr		d0, [r8, #16] // A1
	vmla.f32	q14, q3, d1[0]
	pld			[r9, #64]
	vmla.f32	q15, q3, d1[1]
	vldr		d1, [r8, #24] // A1

	// unroll 1
	vmla.f32	q4, q1, d4[0]
	vldr		d6, [r9, #16] // A2
	vmla.f32	q5, q1, d4[1]
	vldr		d7, [r9, #24] // A2
	vmla.f32	q6, q1, d5[0]
	sub		r4, r4, #4
	vmla.f32	q7, q1, d5[1]
	vldr		d2, [r5, #32] // A0
	vmla.f32	q8, q0, d4[0]
	vldr		d3, [r5, #40] // A0
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10, q0, d5[0]
	vmla.f32	q11, q0, d5[1]
	vldr		d0, [r7, #32] // B
	vmla.f32	q12, q3, d4[0]
	vldr		d1, [r7, #40] // B
	vmla.f32	q13, q3, d4[1]
	vldr		d4, [r8, #32] // A1
	vmla.f32	q14, q3, d5[0]
	vmla.f32	q15, q3, d5[1]
	vldr		d5, [r8, #40] // A1

	// unroll 2
	vmla.f32	q4, q1, d0[0]
	vldr		d6, [r9, #32] // A2
	vmla.f32	q5, q1, d0[1]
	vldr		d7, [r9, #40] // A2
	vmla.f32	q6, q1, d1[0]
	vmla.f32	q7, q1, d1[1]
	vldr		d2, [r5, #48] // A0
	vmla.f32	q8, q2, d0[0]
	vldr		d3, [r5, #56] // A0
	vmla.f32	q9, q2, d0[1]
	vmla.f32	q10, q2, d1[0]
	add		r5, r5, #64
	vmla.f32	q11, q2, d1[1]
	vldr		d4, [r7, #48] // B
	vmla.f32	q12, q3, d0[0]
	vldr		d5, [r7, #56] // B
	vmla.f32	q13, q3, d0[1]
	vldr		d0, [r8, #48] // A1
	vmla.f32	q14, q3, d1[0]
	add		r7, r7, #64
	vmla.f32	q15, q3, d1[1]
	vldr		d1, [r8, #56] // A1

	// unroll 3
	vmla.f32	q4, q1, d4[0]
	vldr		d6, [r9, #48] // A2
	vmla.f32	q5, q1, d4[1]
	vldr		d7, [r9, #56] // A2
	vmla.f32	q6, q1, d5[0]
	add		r9, r9, #64
	vmla.f32	q7, q1, d5[1]
//	vldr		d2, [r5, #0] // A0
	vmla.f32	q8, q0, d4[0]
//	vldr		d3, [r5, #8] // A0
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10, q0, d5[0]
	add		r8, r8, #64
	vmla.f32	q11, q0, d5[1]
//	vldr		d0, [r7, #0] // B
	vmla.f32	q12, q3, d4[0]
//	vldr		d1, [r7, #8] // B
	vmla.f32	q13, q3, d4[1]
//	vldr		d4, [r8, #0] // A1
	vmla.f32	q14, q3, d5[0]
	vmla.f32	q15, q3, d5[1]
//	vldr		d5, [r8, #8] // A1


	b		2f // return

4: // consider clean1-up loop

	cmp		r4, #0
	ble		2f // return


3: // clean1-up loop

	// unroll 0
	vld1.64		{d4, d5}, [r7:128]! // B
	vld1.64		{d0, d1}, [r5:128]! // A0
	vmla.f32	q4, q0, d4[0]
	sub		r4, r4, #1
	vmla.f32	q5, q0, d4[1]
	vmla.f32	q6, q0, d5[0]
	vmla.f32	q7, q0, d5[1]
	vld1.64		{d0, d1}, [r8:128]! // A1
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10, q0, d5[0]
	vmla.f32	q11, q0, d5[1]
	vld1.64		{d0, d1}, [r8:128]! // A1
	vmla.f32	q12, q0, d4[0]
	vmla.f32	q13, q0, d4[1]
	vmla.f32	q14, q0, d5[0]
	vmla.f32	q15, q0, d5[1]

	cmp		r4, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_kernel_gemm_add_nt_12x4_lib4, .-inner_kernel_gemm_add_nt_12x4_lib4
#endif
#endif





// subroutine
//
// input arguments:
// r4   <- alpha
// r5   <- beta
// r6   <- C
// r7   <- sdc
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_SCALE_AB_12X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_12x4_lib4, %function
inner_scale_ab_12x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_12x4_lib4:
#endif
#endif

	flds		s8, [r4, #0] // alpha

	vmul.f32	q4, q4, d4[0]
	flds		s9, [r5, #0] // beta
	vmul.f32	q5, q5, d4[0]
	flds		s10, .LC00 // 0.0
	vmul.f32	q6, q6, d4[0]
	vmul.f32	q7, q7, d4[0]
	fcmpes		s9, s10
	vmul.f32	q8, q8, d4[0]
	vmul.f32	q9, q9, d4[0]
	vmul.f32	q10, q10, d4[0]
	vmul.f32	q11, q11, d4[0]
	vmul.f32	q12, q12, d4[0]
	vmul.f32	q13, q13, d4[0]
	vmul.f32	q14, q14, d4[0]
	vmul.f32	q15, q15, d4[0]
	fmstat

	beq		0f // end

	add		r8, r6, r7
	add		r9, r8, r7

	vld1.64		{d0, d1, d2, d3}, [r6:128]!
	vmla.f32	q4, q0, d4[1]
	vmla.f32	q5, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r6:128]!
	vmla.f32	q6, q0, d4[1]
	vmla.f32	q7, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r8:128]!
	vmla.f32	q8, q0, d4[1]
	vmla.f32	q9, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r8:128]!
	vmla.f32	q10, q0, d4[1]
	vmla.f32	q11, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r9:128]!
	vmla.f32	q12, q0, d4[1]
	vmla.f32	q13, q1, d4[1]
	vld1.64		{d0, d1, d2, d3}, [r9:128]!
	vmla.f32	q14, q0, d4[1]
	vmla.f32	q15, q1, d4[1]

0:

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_scale_ab_12x4_lib4, .-inner_scale_ab_12x4_lib4
#endif
#endif





// subroutine
//
// input arguments:
// r4   <- D
// r5   <- sdd
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_12X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_12x4_lib4, %function
inner_store_12x4_lib4:
#elif defined(OS_MAC)
_inner_store_12x4_lib4:
#endif
#endif

	add		r6, r4, r5
	add		r7, r6, r5

	vst1.64		{d8, d9, d10, d11}, [r4:128]!
	vst1.64		{d12, d13, d14, d15}, [r4:128]!
	vst1.64		{d16, d17, d18, d19}, [r6:128]!
	vst1.64		{d20, d21, d22, d23}, [r6:128]!
	vst1.64		{d24, d25, d26, d27}, [r7:128]!
	vst1.64		{d28, d29, d30, d31}, [r7:128]!

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_store_12x4_lib4, .-inner_store_12x4_lib4
#endif
#endif





// zero double word
	.align 3
.LC00: // { 0 }
	.word 0
	.word 0
	.word 0
	.word 0

//                                r0        r1             r2         r3       sp+0       sp+4          sp+8       sp+12    sp+16   sp+20
// void kernel_sgemm_nt_12x4_lib4(int kmax, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd)

//	.p2align 4,,15
#if defined(OS_LINUX)
	.global	kernel_sgemm_nt_12x4_lib4
	.type	kernel_sgemm_nt_12x4_lib4, %function
kernel_sgemm_nt_12x4_lib4:
#elif defined(OS_MAC)
	.global	kernel_sgemm_nt_12x4_lib4
_kernel_sgemm_nt_12x4_lib4:
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
	vmov	q8, q4
	vmov	q9, q4
	vmov	q10, q4
	vmov	q11, q4
	vmov	q12, q4
	vmov	q13, q4
	vmov	q14, q4
	vmov	q15, q4



	// call inner kernel dgemm nt
	mov		r4, r0 // kmax
	mov		r5, r2 // A
	mov		r6, r3 // sda
	lsl		r6, r6, #4 // 4*sizeof(float)*sda
	ldr		r7, [fp, #0] // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	bl	inner_kernel_gemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	bl	_inner_kernel_gemm_add_nt_12x4_lib4
#endif
#endif



	// call inner blend for generic alpha and beta
	mov		r4, r1 // alpha
	ldr		r5, [fp, #4] // beta
	ldr		r6, [fp, #8] // C
	ldr		r7, [fp, #12] // sdc
	lsl		r7, r7, #4 // 4*sizeof(float)*sdc

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	bl _inner_scale_ab_12x4_lib4
#endif
#endif



	// store n
	ldr		r4, [fp, #16] // D
	ldr		r5, [fp, #20] // sdd
	lsl		r5, r5, #4 // 4*sizeof(float)*sdd

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_store_12x4_lib4
#elif defined(OS_MAC)
	bl _inner_store_12x4_lib4
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
	.size	kernel_sgemm_nt_12x4_lib4, .-kernel_sgemm_nt_12x4_lib4
#endif




