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
	.macro INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_4x4_lib4, %function
inner_kernel_dgemm_add_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_4x4_lib4:
#endif
#endif

	// early return
	cmp		r4, #0
	ble		2f // return

	// prefetch
	pld		[r5, #0]
	pld		[r6, #0]

	// preload A even
	fldd	d16, [r5, #0]
	fldd	d17, [r5, #8]
	fldd	d18, [r5, #16]
	fldd	d19, [r5, #24]

	// preload B even
	fldd	d20, [r6, #0]
	fldd	d21, [r6, #8]
	fldd	d22, [r6, #16]
	fldd	d23, [r6, #24]

	// preload A odd
	fldd	d24, [r5, #32]
	fldd	d25, [r5, #40]
	fldd	d26, [r5, #48]
	fldd	d27, [r5, #56]

	// preload B odd
	fldd	d28, [r6, #32]
	fldd	d29, [r6, #40]
	fldd	d30, [r6, #48]
	fldd	d31, [r6, #56]

	// prefetch
	pld		[r5, #64]
	pld		[r6, #64]

	cmp		r4, #4
	ble		0f // consider clean up loop

	// main loop
1:
	
	// prefetch
	pld		[r5, #128]
	pld		[r6, #128]

	// unroll 0
	fmacd	d0, d16, d20
	fmacd	d1, d17, d20
	fmacd	d2, d18, d20
	fmacd	d3, d19, d20
	fldd	d20, [r6, #64] // B

	fmacd	d4, d16, d21
	fmacd	d5, d17, d21
	fmacd	d6, d18, d21
	fmacd	d7, d19, d21
	fldd	d21, [r6, #72] // B

	fmacd	d8, d16, d22
	fmacd	d9, d17, d22
	fmacd	d10, d18, d22
	fmacd	d11, d19, d22
	fldd	d22, [r6, #80] // B

	fmacd	d12, d16, d23
	fldd	d16, [r5, #64] // A
	fmacd	d13, d17, d23
	fldd	d17, [r5, #72] // A
	fmacd	d14, d18, d23
	fldd	d18, [r5, #80] // A
	fmacd	d15, d19, d23
	fldd	d19, [r5, #88] // A
	fldd	d23, [r6, #88] // B

	// unroll 1
	fmacd	d0, d24, d28
	fmacd	d1, d25, d28
	fmacd	d2, d26, d28
	fmacd	d3, d27, d28
	fldd	d28, [r6, #96] // B

	fmacd	d4, d24, d29
	fmacd	d5, d25, d29
	fmacd	d6, d26, d29
	fmacd	d7, d27, d29
	fldd	d29, [r6, #104] // B

	fmacd	d8, d24, d30
	fmacd	d9, d25, d30
	fmacd	d10, d26, d30
	fmacd	d11, d27, d30
	fldd	d30, [r6, #112] // B

	fmacd	d12, d24, d31
	fldd	d24, [r5, #96] // A
	fmacd	d13, d25, d31
	fldd	d25, [r5, #104] // A
	fmacd	d14, d26, d31
	fldd	d26, [r5, #112] // A
	fmacd	d15, d27, d31
	fldd	d27, [r5, #120] // A
	fldd	d31, [r6, #120] // B

	// prefetch
	pld		[r5, #192]
	pld		[r6, #192]

	add		r5, r5, #128
	add		r6, r6, #128

	// unroll 2
	fmacd	d0, d16, d20
	fmacd	d1, d17, d20
	fmacd	d2, d18, d20
	fmacd	d3, d19, d20
	fldd	d20, [r6, #0] // B

	fmacd	d4, d16, d21
	fmacd	d5, d17, d21
	fmacd	d6, d18, d21
	fmacd	d7, d19, d21
	fldd	d21, [r6, #8] // B

	fmacd	d8, d16, d22
	fmacd	d9, d17, d22
	fmacd	d10, d18, d22
	fmacd	d11, d19, d22
	fldd	d22, [r6, #16] // B

	fmacd	d12, d16, d23
	fldd	d16, [r5, #0] // A
	fmacd	d13, d17, d23
	fldd	d17, [r5, #8] // A
	fmacd	d14, d18, d23
	fldd	d18, [r5, #16] // A
	fmacd	d15, d19, d23
	fldd	d19, [r5, #24] // A
	fldd	d23, [r6, #24] // B

	// unroll 3
	fmacd	d0, d24, d28
	fmacd	d1, d25, d28
	fmacd	d2, d26, d28
	fmacd	d3, d27, d28
	fldd	d28, [r6, #32] // B

	fmacd	d4, d24, d29
	fmacd	d5, d25, d29
	fmacd	d6, d26, d29
	fmacd	d7, d27, d29
	fldd	d29, [r6, #40] // B

	fmacd	d8, d24, d30
	fmacd	d9, d25, d30
	fmacd	d10, d26, d30
	fmacd	d11, d27, d30
	fldd	d30, [r6, #48] // B

	fmacd	d12, d24, d31
	fldd	d24, [r5, #32] // A
	fmacd	d13, d25, d31
	fldd	d25, [r5, #40] // A
	fmacd	d14, d26, d31
	fldd	d26, [r5, #48] // A
	fmacd	d15, d27, d31
	fldd	d27, [r5, #56] // A
	fldd	d31, [r6, #56] // B

	sub		r4, r4, #4
	cmp		r4, #4
	bgt		1b

0:

	cmp		r4, #3
	ble		4f

	// unroll 0
	fmacd	d0, d16, d20
	fmacd	d1, d17, d20
	fmacd	d2, d18, d20
	fmacd	d3, d19, d20
	fldd	d20, [r6, #64] // B

	fmacd	d4, d16, d21
	fmacd	d5, d17, d21
	fmacd	d6, d18, d21
	fmacd	d7, d19, d21
	fldd	d21, [r6, #72] // B

	fmacd	d8, d16, d22
	fmacd	d9, d17, d22
	fmacd	d10, d18, d22
	fmacd	d11, d19, d22
	fldd	d22, [r6, #80] // B

	fmacd	d12, d16, d23
	fldd	d16, [r5, #64] // A
	fmacd	d13, d17, d23
	fldd	d17, [r5, #72] // A
	fmacd	d14, d18, d23
	fldd	d18, [r5, #80] // A
	fmacd	d15, d19, d23
	fldd	d19, [r5, #88] // A
	fldd	d23, [r6, #88] // B

	// unroll 1
	fmacd	d0, d24, d28
	fmacd	d1, d25, d28
	fmacd	d2, d26, d28
	fmacd	d3, d27, d28
	fldd	d28, [r6, #96] // B

	fmacd	d4, d24, d29
	fmacd	d5, d25, d29
	fmacd	d6, d26, d29
	fmacd	d7, d27, d29
	fldd	d29, [r6, #104] // B

	fmacd	d8, d24, d30
	fmacd	d9, d25, d30
	fmacd	d10, d26, d30
	fmacd	d11, d27, d30
	fldd	d30, [r6, #112] // B

	fmacd	d12, d24, d31
	fldd	d24, [r5, #96] // A
	fmacd	d13, d25, d31
	fldd	d25, [r5, #104] // A
	fmacd	d14, d26, d31
	fldd	d26, [r5, #112] // A
	fmacd	d15, d27, d31
	fldd	d27, [r5, #120] // A
	fldd	d31, [r6, #120] // B

	add		r5, r5, #128
	add		r6, r6, #128

	// unroll 2
	fmacd	d0, d16, d20
	fmacd	d1, d17, d20
	fmacd	d2, d18, d20
	fmacd	d3, d19, d20
//	fldd	d20, [r6, #0] // B

	fmacd	d4, d16, d21
	fmacd	d5, d17, d21
	fmacd	d6, d18, d21
	fmacd	d7, d19, d21
//	fldd	d21, [r6, #8] // B

	fmacd	d8, d16, d22
	fmacd	d9, d17, d22
	fmacd	d10, d18, d22
	fmacd	d11, d19, d22
//	fldd	d22, [r6, #16] // B

	fmacd	d12, d16, d23
//	fldd	d16, [r5, #0] // A
	fmacd	d13, d17, d23
//	fldd	d17, [r5, #8] // A
	fmacd	d14, d18, d23
//	fldd	d18, [r5, #16] // A
	fmacd	d15, d19, d23
//	fldd	d19, [r5, #24] // A
//	fldd	d23, [r6, #24] // B

	// unroll 3
	fmacd	d0, d24, d28
	fmacd	d1, d25, d28
	fmacd	d2, d26, d28
	fmacd	d3, d27, d28
//	fldd	d28, [r6, #32] // B

	fmacd	d4, d24, d29
	fmacd	d5, d25, d29
	fmacd	d6, d26, d29
	fmacd	d7, d27, d29
//	fldd	d29, [r6, #40] // B

	fmacd	d8, d24, d30
	fmacd	d9, d25, d30
	fmacd	d10, d26, d30
	fmacd	d11, d27, d30
//	fldd	d30, [r6, #48] // B

	fmacd	d12, d24, d31
//	fldd	d24, [r5, #32] // A
	fmacd	d13, d25, d31
//	fldd	d25, [r5, #40] // A
	fmacd	d14, d26, d31
//	fldd	d26, [r5, #48] // A
	fmacd	d15, d27, d31
//	fldd	d27, [r5, #56] // A
//	fldd	d31, [r6, #56] // B

	sub		r4, r4, #4

	b		2f // return

4: // consider clean1-up loop

	cmp		r4, #0
	ble		2f // return

3: // clean1-up loop

	fldd	d16, [r5, #0] // A
	fldd	d17, [r5, #8] // A
	fldd	d18, [r5, #16] // A
	fldd	d19, [r5, #24] // A

	fldd	d20, [r6, #0] // B
	fmacd	d0, d16, d20
	fmacd	d1, d17, d20
	fmacd	d2, d18, d20
	fmacd	d3, d19, d20

	fldd	d21, [r6, #8] // B
	fmacd	d4, d16, d21
	fmacd	d5, d17, d21
	fmacd	d6, d18, d21
	fmacd	d7, d19, d21

	fldd	d22, [r6, #16] // B
	fmacd	d8, d16, d22
	fmacd	d9, d17, d22
	fmacd	d10, d18, d22
	fmacd	d11, d19, d22

	fldd	d23, [r6, #24] // B
	fmacd	d12, d16, d23
	fmacd	d13, d17, d23
	fmacd	d14, d18, d23
	fmacd	d15, d19, d23

	add		r5, r5, #32
	add		r6, r6, #32

	sub		r4, r4, #1
	cmp		r4, #0
	bgt		3b

2: // return

	
#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4
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
	.macro INNER_BLEND_SCALE_AB_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_4x4_lib4, %function
inner_blend_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_4x4_lib4:
#endif
#endif

	fldd	d16, [r4, #0] // alpha

	fmuld	d0, d0, d16
	fmuld	d1, d1, d16
	fmuld	d2, d2, d16
	fmuld	d3, d3, d16

	fmuld	d4, d4, d16
	fmuld	d5, d5, d16
	fmuld	d6, d6, d16
	fmuld	d7, d7, d16

	fmuld	d8, d8, d16
	fmuld	d9, d9, d16
	fmuld	d10, d10, d16
	fmuld	d11, d11, d16

	fmuld	d12, d12, d16
	fmuld	d13, d13, d16
	fmuld	d14, d14, d16
	fmuld	d15, d15, d16

	fldd	d16, [r5, #0] // beta

	fldd	d17, [r6, #0] // C
	fmacd	d0, d16, d17
	fldd	d17, [r6, #8] // C
	fmacd	d1, d16, d17
	fldd	d17, [r6, #16] // C
	fmacd	d2, d16, d17
	fldd	d17, [r6, #24] // C
	fmacd	d3, d16, d17

	fldd	d17, [r6, #32] // C
	fmacd	d4, d16, d17
	fldd	d17, [r6, #40] // C
	fmacd	d5, d16, d17
	fldd	d17, [r6, #48] // C
	fmacd	d6, d16, d17
	fldd	d17, [r6, #56] // C
	fmacd	d7, d16, d17

	fldd	d17, [r6, #64] // C
	fmacd	d8, d16, d17
	fldd	d17, [r6, #72] // C
	fmacd	d9, d16, d17
	fldd	d17, [r6, #80] // C
	fmacd	d10, d16, d17
	fldd	d17, [r6, #88] // C
	fmacd	d11, d16, d17

	fldd	d17, [r6, #96] // C
	fmacd	d12, d16, d17
	fldd	d17, [r6, #104] // C
	fmacd	d13, d16, d17
	fldd	d17, [r6, #112] // C
	fmacd	d14, d16, d17
	fldd	d17, [r6, #120] // C
	fmacd	d15, d16, d17


#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_4x4_lib4, .-inner_blend_scale_ab_4x4_lib4
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

	fstd	d0, [r4, #0]
	fstd	d1, [r4, #8]
	fstd	d2, [r4, #16]
	fstd	d3, [r4, #24]

	fstd	d4, [r4, #32]
	fstd	d5, [r4, #40]
	fstd	d6, [r4, #48]
	fstd	d7, [r4, #56]

	fstd	d8, [r4, #64]
	fstd	d9, [r4, #72]
	fstd	d10, [r4, #80]
	fstd	d11, [r4, #88]

	fstd	d12, [r4, #96]
	fstd	d13, [r4, #104]
	fstd	d14, [r4, #112]
	fstd	d15, [r4, #120]

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif
#endif





// subroutine
//
// input arguments:
// r4   <- D
//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_STORE_L_4X4_LIB4
#else
//	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_4x4_lib4, %function
inner_store_l_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_l_4x4_lib4:
#endif
#endif

	fstd	d0, [r4, #0]
	fstd	d1, [r4, #8]
	fstd	d2, [r4, #16]
	fstd	d3, [r4, #24]

//	fstd	d4, [r4, #32]
	fstd	d5, [r4, #40]
	fstd	d6, [r4, #48]
	fstd	d7, [r4, #56]

//	fstd	d8, [r4, #64]
//	fstd	d9, [r4, #72]
	fstd	d10, [r4, #80]
	fstd	d11, [r4, #88]

//	fstd	d12, [r4, #96]
//	fstd	d13, [r4, #104]
//	fstd	d14, [r4, #112]
	fstd	d15, [r4, #120]

#if MACRO_LEVEL>=2
	.endm
#else
	mov		pc, lr // return

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_lib4, .-inner_store_l_4x4_lib4
#endif
#endif





//                               r0        r1             r2         r3         sp+0          sp+4       sp+8
// void kernel_dgemm_nt_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D)

//	.p2align 4,,15
#if defined(OS_LINUX)
	.global	kernel_dgemm_nt_4x4_lib4
	.type	kernel_dgemm_nt_4x4_lib4, %function
kernel_dgemm_nt_4x4_lib4:
#elif defined(OS_MAC)
	.global	kernel_dgemm_nt_4x4_lib4
_kernel_dgemm_nt_4x4_lib4:
#endif

	// prologue

	// save GP registers
	stmdb	sp!, {r4 - r10, fp, lr} // save registers
	add		fp, sp, #36 // fp to old sp position

	// save FP registers
	fstmfdd	sp!, {d8-d15}



	// zero accumulation registers
	fldd	d0, .LC00
	fcpyd	d1, d0
	fcpyd	d2, d0
	fcpyd	d3, d0
	fcpyd	d4, d0
	fcpyd	d5, d0
	fcpyd	d6, d0
	fcpyd	d7, d0
	fcpyd	d8, d0
	fcpyd	d9, d0
	fcpyd	d10, d0
	fcpyd	d11, d0
	fcpyd	d12, d0
	fcpyd	d13, d0
	fcpyd	d14, d0
	fcpyd	d15, d0



	// call inner kernel dgemm nt
	mov		r4, r0 // kmax
	mov		r5, r2 // A
	mov		r6, r3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl	inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	bl	_inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif



	// call inner blend for generic alpha and beta
	mov		r4, r1 // alpha
	ldr		r5, [fp, #0] // beta
	ldr		r6, [fp, #4] // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	bl _inner_blend_scale_ab_4x4_lib4
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
	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4
#endif

	.align 3
.LC00: // { 0 }
	.word 0
	.word 0





//                                 r0        r1             r2         r3         sp+0          sp+4       sp+8
// void kernel_dsyrk_nt_l_4x4_lib4(int kmax, double *alpha, double *A, double *B, double *beta, double *C, double *D);

//	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_lib4
	.type kernel_dsyrk_nt_l_4x4_lib4, %function
kernel_dsyrk_nt_l_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_lib4
_kernel_dsyrk_nt_l_4x4_lib4:
#endif

	// prologue

	// save GP registers
	stmdb	sp!, {r4 - r10, fp, lr} // save registers
	add		fp, sp, #36 // fp to old sp position

	// save FP registers
	fstmfdd	sp!, {d8-d15}



	// zero accumulation registers
	fldd	d0, .LC01
	fcpyd	d1, d0
	fcpyd	d2, d0
	fcpyd	d3, d0
	fcpyd	d4, d0
	fcpyd	d5, d0
	fcpyd	d6, d0
	fcpyd	d7, d0
	fcpyd	d8, d0
	fcpyd	d9, d0
	fcpyd	d10, d0
	fcpyd	d11, d0
	fcpyd	d12, d0
	fcpyd	d13, d0
	fcpyd	d14, d0
	fcpyd	d15, d0



	// call inner kernel dgemm nt
	mov		r4, r0 // kmax
	mov		r5, r2 // A
	mov		r6, r3 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl	inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	bl	_inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif



	// call inner blend for generic alpha and beta
	mov		r4, r1 // alpha
	ldr		r5, [fp, #0] // beta
	ldr		r6, [fp, #4] // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	bl _inner_blend_scale_ab_4x4_lib4
#endif
#endif



	// store n
	ldr		r4, [fp, #8] // D

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	bl inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	bl _inner_store_l_4x4_lib4
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
	.size	kernel_dsyrk_nt_l_4x4_lib4, .-kernel_dsyrk_nt_l_4x4_lib4
#endif

	.align 3
.LC01: // { 0 }
	.word 0
	.word 0




