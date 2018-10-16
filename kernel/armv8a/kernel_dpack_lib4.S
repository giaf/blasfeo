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





//                              w0        x1         w2       x3         w4
// void kernel_dpack_nn_12_lib4(int kmax, double *A, int lda, double *C, int sdc)

	.align	4
	.globl kernel_dpack_nn_12_lib4
	.type kernel_dpack_nn_12_lib4, @function
kernel_dpack_nn_12_lib4:
	


	PROLOGUE



	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // lda
	lsl		w10, w10, #3 // 8*lda
	mov		x11, x3 // A
	mov		w12, w4 // sdc
	lsl		w12, w12, #5 // 32*sdc


	cmp		w8, #0
	ble		0f

	add		x13, x11, x12
	add		x14, x13, x12

	cmp		w8, #3
	ble		2f


1: // main loop
	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #0]
	ldp		q0, q1, [x9, #32]
	stp		q0, q1, [x13, #0]
	ldp		q0, q1, [x9, #64]
	add		x9, x9, x10
	stp		q0, q1, [x14, #0]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #32]
	ldp		q0, q1, [x9, #32]
	stp		q0, q1, [x13, #32]
	ldp		q0, q1, [x9, #64]
	add		x9, x9, x10
	stp		q0, q1, [x14, #32]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #64]
	ldp		q0, q1, [x9, #32]
	stp		q0, q1, [x13, #64]
	ldp		q0, q1, [x9, #64]
	add		x9, x9, x10
	stp		q0, q1, [x14, #64]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #96]
	ldp		q0, q1, [x9, #32]
	stp		q0, q1, [x13, #96]
	ldp		q0, q1, [x9, #64]
	add		x9, x9, x10
	stp		q0, q1, [x14, #96]

	sub		w8, w8, #4
	add		x11, x11, #128
	add		x13, x13, #128
	add		x14, x14, #128

	cmp		w8, #3
	bgt		1b


	cmp		w8, #0
	ble		0f


2: // clean-up loop
	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #0]
	ldp		q0, q1, [x9, #32]
	stp		q0, q1, [x13, #0]
	ldp		q0, q1, [x9, #64]
	add		x9, x9, x10
	stp		q0, q1, [x14, #0]

	sub		w8, w8, #1
	add		x11, x11, #32
	add		x13, x13, #32
	add		x14, x14, #32

	cmp		w8, #0
	bgt		2b


0: // return

	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dpack_nn_12_lib4, .-kernel_dpack_nn_12_lib4





//                             w0        x1         w2       x3         w4
// void kernel_dpack_nn_8_lib4(int kmax, double *A, int lda, double *C, int sdc)

	.align	4
	.globl kernel_dpack_nn_8_lib4
	.type kernel_dpack_nn_8_lib4, @function
kernel_dpack_nn_8_lib4:
	


	PROLOGUE



	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // lda
	lsl		w10, w10, #3 // 8*lda
	mov		x11, x3 // A
	mov		w12, w4 // sdc
	lsl		w12, w12, #5 // 32*sdc


	cmp		w8, #0
	ble		0f

	add		x13, x11, x12

	cmp		w8, #3
	ble		2f


1: // main loop
	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #0]
	ldp		q0, q1, [x9, #32]
	add		x9, x9, x10
	stp		q0, q1, [x13, #0]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #32]
	ldp		q0, q1, [x9, #32]
	add		x9, x9, x10
	stp		q0, q1, [x13, #32]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #64]
	ldp		q0, q1, [x9, #32]
	add		x9, x9, x10
	stp		q0, q1, [x13, #64]

	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #96]
	ldp		q0, q1, [x9, #32]
	add		x9, x9, x10
	stp		q0, q1, [x13, #96]

	sub		w8, w8, #4
	add		x11, x11, #128
	add		x13, x13, #128

	cmp		w8, #3
	bgt		1b


	cmp		w8, #0
	ble		0f


2: // clean-up loop
	ldp		q0, q1, [x9, #0]
	stp		q0, q1, [x11, #0]
	ldp		q0, q1, [x9, #32]
	add		x9, x9, x10
	stp		q0, q1, [x13, #0]

	sub		w8, w8, #1
	add		x11, x11, #32
	add		x13, x13, #32

	cmp		w8, #0
	bgt		2b


0: // return

	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dpack_nn_8_lib4, .-kernel_dpack_nn_8_lib4





//                             w0        x1         w2       x3
// void kernel_dpack_nn_4_lib4(int kmax, double *A, int lda, double *C)

	.align	4
	.globl kernel_dpack_nn_4_lib4
	.type kernel_dpack_nn_4_lib4, @function
kernel_dpack_nn_4_lib4:
	


	PROLOGUE



	mov		w8, w0 // kmax
	mov		x9, x1 // A
	mov		w10, w2 // A
	lsl		w10, w10, #3 // 32*sda
	mov		x11, x3 // A


	cmp		w8, #0
	ble		0f

	cmp		w8, #3
	ble		2f


1: // main loop
	ldp		q0, q1, [x9, #0]
	add		x9, x9, x10
	stp		q0, q1, [x11, #0]

	ldp		q0, q1, [x9, #0]
	add		x9, x9, x10
	stp		q0, q1, [x11, #32]

	ldp		q0, q1, [x9, #0]
	add		x9, x9, x10
	stp		q0, q1, [x11, #64]

	ldp		q0, q1, [x9, #0]
	add		x9, x9, x10
	stp		q0, q1, [x11, #96]

	sub		w8, w8, #4
	add		x11, x11, #128

	cmp		w8, #3
	bgt		1b


	cmp		w8, #0
	ble		0f


2: // clean-up loop
	ldp		q0, q1, [x9, #0]
	add		x9, x9, x10
	stp		q0, q1, [x11, #0]

	sub		w8, w8, #1
	add		x11, x11, #32

	cmp		w8, #0
	bgt		2b


0: // return

	EPILOGUE

	mov	x0, #0

	ret

	.size	kernel_dpack_nn_4_lib4, .-kernel_dpack_nn_4_lib4






