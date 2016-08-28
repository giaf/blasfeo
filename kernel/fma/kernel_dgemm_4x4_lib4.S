/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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

#define STACKSIZE 96

//.LSTR:
//	.string	"\n%d %d %d\n"

	.text

// common inner routine with file scope
//
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L002 // return

	// prefetch

	cmpl	$4, %r10d
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		8(%r12), %xmm12 // B[1]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	subl	$4, %r10d


	// unroll 1
	vmovapd 		32(%r11), %xmm8 // A[4]
	vmovapd 		48(%r11), %xmm9 // A[6]

	vmovddup		32(%r12), %xmm12 // B[4]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		40(%r12), %xmm12 // B[5]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		48(%r12), %xmm12 // B[6]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		56(%r12), %xmm12 // B[7]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	// unroll 2
	vmovapd 		64(%r11), %xmm8 // A[8]
	vmovapd 		80(%r11), %xmm9 // A[10]

	vmovddup		64(%r12), %xmm12 // B[8]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		72(%r12), %xmm12 // B[9]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		80(%r12), %xmm12 // B[10]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		88(%r12), %xmm12 // B[11]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	// unroll 3
	vmovapd 		96(%r11), %xmm8 // A[12]
	vmovapd 		112(%r11), %xmm9 // A[14]

	vmovddup		96(%r12), %xmm12 // B[12]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		104(%r12), %xmm12 // B[13]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		112(%r12), %xmm12 // B[14]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		120(%r12), %xmm12 // B[15]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	addq	$128, %r11
	addq	$128, %r12


	cmpl	$4, %r10d


	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L004 // clean1


	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		8(%r12), %xmm12 // B[1]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	// unroll 1
	vmovapd 		32(%r11), %xmm8 // A[4]
	vmovapd 		48(%r11), %xmm9 // A[6]

	vmovddup		32(%r12), %xmm12 // B[4]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		40(%r12), %xmm12 // B[5]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		48(%r12), %xmm12 // B[6]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		56(%r12), %xmm12 // B[7]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	// unroll 2
	vmovapd 		64(%r11), %xmm8 // A[8]
	vmovapd 		80(%r11), %xmm9 // A[10]

	vmovddup		64(%r12), %xmm12 // B[8]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		72(%r12), %xmm12 // B[9]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		80(%r12), %xmm12 // B[10]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		88(%r12), %xmm12 // B[11]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	// unroll 3
	vmovapd 		96(%r11), %xmm8 // A[12]
	vmovapd 		112(%r11), %xmm9 // A[14]

	vmovddup		96(%r12), %xmm12 // B[12]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovddup		104(%r12), %xmm12 // B[13]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		112(%r12), %xmm12 // B[14]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		120(%r12), %xmm12 // B[15]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7


	ret


.L004: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	subl	$1, %r10d

	vmovddup		8(%r12), %xmm12 // B[1]
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd	%xmm9, %xmm12, %xmm7

	addq	$32, %r11
	addq	$32, %r12

	cmpl	$0, %r10d

	jg		.L003 // clean up loop 


.L002: // return

	ret

	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4





// common inner routine with file scope
//
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_sub_nt_4x4_lib4, @function
inner_kernel_dgemm_sub_nt_4x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L025 // return

	// prefetch

	cmpl	$4, %r10d
	jle		.L023 // consider clean-up loop

	// main loop
	.p2align 3
.L024: // main loop
	
	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		8(%r12), %xmm12 // B[1]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	subl	$4, %r10d


	// unroll 1
	vmovapd 		32(%r11), %xmm8 // A[4]
	vmovapd 		48(%r11), %xmm9 // A[6]

	vmovddup		32(%r12), %xmm12 // B[4]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		40(%r12), %xmm12 // B[5]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		48(%r12), %xmm12 // B[6]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		56(%r12), %xmm12 // B[7]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	// unroll 2
	vmovapd 		64(%r11), %xmm8 // A[8]
	vmovapd 		80(%r11), %xmm9 // A[10]

	vmovddup		64(%r12), %xmm12 // B[8]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		72(%r12), %xmm12 // B[9]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		80(%r12), %xmm12 // B[10]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		88(%r12), %xmm12 // B[11]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	// unroll 3
	vmovapd 		96(%r11), %xmm8 // A[12]
	vmovapd 		112(%r11), %xmm9 // A[14]

	vmovddup		96(%r12), %xmm12 // B[12]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		104(%r12), %xmm12 // B[13]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		112(%r12), %xmm12 // B[14]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		120(%r12), %xmm12 // B[15]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	addq	$128, %r12
	addq	$128, %r11


	cmpl	$4, %r10d


	jg		.L024 // main loop 


.L023: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L027 // clean1


	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		8(%r12), %xmm12 // B[1]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	// unroll 1
	vmovapd 		32(%r11), %xmm8 // A[4]
	vmovapd 		48(%r11), %xmm9 // A[6]

	vmovddup		32(%r12), %xmm12 // B[4]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		40(%r12), %xmm12 // B[5]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		48(%r12), %xmm12 // B[6]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		56(%r12), %xmm12 // B[7]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	// unroll 2
	vmovapd 		64(%r11), %xmm8 // A[8]
	vmovapd 		80(%r11), %xmm9 // A[10]

	vmovddup		64(%r12), %xmm12 // B[8]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		72(%r12), %xmm12 // B[9]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		80(%r12), %xmm12 // B[10]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		88(%r12), %xmm12 // B[11]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	// unroll 3
	vmovapd 		96(%r11), %xmm8 // A[12]
	vmovapd 		112(%r11), %xmm9 // A[14]

	vmovddup		96(%r12), %xmm12 // B[12]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovddup		104(%r12), %xmm12 // B[13]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		112(%r12), %xmm12 // B[14]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		120(%r12), %xmm12 // B[15]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7


	ret


.L027: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L025 // return

	// clean-up loop
.L026: // clean up loop
	
	// unroll 0
	vmovapd 		0(%r11), %xmm8 // A[0]
	vmovapd 		16(%r11), %xmm9 // A[2]

	vmovddup		0(%r12), %xmm12 // B[0]
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	subl	$1, %r10d

	vmovddup		8(%r12), %xmm12 // B[1]
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovddup		16(%r12), %xmm12 // B[2]
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovddup		24(%r12), %xmm12 // B[3]
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7

	addq	$32, %r12
	addq	$32, %r11

	cmpl	$0, %r10d

	jg		.L026 // clean up loop 


.L025: // return

	ret

	.size	inner_kernel_dgemm_sub_nt_4x4_lib4, .-inner_kernel_dgemm_sub_nt_4x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_4x4_lib4, @function
inner_edge_dtrmm_add_nt_lu_4x4_lib4:
	
	vmovapd			0(%r10), %xmm8
	vmovapd			16(%r10), %xmm9
	vmovddup		0(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1

	vmovapd			32(%r10), %xmm8
	vmovapd			48(%r10), %xmm9
	vmovddup		32(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	vmovddup		40(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3

	vmovapd			64(%r10), %xmm8
	vmovapd			80(%r10), %xmm9
	vmovddup		64(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	vmovddup		72(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3
	vmovddup		80(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5

	vmovapd			96(%r10), %xmm8
	vmovapd			112(%r10), %xmm9
	vmovddup		96(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	vmovddup		104(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3
	vmovddup		112(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5
	addq			$128, %r10
	vmovddup		120(%r11), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7
	addq			$128, %r11

	ret

	.size	inner_edge_dtrmm_add_nt_lu_4x4_lib4, .-inner_edge_dtrmm_add_nt_lu_4x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4, @function
inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4:
	
	vmovapd			0(%r11), %xmm8
	vmovapd			16(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		0(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %xmm8
	vmovapd			16(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		0(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	addq			$32, %r11
	vmovddup		8(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %xmm8
	vmovapd			16(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		0(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	vmovddup		8(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3
	addq			$32, %r11
	vmovddup		16(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %xmm8
	vmovapd			16(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		0(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm0
	vfmadd231pd		%xmm9, %xmm12, %xmm1
	vmovddup		8(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm2
	vfmadd231pd		%xmm9, %xmm12, %xmm3
	vmovddup		16(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm4
	vfmadd231pd		%xmm9, %xmm12, %xmm5
	addq			$32, %r11
	vmovddup		24(%r12), %xmm12
	vfmadd231pd		%xmm8, %xmm12, %xmm6
	vfmadd231pd		%xmm9, %xmm12, %xmm7
	addq			$32, %r12

.L016:
	ret

	.size	inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4, .-inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_4x4_lib4, @function
inner_edge_dtrmm_sub_nt_lu_4x4_lib4:

	vmovapd			0(%r10), %xmm8
	vmovapd			16(%r10), %xmm9
	vmovddup		0(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1

	vmovapd			32(%r10), %xmm8
	vmovapd			48(%r10), %xmm9
	vmovddup		32(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	vmovddup		40(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3

	vmovapd			64(%r10), %xmm8
	vmovapd			80(%r10), %xmm9
	vmovddup		64(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	vmovddup		72(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3
	vmovddup		80(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5

	vmovapd			96(%r10), %xmm8
	vmovapd			112(%r10), %xmm9
	vmovddup		96(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	vmovddup		104(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3
	vmovddup		112(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5
	addq			$128, %r10
	vmovddup		120(%r11), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7
	addq			$128, %r11

	ret

	.size	inner_edge_dtrmm_sub_nt_lu_4x4_lib4, .-inner_edge_dtrmm_sub_nt_lu_4x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4, @function
inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4:
	
	vmovapd			0(%r11), %xmm8
	vmovapd			16(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		0(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			32(%r11), %xmm8
	vmovapd			48(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		32(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	addq			$32, %r11
	vmovddup		40(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			64(%r11), %xmm8
	vmovapd			80(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		64(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	vmovddup		72(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3
	addq			$32, %r11
	vmovddup		80(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			96(%r11), %xmm8
	vmovapd			112(%r11), %xmm9
	subl			$1, %r10d
	vmovddup		96(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm0
	vfnmadd231pd	%xmm9, %xmm12, %xmm1
	vmovddup		104(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm2
	vfnmadd231pd	%xmm9, %xmm12, %xmm3
	vmovddup		112(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm4
	vfnmadd231pd	%xmm9, %xmm12, %xmm5
	addq			$32, %r11
	vmovddup		120(%r12), %xmm12
	vfnmadd231pd	%xmm8, %xmm12, %xmm6
	vfnmadd231pd	%xmm9, %xmm12, %xmm7
	addq			$32, %r12

.L017:
	ret

	.size	inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4, .-inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4





// common inner routine with file scope
//
// blender for tc==n and td==n
//
// input arguments:
// r10d <- alg
// r11   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

// output arguments:
// r10d <- alg
// r11   <- C
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nn_4x4_lib4, @function
inner_blender_nn_4x4_lib4:
	

	// tc==n

	cmpl	$0, %r10d // alg
	je		.L005 // return

	cmpl	$1, %r10d // alg
	jne		.L006 // alg==-1

	// alg==1
	vmovapd		0(%r11), %xmm15
	vaddpd		%xmm0, %xmm15, %xmm0
	vmovapd		16(%r11), %xmm15
	vaddpd		%xmm1, %xmm15, %xmm1
	vmovapd		32(%r11), %xmm15
	vaddpd		%xmm2, %xmm15, %xmm2
	vmovapd		48(%r11), %xmm15
	vaddpd		%xmm3, %xmm15, %xmm3
	vmovapd		64(%r11), %xmm15
	vaddpd		%xmm4, %xmm15, %xmm4
	vmovapd		80(%r11), %xmm15
	vaddpd		%xmm5, %xmm15, %xmm5
	vmovapd		96(%r11), %xmm15
	vaddpd		%xmm6, %xmm15, %xmm6
	vmovapd		112(%r11), %xmm15
	vaddpd		%xmm7, %xmm15, %xmm7

	ret // return

.L006: // alg==-1

	// alg==-1
	vmovapd		0(%r11), %xmm15
	vsubpd		%xmm0, %xmm15, %xmm0
	vmovapd		16(%r11), %xmm15
	vsubpd		%xmm1, %xmm15, %xmm1
	vmovapd		32(%r11), %xmm15
	vsubpd		%xmm2, %xmm15, %xmm2
	vmovapd		48(%r11), %xmm15
	vsubpd		%xmm3, %xmm15, %xmm3
	vmovapd		64(%r11), %xmm15
	vsubpd		%xmm4, %xmm15, %xmm4
	vmovapd		80(%r11), %xmm15
	vsubpd		%xmm5, %xmm15, %xmm5
	vmovapd		96(%r11), %xmm15
	vsubpd		%xmm6, %xmm15, %xmm6
	vmovapd		112(%r11), %xmm15
	vsubpd		%xmm7, %xmm15, %xmm7

.L005: // return
	
	ret

	.size	inner_blender_nn_4x4_lib4, .-inner_blender_nn_4x4_lib4





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10  <- inv_diag_E
// r11d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// r11d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_edge_dpotrf_4x4_vs_lib4, @function
inner_edge_dpotrf_4x4_vs_lib4:
	
	vxorpd			%xmm15, %xmm15, %xmm15 // 0.0
	vmovsd			.LC04(%rip), %xmm14 // 1.0

	vmovsd			%xmm0, %xmm0, %xmm13
	vucomisd		%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe				.L029
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
.L030:
	cmpl			$2, %r11d
	vmovsd			%xmm13, 0(%r10)
	vmovddup		%xmm13, %xmm13
	vmulpd			%xmm0, %xmm13, %xmm0
	vmulpd			%xmm1, %xmm13, %xmm1

	jl				.L028 // ret

	vpermilpd		$0x3, %xmm0, %xmm13
	vfnmadd231pd	%xmm0, %xmm13, %xmm2
	vfnmadd231pd	%xmm1, %xmm13, %xmm3
	vpermilpd		$0x3, %xmm2, %xmm13
	vucomisd		%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe				.L031
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
.L032:
	cmpl			$3, %r11d
	vmovsd			%xmm13, 8(%r10)
	vmovddup		%xmm13, %xmm13
	vmulpd			%xmm2, %xmm13, %xmm2
	vmulpd			%xmm3, %xmm13, %xmm3

	jl			.L028 // ret

	vpermilpd		$0x0, %xmm1, %xmm13
//	vfnmadd231pd	%xmm0, %xmm13, %xmm4
	vfnmadd231pd	%xmm1, %xmm13, %xmm5
	vpermilpd		$0x0, %xmm3, %xmm13
//	vfnmadd231pd	%xmm2, %xmm13, %xmm4
	vfnmadd231pd	%xmm3, %xmm13, %xmm5
	vmovaps			%xmm5, %xmm13
	vucomisd		%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe				.L033
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
.L034:
	cmpl			$4, %r11d
	vmovsd			%xmm13, 16(%r10)
	vmovddup		%xmm13, %xmm13
//	vmulpd			%xmm4, %xmm13, %xmm4
	vmulpd			%xmm5, %xmm13, %xmm5

	jl			.L028 // ret

	vpermilpd		$0x3, %xmm1, %xmm13
//	vfnmadd231pd	%xmm0, %xmm13, %xmm6
	vfnmadd231pd	%xmm1, %xmm13, %xmm7
	vpermilpd		$0x3, %xmm3, %xmm13
//	vfnmadd231pd	%xmm2, %xmm13, %xmm6
	vfnmadd231pd	%xmm3, %xmm13, %xmm7
	vpermilpd		$0x3, %xmm5, %xmm13
//	vfnmadd231pd	%xmm4, %xmm13, %xmm6
	vfnmadd231pd	%xmm5, %xmm13, %xmm7
	vpermilpd		$0x3, %xmm7, %xmm13
	vucomisd		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe				.L035
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
.L036:
	vmovsd			%xmm13, 24(%r10)
	vmovddup		%xmm13, %xmm13
//	vmulpd			%xmm6, %xmm13, %xmm6
	vmulpd			%xmm7, %xmm13, %xmm7

.L028:
	
	ret

.L029:
	vxorpd	%xmm13, %xmm13, %xmm13
	jmp		.L030

.L031:
	vxorpd	%xmm13, %xmm13, %xmm13
	jmp		.L032

.L033:
	vxorpd	%xmm13, %xmm13, %xmm13
	jmp		.L034

.L035:
	vxorpd	%xmm13, %xmm13, %xmm13
	jmp		.L036

	.size	inner_edge_dpotrf_4x4_vs_lib4, .-inner_edge_dpotrf_4x4_vs_lib4





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_4x4_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_lib4:
	
	vmovddup		0(%r11), %xmm13
	vmulpd			%xmm0, %xmm13, %xmm0
	vmulpd			%xmm1, %xmm13, %xmm1

	vmovddup		8(%r10), %xmm13
	vfnmadd231pd	%xmm0, %xmm13, %xmm2
	vfnmadd231pd	%xmm1, %xmm13, %xmm3
	vmovddup		8(%r11), %xmm13
	vmulpd			%xmm2, %xmm13, %xmm2
	vmulpd			%xmm3, %xmm13, %xmm3

	vmovddup		16(%r10), %xmm13
	vfnmadd231pd	%xmm0, %xmm13, %xmm4
	vfnmadd231pd	%xmm1, %xmm13, %xmm5
	vmovddup		48(%r10), %xmm13
	vfnmadd231pd	%xmm2, %xmm13, %xmm4
	vfnmadd231pd	%xmm3, %xmm13, %xmm5
	vmovddup		16(%r11), %xmm13
	vmulpd			%xmm4, %xmm13, %xmm4
	vmulpd			%xmm5, %xmm13, %xmm5

	vmovddup		24(%r10), %xmm13
	vfnmadd231pd	%xmm0, %xmm13, %xmm6
	vfnmadd231pd	%xmm1, %xmm13, %xmm7
	vmovddup		56(%r10), %xmm13
	vfnmadd231pd	%xmm2, %xmm13, %xmm6
	vfnmadd231pd	%xmm3, %xmm13, %xmm7
	vmovddup		88(%r10), %xmm13
	vfnmadd231pd	%xmm4, %xmm13, %xmm6
	vfnmadd231pd	%xmm5, %xmm13, %xmm7
	vmovddup		24(%r11), %xmm13
	vmulpd			%xmm6, %xmm13, %xmm6
	vmulpd			%xmm7, %xmm13, %xmm7

	ret

	.size	inner_edge_dtrsm_rlt_inv_4x4_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_lib4





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
	
	vmovddup		0(%r11), %xmm13
	cmpl			$2, %r12d
	vmulpd			%xmm0, %xmm13, %xmm0
	vmulpd			%xmm1, %xmm13, %xmm1

	jl				.L037 // ret

	vmovddup		8(%r10), %xmm13
	cmpl			$3, %r12d
	vfnmadd231pd	%xmm0, %xmm13, %xmm2
	vfnmadd231pd	%xmm1, %xmm13, %xmm3
	vmovddup		8(%r11), %xmm13
	vmulpd			%xmm2, %xmm13, %xmm2
	vmulpd			%xmm3, %xmm13, %xmm3

	jl				.L037 // ret

	vmovddup		16(%r10), %xmm13
	cmpl			$4, %r12d
	vfnmadd231pd	%xmm0, %xmm13, %xmm4
	vfnmadd231pd	%xmm1, %xmm13, %xmm5
	vmovddup		48(%r10), %xmm13
	vfnmadd231pd	%xmm2, %xmm13, %xmm4
	vfnmadd231pd	%xmm3, %xmm13, %xmm5
	vmovddup		16(%r11), %xmm13
	vmulpd			%xmm4, %xmm13, %xmm4
	vmulpd			%xmm5, %xmm13, %xmm5

	jl				.L037 // ret

	vmovddup		24(%r10), %xmm13
	vfnmadd231pd	%xmm0, %xmm13, %xmm6
	vfnmadd231pd	%xmm1, %xmm13, %xmm7
	vmovddup		56(%r10), %xmm13
	vfnmadd231pd	%xmm2, %xmm13, %xmm6
	vfnmadd231pd	%xmm3, %xmm13, %xmm7
	vmovddup		88(%r10), %xmm13
	vfnmadd231pd	%xmm4, %xmm13, %xmm6
	vfnmadd231pd	%xmm5, %xmm13, %xmm7
	vmovddup		24(%r11), %xmm13
	vmulpd			%xmm6, %xmm13, %xmm6
	vmulpd			%xmm7, %xmm13, %xmm7

.L037:
	
	ret

	.size	inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_vs_lib4





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10  <- D
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_store_4x4_lib4, @function
inner_store_4x4_lib4:
	
	vmovapd %xmm0,   0(%r10)
	vmovapd %xmm1,  16(%r10)
	vmovapd %xmm2,  32(%r10)
	vmovapd %xmm3,  48(%r10)
	vmovapd %xmm4,  64(%r10)
	vmovapd %xmm5,  80(%r10)
	vmovapd %xmm6,  96(%r10)
	vmovapd %xmm7, 112(%r10)
	
	ret

	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4





// TODO use blendv instead
// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_store_4x4_vs_lib4, @function
inner_store_4x4_vs_lib4:
	
	cmpl		$2, %r11d
	jg			.L038
	je			.L039

	// km==1
	cmpl		$2, %r12d
	vmovsd		%xmm0,  0(%r10)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmovsd		%xmm2, 32(%r10)
	jl			.L015 // end
	vmovsd		%xmm4, 64(%r10)
	je			.L015 // end
	vmovsd		%xmm6, 96(%r10)

	ret

.L039:
	// km==2
	cmpl		$2, %r12d
	vmovapd		%xmm0,  0(%r10)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmovapd		%xmm2, 32(%r10)
	jl			.L015 // end
	vmovapd		%xmm4, 64(%r10)
	je			.L015 // end
	vmovapd		%xmm6, 96(%r10)

	ret

.L038:
	cmpl		$3, %r11d
	jg			.L040

	// km==3
	cmpl		$2, %r12d
	vmovapd		%xmm0,   0(%r10)
	vmovsd		%xmm1,  16(%r10)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmovapd		%xmm2,  32(%r10)
	vmovsd		%xmm3,  48(%r10)
	jl			.L015 // end
	vmovapd		%xmm4,  64(%r10)
	vmovsd		%xmm5,  80(%r10)
	je			.L015 // end
	vmovapd		%xmm6,  96(%r10)
	vmovsd		%xmm7, 112(%r10)

	ret

.L040:
	// km==3
	cmpl		$2, %r12d
	vmovapd		%xmm0,   0(%r10)
	vmovapd		%xmm1,  16(%r10)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmovapd		%xmm2,  32(%r10)
	vmovapd		%xmm3,  48(%r10)
	jl			.L015 // end
	vmovapd		%xmm4,  64(%r10)
	vmovapd		%xmm5,  80(%r10)
	je			.L015 // end
	vmovapd		%xmm6,  96(%r10)
	vmovapd		%xmm7, 112(%r10)

.L015:
	ret

	.size	inner_store_4x4_vs_lib4, .-inner_store_4x4_vs_lib4





// common inner routine with file scope
//
// store n vs lower triangular
//
// input arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// xmm0  <- [d00 d10]
// xmm1  <- [d20 d30]
// xmm2  <- [d01 d11]
// xmm3  <- [d21 d31]
// xmm0  <- [d02 d12]
// xmm1  <- [d22 d32]
// xmm2  <- [d03 d13]
// xmm3  <- [d23 d33]
// xmm8  <- dirty
// xmm9  <- dirty
// xmm10 <- dirty
// xmm11 <- dirty
// xmm12 <- dirty
// xmm13 <- dirty
// xmm14 <- dirty
// xmm15 <- dirty

	.p2align 4,,15
	.type inner_store_l_4x4_vs_lib4, @function
inner_store_l_4x4_vs_lib4:
	
	cmpl		$2, %r11d
	jg			.L041
	je			.L042

	// km==1
	vmovsd		%xmm0,  0(%r10)

	ret

.L042:
	// km==2
	cmpl		$2, %r12d
	vmovapd		%xmm0,  0(%r10)
	jl			.L018 // end
	vmovsd		32(%r10), %xmm15
	vmovsd		%xmm15, %xmm2, %xmm2
	vmovapd		%xmm2, 32(%r10)

	ret

.L041:
	cmpl		$3, %r11d
	jg			.L043

	// km==3
	cmpl		$2, %r12d
	vmovapd		%xmm0,   0(%r10)
	vmovsd		%xmm1,  16(%r10)
	jl			.L018 // end
	cmpl		$3, %r12d
	vmovsd		32(%r10), %xmm15
	vmovsd		%xmm15, %xmm2, %xmm2
	vmovapd		%xmm2,  32(%r10)
	vmovsd		%xmm3,  48(%r10)
	jl			.L018 // end
//	vmovapd		%xmm4,  64(%r10)
	vmovsd		%xmm5,  80(%r10)

	ret

.L043:
	// km==3
	cmpl		$2, %r12d
	vmovapd		%xmm0,   0(%r10)
	vmovapd		%xmm1,  16(%r10)
	jl			.L018 // end
	cmpl		$3, %r12d
	vmovsd		32(%r10), %xmm15
	vmovsd		%xmm15, %xmm2, %xmm2
	vmovapd		%xmm2,  32(%r10)
	vmovapd		%xmm3,  48(%r10)
	jl			.L018 // end
//	vmovapd		%xmm4,  64(%r10)
	vmovapd		%xmm5,  80(%r10)
	je			.L018 // end
//	vmovapd		%xmm6,  96(%r10)
	vmovsd		112(%r10), %xmm15
	vmovsd		%xmm15, %xmm7, %xmm7
	vmovapd		%xmm7, 112(%r10)

.L018:
	ret

	.size	inner_store_l_4x4_vs_lib4, .-inner_store_l_4x4_vs_lib4





//                             rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dgemm_nt_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_nt_4x4_lib4
	.type kernel_dgemm_nt_4x4_lib4, @function
kernel_dgemm_nt_4x4_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nn_4x4_lib4


	// store n

	movq	%r9, %r10

	call inner_store_4x4_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4





//                                  rdi    rsi        rdx        rcx      r8         r9         rsp+8   rsp+16
// void kernel_dgemm_nt_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nt_4x4_vs_lib4
	.type kernel_dgemm_nt_4x4_vs_lib4, @function
kernel_dgemm_nt_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nn_4x4_lib4


	// store n

	movq	%r9, %r10 // D
	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	call inner_store_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dgemm_nt_4x4_vs_lib4, .-kernel_dgemm_nt_4x4_vs_lib4





//                                    rdi    rsi        rdx        rcx      r8         r9         rsp+8   rsp+16
// void kernel_dsyrk_nt_l_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_nt_l_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nn_4x4_lib4


	// store

	movq	%r9, %r10 // D
	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 
	
	call	inner_store_l_4x4_vs_lib4



	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dsyrk_nt_l_4x4_vs_lib4, .-kernel_dsyrk_nt_l_4x4_vs_lib4





//                                  edi    rsi        rdx        ecx      r8         r9         
// void kernel_dtrmm_nt_ru_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dtrmm_nt_ru_4x4_lib4
	.type kernel_dtrmm_nt_ru_4x4_lib4, @function
kernel_dtrmm_nt_ru_4x4_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d
	subl	$4, %r10d
	movq	%rsi, %r11
	addq	$128, %r11
	movq	%rdx, %r12
	addq	$128, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nn_4x4_lib4


	// initial triangle

	movq	%rsi, %r10
	movq	%rdx, %r11

	cmpl	$-1, %ecx // alg
	je		.L019


	call inner_edge_dtrmm_add_nt_lu_4x4_lib4

	jmp		.L020

.L019:

	call inner_edge_dtrmm_sub_nt_lu_4x4_lib4 

.L020:
	// store n

	movq	%r9, %r10

	call inner_store_4x4_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dtrmm_nt_ru_4x4_lib4, .-kernel_dtrmm_nt_ru_4x4_lib4





//                                     edi    rsi        rdx        ecx      r8         r9         rsp+8   rsp+16 
// void kernel_dtrmm_nt_ru_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
// it holds k==kn !!!!!

	.p2align 4,,15
	.globl kernel_dtrmm_nt_ru_4x4_vs_lib4
	.type kernel_dtrmm_nt_ru_4x4_vs_lib4, @function
kernel_dtrmm_nt_ru_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d
	subl	$4, %r10d
	movq	%rsi, %r11
	addq	$128, %r11
	movq	%rdx, %r12
	addq	$128, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nn_4x4_lib4


	// initial triangle

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	cmpl	$-1, %ecx // alg
	je		.L021


	call inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4

	jmp		.L022

.L021:

	call inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4 

.L022:
	// store n

	movq	%r9, %r10 // D
	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	call inner_store_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dtrmm_nt_ru_4x4_vs_lib4, .-kernel_dtrmm_nt_ru_4x4_vs_lib4





//                                     edi    rsi        rdx        ecx        r8         r9                  rsp+8   rsp+16
// void kernel_dpotrf_nt_l_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dpotrf_nt_l_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	$1, %r10d
	movq	%rcx, %r11

	call inner_blender_nn_4x4_lib4


	// factorization

	movq	%r9, %r10  // inv_diag_D 
	movl	STACKSIZE + 16(%rsp), %r11d // kn 

	call inner_edge_dpotrf_4x4_vs_lib4


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	call inner_store_l_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dpotrf_nt_l_4x4_vs_lib4





//                                           edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d  // alg
	movq	STACKSIZE + 16(%rsp), %r11   // C

	call inner_blender_nn_4x4_lib4 // TODO alg has to be applied only to the add part !!!!!


	// factorization

	movq	STACKSIZE + 32(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 48(%rsp), %r11d // kn 

	call inner_edge_dpotrf_4x4_vs_lib4


	// store

	movq	STACKSIZE + 24(%rsp), %r10  // D 
	movl	STACKSIZE + 40(%rsp), %r11d // km 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

	call inner_store_l_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4





//                                      edi    rsi        rdx        ecx        r8         r9         rsp+8  
// void kernel_dtrsm_nt_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	$1, %r10d
	movq	%rcx, %r11

	call inner_blender_nn_4x4_lib4


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 

	call inner_edge_dtrsm_rlt_inv_4x4_lib4


	// store

	movq	%r8, %r10 // D

	call inner_store_4x4_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_lib4





//                                            edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32    rsp+40                     
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	STACKSIZE + 8(%rsp),  %r10d  // alg 
	movq	STACKSIZE + 16(%rsp), %r11   // C

	call inner_blender_nn_4x4_lib4 // TODO alg has to be applied only to the add part !!!!!


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11  // inv_diag_E 

	call inner_edge_dtrsm_rlt_inv_4x4_lib4


	// store

	movq	STACKSIZE + 24(%rsp), %r10   // D

	call inner_store_4x4_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4





//                                         edi    rsi        rdx        ecx        r8       r9           rsp+8               rsp+16  rsp+24
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	$1, %r10d
	movq	%rcx, %r11

	call inner_blender_nn_4x4_lib4


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

	call inner_store_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib4





//                                               edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32     rsp+40              rsp+48  rsp+56
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4:
	
	// prologue

	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper


	// zero accumulation registers

	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm0, %xmm5
	vmovapd	%xmm0, %xmm6
	vmovapd	%xmm0, %xmm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

	call inner_kernel_dgemm_sub_nt_4x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d // alg 
	movq	STACKSIZE + 16(%rsp), %r11  // C 

	call inner_blender_nn_4x4_lib4 // TODO alg has to be applied only to the add part !!!!!


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 56(%rsp), %r12d // kn 

	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4


	// store

	movq	STACKSIZE + 24(%rsp), %r10 // D 
	movl	STACKSIZE + 48(%rsp), %r11d // km 
	movl	STACKSIZE + 56(%rsp), %r12d // kn 

	call inner_store_4x4_vs_lib4


	// epilogue

	vzeroupper

	movq	  (%rsp), %rbx 
	movq	 8(%rsp), %rbp
	movq	16(%rsp), %r12 
	movq	24(%rsp), %r13 
	movq	32(%rsp), %r14 
	movq	40(%rsp), %r15 
	addq	$STACKSIZE, %rsp

	ret

	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4





	// read-only data
	.section	.rodata.cst32,"aM",@progbits,32
#if 1
	.align 32
.LC00: // { -1 -1 -1 1 }
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	1
	.align 32
.LC01: // { -1 -1 -1 -1 }
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	-1
#endif
	.align 32
.LC02: // { 3.5 2.5 1.5 0.5 }
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
	.long	0
	.long	1074921472
	.long	0
	.long	1075183616
	.long	0
	.long	1075445760
	.long	0
	.long	1075707904
	.align 16
.LC04: // { 1 }
	.long	0
	.long	1072693248
	.long	0
	.long	0



	.section	.note.GNU-stack,"",@progbits

