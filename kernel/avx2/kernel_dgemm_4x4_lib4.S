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
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L002 // return

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r12), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vfmadd231pd	%ymm8, %ymm14, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vfmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vfmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfmadd231pd	%ymm10, %ymm14, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vfmadd231pd	%ymm8, %ymm14, %ymm2


	// unroll 3
	vmovapd 0(%r12), %ymm12 // B[0]
	vfmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	cmpl	$4, %r10d

	vfmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfmadd231pd	%ymm10, %ymm14, %ymm2

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L004 // clean1

	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

//	subl	$4, %r10d
	vfmadd231pd	%ymm8, %ymm14, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vfmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vfmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfmadd231pd	%ymm10, %ymm14, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
//	addq	$128, %r12
//	addq	$128, %r11

	vfmadd231pd	%ymm8, %ymm14, %ymm2


	// unroll 3
//	vmovapd 0(%r12), %ymm12 // B[0]
	vfmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm14, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
//	cmpl	$3, %r10d

	vfmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfmadd231pd	%ymm10, %ymm14, %ymm2


//	jmp		.L002 // return
	ret


.L004: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%r12), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	addq	$32, %r11

	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	addq	$32, %r12

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm8, %ymm14, %ymm3

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm8, %ymm14, %ymm2
	subl	$1, %r10d

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
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_sub_nt_4x4_lib4, @function
inner_kernel_dgemm_sub_nt_4x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L025 // return

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r12), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		.L023 // consider clean-up loop

	// main loop
	.p2align 3
.L024: // main loop
	
	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfnmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vfnmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vfnmadd231pd	%ymm8, %ymm14, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vfnmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfnmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vfnmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfnmadd231pd	%ymm10, %ymm14, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfnmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vfnmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vfnmadd231pd	%ymm8, %ymm14, %ymm2


	// unroll 3
	vmovapd 0(%r12), %ymm12 // B[0]
	vfnmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfnmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	cmpl	$4, %r10d

	vfnmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfnmadd231pd	%ymm10, %ymm14, %ymm2

	jg		.L024 // main loop 


.L023: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L027 // clean1

	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfnmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vfnmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

//	subl	$4, %r10d
	vfnmadd231pd	%ymm8, %ymm14, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vfnmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfnmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vfnmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfnmadd231pd	%ymm10, %ymm14, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfnmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vfnmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
//	addq	$128, %r12
//	addq	$128, %r11

	vfnmadd231pd	%ymm8, %ymm14, %ymm2


	// unroll 3
//	vmovapd 0(%r12), %ymm12 // B[0]
	vfnmadd231pd	%ymm10, %ymm13, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfnmadd231pd	%ymm10, %ymm14, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
//	cmpl	$3, %r10d

	vfnmadd231pd	%ymm10, %ymm13, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vfnmadd231pd	%ymm10, %ymm14, %ymm2


	ret


.L027: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L025 // return

	// clean-up loop
.L026: // clean up loop
	
	vmovapd 0(%r12), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	addq	$32, %r11

	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfnmadd231pd	%ymm8, %ymm14, %ymm1
	addq	$32, %r12

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vfnmadd231pd	%ymm8, %ymm14, %ymm3

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vfnmadd231pd	%ymm8, %ymm14, %ymm2
	subl	$1, %r10d

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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- B+4*4*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_4x4_lib4, @function
inner_edge_dtrmm_add_nt_lu_4x4_lib4:
	
	vmovapd			0(%r10), %ymm8
	vbroadcastsd	0(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0

	vmovapd			32(%r10), %ymm8
	vbroadcastsd	32(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	vbroadcastsd	40(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1

	vmovapd			64(%r10), %ymm8
	vbroadcastsd	64(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	vbroadcastsd	72(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1
	vbroadcastsd	80(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm2

	vmovapd			96(%r10), %ymm8
	vbroadcastsd	96(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	vbroadcastsd	104(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1
	vbroadcastsd	112(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm2
	addq			$128, %r10
	vbroadcastsd	120(%r11), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm3
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4, @function
inner_edge_dtrmm_add_nt_lu_4x4_vs_lib4:
	
	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	addq			$32, %r11
	vbroadcastsd	8(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1
	addq			$32, %r11
	vbroadcastsd	16(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm2
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L016

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm2
	addq			$32, %r11
	vbroadcastsd	24(%r12), %ymm12
	vfmadd231pd		%ymm8, %ymm12, %ymm3
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- B+4*4*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_4x4_lib4, @function
inner_edge_dtrmm_sub_nt_lu_4x4_lib4:

	vmovapd			0(%r10), %ymm8
	vbroadcastsd	0(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0

	vmovapd			32(%r10), %ymm8
	vbroadcastsd	32(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vbroadcastsd	40(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1

	vmovapd			64(%r10), %ymm8
	vbroadcastsd	64(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vbroadcastsd	72(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1
	vbroadcastsd	80(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm2

	vmovapd			96(%r10), %ymm8
	vbroadcastsd	96(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vbroadcastsd	104(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1
	vbroadcastsd	112(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm2
	addq			$128, %r10
	vbroadcastsd	120(%r11), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm3
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4, @function
inner_edge_dtrmm_sub_nt_lu_4x4_vs_lib4:
	
	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	addq			$32, %r11
	vbroadcastsd	8(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1
	addq			$32, %r11
	vbroadcastsd	16(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm2
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		.L017

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm2
	addq			$32, %r11
	vbroadcastsd	24(%r12), %ymm12
	vfnmadd231pd	%ymm8, %ymm12, %ymm3
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
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d <- alg
// r11   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nn_4x4_lib4, @function
inner_blender_nn_4x4_lib4:
	

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	cmpl	$0, %r10d // alg
	je		.L005 // return

	cmpl	$1, %r10d // alg
	jne		.L006 // alg==-1

	// alg==1
	vmovapd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	ret // return

.L006: // alg==-1

	// alg==-1
	vmovapd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L005: // return
	
	ret

	.size	inner_blender_nn_4x4_lib4, .-inner_blender_nn_4x4_lib4





// common inner routine with file scope
//
// blender for tc==n and td==t
//
// input arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nt_4x4_lib4, @function
inner_blender_nt_4x4_lib4:
	

	cmpl	$0, %r10d // alg
	je		.L008 // alg==0

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	cmpl	$1, %r10d // alg
	jne		.L007 // alg==-1

	// alg==1
	vmovapd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	jmp		.L009 // tran

.L007: // alg==-1

	// alg==-1
	vmovapd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L009: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpckhpd	%ymm1, %ymm0, %ymm9
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm3, %ymm2, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm3

	ret

.L008: // alg==0
	
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

	ret

	.size	inner_blender_nt_4x4_lib4, .-inner_blender_nt_4x4_lib4





// common inner routine with file scope
//
// blender for tc==t and td==n
//
// input arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tn_4x4_lib4, @function
inner_blender_tn_4x4_lib4:
	

	cmpl	$0, %r10d // alg
	je		.L011 // alg==0

	// tc==n
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

	cmpl	$1, %r10d // alg
	jne		.L010 // alg==-1

	// alg==1
	vmovapd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	jmp		.L012 // tran

.L010: // alg==-1

	// alg==-1
	vmovapd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L012: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpckhpd	%ymm1, %ymm0, %ymm9
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm3, %ymm2, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm3

	ret

.L011: // alg==0
	
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	ret

	.size	inner_blender_tn_4x4_lib4, .-inner_blender_tn_4x4_lib4





// common inner routine with file scope
//
// blender for tc==t and td==t
//
// input arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tt_4x4_lib4, @function
inner_blender_tt_4x4_lib4:
	

	// tc==t
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

	cmpl	$0, %r10d // alg
	je		.L014 // return

	cmpl	$1, %r10d // alg
	jne		.L013 // alg==-1

	// alg==1
	vmovapd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	ret

.L013: // alg==-1

	// alg==-1
	vmovapd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r11), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r11), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r11), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L014: // return==0
	
	ret

	.size	inner_blender_tt_4x4_lib4, .-inner_blender_tt_4x4_lib4





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_edge_dpotrf_4x4_vs_lib4, @function
inner_edge_dpotrf_4x4_vs_lib4:
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
	vmovsd	.LC04(%rip), %xmm14 // 1.0

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			.L029
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L030:
	cmpl		$2, %r11d
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0

	jl			.L028 // ret

	vpermilpd	$0x3, %xmm0, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			.L031
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L032:
	cmpl		$3, %r11d
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1

	jl			.L028 // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd	$0x0, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd	$0x0, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			.L033
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L034:
	cmpl		$4, %r11d
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2

	jl			.L028 // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vextractf128	$0x1, %ymm2, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			.L035
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L036:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3

.L028:
	
	ret

.L029:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L030

.L031:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L032

.L033:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L034

.L035:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L036

	.size	inner_edge_dpotrf_4x4_vs_lib4, .-inner_edge_dpotrf_4x4_vs_lib4





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_4x4_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_lib4:
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

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
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
	
	vbroadcastsd	0(%r11), %ymm13
	cmpl			$2, %r12d
	vmulpd			%ymm0, %ymm13, %ymm0

	jl				.L037 // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r12d
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	jl				.L037 // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r12d
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	jl				.L037 // ret

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

.L037:
	
	ret

	.size	inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_vs_lib4





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:
// r10  <- D
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]

	.p2align 4,,15
	.type inner_store_4x4_lib4, @function
inner_store_4x4_lib4:
	
	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)
	
	ret

	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_store_4x4_vs_lib4, @function
inner_store_4x4_vs_lib4:
	
	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r10)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r10)
	jl			.L015 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r10)
	je			.L015 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r10)

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
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// r12d   <- kn
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_store_l_4x4_vs_lib4, @function
inner_store_l_4x4_vs_lib4:
	
	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r10)
	jl			.L018 // end
	cmpl		$3, %r12d
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmaskmovpd	%ymm1, %ymm15, 32(%r10)
	jl			.L018 // end
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmaskmovpd	%ymm2, %ymm15, 64(%r10)
	je			.L018 // end
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmaskmovpd	%ymm3, %ymm15, 96(%r10)

.L018:
	ret

	.size	inner_store_l_4x4_vs_lib4, .-inner_store_l_4x4_vs_lib4





//                                 rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dgemm_ntnn_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_ntnn_4x4_lib4
	.type kernel_dgemm_ntnn_4x4_lib4, @function
kernel_dgemm_ntnn_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dgemm_ntnn_4x4_lib4, .-kernel_dgemm_ntnn_4x4_lib4





//                                    rdi    rsi        rdx        rcx      r8         r9         rsp+8   rsp+16
// void kernel_dgemm_ntnn_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_ntnn_4x4_vs_lib4
	.type kernel_dgemm_ntnn_4x4_vs_lib4, @function
kernel_dgemm_ntnn_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dgemm_ntnn_4x4_vs_lib4, .-kernel_dgemm_ntnn_4x4_vs_lib4





//                                 rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dgemm_ntnt_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_ntnt_4x4_lib4
	.type kernel_dgemm_ntnt_4x4_lib4, @function
kernel_dgemm_ntnt_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nt_4x4_lib4


	// store t

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

	.size	kernel_dgemm_ntnt_4x4_lib4, .-kernel_dgemm_ntnt_4x4_lib4





//                                    rdi    rsi        rdx        rcx      r8         r9         rsp+8    rsp+16
// void kernel_dgemm_ntnt_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_ntnt_4x4_vs_lib4
	.type kernel_dgemm_ntnt_4x4_vs_lib4, @function
kernel_dgemm_ntnt_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_nt_4x4_lib4


	// store t

	movq	%r9, %r10
	movl	STACKSIZE + 16(%rsp), %r11d // kn 
	movl	STACKSIZE +  8(%rsp), %r12d // km 

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

	.size	kernel_dgemm_ntnt_4x4_vs_lib4, .-kernel_dgemm_ntnt_4x4_vs_lib4





//                                 rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dgemm_nttn_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_nttn_4x4_lib4
	.type kernel_dgemm_nttn_4x4_lib4, @function
kernel_dgemm_nttn_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_tn_4x4_lib4


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

	.size	kernel_dgemm_nttn_4x4_lib4, .-kernel_dgemm_nttn_4x4_lib4





//                                    rdi    rsi        rdx        rcx      r8         r9         rsp+8    rsp+16
// void kernel_dgemm_nttn_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nttn_4x4_vs_lib4
	.type kernel_dgemm_nttn_4x4_vs_lib4, @function
kernel_dgemm_nttn_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_tn_4x4_lib4


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

	.size	kernel_dgemm_nttn_4x4_vs_lib4, .-kernel_dgemm_nttn_4x4_vs_lib4





//                                 rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dgemm_nttt_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_nttt_4x4_lib4
	.type kernel_dgemm_nttt_4x4_lib4, @function
kernel_dgemm_nttt_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender tt

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_tt_4x4_lib4


	// store t

	movq	%r9, %r10 // D

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

	.size	kernel_dgemm_nttt_4x4_lib4, .-kernel_dgemm_nttt_4x4_lib4





//                                    rdi    rsi        rdx        rcx      r8         r9         rsp+8   rsp+16  
// void kernel_dgemm_nttt_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nttt_4x4_vs_lib4
	.type kernel_dgemm_nttt_4x4_vs_lib4, @function
kernel_dgemm_nttt_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender tt

	movl	%ecx, %r10d
	movq	%r8, %r11

	call inner_blender_tt_4x4_lib4


	// store t

	movq	%r9, %r10 // D
	movl	STACKSIZE + 40(%rsp), %r11d // kn 
	movl	STACKSIZE + 32(%rsp), %r12d // km 

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

	.size	kernel_dgemm_nttt_4x4_vs_lib4, .-kernel_dgemm_nttt_4x4_vs_lib4





//                                      rdi    rsi        rdx        rcx      r8         r9         rsp+8   rsp+16
// void kernel_dsyrk_ntnn_l_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_ntnn_l_4x4_vs_lib4
	.type kernel_dsyrk_ntnn_l_4x4_vs_lib4, @function
kernel_dsyrk_ntnn_l_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dsyrk_ntnn_l_4x4_vs_lib4, .-kernel_dsyrk_ntnn_l_4x4_vs_lib4





//                                    edi    rsi        rdx        ecx      r8         r9         
// void kernel_dtrmm_ntnn_lu_4x4_lib4(int k, double *A, double *B, int alg, double *C, double *D);

	.p2align 4,,15
	.globl kernel_dtrmm_ntnn_lu_4x4_lib4
	.type kernel_dtrmm_ntnn_lu_4x4_lib4, @function
kernel_dtrmm_ntnn_lu_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dtrmm_ntnn_lu_4x4_lib4, .-kernel_dtrmm_ntnn_lu_4x4_lib4





//                                       edi    rsi        rdx        ecx      r8         r9         rsp+8   rsp+16 
// void kernel_dtrmm_ntnn_lu_4x4_vs_lib4(int k, double *A, double *B, int alg, double *C, double *D, int km, int kn);
// it holds k==kn !!!!!

	.p2align 4,,15
	.globl kernel_dtrmm_ntnn_lu_4x4_vs_lib4
	.type kernel_dtrmm_ntnn_lu_4x4_vs_lib4, @function
kernel_dtrmm_ntnn_lu_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dtrmm_ntnn_lu_4x4_vs_lib4, .-kernel_dtrmm_ntnn_lu_4x4_vs_lib4





//                                       edi    rsi        rdx        ecx        r8         r9                  rsp+8   rsp+16
// void kernel_dpotrf_ntnn_l_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dpotrf_ntnn_l_4x4_vs_lib4
	.type kernel_dpotrf_ntnn_l_4x4_vs_lib4, @function
kernel_dpotrf_ntnn_l_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dpotrf_ntnn_l_4x4_vs_lib4, .-kernel_dpotrf_ntnn_l_4x4_vs_lib4





//                                             edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4
	.type kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4, @function
kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4, .-kernel_dsyrk_dpotrf_ntnn_l_4x4_vs_lib4





//                                        edi    rsi        rdx        ecx        r8         r9         rsp+8  
// void kernel_dtrsm_ntnn_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dtrsm_ntnn_rl_inv_4x4_lib4
	.type kernel_dtrsm_ntnn_rl_inv_4x4_lib4, @function
kernel_dtrsm_ntnn_rl_inv_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dtrsm_ntnn_rl_inv_4x4_lib4, .-kernel_dtrsm_ntnn_rl_inv_4x4_lib4





//                                              edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32    rsp+40                     
// void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4
	.type kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4, @function
kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4, .-kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_lib4





//                                           edi    rsi        rdx        ecx        r8       r9           rsp+8               rsp+16  rsp+24
// void kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4
	.type kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4, @function
kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4, .-kernel_dtrsm_ntnn_rl_inv_4x4_vs_lib4





//                                                 edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32     rsp+40              rsp+48  rsp+56
// void kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, int alg, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4
	.type kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4, @function
kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4:
	
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

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


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

	.size	kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4, .-kernel_dgemm_dtrsm_ntnn_rl_inv_4x4_vs_lib4





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

