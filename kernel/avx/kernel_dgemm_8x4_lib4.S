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

// 

#define STACKSIZE 96

//.LSTR:
//	.string	"\n%d %d %d\n"

	.text

// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*k*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_add_nt_8x4_lib4, @function
inner_kernel_dgemm_add_nt_8x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L002 // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%r13), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


	// unroll 3
	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5
	cmpl	$4, %r10d

	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L004 // clean1

	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


	// unroll 3
//	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5
//	cmpl	$3, %r10d

//	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6


	ret


.L004: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%r13), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vaddpd	%ymm4, %ymm15, %ymm4
	addq	$32, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	addq	$32, %r13
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm5

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	addq	$32, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm7, %ymm15, %ymm7

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	subl	$1, %r10d
	vmulpd	%ymm9, %ymm14, %ymm15
	vaddpd	%ymm6, %ymm15, %ymm6

	cmpl	$0, %r10d

	jg		.L003 // clean up loop 


.L002: // return

	ret

	.size	inner_kernel_dgemm_add_nt_8x4_lib4, .-inner_kernel_dgemm_add_nt_8x4_lib4





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
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
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*k*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_kernel_dgemm_sub_nt_8x4_lib4, @function
inner_kernel_dgemm_sub_nt_8x4_lib4:
	
	cmpl	$0, %r10d
	jle		.L028 // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%r13), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		.L026 // consider clean-up loop

	// main loop
	.p2align 3
.L027: // main loop
	
	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	// unroll 3
	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5
	cmpl	$4, %r10d

	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	jg		.L027 // main loop 


.L026: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L030 // clean1

	// unroll 0
	vmovapd 32(%r13), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 1
	vmovapd 64(%r13), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	// unroll 2
	vmovapd 96(%r13), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r13
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	addq	$128, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	// unroll 3
//	vmovapd 0(%r13), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5
//	cmpl	$3, %r10d

//	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vmulpd	%ymm11, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6


	ret


.L030: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L028 // return

	// clean-up loop
.L029: // clean up loop
	
	vmovapd 0(%r13), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm9, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm4, %ymm4
	addq	$32, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	addq	$32, %r13
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm5, %ymm5

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	addq	$32, %r15
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm7, %ymm7

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	subl	$1, %r10d
	vmulpd	%ymm9, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm6, %ymm6

	cmpl	$0, %r10d

	jg		.L029 // clean up loop 


.L028: // return

	ret

	.size	inner_kernel_dgemm_sub_nt_8x4_lib4, .-inner_kernel_dgemm_sub_nt_8x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- 4*sda*sizeof(double)
// r12   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- 4*sda*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_8x4_lib4, @function
inner_edge_dtrmm_add_nt_lu_8x4_lib4:
	
	movq	%r10, %r15 // A1 <- A0
	addq	%r11, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r12), %ymm12
	vmovapd			0(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4

	vbroadcastsd	32(%r12), %ymm12
	vmovapd			32(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			32(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	40(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5

	vbroadcastsd	64(%r12), %ymm12
	vmovapd			64(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			64(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	72(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	80(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6

//	vbroadcastsd	96(%r12), %ymm12
//	vmovapd			96(%r10), %ymm8
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm0, %ymm15, %ymm0
//	vmovapd			96(%r15), %ymm9
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm4, %ymm15, %ymm4
//	vbroadcastsd	104(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm1, %ymm15, %ymm1
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm5, %ymm15, %ymm5
//	vbroadcastsd	112(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm2, %ymm15, %ymm2
//	addq			$128, %r10
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm6, %ymm15, %ymm6
//	vbroadcastsd	120(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm3, %ymm15, %ymm3
//	addq			$128, %r12
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm7, %ymm15, %ymm7
	addq			$96, %r10
	addq			$96, %r12

	ret

	.size	inner_edge_dtrmm_add_nt_lu_8x4_lib4, .-inner_edge_dtrmm_add_nt_lu_8x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_add_nt_lu_8x4_vs_lib4, @function
inner_edge_dtrmm_add_nt_lu_8x4_vs_lib4:
	
	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r13
	addq			$32, %r15

	cmpl	$0, %r10d
	jle		.L017

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r11
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	addq			$32, %r15

	cmpl	$0, %r10d
	jle		.L017

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r11
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vaddpd			%ymm6, %ymm15, %ymm6
	addq			$32, %r15

//	cmpl	$0, %r10d
//	jle		.L017

//	vbroadcastsd	0(%r13), %ymm12
//	subl			$1, %r10d
//	vmovapd			0(%r11), %ymm8
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm0, %ymm15, %ymm0
//	vmovapd			0(%r15), %ymm9
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm4, %ymm15, %ymm4
//	vbroadcastsd	8(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm1, %ymm15, %ymm1
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm5, %ymm15, %ymm5
//	vbroadcastsd	16(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm2, %ymm15, %ymm2
//	addq			$32, %r11
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm6, %ymm15, %ymm6
//	vbroadcastsd	24(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vaddpd			%ymm3, %ymm15, %ymm3
//	addq			$32, %r13
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vaddpd			%ymm7, %ymm15, %ymm7
//	addq			$32, %r15

.L017:
	ret

	.size	inner_edge_dtrmm_add_nt_lu_8x4_vs_lib4, .-inner_edge_dtrmm_add_nt_lu_8x4_vs_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- 4*sda*sizeof(double)
// r12   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- 4*sda*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_8x4_lib4, @function
inner_edge_dtrmm_sub_nt_lu_8x4_lib4:
	
	movq	%r10, %r15 // A1 <- A0
	addq	%r11, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r12), %ymm12
	vmovapd			0(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4

	vbroadcastsd	32(%r12), %ymm12
	vmovapd			32(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	vmovapd			32(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	40(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm5, %ymm15, %ymm5

	vbroadcastsd	64(%r12), %ymm12
	vmovapd			64(%r10), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	vmovapd			64(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	72(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm1, %ymm15, %ymm1
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	80(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm2, %ymm15, %ymm2
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm6, %ymm15, %ymm6

//	vbroadcastsd	96(%r12), %ymm12
//	vmovapd			96(%r10), %ymm8
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm0, %ymm15, %ymm0
//	vmovapd			96(%r15), %ymm9
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm4, %ymm15, %ymm4
//	vbroadcastsd	104(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm1, %ymm15, %ymm1
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm5, %ymm15, %ymm5
//	vbroadcastsd	112(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm2, %ymm15, %ymm2
//	addq			$128, %r10
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm6, %ymm15, %ymm6
//	vbroadcastsd	120(%r12), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm3, %ymm15, %ymm3
//	addq			$128, %r12
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm7, %ymm15, %ymm7
	addq			$96, %r10
	addq			$96, %r12

	ret

	.size	inner_edge_dtrmm_sub_nt_lu_8x4_lib4, .-inner_edge_dtrmm_sub_nt_lu_8x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- mak(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*4*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


	.p2align 4,,15
	.type inner_edge_dtrmm_sub_nt_lu_8x4_vs_lib4, @function
inner_edge_dtrmm_sub_nt_lu_8x4_vs_lib4:
	
	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r13
	addq			$32, %r15

	cmpl	$0, %r10d
	jle		.L023

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4
	addq			$32, %r11
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm5, %ymm15, %ymm5
	addq			$32, %r15

	cmpl	$0, %r10d
	jle		.L023

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm8
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm0, %ymm15, %ymm0
	vmovapd			0(%r15), %ymm9
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm4, %ymm15, %ymm4
	vbroadcastsd	8(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r11
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm5, %ymm15, %ymm5
	vbroadcastsd	16(%r13), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r13
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm6, %ymm15, %ymm6
	addq			$32, %r15

//	cmpl	$0, %r10d
//	jle		.L023

//	vbroadcastsd	0(%r13), %ymm12
//	subl			$1, %r10d
//	vmovapd			0(%r11), %ymm8
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm0, %ymm15, %ymm0
//	vmovapd			0(%r15), %ymm9
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm4, %ymm15, %ymm4
//	vbroadcastsd	8(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm1, %ymm15, %ymm1
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm5, %ymm15, %ymm5
//	vbroadcastsd	16(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm2, %ymm15, %ymm2
//	addq			$32, %r11
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm6, %ymm15, %ymm6
//	vbroadcastsd	24(%r13), %ymm12
//	vmulpd			%ymm8, %ymm12, %ymm15
//	vsubpd			%ymm3, %ymm15, %ymm3
//	addq			$32, %r13
//	vmulpd			%ymm9, %ymm12, %ymm15
//	vsubpd			%ymm7, %ymm15, %ymm7
//	addq			$32, %r15

.L023:
	ret

	.size	inner_edge_dtrmm_sub_nt_lu_8x4_vs_lib4, .-inner_edge_dtrmm_sub_nt_lu_8x4_vs_lib4





// common inner routine with file scope
//
// blender for tc==n and td==n
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C
// r12   <- 4*sdc*sizeof(double)
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nn_8x4_lib4, @function
inner_blender_nn_8x4_lib4:
	

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	cmpl	$0, %r10d // alg
	je		.L005 // return

	movq	%r11, %r15 // C1 <- C0
	addq	%r12, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

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

	vmovapd		0(%r15), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

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

	vmovapd		0(%r15), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

.L005: // return
	
	ret

	.size	inner_blender_nn_8x4_lib4, .-inner_blender_nn_8x4_lib4





// common inner routine with file scope
//
// blender for tc==n and td==t
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12  <- 4*sdc*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C
// r12  <- 4*sdc*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nt_8x4_lib4, @function
inner_blender_nt_8x4_lib4:
	

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

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	movq	%r11, %r15 // C1 <- C0
	addq	%r12, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

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

	vmovapd		0(%r15), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

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

	vmovapd		0(%r15), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

.L009: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpckhpd	%ymm1, %ymm0, %ymm9
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm3, %ymm2, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm8
	vunpckhpd	%ymm5, %ymm4, %ymm9
	vunpcklpd	%ymm7, %ymm6, %ymm10
	vunpckhpd	%ymm7, %ymm6, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm4
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm6
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm5
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm7

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

	vunpcklpd	%ymm5, %ymm4, %ymm8
	vunpcklpd	%ymm7, %ymm6, %ymm10
	vunpckhpd	%ymm4, %ymm5, %ymm9
	vunpckhpd	%ymm6, %ymm7, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm4
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm6
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm5
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm7

	ret

	.size	inner_blender_nt_8x4_lib4, .-inner_blender_nt_8x4_lib4





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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
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
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tn_8x4_lib4, @function
inner_blender_tn_8x4_lib4:
	

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

	vunpcklpd	%ymm5, %ymm4, %ymm8
	vunpcklpd	%ymm7, %ymm6, %ymm10
	vunpckhpd	%ymm4, %ymm5, %ymm9
	vunpckhpd	%ymm6, %ymm7, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm4
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm6
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm5
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm7

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

	vmovapd		128(%r11), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r11), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r11), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r11), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

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

	vmovapd		128(%r11), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r11), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r11), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r11), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

.L012: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpckhpd	%ymm1, %ymm0, %ymm9
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm3, %ymm2, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm8
	vunpckhpd	%ymm5, %ymm4, %ymm9
	vunpcklpd	%ymm7, %ymm6, %ymm10
	vunpckhpd	%ymm7, %ymm6, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm4
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm6
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm5
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm7

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

	vblendpd	$0xa, %ymm5, %ymm4, %ymm8
	vblendpd	$0x5, %ymm5, %ymm4, %ymm9
	vblendpd	$0xa, %ymm7, %ymm6, %ymm10
	vblendpd	$0x5, %ymm7, %ymm6, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm4
	vblendpd	$0x3, %ymm10, %ymm8, %ymm6
	vblendpd	$0xc, %ymm11, %ymm9, %ymm5
	vblendpd	$0x3, %ymm11, %ymm9, %ymm7

	ret

	.size	inner_blender_tn_8x4_lib4, .-inner_blender_tn_8x4_lib4





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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
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
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tt_8x4_lib4, @function
inner_blender_tt_8x4_lib4:
	

	// tc==t
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm8
	vunpcklpd	%ymm7, %ymm6, %ymm10
	vunpckhpd	%ymm4, %ymm5, %ymm9
	vunpckhpd	%ymm6, %ymm7, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm4
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm6
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm5
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm7

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

	vmovapd		128(%r11), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r11), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r11), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r11), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

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

	vmovapd		128(%r11), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r11), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r11), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r11), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

.L014: // return==0
	
	ret

	.size	inner_blender_tt_8x4_lib4, .-inner_blender_tt_8x4_lib4





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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_edge_dpotrf_8x4_vs_lib4, @function
inner_edge_dpotrf_8x4_vs_lib4:
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
	vmovsd	.LC04(%rip), %xmm14 // 1.0

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jle			.L032
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L033:
	cmpl		$2, %r11d
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0
	vmulpd		%ymm4, %ymm13, %ymm4

	jl			.L031 // ret

	vpermilpd	$0x3, %xmm0, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm1, %ymm1
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm5, %ymm5
	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jle			.L034
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L035:
	cmpl		$3, %r11d
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1
	vmulpd		%ymm5, %ymm13, %ymm5

	jl			.L031 // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd	$0x0, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd	$0x0, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm6, %ymm6
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jle			.L036
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L037:
	cmpl		$4, %r11d
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2
	vmulpd		%ymm6, %ymm13, %ymm6

	jl			.L031 // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm4, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm5, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7
	vextractf128	$0x1, %ymm2, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vmulpd		%ymm6, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm7, %ymm7
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jle			.L038
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
.L039:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3
	vmulpd		%ymm7, %ymm13, %ymm7

.L031:
	
	ret

.L032:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L033

.L034:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L035

.L036:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L037

.L038:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		.L039

	.size	inner_edge_dpotrf_8x4_vs_lib4, .-inner_edge_dpotrf_8x4_vs_lib4





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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_8x4_lib4, @function
inner_edge_dtrsm_rlt_inv_8x4_lib4:
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4

	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7

	ret

	.size	inner_edge_dtrsm_rlt_inv_8x4_lib4, .-inner_edge_dtrsm_rlt_inv_8x4_lib4





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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm12 <- dirty
// ymm13 <- dirty

	.p2align 4,,15
	.type inner_edge_dtrsm_rlt_inv_8x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_8x4_vs_lib4:
	
	vbroadcastsd	0(%r11), %ymm13
	cmpl			$3, %r12d
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4

	jl				.L040 // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm5, %ymm5
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5

	jl				.L040 // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm6, %ymm6
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6

	jl				.L040 // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm4, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm5, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vmulpd			%ymm6, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm7, %ymm7
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7

.L040:
	
	ret

	.size	inner_edge_dtrsm_rlt_inv_8x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_8x4_vs_lib4





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]

	.p2align 4,,15
	.type inner_store_8x4_lib4, @function
inner_store_8x4_lib4:
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4,  0(%r15)
	vmovapd %ymm5, 32(%r15)
	vmovapd %ymm6, 64(%r15)
	vmovapd %ymm7, 96(%r15)

	ret

	.size	inner_store_8x4_lib4, .-inner_store_8x4_lib4





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_store_8x4_vs_lib4, @function
inner_store_8x4_vs_lib4:
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC03(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r15)
	jl			.L015 // end
	cmpl		$3, %r13d
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r15)
	jl			.L015 // end
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r15)
	je			.L015 // end
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r15)

.L015:
	ret

	.size	inner_store_8x4_vs_lib4, .-inner_store_8x4_vs_lib4





// common inner routine with file scope
//
// store lower n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_store_l_8x4_vs_lib4, @function
inner_store_l_8x4_vs_lib4:
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC03(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r15)
	jl			.L018 // end
	cmpl		$3, %r13d
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r15)
	jl			.L018 // end
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r15)
	je			.L018 // end
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r15)


.L018:
	ret

	.size	inner_store_l_8x4_vs_lib4, .-inner_store_l_8x4_vs_lib4





// common inner routine with file scope
//
// store t
//
// input arguments:
// r10  <- D
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
//
// output arguments:
// r10  <- D
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]

	.p2align 4,,15
	.type inner_store_4x8_lib4, @function 
inner_store_4x8_lib4:
	
	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4, 128(%r10)
	vmovapd %ymm5, 160(%r10)
	vmovapd %ymm6, 192(%r10)
	vmovapd %ymm7, 224(%r10)

	ret

	.size	inner_store_4x8_lib4, .-inner_store_4x8_lib4





// common inner routine with file scope
//
// store t vs
//
// input arguments:
// r10  <- D
// r11d  <- km
// r12d  <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11d  <- km
// r12d  <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_store_4x8_vs_lib4, @function
inner_store_4x8_vs_lib4:
	
	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$6, %r12d
	vmaskmovpd	%ymm0, %ymm15,   0(%r10)
	vmaskmovpd	%ymm1, %ymm15,  32(%r10)
	vmaskmovpd	%ymm2, %ymm15,  64(%r10)
	vmaskmovpd	%ymm3, %ymm15,  96(%r10)
	vmaskmovpd	%ymm4, %ymm15, 128(%r10)
	jl			.L016 // end
	cmpl		$7, %r12d
	vmaskmovpd	%ymm5, %ymm15, 160(%r10)
	jl			.L016 // end
	vmaskmovpd	%ymm6, %ymm15, 192(%r10)
	je			.L016 // end
	vmaskmovpd	%ymm7, %ymm15, 224(%r10)

.L016:
	ret

	.size	inner_store_4x8_vs_lib4, .-inner_store_4x8_vs_lib4





//                                 rdi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24  
// void kernel_dgemm_ntnn_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dgemm_ntnn_8x4_lib4
	.type kernel_dgemm_ntnn_8x4_lib4, @function
kernel_dgemm_ntnn_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// store n

	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	call inner_store_8x4_lib4


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

	.size	kernel_dgemm_ntnn_8x4_lib4, .-kernel_dgemm_ntnn_8x4_lib4





//                                    rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32    rsp+40
// void kernel_dgemm_ntnn_8x4_vs_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_ntnn_8x4_vs_lib4
	.type kernel_dgemm_ntnn_8x4_vs_lib4, @function
kernel_dgemm_ntnn_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

	call inner_store_8x4_vs_lib4


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

	.size	kernel_dgemm_ntnn_8x4_vs_lib4, .-kernel_dgemm_ntnn_8x4_vs_lib4





//                                 rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24  
// void kernel_dgemm_ntnt_8x4_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dgemm_ntnt_8x4_lib4
	.type kernel_dgemm_ntnt_8x4_lib4, @function
kernel_dgemm_ntnt_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nt_8x4_lib4


	// store t

	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	call inner_store_4x8_lib4


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

	.size	kernel_dgemm_ntnt_8x4_lib4, .-kernel_dgemm_ntnt_8x4_lib4





//                                    rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32  rsp+40 
// void kernel_dgemm_ntnt_8x4_vs_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_ntnt_8x4_vs_lib4
	.type kernel_dgemm_ntnt_8x4_vs_lib4, @function
kernel_dgemm_ntnt_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nt_8x4_lib4


	// store t

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 32(%rsp), %r12d // kn 
	movl	STACKSIZE + 40(%rsp), %r11d // km 

	call inner_store_4x8_vs_lib4


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

	.size	kernel_dgemm_ntnt_8x4_vs_lib4, .-kernel_dgemm_ntnt_8x4_vs_lib4





//                                 rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   
// void kernel_dgemm_nttn_8x4_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dgemm_nttn_8x4_lib4
	.type kernel_dgemm_nttn_8x4_lib4, @function
kernel_dgemm_nttn_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11

	call inner_blender_tn_8x4_lib4


	// store n

	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	call inner_store_8x4_lib4


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

	.size	kernel_dgemm_nttn_8x4_lib4, .-kernel_dgemm_nttn_8x4_lib4





//                                    rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32  rsp+40 
// void kernel_dgemm_nttn_8x4_vs_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nttn_8x4_vs_lib4
	.type kernel_dgemm_nttn_8x4_vs_lib4, @function
kernel_dgemm_nttn_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11

	call inner_blender_tn_8x4_lib4


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

	call inner_store_8x4_vs_lib4


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

	.size	kernel_dgemm_nttn_8x4_vs_lib4, .-kernel_dgemm_nttn_8x4_vs_lib4





//                                 rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24  
// void kernel_dgemm_nttt_8x4_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dgemm_nttt_8x4_lib4
	.type kernel_dgemm_nttt_8x4_lib4, @function
kernel_dgemm_nttt_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender tt

	movl	%r8d, %r10d
	movq	%r9, %r11

	call inner_blender_tt_8x4_lib4


	// store t

	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	call inner_store_4x8_lib4



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

	.size	kernel_dgemm_nttt_8x4_lib4, .-kernel_dgemm_nttt_8x4_lib4





//                                    rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32  rsp+40
// void kernel_dgemm_nttt_8x4_vs_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nttt_8x4_vs_lib4
	.type kernel_dgemm_nttt_8x4_vs_lib4, @function
kernel_dgemm_nttt_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender tt

	movl	%r8d, %r10d
	movq	%r9, %r11

	call inner_blender_tt_8x4_lib4


	// store t

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 32(%rsp), %r12d // kn 
	movl	STACKSIZE + 40(%rsp), %r11d // km 

	call inner_store_4x8_vs_lib4



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

	.size	kernel_dgemm_nttt_8x4_vs_lib4, .-kernel_dgemm_nttt_8x4_vs_lib4





//                                      rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32    rsp+40
// void kernel_dsyrk_ntnn_l_8x4_vs_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_ntnn_l_8x4_vs_lib4
	.type kernel_dsyrk_ntnn_l_8x4_vs_lib4, @function
kernel_dsyrk_ntnn_l_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

	call inner_store_l_8x4_vs_lib4

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

	.size	kernel_dsyrk_ntnn_l_8x4_vs_lib4, .-kernel_dsyrk_ntnn_l_8x4_vs_lib4





//                                    edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24
// void kernel_dtrmm_ntnn_lu_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dtrmm_ntnn_lu_8x4_lib4
	.type kernel_dtrmm_ntnn_lu_8x4_lib4, @function
kernel_dtrmm_ntnn_lu_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d
//	subl	$4, %r10d
	subl	$3, %r10d
	movq	%rsi, %r11
//	addq	$128, %r11
	addq	$96, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13
//	addq	$128, %r13
	addq	$96, %r13


	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// initial triangle

	movq	%rsi, %r10
	movl	%edx, %r11d
	sall	$5, %r11d // 4*sda*sizeof(double)
	movslq	%r11d, %r11
	movq	%rcx, %r12

	cmpl	$-1, %r8d // alg
	je		.L019


	call inner_edge_dtrmm_add_nt_lu_8x4_lib4

	jmp		.L020

.L019:

	call inner_edge_dtrmm_sub_nt_lu_8x4_lib4 

.L020:
	// store n

	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	call inner_store_8x4_lib4


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

	.size	kernel_dtrmm_ntnn_lu_8x4_lib4, .-kernel_dtrmm_ntnn_lu_8x4_lib4





//                                    edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32  rsp+40
// void kernel_dtrmm_ntnn_lu_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
	.globl kernel_dtrmm_ntnn_lu_8x4_vs_lib4
	.type kernel_dtrmm_ntnn_lu_8x4_vs_lib4, @function
kernel_dtrmm_ntnn_lu_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d
//	subl	$4, %r10d
	subl	$3, %r10d
	movq	%rsi, %r11
//	addq	$128, %r11
	addq	$96, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13
//	addq	$128, %r13
	addq	$96, %r13


	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// initial triangle

	movq	%rdi, %r10
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	cmpl	$-1, %r8d // alg
	je		.L024


	call inner_edge_dtrmm_add_nt_lu_8x4_vs_lib4

	jmp		.L025

.L024:

	call inner_edge_dtrmm_sub_nt_lu_8x4_vs_lib4 

.L025:
	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

	call inner_store_8x4_vs_lib4


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

	.size	kernel_dtrmm_ntnn_lu_8x4_vs_lib4, .-kernel_dtrmm_ntnn_lu_8x4_vs_lib4





//                                    rdi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32                 rsp+40  rsp+48
// void kernel_dpotrf_ntnn_l_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dpotrf_ntnn_l_8x4_vs_lib4
	.type kernel_dpotrf_ntnn_l_8x4_vs_lib4, @function
kernel_dpotrf_ntnn_l_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// factorization

	movq	STACKSIZE + 32(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 48(%rsp), %r11d // kn 

	call inner_edge_dpotrf_8x4_vs_lib4


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

	call inner_store_l_8x4_vs_lib4


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

	.size	kernel_dpotrf_ntnn_l_8x4_vs_lib4, .-kernel_dpotrf_ntnn_l_8x4_vs_lib4





//                                             rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24   rsp+32     rsp+40   rsp+48     rsp+56   rsp+64              rsp+72  rsp+80
// void kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4
	.type kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4, @function
kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sdam*sizeof(double)
	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE + 24(%rsp), %r10d // alg
	movq	STACKSIZE + 32(%rsp), %r11 // C
	movl	STACKSIZE + 40(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12 

	call inner_blender_nn_8x4_lib4


	// factorization

	movq	STACKSIZE + 64(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 80(%rsp), %r11d // kn 

	call inner_edge_dpotrf_8x4_vs_lib4


	// store n

	movq	STACKSIZE + 48(%rsp), %r10 // store address D
	movl	STACKSIZE + 56(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  72(%rsp), %r12d // km 
	movl	STACKSIZE +  80(%rsp), %r13d // kn 

	call inner_store_l_8x4_vs_lib4


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

	.size	kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4, .-kernel_dsyrk_dpotrf_ntnn_l_8x4_vs_lib4





//                                           rdi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40              rsp+48  rsp+56
// void kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4
	.type kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4, @function
kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 56(%rsp), %r12d // kn 

	call inner_edge_dtrsm_rlt_inv_8x4_vs_lib4


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  48(%rsp), %r12d // km 
	movl	STACKSIZE +  56(%rsp), %r13d // kn 

	call inner_store_8x4_vs_lib4


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

	.size	kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4, .-kernel_dtrsm_ntnn_rl_inv_8x4_vs_lib4





//                                                 rdi     rsi         rdx       rcx          r8     r9          rsp+8     rsp+16      rsp+24   rsp+32     rsp+40   rsp+48     rsp+56   rsp+64     rsp+72              rsp+80  rsp+88
// void kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4
	.type kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4, @function
kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE + 24(%rsp), %r10d // alg
	movq	STACKSIZE + 32(%rsp), %r11  // C
	movl	STACKSIZE + 40(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// solve

	movq	STACKSIZE + 64(%rsp), %r10  // E 
	movq	STACKSIZE + 72(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 88(%rsp), %r12d // kn 

	call inner_edge_dtrsm_rlt_inv_8x4_vs_lib4


	// store n

	movq	STACKSIZE + 48(%rsp), %r10 // store address D
	movl	STACKSIZE + 56(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  80(%rsp), %r12d // km 
	movl	STACKSIZE +  88(%rsp), %r13d // kn 

	call inner_store_8x4_vs_lib4


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

	.size	kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4, .-kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_vs_lib4





//                                        rdi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40 
// void kernel_dtrsm_ntnn_rl_inv_8x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dtrsm_ntnn_rl_inv_8x4_lib4
	.type kernel_dtrsm_ntnn_rl_inv_8x4_lib4, @function
kernel_dtrsm_ntnn_rl_inv_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	%r8d, %r10d
	movq	%r9, %r11
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11  // inv_diag_E 

	call inner_edge_dtrsm_rlt_inv_8x4_lib4


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	call inner_store_8x4_lib4


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

	.size	kernel_dtrsm_ntnn_rl_inv_8x4_lib4, .-kernel_dtrsm_ntnn_rl_inv_8x4_lib4





//                                              rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24   rsp+32     rsp+40   rsp+48     rsp+56   rsp+64     rsp+72  
// void kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, int alg, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
	.globl kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4
	.type kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4, @function
kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4:
	
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
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

	call inner_kernel_dgemm_sub_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE + 24(%rsp), %r10d // alg
	movq	STACKSIZE + 32(%rsp), %r11  // C
	movl	STACKSIZE + 40(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_8x4_lib4


	// solve

	movq	STACKSIZE + 64(%rsp), %r10  // E 
	movq	STACKSIZE + 72(%rsp), %r11  // inv_diag_E 

	call inner_edge_dtrsm_rlt_inv_8x4_lib4


	// store n

	movq	STACKSIZE + 48(%rsp), %r10 // store address D
	movl	STACKSIZE + 56(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	call inner_store_8x4_lib4


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

	.size	kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4, .-kernel_dgemm_dtrsm_ntnn_rl_inv_8x4_lib4





	// read-only data
	.section	.rodata.cst32,"aM",@progbits,32
#if 0
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
