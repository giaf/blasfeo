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
// r10d  <- k
// r11   <- A
// r12   <- sda
// r13   <- B
// r14   <- dirty
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
// r14   <- dirty
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
	.type inner_kernel_dgemm_add_nt_12x4_lib4, @function
inner_kernel_dgemm_add_nt_12x4_lib4:
	
	cmpl	$0, %edi
	jle		.L002 				// return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)
	movq	%r15, %r14 // A2 <- A1
	addq	%r12, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), 	%ymm12 	// A0[0]
	vmovapd 0(%r15), 	%ymm13 	// A1[0]
	vmovapd 0(%r14), 	%ymm14 	// A2[0] 
	vmovapd 0(%r13), 	%ymm15 	// B[0]

	cmpl	$4, 		%r10d
	jle		.L000 				// consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	subl	$4, %r10d

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 		// A0[4]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 		// A1[4]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r14), %ymm14 		// A2[4]
	vmovapd 		32(%r13), %ymm15 		// B[4]

	// unroll 1
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 		// A0[8]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 		// A1[8]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r14), %ymm14 		// A2[8]
	vmovapd 		64(%r13), %ymm15 		// B[8]

	// unroll 2
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,		%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, 	%ymm15, %ymm2
	vmovapd 		96(%r11), 	%ymm12 			// A0[12]
	vfmadd231pd		%ymm13, 	%ymm15, %ymm6
	vmovapd 		96(%r15), 	%ymm13 			// A1[12]
	vfmadd231pd		%ymm14, 	%ymm15, %ymm10
	vmovapd 		96(%r14), 	%ymm14 			// A2[12]
	vmovapd 		96(%r13), 	%ymm15 			// B[12]

	addq			$128, %r11
	addq			$128, %r15
	addq			$128, %r14
	addq			$128, %r13

	// unroll 3
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	cmpl	$3, %r10d

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		0(%r11), %ymm12 		// A0[16]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		0(%r15), %ymm13 		// A1[16]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		0(%r14), %ymm14 		// A2[16]
	vmovapd 		0(%r13), %ymm15 		// B[16]

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L004 // clean1

	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	subl			$4, %r10d
	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 		// A0[4]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 		// A1[4]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r14), %ymm14 		// A2[4]
	vmovapd 		32(%r13), %ymm15 		// B[4]

	// unroll 1
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 		// A0[8]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 		// A1[8]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r14), %ymm14 		// A2[8]
	vmovapd 		64(%r13), %ymm15 		// B[8]

	// unroll 2
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,		%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, 	%ymm15, %ymm2
	vmovapd 		96(%r11), 	%ymm12 			// A0[12]
	vfmadd231pd		%ymm13, 	%ymm15, %ymm6
	vmovapd 		96(%r15), 	%ymm13 			// A1[12]
	vfmadd231pd		%ymm14, 	%ymm15, %ymm10
	vmovapd 		96(%r14), 	%ymm14 			// A2[12]
	vmovapd 		96(%r13), 	%ymm15 			// B[12]


	addq			$128, %r11
	addq			$128, %r15
	addq			$128, %r14
	addq			$128, %r13

	// unroll 3
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	cmpl	$3, %r10d


//	jmp		.L002 // return
	ret


.L004: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%r11), 	%ymm12 	// A0[0]
	vmovapd 0(%r15), 	%ymm13 	// A1[0]
	vmovapd 0(%r14), 	%ymm14 	// A2[0] 
	vmovapd 0(%r13), 	%ymm15 	// B[0]

	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	subl			$1, %r10d

	addq			$32, %r11
	addq			$32, %r15
	addq			$32, %r14
	addq			$32, %r13

	cmpl	$0, %r10d

	jg		.L003 // clean up loop 


.L002: // return

	ret

	.size	inner_kernel_dgemm_add_nt_12x4_lib4, .-inner_kernel_dgemm_add_nt_12x4_lib4



// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- sda
// r13   <- B
// r14   <- dirty
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
// r14   <- dirty
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
	.type inner_kernel_dgemm_sub_nt_12x4_lib4, @function
inner_kernel_dgemm_sub_nt_12x4_lib4:
	
	cmpl	$0, %edi
	jle		.L027 				// return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)
	movq	%r15, %r14 // A2 <- A1
	addq	%r12, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), 	%ymm12 	// A0[0]
	vmovapd 0(%r15), 	%ymm13 	// A1[0]
	vmovapd 0(%r14), 	%ymm14 	// A2[0] 
	vmovapd 0(%r13), 	%ymm15 	// B[0]

	cmpl	$4, 		%r10d
	jle		.L000 				// consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	subl	$4, %r10d

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 		// A0[4]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 		// A1[4]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r14), %ymm14 		// A2[4]
	vmovapd 		32(%r13), %ymm15 		// B[4]

	// unroll 1
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 		// A0[8]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 		// A1[8]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r14), %ymm14 		// A2[8]
	vmovapd 		64(%r13), %ymm15 		// B[8]

	// unroll 2
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,		%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, 	%ymm15, %ymm2
	vmovapd 		96(%r11), 	%ymm12 			// A0[12]
	vfmadd231pd		%ymm13, 	%ymm15, %ymm6
	vmovapd 		96(%r15), 	%ymm13 			// A1[12]
	vfmadd231pd		%ymm14, 	%ymm15, %ymm10
	vmovapd 		96(%r14), 	%ymm14 			// A2[12]
	vmovapd 		96(%r13), 	%ymm15 			// B[12]

	addq			$128, %r11
	addq			$128, %r15
	addq			$128, %r14
	addq			$128, %r13

	// unroll 3
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	cmpl	$3, %r10d

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		0(%r11), %ymm12 		// A0[16]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		0(%r15), %ymm13 		// A1[16]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		0(%r14), %ymm14 		// A2[16]
	vmovapd 		0(%r13), %ymm15 		// B[16]

	jg		.L001 // main loop 


.L026: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		.L004 // clean1

	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	subl			$4, %r10d
	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 		// A0[4]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 		// A1[4]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r14), %ymm14 		// A2[4]
	vmovapd 		32(%r13), %ymm15 		// B[4]

	// unroll 1
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 		// A0[8]
	vfmadd231pd		%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 		// A1[8]
	vfmadd231pd		%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r14), %ymm14 		// A2[8]
	vmovapd 		64(%r13), %ymm15 		// B[8]

	// unroll 2
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,		%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, 	%ymm15, %ymm2
	vmovapd 		96(%r11), 	%ymm12 			// A0[12]
	vfmadd231pd		%ymm13, 	%ymm15, %ymm6
	vmovapd 		96(%r15), 	%ymm13 			// A1[12]
	vfmadd231pd		%ymm14, 	%ymm15, %ymm10
	vmovapd 		96(%r14), 	%ymm14 			// A2[12]
	vmovapd 		96(%r13), 	%ymm15 			// B[12]


	addq			$128, %r11
	addq			$128, %r15
	addq			$128, %r14
	addq			$128, %r13

	// unroll 3
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	cmpl	$3, %r10d


//	jmp		.L002 // return
	ret


.L030: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		.L002 // return

	// clean-up loop
.L029: // clean up loop
	
	vmovapd 0(%r11), 	%ymm12 	// A0[0]
	vmovapd 0(%r15), 	%ymm13 	// A1[0]
	vmovapd 0(%r14), 	%ymm14 	// A2[0] 
	vmovapd 0(%r13), 	%ymm15 	// B[0]

	// unroll 0
	vfmadd231pd		%ymm12, %ymm15, %ymm0
	vfmadd231pd		%ymm13, %ymm15, %ymm4
	vfmadd231pd		%ymm14, %ymm15, %ymm8
	
	vshufpd 		$0x5,	%ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm1
	vfmadd231pd		%ymm13, %ymm15, %ymm5
	vfmadd231pd		%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15

	vfmadd231pd		%ymm12, %ymm15, %ymm3
	vfmadd231pd		%ymm13, %ymm15, %ymm7
	vfmadd231pd		%ymm14, %ymm15, %ymm11

	subl			$1, %r10d

	addq			$32, %r11
	addq			$32, %r15
	addq			$32, %r14
	addq			$32, %r13

	cmpl	$0, %r10d

	jg		.L003 // clean up loop 


.L028: // return

	ret

	.size	inner_kernel_dgemm_sub_nt_12x4_lib4, .-inner_kernel_dgemm_add_nt_12x4_lib4

// common inner routine with file scope
//
// blender for tc==n and td==n
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12  <- sdc*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8 <- [d80 d91 da2 db3]
// ymm9 <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C+4*sdc*sizeof(double)
// r12  <- 4*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8 <- [d80 d90 da0 db0]
// ymm9 <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nn_12x4_lib4, @function
inner_blender_nn_12x4_lib4:
	

	// tc==n 
	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11

	cmpl	$0, %r10d // alg
	je		.L005 // return

	movq	%r11, %r15 // C1 <- C0
	addq	%r12, %r15 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r15, %r14 // C2 <- C1
	addq	%r12, %r14 // C2 <- C1 + 4*sdc*sizeof(double)

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

	vmovapd		0(%r14), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11

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

	vmovapd		0(%r14), %ymm15
	vsubpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vsubpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vsubpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vsubpd		%ymm11, %ymm15, %ymm11

.L005: // return
	
	ret

	.size	inner_blender_nn_12x4_lib4, .-inner_blender_nn_12x4_lib4

// common inner routine with file scope
//
// blender for tc==n and td==t
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12  <- sdc*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8 <- [d80 d91 da2 db3]
// ymm9 <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C+4*sdc*sizeof(double)
// r12  <- 4*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8 <- [d80 d90 da0 db0]
// ymm9 <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_nt_12x4_lib4, @function
inner_blender_nt_12x4_lib4:

	cmpl	$0, %r10d // alg
	je		.L008 // alg==0

	// tc==n 
	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11


	movq	%r11, %r15 // C1 <- C0
	addq	%r12, %r15 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r15, %r14 // C2 <- C1
	addq	%r12, %r14 // C2 <- C1 + 4*sdc*sizeof(double)

	cmpl	$1, %r10d // alg
	jne		.L007 // alg==-1 ++is this actually necessary?

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

	vmovapd		0(%r14), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11

	jmp 		.L009 //tran

	ret // return

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

	vmovapd		0(%r14), %ymm15
	vsubpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vsubpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vsubpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vsubpd		%ymm11, %ymm15, %ymm11

.L009: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpckhpd	%ymm1, %ymm0, %ymm13
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm3, %ymm2, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm12
	vunpckhpd	%ymm5, %ymm4, %ymm13
	vunpcklpd	%ymm7, %ymm6, %ymm14
	vunpckhpd	%ymm7, %ymm6, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm4
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm6
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm5
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm7
	
	vunpcklpd	%ymm9, %ymm8, %ymm12
	vunpckhpd	%ymm9, %ymm8, %ymm13
	vunpcklpd	%ymm11, %ymm10, %ymm14
	vunpckhpd	%ymm11, %ymm10, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm8
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm10
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm9
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm11
	
	ret

.L008: // alg==0 ++blender + trans...still do not get it

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm0, %ymm1, %ymm13
	vunpckhpd	%ymm2, %ymm3, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm12
	vunpcklpd	%ymm7, %ymm6, %ymm14
	vunpckhpd	%ymm4, %ymm5, %ymm13
	vunpckhpd	%ymm6, %ymm7, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm4
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm6
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm5
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm7

	vunpcklpd	%ymm9, 	%ymm8, 	%ymm12
	vunpcklpd	%ymm11, %ymm10, %ymm14
	vunpckhpd	%ymm8, 	%ymm9, 	%ymm13
	vunpckhpd	%ymm10, %ymm11, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm8
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm10
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm9
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm11

	ret

	.size	inner_blender_nt_12x4_lib4, .-inner_blender_nt_12x4_lib4



// common inner routine with file scope
//
// blender for tc==t and td==n
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12  <- sdc*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8 <- [d80 d91 da2 db3]
// ymm9 <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C+4*sdc*sizeof(double)
// r12  <- 4*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8 <- [d80 d90 da0 db0]
// ymm9 <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tn_12x4_lib4, @function
inner_blender_tn_12x4_lib4:
	

	cmpl	$0, %r10d // alg
	je		.L011 // alg==0

	// tc==n ++??
	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm0, %ymm1, %ymm13
	vunpckhpd	%ymm2, %ymm3, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm12
	vunpcklpd	%ymm7, %ymm6, %ymm14
	vunpckhpd	%ymm4, %ymm5, %ymm13
	vunpckhpd	%ymm6, %ymm7, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm4
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm6
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm5
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm7

	vunpcklpd	%ymm9, 	%ymm8, 	%ymm12
	vunpcklpd	%ymm11, %ymm10, %ymm14
	vunpckhpd	%ymm8, 	%ymm9, 	%ymm13
	vunpckhpd	%ymm10, %ymm11, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm8
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm10
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm9
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm11

	cmpl	$1, %r10d // alg
	jne		.L010 // alg==-1

	movq	%r11, %r15 // C1 <- C0
	addq	%r12, %r15 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r15, %r14 // C2 <- C1
	addq	%r12, %r14 // C2 <- C1 + 4*sdc*sizeof(double)


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

	vmovapd		0(%r14), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11

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

	vmovapd		0(%r15), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

	vmovapd		0(%r14), %ymm15
	vsubpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vsubpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vsubpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vsubpd		%ymm11, %ymm15, %ymm11

.L012: // tran

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpckhpd	%ymm1, %ymm0, %ymm13
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm3, %ymm2, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm12
	vunpckhpd	%ymm5, %ymm4, %ymm13
	vunpcklpd	%ymm7, %ymm6, %ymm14
	vunpckhpd	%ymm7, %ymm6, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm4
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm6
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm5
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm7
	
	vunpcklpd	%ymm9, %ymm8, %ymm12
	vunpckhpd	%ymm9, %ymm8, %ymm13
	vunpcklpd	%ymm11, %ymm10, %ymm14
	vunpckhpd	%ymm11, %ymm10, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm8
	vperm2f128	$0x31, %ymm14, %ymm12, %ymm10
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm9
	vperm2f128	$0x31, %ymm15, %ymm13, %ymm11

	ret

.L011: // alg==0
	
	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11

	ret

	.size	inner_blender_tn_12x4_lib4, .-inner_blender_tn_12x4_lib4


// common inner routine with file scope
//
// blender for tc==t and td==t
//
// input arguments:
// r10d  <- alg
// r11   <- C
// r12  <- sdc*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8 <- [d80 d91 da2 db3]
// ymm9 <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d  <- alg
// r11   <- C+4*sdc*sizeof(double)
// r12  <- 4*sdc*sizeof(double)
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8 <- [d80 d90 da0 db0]
// ymm9 <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

	.p2align 4,,15
	.type inner_blender_tt_12x4_lib4, @function
inner_blender_tt_12x4_lib4:
	

	// tc==t

	vunpcklpd	%ymm1, %ymm0, %ymm12
	vunpcklpd	%ymm3, %ymm2, %ymm14
	vunpckhpd	%ymm0, %ymm1, %ymm13
	vunpckhpd	%ymm2, %ymm3, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm0
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm2
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm1
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm3

	vunpcklpd	%ymm5, %ymm4, %ymm12
	vunpcklpd	%ymm7, %ymm6, %ymm14
	vunpckhpd	%ymm4, %ymm5, %ymm13
	vunpckhpd	%ymm6, %ymm7, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm4
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm6
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm5
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm7

	vunpcklpd	%ymm9, 	%ymm8, 	%ymm12
	vunpcklpd	%ymm11, %ymm10, %ymm14
	vunpckhpd	%ymm8, 	%ymm9, 	%ymm13
	vunpckhpd	%ymm10, %ymm11, %ymm15

	vperm2f128	$0x20, %ymm14, %ymm12, %ymm8
	vperm2f128	$0x31, %ymm12, %ymm14, %ymm10
	vperm2f128	$0x20, %ymm15, %ymm13, %ymm9
	vperm2f128	$0x31, %ymm13, %ymm15, %ymm11

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

	vmovapd		0(%r15), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

	vmovapd		0(%r14), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11

	jmp		.L012 // tran

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

	vmovapd		0(%r15), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

	vmovapd		0(%r14), %ymm15
	vsubpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vsubpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vsubpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vsubpd		%ymm11, %ymm15, %ymm11

.L014: // return==0
	
	ret

	.size	inner_blender_tt_12x4_lib4, .-inner_blender_tt_12x4_lib4



//                                 rdi    rsi         rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24  
// void kernel_dgemm_ntnn_12x4_lib4(int k, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
	.globl kernel_dgemm_ntnn_12x4_lib4
	.type kernel_dgemm_ntnn_12x4_lib4, @function
kernel_dgemm_ntnn_12x4_lib4:
	
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11

	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13 // B

	call inner_kernel_dgemm_add_nt_12x4_lib4

	// call inner blender nn

	movl	%r8d, %r10d // alg
	movq	%r9, %r11 // C
	movl	STACKSIZE +  8(%rsp), %r12d // sdc
	sall	$5, %r12d // 4*sdc*sizeof(double)
	movslq	%r12d, %r12

	call inner_blender_nn_12x4_lib4

	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	call inner_store_12x4_lib4

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

	.size	kernel_dgemm_ntnn_12x4_lib4, .-kernel_dgemm_ntnn_12x4_lib4

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
	.type inner_store_12x4_lib4, @function
inner_store_12x4_lib4:
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)
	movq	%r15, %r14 // D2 <- D1
	addq	%r11, %r14 // D2 <- D1 + 4*sdd*sizeof(double)

	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4,  0(%r15)
	vmovapd %ymm5, 32(%r15)
	vmovapd %ymm6, 64(%r15)
	vmovapd %ymm7, 96(%r15)

	vmovapd %ymm8,  0(%r14)
	vmovapd %ymm9, 32(%r14)
	vmovapd %ymm10, 64(%r14)
	vmovapd %ymm11, 96(%r14)

	ret

	.size	inner_store_12x4_lib4, .-inner_store_12x4_lib4




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



	.section	.note.GNU-stack,"",@progbits
