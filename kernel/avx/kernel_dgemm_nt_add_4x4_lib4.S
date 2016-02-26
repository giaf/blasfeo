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
// edi   <- k
// rsi   <- A
// rdx   <- B
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
// edi   <- dirty
// rsi   <- A+4*k*sizeof(double)
// rdx   <- B+4*k*sizeof(double)
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
	
	cmpl	$0, %edi
	jle		.L002 // return

	// prefetch
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vmovapd 0(%rdx), %ymm12 // B[0]

	cmpl	$4, %edi
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 32(%rdx), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%rsi), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %edi
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%rdx), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%rsi), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%rdx), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%rsi), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %rdx
	addq	$128, %rsi

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


	// unroll 3
	vmovapd 0(%rdx), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 0(%rsi), %ymm8 // A0[0]
	cmpl	$3, %edi

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %edi
	jle		.L004 // clean1

	// unroll 0
	vmovapd 32(%rdx), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%rsi), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

//	subl	$4, %edi
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%rdx), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%rsi), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%rdx), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%rsi), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
//	addq	$128, %rdx
//	addq	$128, %rsi

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


	// unroll 3
//	vmovapd 0(%rdx), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
//	vmovapd 0(%rsi), %ymm8 // A0[0]
//	cmpl	$3, %edi

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


//	jmp		.L002 // return
	ret


.L004: // consider clean1-up loop

	cmpl	$0, %edi
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%rdx), %ymm12 // B[0]
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	addq	$32, %rsi

	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	addq	$32, %rdx

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	subl	$1, %edi

	cmpl	$0, %edi

	jg		.L003 // clean up loop 


.L002: // return

	ret

	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// edi   <- k
// rsi   <- A
// rdx   <- B
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
// edi   <- max(k-4,0)
// rsi   <- A+4*4*sizeof(double)
// rdx   <- B+4*4*sizeof(double)
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
	.type inner_edge_dtrmm_add_nt_lu_4x4_lib4, @function
inner_edge_dtrmm_add_nt_lu_4x4_lib4:
	
	cmpl	$0, %edi
	jle		.L019 // return

	vxorpd	%ymm9, %ymm9, %ymm9

	cmpl	$3, %edi
	jle		.L020 // consider clean-up

	// prefetch
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vmovapd 0(%rdx), %ymm12 // B[0]
	vblendpd	$0x1, %ymm12, %ymm9, %ymm12

	// unroll 0
	vmovapd 32(%rdx), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%rsi), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vblendpd	$0x3, %ymm13, %ymm9, %ymm13

	// unroll 1
	vmovapd 64(%rdx), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%rsi), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	vblendpd	$0x7, %ymm12, %ymm9, %ymm12
	subl	$4, %edi

	// unroll 2
	vmovapd 96(%rdx), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%rsi), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %rdx

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	addq	$128, %rsi


	// unroll 3
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	ret

.L020:

	// unroll 0
	vmovapd 	0(%rdx), %ymm12 // B[0]
	vblendpd	$0x1, %ymm12, %ymm9, %ymm12
	vmovapd 	0(%rsi), %ymm8 // A0[0]

	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		$32, %rdx

	vshufpd		$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		$32, %rsi

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
	subl		$1, %edi

	vshufpd 	$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2

	cmpl	$0, %edi
	jle		.L019 // loop

	// unroll 0
	vmovapd 	0(%rdx), %ymm12 // B[0]
	vblendpd	$0x3, %ymm12, %ymm9, %ymm12
	vmovapd 	0(%rsi), %ymm8 // A0[0]

	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		$32, %rdx

	vshufpd		$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		$32, %rsi

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
	subl		$1, %edi

	vshufpd 	$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2

	cmpl	$0, %edi
	jle		.L019 // loop

	// unroll 0
	vmovapd 	0(%rdx), %ymm12 // B[0]
	vblendpd	$0x7, %ymm12, %ymm9, %ymm12
	vmovapd 	0(%rsi), %ymm8 // A0[0]

	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	addq		$32, %rdx

	vshufpd		$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	addq		$32, %rsi

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
	subl		$1, %edi

	vshufpd 	$0x5, %ymm12, %ymm12, %ymm14
	vmulpd		%ymm8, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2

.L019: // return
	
	ret

	.size	inner_edge_dtrmm_add_nt_lu_4x4_lib4, .-inner_edge_dtrmm_add_nt_lu_4x4_lib4





// common inner routine with file scope
//
// blender for tc==n and td==n
//
// input arguments:
// ecx  <- alg
// r8   <- C
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
// ecx  <- alg
// r8   <- C
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

	cmpl	$0, %ecx // alg
	je		.L005 // return

	cmpl	$1, %ecx // alg
	jne		.L006 // alg==-1

	// alg==1
	vmovapd		0(%r8), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	ret // return

.L006: // alg==-1

	// alg==-1
	vmovapd		0(%r8), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L005: // return
	
	ret

	.size	inner_blender_nn_4x4_lib4, .-inner_blender_nn_4x4_lib4





// common inner routine with file scope
//
// blender for tc==n and td==t
//
// input arguments:
// ecx  <- alg
// r8   <- C
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
// ecx  <- alg
// r8   <- C
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
	

	cmpl	$0, %ecx // alg
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

	cmpl	$1, %ecx // alg
	jne		.L007 // alg==-1

	// alg==1
	vmovapd		0(%r8), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	jmp		.L009 // tran

.L007: // alg==-1

	// alg==-1
	vmovapd		0(%r8), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
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
// ecx  <- alg
// r8   <- C
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
// ecx  <- alg
// r8   <- C
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
	

	cmpl	$0, %ecx // alg
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

	cmpl	$1, %ecx // alg
	jne		.L010 // alg==-1

	// alg==1
	vmovapd		0(%r8), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	jmp		.L012 // tran

.L010: // alg==-1

	// alg==-1
	vmovapd		0(%r8), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
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
// ecx  <- alg
// r8   <- C
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
// ecx  <- alg
// r8   <- C
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

	cmpl	$0, %ecx // alg
	je		.L014 // return

	cmpl	$1, %ecx // alg
	jne		.L013 // alg==-1

	// alg==1
	vmovapd		0(%r8), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	ret

.L013: // alg==-1

	// alg==-1
	vmovapd		0(%r8), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r8), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r8), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r8), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L014: // return==0
	
	ret

	.size	inner_blender_tt_4x4_lib4, .-inner_blender_tt_4x4_lib4





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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nn_4x4_lib4


	// store

	vmovapd %ymm0,  0(%r9)
	vmovapd %ymm1, 32(%r9)
	vmovapd %ymm2, 64(%r9)
	vmovapd %ymm3, 96(%r9)


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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nn_4x4_lib4


	// store

	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r9)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r9)
	jl			.L015 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r9)
	je			.L015 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r9)


.L015:
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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nt_4x4_lib4


	// store

	vmovapd %ymm0,  0(%r9)
	vmovapd %ymm1, 32(%r9)
	vmovapd %ymm2, 64(%r9)
	vmovapd %ymm3, 96(%r9)

	vmovapd %ymm4, 128(%r9)
	vmovapd %ymm5, 160(%r9)
	vmovapd %ymm6, 192(%r9)
	vmovapd %ymm7, 224(%r9)


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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nt_4x4_lib4


	// store

	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$6, %r11d
	vmaskmovpd	%ymm0, %ymm15,   0(%r9)
	jl			.L017 // end
	cmpl		$7, %r11d
	vmaskmovpd	%ymm1, %ymm15,  32(%r9)
	jl			.L017 // end
	vmaskmovpd	%ymm2, %ymm15,  64(%r9)
	je			.L017 // end
	vmaskmovpd	%ymm3, %ymm15,  96(%r9)


.L017:
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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_tn_4x4_lib4


	// store

	vmovapd		%ymm0, 0(%r9)
	vmovapd		%ymm1, 32(%r9)
	vmovapd		%ymm2, 64(%r9)
	vmovapd		%ymm3, 96(%r9)


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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_tn_4x4_lib4


	// store

	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r9)
	jl			.L016 // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r9)
	jl			.L016 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r9)
	je			.L016 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r9)


.L016:
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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender tt

	call inner_blender_tt_4x4_lib4


	// store

	vmovapd %ymm0,  0(%r9)
	vmovapd %ymm1, 32(%r9)
	vmovapd %ymm2, 64(%r9)
	vmovapd %ymm3, 96(%r9)


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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender tt

	call inner_blender_tt_4x4_lib4


	// store

	movl	STACKSIZE + 32(%rsp), %r11d // km 
	movl	STACKSIZE + 40(%rsp), %r12d // kn 

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$6, %r11d
	vmaskmovpd	%ymm0, %ymm15,   0(%r9)
	jl			.L017 // end
	cmpl		$7, %r11d
	vmaskmovpd	%ymm1, %ymm15,  32(%r9)
	jl			.L017 // end
	vmaskmovpd	%ymm2, %ymm15,  64(%r9)
	je			.L017 // end
	vmaskmovpd	%ymm3, %ymm15,  96(%r9)



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

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nn_4x4_lib4


	// store

	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r9)
	jl			.L018 // end
	cmpl		$3, %r12d
	vmovapd		32(%r9), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmaskmovpd	%ymm1, %ymm15, 32(%r9)
	jl			.L018 // end
	vmovapd		64(%r9), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmaskmovpd	%ymm2, %ymm15, 64(%r9)
	je			.L018 // end
	vmovapd		96(%r9), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmaskmovpd	%ymm3, %ymm15, 96(%r9)


.L018:
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


	// call inner dgemm kernel nt

	call inner_edge_dtrmm_add_nt_lu_4x4_lib4


	// call inner dgemm kernel nt

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nn_4x4_lib4


	// store

	vmovapd %ymm0,  0(%r9)
	vmovapd %ymm1, 32(%r9)
	vmovapd %ymm2, 64(%r9)
	vmovapd %ymm3, 96(%r9)


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


	// call inner dgemm kernel nt

	call inner_edge_dtrmm_add_nt_lu_4x4_lib4


	// call inner dgemm kernel nt

	call inner_kernel_dgemm_add_nt_4x4_lib4


	// call inner blender nn

	call inner_blender_nn_4x4_lib4


	// store

	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r9)
	jl			.L021 // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r9)
	jl			.L021 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r9)
	je			.L021 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r9)


.L021:
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

