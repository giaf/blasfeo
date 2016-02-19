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
// edi   <- km
// rsi   <- A
// edx   <- sda
// rcx   <- B
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
// edi   <- dirty
// rsi   <- dirty
// edx   <- 4*sda*sizeof(double)
// rcx   <- dirty
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
	
	cmpl	$0, %edi
	jle		.L002 // return

	movq	%rsi, %r15 // A1 <- A0
	sall	$5, %edx // 4*sda*sizeof(double)
	movslq	%edx, %rdx
	addq	%rdx, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vmovapd 0(%rcx), %ymm12 // B[0]

	cmpl	$4, %edi
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 32(%rcx), %ymm13 // B[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%rsi), %ymm10 // A0[4]
	vfmadd231pd	%ymm9, %ymm14, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm7

	subl	$4, %edi
	vfmadd231pd	%ymm8, %ymm14, %ymm2
	vfmadd231pd	%ymm9, %ymm14, %ymm6

	// unroll 1
	vmovapd 64(%rcx), %ymm12 // B[8]
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm1
	vmovapd 64(%rsi), %ymm8 // A0[8]
	vfmadd231pd	%ymm11, %ymm1, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vfmadd231pd	%ymm10, %ymm1, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm7

	vfmadd231pd	%ymm10, %ymm1, %ymm2
	vfmadd231pd	%ymm11, %ymm1, %ymm6

	// unroll 2
	vmovapd 96(%rcx), %ymm13 // B[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%rsi), %ymm10 // A0[12]
	vfmadd231pd	%ymm9, %ymm14, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %rcx
	vfmadd231pd	%ymm9, %ymm12, %ymm7
	addq	$128, %rsi

	vfmadd231pd	%ymm8, %ymm14, %ymm2
	addq	$128, %r15
	vfmadd231pd	%ymm9, %ymm14, %ymm6


	// unroll 3
	vmovapd 0(%rcx), %ymm12 // B[0]
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm1
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vfmadd231pd	%ymm11, %ymm1, %ymm5
	cmpl	$3, %edi

	vmovapd 0(%r15), %ymm9 // A1[0]
	vfmadd231pd	%ymm10, %ymm1, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm7

	vfmadd231pd	%ymm10, %ymm1, %ymm2
	vfmadd231pd	%ymm11, %ymm1, %ymm6

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %edi
	jle		.L004 // clean1

	// unroll 0
	vmovapd 32(%rcx), %ymm13 // B[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 32(%rsi), %ymm10 // A0[4]
	vfmadd231pd	%ymm9, %ymm14, %ymm5

	vmovapd 32(%r15), %ymm11 // A1[4]
	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm7

//	subl	$4, %edi
	vfmadd231pd	%ymm8, %ymm14, %ymm2
	vfmadd231pd	%ymm9, %ymm14, %ymm6

	// unroll 1
	vmovapd 64(%rcx), %ymm12 // B[8]
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm1
	vmovapd 64(%rsi), %ymm8 // A0[8]
	vfmadd231pd	%ymm11, %ymm1, %ymm5

	vmovapd 64(%r15), %ymm9 // A1[8]
	vfmadd231pd	%ymm10, %ymm1, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm7

	vfmadd231pd	%ymm10, %ymm1, %ymm2
	vfmadd231pd	%ymm11, %ymm1, %ymm6

	// unroll 2
	vmovapd 96(%rcx), %ymm13 // B[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vfmadd231pd	%ymm8, %ymm14, %ymm1
	vmovapd 96(%rsi), %ymm10 // A0[12]
	vfmadd231pd	%ymm9, %ymm14, %ymm5

	vmovapd 96(%r15), %ymm11 // A1[12]
	vfmadd231pd	%ymm8, %ymm12, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
//	addq	$128, %rcx
	vfmadd231pd	%ymm9, %ymm12, %ymm7
//	addq	$128, %rsi

	vfmadd231pd	%ymm8, %ymm14, %ymm2
//	addq	$128, %r15
	vfmadd231pd	%ymm9, %ymm14, %ymm6


	// unroll 3
//	vmovapd 0(%rcx), %ymm12 // B[0]
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm4

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm1
//	vmovapd 0(%rsi), %ymm8 // A0[0]
	vfmadd231pd	%ymm11, %ymm1, %ymm5
//	cmpl	$3, %edi

//	vmovapd 0(%r15), %ymm9 // A1[0]
	vfmadd231pd	%ymm10, %ymm1, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm11, %ymm1, %ymm7

	vfmadd231pd	%ymm10, %ymm1, %ymm2
	vfmadd231pd	%ymm11, %ymm1, %ymm6


//	jmp		.L002 // return
	ret


.L004: // consider clean1-up loop

	cmpl	$0, %edi
	jle		.L002 // return

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%rcx), %ymm12 // B[0]
	vmovapd 0(%rsi), %ymm8 // A0[0]
	vmovapd 0(%r15), %ymm9 // A1[0]
	vfmadd231pd	%ymm8, %ymm12, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vfmadd231pd	%ymm9, %ymm12, %ymm4
	addq	$32, %rsi

	vfmadd231pd	%ymm8, %ymm14, %ymm1
	addq	$32, %rdx
	vfmadd231pd	%ymm9, %ymm14, %ymm5

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm8, %ymm14, %ymm3
	addq	$32, %r15
	vfmadd231pd	%ymm9, %ymm14, %ymm7

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm8, %ymm14, %ymm2
	subl	$1, %edi
	vfmadd231pd	%ymm9, %ymm14, %ymm6

	cmpl	$0, %edi

	jg		.L003 // clean up loop 


.L002: // return

	ret

	.size	inner_kernel_dgemm_add_nt_8x4_lib4, .-inner_kernel_dgemm_add_nt_8x4_lib4





// common inner routine with file scope
//
// blender for tc==n and td==n
//
// input arguments:
// r8d  <- alg
// r9   <- C
// r10d <- sdc
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
// r8d  <- alg
// r9   <- C
// r10d <- 4*sdc*sizeof(double)
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

	cmpl	$0, %r8d // alg
	je		.L005 // return

	movq	%r9, %r15 // C1 <- C0
	sall	$5, %r10d // 4*sdc*sizeof(double)
	movslq	%r10d, %r10
	addq	%r10, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$1, %r8d // alg
	jne		.L006 // alg==-1

	// alg==1
	vmovapd		0(%r9), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
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
	vmovapd		0(%r9), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
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
// r8d  <- alg
// r9   <- C
// r10d <- sdc
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
// r8d  <- alg
// r9   <- C
// r10d <- 4*sdc*sizeof(double)
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
	

	cmpl	$0, %r8d // alg
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

	movq	%r9, %r15 // C1 <- C0
	sall	$5, %r10d // 4*sdc*sizeof(double)
	movslq	%r10d, %r10
	addq	%r10, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$1, %r8d // alg
	jne		.L007 // alg==-1

	// alg==1
	vmovapd		0(%r9), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
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
	vmovapd		0(%r9), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
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
// r8d  <- alg
// r9   <- C
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
// r8d  <- alg
// r9   <- C
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
	

	cmpl	$0, %r8d // alg
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

	cmpl	$1, %r8d // alg
	jne		.L010 // alg==-1

	// alg==1
	vmovapd		0(%r9), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		128(%r9), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r9), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r9), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r9), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

	jmp		.L012 // tran

.L010: // alg==-1

	// alg==-1
	vmovapd		0(%r9), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

	vmovapd		128(%r9), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r9), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r9), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r9), %ymm15
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
// r8d  <- alg
// r9   <- C
// r10d <- sdc
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
// r8d  <- alg
// r9   <- C
// r10d <- 4*sdc*sizeof(double)
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

	cmpl	$0, %r8d // alg
	je		.L014 // return

	cmpl	$1, %r8d // alg
	jne		.L013 // alg==-1

	// alg==1
	vmovapd		0(%r9), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		128(%r9), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r9), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r9), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r9), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

	ret

.L013: // alg==-1

	// alg==-1
	vmovapd		0(%r9), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

	vmovapd		128(%r9), %ymm15
	vsubpd		%ymm4, %ymm15, %ymm4
	vmovapd		160(%r9), %ymm15
	vsubpd		%ymm5, %ymm15, %ymm5
	vmovapd		192(%r9), %ymm15
	vsubpd		%ymm6, %ymm15, %ymm6
	vmovapd		224(%r9), %ymm15
	vsubpd		%ymm7, %ymm15, %ymm7

.L014: // return==0
	
	ret

	.size	inner_blender_tt_8x4_lib4, .-inner_blender_tt_8x4_lib4





//                                 rdi     rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24  
// void kernel_dgemm_ntnn_8x4_lib4(int km, double *A, int sda, double *B, int alg, double *C, int sdc, double *D, int sdd);

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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d // sdc

	call inner_blender_nn_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	movq	%r10, %r15 // D1 <- D0
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	addq	%r11, %r15 // D1 <- D0 + 4*sdd

	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4,  0(%r15)
	vmovapd %ymm5, 32(%r15)
	vmovapd %ymm6, 64(%r15)
	vmovapd %ymm7, 96(%r15)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d // sdc

	call inner_blender_nn_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	movq	%r10, %r15 // D1 <- D0
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	addq	%r11, %r15 // D1 <- D0 + 4*sdd

	movl	STACKSIZE +  32(%rsp), %r11d // km 
	movl	STACKSIZE +  40(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC03(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r15)
	jl			.L015 // end
	cmpl		$3, %r12d
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r15)
	jl			.L015 // end
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r15)
	je			.L015 // end
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r15)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d // sdc

	call inner_blender_nt_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4, 128(%r10)
	vmovapd %ymm5, 160(%r10)
	vmovapd %ymm6, 192(%r10)
	vmovapd %ymm7, 224(%r10)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	movl	STACKSIZE +  8(%rsp), %r10d // sdc

	call inner_blender_nt_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 32(%rsp), %r11d // km 
	movl	STACKSIZE + 40(%rsp), %r12d // kn 

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$6, %r11d
	vmaskmovpd	%ymm0, %ymm15,   0(%r10)
	vmaskmovpd	%ymm1, %ymm15,  32(%r10)
	vmaskmovpd	%ymm2, %ymm15,  64(%r10)
	vmaskmovpd	%ymm3, %ymm15,  96(%r10)
	vmaskmovpd	%ymm4, %ymm15, 128(%r10)
	jl			.L017 // end
	cmpl		$7, %r11d
	vmaskmovpd	%ymm5, %ymm15, 160(%r10)
	jl			.L017 // end
	vmaskmovpd	%ymm6, %ymm15, 192(%r10)
	je			.L017 // end
	vmaskmovpd	%ymm7, %ymm15, 224(%r10)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	call inner_blender_tn_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	movq	%r10, %r15 // D1 <- D0
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	addq	%r11, %r15 // D1 <- D0 + 4*sdd

	vmovapd		%ymm0, 0(%r10)
	vmovapd		%ymm1, 32(%r10)
	vmovapd		%ymm2, 64(%r10)
	vmovapd		%ymm3, 96(%r10)

	vmovapd		%ymm4, 0(%r15)
	vmovapd		%ymm5, 32(%r15)
	vmovapd		%ymm6, 64(%r15)
	vmovapd		%ymm7, 96(%r15)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender nn

	call inner_blender_tn_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	movq	%r10, %r15 // D1 <- D0
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11
	addq	%r11, %r15 // D1 <- D0 + 4*sdd

	movl	STACKSIZE +  32(%rsp), %r11d // km 
	movl	STACKSIZE +  40(%rsp), %r12d // kn 

	vcvtsi2sd	%r11d, %xmm15, %xmm15
	vmovupd		.LC03(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmovapd		%ymm0, 0(%r10)
	vmaskmovpd	%ymm4, %ymm15,  0(%r15)
	jl			.L016 // end
	cmpl		$3, %r12d
	vmovapd		%ymm1, 32(%r10)
	vmaskmovpd	%ymm5, %ymm15, 32(%r15)
	jl			.L016 // end
	vmovapd		%ymm2, 64(%r10)
	vmaskmovpd	%ymm6, %ymm15, 64(%r15)
	je			.L016 // end
	vmovapd		%ymm3, 96(%r10)
	vmaskmovpd	%ymm7, %ymm15, 96(%r15)


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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender tt

	call inner_blender_tt_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D

	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)

	vmovapd %ymm4, 128(%r10)
	vmovapd %ymm5, 160(%r10)
	vmovapd %ymm6, 192(%r10)
	vmovapd %ymm7, 224(%r10)



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

	call inner_kernel_dgemm_add_nt_8x4_lib4


	// call inner blender tt

	call inner_blender_tt_8x4_lib4


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // store address D
	movl	STACKSIZE + 32(%rsp), %r11d // km 
	movl	STACKSIZE + 40(%rsp), %r12d // kn 

	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$6, %r11d
	vmaskmovpd	%ymm0, %ymm15,   0(%r10)
	vmaskmovpd	%ymm1, %ymm15,  32(%r10)
	vmaskmovpd	%ymm2, %ymm15,  64(%r10)
	vmaskmovpd	%ymm3, %ymm15,  96(%r10)
	vmaskmovpd	%ymm4, %ymm15, 128(%r10)
	jl			.L017 // end
	cmpl		$7, %r11d
	vmaskmovpd	%ymm5, %ymm15, 160(%r10)
	jl			.L017 // end
	vmaskmovpd	%ymm6, %ymm15, 192(%r10)
	je			.L017 // end
	vmaskmovpd	%ymm7, %ymm15, 224(%r10)



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
