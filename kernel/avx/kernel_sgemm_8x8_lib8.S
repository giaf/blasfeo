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

#define STACKSIZE 96



#if defined(OS_LINUX)
	.text
#elif defined(OS_MAC)
	.section	__TEXT,__text,regular,pure_instructions
#endif

// common inner routine with file scope
//
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
// ymm4  <- []
// ymm5  <- []
// ymm6  <- []
// ymm7  <- []
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
// ymm0  <- [d00 d10 d20 d30 d40 d50 d60 d70]
// ymm1  <- [d01 d11 d21 d31 d41 d51 d61 d71]
// ymm2  <- [d02 d12 d22 d32 d42 d52 d62 d72]
// ymm3  <- [d03 d13 d23 d33 d43 d53 d63 d73]
// ymm4  <- []
// ymm5  <- []
// ymm6  <- []
// ymm7  <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_SGEMM_ADD_NT_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgemm_add_nt_8x8_lib8, @function
inner_kernel_sgemm_add_nt_8x8_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgemm_add_nt_8x8_lib8:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1
	subl	$4, %r10d

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	16(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	32(%r12), %ymm14 // B
	vmovaps			32(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 1
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	48(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	64(%r12), %ymm14 // B
	vmovaps			64(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 2
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	80(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	96(%r12), %ymm14 // B
	vmovaps			96(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 3
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1
	addq	$128, %r11

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	112(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5
	addq	$128, %r12

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6
	cmpl	$4, %r10d

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0

	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1
	subl	$4, %r10d

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	16(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	32(%r12), %ymm14 // B
	vmovaps			32(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 1
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	48(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	64(%r12), %ymm14 // B
	vmovaps			64(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 2
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	80(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
	vbroadcastf128	96(%r12), %ymm14 // B
	vmovaps			96(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	// unroll 3
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1
	addq	$128, %r11

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3
	vbroadcastf128	112(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5
	addq	$128, %r12

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6
//	cmpl	$4, %r10d

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6
	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7

	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7
//	vbroadcastf128	0(%r12), %ymm14 // B
//	vmovaps			0(%r11), %ymm12 // A
//	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0


	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B0
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm0, %ymm0
	subl	$1, %r10d

	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B1
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm1, %ymm1
	addq	$32, %r11

	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B2
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm2, %ymm2

	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B3
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm3, %ymm3

	vbroadcastf128	16(%r12), %ymm14 // B
	vshufps			$0x00, %ymm14, %ymm14, %ymm15 // B4
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm4, %ymm4
	addq	$32, %r12

	vshufps			$0x55, %ymm14, %ymm14, %ymm15 // B5
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm5, %ymm5
	cmpl	$0, %r10d

	vshufps			$0xaa, %ymm14, %ymm14, %ymm15 // B6
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm6, %ymm6

	vshufps			$0xff, %ymm14, %ymm14, %ymm15 // B7
	vmulps			%ymm12, %ymm15, %ymm11
	vaddps			%ymm11, %ymm7, %ymm7

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgemm_add_nt_8x8_lib8, .-inner_kernel_sgemm_add_nt_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_8x8_lib8, @function
inner_scale_ab_8x8_lib8:
#elif defined(OS_MAC)
_inner_scale_ab_8x8_lib8:
#endif
#endif
	
	// alpha
	vbroadcastss	0(%r10), %ymm15

	vmulps		%ymm0, %ymm15, %ymm0
	vmulps		%ymm1, %ymm15, %ymm1
	vmulps		%ymm2, %ymm15, %ymm2
	vmulps		%ymm3, %ymm15, %ymm3
	vmulps		%ymm4, %ymm15, %ymm4
	vmulps		%ymm5, %ymm15, %ymm5
	vmulps		%ymm6, %ymm15, %ymm6
	vmulps		%ymm7, %ymm15, %ymm7

	// beta
	vbroadcastss	0(%r11), %ymm14

	vxorps		%ymm15, %ymm15, %ymm15 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovaps		0(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm0, %ymm15, %ymm0
	vmovaps		32(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm1, %ymm15, %ymm1
	vmovaps		64(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm2, %ymm15, %ymm2
	vmovaps		96(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm3, %ymm15, %ymm3
	vmovaps		128(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm4, %ymm15, %ymm4
	vmovaps		160(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm5, %ymm15, %ymm5
	vmovaps		192(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm6, %ymm15, %ymm6
	vmovaps		224(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_8x8_lib8, .-inner_scale_ab_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
//
// output arguments:
// r10  <- D
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x8_lib8, @function
inner_store_8x8_lib8:
#elif defined(OS_MAC)
_inner_store_8x8_lib8:
#endif
#endif
	
	vmovaps 	%ymm0,  0(%r10)
	vmovaps 	%ymm1, 32(%r10)
	vmovaps 	%ymm2, 64(%r10)
	vmovaps 	%ymm3, 96(%r10)
	vmovaps 	%ymm4, 128(%r10)
	vmovaps 	%ymm5, 160(%r10)
	vmovaps 	%ymm6, 192(%r10)
	vmovaps 	%ymm7, 224(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x8_lib8, .-inner_store_8x8_lib8
#endif
#endif





//                               rdi    rsi           rdx       rcx       r8           r9        rsp+8
// void kernel_sgemm_nt_8x8_lib8(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nt_8x8_lib8
	.type kernel_sgemm_nt_8x8_lib8, @function
kernel_sgemm_nt_8x8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nt_8x8_lib8
_kernel_sgemm_nt_8x8_lib8:
#endif
	
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

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_ADD_NT_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_add_nt_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_add_nt_8x8_lib8
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_scale_ab_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x8_lib8
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_store_8x8_lib8
#endif
#endif


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

#if defined(OS_LINUX)
	.size	kernel_sgemm_nt_8x8_lib8, .-kernel_sgemm_nt_8x8_lib8
#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#endif

#if defined(OS_LINUX)
	.align 32
.LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
	.align 5
LC00: // { 7.5 6.5 5.5 4.5 3.5 2.5 1.5 0.5 }
#endif
	.long	1056964608
	.long	1069547520
	.long	1075838976
	.long	1080033280
	.long	1083179008
	.long	1085276160
	.long	1087373312
	.long	1089470464

#if defined(OS_LINUX)
	.align 32
.LC01: // { 15.5 14.5 13.5 12.5 11.5 10.5 9.5 8.5 }
#elif defined(OS_MAC)
	.align 5
LC01: // { 15.5 14.5 13.5 12.5 11.5 10.5 9.5 8.5 }
#endif
	.long	1091043328
	.long	1092091904
	.long	1093140480
	.long	1094189056
	.long	1095237632
	.long	1096286208
	.long	1097334784
	.long	1098383360

#if defined(OS_LINUX)
	.align 32
.LC02: // { 23.5 22.5 21.5 20.5 19.5 18.5 17.5 16.5 }
#elif defined(OS_MAC)
	.align 5
LC02: // { 23.5 22.5 21.5 20.5 19.5 18.5 17.5 16.5 }
#endif
	.long	1099169792
	.long	1099694080
	.long	1100218368
	.long	1100742656
	.long	1101266944
	.long	1101791232
	.long	1102315520
	.long	1102839808

#if defined(OS_LINUX)
	.align 32
.LC03: // { -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC03: // { -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 }
#endif
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	3212836864
	.long	3212836864



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

