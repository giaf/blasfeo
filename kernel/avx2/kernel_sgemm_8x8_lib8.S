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
// ymm0  <- [d00 d11 d22 d33 d40 d51 d62 d73]
// ymm1  <- [d01 d10 d23 d32 d41 d50 d63 d72]
// ymm2  <- [d03 d12 d21 d30 d43 d52 d61 d70]
// ymm3  <- [d02 d13 d20 d31 d42 d53 d60 d71]
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
// ymm0  <- [d00 d11 d22 d33 d40 d51 d62 d73]
// ymm1  <- [d01 d10 d23 d32 d41 d50 d63 d72]
// ymm2  <- [d03 d12 d21 d30 d43 d52 d61 d70]
// ymm3  <- [d02 d13 d20 d31 d42 d53 d60 d71]
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

	// preload
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vbroadcastf128	16(%r12), %ymm15 // B
	vmovaps			32(%r11), %ymm13 // A

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vfmadd231ps		%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfmadd231ps		%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14 // 01 00 11 10

	vfmadd231ps		%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfmadd231ps		%ymm12, %ymm14, %ymm3
	vbroadcastf128	32(%r12), %ymm14 // B

	vfmadd231ps		%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm7
	vbroadcastf128	48(%r12), %ymm15 // B
	vmovaps			64(%r11), %ymm12 // A


	// unroll 1
	vfmadd231ps		%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm3
	vbroadcastf128	64(%r12), %ymm14 // B

	vfmadd231ps		%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm7
	vbroadcastf128	80(%r12), %ymm15 // B
	vmovaps			96(%r11), %ymm13 // A


	// unroll 2
	vfmadd231ps		%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm3
	vbroadcastf128	96(%r12), %ymm14 // B

	vfmadd231ps		%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm7
	vbroadcastf128	112(%r12), %ymm15 // B
	vmovaps			128(%r11), %ymm12 // A

	subl	$4, %r10d
	addq	$128, %r11
	addq	$128, %r12

	// unroll 3
	vfmadd231ps		%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm3
	vbroadcastf128	0(%r12), %ymm14 // B

	vfmadd231ps		%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm7
	vbroadcastf128	16(%r12), %ymm15 // B
	vmovaps			32(%r11), %ymm13 // A

	cmpl	$4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfmadd231ps		%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfmadd231ps		%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14 // 01 00 11 10

	vfmadd231ps		%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfmadd231ps		%ymm12, %ymm14, %ymm3
	vbroadcastf128	32(%r12), %ymm14 // B

	vfmadd231ps		%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm7
	vbroadcastf128	48(%r12), %ymm15 // B
	vmovaps			64(%r11), %ymm12 // A


	// unroll 1
	vfmadd231ps		%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm3
	vbroadcastf128	64(%r12), %ymm14 // B

	vfmadd231ps		%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm7
	vbroadcastf128	80(%r12), %ymm15 // B
	vmovaps			96(%r11), %ymm13 // A


	// unroll 2
	vfmadd231ps		%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm12, %ymm14, %ymm3
	vbroadcastf128	96(%r12), %ymm14 // B

	vfmadd231ps		%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm12, %ymm15, %ymm7
	vbroadcastf128	112(%r12), %ymm15 // B
//	vmovaps			128(%r11), %ymm12 // A

	subl	$4, %r10d
	addq	$128, %r11
	addq	$128, %r12

	// unroll 3
	vfmadd231ps		%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfmadd231ps		%ymm13, %ymm14, %ymm3
//	vbroadcastf128	0(%r12), %ymm14 // B

	vfmadd231ps		%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfmadd231ps		%ymm13, %ymm15, %ymm7
//	vbroadcastf128	16(%r12), %ymm15 // B
//	vmovaps			32(%r11), %ymm13 // A


//	cmpl	$4, %r10d
	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vfmadd231ps		%ymm12, %ymm14, %ymm0

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm1

	vshufps			$0x4e, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm2

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm3

	vbroadcastf128	16(%r12), %ymm14 // B
	vfmadd231ps		%ymm12, %ymm14, %ymm4

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm5

	vshufps			$0x4e, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm6

	subl	$1, %r10d
	addq	$32, %r11
	addq	$32, %r12

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfmadd231ps		%ymm12, %ymm14, %ymm7

	cmpl	$0, %r10d
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
// input arguments:
// r10d   <- k
// r11   <- A
// r12   <- B
// ymm0  <- [d00 d11 d22 d33 d40 d51 d62 d73]
// ymm1  <- [d01 d10 d23 d32 d41 d50 d63 d72]
// ymm2  <- [d03 d12 d21 d30 d43 d52 d61 d70]
// ymm3  <- [d02 d13 d20 d31 d42 d53 d60 d71]
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
// ymm0  <- [d00 d11 d22 d33 d40 d51 d62 d73]
// ymm1  <- [d01 d10 d23 d32 d41 d50 d63 d72]
// ymm2  <- [d03 d12 d21 d30 d43 d52 d61 d70]
// ymm3  <- [d02 d13 d20 d31 d42 d53 d60 d71]
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
	.macro INNER_KERNEL_SGEMM_SUB_NT_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgemm_sub_nt_8x8_lib8, @function
inner_kernel_sgemm_sub_nt_8x8_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgemm_sub_nt_8x8_lib8:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// preload
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vbroadcastf128	16(%r12), %ymm15 // B
	vmovaps			32(%r11), %ymm13 // A

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vfnmadd231ps	%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfnmadd231ps	%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14 // 01 00 11 10

	vfnmadd231ps	%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfnmadd231ps	%ymm12, %ymm14, %ymm3
	vbroadcastf128	32(%r12), %ymm14 // B

	vfnmadd231ps	%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm7
	vbroadcastf128	48(%r12), %ymm15 // B
	vmovaps			64(%r11), %ymm12 // A


	// unroll 1
	vfnmadd231ps	%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm3
	vbroadcastf128	64(%r12), %ymm14 // B

	vfnmadd231ps	%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm7
	vbroadcastf128	80(%r12), %ymm15 // B
	vmovaps			96(%r11), %ymm13 // A


	// unroll 2
	vfnmadd231ps	%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm3
	vbroadcastf128	96(%r12), %ymm14 // B

	vfnmadd231ps	%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm7
	vbroadcastf128	112(%r12), %ymm15 // B
	vmovaps			128(%r11), %ymm12 // A

	subl	$4, %r10d
	addq	$128, %r11
	addq	$128, %r12

	// unroll 3
	vfnmadd231ps	%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm3
	vbroadcastf128	0(%r12), %ymm14 // B

	vfnmadd231ps	%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm7
	vbroadcastf128	16(%r12), %ymm15 // B
	vmovaps			32(%r11), %ymm13 // A

	cmpl	$4, %r10d
	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfnmadd231ps	%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfnmadd231ps	%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14 // 01 00 11 10

	vfnmadd231ps	%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14 // 10 11 00 01

	vfnmadd231ps	%ymm12, %ymm14, %ymm3
	vbroadcastf128	32(%r12), %ymm14 // B

	vfnmadd231ps	%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm7
	vbroadcastf128	48(%r12), %ymm15 // B
	vmovaps			64(%r11), %ymm12 // A


	// unroll 1
	vfnmadd231ps	%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm3
	vbroadcastf128	64(%r12), %ymm14 // B

	vfnmadd231ps	%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm7
	vbroadcastf128	80(%r12), %ymm15 // B
	vmovaps			96(%r11), %ymm13 // A


	// unroll 2
	vfnmadd231ps	%ymm12, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm12, %ymm14, %ymm3
	vbroadcastf128	96(%r12), %ymm14 // B

	vfnmadd231ps	%ymm12, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm12, %ymm15, %ymm7
	vbroadcastf128	112(%r12), %ymm15 // B
//	vmovaps			128(%r11), %ymm12 // A

	subl	$4, %r10d
	addq	$128, %r11
	addq	$128, %r12

	// unroll 3
	vfnmadd231ps	%ymm13, %ymm14, %ymm0
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm1
	vshufps			$0x4e, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm2
	vshufps			$0xb1, %ymm14, %ymm14, %ymm14

	vfnmadd231ps	%ymm13, %ymm14, %ymm3
//	vbroadcastf128	0(%r12), %ymm14 // B

	vfnmadd231ps	%ymm13, %ymm15, %ymm4
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm5
	vshufps			$0x4e, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm6
	vshufps			$0xb1, %ymm15, %ymm15, %ymm15

	vfnmadd231ps	%ymm13, %ymm15, %ymm7
//	vbroadcastf128	16(%r12), %ymm15 // B
//	vmovaps			32(%r11), %ymm13 // A


//	cmpl	$4, %r10d
	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	// unroll 0
	vbroadcastf128	0(%r12), %ymm14 // B
	vmovaps			0(%r11), %ymm12 // A
	vfnmadd231ps	%ymm12, %ymm14, %ymm0

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm1

	vshufps			$0x4e, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm2

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm3

	vbroadcastf128	16(%r12), %ymm14 // B
	vfnmadd231ps	%ymm12, %ymm14, %ymm4

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm5

	vshufps			$0x4e, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm6

	subl	$1, %r10d
	addq	$32, %r11
	addq	$32, %r12

	vshufps			$0xb1, %ymm14, %ymm14, %ymm14
	vfnmadd231ps	%ymm12, %ymm14, %ymm7

	cmpl	$0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgemm_sub_nt_8x8_lib8, .-inner_kernel_sgemm_sub_nt_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- B
// r13   <- 4*sdb*sizeof(double)
// r14   <= dirty
// ymm0  <- []
// ymm1  <- []
// ymm2  <- []
// ymm3  <- []
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
// r12   <- B+(k/4)*sdb*sizeof(double)+(k%4)
// r13   <- 4*sdb*sizeof(double)
// r14   <= dirty
// ymm0  <- []
// ymm1  <- []
// ymm2  <- []
// ymm3  <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_SGEMM_ADD_NN_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_sgemm_add_nn_8x8_lib8, @function
inner_kernel_sgemm_add_nn_8x8_lib8:
#elif defined(OS_MAC)
_inner_kernel_sgemm_add_nn_8x8_lib8:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r14 // B_next <- B
	addq	%r13, %r14 // B_next <- B + 4*sda*sizeof(double)

	cmpl	$8, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%r14) // software prefetch
	prefetcht0	64(%r14) // software prefetch
	prefetcht0	128(%r14) // software prefetch
	prefetcht0	192(%r14) // software prefetch

	// unroll 0
	vmovaps			0(%r11), %ymm12 // A[0]
	vbroadcastss	0(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	32(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	64(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	96(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	128(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	160(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	192(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	224(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 1
	vmovaps			32(%r11), %ymm12 // A[0]
	vbroadcastss	4(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	36(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	68(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	100(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	132(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	164(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	196(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	228(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 2
	vmovaps			64(%r11), %ymm12 // A[0]
	vbroadcastss	8(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	40(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	72(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	104(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	136(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	168(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	200(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	232(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 3
	vmovaps			96(%r11), %ymm12 // A[0]
	vbroadcastss	12(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	44(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	76(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	108(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	140(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	172(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	204(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	236(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 4
	vmovaps			128(%r11), %ymm12 // A[0]
	vbroadcastss	16(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	48(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	80(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	112(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	144(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	176(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	208(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	240(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 5
	vmovaps			160(%r11), %ymm12 // A[0]
	vbroadcastss	20(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	52(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	84(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	116(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	148(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	180(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	212(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	244(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 6
	vmovaps			192(%r11), %ymm12 // A[0]
	vbroadcastss	24(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	56(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	88(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	120(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	152(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	184(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	216(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	248(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7


	// unroll 7
	vmovaps			224(%r11), %ymm12 // A[0]
	vbroadcastss	28(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	60(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	92(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	124(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	156(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	188(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	220(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	252(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7

	subl	$8, %r10d
	addq	$256, %r11

	mov		%r14, %r12
	addq	%r13, %r14

	cmpl	$7, %r10d
	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	// unroll 0
	vmovaps			0(%r11), %ymm12 // A[0]
	vbroadcastss	0(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	32(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	64(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	96(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	128(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	160(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	192(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	224(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7

	subl	$1, %r10d
	addq	$32, %r11
	addq	$4, %r12

	cmpl	$0, %r10d
	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_sgemm_add_nn_8x8_lib8, .-inner_kernel_sgemm_add_nn_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// edge for B unaligned
//
// input arguments:
// r10   <- k
// r11   <- A
// r12   <- B
// r13   <- bs*sdb*sizeof(double)
// r14   <- offB
// ymm0  <- []
// ymm1  <- []
// ymm2  <- []
// ymm3  <- []
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- k-(4-offB)
// r11   <- A+(4-offB)*bs*sizeof(double)
// r12   <- B-offB+bs*sdb*sizeof(double)
// r13   <- bs*sdb*sizeof(double)
// r14   <- offB
// ymm0  <- []
// ymm1  <- []
// ymm2  <- []
// ymm3  <- []
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_SGEMM_ADD_NN_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_sgemm_add_nn_8x8_lib8, @function
inner_edge_sgemm_add_nn_8x8_lib8:
#elif defined(OS_MAC)
_inner_edge_sgemm_add_nn_8x4_lib8:
#endif
#endif
	
	cmpl			$0, %r14d // offset==0
	jle				2f // end

	cmpl			$0, %r10d // k==0
	jle				2f // end

	movl			$8, %ebx
	subl			%r14d, %ebx // 8-offsetB
	cmpl			%r10d, %ebx
//	jle				0f
//	movl			%r10d, %ebx // kend=min(k,8-offsetB)
//0:
	cmovgl			%r10d, %ebx // kend=min(k,8-offsetB)

	movl			%r14d, %eax
	sall			$2, %eax // offsetB*sizeof(float)
	addq			%rax, %r12 // B+offsetB*sizeof(float)

	// unroll 0
	vmovaps			0(%r11), %ymm12 // A[0]
	vbroadcastss	0(%r12), %ymm13 // B[0]
	vfmadd231ps		%ymm12, %ymm13, %ymm0
	vbroadcastss	32(%r12), %ymm13 // B[1]
	vfmadd231ps		%ymm12, %ymm13, %ymm1
	vbroadcastss	64(%r12), %ymm13 // B[2]
	vfmadd231ps		%ymm12, %ymm13, %ymm2
	vbroadcastss	96(%r12), %ymm13 // B[3]
	vfmadd231ps		%ymm12, %ymm13, %ymm3
	vbroadcastss	128(%r12), %ymm13 // B[4]
	vfmadd231ps		%ymm12, %ymm13, %ymm4
	vbroadcastss	160(%r12), %ymm13 // B[5]
	vfmadd231ps		%ymm12, %ymm13, %ymm5
	vbroadcastss	192(%r12), %ymm13 // B[6]
	vfmadd231ps		%ymm12, %ymm13, %ymm6
	vbroadcastss	224(%r12), %ymm13 // B[7]
	vfmadd231ps		%ymm12, %ymm13, %ymm7

	subl			$1, %r10d // k-1
	subl			$1, %ebx // kend-1
	addq			$32, %r11 // A+1*bs*sizeof(float)
	addq			$4, %r12 // B+1*sizeof(float)

	cmpl			$0, %ebx
	jg				1b

	cmpl			$0, %r10d
	jle				2f // end

	addq			%r13, %r12
	subq			$32, %r12 // B+bs*(sdb-1)*sizeof(float)

2:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_sgemm_add_nn_8x8_lib8, .-inner_edge_sgemm_add_nn_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// strsm
// right
// lower
// transposed
// not-unit
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_STRSM_RLT_INV_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_strsm_rlt_inv_8x8_gen_lib8, @function
inner_edge_strsm_rlt_inv_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_edge_strsm_rlt_inv_8x8_gen_lib8:
#endif
#endif

	vbroadcastss	0(%r11), %ymm13
	vmulps			%ymm0, %ymm13, %ymm0
	vbroadcastss	4(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm1
	vbroadcastss	8(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm2
	vbroadcastss	12(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm3
	vbroadcastss	16(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm4
	vbroadcastss	20(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm5
	vbroadcastss	24(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm6
	vbroadcastss	28(%r10), %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm7

	vbroadcastss	4(%r11), %ymm13
	vmulps			%ymm1, %ymm13, %ymm1
	vbroadcastss	40(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm2
	vbroadcastss	44(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm3
	vbroadcastss	48(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm4
	vbroadcastss	52(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm5
	vbroadcastss	56(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm6
	vbroadcastss	60(%r10), %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm7

	vbroadcastss	8(%r11), %ymm13
	vmulps			%ymm2, %ymm13, %ymm2
	vbroadcastss	76(%r10), %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm3
	vbroadcastss	80(%r10), %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm4
	vbroadcastss	84(%r10), %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm5
	vbroadcastss	88(%r10), %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm6
	vbroadcastss	92(%r10), %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm7

	vbroadcastss	12(%r11), %ymm13
	vmulps			%ymm3, %ymm13, %ymm3
	vbroadcastss	112(%r10), %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm4
	vbroadcastss	116(%r10), %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm5
	vbroadcastss	120(%r10), %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm6
	vbroadcastss	124(%r10), %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm7

	vbroadcastss	16(%r11), %ymm13
	vmulps			%ymm4, %ymm13, %ymm4
	cmpl			$6, %r12d
	jl				0f // ret
	vbroadcastss	148(%r10), %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm5
	vbroadcastss	152(%r10), %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm6
	vbroadcastss	156(%r10), %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm7

	vbroadcastss	20(%r11), %ymm13
	vmulps			%ymm5, %ymm13, %ymm5
	cmpl			$7, %r12d
	jl				0f // ret
	vbroadcastss	184(%r10), %ymm13
	vfnmadd231ps	%ymm5, %ymm13, %ymm6
	vbroadcastss	188(%r10), %ymm13
	vfnmadd231ps	%ymm5, %ymm13, %ymm7

	vbroadcastss	24(%r11), %ymm13
	vmulps			%ymm6, %ymm13, %ymm6
	cmpl			$8, %r12d
	jl				0f // ret
	vbroadcastss	220(%r10), %ymm13
	vfnmadd231ps	%ymm6, %ymm13, %ymm7

	vbroadcastss	28(%r11), %ymm13
	vmulps			%ymm7, %ymm13, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_strsm_rlt_inv_8x8_gen_lib8, .-inner_edge_strsm_rlt_inv_8x8_gen_lib8
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// r11d <- kn
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_SPOTRF_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_spotrf_8x8_gen_lib8, @function
inner_edge_spotrf_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_edge_spotrf_8x8_gen_lib8:
#endif
#endif

	vxorps	%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX)
	vmovss	.LC03(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovss	LC03(%rip), %xmm14 // 1.0
#endif

	vmovss			%xmm0, %xmm0, %xmm13
	vucomiss		%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			1f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
2:
	vmovss			%xmm13, 0(%r10)
	vbroadcastss	%xmm13, %ymm13
//	vpermilps		$0x00, %xmm13, %xmm13
//	vinsertf128		$0x1, %xmm13, %ymm13, %ymm13
	vmulps			%ymm0, %ymm13, %ymm0
	vperm2f128		$0x00, %ymm0, %ymm0, %ymm11
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm1
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm2
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm3
	vperm2f128		$0x11, %ymm0, %ymm0, %ymm11
	vpermilps		$0x00, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm4
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm5
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm0, %ymm13, %ymm7


	vpermilps		$0x55, %xmm1, %xmm13
	vucomiss		%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			3f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
4:
	vmovss			%xmm13, 4(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm1, %ymm13, %ymm1
	vperm2f128		$0x00, %ymm1, %ymm1, %ymm11
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm2
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm3
	vperm2f128		$0x11, %ymm1, %ymm1, %ymm11
	vpermilps		$0x00, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm4
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm5
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm1, %ymm13, %ymm7


	vpermilps		$0xaa, %xmm2, %xmm13
	vucomiss		%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			5f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
6:
	vmovss			%xmm13, 8(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm2, %ymm13, %ymm2
	vperm2f128		$0x00, %ymm2, %ymm2, %ymm11
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm3
	vperm2f128		$0x11, %ymm2, %ymm2, %ymm11
	vpermilps		$0x00, %ymm11, %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm4
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm5
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm2, %ymm13, %ymm7


	vpermilps		$0xff, %xmm3, %xmm13
	vucomiss		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			7f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
8:
	vmovss			%xmm13, 12(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm3, %ymm13, %ymm3
	vperm2f128		$0x11, %ymm3, %ymm3, %ymm11
	vpermilps		$0x00, %ymm11, %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm4
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm5
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm3, %ymm13, %ymm7


	vextractf128	$0x1, %ymm4, %xmm13
//	vpermilps		$0x00, %xmm13, %xmm13
	vucomiss		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			9f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
10:
	vmovss			%xmm13, 16(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm4, %ymm13, %ymm4
	cmpl		$6, %r11d
	jl			0f // ret
	vperm2f128		$0x11, %ymm4, %ymm4, %ymm11
	vpermilps		$0x55, %ymm11, %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm5
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm4, %ymm13, %ymm7


	vextractf128	$0x1, %ymm5, %xmm13
	vpermilps		$0x55, %xmm13, %xmm13
	vucomiss		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			11f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
12:
	vmovss			%xmm13, 20(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm5, %ymm13, %ymm5
	cmpl		$7, %r11d
	jl			0f // ret
	vperm2f128		$0x11, %ymm5, %ymm5, %ymm11
	vpermilps		$0xaa, %ymm11, %ymm13
	vfnmadd231ps	%ymm5, %ymm13, %ymm6
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm5, %ymm13, %ymm7


	vextractf128	$0x1, %ymm6, %xmm13
	vpermilps		$0xaa, %xmm13, %xmm13
	vucomiss		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			13f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
14:
	vmovss			%xmm13, 24(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm6, %ymm13, %ymm6
	cmpl		$8, %r11d
	jl			0f // ret
	vperm2f128		$0x11, %ymm6, %ymm6, %ymm11
	vpermilps		$0xff, %ymm11, %ymm13
	vfnmadd231ps	%ymm6, %ymm13, %ymm7


	vextractf128	$0x1, %ymm7, %xmm13
	vpermilps		$0xff, %xmm13, %xmm13
	vucomiss		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			15f
	vsqrtss			%xmm13, %xmm13, %xmm13
	vdivss			%xmm13, %xmm14, %xmm13
16:
	vmovss			%xmm13, 28(%r10)
	vbroadcastss	%xmm13, %ymm13
	vmulps			%ymm7, %ymm13, %ymm7


	jmp		0f


1:
	vxorps	%ymm13, %ymm13, %ymm13
	jmp		2b

3:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		4b

5:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		6b

7:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		8b

9:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		10b

11:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		12b

13:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		14b

15:
	vxorpd	%ymm13, %ymm13, %ymm13
	jmp		16b

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_spotrf_8x8_gen_lib8, .-inner_edge_spotrf_8x8_gen_lib8
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
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
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
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
	vbroadcastss	0(%r10), %ymm11

	vmulps		%ymm0, %ymm11, %ymm0
	vmulps		%ymm1, %ymm11, %ymm1
	vmulps		%ymm2, %ymm11, %ymm2
	vmulps		%ymm3, %ymm11, %ymm3

	vmulps		%ymm4, %ymm11, %ymm4
	vmulps		%ymm5, %ymm11, %ymm5
	vmulps		%ymm6, %ymm11, %ymm6
	vmulps		%ymm7, %ymm11, %ymm7

	// beta
	vbroadcastss	0(%r11), %ymm14

	vxorps		%ymm15, %ymm15, %ymm15 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovaps		0(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm0
	vmovaps		32(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm1
	vmovaps		64(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm2
	vmovaps		96(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm3
	vmovaps		128(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm4
	vmovaps		160(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm5
	vmovaps		192(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm6
	vmovaps		224(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm7

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
// scale scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_8x8_gen_lib8, @function
inner_scale_ab_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_scale_ab_8x8_gen_lib8:
#endif
#endif
	
	// alpha
	vbroadcastss	0(%r10), %ymm11

	vmulps		%ymm0, %ymm11, %ymm0
	vmulps		%ymm1, %ymm11, %ymm1
	vmulps		%ymm2, %ymm11, %ymm2
	vmulps		%ymm3, %ymm11, %ymm3

	vmulps		%ymm4, %ymm11, %ymm4
	vmulps		%ymm5, %ymm11, %ymm5
	vmulps		%ymm6, %ymm11, %ymm6
	vmulps		%ymm7, %ymm11, %ymm7

	// beta
	vbroadcastss	0(%r11), %ymm15

	vxorps		%ymm14, %ymm14, %ymm14 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			3f // end

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovaps		0(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm0
	vmovaps		32(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm1
	vmovaps		64(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm2
	vmovaps		96(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm3
	vmovaps		128(%r13), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm4
	vmovaps		160(%r13), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm5
	vmovaps		192(%r13), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm6
	vmovaps		224(%r13), %ymm15
	vfmadd231ps	%ymm12, %ymm15, %ymm7

	jmp		7f

0:

	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%r13, %r15 // C0
	addq	%r14, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:


#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_8x8_gen_lib8, .-inner_scale_ab_8x8_gen_lib8
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
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
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
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_8x8_lib8, @function
inner_blend_scale_ab_8x8_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_ab_8x8_lib8:
#endif
#endif
	
	// alpha
	vbroadcastss	0(%r10), %ymm11

	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	vmulps		%ymm0, %ymm11, %ymm0
	vmulps		%ymm1, %ymm11, %ymm1
	vmulps		%ymm2, %ymm11, %ymm2
	vmulps		%ymm3, %ymm11, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	vmulps		%ymm4, %ymm11, %ymm4
	vmulps		%ymm5, %ymm11, %ymm5
	vmulps		%ymm6, %ymm11, %ymm6
	vmulps		%ymm7, %ymm11, %ymm7

	// beta
	vbroadcastss	0(%r11), %ymm14

	vxorps		%ymm15, %ymm15, %ymm15 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			0f // end

	vmovaps		0(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm0
	vmovaps		32(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm1
	vmovaps		64(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm2
	vmovaps		96(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm3
	vmovaps		128(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm4
	vmovaps		160(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm5
	vmovaps		192(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm6
	vmovaps		224(%r12), %ymm15
	vfmadd231ps	%ymm15, %ymm14, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_8x8_lib8, .-inner_blend_scale_ab_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// blend scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_8x8_gen_lib8, @function
inner_blend_scale_ab_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_ab_8x8_gen_lib8:
#endif
#endif
	
	// alpha
	vbroadcastss	0(%r10), %ymm11

	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	vmulps		%ymm0, %ymm11, %ymm0
	vmulps		%ymm1, %ymm11, %ymm1
	vmulps		%ymm2, %ymm11, %ymm2
	vmulps		%ymm3, %ymm11, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	vmulps		%ymm4, %ymm11, %ymm4
	vmulps		%ymm5, %ymm11, %ymm5
	vmulps		%ymm6, %ymm11, %ymm6
	vmulps		%ymm7, %ymm11, %ymm7

	// beta
	vbroadcastss	0(%r11), %ymm15

	vxorps		%ymm14, %ymm14, %ymm14 // 0.0

	vucomiss	%xmm15, %xmm14 // beta==0.0 ?
	je			3f // end

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovaps		0(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm0
	vmovaps		32(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm1
	vmovaps		64(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm2
	vmovaps		96(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm3
	vmovaps		128(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm4
	vmovaps		160(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm5
	vmovaps		192(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm6
	vmovaps		224(%r13), %ymm12
	vfmadd231ps	%ymm12, %ymm15, %ymm7

	jmp		7f

0:

	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%r13, %r15 // C0
	addq	%r14, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$4, %r12d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r12d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r12d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:


#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_8x8_gen_lib8, .-inner_blend_scale_ab_8x8_gen_lib8
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha=1.0 and beta=1.0
//
// input arguments:
// r10   <- C
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- C
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_8x8_lib8, @function
inner_blend_scale_11_8x8_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_11_8x8_lib8:
#endif
#endif
	
	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	vmovaps		0(%r10), %ymm15
	vaddps		%ymm0, %ymm15, %ymm0
	vmovaps		32(%r10), %ymm15
	vaddps		%ymm1, %ymm15, %ymm1
	vmovaps		64(%r10), %ymm15
	vaddps		%ymm2, %ymm15, %ymm2
	vmovaps		96(%r10), %ymm15
	vaddps		%ymm3, %ymm15, %ymm3
	vmovaps		128(%r10), %ymm15
	vaddps		%ymm4, %ymm15, %ymm4
	vmovaps		160(%r10), %ymm15
	vaddps		%ymm5, %ymm15, %ymm5
	vmovaps		192(%r10), %ymm15
	vaddps		%ymm6, %ymm15, %ymm6
	vmovaps		224(%r10), %ymm15
	vaddps		%ymm7, %ymm15, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_11_8x8_lib8, .-inner_blend_scale_11_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// blend scale for generic alpha and beta
//
// input arguments:
// r10  <- offset
// r11   <- C
// r12  <- 4*sdc*sizeof(double)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- offset
// r11   <- C
// r12  <- 4*sdc*sizeof(double)
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_8x8_gen_lib8, @function
inner_blend_scale_11_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_blend_scale_11_8x8_gen_lib8:
#endif
#endif
	
	vblendps	$0xaa, %ymm1, %ymm0, %ymm12 // 1010 1010
	vblendps	$0x55, %ymm1, %ymm0, %ymm13 // 0101 0101
	vblendps	$0xaa, %ymm3, %ymm2, %ymm14
	vblendps	$0x55, %ymm3, %ymm2, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm0 // 1100 1100
	vblendps	$0x33, %ymm15, %ymm12, %ymm2 // 0011 0011
	vblendps	$0xcc, %ymm14, %ymm13, %ymm1
	vblendps	$0x33, %ymm14, %ymm13, %ymm3

	vblendps	$0xaa, %ymm5, %ymm4, %ymm12
	vblendps	$0x55, %ymm5, %ymm4, %ymm13
	vblendps	$0xaa, %ymm7, %ymm6, %ymm14
	vblendps	$0x55, %ymm7, %ymm6, %ymm15

	vblendps	$0xcc, %ymm15, %ymm12, %ymm4
	vblendps	$0x33, %ymm15, %ymm12, %ymm6
	vblendps	$0xcc, %ymm14, %ymm13, %ymm5
	vblendps	$0x33, %ymm14, %ymm13, %ymm7

	cmpl	$0, %r10d
	jg		0f

	// offset==0

	vmovaps		0(%r11), %ymm12
	vaddps		%ymm0, %ymm12, %ymm0
	vmovaps		32(%r11), %ymm12
	vaddps		%ymm1, %ymm12, %ymm1
	vmovaps		64(%r11), %ymm12
	vaddps		%ymm2, %ymm12, %ymm2
	vmovaps		96(%r11), %ymm12
	vaddps		%ymm3, %ymm12, %ymm3
	vmovaps		128(%r11), %ymm12
	vaddps		%ymm4, %ymm12, %ymm4
	vmovaps		160(%r11), %ymm12
	vaddps		%ymm5, %ymm12, %ymm5
	vmovaps		192(%r11), %ymm12
	vaddps		%ymm6, %ymm12, %ymm6
	vmovaps		224(%r11), %ymm12
	vaddps		%ymm7, %ymm12, %ymm7

	jmp		7f

0:

	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%r13, %r15 // C0
	addq	%r14, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:


#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_11_8x8_gen_lib8, .-inner_blend_scale_11_8x8_gen_lib8
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





// common inner routine with file scope
//
// store n generalized
//
// input arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n0 // col index: start from (inc)
// rax  <- n1 // col index: up to (exc)
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
//
// output arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n1-n0
// rax  <- n1-n0
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_8x8_gen_lib8, @function
inner_store_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_store_8x8_gen_lib8:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm14, %xmm14
	vcvtsi2ss	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovups		.LC00(%rip), %ymm12
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %ymm12
#endif
	vshufps		$0x00, %xmm14, %xmm14, %xmm14
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm14, %ymm14, %ymm14
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm12, %ymm14, %ymm14
	vsubps		%ymm15, %ymm12, %ymm15
	vandps		%ymm14, %ymm15, %ymm15

	// shift D and sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm6, %ymm5
	vmovaps		%ymm7, %ymm6
	addq		$32, %r11

	cmpl	$1, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm6, %ymm5
	addq		$32, %r11

	cmpl	$2, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	addq		$32, %r11

0:

	// compute number of cols
	cmpl	$8, %eax
	jle		0f
	movl	$8, %eax
0:
	subl	%r15d, %eax
	movl	%eax, %r15d

	cmpl	$0, %r10d
	jg		0f

	// offset==0
	vmaskmovps	%ymm0, %ymm15,  0(%r11)
	vmaskmovps	%ymm1, %ymm15,  32(%r11)
	vmaskmovps	%ymm2, %ymm15,  64(%r11)
	vmaskmovps	%ymm3, %ymm15,  96(%r11)
	vmaskmovps	%ymm4, %ymm15,  128(%r11)
	cmpl		$6, %r15d
	jl			7f // end
	vmaskmovps	%ymm5, %ymm15, 160(%r11)
	cmpl		$7, %r15d
	jl			7f // end
	vmaskmovps	%ymm6, %ymm15, 192(%r11)
	je			7f // end
	vmaskmovps	%ymm7, %ymm15, 224(%r11)
	//
	jmp		7f

0:
	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%r11, %rbx // D0
	addq	%r12, %rbx // D1 <- D0 + 4*sdd*sizeof(double)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x8_gen_lib8, .-inner_store_8x8_gen_lib8
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
	.macro INNER_STORE_L_8X8_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x8_lib8, @function
inner_store_l_8x8_lib8:
#elif defined(OS_MAC)
_inner_store_l_8x8_lib8:
#endif
#endif
	
	vmovaps 	%ymm0,  0(%r10)
	vmovaps		32(%r10), %ymm14
	vblendps	$0x01, %ymm14, %ymm1, %ymm1
	vmovaps 	%ymm1, 32(%r10)
	vmovaps		64(%r10), %ymm14
	vblendps	$0x03, %ymm14, %ymm2, %ymm2
	vmovaps 	%ymm2, 64(%r10)
	vmovaps		96(%r10), %ymm14
	vblendps	$0x07, %ymm14, %ymm3, %ymm3
	vmovaps 	%ymm3, 96(%r10)
	vmovaps		128(%r10), %ymm14
	vblendps	$0x0f, %ymm14, %ymm4, %ymm4
	vmovaps 	%ymm4, 128(%r10)
	vmovaps		160(%r10), %ymm14
	vblendps	$0x1f, %ymm14, %ymm5, %ymm5
	vmovaps 	%ymm5, 160(%r10)
	vmovaps		192(%r10), %ymm14
	vblendps	$0x3f, %ymm14, %ymm6, %ymm6
	vmovaps 	%ymm6, 192(%r10)
	vmovaps		224(%r10), %ymm14
	vblendps	$0x7f, %ymm14, %ymm7, %ymm7
	vmovaps 	%ymm7, 224(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_8x8_lib8, .-inner_store_l_8x8_lib8
#endif
#endif





// common inner routine with file scope
//
// store lower generalized
//
// input arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n0 // col index: start from (inc)
// rax  <- n1 // col index: up to (exc)
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []
//
// output arguments:
// r10  <- offset
// r11  <- D
// r12  <- 4*sdd*sizeof(double)
// r13  <- m0 // row index: start from (inc)
// r14  <- m1 // row index: up to (exc)
// r15  <- n1-n0
// rax  <- n1-n0
// rbx  <- dirty
// ymm0 <- []
// ymm1 <- []
// ymm2 <- []
// ymm3 <- []
// ymm4 <- []
// ymm5 <- []
// ymm6 <- []
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_8X8_GEN_LIB8
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_8x8_gen_lib8, @function
inner_store_l_8x8_gen_lib8:
#elif defined(OS_MAC)
_inner_store_l_8x8_gen_lib8:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r13d, %xmm14, %xmm14
	vcvtsi2ss	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovups		.LC00(%rip), %ymm12
#elif defined(OS_MAC)
	vmovups		LC00(%rip), %ymm12
#endif
	vshufps		$0x00, %xmm14, %xmm14, %xmm14
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm14, %ymm14, %ymm14
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm12, %ymm14, %ymm14
	vsubps		%ymm15, %ymm12, %ymm15
	vandps		%ymm14, %ymm15, %ymm15

	// shift D and sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm6, %ymm5
	vmovaps		%ymm7, %ymm6
	addq		$32, %r11

	cmpl	$1, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm2, %ymm1
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	vmovaps		%ymm6, %ymm5
	addq		$32, %r11

	cmpl	$2, %r15d
	jle		0f

	vmovaps		%ymm1, %ymm0
	vmovaps		%ymm3, %ymm2
	vmovaps		%ymm4, %ymm3
	vmovaps		%ymm5, %ymm4
	addq		$32, %r11

0:

	// compute number of cols
	cmpl	$8, %eax
	jle		0f
	movl	$8, %eax
0:
	subl	%r15d, %eax
	movl	%eax, %r15d

	cmpl	$0, %r10d
	jg		0f

	// offset==0
	vmaskmovps	%ymm0, %ymm15,  0(%r11)
	vmovaps 	32(%r11), %ymm12
	vblendps	$0x01, %ymm12, %ymm1, %ymm1
	vmaskmovps	%ymm1, %ymm15,  32(%r11)
	vmovaps 	64(%r11), %ymm12
	vblendps	$0x03, %ymm12, %ymm2, %ymm2
	vmaskmovps	%ymm2, %ymm15,  64(%r11)
	vmovaps 	96(%r11), %ymm12
	vblendps	$0x07, %ymm12, %ymm3, %ymm3
	vmaskmovps	%ymm3, %ymm15,  96(%r11)
	vmovaps 	128(%r11), %ymm12
	vblendps	$0x0f, %ymm12, %ymm4, %ymm4
	vmaskmovps	%ymm4, %ymm15,  128(%r11)
	cmpl		$6, %r15d
	jl			7f // end
	vmovaps 	160(%r11), %ymm12
	vblendps	$0x1f, %ymm12, %ymm5, %ymm5
	vmaskmovps	%ymm5, %ymm15, 160(%r11)
	cmpl		$7, %r15d
	jl			7f // end
	vmovaps 	192(%r11), %ymm12
	vblendps	$0x3f, %ymm12, %ymm6, %ymm6
	vmaskmovps	%ymm6, %ymm15, 192(%r11)
	je			7f // end
	vmovaps 	224(%r11), %ymm12
	vblendps	$0x7f, %ymm12, %ymm7, %ymm7
	vmaskmovps	%ymm7, %ymm15, 224(%r11)
	//
	jmp		7f

0:
	// offset > 0
	// 1 2 3 4 5 6 7
	
	movq	%r11, %rbx // D0
	addq	%r12, %rbx // D1 <- D0 + 4*sdd*sizeof(double)

	cmpl	$4, %r10d
	jl		1f
	jg		2f

	// offset==4
	// TODO
	jmp		7f

1:
	// 1 2 3

	cmpl	$2, %r10d
	jl		3f
	jg		4f

	// offset==2
	// TODO
	jmp		7f

3:
	// offset==1
	// TODO
	jmp		7f

4:
	// offset==3
	// TODO
	jmp		7f

2:
	// 5 6 7

	cmpl	$6, %r10d
	jl		5f
	jg		6f

	// offset==6
	// TODO
	jmp		7f

5:
	// offset==5
	// TODO
	jmp		7f

6:
	// offset==7
	// TODO
	jmp		7f

	// end
7:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_8x8_gen_lib8, .-inner_store_8x8_gen_lib8
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
	INNER_BLEND_SCALE_AB_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x8_lib8
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





//                                   rdi    rsi           rdx       rcx       r8           r9           rsp+8     rsp+16   rsp+24       rsp+32    rsp+40   rsp+48  rsp+56  rsp+64  rsp+72
// void kernel_sgemm_nt_8x8_gen_lib8(int k, float *alpha, float *A, float *B, float *beta, int offsetC, float *C, int sdc, int offsetD, float *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nt_8x8_gen_lib8
	.type kernel_sgemm_nt_8x8_gen_lib8, @function
kernel_sgemm_nt_8x8_gen_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nt_8x8_gen_lib8
_kernel_sgemm_nt_8x8_gen_lib8:
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


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movl	%r9d, %r12d // offsetC
	movq	STACKSIZE +  8(%rsp), %r13 // C
	movl	STACKSIZE + 16(%rsp), %r14d // sdc
	sall	$5, %r14d // 8*sdc*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x8_gen_lib8
#endif
#endif


	// store n gen

	movq	STACKSIZE + 24(%rsp), %r10 // offsetD
	movq	STACKSIZE + 32(%rsp), %r11 // D
	movl	STACKSIZE + 40(%rsp), %r12d // sdd
	sall	$5, %r12d // 8*sdb*sizeof(float)
	movl	STACKSIZE + 48(%rsp), %r13d // m0
	movl	STACKSIZE + 56(%rsp), %r14d // m1
	movl	STACKSIZE + 64(%rsp), %r15d // n0
	movl	STACKSIZE + 72(%rsp), %eax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_8x8_gen_lib8
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
	.size	kernel_sgemm_nt_8x8_gen_lib8, .-kernel_sgemm_nt_8x8_gen_lib8
#endif





//                               rdi    rsi           rdx        rcx         r8         r9      rsp+8        rsp+16    rsp+24
// void kernel_sgemm_nn_8x8_lib8(int k, float *alpha, float *A, int offsetB, float *B, int sdb, float *beta, float *C, float *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nn_8x8_lib8
	.type kernel_sgemm_nn_8x8_lib8, @function
kernel_sgemm_nn_8x8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nn_8x8_lib8
_kernel_sgemm_nn_8x8_lib8:
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

	vxorps	%ymm0, %ymm0, %ymm0
	vmovaps	%ymm0, %ymm1
	vmovaps	%ymm0, %ymm2
	vmovaps	%ymm0, %ymm3
	vmovaps	%ymm0, %ymm4
	vmovaps	%ymm0, %ymm5
	vmovaps	%ymm0, %ymm6
	vmovaps	%ymm0, %ymm7


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%r8, %r12  // B
	movl	%r9d, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movl	%ecx, %r14d // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_SGEMM_ADD_NN_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_sgemm_add_nn_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_edge_sgemm_add_nn_8x8_lib8
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_ADD_NN_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_add_nn_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_add_nn_8x8_lib8
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	STACKSIZE + 8(%rsp), %r11 // beta
	movq	STACKSIZE + 16(%rsp), %r12   // C

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

	movq	STACKSIZE + 24(%rsp), %r10 // D

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
	.size	kernel_sgemm_nn_8x8_lib8, .-kernel_sgemm_nn_8x8_lib8
#endif





//                                   rdi    rsi           rdx       rcx       r8        r9       rsp+8        rsp+16    rsp+24    rsp+32    rsp+40   rsp+48     rsp+56   rsp+64  rsp+72  rsp+80  rsp+88
// void kernel_sgemm_nn_8x8_gen_lib4(int k, float *alpha, float *A, int offB, float *B, int sdb, float *beta, int offC, float *C, int sdc, int offD, float *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_sgemm_nn_8x8_gen_lib8
	.type kernel_sgemm_nn_8x8_gen_lib8, @function
kernel_sgemm_nn_8x8_gen_lib8:
#elif defined(OS_MAC)
	.globl _kernel_sgemm_nn_8x8_gen_lib8
_kernel_sgemm_nn_8x8_gen_lib8:
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

	vxorps	%ymm0, %ymm0, %ymm0
	vmovaps	%ymm0, %ymm1
	vmovaps	%ymm0, %ymm2
	vmovaps	%ymm0, %ymm3
	vmovaps	%ymm0, %ymm4
	vmovaps	%ymm0, %ymm5
	vmovaps	%ymm0, %ymm6
	vmovaps	%ymm0, %ymm7


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%r8, %r12  // B
	movl	%r9d, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movl	%ecx, %r14d // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_SGEMM_ADD_NN_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_sgemm_add_nn_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_edge_sgemm_add_nn_8x8_lib8
#endif
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_ADD_NN_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_add_nn_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_add_nn_8x8_lib8
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	STACKSIZE +  8(%rsp), %r11 // beta
	movl	STACKSIZE + 16(%rsp), %r12d // offsetC
	movq	STACKSIZE + 24(%rsp), %r13 // C
	movl	STACKSIZE + 32(%rsp), %r14d // sdc
	sall	$5, %r14d // 4*sdc*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_scale_ab_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_scale_ab_8x8_gen_lib8
#endif
#endif


	// store n gen

	movq	STACKSIZE + 40(%rsp), %r10 // offsetD
	movq	STACKSIZE + 48(%rsp), %r11 // D
	movl	STACKSIZE + 56(%rsp), %r12d // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movl	STACKSIZE + 64(%rsp), %r13d // m0
	movl	STACKSIZE + 72(%rsp), %r14d // m1
	movl	STACKSIZE + 80(%rsp), %r15d // n0
	movl	STACKSIZE + 88(%rsp), %eax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_8x8_gen_lib8
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
	.size	kernel_sgemm_nn_8x8_gen_lib8, .-kernel_sgemm_nn_8x8_gen_lib8
#endif





//                                 rdi    rsi           rdx       rcx       r8           r9        rsp+8
// void kernel_ssyrk_nt_l_8x8_lib8(int k, float *alpha, float *A, float *B, float *beta, float *C, float *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_ssyrk_nt_l_8x8_lib8
	.type kernel_ssyrk_nt_l_8x8_lib8, @function
kernel_ssyrk_nt_l_8x8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_ssyrk_nt_l_8x8_lib8
_kernel_ssyrk_nt_l_8x8_lib8:
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
	INNER_BLEND_SCALE_AB_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_8x8_lib8
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_l_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_store_l_8x8_lib8
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
	.size	kernel_ssyrk_nt_l_8x8_lib8, .-kernel_ssyrk_nt_l_8x8_lib8
#endif





//                                      edi    rsi       rdx       ecx       r8        r9        rsp+8     
// void kernel_strsm_nt_rl_inv_8x8_lib8(int k, float *A, float *B, float *C, float *D, float *E, float *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_strsm_nt_rl_inv_8x8_lib8
	.type kernel_strsm_nt_rl_inv_8x8_lib8, @function
kernel_strsm_nt_rl_inv_8x8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_strsm_nt_rl_inv_8x8_lib8
_kernel_strsm_nt_rl_inv_8x8_lib8:
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

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_SUB_NT_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_sub_nt_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_sub_nt_8x8_lib8
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X8_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x8_lib8
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 
	movl	$4, %r12d // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_STRSM_RLT_INV_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_strsm_rlt_inv_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_edge_strsm_rlt_inv_8x8_gen_lib8
#endif
#endif


	// store

	movq	%r8, %r10 // D

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
	.size	kernel_strsm_nt_rl_inv_8x8_lib8, .-kernel_strsm_nt_rl_inv_8x8_lib8
#endif





//                                         edi    rsi       rdx       ecx       r8        r9        rsp+8               rsp+16  rsp+24  
// void kernel_strsm_nt_rl_inv_8x8_vs_lib8(int k, float *A, float *B, float *C, float *D, float *E, float *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_strsm_nt_rl_inv_8x8_vs_lib8
	.type kernel_strsm_nt_rl_inv_8x8_vs_lib8, @function
kernel_strsm_nt_rl_inv_8x8_vs_lib8:
#elif defined(OS_MAC)
	.globl _kernel_strsm_nt_rl_inv_8x8_vs_lib8
_kernel_strsm_nt_rl_inv_8x8_vs_lib8:
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

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_SUB_NT_8X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_sub_nt_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_sub_nt_8x8_lib8
#endif
#endif


	// call inner blender_loader nn // TODO scale gen

	movl	$0, %r10d // offsetC
	movq	%rcx, %r11 // C
	movl	$0, %r12d // sdc

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x8_gen_lib8
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_STRSM_RLT_INV_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_strsm_rlt_inv_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_edge_strsm_rlt_inv_8x8_gen_lib8
#endif
#endif


	// store

	movl	$0, %r10d // offsetD
	movq	%r8, %r11 // D
	movl	$0, %r12d // sdd
	movl	$0, %r13d // m0
	movl	STACKSIZE + 16(%rsp), %r14d // m1 
	movl	$0, %r15d // n0
	movl	STACKSIZE + 24(%rsp), %eax // n1 

#if MACRO_LEVEL>=1
	INNER_STORE_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_8x8_gen_lib8
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
	.size	kernel_strsm_nt_rl_inv_8x8_vs_lib8, .-kernel_strsm_nt_rl_inv_8x8_vs_lib8
#endif





//                                  edi    rsi       rdx       rcx       r8        r9
// void kernel_spotrf_nt_l_8x8_lib8(int k, float *A, float *B, float *C, float *D, float *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_spotrf_nt_l_8x8_lib8
	.type kernel_spotrf_nt_l_8x8_lib8, @function
kernel_spotrf_nt_l_8x8_lib8:
#elif defined(OS_MAC)
	.globl _kernel_spotrf_nt_l_8x8_lib8
_kernel_spotrf_nt_l_8x8_lib8:
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

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_SUB_NT_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_sub_nt_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_sub_nt_8x8_lib8
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x8_lib8
#endif
#endif


	// factorization

	movq	%r9, %r10  // inv_diag_D 
	movl	$4, %r11d // n1

#if MACRO_LEVEL>=1
	INNER_EDGE_SPOTRF_8X8_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_spotrf_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_edge_spotrf_8x8_gen_lib8
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X8_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_l_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_store_l_8x8_lib8
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
	.size	kernel_spotrf_nt_l_8x8_lib8, .-kernel_spotrf_nt_l_8x8_lib8
#endif





//                                     edi    rsi       rdx       rcx       r8        r9                  rsp+8   rsp+16
// void kernel_spotrf_nt_l_8x8_vs_lib8(int k, float *A, float *B, float *C, float *D, float *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_spotrf_nt_l_8x8_vs_lib8
	.type kernel_spotrf_nt_l_8x8_vs_lib8, @function
kernel_spotrf_nt_l_8x8_vs_lib8:
#elif defined(OS_MAC)
	.globl _kernel_spotrf_nt_l_8x8_vs_lib8
_kernel_spotrf_nt_l_8x8_vs_lib8:
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

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_SGEMM_SUB_NT_8X4_LIB8
#else
#if defined(OS_LINUX)
	call inner_kernel_sgemm_sub_nt_8x8_lib8
#elif defined(OS_MAC)
	callq _inner_kernel_sgemm_sub_nt_8x8_lib8
#endif
#endif


	// call inner blender_loader nn

	movl	$0, %r10d // offsetC
	movq	%rcx, %r11 // C
	movl	$0, %r12d // sdc

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_8x8_gen_lib8
#endif
#endif


	// factorization

	movq	%r9, %r10  // inv_diag_D 
	movl	STACKSIZE + 16(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_SPOTRF_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_edge_spotrf_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_edge_spotrf_8x8_gen_lib8
#endif
#endif


	// store

	movl	$0, %r10d // offsetD
	movq	%r8, %r11 // D
	movl	$0, %r12d // sdd
	movl	$0, %r13d // m0
	movl	STACKSIZE + 8(%rsp), %r14d // m1 
	movl	$0, %r15d // n0
	movl	STACKSIZE + 16(%rsp), %eax // n1 

#if MACRO_LEVEL>=1
	INNER_STORE_L_8X4_GEN_LIB8
#else
#if defined(OS_LINUX)
	call inner_store_l_8x8_gen_lib8
#elif defined(OS_MAC)
	callq _inner_store_l_8x8_gen_lib8
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
	.size	kernel_spotrf_nt_l_8x8_vs_lib8, .-kernel_spotrf_nt_l_8x8_vs_lib8
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

