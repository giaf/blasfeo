/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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

//.LSTR:
//	.string	"\n%d %d %d\n"

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

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_4x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r12), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


	// unroll 3
	vmovapd 0(%r12), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	cmpl	$4, %r10d

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


	// unroll 3
//	vmovapd 0(%r12), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
//	cmpl	$3, %r10d

	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2


	jmp		2f // return


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 0(%r12), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	addq	$32, %r11

	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	addq	$32, %r12

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2
	subl	$1, %r10d

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif





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

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nt_4x4_lib4, @function
inner_kernel_dgemm_sub_nt_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nt_4x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	// prefetch
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmovapd 0(%r12), %ymm12 // B[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2


	// unroll 3
	vmovapd 0(%r12), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 0(%r11), %ymm8 // A0[0]
	cmpl	$4, %r10d

	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2

	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1

	// unroll 0
	vmovapd 32(%r12), %ymm13 // B[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 32(%r11), %ymm10 // A0[4]

	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	subl	$4, %r10d
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2

	// unroll 1
	vmovapd 64(%r12), %ymm12 // B[8]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 64(%r11), %ymm8 // A0[8]

	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2

	// unroll 2
	vmovapd 96(%r12), %ymm13 // B[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm12
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	vmovapd 96(%r11), %ymm10 // A0[12]

	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	addq	$128, %r12
	addq	$128, %r11

	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2


	// unroll 3
//	vmovapd 0(%r12), %ymm12 // B[0]
	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm13
	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
//	vmovapd 0(%r11), %ymm8 // A0[0]
//	cmpl	$3, %r10d

	vmulpd	%ymm10, %ymm13, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm14

	vmulpd	%ymm10, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 0(%r12), %ymm12 // B[0]
	vmovapd 0(%r11), %ymm8 // A0[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vsubpd	%ymm15, %ymm0, %ymm0
	addq	$32, %r11

	vshufpd $0x5, %ymm12, %ymm12, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm1, %ymm1
	addq	$32, %r12

	vperm2f128 $0x1, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm3, %ymm3

	vshufpd $0x5, %ymm14, %ymm14, %ymm14
	vmulpd	%ymm8, %ymm14, %ymm15
	vsubpd	%ymm15, %ymm2, %ymm2
	subl	$1, %r10d

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nt_4x4_lib4, .-inner_kernel_dgemm_sub_nt_4x4_lib4
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NN_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nn_4x4_lib4, @function
inner_kernel_dgemm_add_nn_4x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nn_4x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r14 // B_next <- B
	addq	%r13, %r14 // B_next <- B + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%r14) // software prefetch
	prefetcht0	64(%r14) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A[0]
	vbroadcastsd	0(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A[0]
	vbroadcastsd	8(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	40(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	72(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A[0]
	vbroadcastsd	16(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	48(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	80(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	112(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A[0]
	vbroadcastsd	24(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	56(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	88(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	120(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3

	addq	$128, %r11

	mov		%r14, %r12
	addq	%r13, %r14

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A[0]
	vbroadcastsd	0(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vaddpd			%ymm15, %ymm3, %ymm3

	subl	$1, %r10d

	addq	$32, %r11
	addq	$8, %r12

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nn_4x4_lib4, .-inner_kernel_dgemm_add_nn_4x4_lib4
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
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
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nn_4x4_lib4, @function
inner_kernel_dgemm_sub_nn_4x4_lib4:
#elif defined(OS_MAC)
#endif
_inner_kernel_dgemm_sub_nn_4x4_lib4:
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r12, %r14 // B_next <- B
	addq	%r13, %r14 // B_next <- B + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%r14) // software prefetch
	prefetcht0	64(%r14) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A[0]
	vbroadcastsd	0(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A[0]
	vbroadcastsd	8(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	40(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	72(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A[0]
	vbroadcastsd	16(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	48(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	80(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	112(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A[0]
	vbroadcastsd	24(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	56(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	88(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	120(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	addq	$128, %r11

	mov		%r14, %r12
	addq	%r13, %r14

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A[0]
	vbroadcastsd	0(%r12), %ymm13 // B[0]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	32(%r12), %ymm13 // B[1]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r12), %ymm13 // B[2]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	96(%r12), %ymm13 // B[3]
	vmulpd			%ymm12, %ymm13, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

	subl	$1, %r10d

	addq	$32, %r11
	addq	$8, %r12

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nn_4x4_lib4, .-inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif





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


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_4x4_lib4, @function
inner_edge_dtrmm_nt_ru_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_4x4_lib4:
#endif
#endif
	
	vmovapd			0(%r10), %ymm8
	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	vmovapd			32(%r10), %ymm8
	vbroadcastsd	32(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	40(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	vmovapd			64(%r10), %ymm8
	vbroadcastsd	64(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	72(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	80(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	vmovapd			96(%r10), %ymm8
	vbroadcastsd	96(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	104(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	112(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	120(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	addq			$128, %r10
	addq			$128, %r11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_4x4_lib4, .-inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
#endif





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


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_4x4_vs_lib4, @function
inner_edge_dtrmm_nt_ru_4x4_vs_lib4:
#elif defined(OS_MAC)
#endif
_inner_edge_dtrmm_nt_ru_4x4_vs_lib4:
#endif
	
	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r11
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r12

	cmpl	$0, %r10d
	jle		0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r11
	vbroadcastsd	24(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq			$32, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_4x4_vs_lib4, .-inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for dlauum
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


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DLAUUM_NT_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dlauum_nt_4x4_lib4, @function
inner_edge_dlauum_nt_4x4_lib4:
#elif defined(OS_MAC)
#endif
_inner_edge_dlauum_nt_4x4_lib4:
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			0(%r10), %ymm8
	vblendpd		$0x1, %ymm8, %ymm14, %ymm8
	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0

	vmovapd			32(%r10), %ymm8
	vblendpd		$0x3, %ymm8, %ymm14, %ymm8
	vbroadcastsd	32(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	40(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1

	vmovapd			64(%r10), %ymm8
	vblendpd		$0x7, %ymm8, %ymm14, %ymm8
	vbroadcastsd	64(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	72(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	80(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	vmovapd			96(%r10), %ymm8
	vbroadcastsd	96(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	104(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	112(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	120(%r11), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3

	addq			$128, %r10
	addq			$128, %r11

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dlauum_nt_4x4_lib4, .-inner_edge_dlauum_nt_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for dlauum
//
// input arguments:
// r10   <- k
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
// r10   <- 0
// r11   <- A+4*k*sizeof(double)
// r12   <- B+4*k*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm12 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DLAUUM_NT_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dlauum_nt_4x4_vs_lib4, @function
inner_edge_dlauum_nt_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dlauum_nt_4x4_vs_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vblendpd		$0x1, %ymm8, %ymm14, %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	addq			$32, %r11
	addq			$32, %r12

	cmpl			$0, %r10d
	jle				0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vblendpd		$0x3, %ymm8, %ymm14, %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	addq			$32, %r11
	addq			$32, %r12

	cmpl			$0, %r10d
	jle				0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vblendpd		$0x7, %ymm8, %ymm14, %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	addq			$32, %r11
	addq			$32, %r12

	cmpl			$0, %r10d
	jle				0f

	vmovapd			0(%r11), %ymm8
	subl			$1, %r10d
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2
	vbroadcastsd	24(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	addq			$32, %r11
	addq			$32, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dlauum_nt_4x4_vs_lib4, .-inner_edge_dlauum_nt_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// blend
//
// input arguments:
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
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_4x4_lib4, @function
inner_blend_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_4x4_lib4:
#endif	
#endif	
	

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_4x4_lib4, .-inner_blend_4x4_lib4
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
// r10   <- alpha
// r11   <- beta
// r12   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_4x4_lib4, @function
inner_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_4x4_lib4:
#endif
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vmovapd		0(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_4x4_lib4, .-inner_scale_ab_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
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
// r10   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_11_4x4_lib4, @function
inner_scale_11_4x4_lib4:
#elif defined(OS_MAC)
_inner_scale_11_4x4_lib4:
#endif	
#endif	
	
	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_11_4x4_lib4, .-inner_scale_11_4x4_lib4
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
// r10   <- alpha
// r11   <- beta
// r12   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_4x4_lib4, @function
inner_blend_scale_ab_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_4x4_lib4:
#endif
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm14

	vmovapd		0(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vmulpd		%ymm15, %ymm14, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_4x4_lib4, .-inner_blend_scale_ab_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
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
// r10   <- alpha
// r11   <- beta
// r12  <- offset
// r13   <- C
// r14  <- 4*sdc*sizeof(double)
// r15  <- n0 // col index: start from (inc)
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_4X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_4x4_gen_lib4, @function
inner_blend_scale_ab_4x4_gen_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_4x4_gen_lib4:
#endif
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	// beta
	vbroadcastsd	0(%r11), %ymm15

	// shift sol for cols
	cmpl	$0, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm2, %ymm1
	vmovapd		%ymm3, %ymm2

	cmpl	$1, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0
	vmovapd		%ymm2, %ymm1

	cmpl	$2, %r15d
	jle		0f

	vmovapd		%ymm1, %ymm0

0:

	cmpl	$0, %r12d
	jg		0f

	// offset==0

	vmovapd		0(%r13), %ymm12
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmovapd		32(%r13), %ymm12
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm1, %ymm12, %ymm1
	vmovapd		64(%r13), %ymm12
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmovapd		96(%r13), %ymm12
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm3, %ymm12, %ymm3

	jmp		3f

0:

	movq	%r13, %r15 // C0
	addq	%r14, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	cmpl	$1, %r12d
	jg		1f

	// offset==1

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r15), %ymm14
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vmovapd		32(%r13), %ymm13
	vmovapd		32(%r15), %ymm14
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm1, %ymm13, %ymm1
	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r15), %ymm14
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vmovapd		96(%r13), %ymm13
	vmovapd		96(%r15), %ymm14
	vblendpd	$0x1, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm14, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm3, %ymm13, %ymm3

	jmp		3f

1:

	cmpl	$2, %r12d
	jg		2f

	// offset==2

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r15), %ymm14
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vmovapd		32(%r13), %ymm13
	vmovapd		32(%r15), %ymm14
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm1, %ymm13, %ymm1
	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r15), %ymm14
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vmovapd		96(%r13), %ymm13
	vmovapd		96(%r15), %ymm14
	vblendpd	$0x3, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm3, %ymm13, %ymm3

	jmp		3f

2:

	// offset==3

	vmovapd		0(%r13), %ymm12
	vmovapd		0(%r15), %ymm14
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vmovapd		32(%r13), %ymm13
	vmovapd		32(%r15), %ymm14
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm0, %ymm12, %ymm0
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm1, %ymm13, %ymm1
	vmovapd		64(%r13), %ymm12
	vmovapd		64(%r15), %ymm14
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vmovapd		96(%r13), %ymm13
	vmovapd		96(%r15), %ymm14
	vblendpd	$0x7, %ymm14, %ymm13, %ymm13
	vperm2f128	$0x01, %ymm12, %ymm12, %ymm14
	vshufpd		$0x5, %ymm12, %ymm14, %ymm12
	vperm2f128	$0x01, %ymm13, %ymm13, %ymm14
	vshufpd		$0x5, %ymm13, %ymm14, %ymm13
	vmulpd		%ymm12, %ymm15, %ymm12
	vaddpd		%ymm2, %ymm12, %ymm2
	vmulpd		%ymm13, %ymm15, %ymm13
	vaddpd		%ymm3, %ymm13, %ymm3

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_4x4_gen_lib4, .-inner_blend_scale_ab_4x4_gen_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta=0.0
//
// input arguments:
// r10   <- alpha
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
// r10   <- alpha
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_A0_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_a0_4x4_lib4, @function
inner_blend_scale_a0_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_a0_4x4_lib4:
#endif
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %ymm15

	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_a0_4x4_lib4, .-inner_blend_scale_a0_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blender_loader for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
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
// r10   <- C
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_4x4_lib4, @function
inner_blend_scale_11_4x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_11_4x4_lib4:
#endif	
#endif	
	
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_11_4x4_lib4, .-inner_blend_scale_11_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_4x4_lib4, @function
inner_edge_dpotrf_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_4x4_lib4:
#endif
#endif
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX)
	vmovsd	.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd	LC04(%rip), %xmm14 // 1.0
#endif

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			1f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
2:
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0


	vpermilpd	$0x3, %xmm0, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm1, %ymm1
	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			3f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
4:
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1


	vpermilpd	$0x0, %ymm0, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vpermilpd	$0x0, %ymm1, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			5f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
6:
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2


	vpermilpd	$0xc, %ymm0, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vpermilpd	$0xc, %ymm1, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vpermilpd	$0xc, %ymm2, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			7f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
8:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3

	jmp		0f

1:
	vxorpd	%ymm13, %ymm13, %ymm13
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

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_4x4_lib4, .-inner_edge_dpotrf_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_4x4_vs_lib4, @function
inner_edge_dpotrf_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_4x4_vs_lib4:
#endif
#endif
	
	vxorpd	%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX)
	vmovsd	.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd	LC04(%rip), %xmm14 // 1.0
#endif

	vmovsd		%xmm0, %xmm0, %xmm13
	vucomisd	%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe			1f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
2:
	cmpl		$2, %r11d
	vmovsd		%xmm13, 0(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm0

	jl			0f // ret

	vpermilpd	$0x3, %xmm0, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm1, %ymm1
	vpermilpd	$0x3, %xmm1, %xmm13
	vucomisd	%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe			3f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
4:
	cmpl		$3, %r11d
	vmovsd		%xmm13, 8(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm1

	jl			0f // ret

	vpermilpd	$0x0, %ymm0, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vpermilpd	$0x0, %ymm1, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm2, %ymm2
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd	%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe			5f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
6:
	cmpl		$4, %r11d
	vmovsd		%xmm13, 16(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm2

	jl			0f // ret

	vpermilpd	$0xc, %ymm0, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vpermilpd	$0xc, %ymm1, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vpermilpd	$0xc, %ymm2, %ymm13
	vperm2f128	$0x11, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm12
	vsubpd		%ymm12, %ymm3, %ymm3
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vucomisd	%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe			7f
	vsqrtsd		%xmm13, %xmm13, %xmm13
	vdivsd		%xmm13, %xmm14, %xmm13
8:
	vmovsd		%xmm13, 24(%r10)
	vmovddup	%xmm13, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm3, %ymm13, %ymm3

	jmp		0f

1:
	vxorpd	%ymm13, %ymm13, %ymm13
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

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_4x4_vs_lib4, .-inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0

	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_4x4_vs_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	cmpl			$2, %r12d
	vmulpd			%ymm0, %ymm13, %ymm0

	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r12d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1

	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r12d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2

	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_4x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_4x4_lib4, @function
inner_edge_dtrsm_rlt_one_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_4x4_lib4:
#endif
#endif
	
	vbroadcastsd	8(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1

	vbroadcastsd	16(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_4x4_lib4, .-inner_edge_dtrsm_rlt_one_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// unit diagonal
//
// input arguments:
// r10  <- D
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- D
// r11d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_4x4_vs_lib4, @function
inner_edge_dtrsm_rlt_one_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_4x4_vs_lib4:
#endif
#endif
	
	cmpl			$2, %r11d

	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm1, %ymm1

	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r11d
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2
	vbroadcastsd	48(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm2, %ymm2

	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	56(%r10), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3
	vbroadcastsd	88(%r10), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm12
	vsubpd			%ymm12, %ymm3, %ymm3

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_4x4_vs_lib4, .-inner_edge_dtrsm_rlt_one_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = upper
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_4x4_lib4, @function
inner_edge_dtrsm_rut_inv_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_4x4_lib4:
#endif
#endif
	
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_4x4_lib4, .-inner_edge_dtrsm_rut_inv_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = lower
// tran = transposed
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- D
// r11  <- inv_diag_D
// r12d <- kn
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_rut_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_4x4_vs_lib4:
#endif
#endif
	
	cmpl			$3, %r12d
	jle				0f

	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:
	cmpl			$2, %r12d
	jle				1f

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

1:
	cmpl			$1, %r12d
	jle				2f

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

2:

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_4x4_vs_lib4, .-inner_edge_dtrsm_rut_inv_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = right
// uplo = up
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
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
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_INV_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_inv_4x4_lib4, @function
inner_edge_dtrsm_run_inv_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_inv_4x4_lib4:
#endif
#endif

	// first column
	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0

	// second column
	vbroadcastsd	32(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1
	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1

	// third column
	vbroadcastsd	64(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	72(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2
	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2

	// fourth column
	vbroadcastsd	96(%r10), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	104(%r10), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	112(%r10), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_inv_4x4_lib4, .-inner_edge_dtrsm_run_inv_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = lower
// tran = normal
// unit diagonal
//
// input arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lln_one_4x4_lib4, @function
inner_edge_dtrsm_lln_one_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lln_one_4x4_lib4:
#endif
#endif

	vxorpd		%ymm14, %ymm14, %ymm14

	vmovapd		0(%r10), %ymm12
	vblendpd	$0x1, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3

	vmovapd		32(%r10), %ymm12
	vblendpd	$0x3, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x00, %ymm0, %ymm0, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x00, %ymm1, %ymm1, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x00, %ymm2, %ymm2, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x00, %ymm3, %ymm3, %ymm13
	vpermilpd	$0xf, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3

	vmovapd		64(%r10), %ymm12
	vblendpd	$0x7, %ymm14, %ymm12, %ymm12
	vperm2f128	$0x11, %ymm0, %ymm0, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm0, %ymm0
	vperm2f128	$0x11, %ymm1, %ymm1, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vperm2f128	$0x11, %ymm2, %ymm2, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vperm2f128	$0x11, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm12, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lln_one_4x4_lib4, .-inner_edge_dtrsm_lln_one_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = upper
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
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
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_4x4_lib4, @function
inner_edge_dtrsm_lun_inv_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_4x4_lib4:
#endif
#endif
	
	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r11), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r11), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r11), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3


	vbroadcastsd	0(%r11), %ymm12

	vmulpd			%ymm0, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm0, %ymm0

	vmulpd			%ymm1, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm1, %ymm1

	vmulpd			%ymm2, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm2, %ymm2

	vmulpd			%ymm3, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lun_inv_4x4_lib4, .-inner_edge_dtrsm_lun_inv_4x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution:
// side = left
// uplo = upper
// tran = normal
// requires explicit inverse of diagonal
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- km
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- km
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_4x4_vs_lib4, @function
inner_edge_dtrsm_lun_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_4x4_vs_lib4:
#endif
#endif
	
	cmpl	$3, %r12d
	jle		0f

	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r11), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0xf, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

0:
	cmpl	$2, %r12d
	jle		1f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r11), %ymm12

	vperm2f128		$0x11, %ymm0, %ymm0, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vperm2f128		$0x11, %ymm1, %ymm1, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vperm2f128		$0x11, %ymm2, %ymm2, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vperm2f128		$0x11, %ymm3, %ymm3, %ymm14
	vpermilpd		$0x0, %ymm14, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

1:
	cmpl	$1, %r12d
	jle		2f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r11), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm1, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm2, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vmulpd			%ymm13, %ymm14, %ymm15
	vsubpd			%ymm15, %ymm3, %ymm3

2:

	vbroadcastsd	0(%r11), %ymm12

	vmulpd			%ymm0, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm0, %ymm0

	vmulpd			%ymm1, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm1, %ymm1

	vmulpd			%ymm2, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm2, %ymm2

	vmulpd			%ymm3, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lun_inv_4x4_vs_lib4, .-inner_edge_dtrsm_lun_inv_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// LU factorization without pivoting
//
// input arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- inv_diag_E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGETRF_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgetrf_4x4_lib4, @function
inner_edge_dgetrf_4x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgetrf_4x4_lib4:
#endif
#endif
	
#if defined(OS_LINUX)
	vmovsd	.LC04(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd	LC04(%rip), %xmm14 // 1.0
#endif
	vmovddup	%xmm14, %xmm14

	// first column
//	vblendpd	$0x1, %ymm0, %ymm12, %ymm12
	vmovapd		%ymm0, %ymm12
	vmovddup	%xmm0, %xmm13
	vdivpd		%xmm13, %xmm14, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmovsd		%xmm13, 0(%r10)
	vmulpd		%ymm0, %ymm13, %ymm0
	vblendpd	$0x1, %ymm12, %ymm0, %ymm0

	// second column
	vmovddup	%xmm1, %xmm12
	vperm2f128	$0x00, %ymm12, %ymm12, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm1, %ymm1
	vblendpd	$0x2, %ymm1, %ymm13, %ymm12

	vpermilpd	$0x3, %xmm1, %xmm13
	vdivpd		%xmm13, %xmm14, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmovsd		%xmm13, 8(%r10)
	vmulpd		%ymm1, %ymm13, %ymm1
	vblendpd	$0x3, %ymm12, %ymm1, %ymm1

	// third column
	vmovddup	%xmm2, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vblendpd	$0x2, %ymm2, %ymm13, %ymm12

	vpermilpd	$0x3, %xmm2, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm2, %ymm2
	vblendpd	$0x4, %ymm2, %ymm12, %ymm12

	vextractf128	$0x1, %ymm2, %xmm13
	vmovddup	%xmm13, %xmm13
	vdivpd		%xmm13, %xmm14, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmovsd		%xmm13, 16(%r10)
	vmulpd		%ymm2, %ymm13, %ymm2
	vblendpd	$0x7, %ymm12, %ymm2, %ymm2

	// fourth column
	vmovddup	%xmm3, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm0, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vblendpd	$0x2, %ymm3, %ymm13, %ymm12

	vpermilpd	$0x3, %xmm3, %xmm13
	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmulpd		%ymm1, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vblendpd	$0x4, %ymm3, %ymm12, %ymm12

	vperm2f128	$0x11, %ymm3, %ymm3, %ymm13
	vpermilpd	$0x0, %ymm13, %ymm13
	vmulpd		%ymm2, %ymm13, %ymm15
	vsubpd		%ymm15, %ymm3, %ymm3
	vblendpd	$0x8, %ymm3, %ymm12, %ymm12
	
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd	$0x3, %xmm13, %xmm13
	vdivpd		%xmm13, %xmm14, %xmm13
//	vperm2f128	$0x00, %ymm13, %ymm13, %ymm13
	vmovsd		%xmm13, 24(%r10)
//	vmulpd		%ymm3, %ymm13, %ymm3
	vblendpd	$0x7, %ymm12, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgetrf_4x4_lib4, .-inner_edge_dgetrf_4x4_lib4
#endif
#endif





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

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_lib4, @function
inner_store_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_lib4:
#endif
#endif
	
	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_lib4, .-inner_store_4x4_lib4
#endif
#endif





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

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_vs_lib4, @function
inner_store_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_vs_lib4:
#endif
#endif
	
	vcvtsi2sd	%r11d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovupd		.LC02(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r10)
	jl			0f // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r10)
	jl			0f // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r10)
	je			0f // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r10)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_vs_lib4, .-inner_store_4x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// store n lower triangular
//
// input arguments:
// r10   <- D
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_4x4_lib4, @function
inner_store_l_4x4_lib4:
#elif defined(OS_MAC)
_inner_store_l_4x4_lib4:
#endif
#endif
	
	vmovapd		%ymm0, 0(%r10)
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmovapd		%ymm1, 32(%r10)
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmovapd		%ymm2, 64(%r10)
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmovapd		%ymm3, 96(%r10)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_lib4, .-inner_store_l_4x4_lib4
#endif
#endif





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

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_4X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_4x4_vs_lib4, @function
inner_store_l_4x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_l_4x4_vs_lib4:
#endif
#endif
	
	vcvtsi2sd	%r11d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovupd		.LC02(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r10)
	jl			0f // end
	cmpl		$3, %r12d
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmaskmovpd	%ymm1, %ymm15, 32(%r10)
	jl			0f // end
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmaskmovpd	%ymm2, %ymm15, 64(%r10)
	je			0f // end
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmaskmovpd	%ymm3, %ymm15, 96(%r10)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_4x4_vs_lib4, .-inner_store_l_4x4_vs_lib4
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
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
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
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4X4_GEN_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4x4_gen_lib4, @function
inner_store_4x4_gen_lib4:
#elif defined(OS_MAC)
_inner_store_4x4_gen_lib4:
#endif
#endif
	
	// compute mask for rows
	vcvtsi2sd	%r13d, %xmm14, %xmm14
	vcvtsi2sd	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovupd		.LC02(%rip), %ymm12
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm12
#endif
	vmovddup	%xmm14, %xmm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm14, %ymm14, %ymm14
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm12, %ymm14, %ymm14
	vsubpd		%ymm15, %ymm12, %ymm15
	vandpd		%ymm14, %ymm15, %ymm15

	// compute number of cols
	cmpl	$4, %eax
	jle		0f
	movl	$4, %eax
0:
	subl	%r15d, %eax
	movl	%eax, %r15d

	cmpl	$0, %r10d
	jg		0f

	// offset==0

	cmpl		$2, %r15d
	vmaskmovpd	%ymm0, %ymm15,  0(%r11)
	jl			3f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm15, 32(%r11)
	jl			3f // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r11)
	je			3f // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r11)

	jmp		3f

0:
	
	movq	%r11, %rbx // D0
	addq	%r12, %rbx // D1 <- D0 + 4*sdd*sizeof(double)

	cmpl	$1, %r10d
	jg		1f

	// offset==1

	vperm2f128	$0x01, %ymm0, %ymm0, %ymm12
	vshufpd		$0x5, %ymm0, %ymm12, %ymm0
	vperm2f128	$0x01, %ymm1, %ymm1, %ymm12
	vshufpd		$0x5, %ymm1, %ymm12, %ymm1
	vperm2f128	$0x01, %ymm2, %ymm2, %ymm12
	vshufpd		$0x5, %ymm2, %ymm12, %ymm2
	vperm2f128	$0x01, %ymm3, %ymm3, %ymm12
	vshufpd		$0x5, %ymm3, %ymm12, %ymm3

	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm15, %ymm12, %ymm15

#if defined(OS_LINUX)
	vmovupd		.LC08(%rip), %ymm12
	vmovupd		.LC05(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC08(%rip), %ymm12
	vmovupd		LC05(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm15, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	cmpl		$2, %r15d
	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm0, %ymm13, 0(%rbx)
	jl			3f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm1, %ymm13, 32(%rbx)
	jl			3f // end
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm2, %ymm13, 64(%rbx)
	je			3f // end
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm3, %ymm13, 96(%rbx)

	jmp		3f

1:

	cmpl	$2, %r10d
	jg		2f

	// offset==2

	vperm2f128	$0x01, %ymm0, %ymm0, %ymm0
	vperm2f128	$0x01, %ymm1, %ymm1, %ymm1
	vperm2f128	$0x01, %ymm2, %ymm2, %ymm2
	vperm2f128	$0x01, %ymm3, %ymm3, %ymm3

	vperm2f128	$0x01, %ymm15, %ymm15, %ymm15

#if defined(OS_LINUX)
	vmovupd		.LC09(%rip), %ymm12
	vmovupd		.LC06(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC09(%rip), %ymm12
	vmovupd		LC06(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm15, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	cmpl		$2, %r15d
	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm0, %ymm13, 0(%rbx)
	jl			3f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm1, %ymm13, 32(%rbx)
	jl			3f // end
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm2, %ymm13, 64(%rbx)
	je			3f // end
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm3, %ymm13, 96(%rbx)

	jmp		3f

2:

	// offset==3

	vperm2f128	$0x01, %ymm0, %ymm0, %ymm12
	vshufpd		$0x5, %ymm12, %ymm0, %ymm0
	vperm2f128	$0x01, %ymm1, %ymm1, %ymm12
	vshufpd		$0x5, %ymm12, %ymm1, %ymm1
	vperm2f128	$0x01, %ymm2, %ymm2, %ymm12
	vshufpd		$0x5, %ymm12, %ymm2, %ymm2
	vperm2f128	$0x01, %ymm3, %ymm3, %ymm12
	vshufpd		$0x5, %ymm12, %ymm3, %ymm3

	vperm2f128	$0x01, %ymm15, %ymm15, %ymm12
	vshufpd		$0x5, %ymm12, %ymm15, %ymm15

#if defined(OS_LINUX)
	vmovupd		.LC10(%rip), %ymm12
	vmovupd		.LC07(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC10(%rip), %ymm12
	vmovupd		LC07(%rip), %ymm13
#endif
	vandpd		%ymm12, %ymm15, %ymm12
	vandpd		%ymm13, %ymm15, %ymm13

	cmpl		$2, %r15d
	vmaskmovpd	%ymm0, %ymm12, 0(%r11)
	vmaskmovpd	%ymm0, %ymm13, 0(%rbx)
	jl			3f // end
	cmpl		$3, %r15d
	vmaskmovpd	%ymm1, %ymm12, 32(%r11)
	vmaskmovpd	%ymm1, %ymm13, 32(%rbx)
	jl			3f // end
	vmaskmovpd	%ymm2, %ymm12, 64(%r11)
	vmaskmovpd	%ymm2, %ymm13, 64(%rbx)
	je			3f // end
	vmaskmovpd	%ymm3, %ymm12, 96(%r11)
	vmaskmovpd	%ymm3, %ymm13, 96(%rbx)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4x4_gen_lib4, .-inner_store_4x4_gen_lib4
#endif
#endif





//                                  rdi    rsi            rdx        rcx        r8
// void kernel_dgemm_nt_4x4_a0_lib4(int k, double *alpha, double *A, double *B, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_a0_lib4
	.type kernel_dgemm_nt_4x4_a0_lib4, @function
kernel_dgemm_nt_4x4_a0_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_a0_lib4
_kernel_dgemm_nt_4x4_a0_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend for generic alpha and beta=0.0

	movq	%rsi, %r10 // alpha

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_A0_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_a0_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_a0_4x4_lib4
#endif
#endif


	// store n

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dgemm_nt_4x4_a0_lib4, .-kernel_dgemm_nt_4x4_a0_lib4
#endif





//                               rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dgemm_nt_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_lib4
	.type kernel_dgemm_nt_4x4_lib4, @function
kernel_dgemm_nt_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_lib4
_kernel_dgemm_nt_4x4_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dgemm_nt_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_vs_lib4
	.type kernel_dgemm_nt_4x4_vs_lib4, @function
kernel_dgemm_nt_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_vs_lib4
_kernel_dgemm_nt_4x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dgemm_nt_4x4_vs_lib4, .-kernel_dgemm_nt_4x4_vs_lib4
#endif





//                                   rdi    rsi            rdx        rcx        r8            r9           rsp+8      rsp+16   rsp+24       rsp+32     rsp+40   rsp+48  rsp+56  rsp+64  rsp+72
// void kernel_dgemm_nt_4x4_gen_lib4(int k, double *alpha, double *A, double *B, double *beta, int offsetC, double *C, int sdc, int offsetD, double *D, int sdd, int m0, int m1, int n0, int n1);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_4x4_gen_lib4
	.type kernel_dgemm_nt_4x4_gen_lib4, @function
kernel_dgemm_nt_4x4_gen_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_4x4_gen_lib4
_kernel_dgemm_nt_4x4_gen_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movl	%r9d, %r12d // offsetC
	movq	STACKSIZE +  8(%rsp), %r13 // C
	movl	STACKSIZE + 16(%rsp), %r14d // sdc
	sall	$5, %r14d // 4*sdc*sizeof(double)
//	movslq	%r14d, %r14
	movl	STACKSIZE + 64(%rsp), %r15d // n0

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_GEN_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_gen_lib4
#endif
#endif


	// store n gen

	movq	STACKSIZE + 24(%rsp), %r10 // offsetD
	movq	STACKSIZE + 32(%rsp), %r11 // D
	movl	STACKSIZE + 40(%rsp), %r12d // sdd
	sall	$5, %r12d // 4*sdb*sizeof(double)
//	movslq	%r12d, %r12
	movl	STACKSIZE + 48(%rsp), %r13d // m0
	movl	STACKSIZE + 56(%rsp), %r14d // m1
	movl	STACKSIZE + 64(%rsp), %r15d // n0
	movl	STACKSIZE + 72(%rsp), %eax // n1

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_GEN_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_gen_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_gen_lib4
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
	.size	kernel_dgemm_nt_4x4_gen_lib4, .-kernel_dgemm_nt_4x4_gen_lib4
#endif





//                               rdi    rsi            rdx        rcx        r8       r9            rsp+8      rsp+16
// void kernel_dgemm_nn_4x4_lib4(int k, double *alpha, double *A, double *B, int sdb, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_4x4_lib4
	.type kernel_dgemm_nn_4x4_lib4, @function
kernel_dgemm_nn_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_4x4_lib4
_kernel_dgemm_nn_4x4_lib4:
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


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B
	movl	%r8d, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_4x4_lib4
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE + 8(%rsp), %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dgemm_nn_4x4_lib4, .-kernel_dgemm_nn_4x4_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8       r9            rsp+8      rsp+16     rsp+24  rsp+32
// void kernel_dgemm_nn_4x4_vs_lib4(int k, double *alpha, double *A, double *B, int sdb, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_4x4_vs_lib4
	.type kernel_dgemm_nn_4x4_vs_lib4, @function
kernel_dgemm_nn_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_4x4_vs_lib4
_kernel_dgemm_nn_4x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B
	movl	%r8d, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_4x4_lib4
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE + 8(%rsp), %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // km 
	movl	STACKSIZE + 32(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dgemm_nn_4x4_vs_lib4, .-kernel_dgemm_nn_4x4_vs_lib4
#endif





//                               rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dsyrk_nt_l_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_lib4
	.type kernel_dsyrk_nt_l_4x4_lib4, @function
kernel_dsyrk_nt_l_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_lib4
_kernel_dsyrk_nt_l_4x4_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D


#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	call	inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	callq	_inner_store_l_4x4_lib4
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
	.size	kernel_dsyrk_nt_l_4x4_lib4, .-kernel_dsyrk_nt_l_4x4_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dsyrk_nt_l_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_4x4_vs_lib4
_kernel_dsyrk_nt_l_4x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend 

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 


#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call	inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq	_inner_store_l_4x4_vs_lib4
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
	.size	kernel_dsyrk_nt_l_4x4_vs_lib4, .-kernel_dsyrk_nt_l_4x4_vs_lib4
#endif





//                                  rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dtrmm_nt_ru_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_lib4
	.type kernel_dtrmm_nt_ru_4x4_lib4, @function
kernel_dtrmm_nt_ru_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_lib4
_kernel_dtrmm_nt_ru_4x4_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movq	%rdx, %r10
	movq	%rcx, %r11

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrmm_nt_ru_4x4_lib4, .-kernel_dtrmm_nt_ru_4x4_lib4
#endif





//                                     rdi    rsi            rdx        rcx        r8            r9         rsp+8     rsp+16   rsp+24
// void kernel_dtrmm_nt_ru_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);
// it holds k==kn !!!!!

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_4x4_vs_lib4
	.type kernel_dtrmm_nt_ru_4x4_vs_lib4, @function
kernel_dtrmm_nt_ru_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_4x4_vs_lib4
_kernel_dtrmm_nt_ru_4x4_vs_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movq	%rcx, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_4x4_vs_lib4
#endif
#endif


	// call inner loader nn

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE +  8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrmm_nt_ru_4x4_vs_lib4, .-kernel_dtrmm_nt_ru_4x4_vs_lib4
#endif





//                                  edi    rsi        rdx        rcx        r8         r9
// void kernel_dpotrf_nt_l_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_4x4_lib4
	.type kernel_dpotrf_nt_l_4x4_lib4, @function
kernel_dpotrf_nt_l_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_4x4_lib4
_kernel_dpotrf_nt_l_4x4_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	%r9, %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib4
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
	.size	kernel_dpotrf_nt_l_4x4_lib4, .-kernel_dpotrf_nt_l_4x4_lib4
#endif





//                                     edi    rsi        rdx        rcx        r8         r9                  rsp+8   rsp+16
// void kernel_dpotrf_nt_l_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dpotrf_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_4x4_vs_lib4
_kernel_dpotrf_nt_l_4x4_vs_lib4:
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


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	%r9, %r10  // inv_diag_D 
	movl	STACKSIZE + 16(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib4
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
	.size	kernel_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dpotrf_nt_l_4x4_vs_lib4
#endif





//                                        edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24
// void kernel_dsyrk_dpotrf_nt_l_4x4_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_4x4_lib4
	.type kernel_dsyrk_dpotrf_nt_l_4x4_lib4, @function
kernel_dsyrk_dpotrf_nt_l_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_4x4_lib4
_kernel_dsyrk_dpotrf_nt_l_4x4_lib4:
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


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 24(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10  // D 

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib4
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
	.size	kernel_dsyrk_dpotrf_nt_l_4x4_lib4, .-kernel_dsyrk_dpotrf_nt_l_4x4_lib4
#endif





//                                           edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24             rsp+32   rsp+40
// void kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
	.type kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
_kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4:
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


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 24(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 40(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_4x4_vs_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10  // D 
	movl	STACKSIZE + 32(%rsp), %r11d // km 
	movl	STACKSIZE + 40(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_vs_lib4
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
	.size	kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_l_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx        r8         r9         rsp+8     
// void kernel_dtrsm_nt_rl_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_lib4
_kernel_dtrsm_nt_rl_inv_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_lib4
#endif





//                                            edi     rsi         rdx         ecx     r8          r9          rsp+8      rsp+16     rsp+24     rsp+32
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4:
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


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10   // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10   // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx        r8         r9         rsp+8               rsp+16  rsp+24  
// void kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
_kernel_dtrsm_nt_rl_inv_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE + 8(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dtrsm_nt_rl_inv_4x4_vs_lib4
#endif





//                                               edi     rsi         rdx         ecx     r8          r9          rsp+8    rsp+16     rsp+24     rsp+32                rsp+40 rsp+48
// void kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4(int kp, double *Ap, double *Bp, int km, double *Am, double *Bm, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4:
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


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movq	%rdx, %r12  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%ecx, %r10d // km
	movq	%r8, %r11   // Am
	movq	%r9, %r12   // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	STACKSIZE + 8(%rsp), %r10  // C 

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_4x4_vs_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D 
	movl	STACKSIZE + 40(%rsp), %r11d // km 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx        r8         r9
// void kernel_dtrsm_nt_rl_one_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_lib4
	.type kernel_dtrsm_nt_rl_one_4x4_lib4, @function
kernel_dtrsm_nt_rl_one_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_lib4
_kernel_dtrsm_nt_rl_one_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_one_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nt_rl_one_4x4_lib4, .-kernel_dtrsm_nt_rl_one_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx        r8         r9         rsp+8   rsp+16
// void kernel_dtrsm_nt_rl_one_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_4x4_vs_lib4
	.type kernel_dtrsm_nt_rl_one_4x4_vs_lib4, @function
kernel_dtrsm_nt_rl_one_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_4x4_vs_lib4
_kernel_dtrsm_nt_rl_one_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movl	STACKSIZE + 16(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_one_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE +  8(%rsp), %r11d // km 
	movl	STACKSIZE + 16(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nt_rl_one_4x4_vs_lib4, .-kernel_dtrsm_nt_rl_one_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx        r8         r9         rsp+8
// void kernel_dtrsm_nt_ru_inv_4x4_lib4(int k, double *A, double *B, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_lib4
	.type kernel_dtrsm_nt_ru_inv_4x4_lib4, @function
kernel_dtrsm_nt_ru_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_lib4
_kernel_dtrsm_nt_ru_inv_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE +  8(%rsp), %r11 // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rut_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nt_ru_inv_4x4_lib4, .-kernel_dtrsm_nt_ru_inv_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx        r8         r9         rsp+8                rsp+16  rsp+24
// void kernel_dtrsm_nt_ru_inv_4x4_vs_lib4(int k, double *A, double *B, double *C, double *D, double *E, double  *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_4x4_vs_lib4
	.type kernel_dtrsm_nt_ru_inv_4x4_vs_lib4, @function
kernel_dtrsm_nt_ru_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_4x4_vs_lib4
_kernel_dtrsm_nt_ru_inv_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d
	movq	%rsi, %r11
	movq	%rdx, %r12

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%rcx, %r10 // C

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	%r9, %r10  // E 
	movq	STACKSIZE +  8(%rsp), %r11 // inv_diag_E 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rut_inv_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_4x4_vs_lib4
#endif
#endif


	// store

	movq	%r8, %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nt_ru_inv_4x4_vs_lib4, .-kernel_dtrsm_nt_ru_inv_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx      r8         r9         rsp+8      rsp+16
// void kernel_dtrsm_nn_ru_inv_4x4_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_4x4_lib4
	.type kernel_dtrsm_nn_ru_inv_4x4_lib4, @function
kernel_dtrsm_nn_ru_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_4x4_lib4
_kernel_dtrsm_nn_ru_inv_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 
	movq	STACKSIZE + 16(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_run_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nn_ru_inv_4x4_lib4, .-kernel_dtrsm_nn_ru_inv_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx      r8         r9         rsp+8      rsp+16              rsp+24  rsp+32
// void kernel_dtrsm_nn_ru_inv_4x4_vs_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_4x4_vs_lib4
	.type kernel_dtrsm_nn_ru_inv_4x4_vs_lib4, @function
kernel_dtrsm_nn_ru_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_4x4_vs_lib4
_kernel_dtrsm_nn_ru_inv_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 
	movq	STACKSIZE + 16(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_run_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

	movq	STACKSIZE + 24(%rsp), %r11  // km 
	movq	STACKSIZE + 32(%rsp), %r12  // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nn_ru_inv_4x4_vs_lib4, .-kernel_dtrsm_nn_ru_inv_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx      r8         r9         rsp+8
// void kernel_dtrsm_nn_ll_one_4x4_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_4x4_lib4
	.type kernel_dtrsm_nn_ll_one_4x4_lib4, @function
kernel_dtrsm_nn_ll_one_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_4x4_lib4
_kernel_dtrsm_nn_ll_one_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lln_one_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nn_ll_one_4x4_lib4, .-kernel_dtrsm_nn_ll_one_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx      r8         r9         rsp+8      rsp+16  rsp+24
// void kernel_dtrsm_nn_ll_one_4x4_vs_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_4x4_vs_lib4
	.type kernel_dtrsm_nn_ll_one_4x4_vs_lib4, @function
kernel_dtrsm_nn_ll_one_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_4x4_vs_lib4
_kernel_dtrsm_nn_ll_one_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lln_one_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

	movq	STACKSIZE + 16(%rsp), %r11  // km 
	movq	STACKSIZE + 24(%rsp), %r12  // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nn_ll_one_4x4_vs_lib4, .-kernel_dtrsm_nn_ll_one_4x4_vs_lib4
#endif





//                                      edi    rsi        rdx        ecx      r8         r9         rsp+8      rsp+16
// void kernel_dtrsm_nn_lu_inv_4x4_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_4x4_lib4
	.type kernel_dtrsm_nn_lu_inv_4x4_lib4, @function
kernel_dtrsm_nn_lu_inv_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_4x4_lib4
_kernel_dtrsm_nn_lu_inv_4x4_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 
	movq	STACKSIZE + 16(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lun_inv_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dtrsm_nn_lu_inv_4x4_lib4, .-kernel_dtrsm_nn_lu_inv_4x4_lib4
#endif





//                                         edi    rsi        rdx        ecx      r8         r9         rsp+8      rsp+16              rsp+24  rsp+32
// void kernel_dtrsm_nn_lu_inv_4x4_vs_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_4x4_vs_lib4
	.type kernel_dtrsm_nn_lu_inv_4x4_vs_lib4, @function
kernel_dtrsm_nn_lu_inv_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_4x4_vs_lib4
_kernel_dtrsm_nn_lu_inv_4x4_vs_lib4:
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


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 8(%rsp), %r10  // E 
	movq	STACKSIZE + 16(%rsp), %r11  // inv_diag_E 
	movq	STACKSIZE + 24(%rsp), %r12  // km 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_4X4_VS_LIB4 // TODO
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lun_inv_4x4_vs_lib4 // TODO
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_4x4_vs_lib4 // TODO
#endif
#endif


	// store

	movq	%r9, %r10 // D

	movq	STACKSIZE + 24(%rsp), %r11  // km 
	movq	STACKSIZE + 32(%rsp), %r12  // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dtrsm_nn_lu_inv_4x4_vs_lib4, .-kernel_dtrsm_nn_lu_inv_4x4_vs_lib4
#endif





//                                edi    rsi        rdx        rcx      r8         r9         rsp+8
// void kernel_dgetrf_nn_4x4_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_4x4_lib4
	.type kernel_dgetrf_nn_4x4_lib4, @function
kernel_dgetrf_nn_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_4x4_lib4
_kernel_dgetrf_nn_4x4_lib4:
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


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 8(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_lib4
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
	.size	kernel_dgetrf_nn_4x4_lib4, .-kernel_dgetrf_nn_4x4_lib4
#endif





//                                   edi    rsi        rdx        rcx      r8         r9         rsp+8               rsp+16  rsp+24
// void kernel_dgetrf_nn_4x4_vs_lib4(int k, double *A, double *B, int sdb, double *C, double *D, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_4x4_vs_lib4
	.type kernel_dgetrf_nn_4x4_vs_lib4, @function
kernel_dgetrf_nn_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_4x4_vs_lib4
_kernel_dgetrf_nn_4x4_vs_lib4:
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


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // B
	movl	%ecx, %r13d // sdb
	sall	$5, %r13d // 4*sdb*sizeof(double)
	movslq	%r13d, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_4x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r8, %r10 // C

#if MACRO_LEVEL>=1
	INNER_SCALE_11_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_4x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 8(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_4x4_lib4
#endif
#endif


	// store

	movq	%r9, %r10 // D

	movq	STACKSIZE + 16(%rsp), %r11  // km 
	movq	STACKSIZE + 24(%rsp), %r12  // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dgetrf_nn_4x4_vs_lib4, .-kernel_dgetrf_nn_4x4_vs_lib4
#endif





#if 0
//                                   rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dlauum_nt_4x4_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dlauum_nt_4x4_lib4
	.type kernel_dlauum_nt_4x4_lib4, @function
kernel_dlauum_nt_4x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dlauum_nt_4x4_lib4
_kernel_dlauum_nt_4x4_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movq	%rcx, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DLAUUM_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dlauum_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dlauum_nt_4x4_lib4
#endif
#endif


	// call inner loader nn

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D

#if MACRO_LEVEL>=1
	INNER_STORE_L_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_4x4_lib4
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
	.size	kernel_dlauum_nt_4x4_vs_lib4, .-kernel_dlauum_nt_4x4_vs_lib4
#endif





//                                   rdi    rsi            rdx        rcx        r8            r9         rsp+8      rsp+16  rsp+24
// void kernel_dlauum_nt_4x4_vs_lib4(int k, double *alpha, double *A, double *B, double *beta, double *C, double *D, int km, int kn);
// it holds k==kn==km !!!!!

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dlauum_nt_4x4_vs_lib4
	.type kernel_dlauum_nt_4x4_vs_lib4, @function
kernel_dlauum_nt_4x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dlauum_nt_4x4_vs_lib4
_kernel_dlauum_nt_4x4_vs_lib4:
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


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d // k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movq	%rcx, %r12 // B
	addq	$128, %r12 // B+4*bs

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_4x4_lib4
#endif
#endif


	// call inner blender nn

#if MACRO_LEVEL>=1
	INNER_BLEND_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_4x4_lib4
#endif
#endif


	// initial triangle

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movq	%rcx, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DLAUUM_NT_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dlauum_nt_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dlauum_nt_4x4_vs_lib4
#endif
#endif


	// call inner loader nn

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11 // beta
	movq	%r9, %r12   // C

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_4X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_4x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_4x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // km 
	movl	STACKSIZE + 24(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_4X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_4x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_4x4_vs_lib4
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
	.size	kernel_dlauum_nt_4x4_vs_lib4, .-kernel_dlauum_nt_4x4_vs_lib4
#endif
#endif





	// read-only data
#if defined(OS_LINUX)
	.section	.rodata.cst32,"aM",@progbits,32
#elif defined(OS_MAC)
	.section	__TEXT,__const
#endif
#if 1
#if defined(OS_LINUX)
	.align 32
.LC00: // { -1 -1 -1 1 }
#elif defined(OS_MAC)
	.align 5
LC00: // { -1 -1 -1 1 }
#endif
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	1
#if defined(OS_LINUX)
	.align 32
.LC01: // { -1 -1 -1 -1 }
#elif defined(OS_MAC)
	.align 5
LC01: // { -1 -1 -1 -1 }
#endif
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	-1
#endif
#if defined(OS_LINUX)
	.align 32
.LC02: // { 3.5 2.5 1.5 0.5 }
#elif defined(OS_MAC)
	.align 5
LC02: // { 3.5 2.5 1.5 0.5 }
#endif
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
#if defined(OS_LINUX)
	.align 32
.LC03: // { 7.5 6.5 5.5 4.5 }
#elif defined(OS_MAC)
	.align 5
LC03: // { 7.5 6.5 5.5 4.5 }
#endif
	.long	0
	.long	1074921472
	.long	0
	.long	1075183616
	.long	0
	.long	1075445760
	.long	0
	.long	1075707904
#if defined(OS_LINUX)
	.align 32
.LC04: // { 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC04: // { 1.0 1.0 1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
#if defined(OS_LINUX)
	.align 32
.LC05: // { 1.0 1.0 1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC05: // { 1.0 1.0 1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
#if defined(OS_LINUX)
	.align 32
.LC06: // { 1.0 1.0 -1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC06: // { 1.0 1.0 -1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
#if defined(OS_LINUX)
	.align 32
.LC07: // { 1.0 -1.0 -1.0 -1.0 }
#elif defined(OS_MAC)
	.align 5
LC07: // { 1.0 -1.0 -1.0 -1.0 }
#endif
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	1072693248
#if defined(OS_LINUX)
	.align 32
.LC08: // { -1.0 -1.0 -1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC08: // { -1.0 -1.0 -1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
#if defined(OS_LINUX)
	.align 32
.LC09: // { -1.0 -1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC09: // { -1.0 -1.0 1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
#if defined(OS_LINUX)
	.align 32
.LC10: // { -1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
	.align 5
LC10: // { -1.0 1.0 1.0 1.0 }
#endif
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	-1074790400



#if defined(OS_LINUX)
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

