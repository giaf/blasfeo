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
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty



#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nt_12x4_lib4, @function
inner_kernel_dgemm_add_nt_12x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nt_12x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	movq	%r15, %r14 // A2 <- A1
	addq	%r12, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), %ymm12 // A0[0]
	vmovapd 0(%r13), %ymm15 // B[0]
	vmovapd 0(%r15), %ymm13 // A1[0]
	vmovapd 0(%r14), %ymm14 // A2[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	subl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd		$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd		$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd		32(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd		32(%r15), %ymm13 // A1[4]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd		32(%r13), %ymm15 // B[4]
	vmovapd		32(%r14), %ymm14 // A2[4]

	// unroll 1
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 	64(%r11), %ymm12 // A0[8]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 	64(%r15), %ymm13 // A1[8]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 	64(%r13), %ymm15 // B[8]
	vmovapd 	64(%r14), %ymm14 // A2[8]


	// unroll 2
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 	96(%r11), %ymm12 // A0[12]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 	96(%r15), %ymm13 // A1[12]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 	96(%r13), %ymm15 // B[12]
	vmovapd 	96(%r14), %ymm14 // A2[12]


	// unroll 3
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	addq		$128, %r11
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	addq		$128, %r15
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	addq		$128, %r13
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	addq		$128, %r14
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	cmpl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 	0(%r11), %ymm12 // A0[0]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 	0(%r15), %ymm13 // A1[0]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 	0(%r13), %ymm15 // B[0]
	vmovapd 	0(%r14), %ymm14 // A2[0]


	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	subl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd		$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd		$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd		32(%r11), %ymm12 // A0[4]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd		32(%r15), %ymm13 // A1[4]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd		32(%r13), %ymm15 // B[4]
	vmovapd		32(%r14), %ymm14 // A2[4]

	// unroll 1
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 	64(%r11), %ymm12 // A0[8]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 	64(%r15), %ymm13 // A1[8]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 	64(%r13), %ymm15 // B[8]
	vmovapd 	64(%r14), %ymm14 // A2[8]


	// unroll 2
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 	96(%r11), %ymm12 // A0[12]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 	96(%r15), %ymm13 // A1[12]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 	96(%r13), %ymm15 // B[12]
	vmovapd 	96(%r14), %ymm14 // A2[12]


	// unroll 3
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	addq		$128, %r11
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	addq		$128, %r15
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	addq		$128, %r13
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	addq		$128, %r14
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 $0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
//	cmpl		$4, %r10d
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
//	vmovapd 	0(%r11), %ymm12 // A0[0]
	vfmadd231pd	%ymm13, %ymm15, %ymm6
//	vmovapd 	0(%r15), %ymm13 // A1[0]
	vfmadd231pd	%ymm14, %ymm15, %ymm10
//	vmovapd 	0(%r13), %ymm15 // B[0]
//	vmovapd 	0(%r14), %ymm14 // A2[0]


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 	0(%r11), %ymm12 // A0[4]
	vmovapd 	0(%r15), %ymm13 // A1[4]
	vmovapd 	0(%r13), %ymm15 // B[4]
	vmovapd 	0(%r14), %ymm14 // A2[4]
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	addq		$32, %r11
	vfmadd231pd	%ymm13, %ymm15, %ymm4
	addq		$32, %r13
	vfmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm1
	addq		$32, %r15
	vfmadd231pd	%ymm13, %ymm15, %ymm5
	addq		$32, %r14
	vfmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm3
	subl		$1, %r10d
	vfmadd231pd	%ymm13, %ymm15, %ymm7
	cmpl		$0, %r10d
	vfmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm12, %ymm15, %ymm2
	vfmadd231pd	%ymm13, %ymm15, %ymm6
	vfmadd231pd	%ymm14, %ymm15, %ymm10


	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nt_12x4_lib4, .-inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nt_12x4_lib4, @function
inner_kernel_dgemm_sub_nt_12x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nt_12x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	movq	%r15, %r14 // A2 <- A1
	addq	%r12, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	// prefetch
	vmovapd 0(%r11), %ymm12 // A0[0]
	vmovapd 0(%r13), %ymm15 // B[0]
	vmovapd 0(%r15), %ymm13 // A1[0]
	vmovapd 0(%r14), %ymm14 // A2[0]

	cmpl	$4, %r10d
	jle		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	// unroll 0
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	subl			$4, %r10d
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd			$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 // A0[4]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 // A1[4]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r13), %ymm15 // B[4]
	vmovapd 		32(%r14), %ymm14 // A2[4]

	// unroll 1
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 // A0[8]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 // A1[8]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r13), %ymm15 // B[8]
	vmovapd 		64(%r14), %ymm14 // A2[8]


	// unroll 2
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128		 $0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		96(%r11), %ymm12 // A0[12]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		96(%r15), %ymm13 // A1[12]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		96(%r13), %ymm15 // B[12]
	vmovapd 		96(%r14), %ymm14 // A2[12]


	// unroll 3
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	addq			$128, %r11
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	addq			$128, %r15
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	addq			$128, %r13
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	addq			$128, %r14
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	cmpl			$4, %r10d
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		0(%r11), %ymm12 // A0[0]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		0(%r15), %ymm13 // A1[0]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		0(%r13), %ymm15 // B[0]
	vmovapd 		0(%r14), %ymm14 // A2[0]


	jg		1b // main loop 


0: // consider clean4-up
	
	cmpl	$3, %r10d
	jle		4f // clean1


	// unroll 0
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	subl			$4, %r10d
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd			$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		32(%r11), %ymm12 // A0[4]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		32(%r15), %ymm13 // A1[4]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		32(%r13), %ymm15 // B[4]
	vmovapd 		32(%r14), %ymm14 // A2[4]

	// unroll 1
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		64(%r11), %ymm12 // A0[8]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		64(%r15), %ymm13 // A1[8]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		64(%r13), %ymm15 // B[8]
	vmovapd 		64(%r14), %ymm14 // A2[8]


	// unroll 2
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128		 $0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vmovapd 		96(%r11), %ymm12 // A0[12]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vmovapd 		96(%r15), %ymm13 // A1[12]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd 		96(%r13), %ymm15 // B[12]
	vmovapd 		96(%r14), %ymm14 // A2[12]


	// unroll 3
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	addq			$128, %r11
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	addq			$128, %r15
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	addq			$128, %r13
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	addq			$128, %r14
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 		$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
//	cmpl			$4, %r10d
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 		$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
//	vmovapd 		0(%r11), %ymm12 // A0[0]
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
//	vmovapd 		0(%r15), %ymm13 // A1[0]
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
//	vmovapd 		0(%r13), %ymm15 // B[0]
//	vmovapd 		0(%r14), %ymm14 // A2[0]


	jmp		2f


4: // consider clean1-up loop

	cmpl	$0, %r10d
	jle		2f // return

	// clean-up loop
3: // clean up loop
	
	vmovapd 	0(%r11), %ymm12 // A0[4]
	vmovapd 	0(%r15), %ymm13 // A1[4]
	vmovapd 	0(%r13), %ymm15 // B[4]
	vmovapd 	0(%r14), %ymm14 // A2[4]
	vfnmadd231pd	%ymm12, %ymm15, %ymm0
	addq		$32, %r11
	vfnmadd231pd	%ymm13, %ymm15, %ymm4
	addq		$32, %r13
	vfnmadd231pd	%ymm14, %ymm15, %ymm8

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm1
	addq		$32, %r15
	vfnmadd231pd	%ymm13, %ymm15, %ymm5
	addq		$32, %r14
	vfnmadd231pd	%ymm14, %ymm15, %ymm9

	vperm2f128 	$0x1, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm3
	subl		$1, %r10d
	vfnmadd231pd	%ymm13, %ymm15, %ymm7
	cmpl		$0, %r10d
	vfnmadd231pd	%ymm14, %ymm15, %ymm11

	vshufpd 	$0x5, %ymm15, %ymm15, %ymm15
	vfnmadd231pd	%ymm12, %ymm15, %ymm2
	vfnmadd231pd	%ymm13, %ymm15, %ymm6
	vfnmadd231pd	%ymm14, %ymm15, %ymm10


	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nt_12x4_lib4, .-inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// rbx   <- dirty
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
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- k
// r11   <- A+4*sda*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+(k/4)*sdb*sizeof(double)+(k%4)
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// rbx   <- dirty
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
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_ADD_NN_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_add_nn_12x4_lib4, @function
inner_kernel_dgemm_add_nn_12x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_add_nn_12x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)
	addq	%r15, %r12 // A2 <- A1 + 4*sda*sizeof(double)

	movq	%r13, %rax // B_next <- B
	addq	%r14, %rax // B_next <- B + 4*sda*sizeof(double)
	movq	%rax, %rbx // B_next2 <- B_next
	addq	%r14, %rbx // B_next2 <- B_next + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%rbx) // software prefetch
	prefetcht0	64(%rbx) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r15), %ymm14 // A1[0]
	vmovapd			0(%r12), %ymm15 // A2[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vfmadd231pd		%ymm12, %ymm13, %ymm0
	vfmadd231pd		%ymm14, %ymm13, %ymm4
	vfmadd231pd		%ymm15, %ymm13, %ymm8
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vfmadd231pd		%ymm12, %ymm13, %ymm1
	vfmadd231pd		%ymm14, %ymm13, %ymm5
	vfmadd231pd		%ymm15, %ymm13, %ymm9
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vfmadd231pd		%ymm12, %ymm13, %ymm2
	vfmadd231pd		%ymm14, %ymm13, %ymm6
	vfmadd231pd		%ymm15, %ymm13, %ymm10
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vfmadd231pd		%ymm12, %ymm13, %ymm3
	vfmadd231pd		%ymm14, %ymm13, %ymm7
	vfmadd231pd		%ymm15, %ymm13, %ymm11

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A0[0]
	vmovapd			32(%r15), %ymm14 // A1[0]
	vmovapd			32(%r12), %ymm15 // A2[0]
	vbroadcastsd	8(%r13), %ymm13 // B[0]
	vfmadd231pd		%ymm12, %ymm13, %ymm0
	vfmadd231pd		%ymm14, %ymm13, %ymm4
	vfmadd231pd		%ymm15, %ymm13, %ymm8
	vbroadcastsd	40(%r13), %ymm13 // B[1]
	vfmadd231pd		%ymm12, %ymm13, %ymm1
	vfmadd231pd		%ymm14, %ymm13, %ymm5
	vfmadd231pd		%ymm15, %ymm13, %ymm9
	vbroadcastsd	72(%r13), %ymm13 // B[2]
	vfmadd231pd		%ymm12, %ymm13, %ymm2
	vfmadd231pd		%ymm14, %ymm13, %ymm6
	vfmadd231pd		%ymm15, %ymm13, %ymm10
	vbroadcastsd	104(%r13), %ymm13 // B[3]
	vfmadd231pd		%ymm12, %ymm13, %ymm3
	vfmadd231pd		%ymm14, %ymm13, %ymm7
	vfmadd231pd		%ymm15, %ymm13, %ymm11


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A0[0]
	vmovapd			64(%r15), %ymm14 // A1[0]
	vmovapd			64(%r12), %ymm15 // A2[0]
	vbroadcastsd	16(%r13), %ymm13 // B[0]
	vfmadd231pd		%ymm12, %ymm13, %ymm0
	vfmadd231pd		%ymm14, %ymm13, %ymm4
	vfmadd231pd		%ymm15, %ymm13, %ymm8
	vbroadcastsd	48(%r13), %ymm13 // B[1]
	vfmadd231pd		%ymm12, %ymm13, %ymm1
	vfmadd231pd		%ymm14, %ymm13, %ymm5
	vfmadd231pd		%ymm15, %ymm13, %ymm9
	vbroadcastsd	80(%r13), %ymm13 // B[2]
	vfmadd231pd		%ymm12, %ymm13, %ymm2
	vfmadd231pd		%ymm14, %ymm13, %ymm6
	vfmadd231pd		%ymm15, %ymm13, %ymm10
	vbroadcastsd	112(%r13), %ymm13 // B[3]
	vfmadd231pd		%ymm12, %ymm13, %ymm3
	vfmadd231pd		%ymm14, %ymm13, %ymm7
	vfmadd231pd		%ymm15, %ymm13, %ymm11


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A0[0]
	vmovapd			96(%r15), %ymm14 // A1[0]
	vmovapd			96(%r12), %ymm15 // A2[0]
	vbroadcastsd	24(%r13), %ymm13 // B[0]
	vfmadd231pd		%ymm12, %ymm13, %ymm0
	vfmadd231pd		%ymm14, %ymm13, %ymm4
	vfmadd231pd		%ymm15, %ymm13, %ymm8
	vbroadcastsd	56(%r13), %ymm13 // B[1]
	vfmadd231pd		%ymm12, %ymm13, %ymm1
	vfmadd231pd		%ymm14, %ymm13, %ymm5
	vfmadd231pd		%ymm15, %ymm13, %ymm9
	vbroadcastsd	88(%r13), %ymm13 // B[2]
	vfmadd231pd		%ymm12, %ymm13, %ymm2
	vfmadd231pd		%ymm14, %ymm13, %ymm6
	vfmadd231pd		%ymm15, %ymm13, %ymm10
	vbroadcastsd	120(%r13), %ymm13 // B[3]
	vfmadd231pd		%ymm12, %ymm13, %ymm3
	vfmadd231pd		%ymm14, %ymm13, %ymm7
	vfmadd231pd		%ymm15, %ymm13, %ymm11

	addq	$128, %r11
	addq	$128, %r15
	addq	$128, %r12

	mov		%rax, %r13
	mov		%rbx, %rax
	addq	%r14, %rbx

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r15), %ymm14 // A1[0]
	vmovapd			0(%r12), %ymm15 // A2[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vfmadd231pd		%ymm12, %ymm13, %ymm0
	vfmadd231pd		%ymm14, %ymm13, %ymm4
	vfmadd231pd		%ymm15, %ymm13, %ymm8
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vfmadd231pd		%ymm12, %ymm13, %ymm1
	vfmadd231pd		%ymm14, %ymm13, %ymm5
	vfmadd231pd		%ymm15, %ymm13, %ymm9
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vfmadd231pd		%ymm12, %ymm13, %ymm2
	vfmadd231pd		%ymm14, %ymm13, %ymm6
	vfmadd231pd		%ymm15, %ymm13, %ymm10
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vfmadd231pd		%ymm12, %ymm13, %ymm3
	vfmadd231pd		%ymm14, %ymm13, %ymm7
	vfmadd231pd		%ymm15, %ymm13, %ymm11

	subl	$1, %r10d

	addq	$32, %r11
	addq	$32, %r15
	addq	$32, %r12
	addq	$8, %r13

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_add_nn_12x4_lib4, .-inner_kernel_dgemm_add_nn_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// rbx   <- dirty
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
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- k
// r11   <- A+4*sda*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+(k/4)*sdb*sizeof(double)+(k%4)
// r14   <- 4*sdb*sizeof(double)
// r15   <- dirty
// rax   <- dirty
// rbx   <- dirty
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
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemm_sub_nn_12x4_lib4, @function
inner_kernel_dgemm_sub_nn_12x4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemm_sub_nn_12x4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)
	addq	%r15, %r12 // A2 <- A1 + 4*sda*sizeof(double)

	movq	%r13, %rax // B_next <- B
	addq	%r14, %rax // B_next <- B + 4*sda*sizeof(double)
	movq	%rax, %rbx // B_next2 <- B_next
	addq	%r14, %rbx // B_next2 <- B_next + 4*sda*sizeof(double)

	cmpl	$4, %r10d
	jl		0f // consider clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	prefetcht0	0(%rbx) // software prefetch
	prefetcht0	64(%rbx) // software prefetch

	// unroll 0
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r15), %ymm14 // A1[0]
	vmovapd			0(%r12), %ymm15 // A2[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	subl	$4, %r10d

	// unroll 1
	vmovapd			32(%r11), %ymm12 // A0[0]
	vmovapd			32(%r15), %ymm14 // A1[0]
	vmovapd			32(%r12), %ymm15 // A2[0]
	vbroadcastsd	8(%r13), %ymm13 // B[0]
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vbroadcastsd	40(%r13), %ymm13 // B[1]
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vbroadcastsd	72(%r13), %ymm13 // B[2]
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vbroadcastsd	104(%r13), %ymm13 // B[3]
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11


	// unroll 2
	vmovapd			64(%r11), %ymm12 // A0[0]
	vmovapd			64(%r15), %ymm14 // A1[0]
	vmovapd			64(%r12), %ymm15 // A2[0]
	vbroadcastsd	16(%r13), %ymm13 // B[0]
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vbroadcastsd	48(%r13), %ymm13 // B[1]
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vbroadcastsd	80(%r13), %ymm13 // B[2]
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vbroadcastsd	112(%r13), %ymm13 // B[3]
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11


	// unroll 3
	vmovapd			96(%r11), %ymm12 // A0[0]
	vmovapd			96(%r15), %ymm14 // A1[0]
	vmovapd			96(%r12), %ymm15 // A2[0]
	vbroadcastsd	24(%r13), %ymm13 // B[0]
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vbroadcastsd	56(%r13), %ymm13 // B[1]
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vbroadcastsd	88(%r13), %ymm13 // B[2]
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vbroadcastsd	120(%r13), %ymm13 // B[3]
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	addq	$128, %r11
	addq	$128, %r15
	addq	$128, %r12

	mov		%rax, %r13
	mov		%rbx, %rax
	addq	%r14, %rbx

	cmpl	$3, %r10d

	jg		1b // main loop 


0: // consider clean1-up loop
	
	cmpl	$0, %r10d
	jle		2f // return

3: // clean1-up loop
	
	vmovapd			0(%r11), %ymm12 // A0[0]
	vmovapd			0(%r15), %ymm14 // A1[0]
	vmovapd			0(%r12), %ymm15 // A2[0]
	vbroadcastsd	0(%r13), %ymm13 // B[0]
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vbroadcastsd	32(%r13), %ymm13 // B[1]
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vbroadcastsd	64(%r13), %ymm13 // B[2]
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vbroadcastsd	96(%r13), %ymm13 // B[3]
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	subl	$1, %r10d

	addq	$32, %r11
	addq	$32, %r15
	addq	$32, %r12
	addq	$8, %r13

	cmpl	$0, %r10d

	jg		3b // clean up loop 


2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemm_sub_nn_12x4_lib4, .-inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10   <- A
// r11   <- 4*sda*sizeof(double)
// r12   <- B
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- A+4*4*sizeof(double)
// r11   <- 4*sda*sizeof(double)
// r12   <- B+4*4*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_12x4_lib4, @function
inner_edge_dtrmm_nt_ru_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_12x4_lib4:
#endif
#endif
	
	movq	%r10, %r15 // A1 <- A0
	addq	%r11, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	movq	%r15, %r14 // A2 <- A1
	addq	%r11, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r12), %ymm12
	vmovapd			0(%r10), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			0(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			0(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	vbroadcastsd	32(%r12), %ymm12
	vmovapd			32(%r10), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			32(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			32(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9

	vbroadcastsd	64(%r12), %ymm12
	vmovapd			64(%r10), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			64(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			64(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	72(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	80(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10

	vbroadcastsd	96(%r12), %ymm12
	vmovapd			96(%r10), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			96(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			96(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	104(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	112(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	120(%r12), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	addq			$128, %r10
	addq			$128, %r12

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_12x4_lib4, .-inner_edge_dtrmm_nt_ru_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B upper triangular
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- 4*sda*sizeof(double)
// r13   <- B
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10d  <- max(k-4,0)
// r11   <- A+4*4*sizeof(double)
// r12   <- 4*sda*sizeof(double)
// r13   <- B+4*4*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NT_RU_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nt_ru_12x4_vs_lib4, @function
inner_edge_dtrmm_nt_ru_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nt_ru_12x4_vs_lib4:
#endif
#endif
	
	movq	%r11, %r15 // A1 <- A0
	addq	%r12, %r15 // A1 <- A0 + 4*sda*sizeof(double)

	movq	%r15, %r14 // A2 <- A1
	addq	%r12, %r14 // A2 <- A1 + 4*sda*sizeof(double)

	vbroadcastsd	0(%r13), %ymm12
	addq			$32, %r13
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm13
	addq			$32, %r11
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			0(%r15), %ymm14
	addq			$32, %r15
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			0(%r14), %ymm15
	addq			$32, %r14
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			0(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			0(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	addq			$32, %r11
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	addq			$32, %r13
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	addq			$32, %r15
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	addq			$32, %r14

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			0(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			0(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	addq			$32, %r11
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	addq			$32, %r13
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	addq			$32, %r15
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	addq			$32, %r14

	cmpl	$0, %r10d
	jle		0f

	vbroadcastsd	0(%r13), %ymm12
	subl			$1, %r10d
	vmovapd			0(%r11), %ymm13
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vmovapd			0(%r15), %ymm14
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vmovapd			0(%r14), %ymm15
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	addq			$32, %r11
	vbroadcastsd	24(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	addq			$32, %r13
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	addq			$32, %r15
	vfmadd231pd		%ymm15, %ymm12, %ymm11
	addq			$32, %r14

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nt_ru_12x4_vs_lib4, .-inner_edge_dtrmm_nt_ru_12x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// edge for B lower triangular
//
// input arguments:
// r10   <- k
// r11   <- A0
// r12   <- bs*sda*sizeof(double)
// r13   <- B
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d40 d50 d60 d70]
// ymm9  <- [d41 d51 d61 d71]
// ymm10  <- [d42 d52 d62 d72]
// ymm11 <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

//
// output arguments:
// r10   <- k-(4-offB)
// r11   <- A+(4-offB)*bs*sizeof(double)
// r12   <- bs*sda*sizeof(double)
// r13   <- B-offB+bs*sdb*sizeof(double)
// r14   <- bs*sdb*sizeof(double)
// r15   <- offB
// rax   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty


#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMM_NN_RL_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmm_nn_rl_12x4_lib4, @function
inner_edge_dtrmm_nn_rl_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmm_nn_rl_12x4_lib4:
#endif
#endif
	
	movq	%r11, %rax // A1 <- A0
	addq	%r12, %rax // A1 <- A0 + 4*sda*sizeof(double)
	movq	%rax, %rbx // A2 <- A1
	addq	%r12, %rbx // A2 <- A1 + 4*sda*sizeof(double)

	cmpl	$0, %r15d
	jg		0f

	// offB==0

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	vmovapd			32(%r11), %ymm13
	vmovapd			32(%rax), %ymm14
	vmovapd			32(%rbx), %ymm15
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9

	vmovapd			64(%r11), %ymm13
	vmovapd			64(%rax), %ymm14
	vmovapd			64(%rbx), %ymm15
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	48(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	80(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10

	vmovapd			96(%r11), %ymm13
	vmovapd			96(%rax), %ymm14
	vmovapd			96(%rbx), %ymm15
	vbroadcastsd	24(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	56(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	88(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	120(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			$128, %rax // A1+4*bs*sizeof(double)
	addq			$128, %rbx // A2+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

	jmp		3f

0:
	cmpl	$1, %r15d
	jg		1f

	// offB==1

	addq			$8, %r13 // B+1*sizeof(double)

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	vmovapd			32(%r11), %ymm13
	vmovapd			32(%rax), %ymm14
	vmovapd			32(%rbx), %ymm15
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9

	vmovapd			64(%r11), %ymm13
	vmovapd			64(%rax), %ymm14
	vmovapd			64(%rbx), %ymm15
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	48(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	80(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10

	subl			$3, %r10d // k-3
	addq			$96, %r11 // A0+3*bs*sizeof(double)
	addq			$96, %rax // A1+3*bs*sizeof(double)
	addq			$96, %rbx // A2+3*bs*sizeof(double)
	addq			%r14, %r13
	subq			$8, %r13 // B+bs*sdb*sizeof(double)-1

	jmp		3f

1:
	cmpl	$2, %r15d
	jg		2f

	// offB==2

	addq			$16, %r13 // B+2*sizeof(double)

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	vmovapd			32(%r11), %ymm13
	vmovapd			32(%rax), %ymm14
	vmovapd			32(%rbx), %ymm15
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9

	subl			$2, %r10d // k-2
	addq			$64, %r11 // A0+2*bs*sizeof(double)
	addq			$64, %rax // A1+2*bs*sizeof(double)
	addq			$64, %rbx // A2+2*bs*sizeof(double)
	addq			%r14, %r13
	subq			$16, %r13 // B+bs*sdb*sizeof(double)-2

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	32(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	64(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10

	vmovapd			32(%r11), %ymm13
	vmovapd			32(%rax), %ymm14
	vmovapd			32(%rbx), %ymm15
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	72(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	104(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	vmovapd			64(%r11), %ymm13
	vmovapd			64(%rax), %ymm14
	vmovapd			64(%rbx), %ymm15
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	48(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	80(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	112(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	vmovapd			96(%r11), %ymm13
	vmovapd			96(%rax), %ymm14
	vmovapd			96(%rbx), %ymm15
	vbroadcastsd	24(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	56(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	88(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	120(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			$128, %rax // A1+4*bs*sizeof(double)
	addq			$128, %rbx // A2+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

	jmp		3f

2:
	// offB==3

	addq			$24, %r13 // B+3*sizeof(double)

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8

	subl			$1, %r10d // k-1
	addq			$32, %r11 // A0+1*bs*sizeof(double)
	addq			$32, %rax // A1+1*bs*sizeof(double)
	addq			$32, %rbx // A2+1*bs*sizeof(double)
	addq			%r14, %r13
	subq			$24, %r13 // B+bs*sdb*sizeof(double)-3

	vmovapd			0(%r11), %ymm13
	vmovapd			0(%rax), %ymm14
	vmovapd			0(%rbx), %ymm15
	vbroadcastsd	0(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	32(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9

	vmovapd			32(%r11), %ymm13
	vmovapd			32(%rax), %ymm14
	vmovapd			32(%rbx), %ymm15
	vbroadcastsd	8(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	40(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	72(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10

	vmovapd			64(%r11), %ymm13
	vmovapd			64(%rax), %ymm14
	vmovapd			64(%rbx), %ymm15
	vbroadcastsd	16(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	48(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	80(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	112(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	vmovapd			96(%r11), %ymm13
	vmovapd			96(%rax), %ymm14
	vmovapd			96(%rbx), %ymm15
	vbroadcastsd	24(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm0
	vfmadd231pd		%ymm14, %ymm12, %ymm4
	vfmadd231pd		%ymm15, %ymm12, %ymm8
	vbroadcastsd	56(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm1
	vfmadd231pd		%ymm14, %ymm12, %ymm5
	vfmadd231pd		%ymm15, %ymm12, %ymm9
	vbroadcastsd	88(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm2
	vfmadd231pd		%ymm14, %ymm12, %ymm6
	vfmadd231pd		%ymm15, %ymm12, %ymm10
	vbroadcastsd	120(%r13), %ymm12
	vfmadd231pd		%ymm13, %ymm12, %ymm3
	vfmadd231pd		%ymm14, %ymm12, %ymm7
	vfmadd231pd		%ymm15, %ymm12, %ymm11

	subl			$4, %r10d // k-4
	addq			$128, %r11 // A0+4*bs*sizeof(double)
	addq			$128, %rax // A1+4*bs*sizeof(double)
	addq			$128, %rbx // A2+4*bs*sizeof(double)
	addq			%r14, %r13 // B+bs*sdb*sizeof(double)

3:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmm_nn_rl_12x4_lib4, .-inner_edge_dtrmm_nn_rl_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend
//
// input arguments:
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_12x4_lib4, @function
inner_blend_12x4_lib4:
#elif defined(OS_MAC)
_inner_blend_12x4_lib4:
#endif
#endif
	
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

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_12x4_lib4, .-inner_blend_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 d63 db3]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 d63 db3]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_11_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_11_12x4_lib4, @function
inner_scale_11_12x4_lib4:
#elif defined(OS_MAC)
_inner_scale_11_12x4_lib4:
#endif
#endif
	

	movq	%r10, %r12 // C1 <- C0
	addq	%r11, %r12 // C1 <- C0 + 4*sdc*sizeof(double)
	movq	%r12, %r13 // C2 <- C1
	addq	%r11, %r13 // C2 <- C1 + 4*sdc*sizeof(double)

	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3

	vmovapd		0(%r12), %ymm15
	vaddpd		%ymm4, %ymm15, %ymm4
	vmovapd		32(%r12), %ymm15
	vaddpd		%ymm5, %ymm15, %ymm5
	vmovapd		64(%r12), %ymm15
	vaddpd		%ymm6, %ymm15, %ymm6
	vmovapd		96(%r12), %ymm15
	vaddpd		%ymm7, %ymm15, %ymm7

	vmovapd		0(%r13), %ymm15
	vaddpd		%ymm8, %ymm15, %ymm8
	vmovapd		32(%r13), %ymm15
	vaddpd		%ymm9, %ymm15, %ymm9
	vmovapd		64(%r13), %ymm15
	vaddpd		%ymm10, %ymm15, %ymm10
	vmovapd		96(%r13), %ymm15
	vaddpd		%ymm11, %ymm15, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_11_12x4_lib4, .-inner_scale_11_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_ab_12x4_lib4, @function
inner_scale_ab_12x4_lib4:
#elif defined(OS_MAC)
_inner_scale_ab_12x4_lib4:
#endif
#endif
		
	vbroadcastsd 0(%r10), %ymm15 // beta

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

	movq	%r12, %r15 // C1 <- C0
	addq	%r13, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	movq	%r15, %r14 // C2 <- C1
	addq	%r13, %r14 // C2 <- C1 + 4*sdc*sizeof(double)

	vbroadcastsd 0(%r11), %ymm14 // beta

	vmovapd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3

	vmovapd		0(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7

	vmovapd		0(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_ab_12x4_lib4, .-inner_scale_ab_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta=0
//
// input arguments:
// r10   <- &alpha
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- &alpha
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_A0_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_scale_a0_12x4_lib4, @function
inner_scale_a0_12x4_lib4:
#elif defined(OS_MAC)
_inner_scale_a0_12x4_lib4:
#endif
#endif
		
	vbroadcastsd 0(%r10), %ymm15 // beta

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_scale_a0_12x4_lib4, .-inner_scale_a0_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend and scale for generic alpha and beta
//
// input arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- &alpha
// r11   <- &beta
// r12   <- C
// r13   <- 4*sdc*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_AB_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_ab_12x4_lib4, @function
inner_blend_scale_ab_12x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_ab_12x4_lib4:
#endif
#endif
		

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vbroadcastsd 0(%r10), %ymm15 // beta

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vbroadcastsd 0(%r10), %ymm15 // beta

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11

	vbroadcastsd 0(%r10), %ymm15 // beta

	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

	movq	%r12, %r15 // C1 <- C0
	addq	%r13, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	movq	%r15, %r14 // C2 <- C1
	addq	%r13, %r14 // C2 <- C1 + 4*sdc*sizeof(double)

	vbroadcastsd 0(%r11), %ymm14 // beta

	vmovapd		0(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm0
	vmovapd		32(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm1
	vmovapd		64(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm2
	vmovapd		96(%r12), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm3

	vmovapd		0(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm4
	vmovapd		32(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm5
	vmovapd		64(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm6
	vmovapd		96(%r15), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm7

	vmovapd		0(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm8
	vmovapd		32(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm9
	vmovapd		64(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd		96(%r14), %ymm15
	vfmadd231pd	%ymm14, %ymm15, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_ab_12x4_lib4, .-inner_blend_scale_ab_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for generic alpha and beta=0.0
//
// input arguments:
// r10   <- &alpha
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- &alpha
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_A0_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_a0_12x4_lib4, @function
inner_blend_scale_a0_12x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_a0_12x4_lib4:
#endif
#endif
		

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm12
	vblendpd	$0x5, %ymm1, %ymm0, %ymm13
	vblendpd	$0xa, %ymm3, %ymm2, %ymm14
	vblendpd	$0x5, %ymm3, %ymm2, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm0
	vblendpd	$0x3, %ymm14, %ymm12, %ymm2
	vblendpd	$0xc, %ymm15, %ymm13, %ymm1
	vblendpd	$0x3, %ymm15, %ymm13, %ymm3

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm0, %ymm15, %ymm0
	vmulpd		%ymm1, %ymm15, %ymm1
	vmulpd		%ymm2, %ymm15, %ymm2
	vmulpd		%ymm3, %ymm15, %ymm3

	vblendpd	$0xa, %ymm5, %ymm4, %ymm12
	vblendpd	$0x5, %ymm5, %ymm4, %ymm13
	vblendpd	$0xa, %ymm7, %ymm6, %ymm14
	vblendpd	$0x5, %ymm7, %ymm6, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm4
	vblendpd	$0x3, %ymm14, %ymm12, %ymm6
	vblendpd	$0xc, %ymm15, %ymm13, %ymm5
	vblendpd	$0x3, %ymm15, %ymm13, %ymm7

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm4, %ymm15, %ymm4
	vmulpd		%ymm5, %ymm15, %ymm5
	vmulpd		%ymm6, %ymm15, %ymm6
	vmulpd		%ymm7, %ymm15, %ymm7

	vblendpd	$0xa, %ymm9, %ymm8, %ymm12
	vblendpd	$0x5, %ymm9, %ymm8, %ymm13
	vblendpd	$0xa, %ymm11, %ymm10, %ymm14
	vblendpd	$0x5, %ymm11, %ymm10, %ymm15

	vblendpd	$0xc, %ymm14, %ymm12, %ymm8
	vblendpd	$0x3, %ymm14, %ymm12, %ymm10
	vblendpd	$0xc, %ymm15, %ymm13, %ymm9
	vblendpd	$0x3, %ymm15, %ymm13, %ymm11

	vbroadcastsd 0(%r10), %ymm15 // alpha

	vmulpd		%ymm8, %ymm15, %ymm8
	vmulpd		%ymm9, %ymm15, %ymm9
	vmulpd		%ymm10, %ymm15, %ymm10
	vmulpd		%ymm11, %ymm15, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_a0_12x4_lib4, .-inner_blend_scale_a0_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// blender for alpha = 1.0 and beta = 1.0
//
// input arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- C
// r11   <- 4*sdc*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_SCALE_11_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_scale_11_12x4_lib4, @function
inner_blend_scale_11_12x4_lib4:
#elif defined(OS_MAC)
_inner_blend_scale_11_12x4_lib4:
#endif
#endif
	
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

	movq	%r10, %r15 // C1 <- C0
	addq	%r11, %r15 // C1 <- C0 + 4*sdc*sizeof(double)

	movq	%r15, %r14 // C2 <- C1
	addq	%r11, %r14 // C2 <- C1 + 4*sdc*sizeof(double)

	vmovapd		0(%r10), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r10), %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r10), %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r10), %ymm15
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

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_blend_scale_11_12x4_lib4, .-inner_blend_scale_11_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10   <- inv_diag_E
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_12x4_lib4, @function
inner_edge_dpotrf_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_12x4_lib4:
#endif
#endif
	
	vxorpd			%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX)
	vmovsd			.LC05(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC05(%rip), %xmm14 // 1.0
#endif

	vmovsd			%xmm0, %xmm0, %xmm13
	vucomisd		%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe				1f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
2:
	vmovsd			%xmm13, 0(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8


	vpermilpd		$0x3, %xmm0, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vpermilpd		$0x3, %xmm1, %xmm13
	vucomisd		%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe				3f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
4:
	vmovsd			%xmm13, 8(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9


	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd		$0x0, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd		$0x0, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd		%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe				5f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
6:
	vmovsd			%xmm13, 16(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10


	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vextractf128	$0x1, %ymm2, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vucomisd		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe				7f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
8:
	vmovsd			%xmm13, 24(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11

	jmp				0f

1:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				2b

3:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				4b

5:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				6b

7:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				8

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_12x4_lib4, .-inner_edge_dpotrf_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// cholesky factorization 
//
// input arguments:
// r10   <- inv_diag_E
// r11d  <- kn
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 da0 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DPOTRF_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dpotrf_12x4_vs_lib4, @function
inner_edge_dpotrf_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dpotrf_12x4_vs_lib4:
#endif
#endif
	
	vxorpd			%ymm15, %ymm15, %ymm15 // 0.0
#if defined(OS_LINUX)
	vmovsd			.LC05(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC05(%rip), %xmm14 // 1.0
#endif

	vmovsd			%xmm0, %xmm0, %xmm13
	vucomisd		%xmm15, %xmm13 // d_00 > 0.0 ?
	jbe				1f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
2:
	cmpl			$2, %r11d
	vmovsd			%xmm13, 0(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8

	jl				0f // ret

	vpermilpd		$0x3, %xmm0, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vpermilpd		$0x3, %xmm1, %xmm13
	vucomisd		%xmm15, %xmm13 // d_11 > 0.0 ?
	jbe				3f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
4:
	cmpl			$3, %r11d
	vmovsd			%xmm13, 8(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9

	jl				0f // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd		$0x0, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd		$0x0, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vextractf128	$0x1, %ymm2, %xmm13
	vucomisd		%xmm15, %xmm13 // d_22 > 0.0 ?
	jbe				5f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
6:
	cmpl			$4, %r11d
	vmovsd			%xmm13, 16(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10

	jl				0f // ret

	vextractf128	$0x1, %ymm0, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vextractf128	$0x1, %ymm1, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vextractf128	$0x1, %ymm2, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vextractf128	$0x1, %ymm3, %xmm13
	vpermilpd		$0x3, %xmm13, %xmm13
	vucomisd		%xmm15, %xmm13 // d_33 > 0.0 ?
	jbe				7f
	vsqrtsd			%xmm13, %xmm13, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
8:
	vmovsd			%xmm13, 24(%r10)
	vmovddup		%xmm13, %xmm13
	vperm2f128		$0x00, %ymm13, %ymm13, %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11

	jmp				0f

1:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				2b

3:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				4b

5:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				6b

7:
	vxorpd			%ymm13, %ymm13, %ymm13
	jmp				8

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dpotrf_12x4_vs_lib4, .-inner_edge_dpotrf_12x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10   <- E
// r11   <- inv_diag_E
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10   <- E
// r11   <- inv_diag_E
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_12x4_lib4, @function
inner_edge_dtrsm_rlt_inv_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_12x4_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8

	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_12x4_lib4, .-inner_edge_dtrsm_rlt_inv_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution for cholesky factorization 
//
// input arguments:
// r10   <- D
// r11   <- inv_diag_D
// r12d  <- kn
// ymm0  <- [d00 d11 d22 d33]
// ymm1  <- [d01 d10 d23 d32]
// ymm2  <- [d03 d12 d21 d30]
// ymm3  <- [d02 d13 d20 d31]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10   <- D
// r11   <- inv_diag_D
// r12d  <- kn
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d51 d62 d73]
// ymm5  <- [d41 d50 d63 d72]
// ymm6  <- [d43 d52 d61 d70]
// ymm7  <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_INV_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_inv_12x4_vs_lib4, @function
inner_edge_dtrsm_rlt_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_inv_12x4_vs_lib4:
#endif
#endif
	
	vbroadcastsd	0(%r11), %ymm13
	cmpl			$2, %r12d
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8

	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	cmpl			$3, %r12d
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vbroadcastsd	8(%r11), %ymm13
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9

	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	cmpl			$4, %r12d
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vbroadcastsd	16(%r11), %ymm13
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10

	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vbroadcastsd	24(%r11), %ymm13
	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_inv_12x4_vs_lib4, .-inner_edge_dtrsm_rlt_inv_12x4_vs_lib4
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_12x4_lib4, @function
inner_edge_dtrsm_rlt_one_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_12x4_lib4:
#endif
#endif
	
	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_12x4_lib4, .-inner_edge_dtrsm_rlt_one_12x4_lib4
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RLT_ONE_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rlt_one_12x4_vs_lib4, @function
inner_edge_dtrsm_rlt_one_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rlt_one_12x4_vs_lib4:
#endif
#endif
	
	cmpl			$2, %r11d
	jl				0f // ret

	vbroadcastsd	8(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9

	cmpl			$3, %r11d
	jl				0f // ret

	vbroadcastsd	16(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vbroadcastsd	48(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10

	cmpl			$4, %r11d
	jl				0f // ret

	vbroadcastsd	24(%r10), %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vbroadcastsd	56(%r10), %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vbroadcastsd	88(%r10), %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11

0:
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rlt_one_12x4_vs_lib4, .-inner_edge_dtrsm_rlt_one_12x4_vs_lib4
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_12x4_lib4, @function
inner_edge_dtrsm_rut_inv_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_12x4_lib4:
#endif
#endif
	
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7
	vmulpd			%ymm11, %ymm12, %ymm11
	vbroadcastsd	112(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm2
	vfnmadd231pd	%ymm7, %ymm12, %ymm6
	vfnmadd231pd	%ymm11, %ymm12, %ymm10
	vbroadcastsd	104(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm1
	vfnmadd231pd	%ymm7, %ymm12, %ymm5
	vfnmadd231pd	%ymm11, %ymm12, %ymm9
	vbroadcastsd	96(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm0
	vfnmadd231pd	%ymm7, %ymm12, %ymm4
	vfnmadd231pd	%ymm11, %ymm12, %ymm8

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6
	vmulpd			%ymm10, %ymm12, %ymm10
	vbroadcastsd	72(%r10), %ymm12
	vfnmadd231pd	%ymm2, %ymm12, %ymm1
	vfnmadd231pd	%ymm6, %ymm12, %ymm5
	vfnmadd231pd	%ymm10, %ymm12, %ymm9
	vbroadcastsd	64(%r10), %ymm12
	vfnmadd231pd	%ymm2, %ymm12, %ymm0
	vfnmadd231pd	%ymm6, %ymm12, %ymm4
	vfnmadd231pd	%ymm10, %ymm12, %ymm8

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5
	vmulpd			%ymm9, %ymm12, %ymm9
	vbroadcastsd	32(%r10), %ymm12
	vfnmadd231pd	%ymm1, %ymm12, %ymm0
	vfnmadd231pd	%ymm5, %ymm12, %ymm4
	vfnmadd231pd	%ymm9, %ymm12, %ymm8

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4
	vmulpd			%ymm8, %ymm12, %ymm8

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_12x4_lib4, .-inner_edge_dtrsm_rut_inv_12x4_lib4
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
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
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
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d51 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUN_INV_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_run_inv_12x4_lib4, @function
inner_edge_dtrsm_run_inv_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_run_inv_12x4_lib4:
#endif
#endif

	// first column
	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4
	vmulpd			%ymm8, %ymm12, %ymm8

	// second column
	vbroadcastsd	32(%r10), %ymm12
	vfnmadd231pd	%ymm0, %ymm12, %ymm1
	vfnmadd231pd	%ymm4, %ymm12, %ymm5
	vfnmadd231pd	%ymm8, %ymm12, %ymm9
	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5
	vmulpd			%ymm9, %ymm12, %ymm9

	// third column
	vbroadcastsd	64(%r10), %ymm12
	vfnmadd231pd	%ymm0, %ymm12, %ymm2
	vfnmadd231pd	%ymm4, %ymm12, %ymm6
	vfnmadd231pd	%ymm8, %ymm12, %ymm10
	vbroadcastsd	72(%r10), %ymm12
	vfnmadd231pd	%ymm1, %ymm12, %ymm2
	vfnmadd231pd	%ymm5, %ymm12, %ymm6
	vfnmadd231pd	%ymm9, %ymm12, %ymm10
	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6
	vmulpd			%ymm10, %ymm12, %ymm10

	// fourth column
	vbroadcastsd	96(%r10), %ymm12
	vfnmadd231pd	%ymm0, %ymm12, %ymm3
	vfnmadd231pd	%ymm4, %ymm12, %ymm7
	vfnmadd231pd	%ymm8, %ymm12, %ymm11
	vbroadcastsd	104(%r10), %ymm12
	vfnmadd231pd	%ymm1, %ymm12, %ymm3
	vfnmadd231pd	%ymm5, %ymm12, %ymm7
	vfnmadd231pd	%ymm9, %ymm12, %ymm11
	vbroadcastsd	112(%r10), %ymm12
	vfnmadd231pd	%ymm2, %ymm12, %ymm3
	vfnmadd231pd	%ymm6, %ymm12, %ymm7
	vfnmadd231pd	%ymm10, %ymm12, %ymm11
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7
	vmulpd			%ymm11, %ymm12, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_run_inv_12x4_lib4, .-inner_edge_dtrsm_run_inv_12x4_lib4
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
// ymm4 <- [d40 d51 d62 d73]
// ymm5 <- [d41 d50 d63 d72]
// ymm6 <- [d43 d52 d61 d70]
// ymm7 <- [d42 d53 d60 d71]
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
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
// ymm8  <- [d80 d91 da2 db3]
// ymm9  <- [d81 d90 da3 db2]
// ymm10 <- [d83 d92 da1 db0]
// ymm11 <- [d82 d93 d60 db1]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_RUT_INV_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_rut_inv_12x4_vs_lib4, @function
inner_edge_dtrsm_rut_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_rut_inv_12x4_vs_lib4:
#endif
#endif
	
	cmpl			$3, %r12d
	jle				0f

	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm3, %ymm12, %ymm3
	vmulpd			%ymm7, %ymm12, %ymm7
	vmulpd			%ymm11, %ymm12, %ymm11
	vbroadcastsd	112(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm2
	vfnmadd231pd	%ymm7, %ymm12, %ymm6
	vfnmadd231pd	%ymm11, %ymm12, %ymm10
	vbroadcastsd	104(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm1
	vfnmadd231pd	%ymm7, %ymm12, %ymm5
	vfnmadd231pd	%ymm11, %ymm12, %ymm9
	vbroadcastsd	96(%r10), %ymm12
	vfnmadd231pd	%ymm3, %ymm12, %ymm0
	vfnmadd231pd	%ymm7, %ymm12, %ymm4
	vfnmadd231pd	%ymm11, %ymm12, %ymm8

0:
	cmpl			$2, %r12d
	jle				1f

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm2, %ymm12, %ymm2
	vmulpd			%ymm6, %ymm12, %ymm6
	vmulpd			%ymm10, %ymm12, %ymm10
	vbroadcastsd	72(%r10), %ymm12
	vfnmadd231pd	%ymm2, %ymm12, %ymm1
	vfnmadd231pd	%ymm6, %ymm12, %ymm5
	vfnmadd231pd	%ymm10, %ymm12, %ymm9
	vbroadcastsd	64(%r10), %ymm12
	vfnmadd231pd	%ymm2, %ymm12, %ymm0
	vfnmadd231pd	%ymm6, %ymm12, %ymm4
	vfnmadd231pd	%ymm10, %ymm12, %ymm8

1:
	cmpl			$1, %r12d
	jle				2f

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm1, %ymm12, %ymm1
	vmulpd			%ymm5, %ymm12, %ymm5
	vmulpd			%ymm9, %ymm12, %ymm9
	vbroadcastsd	32(%r10), %ymm12
	vfnmadd231pd	%ymm1, %ymm12, %ymm0
	vfnmadd231pd	%ymm5, %ymm12, %ymm4
	vfnmadd231pd	%ymm9, %ymm12, %ymm8

2:

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm0
	vmulpd			%ymm4, %ymm12, %ymm4
	vmulpd			%ymm8, %ymm12, %ymm8

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_rut_inv_12x4_vs_lib4, .-inner_edge_dtrsm_rut_inv_12x4_vs_lib4
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
// r10   <- E0
// r11   <- 4*sde*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- E0
// r11   <- 4*sde*sizeof(double)
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LLN_ONE_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lln_one_12x4_lib4, @function
inner_edge_dtrsm_lln_one_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lln_one_12x4_lib4:
#endif
#endif

	movq	%r10, %r12 // E1 <- E0
	addq	%r11, %r12 // E1 <- E0 + 4*sde*sizeof(double)
	movq	%r12, %r13 // E2 <- E1
	addq	%r11, %r13 // E2 <- E1 + 4*sde*sizeof(double)

	// left block-column

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			0(%r10), %ymm12
	vblendpd		$0x1, %ymm15, %ymm12, %ymm12
	vmovapd			0(%r12), %ymm14
	vmovapd			0(%r13), %ymm15
	vpermpd			$0x00, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x00, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x00, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x00, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			32(%r10), %ymm12
	vblendpd		$0x3, %ymm15, %ymm12, %ymm12
	vmovapd			32(%r12), %ymm14
	vmovapd			32(%r13), %ymm15
	vpermpd			$0x55, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x55, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x55, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x55, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			64(%r10), %ymm12
	vblendpd		$0x7, %ymm15, %ymm12, %ymm12
	vmovapd			64(%r12), %ymm14
	vmovapd			64(%r13), %ymm15
	vpermpd			$0xaa, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vmovapd			96(%r12), %ymm14
	vmovapd			96(%r13), %ymm15
	vpermpd			$0xff, %ymm0, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xff, %ymm1, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xff, %ymm2, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xff, %ymm3, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	addq		$128, %r12
	addq		$128, %r13


	// middle block-column

	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			0(%r12), %ymm12
	vblendpd		$0x1, %ymm14, %ymm12, %ymm12
	vmovapd			0(%r13), %ymm14
	vpermpd			$0x00, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0x00, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0x00, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0x00, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			32(%r12), %ymm12
	vblendpd		$0x3, %ymm14, %ymm12, %ymm12
	vmovapd			32(%r13), %ymm14
	vpermpd			$0x55, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0x55, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0x55, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0x55, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			64(%r12), %ymm12
	vblendpd		$0x7, %ymm14, %ymm12, %ymm12
	vmovapd			64(%r13), %ymm14
	vpermpd			$0xaa, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vmovapd			96(%r13), %ymm14
	vpermpd			$0xff, %ymm4, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0xff, %ymm5, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0xff, %ymm6, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0xff, %ymm7, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm11


	addq		$128, %r13


	// right block-column

	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			0(%r13), %ymm12
	vblendpd		$0x1, %ymm14, %ymm12, %ymm12
	vpermpd			$0x00, %ymm8, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm8
	vpermpd			$0x00, %ymm9, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm9
	vpermpd			$0x00, %ymm10, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm10
	vpermpd			$0x00, %ymm11, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm11

	vmovapd			32(%r13), %ymm12
	vblendpd		$0x3, %ymm14, %ymm12, %ymm12
	vpermpd			$0x55, %ymm8, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm8
	vpermpd			$0x55, %ymm9, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm9
	vpermpd			$0x55, %ymm10, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm10
	vpermpd			$0x55, %ymm11, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm11

	vmovapd			64(%r13), %ymm12
	vblendpd		$0x7, %ymm14, %ymm12, %ymm12
	vpermpd			$0xaa, %ymm8, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm9, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm10, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm11, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm11



#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsm_lln_one_12x4_lib4, .-inner_edge_dtrsm_lln_one_12x4_lib4
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
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
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
//
// output arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_12x4_lib4, @function
inner_edge_dtrsm_lun_inv_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_12x4_lib4:
#endif
#endif
	
	movq	%r10, %r13 // E1 <- E0
	addq	%r11, %r13 // E1 <- E0 + 4*sde*sizeof(double)
	movq	%r13, %r14 // E2 <- E1
	addq	%r11, %r14 // E2 <- E1 + 4*sde*sizeof(double)

	// bottom-right

	vmovapd			352(%r14), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	88(%r12), %ymm12
	vmovapd			352(%r13), %ymm15
//	vmovapd			352(%r10), %ymm11

	vpermpd			$0xff, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	352(%r10), %ymm14, %ymm0

	vpermpd			$0xff, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	352(%r10), %ymm14, %ymm1

	vpermpd			$0xff, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	352(%r10), %ymm14, %ymm2

	vpermpd			$0xff, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	352(%r10), %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			320(%r14), %xmm13
	vbroadcastsd	80(%r12), %ymm12
	vmovapd			320(%r13), %ymm15
//	vmovapd			320(%r10), %ymm11

	vpermpd			$0xaa, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	320(%r10), %ymm14, %ymm0

	vpermpd			$0xaa, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	320(%r10), %ymm14, %ymm1

	vpermpd			$0xaa, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	320(%r10), %ymm14, %ymm2

	vpermpd			$0xaa, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	320(%r10), %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			288(%r14), %xmm13
	vbroadcastsd	72(%r12), %ymm12
	vmovapd			288(%r13), %ymm15
//	vmovapd			288(%r10), %ymm11

	vpermpd			$0x55, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	288(%r10), %ymm14, %ymm0

	vpermpd			$0x55, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	288(%r10), %ymm14, %ymm1

	vpermpd			$0x55, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	288(%r10), %ymm14, %ymm2

	vpermpd			$0x55, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	288(%r10), %ymm14, %ymm3


	vbroadcastsd	64(%r12), %ymm12
	vmovapd			256(%r13), %ymm15
//	vmovapd			256(%r10), %ymm11

	vpermpd			$0x00, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	256(%r10), %ymm14, %ymm0

	vpermpd			$0x00, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	256(%r10), %ymm14, %ymm1

	vpermpd			$0x00, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	256(%r10), %ymm14, %ymm2

	vpermpd			$0x00, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	256(%r10), %ymm14, %ymm3


	// middle-middle

	vmovapd			224(%r13), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	56(%r12), %ymm12
	vmovapd			224(%r10), %ymm15

	vpermpd			$0xff, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0xff, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0xff, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0xff, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			192(%r13), %xmm13
	vbroadcastsd	48(%r12), %ymm12
	vmovapd			192(%r10), %ymm15

	vpermpd			$0xaa, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0xaa, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0xaa, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0xaa, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			160(%r13), %xmm13
	vbroadcastsd	40(%r12), %ymm12
	vmovapd			160(%r10), %ymm15

	vpermpd			$0x55, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0x55, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0x55, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0x55, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vbroadcastsd	32(%r12), %ymm12
	vmovapd			128(%r10), %ymm15

	vpermpd			$0x00, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0x00, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0x00, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0x00, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	// top-left

	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r12), %ymm12

	vpermpd			$0xff, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermpd			$0xff, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermpd			$0xff, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermpd			$0xff, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r12), %ymm12

	vpermpd			$0xaa, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermpd			$0xaa, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermpd			$0xaa, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermpd			$0xaa, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r12), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vbroadcastsd	0(%r12), %ymm12

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
	.size	inner_edge_dtrsm_lun_inv_12x4_lib4, .-inner_edge_dtrsm_lun_inv_12x4_lib4
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
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// r13  <- km
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
//
// output arguments:
// r10  <- E
// r11  <- 4*sde*sizeof(double)
// r12  <- inv_diag_E
// r13  <- km
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSM_LUN_INV_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsm_lun_inv_12x4_vs_lib4, @function
inner_edge_dtrsm_lun_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsm_lun_inv_12x4_vs_lib4:
#endif
#endif
	
	movq	%r10, %r15 // E1 <- E0
	addq	%r11, %r15 // E1 <- E0 + 4*sde*sizeof(double)
	movq	%r15, %r14 // E2 <- E1
	addq	%r11, %r14 // E2 <- E1 + 4*sde*sizeof(double)

	// bottom-right

	cmpl	$11, %r13d
	jle		0f

	vmovapd			352(%r14), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	88(%r12), %ymm12
	vmovapd			352(%r15), %ymm15
//	vmovapd			352(%r10), %ymm11

	vpermpd			$0xff, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	352(%r10), %ymm14, %ymm0

	vpermpd			$0xff, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	352(%r10), %ymm14, %ymm1

	vpermpd			$0xff, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	352(%r10), %ymm14, %ymm2

	vpermpd			$0xff, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	352(%r10), %ymm14, %ymm3

0:
	cmpl	$10, %r13d
	jle		1f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			320(%r14), %xmm13
	vbroadcastsd	80(%r12), %ymm12
	vmovapd			320(%r15), %ymm15
//	vmovapd			320(%r10), %ymm11

	vpermpd			$0xaa, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	320(%r10), %ymm14, %ymm0

	vpermpd			$0xaa, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	320(%r10), %ymm14, %ymm1

	vpermpd			$0xaa, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	320(%r10), %ymm14, %ymm2

	vpermpd			$0xaa, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	320(%r10), %ymm14, %ymm3

1:
	cmpl	$9, %r13d
	jle		2f

	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			288(%r14), %xmm13
	vbroadcastsd	72(%r12), %ymm12
	vmovapd			288(%r15), %ymm15
//	vmovapd			288(%r10), %ymm11

	vpermpd			$0x55, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm13, %ymm14, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	288(%r10), %ymm14, %ymm0

	vpermpd			$0x55, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm13, %ymm14, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	288(%r10), %ymm14, %ymm1

	vpermpd			$0x55, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm13, %ymm14, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	288(%r10), %ymm14, %ymm2

	vpermpd			$0x55, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	288(%r10), %ymm14, %ymm3

2:

	vbroadcastsd	64(%r12), %ymm12
	vmovapd			256(%r15), %ymm15
//	vmovapd			256(%r10), %ymm11

	vpermpd			$0x00, %ymm8, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm8, %ymm8
	vfnmadd231pd	%ymm15, %ymm14, %ymm4
	vfnmadd231pd	256(%r10), %ymm14, %ymm0

	vpermpd			$0x00, %ymm9, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm9, %ymm9
	vfnmadd231pd	%ymm15, %ymm14, %ymm5
	vfnmadd231pd	256(%r10), %ymm14, %ymm1

	vpermpd			$0x00, %ymm10, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm10, %ymm10
	vfnmadd231pd	%ymm15, %ymm14, %ymm6
	vfnmadd231pd	256(%r10), %ymm14, %ymm2

	vpermpd			$0x00, %ymm11, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm11, %ymm11
	vfnmadd231pd	%ymm15, %ymm14, %ymm7
	vfnmadd231pd	256(%r10), %ymm14, %ymm3


	// middle-middle

	vmovapd			224(%r15), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	56(%r12), %ymm12
	vmovapd			224(%r10), %ymm15

	vpermpd			$0xff, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0xff, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0xff, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0xff, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0 // ?????????????
	vmovapd			192(%r15), %xmm13
	vbroadcastsd	48(%r12), %ymm12
	vmovapd			192(%r10), %ymm15

	vpermpd			$0xaa, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0xaa, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0xaa, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0xaa, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			160(%r15), %xmm13
	vbroadcastsd	40(%r12), %ymm12
	vmovapd			160(%r10), %ymm15

	vpermpd			$0x55, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0x55, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0x55, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm13, %ymm14, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0x55, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm13, %ymm14, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	vbroadcastsd	32(%r12), %ymm12
	vmovapd			128(%r10), %ymm15

	vpermpd			$0x00, %ymm4, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm4, %ymm4
	vfnmadd231pd	%ymm15, %ymm14, %ymm0

	vpermpd			$0x00, %ymm5, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm5, %ymm5
	vfnmadd231pd	%ymm15, %ymm14, %ymm1

	vpermpd			$0x00, %ymm6, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm6, %ymm6
	vfnmadd231pd	%ymm15, %ymm14, %ymm2

	vpermpd			$0x00, %ymm7, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x1, %ymm14, %ymm7, %ymm7
	vfnmadd231pd	%ymm15, %ymm14, %ymm3


	// top-left

	vmovapd			96(%r10), %ymm13
	vxorpd			%ymm14, %ymm14, %ymm14 // 0.0
	vblendpd		$0x7, %ymm13, %ymm14, %ymm13
	vbroadcastsd	24(%r12), %ymm12

	vpermpd			$0xff, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermpd			$0xff, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermpd			$0xff, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermpd			$0xff, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x8, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovapd			64(%r10), %xmm13
	vbroadcastsd	16(%r12), %ymm12

	vpermpd			$0xaa, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermpd			$0xaa, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermpd			$0xaa, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermpd			$0xaa, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x4, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vxorpd			%ymm13, %ymm13, %ymm13 // 0.0
	vmovsd			32(%r10), %xmm13
	vbroadcastsd	8(%r12), %ymm12

	vpermilpd		$0xf, %ymm0, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm0, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm0

	vpermilpd		$0xf, %ymm1, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm1, %ymm1
	vfnmadd231pd	%ymm13, %ymm14, %ymm1

	vpermilpd		$0xf, %ymm2, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm2, %ymm2
	vfnmadd231pd	%ymm13, %ymm14, %ymm2

	vpermilpd		$0xf, %ymm3, %ymm14
	vmulpd			%ymm14, %ymm12, %ymm14
	vblendpd		$0x2, %ymm14, %ymm3, %ymm3
	vfnmadd231pd	%ymm13, %ymm14, %ymm3


	vbroadcastsd	0(%r12), %ymm12

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
	.size	inner_edge_dtrsm_lun_inv_12x4_vs_lib4, .-inner_edge_dtrsm_lun_inv_12x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// LU factorization without pivoting
// left kernel
//
// input arguments:
// r10  <- inv_diag_E
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
//
// output arguments:
// r10  <- inv_diag_E
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGETRF_L_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgetrf_l_12x4_lib4, @function
inner_edge_dgetrf_l_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgetrf_l_12x4_lib4:
#endif
#endif
	
#if defined(OS_LINUX)
	vmovsd			.LC05(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC05(%rip), %xmm14 // 1.0
#endif
//	vmovddup		%xmm14, %xmm14

	// first column
//	vblendpd		$0x1, %ymm0, %ymm12, %ymm12
	vmovapd			%ymm0, %ymm12
	vdivsd			%xmm0, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 0(%r10)
	vmulpd			%ymm0, %ymm13, %ymm0
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8
	vblendpd		$0x1, %ymm12, %ymm0, %ymm0

	// second column
	vpermpd			$0x00, %ymm1, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm1
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vblendpd		$0x2, %ymm1, %ymm13, %ymm12

	vpermilpd		$0x3, %xmm1, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 8(%r10)
	vmulpd			%ymm1, %ymm13, %ymm1
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9
	vblendpd		$0x3, %ymm12, %ymm1, %ymm1

	// third column
	vpermpd			$0x00, %ymm2, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vblendpd		$0x2, %ymm2, %ymm13, %ymm12

	vpermpd			$0x55, %ymm2, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vblendpd		$0x4, %ymm2, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm2, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 16(%r10)
	vmulpd			%ymm2, %ymm13, %ymm2
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10
	vblendpd		$0x7, %ymm12, %ymm2, %ymm2

	// fourth column
	vpermpd			$0x00, %ymm3, %ymm13
	vfnmadd231pd	%ymm0, %ymm13, %ymm3
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vblendpd		$0x2, %ymm3, %ymm13, %ymm12

	vpermpd			$0x55, %ymm3, %ymm13
	vfnmadd231pd	%ymm1, %ymm13, %ymm3
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vblendpd		$0x4, %ymm3, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm3, %ymm13
	vfnmadd231pd	%ymm2, %ymm13, %ymm3
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vblendpd		$0x8, %ymm3, %ymm12, %ymm12
	
	vpermpd			$0xff, %ymm3, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 24(%r10)
//	vmulpd			%ymm3, %ymm13, %ymm3
	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11
	vblendpd		$0x7, %ymm12, %ymm3, %ymm3

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgetrf_l_12x4_lib4, .-inner_edge_dgetrf_l_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// LU factorization without pivoting
// middle kernel
//
// input arguments:
// r10  <- E
// r11  <- sde
// r12  <- inv_diag_D
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
//
// output arguments:
// r10  <- E
// r11  <- sde
// r12  <- inv_diag_D
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGETRF_M_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgetrf_m_12x4_lib4, @function
inner_edge_dgetrf_m_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgetrf_m_12x4_lib4:
#endif
#endif
	
	movq	%r10, %r14 // E1 <- E0
	addq	%r11, %r14 // E1 <- E0 + 4*sde*sizeof(double)
	movq	%r14, %r13 // E2 <- E1
	addq	%r11, %r13 // E2 <- E1 + 4*sde*sizeof(double)

	// solve upper 4x4 & correct lower 8x4

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			0(%r10), %ymm12
	vblendpd		$0x1, %ymm15, %ymm12, %ymm12
	vmovapd			0(%r14), %ymm14
	vmovapd			0(%r13), %ymm15
	vpermpd			$0x00, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x00, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x00, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x00, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			32(%r10), %ymm12
	vblendpd		$0x3, %ymm15, %ymm12, %ymm12
	vmovapd			32(%r14), %ymm14
	vmovapd			32(%r13), %ymm15
	vpermpd			$0x55, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x55, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x55, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x55, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			64(%r10), %ymm12
	vblendpd		$0x7, %ymm15, %ymm12, %ymm12
	vmovapd			64(%r14), %ymm14
	vmovapd			64(%r13), %ymm15
	vpermpd			$0xaa, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vmovapd			96(%r14), %ymm14
	vmovapd			96(%r13), %ymm15
	vpermpd			$0xff, %ymm0, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xff, %ymm1, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xff, %ymm2, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xff, %ymm3, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11


	// factorize lower 8x4

#if defined(OS_LINUX)
	vmovsd			.LC05(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC05(%rip), %xmm14 // 1.0
#endif
//	vmovddup		%xmm14, %xmm14

	// first column
//	vblendpd		$0x1, %ymm4, %ymm12, %ymm12
	vmovapd			%ymm4, %ymm12
	vdivsd			%xmm4, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 0(%r12)
	vmulpd			%ymm4, %ymm13, %ymm4
	vmulpd			%ymm8, %ymm13, %ymm8
	vblendpd		$0x1, %ymm12, %ymm4, %ymm4

	// second column
	vpermpd			$0x00, %ymm5, %ymm13
	vfnmadd231pd	%ymm4, %ymm13, %ymm5
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vblendpd		$0x2, %ymm5, %ymm13, %ymm12

	vpermilpd		$0x3, %xmm5, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 8(%r12)
	vmulpd			%ymm5, %ymm13, %ymm5
	vmulpd			%ymm9, %ymm13, %ymm9
	vblendpd		$0x3, %ymm12, %ymm5, %ymm5

	// third column
	vpermpd			$0x00, %ymm6, %ymm13
	vfnmadd231pd	%ymm4, %ymm13, %ymm6
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vblendpd		$0x2, %ymm6, %ymm13, %ymm12

	vpermpd			$0x55, %ymm6, %ymm13
	vfnmadd231pd	%ymm5, %ymm13, %ymm6
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vblendpd		$0x4, %ymm6, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm6, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 16(%r12)
	vmulpd			%ymm6, %ymm13, %ymm6
	vmulpd			%ymm10, %ymm13, %ymm10
	vblendpd		$0x7, %ymm12, %ymm6, %ymm6

	// fourth column
	vpermpd			$0x00, %ymm7, %ymm13
	vfnmadd231pd	%ymm4, %ymm13, %ymm7
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vblendpd		$0x2, %ymm7, %ymm13, %ymm12

	vpermpd			$0x55, %ymm7, %ymm13
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vblendpd		$0x4, %ymm7, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm7, %ymm13
	vfnmadd231pd	%ymm6, %ymm13, %ymm7
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vblendpd		$0x8, %ymm7, %ymm12, %ymm12
	
	vpermpd			$0xff, %ymm7, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 24(%r12)
//	vmulpd			%ymm7, %ymm13, %ymm7
	vmulpd			%ymm11, %ymm13, %ymm11
	vblendpd		$0x7, %ymm12, %ymm7, %ymm7

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgetrf_m_12x4_lib4, .-inner_edge_dgetrf_m_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// LU factorization without pivoting
// right kernel
//
// input arguments:
// r10  <- E
// r11  <- sde
// r12  <- inv_diag_D
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
//
// output arguments:
// r10  <- E
// r11  <- sde
// r12  <- inv_diag_D
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

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DGETRF_R_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dgetrf_r_12x4_lib4, @function
inner_edge_dgetrf_r_12x4_lib4:
#elif defined(OS_MAC)
_inner_edge_dgetrf_r_12x4_lib4:
#endif
#endif
	
	movq	%r10, %r14 // E1 <- E0
	addq	%r11, %r14 // E1 <- E0 + 4*sde*sizeof(double)
	movq	%r14, %r13 // E2 <- E1
	addq	%r11, %r13 // E2 <- E1 + 4*sde*sizeof(double)

	// solve upper 8x4 & correct lower 4x4

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			0(%r10), %ymm12
	vblendpd		$0x1, %ymm15, %ymm12, %ymm12
	vmovapd			0(%r14), %ymm14
	vmovapd			0(%r13), %ymm15
	vpermpd			$0x00, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x00, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x00, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x00, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			32(%r10), %ymm12
	vblendpd		$0x3, %ymm15, %ymm12, %ymm12
	vmovapd			32(%r14), %ymm14
	vmovapd			32(%r13), %ymm15
	vpermpd			$0x55, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0x55, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0x55, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0x55, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vxorpd			%ymm15, %ymm15, %ymm15
	vmovapd			64(%r10), %ymm12
	vblendpd		$0x7, %ymm15, %ymm12, %ymm12
	vmovapd			64(%r14), %ymm14
	vmovapd			64(%r13), %ymm15
	vpermpd			$0xaa, %ymm0, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm0
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm1, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm1
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm2, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm2
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm3, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm3
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	vmovapd			96(%r14), %ymm14
	vmovapd			96(%r13), %ymm15
	vpermpd			$0xff, %ymm0, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm4
	vfnmadd231pd	%ymm15, %ymm13, %ymm8
	vpermpd			$0xff, %ymm1, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm5
	vfnmadd231pd	%ymm15, %ymm13, %ymm9
	vpermpd			$0xff, %ymm2, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm6
	vfnmadd231pd	%ymm15, %ymm13, %ymm10
	vpermpd			$0xff, %ymm3, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm7
	vfnmadd231pd	%ymm15, %ymm13, %ymm11

	addq		$128, %r14
	addq		$128, %r13


	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			0(%r14), %ymm12
	vblendpd		$0x1, %ymm14, %ymm12, %ymm12
	vmovapd			0(%r13), %ymm14
	vpermpd			$0x00, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0x00, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0x00, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0x00, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			32(%r14), %ymm12
	vblendpd		$0x3, %ymm14, %ymm12, %ymm12
	vmovapd			32(%r13), %ymm14
	vpermpd			$0x55, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0x55, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0x55, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0x55, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vxorpd			%ymm14, %ymm14, %ymm14
	vmovapd			64(%r14), %ymm12
	vblendpd		$0x7, %ymm14, %ymm12, %ymm12
	vmovapd			64(%r13), %ymm14
	vpermpd			$0xaa, %ymm4, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm4
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0xaa, %ymm5, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm5
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0xaa, %ymm6, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm6
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0xaa, %ymm7, %ymm13
	vfnmadd231pd	%ymm12, %ymm13, %ymm7
	vfnmadd231pd	%ymm14, %ymm13, %ymm11

	vmovapd			96(%r13), %ymm14
	vpermpd			$0xff, %ymm4, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm8
	vpermpd			$0xff, %ymm5, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm9
	vpermpd			$0xff, %ymm6, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm10
	vpermpd			$0xff, %ymm7, %ymm13
	vfnmadd231pd	%ymm14, %ymm13, %ymm11



	// factorize lower 8x4

#if defined(OS_LINUX)
	vmovsd			.LC05(%rip), %xmm14 // 1.0
#elif defined(OS_MAC)
	vmovsd			LC05(%rip), %xmm14 // 1.0
#endif
//	vmovddup		%xmm14, %xmm14

	// first column
//	vblendpd		$0x1, %ymm8, %ymm12, %ymm12
	vmovapd			%ymm8, %ymm12
	vdivsd			%xmm8, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 0(%r12)
	vmulpd			%ymm8, %ymm13, %ymm8
	vblendpd		$0x1, %ymm12, %ymm8, %ymm8

	// second column
	vpermpd			$0x00, %ymm9, %ymm13
	vfnmadd231pd	%ymm8, %ymm13, %ymm9
	vblendpd		$0x2, %ymm9, %ymm13, %ymm12

	vpermilpd		$0x3, %xmm9, %xmm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 8(%r12)
	vmulpd			%ymm9, %ymm13, %ymm9
	vblendpd		$0x3, %ymm12, %ymm9, %ymm9

	// third column
	vpermpd			$0x00, %ymm10, %ymm13
	vfnmadd231pd	%ymm8, %ymm13, %ymm10
	vblendpd		$0x2, %ymm10, %ymm13, %ymm12

	vpermpd			$0x55, %ymm10, %ymm13
	vfnmadd231pd	%ymm9, %ymm13, %ymm10
	vblendpd		$0x4, %ymm10, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm10, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 16(%r12)
	vmulpd			%ymm10, %ymm13, %ymm10
	vblendpd		$0x7, %ymm12, %ymm10, %ymm10

	// fourth column
	vpermpd			$0x00, %ymm11, %ymm13
	vfnmadd231pd	%ymm8, %ymm13, %ymm11
	vblendpd		$0x2, %ymm11, %ymm13, %ymm12

	vpermpd			$0x55, %ymm11, %ymm13
	vfnmadd231pd	%ymm9, %ymm13, %ymm11
	vblendpd		$0x4, %ymm11, %ymm12, %ymm12

	vpermpd			$0xaa, %ymm11, %ymm13
	vfnmadd231pd	%ymm10, %ymm13, %ymm11
	vblendpd		$0x8, %ymm11, %ymm12, %ymm12
	
	vpermpd			$0xff, %ymm11, %ymm13
	vdivsd			%xmm13, %xmm14, %xmm13
//	vpermpd			$0x00, %ymm13, %ymm13
	vmovsd			%xmm13, 24(%r12)
//	vmulpd			%ymm11, %ymm13, %ymm11
	vblendpd		$0x7, %ymm12, %ymm11, %ymm11

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dgetrf_r_12x4_lib4, .-inner_edge_dgetrf_r_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
//
// output arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_12x4_lib4, @function
inner_store_12x4_lib4:
#elif defined(OS_MAC)
_inner_store_12x4_lib4:
#endif
#endif
	
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

	vmovapd %ymm8,   0(%r14)
	vmovapd %ymm9,  32(%r14)
	vmovapd %ymm10, 64(%r14)
	vmovapd %ymm11, 96(%r14)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_12x4_lib4, .-inner_store_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r14  <- dirty
// r15  <- dirty
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]
// ymm4 <- [d40 d50 d60 d70]
// ymm5 <- [d41 d50 d61 d71]
// ymm6 <- [d42 d52 d62 d72]
// ymm7 <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
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
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d91 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_12x4_vs_lib4, @function
inner_store_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_12x4_vs_lib4:
#endif
#endif
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	movq	%r15, %r14 // D2 <- D1
	addq	%r11, %r14 // D2 <- D1 + 4*sdd*sizeof(double)

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovupd		.LC04(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC04(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmovapd		%ymm4, 0(%r15)
	vmaskmovpd	%ymm8, %ymm15,  0(%r14)
	jl			0f // end
	cmpl		$3, %r13d
	vmovapd		%ymm1, 32(%r10)
	vmovapd		%ymm5, 32(%r15)
	vmaskmovpd	%ymm9, %ymm15, 32(%r14)
	jl			0f // end
	vmovapd		%ymm2, 64(%r10)
	vmovapd		%ymm6, 64(%r15)
	vmaskmovpd	%ymm10, %ymm15, 64(%r14)
	je			0f // end
	vmovapd		%ymm3, 96(%r10)
	vmovapd		%ymm7, 96(%r15)
	vmaskmovpd	%ymm11, %ymm15, 96(%r14)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_12x4_vs_lib4, .-inner_store_12x4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// store lower n
//
// input arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d50 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_12X4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_12x4_lib4, @function
inner_store_l_12x4_lib4:
#elif defined(OS_MAC)
_inner_store_l_12x4_lib4:
#endif
#endif
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	movq	%r15, %r14 // D2 <- D1
	addq	%r11, %r14 // D2 <- D1 + 4*sdd*sizeof(double)

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

	vmovapd		%ymm4, 0(%r15)
	vmovapd		%ymm5, 32(%r15)
	vmovapd		%ymm6, 64(%r15)
	vmovapd		%ymm7, 96(%r15)

	vmovapd		%ymm8, 0(%r14)
	vmovapd		%ymm9, 32(%r14)
	vmovapd		%ymm10, 64(%r14)
	vmovapd		%ymm11, 96(%r14)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_12x4_lib4, .-inner_store_l_12x4_lib4
#endif
#endif





// common inner routine with file scope
//
// store lower n vs
//
// input arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d50 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11   <- 4*sdd*sizeof(double)
// r12d  <- km
// r13d  <- kn
// r14   <- dirty
// r15   <- dirty
// ymm0  <- [d00 d10 d20 d30]
// ymm1  <- [d01 d11 d21 d31]
// ymm2  <- [d02 d12 d22 d32]
// ymm3  <- [d03 d13 d23 d33]
// ymm4  <- [d40 d50 d60 d70]
// ymm5  <- [d41 d51 d61 d71]
// ymm6  <- [d42 d52 d62 d72]
// ymm7  <- [d43 d53 d63 d73]
// ymm8  <- [d80 d90 da0 db0]
// ymm9  <- [d81 d90 da1 db1]
// ymm10 <- [d82 d92 da2 db2]
// ymm11 <- [d83 d93 da3 db3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_L_12X4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_l_12x4_vs_lib4, @function
inner_store_l_12x4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_l_12x4_vs_lib4:
#endif
#endif
	
	movq	%r10, %r15 // D1 <- D0
	addq	%r11, %r15 // D1 <- D0 + 4*sdd*sizeof(double)

	movq	%r15, %r14 // D2 <- D1
	addq	%r11, %r14 // D2 <- D1 + 4*sdd*sizeof(double)

	vcvtsi2sd	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX)
	vmovupd		.LC04(%rip), %ymm14
#elif defined(OS_MAC)
	vmovupd		LC04(%rip), %ymm14
#endif
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15

	cmpl		$2, %r13d
	vmovapd		%ymm0, 0(%r10)
	vmovapd		%ymm4, 0(%r15)
	vmaskmovpd	%ymm8, %ymm15,  0(%r14)
	jl			0f // end
	cmpl		$3, %r13d
	vmovapd		32(%r10), %ymm14
	vblendpd	$0x1, %ymm14, %ymm1, %ymm1	
	vmovapd		%ymm1, 32(%r10)
	vmovapd		%ymm5, 32(%r15)
	vmaskmovpd	%ymm9, %ymm15, 32(%r14)
	jl			0f // end
	vmovapd		64(%r10), %ymm14
	vblendpd	$0x3, %ymm14, %ymm2, %ymm2	
	vmovapd		%ymm2, 64(%r10)
	vmovapd		%ymm6, 64(%r15)
	vmaskmovpd	%ymm10, %ymm15, 64(%r14)
	je			0f // end
	vmovapd		96(%r10), %ymm14
	vblendpd	$0x7, %ymm14, %ymm3, %ymm3	
	vmovapd		%ymm3, 96(%r10)
	vmovapd		%ymm7, 96(%r15)
	vmaskmovpd	%ymm11, %ymm15, 96(%r14)

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_l_12x4_vs_lib4, .-inner_store_l_12x4_vs_lib4
#endif
#endif





//                                   rdi    rsi            rdx        rcx      r8         r9         rsp+8
// 12oid kernel_dgemm_nt_12x4_a0_lib4(int k, double *alpha, double *A, int sda, double *B, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_a0_lib4
	.type kernel_dgemm_nt_12x4_a0_lib4, @function
kernel_dgemm_nt_12x4_a0_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_a0_lib4
_kernel_dgemm_nt_12x4_a0_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_A0_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_a0_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_a0_12x4_lib4
#endif
#endif


	// store n

	movq	%r9, %r10 // D
	movl	STACKSIZE + 8(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgemm_nt_12x4_a0_lib4, .-kernel_dgemm_nt_12x4_a0_lib4
#endif





//                                rdi    rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dgemm_nt_12x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_lib4
	.type kernel_dgemm_nt_12x4_lib4, @function
kernel_dgemm_nt_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_lib4
_kernel_dgemm_nt_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgemm_nt_12x4_lib4, .-kernel_dgemm_nt_12x4_lib4
#endif





//                                   rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32   rsp+40  rsp+48
// void kernel_dgemm_nt_12x4_vs_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nt_12x4_vs_lib4
	.type kernel_dgemm_nt_12x4_vs_lib4, @function
kernel_dgemm_nt_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nt_12x4_vs_lib4
_kernel_dgemm_nt_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgemm_nt_12x4_vs_lib4, .-kernel_dgemm_nt_12x4_vs_lib4
#endif





//                                rdi    rsi            rdx        rcx      r8         r9       rsp+8         rsp+16     rsp+24   rsp+32     rsp_40
// void kernel_dgemm_nn_12x4_lib4(int k, double *alpha, double *A, int sda, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_12x4_lib4
	.type kernel_dgemm_nn_12x4_lib4, @function
kernel_dgemm_nn_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_12x4_lib4
_kernel_dgemm_nn_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B
	movl	%r9d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_12x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	STACKSIZE +  8(%rsp), %r11 // beta
	movq	STACKSIZE +  16(%rsp), %r12 // C
	movl	STACKSIZE +  24(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 32(%rsp), %r10 // D
	movl	STACKSIZE + 40(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgemm_nn_12x4_lib4, .-kernel_dgemm_nn_12x4_lib4
#endif





//                                   rdi     rsi            rdx        rcx      r8         r9       rsp+8         rsp+16     rsp+24   rsp+32     rsp+40   rsp+48  rsp+56
// void kernel_dgemm_nn_12x4_vs_lib4(int km, double *alpha, double *A, int sda, double *B, int sdb, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_nn_12x4_vs_lib4
	.type kernel_dgemm_nn_12x4_vs_lib4, @function
kernel_dgemm_nn_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_nn_12x4_vs_lib4
_kernel_dgemm_nn_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B
	movl	%r9d, %r14d // sda
	sall	$5, %r14d // 4*sda*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_12x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	STACKSIZE +  8(%rsp), %r11 // beta
	movq	STACKSIZE +  16(%rsp), %r12 // C
	movl	STACKSIZE +  24(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 32(%rsp), %r10 // store address D
	movl	STACKSIZE + 40(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  48(%rsp), %r12d // km 
	movl	STACKSIZE +  56(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgemm_nn_12x4_vs_lib4, .-kernel_dgemm_nn_12x4_vs_lib4
#endif





//                                  rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dsyrk_nt_l_12x4_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_12x4_lib4
	.type kernel_dsyrk_nt_l_12x4_lib4, @function
kernel_dsyrk_nt_l_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_12x4_lib4
_kernel_dsyrk_nt_l_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_lib4
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
	.size	kernel_dsyrk_nt_l_12x4_lib4, .-kernel_dsyrk_nt_l_12x4_lib4
#endif





//                                     rdi     rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32   rsp+40  rsp+48
// void kernel_dsyrk_nt_l_12x4_vs_lib4(int km, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_nt_l_12x4_vs_lib4
	.type kernel_dsyrk_nt_l_12x4_vs_lib4, @function
kernel_dsyrk_nt_l_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_nt_l_12x4_vs_lib4
_kernel_dsyrk_nt_l_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner blend scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_vs_lib4
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
	.size	kernel_dsyrk_nt_l_12x4_vs_lib4, .-kernel_dsyrk_nt_l_12x4_vs_lib4
#endif





//                                   rdi    rsi            rdx        rcx      r8           r9         rsp+8    rsp+16     rsp+24
// void kernel_dtrmm_nn_rl_12x4_lib4(int k, double *alpha, double *A, int sda, int offsetB, double *B, int sdb, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nn_rl_12x4_lib4
	.type kernel_dtrmm_nn_rl_12x4_lib4, @function
kernel_dtrmm_nn_rl_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nn_rl_12x4_lib4
_kernel_dtrmm_nn_rl_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// initial triangle

	movl	%edi, %r10d // k
	movq	%rdx, %r11 // A
	movl	%ecx, %r12d // sdb
	sall	$5, %r12d // 4*sdb*sizeof(double)
	movq	%r9, %r13 // B
	movl	STACKSIZE +  8(%rsp), %r14d // sdb
	sall	$5, %r14d // 4*sdb*sizeof(double)
	movl	%r8d, %r15d // offsetB

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NN_RL_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nn_rl_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nn_rl_12x4_lib4
#endif
#endif

	// call inner dgemm kernel nt after initial triangle

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nn_12x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha

#if MACRO_LEVEL>=1
	INNER_SCALE_A0_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_a0_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_a0_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrmm_nn_rl_12x4_lib4, .-kernel_dtrmm_nn_rl_12x4_lib4
#endif





//                                   rdi    rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dtrmm_nt_ru_12x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_12x4_lib4
	.type kernel_dtrmm_nt_ru_12x4_lib4, @function
kernel_dtrmm_nt_ru_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_12x4_lib4
_kernel_dtrmm_nt_ru_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d //k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B
	addq	$128, %r13 // B+4*bs


#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_12x4_lib4
#endif
#endif


	// initial triangle

	movq	%rdx, %r10 // A
	movl	%ecx, %r11d // sda
	sall	$5, %r11d // 4*sda*sizeof(double)
//	movslq	%r11d, %r11
	movq	%r8, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_12x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrmm_nt_ru_12x4_lib4, .-kernel_dtrmm_nt_ru_12x4_lib4
#endif





//                                   rdi    rsi            rdx        rcx      r8         r9            rsp+8      rsp+16   rsp+24     rsp+32
// void kernel_dtrmm_nt_ru_12x4_lib4(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int sdc, double *D, int sdd);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmm_nt_ru_12x4_vs_lib4
	.type kernel_dtrmm_nt_ru_12x4_vs_lib4, @function
kernel_dtrmm_nt_ru_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmm_nt_ru_12x4_vs_lib4
_kernel_dtrmm_nt_ru_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt after initial triangle

	movl	%edi, %r10d // k
	subl	$4, %r10d //k-4
	movq	%rdx, %r11 // A
	addq	$128, %r11 // A+4*bs
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%r8, %r13 // B
	addq	$128, %r13 // B+4*bs


#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


// call inner blend

#if MACRO_LEVEL>=1
	INNER_BLEND_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_12x4_lib4
#endif
#endif


	// initial triangle

	movq	%rdx, %r10 // A
	movl	%ecx, %r11d // sda
	sall	$5, %r11d // 4*sda*sizeof(double)
//	movslq	%r11d, %r11
	movq	%r8, %r12 // B

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMM_NT_RU_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrmm_nt_ru_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrmm_nt_ru_12x4_lib4
#endif
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11 // beta
	movq	STACKSIZE +  8(%rsp), %r12 // C
	movl	STACKSIZE +  16(%rsp), %r13d // sdc
	sall	$5, %r13d // 4*sdc*sizeof(double)
//	movslq	%r13d, %r13

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_ab_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_ab_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 24(%rsp), %r10 // D
	movl	STACKSIZE + 32(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrmm_nt_ru_12x4_vs_lib4, .-kernel_dtrmm_nt_ru_12x4_vs_lib4
#endif





//                                   rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24
// void kernel_dpotrf_nt_l_12x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_12x4_lib4
	.type kernel_dpotrf_nt_l_12x4_lib4, @function
kernel_dpotrf_nt_l_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_12x4_lib4
_kernel_dpotrf_nt_l_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10 // C
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 24(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_lib4
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
	.size	kernel_dpotrf_nt_l_12x4_lib4, .-kernel_dpotrf_nt_l_12x4_lib4
#endif





//                                      rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24              rsp+32  rsp+40 
// void kernel_dpotrf_nt_l_12x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dpotrf_nt_l_12x4_vs_lib4
	.type kernel_dpotrf_nt_l_12x4_vs_lib4, @function
kernel_dpotrf_nt_l_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dpotrf_nt_l_12x4_vs_lib4
_kernel_dpotrf_nt_l_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10 // C
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 24(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 40(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_vs_lib4
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
	.size	kernel_dpotrf_nt_l_12x4_vs_lib4, .-kernel_dpotrf_nt_l_12x4_vs_lib4
#endif





//                                         rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56
// void kernel_dsyrk_dpotrf_nt_l_12x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_12x4_lib4
	.type kernel_dsyrk_dpotrf_nt_l_12x4_lib4, @function
kernel_dsyrk_dpotrf_nt_l_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_12x4_lib4
_kernel_dsyrk_dpotrf_nt_l_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sdam*sizeof(double)
//	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	STACKSIZE + 24(%rsp), %r10 // C
	movl	STACKSIZE + 32(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11 

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 56(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 40(%rsp), %r10 // D
	movl	STACKSIZE + 48(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_lib4
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
	.size	kernel_dsyrk_dpotrf_nt_l_12x4_lib4, .-kernel_dsyrk_dpotrf_nt_l_12x4_lib4
#endif





//                                            rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56              rsp+64  rsp+72
// void kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4
	.type kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4
_kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sdam*sizeof(double)
//	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	STACKSIZE + 24(%rsp), %r10 // C
	movl	STACKSIZE + 32(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11 

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 56(%rsp), %r10  // inv_diag_D 
	movl	STACKSIZE + 72(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DPOTRF_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dpotrf_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dpotrf_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 40(%rsp), %r10 // D
	movl	STACKSIZE + 48(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  64(%rsp), %r12d // km 
	movl	STACKSIZE +  72(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_L_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_l_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_l_12x4_vs_lib4
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
	.size	kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_l_12x4_vs_lib4
#endif





//                                          rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32              rsp+40  rsp+48 
// void kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_12x4_vs_lib4
	.type kernel_dtrsm_nt_rl_inv_12x4_vs_lib4, @function
kernel_dtrsm_nt_rl_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_12x4_vs_lib4
_kernel_dtrsm_nt_rl_inv_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10 // C
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nt_rl_inv_12x4_vs_lib4, .-kernel_dtrsm_nt_rl_inv_12x4_vs_lib4
#endif





//                                                rdi     rsi         rdx       rcx          r8     r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56     rsp+64              rsp+72  rsp+80
// void kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	STACKSIZE + 24(%rsp), %r10  // C
	movl	STACKSIZE + 32(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 56(%rsp), %r10  // E 
	movq	STACKSIZE + 64(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 80(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 40(%rsp), %r10 // D
	movl	STACKSIZE + 48(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  72(%rsp), %r12d // km 
	movl	STACKSIZE +  80(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_12x4_vs_lib4
#endif





//                                       rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32  
// void kernel_dtrsm_nt_rl_inv_12x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_inv_12x4_lib4
	.type kernel_dtrsm_nt_rl_inv_12x4_lib4, @function
kernel_dtrsm_nt_rl_inv_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_inv_12x4_lib4
_kernel_dtrsm_nt_rl_inv_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // store address D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nt_rl_inv_12x4_lib4, .-kernel_dtrsm_nt_rl_inv_12x4_lib4
#endif





//                                             rdi     rsi         rdx       rcx         r8      r9          rsp+8     rsp+16      rsp+24     rsp+32   rsp+40     rsp+48   rsp+56     rsp+64
// void kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4(int kp, double *Ap, int sdap, double *Bp, int km, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4
	.type kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4, @function
kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4
_kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt add

	movl	%edi, %r10d // kp
	movq	%rsi, %r11  // Ap
	movl	%edx, %r12d // sdap
	sall	$5, %r12d   // 4*sdap*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13  // Bp

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_ADD_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_add_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_add_nt_12x4_lib4
#endif
#endif


	// call inner dgemm kernel nt sub

	movl	%r8d, %r10d                 // km
	movq	%r9, %r11                   // Am
	movl	STACKSIZE +  8(%rsp), %r12d // sdam
	sall	$5, %r12d                   // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	STACKSIZE + 16(%rsp), %r13  // Bm

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	STACKSIZE + 24(%rsp), %r10  // C
	movl	STACKSIZE + 32(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 56(%rsp), %r10  // E 
	movq	STACKSIZE + 64(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_inv_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 40(%rsp), %r10 // store address D
	movl	STACKSIZE + 48(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4, .-kernel_dgemm_dtrsm_nt_rl_inv_12x4_lib4
#endif





//                                       rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24
// void kernel_dtrsm_nt_rl_one_12x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_12x4_lib4
	.type kernel_dtrsm_nt_rl_one_12x4_lib4, @function
kernel_dtrsm_nt_rl_one_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_12x4_lib4
_kernel_dtrsm_nt_rl_one_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_one_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // store address D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nt_rl_one_12x4_lib4, .-kernel_dtrsm_nt_rl_one_12x4_lib4
#endif





//                                          rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32  rsp+40
// void kernel_dtrsm_nt_rl_one_12x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_rl_one_12x4_vs_lib4
	.type kernel_dtrsm_nt_rl_one_12x4_vs_lib4, @function
kernel_dtrsm_nt_rl_one_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_rl_one_12x4_vs_lib4
_kernel_dtrsm_nt_rl_one_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movl	STACKSIZE + 40(%rsp), %r11d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RLT_ONE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rlt_one_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rlt_one_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // store address D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  32(%rsp), %r12d // km 
	movl	STACKSIZE +  40(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nt_rl_one_12x4_vs_lib4, .-kernel_dtrsm_nt_rl_one_12x4_vs_lib4
#endif





//                                       rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32 
// void kernel_dtrsm_nt_ru_inv_12x4_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_12x4_lib4
	.type kernel_dtrsm_nt_ru_inv_12x4_lib4, @function
kernel_dtrsm_nt_ru_inv_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_12x4_lib4
_kernel_dtrsm_nt_ru_inv_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rut_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_12x4_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // store address D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nt_ru_inv_12x4_lib4, .-kernel_dtrsm_nt_ru_inv_12x4_lib4
#endif





//                                          rdi    rsi        rdx      rcx        r8         r9       rsp+8      rsp+16   rsp+24     rsp+32              rsp+40  rsp+48
// void kernel_dtrsm_nt_ru_inv_12x4_vs_lib4(int k, double *A, int sda, double *B, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nt_ru_inv_12x4_vs_lib4
	.type kernel_dtrsm_nt_ru_inv_12x4_vs_lib4, @function
kernel_dtrsm_nt_ru_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nt_ru_inv_12x4_vs_lib4
_kernel_dtrsm_nt_ru_inv_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt

	movl	%edi, %r10d
	movq	%rsi, %r11
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NT_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nt_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nt_12x4_lib4
#endif
#endif


	// call inner blender nn

	movq	%r8, %r10
	movl	%r9d, %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_BLEND_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_blend_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_blend_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 24(%rsp), %r10  // E 
	movq	STACKSIZE + 32(%rsp), %r11  // inv_diag_E 
	movl	STACKSIZE + 48(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUT_INV_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_rut_inv_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_rut_inv_12x4_vs_lib4
#endif
#endif


	// store n

	movq	STACKSIZE + 8(%rsp), %r10 // store address D
	movl	STACKSIZE + 16(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE +  40(%rsp), %r12d // km 
	movl	STACKSIZE +  48(%rsp), %r13d // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nt_ru_inv_12x4_vs_lib4, .-kernel_dtrsm_nt_ru_inv_12x4_vs_lib4
#endif





//                                       edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40
// void kernel_dtrsm_nn_ru_inv_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_12x4_lib4
	.type kernel_dtrsm_nn_ru_inv_12x4_lib4, @function
kernel_dtrsm_nn_ru_inv_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_12x4_lib4
_kernel_dtrsm_nn_ru_inv_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_run_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nn_ru_inv_12x4_lib4, .-kernel_dtrsm_nn_ru_inv_12x4_lib4
#endif





//                                          edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40              rsp+48  rsp+56
// void kernel_dtrsm_nn_ru_inv_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, double *inv_diag_E, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ru_inv_12x4_vs_lib4
	.type kernel_dtrsm_nn_ru_inv_12x4_vs_lib4, @function
kernel_dtrsm_nn_ru_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ru_inv_12x4_vs_lib4
_kernel_dtrsm_nn_ru_inv_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movq	STACKSIZE + 40(%rsp), %r11 // inv_diag_E

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_RUN_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_run_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_run_inv_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE + 48(%rsp), %r12d // km
	movl	STACKSIZE + 56(%rsp), %r13d // kn

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nn_ru_inv_12x4_vs_lib4, .-kernel_dtrsm_nn_ru_inv_12x4_vs_lib4
#endif





//                                       edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40
// void kernel_dtrsm_nn_ll_one_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_12x4_lib4
	.type kernel_dtrsm_nn_ll_one_12x4_lib4, @function
kernel_dtrsm_nn_ll_one_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_12x4_lib4
_kernel_dtrsm_nn_ll_one_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movl	STACKSIZE + 40(%rsp), %r11d // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lln_one_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nn_ll_one_12x4_lib4, .-kernel_dtrsm_nn_ll_one_12x4_lib4
#endif





//                                          edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48  tsp+56
// void kernel_dtrsm_nn_ll_one_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_ll_one_12x4_vs_lib4
	.type kernel_dtrsm_nn_ll_one_12x4_vs_lib4, @function
kernel_dtrsm_nn_ll_one_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_ll_one_12x4_vs_lib4
_kernel_dtrsm_nn_ll_one_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movl	STACKSIZE + 40(%rsp), %r11d // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LLN_ONE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lln_one_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lln_one_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

	movl	STACKSIZE + 48(%rsp), %r12d // km
	movl	STACKSIZE + 56(%rsp), %r13d // kn

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nn_ll_one_12x4_vs_lib4, .-kernel_dtrsm_nn_ll_one_12x4_vs_lib4
#endif





//                                       edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48
// void kernel_dtrsm_nn_lu_inv_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, double *inv_diag_E);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_12x4_lib4
	.type kernel_dtrsm_nn_lu_inv_12x4_lib4, @function
kernel_dtrsm_nn_lu_inv_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_12x4_lib4
_kernel_dtrsm_nn_lu_inv_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movl	STACKSIZE + 40(%rsp), %r11d // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
//	movslq	%r11d, %r11
	movq	STACKSIZE + 48(%rsp), %r12  // inv_diag_E 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lun_inv_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dtrsm_nn_lu_inv_12x4_lib4, .-kernel_dtrsm_nn_lu_inv_12x4_lib4
#endif





//                                          edi    rsi        rdx      ecx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32     rsp+40   rsp+48               rsp+56  rsp+64
// void kernel_dtrsm_nn_lu_inv_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *E, int sde, double *inv_diag_E), int km, int kn;

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsm_nn_lu_inv_12x4_vs_lib4
	.type kernel_dtrsm_nn_lu_inv_12x4_vs_lib4, @function
kernel_dtrsm_nn_lu_inv_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsm_nn_lu_inv_12x4_vs_lib4
_kernel_dtrsm_nn_lu_inv_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nt 

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// solve

	movq	STACKSIZE + 32(%rsp), %r10  // E 
	movl	STACKSIZE + 40(%rsp), %r11d // sde
	sall	$5, %r11d // 4*sde*sizeof(double)
//	movslq	%r11d, %r11
	movq	STACKSIZE + 48(%rsp), %r12  // inv_diag_E 
	movl	STACKSIZE + 56(%rsp), %r13d  // km 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSM_LUN_INV_12X4_vs_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dtrsm_lun_inv_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dtrsm_lun_inv_12x4_vs_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11
	movl	STACKSIZE + 56(%rsp), %r12d  // km 
	movl	STACKSIZE + 64(%rsp), %r13d  // kn 

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dtrsm_nn_lu_inv_12x4_vs_lib4, .-kernel_dtrsm_nn_lu_inv_12x4_vs_lib4
#endif





//                                   edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32
// void kernel_dgetrf_nn_l_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_l_12x4_lib4
	.type kernel_dgetrf_nn_l_12x4_lib4, @function
kernel_dgetrf_nn_l_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_l_12x4_lib4
_kernel_dgetrf_nn_l_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 32(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_L_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_l_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_l_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgetrf_nn_l_12x4_lib4, .-kernel_dgetrf_nn_l_12x4_lib4
#endif





//                                      edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32              rsp+40  rsp+48
// void kernel_dgetrf_nn_l_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_l_12x4_vs_lib4
	.type kernel_dgetrf_nn_l_12x4_vs_lib4, @function
kernel_dgetrf_nn_l_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_l_12x4_vs_lib4
_kernel_dgetrf_nn_l_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 32(%rsp), %r10  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_L_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_l_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_l_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11
	movl	STACKSIZE + 40(%rsp), %r12d  // km
	movl	STACKSIZE + 48(%rsp), %r13d  // km

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgetrf_nn_l_12x4_vs_lib4, .-kernel_dgetrf_nn_l_12x4_vs_lib4
#endif





//                                   edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32
// void kernel_dgetrf_nn_m_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_m_12x4_lib4
	.type kernel_dgetrf_nn_m_12x4_lib4, @function
kernel_dgetrf_nn_m_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_m_12x4_lib4
_kernel_dgetrf_nn_m_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 16(%rsp), %r10 // D
	subq	$128, %r10 // E
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	STACKSIZE + 32(%rsp), %r12  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_M_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_m_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_m_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgetrf_nn_m_12x4_lib4, .-kernel_dgetrf_nn_m_12x4_lib4
#endif





//                                      edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32              rsp+40  rsp+48
// void kernel_dgetrf_nn_m_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_m_12x4_vs_lib4
	.type kernel_dgetrf_nn_m_12x4_vs_lib4, @function
kernel_dgetrf_nn_m_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_m_12x4_vs_lib4
_kernel_dgetrf_nn_m_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 16(%rsp), %r10 // D
	subq	$128, %r10 // E
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	STACKSIZE + 32(%rsp), %r12  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_M_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_m_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_m_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11
	movl	STACKSIZE + 40(%rsp), %r12d  // km
	movl	STACKSIZE + 48(%rsp), %r13d  // km

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgetrf_nn_m_12x4_vs_lib4, .-kernel_dgetrf_nn_m_12x4_vs_lib4
#endif





//                                   edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32
// void kernel_dgetrf_nn_r_12x4_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_r_12x4_lib4
	.type kernel_dgetrf_nn_r_12x4_lib4, @function
kernel_dgetrf_nn_r_12x4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_r_12x4_lib4
_kernel_dgetrf_nn_r_12x4_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 16(%rsp), %r10 // D
	subq	$256, %r10 // E
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	STACKSIZE + 32(%rsp), %r12  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_R_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_r_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_r_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_lib4
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
	.size	kernel_dgetrf_nn_r_12x4_lib4, .-kernel_dgetrf_nn_r_12x4_lib4
#endif





//                                      edi    rsi        rdx      rcx        r8       r9         rsp+8    rsp+16     rsp+24   rsp+32              rsp+40  rsp+48
// void kernel_dgetrf_nn_r_12x4_vs_lib4(int k, double *A, int sda, double *B, int sdb, double *C, int sdc, double *D, int sdd, double *inv_diag_D, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgetrf_nn_r_12x4_vs_lib4
	.type kernel_dgetrf_nn_r_12x4_vs_lib4, @function
kernel_dgetrf_nn_r_12x4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgetrf_nn_r_12x4_vs_lib4
_kernel_dgetrf_nn_r_12x4_vs_lib4:
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
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm11


	// call inner dgemm kernel nn

	movl	%edi, %r10d // k
	movq	%rsi, %r11 // A
	movl	%edx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
//	movslq	%r12d, %r12
	movq	%rcx, %r13 // B
	movl	%r8d, %r14d // sda
	sall	$5, %r14d // 4*sdb*sizeof(double)
//	movslq	%r14d, %r14

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_SUB_NN_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_kernel_dgemm_sub_nn_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_kernel_dgemm_sub_nn_12x4_lib4
#endif
#endif


	// call inner blender_loader nn

	movq	%r9, %r10 // C
	movl	STACKSIZE +  8(%rsp), %r11d // sdc
	sall	$5, %r11d // 4*sdc*sizeof(double)
//	movslq	%r11d, %r11

#if MACRO_LEVEL>=1
	INNER_SCALE_11_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_scale_11_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_scale_11_12x4_lib4
#endif
#endif


	// factorization

	movq	STACKSIZE + 16(%rsp), %r10 // D
	subq	$256, %r10 // E
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
	movq	STACKSIZE + 32(%rsp), %r12  // inv_diag_D 

#if MACRO_LEVEL>=1
	INNER_EDGE_DGETRF_R_12X4_LIB4
#else
#if defined(OS_LINUX)
	call inner_edge_dgetrf_r_12x4_lib4
#elif defined(OS_MAC)
	callq _inner_edge_dgetrf_r_12x4_lib4
#endif
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // D
	movl	STACKSIZE + 24(%rsp), %r11d // sdd
	sall	$5, %r11d // 4*sdd*sizeof(double)
//	movslq	%r11d, %r11
	movl	STACKSIZE + 40(%rsp), %r12d  // km
	movl	STACKSIZE + 48(%rsp), %r13d  // km

#if MACRO_LEVEL>=1
	INNER_STORE_12X4_VS_LIB4
#else
#if defined(OS_LINUX)
	call inner_store_12x4_vs_lib4
#elif defined(OS_MAC)
	callq _inner_store_12x4_vs_lib4
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
	.size	kernel_dgetrf_nn_r_12x4_vs_lib4, .-kernel_dgetrf_nn_r_12x4_vs_lib4
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
LC00: // { -1 -1 -1 1 }
	.align 5
#endif
	.quad	-1
	.quad	-1
	.quad	-1
	.quad	1
#if defined(OS_LINUX)
	.align 32
.LC01: // { -1 -1 -1 -1 }
#elif defined(OS_MAC)
LC01: // { -1 -1 -1 -1 }
	.align 5
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
LC02: // { 3.5 2.5 1.5 0.5 }
	.align 5
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
LC03: // { 7.5 6.5 5.5 4.5 }
	.align 5
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
.LC04: // { 11.5 10.5 9.5 8.5 }
#elif defined(OS_MAC)
LC04: // { 11.5 10.5 9.5 8.5 }
	.align 5
#endif
	.long	0
	.long	1075904512
	.long	0
	.long	1076035584
	.long	0
	.long	1076166656
 	.long	0
	.long	1076297728
#if defined(OS_LINUX)
	.align 32
.LC05: // { 1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
LC05: // { 1.0 1.0 1.0 1.0 }
	.align 5
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
	.section	.note.GNU-stack,"",@progbits
#elif defined(OS_MAC)
	.subsections_via_symbols
#endif

