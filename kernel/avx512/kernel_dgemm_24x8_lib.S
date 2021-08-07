/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2021 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/



// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r10   <- C
// r13   <- ldc
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_24X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_24x8_lib)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm25

	vmulpd		%zmm0, %zmm25, %zmm0
	vmulpd		%zmm1, %zmm25, %zmm1
	vmulpd		%zmm2, %zmm25, %zmm2
	vmulpd		%zmm3, %zmm25, %zmm3
	vmulpd		%zmm4, %zmm25, %zmm4
	vmulpd		%zmm5, %zmm25, %zmm5
	vmulpd		%zmm6, %zmm25, %zmm6
	vmulpd		%zmm7, %zmm25, %zmm7
	vmulpd		%zmm8, %zmm25, %zmm8
	vmulpd		%zmm9, %zmm25, %zmm9
	vmulpd		%zmm10, %zmm25, %zmm10
	vmulpd		%zmm11, %zmm25, %zmm11
	vmulpd		%zmm12, %zmm25, %zmm12
	vmulpd		%zmm13, %zmm25, %zmm13
	vmulpd		%zmm14, %zmm25, %zmm14
	vmulpd		%zmm15, %zmm25, %zmm15
	vmulpd		%zmm16, %zmm25, %zmm16
	vmulpd		%zmm17, %zmm25, %zmm17
	vmulpd		%zmm18, %zmm25, %zmm18
	vmulpd		%zmm19, %zmm25, %zmm19
	vmulpd		%zmm20, %zmm25, %zmm20
	vmulpd		%zmm21, %zmm25, %zmm21
	vmulpd		%zmm22, %zmm25, %zmm22
	vmulpd		%zmm23, %zmm25, %zmm23

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm8
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm16
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm9
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm17
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm10
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm18
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm11
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm19
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm12
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm20
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm13
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm21
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm14
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm22
	addq		%r13, %r12
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm15
	vmovupd		128(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm23
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_24x8_lib)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc
// r14   <- m1
// r15   <- n1
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r10   <- C
// r13   <- ldc
// r14   <- m1
// r15   <- n1
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_24X8_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_24x8_vs_lib)
#endif
	
	// alpha
	vbroadcastsd	0(%r10), %zmm25

	vmulpd		%zmm0, %zmm25, %zmm0
	vmulpd		%zmm1, %zmm25, %zmm1
	vmulpd		%zmm2, %zmm25, %zmm2
	vmulpd		%zmm3, %zmm25, %zmm3
	vmulpd		%zmm4, %zmm25, %zmm4
	vmulpd		%zmm5, %zmm25, %zmm5
	vmulpd		%zmm6, %zmm25, %zmm6
	vmulpd		%zmm7, %zmm25, %zmm7
	vmulpd		%zmm8, %zmm25, %zmm8
	vmulpd		%zmm9, %zmm25, %zmm9
	vmulpd		%zmm10, %zmm25, %zmm10
	vmulpd		%zmm11, %zmm25, %zmm11
	vmulpd		%zmm12, %zmm25, %zmm12
	vmulpd		%zmm13, %zmm25, %zmm13
	vmulpd		%zmm14, %zmm25, %zmm14
	vmulpd		%zmm15, %zmm25, %zmm15
	vmulpd		%zmm16, %zmm25, %zmm16
	vmulpd		%zmm17, %zmm25, %zmm17
	vmulpd		%zmm18, %zmm25, %zmm18
	vmulpd		%zmm19, %zmm25, %zmm19
	vmulpd		%zmm20, %zmm25, %zmm20
	vmulpd		%zmm21, %zmm25, %zmm21
	vmulpd		%zmm22, %zmm25, %zmm22
	vmulpd		%zmm23, %zmm25, %zmm23

	// beta
	vbroadcastsd	0(%r11), %zmm24

	vxorpd		%zmm25, %zmm25, %zmm25 // 0.0

	vucomisd	%xmm25, %xmm24 // beta==0.0 ?
	je			0f // end

	// compute mask for rows
	vcvtsi2sd	%r14d, %xmm25, %xmm25
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %zmm26
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %zmm26
#endif
	vbroadcastsd	%xmm25, %zmm25
	vsubpd		%zmm25, %zmm26, %zmm25
	vpmovq2m	%zmm25, %k1

	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm0
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm8
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm16
	addq		%r13, %r12
	cmpl		$ 2, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm1
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm9
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm17
	addq		%r13, %r12
	cmpl		$ 3, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm2
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm10
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm18
	addq		%r13, %r12
	cmpl		$ 4, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm3
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm11
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm19
	addq		%r13, %r12
	cmpl		$ 5, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm4
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm12
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm20
	addq		%r13, %r12
	cmpl		$ 6, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm5
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm13
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm21
	addq		%r13, %r12
	cmpl		$ 7, %r15d
	jl			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm6
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm14
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm22
	addq		%r13, %r12
	cmpl		$ 7, %r15d
	je			0f // end
	vmovupd		0(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm7
	vmovupd		64(%r12), %zmm25
	vfmadd231pd	%zmm24, %zmm25, %zmm15
	vmovupd		128(%r12), %zmm25 {%k1}{z}
	vfmadd231pd	%zmm24, %zmm25, %zmm23
//	addq		%r13, %r12

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_24x8_vs_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10  <- D
// r11  <- ldd
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_24X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_24x8_lib)
#endif
	
	vmovupd %zmm0, 0(%r10)
	vmovupd %zmm8, 64(%r10)
	vmovupd %zmm16, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm1, 0(%r10)
	vmovupd %zmm9, 64(%r10)
	vmovupd %zmm17, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm2, 0(%r10)
	vmovupd %zmm10, 64(%r10)
	vmovupd %zmm18, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm3, 0(%r10)
	vmovupd %zmm11, 64(%r10)
	vmovupd %zmm19, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm4, 0(%r10)
	vmovupd %zmm12, 64(%r10)
	vmovupd %zmm20, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm5, 0(%r10)
	vmovupd %zmm13, 64(%r10)
	vmovupd %zmm21, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm6, 0(%r10)
	vmovupd %zmm14, 64(%r10)
	vmovupd %zmm22, 128(%r10)
	addq		%r11, %r10
	vmovupd %zmm7, 0(%r10)
	vmovupd %zmm15, 64(%r10)
	vmovupd %zmm23, 128(%r10)
//	addq		%r11, %r10

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_24x8_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd
// r12  <- m1
// r13  <- n1
// zmm0 <- []
// ...
// zmm15 <- []
//
// output arguments:
// r10  <- D
// r11  <- ldd
// r12  <- m1
// r13  <- n1
// zmm0 <- []
// ...
// zmm15 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_24X8_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_24x8_vs_lib)
#endif
	
	// compute mask for rows
	vcvtsi2sd	%r12d, %xmm25, %xmm25
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovupd		.LC02(%rip), %zmm24
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %zmm24
#endif
	vbroadcastsd	%xmm25, %zmm25
	vsubpd		%zmm25, %zmm24, %zmm25
	vpmovq2m	%zmm25, %k1

	vmovupd %zmm0, 0(%r10)
	vmovupd %zmm8, 64(%r10)
	vmovupd %zmm16, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 2, %r13d
	jl			0f // end
	vmovupd %zmm1, 0(%r10)
	vmovupd %zmm9, 64(%r10)
	vmovupd %zmm17, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 3, %r13d
	jl			0f // end
	vmovupd %zmm2, 0(%r10)
	vmovupd %zmm10, 64(%r10)
	vmovupd %zmm18, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 4, %r13d
	jl			0f // end
	vmovupd %zmm3, 0(%r10)
	vmovupd %zmm11, 64(%r10)
	vmovupd %zmm19, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 5, %r13d
	jl			0f // end
	vmovupd %zmm4, 0(%r10)
	vmovupd %zmm12, 64(%r10)
	vmovupd %zmm20, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 6, %r13d
	jl			0f // end
	vmovupd %zmm5, 0(%r10)
	vmovupd %zmm13, 64(%r10)
	vmovupd %zmm21, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 7, %r13d
	jl			0f // end
	vmovupd %zmm6, 0(%r10)
	vmovupd %zmm14, 64(%r10)
	vmovupd %zmm22, 128(%r10) {%k1}
	addq		%r11, %r10
	cmpl		$ 7, %r13d
	je			0f // end
	vmovupd %zmm7, 0(%r10)
	vmovupd %zmm15, 64(%r10)
	vmovupd %zmm23, 128(%r10) {%k1}
//	addq		%r11, %r10

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_24x8_vs_lib)
#endif





// common inner routine with file scope
//
// prefetch
//
// input arguments:
// r10  <- D
// r11  <- ldd
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_PREFETCH0_24X8_LIB
#else
	.p2align 4,,15
	FUN_START(inner_prefetch0_24x8_lib)
#endif

	leaq		0(%r11, %r11, 2), %r12
	leaq		0(%r10, %r11, 4), %r13

	prefetcht0	0(%r10)
	prefetcht0	64(%r10)
	prefetcht0	128(%r10)
	prefetcht0	191(%r10)
	prefetcht0	0(%r10, %r11)
	prefetcht0	64(%r10, %r11)
	prefetcht0	128(%r10, %r11)
	prefetcht0	191(%r10, %r11)
	prefetcht0	0(%r10, %r11, 2)
	prefetcht0	64(%r10, %r11, 2)
	prefetcht0	128(%r10, %r11, 2)
	prefetcht0	191(%r10, %r11, 2)
	prefetcht0	0(%r10, %r12)
	prefetcht0	64(%r10, %r12)
	prefetcht0	128(%r10, %r12)
	prefetcht0	191(%r10, %r12)
	prefetcht0	0(%r13)
	prefetcht0	64(%r13)
	prefetcht0	128(%r13)
	prefetcht0	191(%r13)
	prefetcht0	0(%r13, %r11)
	prefetcht0	64(%r13, %r11)
	prefetcht0	128(%r13, %r11)
	prefetcht0	191(%r13, %r11)
	prefetcht0	0(%r13, %r11, 2)
	prefetcht0	64(%r13, %r11, 2)
	prefetcht0	128(%r13, %r11, 2)
	prefetcht0	191(%r13, %r11, 2)
	prefetcht0	0(%r13, %r12)
	prefetcht0	64(%r13, %r12)
	prefetcht0	128(%r13, %r12)
	prefetcht0	191(%r13, %r12)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_prefetch0_24x8_lib)
#endif





//                                   1      2              3          4        5          6             7          8        9          10
// void kernel_dgemm_nt_24x8_lib88cc(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_24x8_lib88cc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// prefetch C

	movq		ARG6, %r10 // beta
	vmovsd		0(%r10), %xmm14
	vxorpd		%xmm15, %xmm15, %xmm15 // 0.0
	vucomisd	%xmm15, %xmm14 // beta==0.0 ?
	je			100f // end

	movq	ARG7, %r10 // C
	movq	ARG8, %r11 // ldc
	sall	$ 3, %r11d

#if MACRO_LEVEL>=1
	INNER_PREFETCH0_24X8_LIB
#else
	CALL(inner_prefetch0_24x8_lib)
#endif

100:


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$ 6, %r12d // 8*sda*sizeof(double)
	movq	ARG5, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_24X8_LIB8
#else
	CALL(inner_kernel_dgemm_nt_24x8_lib8)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$ 3, %r13d

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_24X8_LIB
#else
	CALL(inner_scale_ab_24x8_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$ 3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_24X8_LIB
#else
	CALL(inner_store_24x8_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_24x8_lib88cc)





//                                      1      2              3          4        5          6             7          8        9          10       11      12
// void kernel_dgemm_nt_24x8_vs_lib88cc(int k, double *alpha, double *A, int sda, double *B, double *beta, double *C, int ldc, double *D, int ldd, int m1, int n1);

	.p2align 4,,15
	GLOB_FUN_START(kernel_dgemm_nt_24x8_vs_lib88cc)

	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nn

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$ 6, %r12d // 8*sda*sizeof(double)
	movq	ARG5, %r13  // B
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMM_NT_24X8_VS_LIB8
#else
	CALL(inner_kernel_dgemm_nt_24x8_vs_lib8)
#endif


	// call inner blend

	movq	ARG2, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movq	ARG8, %r13   // ldc
	sall	$ 3, %r13d
	movq	ARG11, %r14 // m1
	movq	ARG12, %r15 // n1

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_24X8_VS_LIB
#else
	CALL(inner_scale_ab_24x8_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movq	ARG10, %r11 // ldd
	sall	$ 3, %r11d
	movq	ARG11, %r12 // m1
	movq	ARG12, %r13 // n1

#if MACRO_LEVEL>=1
	INNER_STORE_24X8_VS_LIB
#else
	CALL(inner_store_24x8_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_dgemm_nt_24x8_vs_lib88cc)






