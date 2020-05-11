/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
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
// r13   <- ldc*sizeof(float)
// ymm0 <- []
// ...
// ymm7 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc*sizeof(float)
// ymm0 <- []
// ...
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_16X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_16x4_lib)
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

	leaq		0(%r13, %r13, 2), %r14

	vmovups		0(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm0, %ymm15, %ymm0
	vmovups		32(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm4, %ymm15, %ymm4
	vmovups		0(%r12, %r13), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm1, %ymm15, %ymm1
	vmovups		32(%r12, %r13), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm5, %ymm15, %ymm5
	vmovups		0(%r12, %r13, 2), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm2, %ymm15, %ymm2
	vmovups		32(%r12, %r13, 2), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm6, %ymm15, %ymm6
	vmovups		0(%r12, %r14), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm3, %ymm15, %ymm3
	vmovups		32(%r12, %r14), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_16x4_lib)
#endif





// common inner routine with file scope
//
// scale for generic alpha and beta
//
// input arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc*sizeof(float)
// r14   <- km
// r15   <- kn
// ymm0 <- []
// ...
// ymm7 <- []
//
// output arguments:
// r10   <- alpha
// r11   <- beta
// r12   <- C
// r13   <- ldc*sizeof(float)
// r14   <- km
// r15   <- kn
// ymm0 <- []
// ...
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_SCALE_AB_16X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_scale_ab_16x4_vs_lib)
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

	// compute mask for rows
	vcvtsi2ss	%r14d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC01(%rip), %ymm13
#elif defined(OS_MAC)
	vmovups		LC01(%rip), %ymm13
#endif
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm15, %ymm13, %ymm13

	leaq		0(%r13, %r13, 2), %rax

	vmovups		0(%r12), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm0, %ymm15, %ymm0
	vmaskmovps	32(%r12), %ymm13, %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm4, %ymm15, %ymm4
	cmpl		$2, %r13d
	jl			0f // end
	vmovups		0(%r12, %r13), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm1, %ymm15, %ymm1
	vmaskmovps	32(%r12, %r13), %ymm13, %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm5, %ymm15, %ymm5
	cmpl		$3, %r13d
	jl			0f // end
	vmovups		0(%r12, %r13, 2), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm2, %ymm15, %ymm2
	vmaskmovps	32(%r12, %r13, 2), %ymm13, %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm6, %ymm15, %ymm6
	je			0f // end
	vmovups		0(%r12, %rax), %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm3, %ymm15, %ymm3
	vmaskmovps	32(%r12, %rax), %ymm13, %ymm15
	vmulps		%ymm15, %ymm14, %ymm15
	vaddps		%ymm7, %ymm15, %ymm7

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_scale_ab_16x4_vs_lib)
#endif





// common inner routine with file scope
//
// store n
//
// input arguments:
// r10  <- D
// r11  <- ldd*sizeof(float)
// ymm0 <- []
// ...
// ymm7 <- []
//
// output arguments:
// r10  <- D
// r11  <- ldd*sizeof(float)
// ymm0 <- []
// ...
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_16X4_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_16x4_lib)
#endif
	
	leaq		0(%r11, %r11, 2), %r12

	vmovups 	%ymm0,  0(%r10)
	vmovups 	%ymm4,  32(%r10)
	vmovups 	%ymm1,  0(%r10, %r11)
	vmovups 	%ymm5,  32(%r10, %r11)
	vmovups 	%ymm2,  0(%r10, %r11, 2)
	vmovups 	%ymm6,  32(%r10, %r11, 2)
	vmovups 	%ymm3,  0(%r10, %r12)
	vmovups 	%ymm7,  32(%r10, %r12)


#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_16x4_lib)
#endif





// common inner routine with file scope
//
// store n vs
//
// input arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12  <- km
// r13  <- kn
// ymm0 <- []
// ...
// ymm7 <- []
//
// output arguments:
// r10  <- D
// r11  <- 4*sdd*sizeof(double)
// r12  <- km
// r13  <- kn
// ymm0 <- []
// ...
// ymm7 <- []

#if MACRO_LEVEL>=1
	.macro INNER_STORE_16X4_VS_LIB
#else
	.p2align 4,,15
	FUN_START(inner_store_16x4_vs_lib)
#endif
	
	// compute mask for rows
	vcvtsi2ss	%r12d, %xmm15, %xmm15
#if defined(OS_LINUX) | defined(OS_WINDOWS)
	vmovups		.LC01(%rip), %ymm13
#elif defined(OS_MAC)
	vmovups		LC01(%rip), %ymm13
#endif
	vshufps		$0x00, %xmm15, %xmm15, %xmm15
	vinsertf128	$0x1, %xmm15, %ymm15, %ymm15
	vsubps		%ymm15, %ymm13, %ymm15

	leaq		0(%r11, %r11, 2), %r14

	vmovups		%ymm0, 0(%r10)
	vmaskmovps	%ymm4, %ymm15, 32(%r10)
	cmpl		$2, %r13d
	jl			0f // end
	vmovups		%ymm1, 0(%r10, %r11)
	vmaskmovps	%ymm5, %ymm15, 32(%r10, %r11)
	cmpl		$3, %r13d
	jl			0f // end
	vmovups		%ymm2, 0(%r10, %r11, 2)
	vmaskmovps	%ymm6, %ymm15, 32(%r10, %r11, 2)
	je			0f // end
	vmovups		%ymm3, 0(%r10, %r14)
	vmaskmovps	%ymm7, %ymm15, 32(%r10, %r14)
	//
//	jmp		0f

0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

	FUN_END(inner_store_16x4_vs_lib)
#endif





//                                   1      2             3         4        5         6            7         8        9         10
// void kernel_sgemm_nt_16x4_lib88cc(int k, float *alpha, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd);

	.p2align 4,,15
	GLOB_FUN_START(kernel_sgemm_nt_16x4_lib88cc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 8*sda*sizeof(float)
	movq	ARG5, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_16X4_LIB8
#else
	CALL(inner_kernel_gemm_add_nt_16x4_lib8)
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movl	ARG8, %r13d // ldc
	sall	$2, %r13d // ldc*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X4_LIB
#else
	CALL(inner_scale_ab_16x4_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movl	ARG10, %r11d // ldd
	sall	$2, %r11d // ldd*sizeof(float)

#if MACRO_LEVEL>=1
	INNER_STORE_16X4_LIB
#else
	CALL(inner_store_16x4_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_sgemm_nt_16x4_lib88cc)





//                                      1      2             3         4        5         6            7         8        9         10       12      13
// void kernel_sgemm_nt_16x4_vs_lib88cc(int k, float *alpha, float *A, int sda, float *B, float *beta, float *C, int ldc, float *D, int ldd, int km, int kn);

	.p2align 4,,15
	GLOB_FUN_START(kernel_sgemm_nt_16x4_vs_lib88cc)
	
	PROLOGUE

	// zero accumulation registers

	ZERO_ACC


	// call inner dgemm kernel nt

	movq	ARG1, %r10 // k
	movq	ARG3, %r11  // A
	movq	ARG4, %r12 // sda
	sall	$5, %r12d // 8*sda*sizeof(float)
	movq	ARG5, %r13  // B

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMM_ADD_NT_16X4_LIB8
#else
	CALL(inner_kernel_gemm_add_nt_16x4_lib8)
#endif


	// call inner scale

	movq	%rsi, %r10 // alpha
	movq	ARG6, %r11 // beta
	movq	ARG7, %r12   // C
	movl	ARG8, %r13d // ldc
	sall	$2, %r13d // ldc*sizeof(float)
	movq	ARG11, %r14 // km
	movq	ARG12, %r15 // kn

#if MACRO_LEVEL>=1
	INNER_SCALE_AB_16X4_VS_LIB
#else
	CALL(inner_scale_ab_16x4_vs_lib)
#endif


	// store n

	movq	ARG9, %r10 // D
	movl	ARG10, %r11d // ldd
	sall	$2, %r11d // ldd*sizeof(float)
	movq	ARG11, %r12 // km
	movq	ARG12, %r13 // kn

#if MACRO_LEVEL>=1
	INNER_STORE_16X4_VS_LIB
#else
	CALL(inner_store_16x4_vs_lib)
#endif


	EPILOGUE

	ret

	FUN_END(kernel_sgemm_nt_16x4_vs_lib88cc)







