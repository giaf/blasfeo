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

// common inner kernel with file scope
//
// input arguments:
// edi  <- km
// rsi  <- A
// rdx  <- B
// ecx  <- alg
// r8d  <- tc
// r9   <- C
// r10d <- td
// ymm0 <- [d00 d11 d22 d33]
// ymm1 <- [d01 d10 d23 d32]
// ymm2 <- [d03 d12 d21 d30]
// ymm3 <- [d02 d13 d20 d31]
//
// output arguments:
// edi  <- dirty
// rsi  <- dirty
// rdx  <- dirty
// ecx  <- alg
// r8d  <- tc
// r9   <- C
// r10d <- td
// ymm0 <- [d00 d10 d20 d30]
// ymm1 <- [d01 d11 d21 d31]
// ymm2 <- [d02 d12 d22 d32]
// ymm3 <- [d03 d13 d23 d33]

	.p2align 4,,15
	.type inner_kernel_dgemm_add_nt_4x4_lib4, @function
inner_kernel_dgemm_add_nt_4x4_lib4:
	
	cmpl	$0, %edi
	jle		.L002 // blend

	// prefetch
	vmovapd 0(%rsi), %ymm8 // A[0]
	vmovapd 0(%rdx), %ymm12 // B[0]

	cmpl	$4, %edi
	jle		.L000 // consider clean-up loop

	// main loop
	.p2align 3
.L001: // main loop
	
	// unroll 0
	vmovapd 32(%rsi), %ymm10 // A[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%rdx), %ymm12 // B[4]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	subl	$4, %edi
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%rsi), %ymm8 // A[8]
	vmulpd	%ymm10, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%rdx), %ymm12 // B[8]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%rsi), %ymm10 // A[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	addq	$128, %rsi
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%rdx), %ymm12 // B[12]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	addq	$128, %rdx
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 3
	vmovapd 0(%rsi), %ymm8 // A[8]
	vmulpd	%ymm10, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	cmpl	$3, %edi
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 0(%rdx), %ymm12 // B[8]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	jg		.L001 // main loop 


.L000: // consider clean4-up
	
	cmpl	$3, %edi
	jle		.L011 // clean1

	// unroll 0
	vmovapd 32(%rsi), %ymm10 // A[4]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 32(%rdx), %ymm12 // B[4]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
//	subl	$4, %edi
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 1
	vmovapd 64(%rsi), %ymm8 // A[8]
	vmulpd	%ymm10, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 64(%rdx), %ymm12 // B[8]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 2
	vmovapd 96(%rsi), %ymm10 // A[12]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
//	addq	$128, %rsi
	vaddpd	%ymm1, %ymm15, %ymm1
	vmovapd 96(%rdx), %ymm12 // B[12]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
//	addq	$128, %rdx
	vaddpd	%ymm2, %ymm15, %ymm2

	// unroll 3
//	vmovapd 0(%rsi), %ymm8 // A[8]
	vmulpd	%ymm10, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	cmpl	$3, %edi
	vaddpd	%ymm1, %ymm15, %ymm1
//	vmovapd 0(%rdx), %ymm12 // B[8]
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm10, %ymm13, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	jmp		.L002 // blend


.L011: // consider clean1-up loop

	cmpl	$0, %edi
	jle		.L002 // blend

	// clean-up loop
.L003: // clean up loop
	
	vmovapd 0(%rsi), %ymm8 // A[0]
	vmovapd 0(%rdx), %ymm12 // B[0]
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	vshufpd $0x5, %ymm12, %ymm12, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	addq	$32, %rsi
	vaddpd	%ymm1, %ymm15, %ymm1
	vperm2f128 $0x1, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	addq	$32, %rdx
	vaddpd	%ymm3, %ymm15, %ymm3
	vshufpd $0x5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm8, %ymm13, %ymm15
	subl	$1, %edi
	vaddpd	%ymm2, %ymm15, %ymm2

	cmpl	$0, %edi

	jg		.L003 // clean up loop 

	

.L002: // blend

	cmpl	$0, %ecx // alg
	jne		.L006 // alg!=0

	// alg==0
	cmpl	$0, %r10d // td
	jne		.L005 // td==t

	// td==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	jmp			.L004 // return

.L005: // td==t

	// td== t
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

	jmp			.L004 // return

.L006: // alg!=0

	cmpl	$0, %r8d // tc
	jne		.L008 // tc==t

	// tc==n
	vblendpd	$0xa, %ymm1, %ymm0, %ymm8
	vblendpd	$0x5, %ymm1, %ymm0, %ymm9
	vblendpd	$0xa, %ymm3, %ymm2, %ymm10
	vblendpd	$0x5, %ymm3, %ymm2, %ymm11

	vblendpd	$0xc, %ymm10, %ymm8, %ymm0
	vblendpd	$0x3, %ymm10, %ymm8, %ymm2
	vblendpd	$0xc, %ymm11, %ymm9, %ymm1
	vblendpd	$0x3, %ymm11, %ymm9, %ymm3

	jmp			.L009 // alg

.L008: // tc==t

	// td== t
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm0, %ymm1, %ymm9
	vunpckhpd	%ymm2, %ymm3, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm8, %ymm10, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm9, %ymm11, %ymm3

.L009: // alg

	cmpl	$1, %ecx // alg
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

	jmp			.L010 // check td

.L007: // alg==-1

	// alg==1
	vmovapd		0(%r9), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0
	vmovapd		32(%r9), %ymm15
	vsubpd		%ymm1, %ymm15, %ymm1
	vmovapd		64(%r9), %ymm15
	vsubpd		%ymm2, %ymm15, %ymm2
	vmovapd		96(%r9), %ymm15
	vsubpd		%ymm3, %ymm15, %ymm3

.L010: // check td

	cmpl	%r8d, %r10d
	je		.L004 // ret

	// (tc==0 && td==1) || (tc==1 && td==0)
	vunpcklpd	%ymm1, %ymm0, %ymm8
	vunpckhpd	%ymm1, %ymm0, %ymm9
	vunpcklpd	%ymm3, %ymm2, %ymm10
	vunpckhpd	%ymm3, %ymm2, %ymm11

	vperm2f128	$0x20, %ymm10, %ymm8, %ymm0
	vperm2f128	$0x31, %ymm10, %ymm8, %ymm2
	vperm2f128	$0x20, %ymm11, %ymm9, %ymm1
	vperm2f128	$0x31, %ymm11, %ymm9, %ymm3

.L004: // return

	ret

	.size	inner_kernel_dgemm_add_nt_4x4_lib4, .-inner_kernel_dgemm_add_nt_4x4_lib4





//                               rdi     rsi        rdx        rcx      r8         r9      rsp+4      rsp+16
// void kernel_dgemm_nt_4x4_lib4(int km, double *A, double *B, int alg, int tc, double *C, int td, double *D);

	.p2align 4,,15
	.globl kernel_dgemm_nt_4x4_lib4
	.type kernel_dgemm_nt_4x4_lib4, @function
kernel_dgemm_nt_4x4_lib4:
	
	// prologue
	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper

//	leaq	.LSTR(%rip), %rdi
//	movl	STACKSIZE +  8(%rsp), %esi
//	movl	STACKSIZE + 16(%rsp), %edx
//	movl	STACKSIZE + 24(%rsp), %ecx
//	subq	$16, %rsp
//	movq	%r9,  (%rsp)
//	movq	%r8, 8(%rsp)
//	call	printf@PLT
//	addq	$16, %rsp

	// zero accumulation registers
	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// additional arguments
	movl	STACKSIZE + 8(%rsp), %r10d // td


	// call inner kernel
	call inner_kernel_dgemm_add_nt_4x4_lib4



	// store
	movq	STACKSIZE +  16(%rsp), %r10 // store address
	vmovapd %ymm0,  0(%r10)
	vmovapd %ymm1, 32(%r10)
	vmovapd %ymm2, 64(%r10)
	vmovapd %ymm3, 96(%r10)



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

	.size	kernel_dgemm_nt_4x4_lib4, .-kernel_dgemm_nt_4x4_lib4





//                                  rdi     rsi        rdx        rcx      r8         r9      rsp+8      rsp+16  rsp+24  rsp+32
// void kernel_dgemm_nt_4x4_vs_lib4(int km, double *A, double *B, int alg, int tc, double *C, int td, double *D, int km, int kn);

	.p2align 4,,15
	.globl kernel_dgemm_nt_4x4_vs_lib4
	.type kernel_dgemm_nt_4x4_vs_lib4, @function
kernel_dgemm_nt_4x4_vs_lib4:
	
	// prologue
	subq	$STACKSIZE, %rsp
	movq	%rbx,   (%rsp)
	movq	%rbp,  8(%rsp)
	movq	%r12, 16(%rsp)
	movq	%r13, 24(%rsp)
	movq	%r14, 32(%rsp)
	movq	%r15, 40(%rsp)

	vzeroupper

//	leaq	.LSTR(%rip), %rdi
//	movl	STACKSIZE +  8(%rsp), %esi
//	movl	STACKSIZE + 16(%rsp), %edx
//	movl	STACKSIZE + 24(%rsp), %ecx
//	subq	$16, %rsp
//	movq	%r9,  (%rsp)
//	movq	%r8, 8(%rsp)
//	call	printf@PLT
//	addq	$16, %rsp

	// zero accumulation registers
	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// additional arguments
	movl	STACKSIZE + 8(%rsp), %r10d // td


	// call inner kernel
	call inner_kernel_dgemm_add_nt_4x4_lib4



	// store
	cmpl	$0, %r10d

	movq	STACKSIZE +  16(%rsp), %r11 // store address
	movl	STACKSIZE +  24(%rsp), %r12d // km 
	movl	STACKSIZE +  32(%rsp), %r13d // kn 

	jne		.L013 // store t

	// store n
	vcvtsi2sd	%r12d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$2, %r13d
	vmaskmovpd	%ymm0, %ymm15,  0(%r11)
	jl			.L012 // end
	cmpl		$3, %r13d
	vmaskmovpd	%ymm1, %ymm15, 32(%r11)
	jl			.L012 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r11)
	je			.L012 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r11)

	jmp		.L012

.L013:
	// store t
	vcvtsi2sd	%r13d, %xmm15, %xmm15
	vmovupd		.LC02(%rip), %ymm14
	vmovddup	%xmm15, %xmm15
	vinsertf128	$1, %xmm15, %ymm15, %ymm15
	vsubpd		%ymm15, %ymm14, %ymm15
	cmpl		$2, %r12d
	vmaskmovpd	%ymm0, %ymm15,  0(%r11)
	jl			.L012 // end
	cmpl		$3, %r12d
	vmaskmovpd	%ymm1, %ymm15, 32(%r11)
	jl			.L012 // end
	vmaskmovpd	%ymm2, %ymm15, 64(%r11)
	je			.L012 // end
	vmaskmovpd	%ymm3, %ymm15, 96(%r11)



.L012: // end

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

	.size	kernel_dgemm_nt_4x4_vs_lib4, .-kernel_dgemm_nt_4x4_vs_lib4



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



	.section	.note.GNU-stack,"",@progbits
