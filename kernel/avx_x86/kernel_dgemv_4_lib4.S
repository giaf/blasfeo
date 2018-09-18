/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#if defined(OS_LINUX)

#define STACKSIZE 16
#define ARG1  STACKSIZE +  4(%esp)
#define ARG2  STACKSIZE +  8(%esp)
#define ARG3  STACKSIZE + 12(%esp)
#define ARG4  STACKSIZE + 16(%esp)
#define ARG5  STACKSIZE + 20(%esp)
#define ARG6  STACKSIZE + 24(%esp)
#define ARG7  STACKSIZE + 28(%esp)
#define ARG8  STACKSIZE + 32(%esp)
#define ARG9  STACKSIZE + 36(%esp)
#define ARG10 STACKSIZE + 40(%esp)

#if 1

#define PROLOGUE \
	subl	$16, %esp; \
	movl	%ebx, 0(%esp); \
	movl	%esi, 4(%esp); \
	movl	%edi, 8(%esp); \
	movl	%ebp, 12(%esp);
#define EPILOGUE \
	movl	0(%esp), %ebx; \
	movl	4(%esp), %esi; \
	movl	8(%esp), %edi; \
	movl	12(%esp), %ebp; \
	addl	$16, %esp;

#else

#define PROLOGUE \
	pushl	%ebp; \
	pushl	%edi; \
	pushl	%esi; \
	pushl	%ebx;
#define EPILOGUE \
	popl	%ebx; \
	popl	%esi; \
	popl	%edi; \
	popl	%ebp;

#endif

#else

#error wrong OS

#endif



	.text



// common inner routine with file scope
//
// input arguments:
// eax   <- k
// ebx   <- A
// ecx   <- x
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d

//
// output arguments:

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_ADD_N_4_LIB4
#else
	.align 16
	.type inner_kernel_gemv_add_n_4_lib4, @function
inner_kernel_gemv_add_n_4_lib4:
#endif
	
	cmpl	$0, %eax
	jle		2f // return

	cmpl	$4, %eax
	jl		0f // clean-up loop

	// main loop
	.align 8
1: // main loop
	
	vmovapd	0(%ebx), %ymm4
	vbroadcastsd	0(%ecx), %ymm5
	vmulpd	%ymm4, %ymm5, %ymm7
	vaddpd	%ymm0, %ymm7, %ymm0
	
	vmovapd	32(%ebx), %ymm4
	vbroadcastsd	8(%ecx), %ymm5
	vmulpd	%ymm4, %ymm5, %ymm7
	vaddpd	%ymm1, %ymm7, %ymm1
	
	vmovapd	64(%ebx), %ymm4
	vbroadcastsd	16(%ecx), %ymm5
	vmulpd	%ymm4, %ymm5, %ymm7
	vaddpd	%ymm2, %ymm7, %ymm2

	vmovapd	96(%ebx), %ymm4
	vbroadcastsd	24(%ecx), %ymm5
	vmulpd	%ymm4, %ymm5, %ymm7
	vaddpd	%ymm3, %ymm7, %ymm3
	
	subl	$4, %eax
	addl	$128, %ebx
	addl	$32, %ecx
	
	cmpl	$3, %eax

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	vmovapd	0(%ebx), %ymm4
	vbroadcastsd	0(%ecx), %ymm5
	vmulpd	%ymm4, %ymm5, %ymm7
	vaddpd	%ymm0, %ymm7, %ymm0
	
	addl	$32, %ebx
	addl	$8, %ecx
	
	subl	$1, %eax
	cmpl	$0, %eax

	jg		0b // clean

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

	.size	inner_kernel_gemv_add_n_4_lib4, .-inner_kernel_gemv_add_n_4_lib4
#endif





// common inner routine with file scope
//
// input arguments:
// eax  <- k
// ebx   <- A
// ecx   <- bs*sda*sizeof(double) = 32*sda
// edx   <- x
// ymm0  <- [z0a z0b z0c z0d]
// ymm1  <- [z1a z1b z1c z1d]
// ymm2  <- [z2a z2b z2c z2d]
// ymm3  <- [z3a z3b z3c z3d]

//
// output arguments:
// eax  <- 0
// ebx   <- A+4*k*sizeof(double)
// ecx   <- bs*sda*sizeof(double) = 32*sda
// edx   <- x+k*sizeof(double)
// ymm0  <- [z0a z0b z0c z0d]
// ymm1  <- [z1a z1b z1c z1d]
// ymm2  <- [z2a z2b z2c z2d]
// ymm3  <- [z3a z3b z3c z3d]

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_GEMV_ADD_T_4_LIB4
#else
	.align 16
	.type inner_kernel_gemv_add_t_4_lib4, @function
inner_kernel_gemv_add_t_4_lib4:
#endif

	cmpl	$0, %eax
	jle		2f // return

	cmpl	$4, %eax
	jl		0f // clean-up loop

	// main loop
	.align 8
1: // main loop
	
	prefetcht0	0(%ebx, %ecx, 1) // software prefetch
	prefetcht0	64(%ebx, %ecx, 1) // software prefetch

	vmovupd	0(%edx), %ymm4

	vmovapd	0(%ebx), %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm0, %ymm7, %ymm0
	
	subl	$4, %eax

	vmovapd	32(%ebx), %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm1, %ymm7, %ymm1
	
	vmovapd	64(%ebx), %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm2, %ymm7, %ymm2

	vmovapd	96(%ebx), %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm3, %ymm7, %ymm3
	
	addl	%ecx, %ebx
	addl	$32, %edx
	
	cmpl	$3, %eax

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	vcvtsi2sd	%eax, %xmm6, %xmm6
	vmovupd		.LC02, %ymm5
	vmovddup	%xmm6, %xmm6
	vinsertf128	$1, %xmm6, %ymm6, %ymm6
	vsubpd		%ymm6, %ymm5, %ymm6

	vmaskmovpd	0(%edx), %ymm6, %ymm4

	vmaskmovpd	0(%ebx), %ymm6, %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm0, %ymm7, %ymm0
	
	vmaskmovpd	32(%ebx), %ymm6, %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm1, %ymm7, %ymm1
	
	vmaskmovpd	64(%ebx), %ymm6, %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm2, %ymm7, %ymm2

	vmaskmovpd	96(%ebx), %ymm6, %ymm5
	vmulpd	%ymm5, %ymm4, %ymm7
	vaddpd	%ymm3, %ymm7, %ymm3
		
	sall	$3, %eax
	addl	%eax, %ebx
	addl	%eax, %edx
	xorl	%eax, %eax
	
	
2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_gemv_add_t_4_lib4, .-inner_kernel_gemv_add_t_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for ta==n, scale for generic alpha and beta
//
// input arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// ymm0 <- [z0 z1 z2 z3]_a
// ymm1 <- [z0 z1 z2 z3]_b
// ymm2 <- [z0 z1 z2 z3]_c
// ymm3 <- [z0 z1 z2 z3]_d
//
// output arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// ymm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	.align 16
	.type inner_blend_n_scale_ab_4_lib4, @function
inner_blend_n_scale_ab_4_lib4:
#endif

	// reduction
	vaddpd	%ymm0, %ymm1, %ymm0
	vaddpd	%ymm2, %ymm3, %ymm2
	vaddpd	%ymm0, %ymm2, %ymm0

	// alpha
	vbroadcastsd	0(%eax), %ymm7
	vmulpd	%ymm0, %ymm7, %ymm0

	// beta
	vbroadcastsd	0(%ebx), %ymm7
	vmovupd		0(%ecx), %ymm6
	vmulpd		%ymm7, %ymm6, %ymm6
	vaddpd		%ymm0, %ymm6, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
	.size	inner_blend_n_scale_ab_4_lib4, .-inner_blend_n_scale_ab_4_lib4
#endif





// common inner routine with file scope
//
// blend for ta==t, scale for generic alpha and beta
//
// input arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// ymm0 <- [z0a z0b z0c z0d]
// ymm1 <- [z1a z1b z1c z1d]
// ymm2 <- [z2a z2b z2c z2d]
// ymm3 <- [z3a z3b z3c z3d]
//
// output arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// ymm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	.align 16
	.type inner_blend_t_scale_ab_4_lib4, @function
inner_blend_t_scale_ab_4_lib4:
#endif

	// reduction
	vhaddpd	%ymm1, %ymm0, %ymm0
	vhaddpd	%ymm3, %ymm2, %ymm2
	vperm2f128	$0x2, %ymm0, %ymm2, %ymm1
	vperm2f128	$0x13, %ymm0, %ymm2, %ymm0
	vaddpd	%ymm0, %ymm1, %ymm0

	// alpha
	vbroadcastsd	0(%eax), %ymm7
	vmulpd	%ymm0, %ymm7, %ymm0

	// beta
	vbroadcastsd	0(%ebx), %ymm7
	vmovupd		0(%ecx), %ymm6
	vmulpd		%ymm7, %ymm6, %ymm6
	vaddpd		%ymm0, %ymm6, %ymm0
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
	.size	inner_blend_t_scale_ab_4_lib4, .-inner_blend_t_scale_ab_4_lib4
#endif





// common inner routine with file scope
//
// store 
//
// input arguments:
// eax  <- z
// ymm0 <- [z0 z1 z2 z3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.align 16
	.type inner_store_4_lib4, @function
inner_store_4_lib4:
#endif
	
	vmovupd %ymm0,  0(%eax)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

	.size	inner_store_4_lib4, .-inner_store_4_lib4
#endif





// common inner routine with file scope
//
// store vs
//
// input arguments:
// eax   <- D
// ebx   <- km
// ymm0  <- [z0 z1 z2 z3]
//
// output arguments:
// eax   <- D
// ebx   <- km
// ymm0  <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_VS_LIB4
#else
	.align 16
	.type inner_store_4_vs_lib4, @function
inner_store_4_vs_lib4:
#endif
	
	vcvtsi2sd	%ebx, %xmm7, %xmm7
	vmovupd		.LC02, %ymm6
	vmovddup	%xmm7, %xmm7
	vinsertf128	$1, %xmm7, %ymm7, %ymm7
	vsubpd		%ymm7, %ymm6, %ymm7

	vmaskmovpd	%ymm0, %ymm7,  0(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4_vs_lib4, .-inner_store_4_vs_lib4
#endif
#endif





//                            1      2              3          4          5             6          7
// void kernel_dgemv_n_4_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z);

	.align 16
	.globl kernel_dgemv_n_4_lib4
	.type kernel_dgemv_n_4_lib4, @function
kernel_dgemv_n_4_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemv kernel n

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_N_4_LIB4
#else
	call inner_kernel_gemv_add_n_4_lib4
#endif


	// call inner blend n scale ab

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx   // beta
	movl	ARG6, %ecx   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	call inner_blend_n_scale_ab_4_lib4
#endif


	// store

	movl	ARG7, %eax // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	call inner_store_4_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemv_n_4_lib4, .-kernel_dgemv_n_4_lib4






//                               1      2              3          4          5             6          7          8
// void kernel_dgemv_n_4_vs_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z, int k1);

	.align 16
	.globl kernel_dgemv_n_4_vs_lib4
	.type kernel_dgemv_n_4_vs_lib4, @function
kernel_dgemv_n_4_vs_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemv kernel n

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_N_4_LIB4
#else
	call inner_kernel_gemv_add_n_4_lib4
#endif


	// call inner blend n scale ab

	movl	ARG2, %eax // alpha
	movl	ARG5, %ebx   // beta
	movl	ARG6, %ecx   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	call inner_blend_n_scale_ab_4_lib4
#endif


	// store

	movl	ARG7, %eax // z 
	movl	ARG8, %ebx // k1 

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	call inner_store_4_vs_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemv_n_4_vs_lib4, .-kernel_dgemv_n_4_vs_lib4






//                            1      2              3          4        5          6             7         8
// void kernel_dgemv_t_4_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z);

	.align 16
	.globl kernel_dgemv_t_4_lib4
	.type kernel_dgemv_t_4_lib4, @function
kernel_dgemv_t_4_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemv kernel n

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx // sda
	sall	$5, %ecx // 4*sda*sizeof(double)
	movl	ARG5, %edx  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_T_4_LIB4
#else
	call inner_kernel_gemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	ARG2, %eax // alpha
	movl	ARG6, %ebx   // beta
	movl	ARG7, %ecx // y 

#if MACRO_LEVEL>=1
	INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	call inner_blend_t_scale_ab_4_lib4
#endif


	// store

	movl	ARG8, %eax // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	call inner_store_4_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemv_t_4_lib4, .-kernel_dgemv_t_4_lib4





//                               1      2              3          4        5          6             7         8           9
// void kernel_dgemv_t_4_vs_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z, int k1);

	.align 16
	.globl kernel_dgemv_t_4_vs_lib4
	.type kernel_dgemv_t_4_vs_lib4, @function
kernel_dgemv_t_4_vs_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorpd	%ymm0, %ymm0, %ymm0
	vmovapd	%ymm0, %ymm1
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm3


	// call inner dgemv kernel n

	movl	ARG1, %eax // k
	movl	ARG3, %ebx  // A
	movl	ARG4, %ecx // sda
	sall	$5, %ecx // 4*sda*sizeof(double)
	movl	ARG5, %edx  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_GEMV_ADD_T_4_LIB4
#else
	call inner_kernel_gemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	ARG2, %eax // alpha
	movl	ARG6, %ebx   // beta
	movl	ARG7, %ecx // y 

#if MACRO_LEVEL>=1
	INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	call inner_blend_t_scale_ab_4_lib4
#endif


	// store

	movl	ARG8, %eax // z 
	movl	ARG9, %ebx // k1 

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	call inner_store_4_vs_lib4
#endif


	EPILOGUE

	ret

	.size	kernel_dgemv_t_4_vs_lib4, .-kernel_dgemv_t_4_vs_lib4





	// read-only data
	.section	.rodata.cst32,"aM",@progbits,32

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

	.align 32
.LC04: // { 1.0 1.0 1.0 1.0 }
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248



	.section	.note.GNU-stack,"",@progbits

