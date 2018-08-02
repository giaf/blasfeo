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
// xmm0  <- [z0 z1]_a
// xmm1  <- [z2 z3]_b
// xmm2  <- [z0 z1]_a
// xmm3  <- [z2 z3]_b

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
	
	movddup		0(%ecx), %xmm5
	movapd		0(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm0
	movapd		16(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1

	movddup		8(%ecx), %xmm5
	movapd		32(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm2
	movapd		48(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm3

	movddup		16(%ecx), %xmm5
	movapd		64(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm0
	movapd		80(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1

	movddup		24(%ecx), %xmm5
	movapd		96(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm2
	movapd		112(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm3

	subl	$4, %eax
	addl	$128, %ebx
	addl	$32, %ecx
	
	cmpl	$3, %eax
	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	movddup		0(%ecx), %xmm5
	movapd		0(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm0
	movapd		16(%ebx), %xmm4
	mulpd		%xmm5, %xmm4
	addpd		%xmm4, %xmm1

	subl	$1, %eax
	addl	$32, %ebx
	addl	$8, %ecx

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
// xmm0  <- [z0a z0b]
// xmm1  <- [z2a z2b]
// xmm2  <- [z1a z1b]
// xmm3  <- [z3a z3b]

//
// output arguments:

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

	movupd		0(%edx), %xmm4

	movapd		0(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0

	movapd		32(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm2

	movapd		64(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm1

	movapd		96(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3

	movupd		16(%edx), %xmm4

	movapd		16(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm0

	movapd		48(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm2

	movapd		80(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm1

	movapd		112(%ebx), %xmm5
	mulpd		%xmm4, %xmm5
	addpd		%xmm5, %xmm3

	subl	$4, %eax
	addl	%ecx, %ebx
	addl	$32, %edx
	
	cmpl	$3, %eax
	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	movsd		0(%edx), %xmm4

	movsd		0(%ebx), %xmm5
	mulsd		%xmm4, %xmm5
	addsd		%xmm5, %xmm0

	movsd		32(%ebx), %xmm5
	mulsd		%xmm4, %xmm5
	addsd		%xmm5, %xmm2

	movsd		64(%ebx), %xmm5
	mulsd		%xmm4, %xmm5
	addsd		%xmm5, %xmm1

	movsd		96(%ebx), %xmm5
	mulsd		%xmm4, %xmm5
	addsd		%xmm5, %xmm3
	
	subl	$1, %eax
	addl	$8, %ebx
	addl	$8, %edx
	
	cmpl	$0, %eax
	jg		0b // main loop 

	
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
// xmm0  <- [z0 z1]_a
// xmm1  <- [z2 z3]_b
// xmm2  <- [z0 z1]_a
// xmm3  <- [z2 z3]_b
//
// output arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// xmm0 <- [z0 z1]
// xmm1 <- [z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	.align 16
	.type inner_blend_n_scale_ab_4_lib4, @function
inner_blend_n_scale_ab_4_lib4:
#endif

	// reduction
	addpd		%xmm2, %xmm0
	addpd		%xmm3, %xmm1

	// alpha
	movddup		0(%eax), %xmm7
	mulpd		%xmm7, %xmm0
	mulpd		%xmm7, %xmm1

	// beta
	movddup		0(%ebx), %xmm7
	movupd		0(%ecx), %xmm6
	mulpd		%xmm7, %xmm6
	addpd		%xmm6, %xmm0
	movupd		16(%ecx), %xmm6
	mulpd		%xmm7, %xmm6
	addpd		%xmm6, %xmm1

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
// xmm0  <- [z0a z0b]
// xmm1  <- [z2a z2b]
// xmm2  <- [z1a z1b]
// xmm3  <- [z3a z3b]
//
// output arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// xmm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	.align 16
	.type inner_blend_t_scale_ab_4_lib4, @function
inner_blend_t_scale_ab_4_lib4:
#endif

	// reduction
	haddpd		%xmm2, %xmm0
	haddpd		%xmm3, %xmm1

	// alpha
	movddup		0(%eax), %xmm7
	mulpd		%xmm7, %xmm0
	mulpd		%xmm7, %xmm1

	// beta
	movddup		0(%ebx), %xmm7
	movupd		0(%ecx), %xmm6
	mulpd		%xmm7, %xmm6
	addpd		%xmm6, %xmm0
	movupd		16(%ecx), %xmm6
	mulpd		%xmm7, %xmm6
	addpd		%xmm6, %xmm1
	
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
// xmm0 <- [z0 z1]
// xmm1 <- [z2 z3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.align 16
	.type inner_store_4_lib4, @function
inner_store_4_lib4:
#endif
	
	movupd		%xmm0,  0(%eax)
	movupd		%xmm1, 16(%eax)
	
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
// xmm0  <- [z0 z1 z2 z3]
//
// output arguments:
// eax   <- D
// ebx   <- km
// xmm0  <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_VS_LIB4
#else
	.align 16
	.type inner_store_4_vs_lib4, @function
inner_store_4_vs_lib4:
#endif
	
	cmpl	$0, %ebx
	jle		0f // return

	movsd 	%xmm0, 0(%eax)

	cmpl	$1, %ebx
	jle		0f // return

	movhpd 	%xmm0, 8(%eax)

	cmpl	$2, %ebx
	jle		0f // return

	movsd 	%xmm1, 16(%eax)

	cmpl	$3, %ebx
	jle		0f // return

	movhpd 	%xmm1, 24(%eax)

0:

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

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


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

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


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

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


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

	xorpd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm3


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





	.section	.note.GNU-stack,"",@progbits

