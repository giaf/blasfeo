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
// xmm0  <- [z0 z1 z2 z3]_a
// xmm1  <- [z0 z1 z2 z3]_b
// xmm2  <- [z0 z1 z2 z3]_c
// xmm3  <- [z0 z1 z2 z3]_d

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
	
	vmovaps	0(%ebx), %xmm4
	vbroadcastss	0(%ecx), %xmm5
	vmulps	%xmm4, %xmm5, %xmm7
	vaddps	%xmm0, %xmm7, %xmm0
	
	vmovaps	16(%ebx), %xmm4
	vbroadcastss	4(%ecx), %xmm5
	vmulps	%xmm4, %xmm5, %xmm7
	vaddps	%xmm1, %xmm7, %xmm1
	
	vmovaps	32(%ebx), %xmm4
	vbroadcastss	8(%ecx), %xmm5
	vmulps	%xmm4, %xmm5, %xmm7
	vaddps	%xmm2, %xmm7, %xmm2

	vmovaps	48(%ebx), %xmm4
	vbroadcastss	12(%ecx), %xmm5
	vmulps	%xmm4, %xmm5, %xmm7
	vaddps	%xmm3, %xmm7, %xmm3
	
	subl	$4, %eax
	addl	$64, %ebx
	addl	$16, %ecx
	
	cmpl	$3, %eax

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	vmovaps	0(%ebx), %xmm4
	vbroadcastss	0(%ecx), %xmm5
	vmulps	%xmm4, %xmm5, %xmm7
	vaddps	%xmm0, %xmm7, %xmm0
	
	addl	$16, %ebx
	addl	$4, %ecx
	
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
// xmm0  <- [z0a z0b z0c z0d]
// xmm1  <- [z1a z1b z1c z1d]
// xmm2  <- [z2a z2b z2c z2d]
// xmm3  <- [z3a z3b z3c z3d]

//
// output arguments:
// eax  <- 0
// ebx   <- A+4*k*sizeof(double)
// ecx   <- bs*sda*sizeof(double) = 32*sda
// edx   <- x+k*sizeof(double)
// xmm0  <- [z0a z0b z0c z0d]
// xmm1  <- [z1a z1b z1c z1d]
// xmm2  <- [z2a z2b z2c z2d]
// xmm3  <- [z3a z3b z3c z3d]

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
	
	vmovups	0(%edx), %xmm4

	vmovaps	0(%ebx), %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm0, %xmm7, %xmm0
	
	subl	$4, %eax

	vmovaps	16(%ebx), %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm1, %xmm7, %xmm1
	
	vmovaps	32(%ebx), %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm2, %xmm7, %xmm2

	vmovaps	48(%ebx), %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm3, %xmm7, %xmm3
	
	addl	%ecx, %ebx
	addl	$16, %edx
	
	cmpl	$3, %eax

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %eax
	jle		2f // return

0: // clean-up
	
	vcvtsi2ss	%eax, %xmm6, %xmm6
	vmovups		.LC00, %xmm5
	vshufps		$0x00, %xmm7, %xmm7, %xmm7
	vsubps		%xmm6, %xmm5, %xmm6

	vmaskmovpd	0(%edx), %xmm6, %xmm4

	vmaskmovpd	0(%ebx), %xmm6, %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm0, %xmm7, %xmm0
	
	vmaskmovpd	16(%ebx), %xmm6, %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm1, %xmm7, %xmm1
	
	vmaskmovpd	32(%ebx), %xmm6, %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm2, %xmm7, %xmm2

	vmaskmovpd	48(%ebx), %xmm6, %xmm5
	vmulps	%xmm5, %xmm4, %xmm7
	vaddps	%xmm3, %xmm7, %xmm3
		
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
// xmm0 <- [z0 z1 z2 z3]_a
// xmm1 <- [z0 z1 z2 z3]_b
// xmm2 <- [z0 z1 z2 z3]_c
// xmm3 <- [z0 z1 z2 z3]_d
//
// output arguments:
// eax  <- alpha
// ebx  <- beta
// ecx  <- y
// xmm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	.align 16
	.type inner_blend_n_scale_ab_4_lib4, @function
inner_blend_n_scale_ab_4_lib4:
#endif

	// reduction
	vaddps	%xmm0, %xmm1, %xmm0
	vaddps	%xmm2, %xmm3, %xmm2
	vaddps	%xmm0, %xmm2, %xmm0

	// alpha
	vbroadcastss	0(%eax), %xmm7
	vmulps	%xmm0, %xmm7, %xmm0

	// beta
	vbroadcastss	0(%ebx), %xmm7
	vmovups		0(%ecx), %xmm6
	vmulps		%xmm7, %xmm6, %xmm6
	vaddps		%xmm0, %xmm6, %xmm0

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
// xmm0 <- [z0a z0b z0c z0d]
// xmm1 <- [z1a z1b z1c z1d]
// xmm2 <- [z2a z2b z2c z2d]
// xmm3 <- [z3a z3b z3c z3d]
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
	vhaddps	%xmm1, %xmm0, %xmm0
	vhaddps	%xmm3, %xmm2, %xmm2
	vhaddps	%xmm2, %xmm0, %xmm0

	// alpha
	vbroadcastss	0(%eax), %xmm7
	vmulps	%xmm0, %xmm7, %xmm0

	// beta
	vbroadcastss	0(%ebx), %xmm7
	vmovups		0(%ecx), %xmm6
	vmulps		%xmm7, %xmm6, %xmm6
	vaddps		%xmm0, %xmm6, %xmm0
	
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
// xmm0 <- [z0 z1 z2 z3]
//
// output arguments:

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.align 16
	.type inner_store_4_lib4, @function
inner_store_4_lib4:
#endif
	
	vmovups %xmm0,  0(%eax)
	
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
	
	vcvtsi2ss	%ebx, %xmm7, %xmm7
	vmovups		.LC00, %xmm6
	vshufps		$0x00, %xmm7, %xmm7, %xmm7
	vsubps		%xmm7, %xmm6, %xmm7

	vmaskmovpd	%xmm0, %xmm7,  0(%eax)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4_vs_lib4, .-inner_store_4_vs_lib4
#endif
#endif





//                            1      2              3          4          5             6          7
// void kernel_sgemv_n_4_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z);

	.align 16
	.globl kernel_sgemv_n_4_lib4
	.type kernel_sgemv_n_4_lib4, @function
kernel_sgemv_n_4_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, %xmm1
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm0, %xmm3


	// call inner sgemv kernel n

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

	.size	kernel_sgemv_n_4_lib4, .-kernel_sgemv_n_4_lib4






//                               1      2              3          4          5             6          7          8
// void kernel_sgemv_n_4_vs_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z, int k1);

	.align 16
	.globl kernel_sgemv_n_4_vs_lib4
	.type kernel_sgemv_n_4_vs_lib4, @function
kernel_sgemv_n_4_vs_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, %xmm1
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm0, %xmm3


	// call inner sgemv kernel n

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

	.size	kernel_sgemv_n_4_vs_lib4, .-kernel_sgemv_n_4_vs_lib4






//                            1      2              3          4        5          6             7         8
// void kernel_sgemv_t_4_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z);

	.align 16
	.globl kernel_sgemv_t_4_lib4
	.type kernel_sgemv_t_4_lib4, @function
kernel_sgemv_t_4_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, %xmm1
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm0, %xmm3


	// call inner sgemv kernel n

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

	.size	kernel_sgemv_t_4_lib4, .-kernel_sgemv_t_4_lib4





//                               1      2              3          4        5          6             7         8           9
// void kernel_sgemv_t_4_vs_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z, int k1);

	.align 16
	.globl kernel_sgemv_t_4_vs_lib4
	.type kernel_sgemv_t_4_vs_lib4, @function
kernel_sgemv_t_4_vs_lib4:
	
	PROLOGUE

	// zero accumulation registers

	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, %xmm1
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm0, %xmm3


	// call inner sgemv kernel n

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

	.size	kernel_sgemv_t_4_vs_lib4, .-kernel_sgemv_t_4_vs_lib4





	// read-only data
	.section	.rodata.cst32,"aM",@progbits,32
	.align 16
.LC00: // { 3.5 2.5 1.5 0.5 }
	.long	1056964608
	.long	1069547520
	.long	1075838976
	.long	1080033280

	.section	.note.GNU-stack,"",@progbits

