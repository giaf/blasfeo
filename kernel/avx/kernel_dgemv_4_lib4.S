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
// r12   <- x
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
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
// r12   <- x+k*sizeof(double)
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemv_add_n_4_lib4, @function
inner_kernel_dgemv_add_n_4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemv_add_n_4_lib4:
#endif
#endif
	
	cmpl	$0, %r10d
	jle		2f // return

	cmpl	$4, %r10d
	jl		0f // clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	vmovapd	0(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	
	subl	$4, %r10d

	vmovapd	32(%r11), %ymm8
	vbroadcastsd	8(%r12), %ymm12
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	
	vmovapd	64(%r11), %ymm8
	vbroadcastsd	16(%r12), %ymm12
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	vmovapd	96(%r11), %ymm8
	vbroadcastsd	24(%r12), %ymm12
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	
	addq	$128, %r11
	addq	$32, %r12
	
	cmpl	$3, %r10d

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

0: // clean-up
	
	vmovapd	0(%r11), %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	
	addq	$32, %r11
	addq	$8, %r12
	
	subl	$1, %r10d
	cmpl	$0, %r10d

	jg		0b // clean

2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemv_add_n_4_lib4, .-inner_kernel_dgemv_add_n_4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- bs*sda*sizeof(double) = 32*sda
// r13   <- x
// ymm0  <- [z0a z0b z0c z0d]
// ymm1  <- [z1a z1b z1c z1d]
// ymm2  <- [z2a z2b z2c z2d]
// ymm3  <- [z3a z3b z3c z3d]
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
// r12   <- bs*sda*sizeof(double) = 32*sda
// r13   <- x+k*sizeof(double)
// ymm0  <- [z0a z0b z0c z0d]
// ymm1  <- [z1a z1b z1c z1d]
// ymm2  <- [z2a z2b z2c z2d]
// ymm3  <- [z3a z3b z3c z3d]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=2
	.macro INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dgemv_add_t_4_lib4, @function
inner_kernel_dgemv_add_t_4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dgemv_add_t_4_lib4:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	cmpl	$4, %r10d
	jl		0f // clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	vmovapd	0(%r13), %ymm12

	vmovapd	0(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	
	subl	$4, %r10d

	vmovapd	32(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	
	vmovapd	64(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	vmovapd	96(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	
	addq	%r12, %r11
	addq	$32, %r13
	
	cmpl	$3, %r10d

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

0: // clean-up
	
	vcvtsi2sd	%r10d, %xmm14, %xmm14
#if defined(OS_LINUX)
	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
	vmovupd		LC02(%rip), %ymm13
#endif
	vmovddup	%xmm14, %xmm14
	vinsertf128	$1, %xmm14, %ymm14, %ymm14
	vsubpd		%ymm14, %ymm13, %ymm14

	vmaskmovpd	0(%r13), %ymm14, %ymm12

	vmovapd	0(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	
	vmovapd	32(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	
	vmovapd	64(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	vmovapd	96(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
		
	sall	$3, %r10d
	movslq	%r10d, %r10
	addq	%r10, %r11
	addq	%r10, %r13
	xorl	%r10d, %r10d
	
	
2: // return

#if MACRO_LEVEL>=2
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dgemv_add_t_4_lib4, .-inner_kernel_dgemv_add_t_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for ta==n, scale for generic alpha and beta
//
// input arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// ymm0 <- [z0 z1 z2 z3]_a
// ymm1 <- [z0 z1 z2 z3]_b
// ymm2 <- [z0 z1 z2 z3]_c
// ymm3 <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// ymm0 <- [z0 z1 z2 z3]
// ymm1 <- dirty
// ymm2 <- dirty
// ymm3 <- dirty
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_n_scale_ab_4_lib4, @function
inner_blend_n_scale_ab_4_lib4:
#elif defined(OS_MAC)
_inner_blend_n_scale_ab_4_lib4:
#endif
#endif

	// reduction
	vaddpd	%ymm0, %ymm1, %ymm0
	vaddpd	%ymm2, %ymm3, %ymm2
	vaddpd	%ymm0, %ymm2, %ymm0

	// alpha
	vbroadcastsd	0(%r10), %ymm15
	vmulpd	%ymm0, %ymm15, %ymm0

	// beta
	vbroadcastsd	0(%r11), %ymm15
	vmovupd		0(%r12), %ymm14
	vmulpd		%ymm15, %ymm14, %ymm14
	vaddpd		%ymm0, %ymm14, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blend_n_scale_ab_4_lib4, .-inner_blend_n_scale_ab_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blend for ta==t, scale for generic alpha and beta
//
// input arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// ymm0 <- [z0a z0b z0c z0d]
// ymm1 <- [z1a z1b z1c z1d]
// ymm2 <- [z2a z2b z2c z2d]
// ymm3 <- [z3a z3b z3c z3d]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10  <- alpha
// r11  <- beta
// r12  <- y
// ymm0 <- [z0 z1 z2 z3]
// ymm1 <- dirty
// ymm2 <- dirty
// ymm3 <- dirty
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blend_t_scale_ab_4_lib4, @function
inner_blend_t_scale_ab_4_lib4:
#elif defined(OS_MAC)
_inner_blend_t_scale_ab_4_lib4:
#endif
#endif

	// reduction
	vhaddpd	%ymm1, %ymm0, %ymm0
	vhaddpd	%ymm3, %ymm2, %ymm2
	vperm2f128	$0x2, %ymm0, %ymm2, %ymm1
	vperm2f128	$0x13, %ymm0, %ymm2, %ymm0
	vaddpd	%ymm0, %ymm1, %ymm0

	// alpha
	vbroadcastsd	0(%r10), %ymm15
	vmulpd	%ymm0, %ymm15, %ymm0

	// beta
	vbroadcastsd	0(%r11), %ymm15
	vmovupd		0(%r12), %ymm14
	vmulpd		%ymm15, %ymm14, %ymm14
	vaddpd		%ymm0, %ymm14, %ymm0
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blend_t_scale_ab_4_lib4, .-inner_blend_t_scale_ab_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blender for ta==n
//
// input arguments:
// r10d <- alg
// r11   <- y
// ymm0 <- [z0 z1 z2 z3]_a
// ymm1 <- [z0 z1 z2 z3]_b
// ymm2 <- [z0 z1 z2 z3]_c
// ymm3 <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d <- alg
// r11   <- y
// ymm0 <- [z0 z1 z2 z3]
// ymm1 <- dirty
// ymm2 <- dirty
// ymm3 <- dirty
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLENDER_N_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blender_n_4_lib4, @function
inner_blender_n_4_lib4:
#elif defined(OS_MAC)
_inner_blender_n_4_lib4:
#endif
#endif

	// reduction
	vaddpd	%ymm0, %ymm1, %ymm0
	vaddpd	%ymm2, %ymm3, %ymm2
	vaddpd	%ymm0, %ymm2, %ymm0

	cmpl	$0, %r10d // alg
	je		0f // return

	cmpl	$1, %r10d // alg
	jne		1f // alg==-1

	// alg==1
	vmovupd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0

	jmp		0f // return

1:

	// alg==-1
	vmovupd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0

0: // return
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blender_n_4_lib4, .-inner_blender_n_4_lib4
#endif
#endif





// common inner routine with file scope
//
// blender for ta==t
//
// input arguments:
// r10d <- alg
// r11   <- y
// ymm0 <- [z0a z0b z0c z0d]
// ymm1 <- [z1a z1b z1c z1d]
// ymm2 <- [z2a z2b z2c z2d]
// ymm3 <- [z3a z3b z3c z3d]
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10d <- alg
// r11   <- y
// ymm0 <- [z0 z1 z2 z3]
// ymm1 <- dirty
// ymm2 <- dirty
// ymm3 <- dirty
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_BLENDER_T_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_blender_t_4_lib4, @function
inner_blender_t_4_lib4:
#elif defined(OS_MAC)
_inner_blender_t_4_lib4:
#endif
#endif

	// reduction
	vhaddpd	%ymm1, %ymm0, %ymm0
	vhaddpd	%ymm3, %ymm2, %ymm2
	vperm2f128	$0x2, %ymm0, %ymm2, %ymm1
	vperm2f128	$0x13, %ymm0, %ymm2, %ymm0
	vaddpd	%ymm0, %ymm1, %ymm0

	cmpl	$0, %r10d // alg
	je		0f // return

	cmpl	$1, %r10d // alg
	jne		1f // alg==-1

	// alg==1
	vmovupd		0(%r11), %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0

	jmp		0f // return

1:

	// alg==-1
	vmovupd		0(%r11), %ymm15
	vsubpd		%ymm0, %ymm15, %ymm0

0: // return
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret
	
#if defined(OS_LINUX)
	.size	inner_blender_t_4_lib4, .-inner_blender_t_4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution with vector RHS
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LN_INV_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_ln_inv_4_lib4, @function
inner_edge_dtrsv_ln_inv_4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_ln_inv_4_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0

	vmovapd			0(%r10), %ymm13
	vblendpd		$0x1, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	vblendpd		$0x2, %ymm1, %ymm0, %ymm0

	vmovapd			32(%r10), %ymm13
	vblendpd		$0x3, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x3, %ymm0, %ymm12
	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	vblendpd		$0x4, %ymm1, %ymm0, %ymm0

	vmovapd			64(%r10), %ymm13
	vblendpd		$0x7, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	vblendpd		$0x8, %ymm1, %ymm0, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_ln_inv_4_lib4, .-inner_edge_dtrsv_ln_inv_4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution with vector RHS, variable size version
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12d <- kn
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12d <- kn
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LN_INV_4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_ln_inv_4_vs_lib4, @function
inner_edge_dtrsv_ln_inv_4_vs_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_ln_inv_4_vs_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vbroadcastsd	0(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	cmpl			$2, %r12d
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0
	vmovapd			0(%r10), %ymm13
	vblendpd		$0x1, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	jl				0f // ret

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	cmpl			$3, %r12d
	vblendpd		$0x2, %ymm1, %ymm0, %ymm0
	vmovapd			32(%r10), %ymm13
	vblendpd		$0x3, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x3, %ymm0, %ymm12
	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	jl				0f // ret

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	cmpl			$4, %r12d
	vblendpd		$0x4, %ymm1, %ymm0, %ymm0
	vmovapd			64(%r10), %ymm13
	vblendpd		$0x7, %ymm14, %ymm13, %ymm13
	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm13, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	jl				0f // ret

	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm0, %ymm12, %ymm1
	vblendpd		$0x8, %ymm1, %ymm0, %ymm0

	// return
0:

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_ln_inv_4_vs_lib4, .-inner_edge_dtrsv_ln_inv_4_vs_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution with vector RHS
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LT_INV_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_lt_inv_4_lib4, @function
inner_edge_dtrsv_lt_inv_4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_lt_inv_4_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			16(%r10), %xmm12
	vmovapd			48(%r10), %xmm13
	vunpcklpd		%xmm13, %xmm12, %xmm9
	vblendpd		$0xc, %ymm14, %ymm9, %ymm9
	vunpckhpd		%xmm13, %xmm12, %xmm10
	vmovsd			8(%r10), %xmm8
	vblendpd		$0xe, %ymm14, %ymm8, %ymm8
	vmovsd			88(%r10), %xmm11
	vinsertf128		$0x1, %xmm11, %ymm10, %ymm10
	vblendpd		$0x8, %ymm14, %ymm10, %ymm10

	vbroadcastsd	24(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x8, %ymm1, %ymm0, %ymm0

	vpermilpd		$0xf, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm10, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x4, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x2, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x3, %ymm0, %ymm12
//	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
//	vbroadcastsd	8(%r11), %ymm12
	vmovsd			0(%r11), %xmm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_lt_inv_4_lib4, .-inner_edge_dtrsv_lt_inv_4_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution with vector RHS
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LT_INV_3_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_lt_inv_3_lib4, @function
inner_edge_dtrsv_lt_inv_3_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_lt_inv_3_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	cmpl			$4, %r12d

	vmovapd			16(%r10), %xmm12
	vmovapd			48(%r10), %xmm13
	vunpcklpd		%xmm13, %xmm12, %xmm9
	vblendpd		$0xc, %ymm14, %ymm9, %ymm9
	vunpckhpd		%xmm13, %xmm12, %xmm10
	vmovsd			8(%r10), %xmm8
	vblendpd		$0xe, %ymm14, %ymm8, %ymm8
	vmovsd			88(%r10), %xmm11
	vinsertf128		$0x1, %xmm11, %ymm10, %ymm10
	vblendpd		$0x8, %ymm14, %ymm10, %ymm10

//	vbroadcastsd	24(%r11), %ymm12
//	vmulpd			%ymm12, %ymm0, %ymm1
//	vblendpd		$0x8, %ymm1, %ymm0, %ymm0

	vmovapd			0(%r13), %ymm12
	vblendpd		$0x8, %ymm12, %ymm0, %ymm0
	
	jl				0f

	vpermilpd		$0xf, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm10, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	vbroadcastsd	16(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x4, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x2, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x3, %ymm0, %ymm12
//	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
//	vbroadcastsd	8(%r11), %ymm12

	vmovsd			0(%r11), %xmm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_lt_inv_3_lib4, .-inner_edge_dtrsv_lt_inv_3_lib4
#endif
#endif





// common inner routine with file scope
//
// triangular substitution with vector RHS
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LT_INV_2_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_lt_inv_2_lib4, @function
inner_edge_dtrsv_lt_inv_2_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_lt_inv_2_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	cmpl			$3, %r12d

	vmovapd			16(%r10), %xmm12
	vmovapd			48(%r10), %xmm13
	vunpcklpd		%xmm13, %xmm12, %xmm9
	vblendpd		$0xc, %ymm14, %ymm9, %ymm9
	vunpckhpd		%xmm13, %xmm12, %xmm10
	vmovsd			8(%r10), %xmm8
	vblendpd		$0xe, %ymm14, %ymm8, %ymm8
//	vmovsd			88(%r10), %xmm11
//	vinsertf128		$0x1, %xmm11, %ymm10, %ymm10
//	vblendpd		$0x8, %ymm14, %ymm10, %ymm10
	vblendpd		$0xc, %ymm14, %ymm10, %ymm10

//	vbroadcastsd	24(%r11), %ymm12
//	vmulpd			%ymm12, %ymm0, %ymm1
//	vblendpd		$0x8, %ymm1, %ymm0, %ymm0

	vmovapd			0(%r13), %ymm12
	vblendpd		$0xc, %ymm12, %ymm0, %ymm0
	
	je				0f
	jl				1f

	vpermilpd		$0xf, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm10, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

//	vbroadcastsd	16(%r11), %ymm12
//	vmulpd			%ymm12, %ymm0, %ymm1
//	vblendpd		$0x4, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

1:

	vbroadcastsd	8(%r11), %ymm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x2, %ymm1, %ymm0, %ymm0

	vpermilpd		$0x3, %ymm0, %ymm12
//	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
//	vbroadcastsd	8(%r11), %ymm12

	vmovsd			0(%r11), %xmm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_lt_inv_2_lib4, .-inner_edge_dtrsv_lt_inv_2_lib4
#endif
#endif





// common inner routine with file scope 
//
// triangular substitution with vector RHS
//
// input arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty
//
// output arguments:
// r10  <- E
// r11  <- inv_diag_E
// r12  <- k
// r13  <- x
// ymm0 <- [z0 z1 z2 z3]
// ymm12 <- dirty
// ymm13 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRSV_LT_INV_1_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrsv_lt_inv_1_lib4, @function
inner_edge_dtrsv_lt_inv_1_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrsv_lt_inv_1_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			0(%r13), %ymm12
	vblendpd		$0xe, %ymm12, %ymm0, %ymm0
	
	cmpl			$3, %r12d
	je				0f

	cmpl			$2, %r12d
	je				1f
	jl				2f

	vmovsd			24(%r10), %xmm10
	vblendpd		$0xe, %ymm14, %ymm10, %ymm10
	vpermilpd		$0xf, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm10, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

0:

	vmovsd			16(%r10), %xmm9
	vblendpd		$0xe, %ymm14, %ymm9, %ymm9
	vpermilpd		$0x0, %ymm0, %ymm12
	vperm2f128		$0x11, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm9, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0

1:

	vmovsd			8(%r10), %xmm8
	vblendpd		$0xe, %ymm14, %ymm8, %ymm8
	vpermilpd		$0x3, %ymm0, %ymm12
//	vperm2f128		$0x00, %ymm12, %ymm12, %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vsubpd			%ymm15, %ymm0, %ymm0
//	vbroadcastsd	8(%r11), %ymm12

2:

	vmovsd			0(%r11), %xmm12
	vmulpd			%ymm12, %ymm0, %ymm1
	vblendpd		$0x1, %ymm1, %ymm0, %ymm0

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrsv_lt_inv_1_lib4, .-inner_edge_dtrsv_lt_inv_1_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- x
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
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
// r10d  <- k-4
// r11   <- A+4*4*sizeof(double)
// r12   <- x+4*sizeof(double)
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_EDGE_DTRMV_UN_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_edge_dtrmv_un_4_lib4, @function
inner_edge_dtrmv_un_4_lib4:
#elif defined(OS_MAC)
_inner_edge_dtrmv_un_4_lib4:
#endif
#endif
	
	vxorpd			%ymm14, %ymm14, %ymm14

	vmovapd			0(%r11), %ymm8
	vblendpd		$0x1, %ymm8, %ymm14, %ymm8
	vbroadcastsd	0(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm0, %ymm15, %ymm0
	
	subl			$4, %r10d

	vmovapd			32(%r11), %ymm8
	vblendpd		$0x3, %ymm8, %ymm14, %ymm8
	vbroadcastsd	8(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm1, %ymm15, %ymm1
	
	vmovapd			64(%r11), %ymm8
	vblendpd		$0x7, %ymm8, %ymm14, %ymm8
	vbroadcastsd	16(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm2, %ymm15, %ymm2

	vmovapd			96(%r11), %ymm8
	vbroadcastsd	24(%r12), %ymm12
	vmulpd			%ymm8, %ymm12, %ymm15
	vaddpd			%ymm3, %ymm15, %ymm3
	
	addq			$128, %r11
	addq			$32, %r12
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_edge_dtrmv_un_4_lib4, .-inner_edge_dtrmv_un_4_lib4
#endif
#endif





// common inner routine with file scope
//
// input arguments:
// r10d  <- k
// r11   <- A
// r12   <- bs*sda*sizeof(double) = 32*sda
// r13   <- x
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
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
// r12   <- bs*sda*sizeof(double) = 32*sda
// r13   <- x+k*sizeof(double)
// ymm0  <- [z0 z1 z2 z3]_a
// ymm1  <- [z0 z1 z2 z3]_b
// ymm2  <- [z0 z1 z2 z3]_c
// ymm3  <- [z0 z1 z2 z3]_d
// ymm8  <- dirty
// ymm9  <- dirty
// ymm10 <- dirty
// ymm11 <- dirty
// ymm12 <- dirty
// ymm13 <- dirty
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_KERNEL_DTRMV_UT_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_kernel_dtrmv_ut_4_lib4, @function
inner_kernel_dtrmv_ut_4_lib4:
#elif defined(OS_MAC)
_inner_kernel_dtrmv_ut_4_lib4:
#endif
#endif

	cmpl	$0, %r10d
	jle		2f // return

	cmpl	$4, %r10d
	jle		0f // clean-up loop

	// main loop
	.p2align 3
1: // main loop
	
	vmovapd	0(%r13), %ymm12

	vmovapd	0(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm0, %ymm15, %ymm0
	
	subl	$4, %r10d

	vmovapd	32(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm1, %ymm15, %ymm1
	
	vmovapd	64(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm2, %ymm15, %ymm2

	vmovapd	96(%r11), %ymm8
	vmulpd	%ymm8, %ymm12, %ymm15
	vaddpd	%ymm3, %ymm15, %ymm3
	
	addq	%r12, %r11
	addq	$32, %r13
	
	cmpl	$4, %r10d

	jg		1b // main loop 


	// consider clean-up
	cmpl	$0, %r10d
	jle		2f // return

0: // clean-up
	
//	vcvtsi2sd	%r10d, %xmm14, %xmm14
#if defined(OS_LINUX)
//	vmovupd		.LC02(%rip), %ymm13
#elif defined(OS_MAC)
//	vmovupd		LC02(%rip), %ymm13
#endif
//	vmovddup	%xmm14, %xmm14
//	vinsertf128	$1, %xmm14, %ymm14, %ymm14
//	vsubpd		%ymm14, %ymm13, %ymm14
//
//	vmaskmovpd	0(%r13), %ymm14, %ymm12

	vmovapd		0(%r13), %ymm12

	vxorpd		%ymm14, %ymm14, %ymm14

	vmovapd		0(%r11), %ymm8
	vblendpd	$0x1, %ymm8, %ymm14, %ymm8
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm0, %ymm15, %ymm0
	
	vmovapd	32(%r11), %ymm8
	vblendpd	$0x3, %ymm8, %ymm14, %ymm8
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm1, %ymm15, %ymm1
	
	vmovapd		64(%r11), %ymm8
	vblendpd	$0x7, %ymm8, %ymm14, %ymm8
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm2, %ymm15, %ymm2

	vmovapd		96(%r11), %ymm8
	vmulpd		%ymm8, %ymm12, %ymm15
	vaddpd		%ymm3, %ymm15, %ymm3
		
	sall		$3, %r10d
	movslq		%r10d, %r10
	addq		%r10, %r11
	addq		%r10, %r13
	xorl		%r10d, %r10d
	
	
2: // return

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_kernel_dtrmv_ut_4_lib4, .-inner_kernel_dtrmv_ut_4_lib4
#endif
#endif





// common inner routine with file scope
//
// store 
//
// input arguments:
// r10  <- z
// ymm0 <- [z0 z1 z2 z3]
//
// output arguments:
// r10  <- z
// ymm0 <- [z0 z1 z2 z3]

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4_lib4, @function
inner_store_4_lib4:
#elif defined(OS_MAC)
_inner_store_4_lib4:
#endif
#endif
	
	vmovupd %ymm0,  0(%r10)
	
#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4_lib4, .-inner_store_4_lib4
#endif
#endif





// common inner routine with file scope
//
// store vs
//
// input arguments:
// r10   <- D
// r11d   <- km
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty
//
// output arguments:
// r10   <- D
// r11d   <- km
// ymm0  <- [z0 z1 z2 z3]
// ymm14 <- dirty
// ymm15 <- dirty

#if MACRO_LEVEL>=1
	.macro INNER_STORE_4_VS_LIB4
#else
	.p2align 4,,15
#if defined(OS_LINUX)
	.type inner_store_4_vs_lib4, @function
inner_store_4_vs_lib4:
#elif defined(OS_MAC)
_inner_store_4_vs_lib4:
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

	vmaskmovpd	%ymm0, %ymm15,  0(%r10)

#if MACRO_LEVEL>=1
	.endm
#else
	ret

#if defined(OS_LINUX)
	.size	inner_store_4_vs_lib4, .-inner_store_4_vs_lib4
#endif
#endif





//                            rdi    rsi            rdx        rcx        r8            r9         rsp+8
// void kernel_dgemv_n_4_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemv_n_4_lib4
	.type kernel_dgemv_n_4_lib4, @function
kernel_dgemv_n_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemv_n_4_lib4
_kernel_dgemv_n_4_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif


	// call inner blend n scale ab

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11   // beta
	movq	%r9, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	callq _inner_blend_n_scale_ab_4_lib4
#endif


	// store

	movq	STACKSIZE + 8(%rsp), %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dgemv_n_4_lib4, .-kernel_dgemv_n_4_lib4
#endif





//                               rdi    rsi            rdx        rcx        r8            r9         rsp+8      rsp+16
// void kernel_dgemv_n_4_vs_lib4(int k, double *alpha, double *A, double *x, double *beta, double *y, double *z, int km);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemv_n_4_vs_lib4
	.type kernel_dgemv_n_4_vs_lib4, @function
kernel_dgemv_n_4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemv_n_4_vs_lib4
_kernel_dgemv_n_4_vs_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movq	%rcx, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif


	// call inner blend n scale ab

	movq	%rsi, %r10 // alpha
	movq	%r8, %r11   // beta
	movq	%r9, %r12   // y

#if MACRO_LEVEL>=1
	INNER_BLEND_N_SCALE_AB_4_LIB4
#else
	callq _inner_blend_n_scale_ab_4_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 
	movl	STACKSIZE + 16(%rsp), %r11d // km 

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dgemv_n_4_vs_lib4, .-kernel_dgemv_n_4_vs_lib4
#endif





//                            rdi    rsi           rdx         rcx      r8         r9            rsp+8      rsp+16
// void kernel_dgemv_t_4_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemv_t_4_lib4
	.type kernel_dgemv_t_4_lib4, @function
kernel_dgemv_t_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemv_t_4_lib4
_kernel_dgemv_t_4_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%r8, %r13  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11   // beta
	movq	STACKSIZE + 8(%rsp), %r12 // y 

#if MACRO_LEVEL>=1
	INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	callq _inner_blend_t_scale_ab_4_lib4
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dgemv_t_4_lib4, .-kernel_dgemv_t_4_lib4
#endif





//                               rdi    rsi           rdx         rcx      r8         r9            rsp+8      rsp+16     rsp+24
// void kernel_dgemv_t_4_vs_lib4(int k, double *alpha, double *A, int sda, double *x, double *beta, double *y, double *z, int km);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dgemv_t_4_vs_lib4
	.type kernel_dgemv_t_4_vs_lib4, @function
kernel_dgemv_t_4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dgemv_t_4_vs_lib4
_kernel_dgemv_t_4_vs_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rdx, %r11  // A
	movl	%ecx, %r12d // sda
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%r8, %r13  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movq	%rsi, %r10 // alpha
	movq	%r9, %r11   // beta
	movq	STACKSIZE + 8(%rsp), %r12 // y 

#if MACRO_LEVEL>=1
	INNER_BLEND_T_SCALE_AB_4_LIB4
#else
	callq _inner_blend_t_scale_ab_4_lib4
#endif


	// store

	movq	STACKSIZE + 16(%rsp), %r10 // z 
	movl	STACKSIZE + 24(%rsp), %r11d // km 

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dgemv_t_4_vs_lib4, .-kernel_dgemv_t_4_vs_lib4
#endif





//                                 rdi    rsi        rdx                 rcx        r8         r9  
// void kernel_dtrsv_ln_inv_4_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_ln_inv_4_lib4
	.type kernel_dtrsv_ln_inv_4_lib4, @function
kernel_dtrsv_ln_inv_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_ln_inv_4_lib4
_kernel_dtrsv_ln_inv_4_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rcx, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif

	movq	%r11, %r13 // A+k*sizeof(double)


	// call inner blender n

	movl	$-1, %r10d // alg
	movq	%r8, %r11   // y

#if MACRO_LEVEL>=1
	INNER_BLENDER_N_4_LIB4
#else
	callq _inner_blender_n_4_lib4
#endif


	// solution

	movq	%r13, %r10 // A+k*sizeof(double)
	movq	%rdx, %r11 // inv_diag_A

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LN_INV_4_LIB4
#else
	callq _inner_edge_dtrsv_ln_inv_4_lib4
#endif


	// store

	movq	%r9, %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dtrsv_ln_inv_4_lib4, .-kernel_dtrsv_ln_inv_4_lib4
#endif





//                                    rdi    rsi        rdx                 rcx        r8         r9         rsp+8   rsp+16 
// void kernel_dtrsv_ln_inv_4_vs_lib4(int k, double *A, double *inv_diag_A, double *x, double *y, double *z, int km, int kn);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_ln_inv_4_vs_lib4
	.type kernel_dtrsv_ln_inv_4_vs_lib4, @function
kernel_dtrsv_ln_inv_4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_ln_inv_4_vs_lib4
_kernel_dtrsv_ln_inv_4_vs_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rcx, %r12  // x

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif

	movq	%r11, %r13


	// call inner blender n

	movl	$-1, %r10d // alg
	movq	%r8, %r11   // y

#if MACRO_LEVEL>=1
	INNER_BLENDER_N_4_LIB4
#else
	callq _inner_blender_n_4_lib4
#endif


	// solution

	movq	%r13, %r10 // A+k*sizeof(double)
	movq	%rdx, %r11 // inv_diag_A
	movl	STACKSIZE +  16(%rsp), %r12d // kn 

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LN_INV_4_VS_LIB4
#else
	callq _inner_edge_dtrsv_ln_inv_4_vs_lib4
#endif


	// store vs

	movq	%r9, %r10 // z 
	movl	STACKSIZE +  8(%rsp), %r11d // km 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dtrsv_ln_inv_4_vs_lib4, .-kernel_dtrsv_ln_inv_4_vs_lib4
#endif





//                                 rdi    rsi        rdx      rcx                 r8         r9         rsp+8   
// void kernel_dtrsv_lt_inv_4_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_lt_inv_4_lib4
	.type kernel_dtrsv_lt_inv_4_lib4, @function
kernel_dtrsv_lt_inv_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_lt_inv_4_lib4
_kernel_dtrsv_lt_inv_4_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	subl	$4, %r10d
	movq	%rsi, %r11 // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	addq	%r12, %r11 // A+4*sda*sizeof(double)
	movq	%r8, %r13 // x
	addq	$32, %r13 // x+4 

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	$-1, %r10d // alg
	movq	%r9, %r11 // y 

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// solution

	movq	%rsi, %r10 // A
	movq	%rcx, %r11 // inv_diag_A

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LT_INV_4_LIB4
#else
	callq _inner_edge_dtrsv_lt_inv_4_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dtrsv_lt_inv_4_lib4, .-kernel_dtrsv_lt_inv_4_lib4
#endif





//                                 rdi    rsi        rdx      rcx                 r8         r9         rsp+8   
// void kernel_dtrsv_lt_inv_3_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_lt_inv_3_lib4
	.type kernel_dtrsv_lt_inv_3_lib4, @function
kernel_dtrsv_lt_inv_3_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_lt_inv_3_lib4
_kernel_dtrsv_lt_inv_3_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	subl	$4, %r10d
	movq	%rsi, %r11 // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	addq	%r12, %r11 // A+4*sda*sizeof(double)
	movq	%r8, %r13 // x
	addq	$32, %r13 // x+4 

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	$-1, %r10d // alg
	movq	%r9, %r11 // y 

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// solution

	movq	%rsi, %r10 // A
	movq	%rcx, %r11 // inv_diag_A
	movl	%edi, %r12d // k
	movq	%r8, %r13 // x

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LT_INV_3_LIB4
#else
	callq _inner_edge_dtrsv_lt_inv_3_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 
	movl	$3, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dtrsv_lt_inv_3_lib4, .-kernel_dtrsv_lt_inv_3_lib4
#endif





//                                 rdi    rsi        rdx      rcx                 r8         r9         rsp+8 
// void kernel_dtrsv_lt_inv_2_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_lt_inv_2_lib4
	.type kernel_dtrsv_lt_inv_2_lib4, @function
kernel_dtrsv_lt_inv_2_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_lt_inv_2_lib4
_kernel_dtrsv_lt_inv_2_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	subl	$4, %r10d
	movq	%rsi, %r11 // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	addq	%r12, %r11 // A+4*sda*sizeof(double)
	movq	%r8, %r13 // x
	addq	$32, %r13 // x+4 

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	$-1, %r10d // alg
	movq	%r9, %r11 // y 

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// solution

	movq	%rsi, %r10 // A
	movq	%rcx, %r11 // inv_diag_A
	movl	%edi, %r12d // k
	movq	%r8, %r13 // x

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LT_INV_2_LIB4
#else
	callq _inner_edge_dtrsv_lt_inv_2_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 
	movl	$2, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dtrsv_lt_inv_2_lib4, .-kernel_dtrsv_lt_inv_2_lib4
#endif





//                                 rdi    rsi        rdx      rcx                 r8         r9         rsp+8 
// void kernel_dtrsv_lt_inv_1_lib4(int k, double *A, int sda, double *inv_diag_A, double *x, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrsv_lt_inv_1_lib4
	.type kernel_dtrsv_lt_inv_1_lib4, @function
kernel_dtrsv_lt_inv_1_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrsv_lt_inv_1_lib4
_kernel_dtrsv_lt_inv_1_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	subl	$4, %r10d
	movq	%rsi, %r11 // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	addq	%r12, %r11 // A+4*sda*sizeof(double)
	movq	%r8, %r13 // x
	addq	$32, %r13 // x+4 

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_T_4_LIB4
#else
	callq _inner_kernel_dgemv_add_t_4_lib4
#endif


	// call inner blender t

	movl	$-1, %r10d // alg
	movq	%r9, %r11 // y 

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// solution

	movq	%rsi, %r10 // A
	movq	%rcx, %r11 // inv_diag_A
	movl	%edi, %r12d // k
	movq	%r8, %r13 // x

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRSV_LT_INV_1_LIB4
#else
	callq _inner_edge_dtrsv_lt_inv_1_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 
	movl	$1, %r11d

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dtrsv_lt_inv_1_lib4, .-kernel_dtrsv_lt_inv_1_lib4
#endif





//                            rdi    rsi        rdx        rcx      r8         r9         
// void kernel_dtrmv_un_4_lib4(int k, double *A, double *x, int alg, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmv_un_4_lib4
	.type kernel_dtrmv_un_4_lib4, @function
kernel_dtrmv_un_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmv_un_4_lib4
_kernel_dtrmv_un_4_lib4:
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


	// call inner dtrmv edge & dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movq	%rdx, %r12  // x

#if MACRO_LEVEL>=1
	INNER_EDGE_DTRMV_UN_4_LIB4
#else
	callq _inner_edge_dtrmv_un_4_lib4
#endif

#if MACRO_LEVEL>=2
	INNER_KERNEL_DGEMV_ADD_N_4_LIB4
#else
	callq _inner_kernel_dgemv_add_n_4_lib4
#endif


	// call inner blender n

	movl	%ecx, %r10d // alg
	movq	%r8, %r11   // y

#if MACRO_LEVEL>=1
	INNER_BLENDER_N_4_LIB4
#else
	callq _inner_blender_n_4_lib4
#endif


	// store

	movq	%r9, %r10 // z

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dtrmv_un_4_lib4, .-kernel_dtrmv_un_4_lib4
#endif





//                             rdi    rsi        rdx      rcx        r8       r9         rsp+8
// void kernel_dtrmv_ut_4_lib4(int k, double *A, int sda, double *x, int alg, double *y, double *z);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmv_ut_4_lib4
	.type kernel_dtrmv_ut_4_lib4, @function
kernel_dtrmv_ut_4_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmv_ut_4_lib4
_kernel_dtrmv_ut_4_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13  // x

#if MACRO_LEVEL>=1
	INNER_KERNEL_DTRMV_UT_4_LIB4
#else
	callq _inner_kernel_dtrmv_ut_4_lib4
#endif


	// call inner blender t

	movl	%r8d, %r10d // alg
	movq	%r9, %r11   // y

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 

#if MACRO_LEVEL>=1
	INNER_STORE_4_LIB4
#else
	callq _inner_store_4_lib4
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
	.size	kernel_dtrmv_ut_4_lib4, .-kernel_dtrmv_ut_4_lib4
#endif





//                                rdi    rsi        rdx      rcx        r8       r9         rsp+8      rsp+16
// void kernel_dtrmv_ut_4_vs_lib4(int k, double *A, int sda, double *x, int alg, double *C, double *D, int km);

	.p2align 4,,15
#if defined(OS_LINUX)
	.globl kernel_dtrmv_ut_4_vs_lib4
	.type kernel_dtrmv_ut_4_vs_lib4, @function
kernel_dtrmv_ut_4_vs_lib4:
#elif defined(OS_MAC)
	.globl _kernel_dtrmv_ut_4_vs_lib4
_kernel_dtrmv_ut_4_vs_lib4:
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


	// call inner dgemv kernel n

	movl	%edi, %r10d // k
	movq	%rsi, %r11  // A
	movl	%edx, %r12d
	sall	$5, %r12d // 4*sda*sizeof(double)
	movslq	%r12d, %r12
	movq	%rcx, %r13  // x

#if MACRO_LEVEL>=1
	INNER_KERNEL_DTRMV_UT_4_LIB4
#else
	callq _inner_kernel_dtrmv_ut_4_lib4
#endif


	// call inner blender t

	movl	%r8d, %r10d // alg
	movq	%r9, %r11   // y

#if MACRO_LEVEL>=1
	INNER_BLENDER_T_4_LIB4
#else
	callq _inner_blender_t_4_lib4
#endif


	// store

	movq	STACKSIZE +  8(%rsp), %r10 // z 
	movl	STACKSIZE + 16(%rsp), %r11d // km 

#if MACRO_LEVEL>=1
	INNER_STORE_4_VS_LIB4
#else
	callq _inner_store_4_vs_lib4
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
	.size	kernel_dtrmv_ut_4_vs_lib4, .-kernel_dtrmv_ut_4_vs_lib4
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
.LC04: // { 1.0 1.0 1.0 1.0 }
#elif defined(OS_MAC)
LC04: // { 1.0 1.0 1.0 1.0 }
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

