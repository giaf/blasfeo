###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS for embedded optimization.                                                      #
# Copyright (C) 2019 by Gianluca Frison.                                                          #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# The 2-Clause BSD License                                                                        #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without                              #
# modification, are permitted provided that the following conditions are met:                     #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
#    list of conditions and the following disclaimer.                                             #
# 2. Redistributions in binary form must reproduce the above copyright notice,                    #
#    this list of conditions and the following disclaimer in the documentation                    #
#    and/or other materials provided with the distribution.                                       #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################

include ./Makefile.rule

OBJS =

OBJS += \
		auxiliary/blasfeo_processor_features.o \
		auxiliary/blasfeo_stdlib.o \

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib8.o \
		auxiliary/m_aux_lib48.o \

# kernels
OBJS += \
		kernel/avx2/kernel_dgemm_12x4_lib4.o \
		kernel/avx2/kernel_dgemm_8x8_lib4.o \
		kernel/avx2/kernel_dgemm_8x4_lib4.o \
		kernel/avx2/kernel_dgemm_4x4_lib4.o \
		kernel/avx2/kernel_dgemv_8_lib4.o \
		kernel/avx2/kernel_dsymv_6_lib4.o \
		kernel/avx2/kernel_dgetrf_pivot_lib4.o \
		kernel/avx2/kernel_dgebp_lib4.o \
		kernel/avx2/kernel_dgelqf_4_lib4.o \
		kernel/avx2/kernel_dgetr_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx/kernel_dgemv_4_lib4.o \
		kernel/avx/kernel_dgemm_diag_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
		kernel/avx/kernel_dpack_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/avx2/kernel_sgemm_24x4_lib8.o \
		kernel/avx2/kernel_sgemm_16x4_lib8.o \
		kernel/avx2/kernel_sgemm_8x8_lib8.o \
		kernel/avx2/kernel_sgemm_8x4_lib8.o \
		kernel/avx/kernel_sgemm_diag_lib8.o \
		kernel/avx/kernel_sgemv_8_lib8.o \
		kernel/avx/kernel_sgemv_4_lib8.o \
		kernel/avx/kernel_sgecpsc_lib8.o \
		kernel/avx/kernel_sgetr_lib8.o \
		kernel/avx/kernel_sgead_lib8.o \
		kernel/avx/kernel_spack_lib8.o \
		kernel/generic/kernel_sgemm_8x4_lib8.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x64.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib8.o \
		blasfeo_api/s_blas2_lib8.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib8.o \
		blasfeo_api/s_blas3_diag_lib8.o \
		blasfeo_api/s_lapack_lib8.o \

endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib8.o \
		auxiliary/m_aux_lib48.o \

# kernels
OBJS += \
		kernel/avx/kernel_dgemm_12x4_lib4.o \
		kernel/avx/kernel_dgemm_8x4_lib4.o \
		kernel/avx/kernel_dgemm_4x4_lib4.o \
		kernel/avx/kernel_dgemm_diag_lib4.o \
		kernel/avx/kernel_dgemv_12_lib4.o \
		kernel/avx/kernel_dgemv_8_lib4.o \
		kernel/avx/kernel_dgemv_4_lib4.o \
		kernel/avx/kernel_dsymv_6_lib4.o \
		kernel/avx/kernel_dgetrf_pivot_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx/kernel_dgebp_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
		kernel/avx/kernel_dgetr_lib4.o \
		kernel/avx/kernel_dpack_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/avx/kernel_sgemm_16x4_lib8.o \
		kernel/avx/kernel_sgemm_8x8_lib8.o \
		kernel/avx/kernel_sgemm_8x4_lib8.o \
		kernel/avx/kernel_sgemm_diag_lib8.o \
		kernel/avx/kernel_sgemv_8_lib8.o \
		kernel/avx/kernel_sgemv_4_lib8.o \
		kernel/avx/kernel_sgecpsc_lib8.o \
		kernel/avx/kernel_sgetr_lib8.o \
		kernel/avx/kernel_sgead_lib8.o \
		kernel/avx/kernel_spack_lib8.o \
		kernel/generic/kernel_sgemm_8x4_lib8.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x64.o\

# blas
OBJS  += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib8.o \
		blasfeo_api/s_blas2_lib8.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib8.o \
		blasfeo_api/s_blas3_diag_lib8.o \
		blasfeo_api/s_lapack_lib8.o \

endif

ifeq ($(TARGET), X64_INTEL_CORE)
# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/sse3/kernel_dgemm_4x4_lib4.o \
		kernel/sse3/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/sse3/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x64.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/fma/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x64.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), X86_AMD_JAGUAR)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/avx_x86/kernel_dgemm_4x4_lib4.o \
		kernel/avx_x86/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/avx_x86/kernel_sgemm_4x4_lib4.o \
		kernel/avx_x86/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x86.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), X86_AMD_BARCELONA)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/sse3_x86/kernel_dgemm_4x2_lib4.o \
		kernel/sse3_x86/kernel_dgemm_2x2_lib4.o \
		kernel/sse3_x86/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_x86.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv8a/kernel_dgemm_8x4_lib4.o \
		kernel/armv8a/kernel_dgemm_4x4_lib4.o \
		kernel/armv8a/kernel_dpack_lib4.o \
		kernel/armv8a/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/armv8a/kernel_sgemm_16x4_lib4.o \
		kernel/armv8a/kernel_sgemm_12x4_lib4.o \
		kernel/armv8a/kernel_sgemm_8x8_lib4.o \
		kernel/armv8a/kernel_sgemm_8x4_lib4.o \
		kernel/armv8a/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_generic.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv8a/kernel_dgemm_12x4_lib4.o \
		kernel/armv8a/kernel_dgemm_8x4_lib4.o \
		kernel/armv8a/kernel_dgemm_4x4_lib4.o \
		kernel/armv8a/kernel_dpack_lib4.o \
		kernel/armv8a/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/armv8a/kernel_sgemm_16x4_lib4.o \
		kernel/armv8a/kernel_sgemm_12x4_lib4.o \
		kernel/armv8a/kernel_sgemm_8x8_lib4.o \
		kernel/armv8a/kernel_sgemm_8x4_lib4.o \
		kernel/armv8a/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_generic.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv7a/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/armv7a/kernel_sgemm_12x4_lib4.o \
		kernel/armv7a/kernel_sgemm_8x4_lib4.o \
		kernel/armv7a/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_generic.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), $(filter $(TARGET), ARMV7A_ARM_CORTEX_A9 ARMV7A_ARM_CORTEX_A7))
# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv7a/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/armv7a/kernel_sgemm_8x4_lib4.o \
		kernel/armv7a/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_generic.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif

ifeq ($(TARGET), GENERIC)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/generic/kernel_dgemm_4x4_lib4.o \
		kernel/generic/kernel_dgemm_diag_lib4.o \
		kernel/generic/kernel_dgemv_4_lib4.o \
		kernel/generic/kernel_dsymv_4_lib4.o \
		kernel/generic/kernel_dgecp_lib4.o \
		kernel/generic/kernel_dgetr_lib4.o \
		kernel/generic/kernel_dgetrf_pivot_lib4.o \
		kernel/generic/kernel_dgeqrf_4_lib4.o \
		kernel/generic/kernel_dpack_lib4.o \
		kernel/generic/kernel_ddot_lib.o \
		kernel/generic/kernel_daxpy_lib.o \
		\
		kernel/generic/kernel_sgemm_4x4_lib4.o \
		kernel/generic/kernel_sgemm_diag_lib4.o \
		kernel/generic/kernel_sgemv_4_lib4.o \
		kernel/generic/kernel_ssymv_4_lib4.o \
		kernel/generic/kernel_sgetrf_pivot_lib4.o \
		kernel/generic/kernel_sgecp_lib4.o \
		kernel/generic/kernel_sgetr_lib4.o \
		kernel/generic/kernel_spack_lib4.o \
		kernel/generic/kernel_sdot_lib.o \
		kernel/generic/kernel_saxpy_lib.o \
		\
		kernel/kernel_align_generic.o\

# blas
OBJS += \
		blasfeo_api/d_blas1_lib4.o \
		blasfeo_api/d_blas2_lib4.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib4.o \
		blasfeo_api/d_blas3_diag_lib4.o \
		blasfeo_api/d_lapack_lib4.o \
		\
		blasfeo_api/s_blas1_lib4.o \
		blasfeo_api/s_blas2_lib4.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib4.o \
		blasfeo_api/s_blas3_diag_lib4.o \
		blasfeo_api/s_lapack_lib4.o \

endif # GENERIC

ifeq ($(BLAS_API), 1)
OBJS += \
		blas_api/dcopy.o \
		blas_api/daxpy.o \
		blas_api/ddot.o \
		blas_api/dgemm.o \
		blas_api/dsyrk.o \
		blas_api/dtrmm.o \
		blas_api/dtrsm.o \
		blas_api/dgesv.o \
		blas_api/dgetrf.o \
		blas_api/dgetrs.o \
		blas_api/dlaswp.o \
		blas_api/dposv.o \
		blas_api/dpotrf.o \
		blas_api/dpotrs.o \
		blas_api/dtrtrs.o \
		\
		blas_api/saxpy.o \
		blas_api/sdot.o \
		blas_api/sgemm.o

ifeq ($(COMPLEMENT_WITH_NETLIB_BLAS), 1)
include $(CURRENT_DIR)/netlib/Makefile.netlib_blas
OBJS += $(NETLIB_BLAS_OBJS)
endif # COMPLEMENT_WITH_NETLIB_BLAS

ifeq ($(COMPLEMENT_WITH_NETLIB_LAPACK), 1)
include $(CURRENT_DIR)/netlib/Makefile.netlib_lapack
OBJS += $(NETLIB_LAPACK_OBJS)
endif # COMPLEMENT_WITH_NETLIB_LAPACK

ifeq ($(CBLAS_API), 1)
include $(CURRENT_DIR)/netlib/Makefile.netlib_cblas
OBJS += $(NETLIB_CBLAS_OBJS)
endif # CBLAS_API

ifeq ($(LAPACKE_API), 1)
include $(CURRENT_DIR)/netlib/Makefile.netlib_lapacke
OBJS += $(NETLIB_LAPACKE_OBJS)
endif # LAPACKE_API

endif # BLAS_API

else # LA_HIGH_PERFORMANCE vs LA_REFERENCE | LA_BLAS

# aux
OBJS += \
		auxiliary/d_aux_lib.o \
		auxiliary/s_aux_lib.o \
		auxiliary/m_aux_lib.o \

# blas
OBJS += \
		blasfeo_api/d_blas1_lib.o \
		blasfeo_api/d_blas2_lib.o \
		blasfeo_api/d_blas2_diag_lib.o \
		blasfeo_api/d_blas3_lib.o \
		blasfeo_api/d_blas3_diag_lib.o \
		blasfeo_api/d_lapack_lib.o \
		\
		blasfeo_api/s_blas1_lib.o \
		blasfeo_api/s_blas2_lib.o \
		blasfeo_api/s_blas2_diag_lib.o \
		blasfeo_api/s_blas3_lib.o \
		blasfeo_api/s_blas3_diag_lib.o \
		blasfeo_api/s_lapack_lib.o \

ifeq ($(BLAS_API), 1)
OBJS += \
		blas_api/dgemm_ref.o \

endif

endif # LA choice

ifeq ($(EXT_DEP), 1)
# ext dep
ifeq ($(LA), HIGH_PERFORMANCE)
OBJS += \
		auxiliary/d_aux_ext_dep_lib4.o \
		auxiliary/s_aux_ext_dep_lib4.o
endif
OBJS += \
		auxiliary/d_aux_ext_dep_lib.o \
		auxiliary/s_aux_ext_dep_lib.o \
		auxiliary/v_aux_ext_dep_lib.o \
		auxiliary/i_aux_ext_dep_lib.o \
		auxiliary/timing.o

endif



ifeq ($(TESTING_MODE), 1)
# reference routine for testing
OBJS_REF =
# aux
OBJS_REF += \
		auxiliary/d_aux_libref.o \
		auxiliary/s_aux_libref.o \
		auxiliary/d_aux_ext_dep_libref.o \
		auxiliary/s_aux_ext_dep_libref.o \
		blasfeo_api/d_blas1_libref.o \
		blasfeo_api/d_blas2_libref.o \
		blasfeo_api/d_blas2_diag_libref.o \
		blasfeo_api/d_blas3_libref.o \
		blasfeo_api/d_blas3_diag_libref.o \
		blasfeo_api/d_lapack_libref.o \
		blasfeo_api/s_blas1_libref.o \
		blasfeo_api/s_blas2_libref.o \
		blasfeo_api/s_blas2_diag_libref.o \
		blasfeo_api/s_blas3_libref.o \
		blasfeo_api/s_blas3_diag_libref.o \
		blasfeo_api/s_lapack_libref.o \
#
endif



ifeq ($(SANDBOX_MODE), 1)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += sandbox/kernel_avx2.o
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += sandbox/kernel_avx.o
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += sandbox/kernel_sse3.o
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += sandbox/kernel_avx.o
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
OBJS += sandbox/kernel_avx_x86.o
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
OBJS += sandbox/kernel_sse3_x86.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += sandbox/kernel_armv8a.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
OBJS += sandbox/kernel_armv8a.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += sandbox/kernel_armv7a.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A7)
OBJS += sandbox/kernel_armv7a.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A9)
OBJS += sandbox/kernel_armv7a.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += sandbox/kernel_generic.o
endif

#OBJS += sandbox/kernel_c_dummy.o
#OBJS += sandbox/kernel_asm_dummy.o

endif



# Define targets


all: clean static_library


# compile static library
static_library: target
	( cd kernel; $(MAKE) obj)
	( cd auxiliary; $(MAKE) obj)
	( cd blasfeo_api; $(MAKE) obj)
ifeq ($(BLAS_API), 1)
	( cd blas_api; $(MAKE) obj)
ifeq ($(COMPLEMENT_WITH_NETLIB_BLAS), 1)
	( cd netlib; $(MAKE) obj_blas)
endif
ifeq ($(COMPLEMENT_WITH_NETLIB_LAPACK), 1)
	( cd netlib; $(MAKE) obj_lapack)
endif
ifeq ($(CBLAS_API), 1)
	( cd netlib; $(MAKE) obj_cblas)
endif
ifeq ($(LAPACKE_API), 1)
	( cd netlib; $(MAKE) obj_lapacke)
endif
endif
ifeq ($(SANDBOX_MODE), 1)
	( cd sandbox; $(MAKE) obj)
endif
	$(AR) rcs libblasfeo.a $(OBJS)
	mv libblasfeo.a ./lib/
	@echo
	@echo " libblasfeo.a static library build complete."
ifeq ($(TESTING_MODE), 1)
	$(AR) rcs libblasfeo_ref.a $(OBJS_REF)
	mv libblasfeo_ref.a ./lib/
	@echo " libblasfeo_ref.a static library build complete."
endif
	@echo


# compile shared library
shared_library: target
	( cd auxiliary; $(MAKE) obj)
	( cd kernel; $(MAKE) obj)
	( cd blasfeo_api; $(MAKE) obj)
ifeq ($(BLAS_API), 1)
	( cd blas_api; $(MAKE) obj)
ifeq ($(COMPLEMENT_WITH_NETLIB_BLAS), 1)
	( cd netlib; $(MAKE) obj_blas)
endif
ifeq ($(COMPLEMENT_WITH_NETLIB_LAPACK), 1)
	( cd netlib; $(MAKE) obj_lapack)
endif
ifeq ($(CBLAS_API), 1)
	( cd netlib; $(MAKE) obj_cblas)
endif
ifeq ($(LAPACKE_API), 1)
	( cd netlib; $(MAKE) obj_lapacke)
endif
endif
ifeq ($(SANDBOX_MODE), 1)
	( cd sandbox; $(MAKE) obj)
endif
	$(CC) -shared -o libblasfeo.so $(OBJS) -lm #-Wl,-Bsymbolic
	mv libblasfeo.so ./lib/
	@echo
	@echo " libblasfeo.so shared library build complete."
ifeq ($(TESTING_MODE), 1)
	$(CC) -shared -o libblasfeo_ref.so $(OBJS_REF) -lm
	mv libblasfeo_ref.so ./lib/
	@echo " libblasfeo_ref.so shared library build complete."
endif
	@echo


# generate target header
target:
	touch ./include/blasfeo_target.h
ifeq ($(TARGET), X64_INTEL_HASWELL)
	echo "#ifndef TARGET_X64_INTEL_HASWELL" >  ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_AVX2" >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_AVX2" >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_FMA"  >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_FMA"  >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	echo "#ifndef TARGET_X64_INTEL_SANDY_BRIDGE" >  ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
	echo "#endif"                                >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_AVX"       >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_AVX"       >> ./include/blasfeo_target.h
	echo "#endif"                                >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_CORE)
	echo "#ifndef TARGET_X64_INTEL_CORE"    >  ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_CORE"    >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_SSE3" >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_SSE3" >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
	echo "#ifndef TARGET_X64_AMD_BULLDOZER" >  ./include/blasfeo_target.h
	echo "#define TARGET_X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_AVX"  >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_AVX"  >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_FMA"  >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_FMA"  >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
	echo "#ifndef TARGET_X86_AMD_JAGUAR" >  ./include/blasfeo_target.h
	echo "#define TARGET_X86_AMD_JAGUAR" >> ./include/blasfeo_target.h
	echo "#endif"                        >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
	echo "#ifndef TARGET_X86_AMD_BARCELONA" >  ./include/blasfeo_target.h
	echo "#define TARGET_X86_AMD_BARCELONA" >> ./include/blasfeo_target.h
	echo "#endif"                           >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
	echo "#ifndef TARGET_ARMV8A_ARM_CORTEX_A57" >  ./include/blasfeo_target.h
	echo "#define TARGET_ARMV8A_ARM_CORTEX_A57" >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_VFPv4"    >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_VFPv4"    >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_NEONv2"   >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_NEONv2"   >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
	echo "#ifndef TARGET_ARMV8A_ARM_CORTEX_A53" >  ./include/blasfeo_target.h
	echo "#define TARGET_ARMV8A_ARM_CORTEX_A53" >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_VFPv4"    >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_VFPv4"    >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_NEONv2"   >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_NEONv2"   >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
	echo "#ifndef TARGET_ARMV7A_ARM_CORTEX_A15" >  ./include/blasfeo_target.h
	echo "#define TARGET_ARMV7A_ARM_CORTEX_A15" >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_VFPv3"    >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_VFPv3"    >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_NEON"     >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_NEON"     >> ./include/blasfeo_target.h
	echo "#endif"                               >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A7)
	echo "#ifndef TARGET_ARMV7A_ARM_CORTEX_A7" >  ./include/blasfeo_target.h
	echo "#define TARGET_ARMV7A_ARM_CORTEX_A7" >> ./include/blasfeo_target.h
	echo "#endif"                              >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_VFPv3"   >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_VFPv3"   >> ./include/blasfeo_target.h
	echo "#endif"                              >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_NEON"    >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_NEON"    >> ./include/blasfeo_target.h
	echo "#endif"                              >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A9)
	echo "#ifndef TARGET_ARMV7A_ARM_CORTEX_A9" >  ./include/blasfeo_target.h
	echo "#define TARGET_ARMV7A_ARM_CORTEX_A9" >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_VFPv3"   >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_VFPv3"   >> ./include/blasfeo_target.h
	echo "#endif"                              >> ./include/blasfeo_target.h
	echo "#ifndef TARGET_NEED_FEATURE_NEON"    >> ./include/blasfeo_target.h
	echo "#define TARGET_NEED_FEATURE_NEON"    >> ./include/blasfeo_target.h
	echo "#endif"                              >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), GENERIC)
	echo "#ifndef TARGET_GENERIC" >  ./include/blasfeo_target.h
	echo "#define TARGET_GENERIC" >> ./include/blasfeo_target.h
	echo "#endif"                 >> ./include/blasfeo_target.h
endif
ifeq ($(LA), HIGH_PERFORMANCE)
	echo "#ifndef LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#define LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), EXTERNAL_BLAS_WRAPPER)
	echo "#ifndef LA_EXTERNAL_BLAS_WRAPPER" >> ./include/blasfeo_target.h
	echo "#define LA_EXTERNAL_BLAS_WRAPPER" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), REFERENCE)
	echo "#ifndef LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#define LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(EXT_DEP), 1)
	echo "#ifndef EXT_DEP" >> ./include/blasfeo_target.h
	echo "#define EXT_DEP" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(BLAS_API), 1)
	echo "#ifndef BLAS_API" >> ./include/blasfeo_target.h
	echo "#define BLAS_API" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(FORTRAN_BLAS_API), 1)
	echo "#ifndef FORTRAN_BLAS_API" >> ./include/blasfeo_target.h
	echo "#define FORTRAN_BLAS_API" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif


# install static library & headers
install_static:
	mkdir -p $(PREFIX)/blasfeo
	mkdir -p $(PREFIX)/blasfeo/lib
	cp -f ./lib/libblasfeo.a $(PREFIX)/blasfeo/lib/
	mkdir -p $(PREFIX)/blasfeo/include
	cp -f ./include/*.h $(PREFIX)/blasfeo/include/


# install share library & headers
install_shared:
	mkdir -p $(PREFIX)/blasfeo
	mkdir -p $(PREFIX)/blasfeo/lib
	cp -f ./lib/libblasfeo.so $(PREFIX)/blasfeo/lib/
	mkdir -p $(PREFIX)/blasfeo/include
	cp -f ./include/*.h $(PREFIX)/blasfeo/include/


# clean .o files
clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blasfeo_api clean
	make -C blas_api clean
	make -C netlib clean
	make -C examples clean
	make -C tests clean
	make -C benchmarks clean
	make -C sandbox clean

# deep clean
deep_clean: clean
	rm -f ./include/blasfeo_target.h
	rm -f ./lib/libblasfeo.a
	rm -f ./lib/libblasfeo.so
	rm -f ./lib/libblasfeo_ref.a
	make -C netlib deep_clean
	make -C examples deep_clean
	make -C tests deep_clean
	make -C benchmarks deep_clean

clean_blas_api:
	make -C blas_api clean



### benchmarks


# single benchmark

deploy_to_benchmarks:
	mkdir -p ./benchmarks/$(BINARY_DIR)/
	mkdir -p ./benchmarks/$(BINARY_DIR)/BLASFEO_API/
ifeq ($(BLAS_API), 1)
	mkdir -p ./benchmarks/$(BINARY_DIR)/BLAS_API/
endif
	cp ./lib/libblasfeo.a ./benchmarks/$(BINARY_DIR)/

build_benchmarks:
	make -C benchmarks build
	@echo
	@echo "Benchmarks build complete."
	@echo

benchmarks: deploy_to_benchmarks build_benchmarks

run_benchmarks:
	make -C benchmarks run

figures_benchmark_one:
	make -C benchmarks figures_benchmark_one


# BLASFEO API benchmark

build_benchmarks_blasfeo_api_all:
	make -C benchmarks blasfeo_api_all
	@echo
	@echo "Benchmarks build complete."
	@echo

benchmarks_blasfeo_api_all: deploy_to_benchmarks build_benchmarks_blasfeo_api_all

run_benchmarks_blasfeo_api_all:
	make -C benchmarks run_blasfeo_api_all

figures_benchmark_blasfeo_api_all:
	make -C benchmarks figures_benchmark_blasfeo_api_all


# BLAS API benchmark

build_benchmarks_blas_api_all:
	make -C benchmarks blas_api_all
	@echo
	@echo "Benchmarks build complete."
	@echo

benchmarks_blas_api_all: deploy_to_benchmarks build_benchmarks_blas_api_all

run_benchmarks_blas_api_all:
	make -C benchmarks run_blas_api_all

figures_benchmark_blas_api_all:
	make -C benchmarks figures_benchmark_blas_api_all



### examples

deploy_to_examples:
	mkdir -p ./examples/$(BINARY_DIR)/
	cp ./lib/libblasfeo.a ./examples/$(BINARY_DIR)/

build_examples:
	make -C examples build
	@echo
	@echo "Examples build complete."
	@echo

examples: deploy_to_examples build_examples

run_examples:
	make -C examples run



### sandbox

build_sandbox:
	make -C sandbox build
	@echo
	@echo "Sandbox build complete."
	@echo

disassembly_sandbox:
	make -C sandbox disassembly

sandbox: build_sandbox

run_sandbox:
	make -C sandbox run



### tests

# copy static library into test path
deploy_to_tests:
	mkdir -p ./tests/$(BINARY_DIR)
	cp ./lib/libblasfeo.a ./tests/$(BINARY_DIR)/
ifeq ($(TESTING_MODE), 1)
	cp ./lib/libblasfeo_ref.a ./tests/$(BINARY_DIR)/
endif

# test one, one single test

build_tests_one:
	make -C tests one
	@echo
	@echo " Build test_one complete."
	@echo

tests_one: deploy_to_tests build_tests_one

run_tests_one:
	make -C tests run_one

# aux test
build_tests_aux:
	make -C tests aux
	@echo
	@echo " Build tests_aux complete."
	@echo

tests_aux: deploy_to_tests build_tests_aux

run_tests_aux:
	make -C tests run_aux

# blas test
build_tests_blas:
	make -C tests blas
	@echo
	@echo " Build tests_blas complete."
	@echo

tests_blas: deploy_to_tests build_tests_blas

run_tests_blas:
	make -C tests run_blas

### shortcuts

tests_all: tests_aux tests_blas
run_tests_all: run_tests_blas run_tests_aux
build_tests_all: build_tests_blas build_tests_aux

tests_clean_all:
	make -C tests clean_all
examples_clean_all:
	make -C examples clean_all
benchmarks_clean_all:
	make -C benchmarks clean_all

# test_rebuild: if tests sources is modified
# build tests (use existing library); run tests
update_test: build_test_all run_test_all

# lib_rebuild: modified blasfeo lib code
# build lib; copy lib; build test; run test
update_lib_test: static_library test_all run_test_all

# hard_rebuild: modified blasfeo lib flags affecting macros
# delete lib; build lib; copy lib; build test; run test
update_deep_test: clean static_library test_all run_test_all
