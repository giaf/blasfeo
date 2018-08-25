###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS For Embedded Optimization.                                                      #
# Copyright (C) 2016-2018 by Gianluca Frison.                                                     #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# This program is free software: you can redistribute it and/or modify                            #
# it under the terms of the GNU General Public License as published by                            #
# the Free Software Foundation, either version 3 of the License, or                               #
# (at your option) any later version                                                              #.
#                                                                                                 #
# This program is distributed in the hope that it will be useful,                                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   #
# GNU General Public License for more details.                                                    #
#                                                                                                 #
# You should have received a copy of the GNU General Public License                               #
# along with this program.  If not, see <https://www.gnu.org/licenses/>.                          #
#                                                                                                 #
# The authors designate this particular file as subject to the "Classpath" exception              #
# as provided by the authors in the LICENSE file that accompained this code.                      #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################

include ./Makefile.rule

OBJS =

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)

# aux
OBJS += auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib8.o \
		auxiliary/m_aux_lib48.o \

# kernels
OBJS += \
		kernel/avx2/kernel_dgemm_12x4_lib4.o \
		kernel/avx2/kernel_dgemm_8x8_lib4.o \
		kernel/avx2/kernel_dgemm_8x4_lib4.o \
		kernel/avx2/kernel_dgemm_4x4_lib4.o \
		kernel/avx/kernel_dgemm_diag_lib4.o \
		kernel/avx2/kernel_dgemv_8_lib4.o \
		kernel/avx/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/avx2/kernel_dsymv_6_lib4.o \
		kernel/avx2/kernel_dgetrf_pivot_4_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx2/kernel_dgebp_lib4.o \
		kernel/avx2/kernel_dgelqf_4_lib4.o \
		kernel/avx2/kernel_dgetr_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
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

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib8.o \
		blas/s_blas2_lib8.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib8.o \
		blas/s_blas3_diag_lib8.o \
		blas/s_lapack_lib8.o \

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
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/avx/kernel_dsymv_6_lib4.o \
		kernel/avx/kernel_dgetrf_pivot_4_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx/kernel_dgebp_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
		kernel/avx/kernel_dgetr_lib4.o \
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

# blas
OBJS  += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib8.o \
		blas/s_blas2_lib8.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib8.o \
		blas/s_blas3_diag_lib8.o \
		blas/s_lapack_lib8.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/avx_x86/kernel_sgemm_4x4_lib4.o \
		kernel/avx_x86/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/armv8a/kernel_sgemm_16x4_lib4.o \
		kernel/armv8a/kernel_sgemm_12x4_lib4.o \
		kernel/armv8a/kernel_sgemm_8x8_lib4.o \
		kernel/armv8a/kernel_sgemm_8x4_lib4.o \
		kernel/armv8a/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/armv8a/kernel_sgemm_16x4_lib4.o \
		kernel/armv8a/kernel_sgemm_12x4_lib4.o \
		kernel/armv8a/kernel_sgemm_8x8_lib4.o \
		kernel/armv8a/kernel_sgemm_8x4_lib4.o \
		kernel/armv8a/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

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
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/armv7a/kernel_sgemm_12x4_lib4.o \
		kernel/armv7a/kernel_sgemm_8x4_lib4.o \
		kernel/armv7a/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

ifeq ($(TARGET), GENERIC)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

else # LA_REFERENCE | LA_BLAS

# aux
OBJS += \
		auxiliary/d_aux_lib.o \
		auxiliary/s_aux_lib.o \
		auxiliary/m_aux_lib.o \

# blas
OBJS += \
		blas/d_blas1_lib.o \
		blas/d_blas2_lib.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib.o \
		blas/d_blas3_diag_lib.o \
		blas/d_lapack_lib.o \
		\
		blas/s_blas1_lib.o \
		blas/s_blas2_lib.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib.o \
		blas/s_blas3_diag_lib.o \
		blas/s_lapack_lib.o \

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
		blas/d_blas3_libref.o \
		blas/s_blas3_libref.o \
#
endif


# Define targets


all: clean static_library


# compile static library
static_library: target
	( cd kernel; $(MAKE) obj)
	( cd auxiliary; $(MAKE) obj)
	( cd blas; $(MAKE) obj)
	$(AR) rcs libblasfeo.a $(OBJS)
	mv libblasfeo.a ./lib/
ifeq ($(TESTING_MODE), 1)
	$(AR) rcs libblasfeo_ref.a $(OBJS_REF)
	mv libblasfeo_ref.a ./lib/
endif
	@echo
	@echo " libblasfeo.a static library build complete."
ifeq ($(TESTING_MODE), 1)
	@echo " libblasfeo_ref.a static library build complete."
endif
	@echo


# compile shared library
shared_library: target
	( cd auxiliary; $(MAKE) obj)
	( cd kernel; $(MAKE) obj)
	( cd blas; $(MAKE) obj)
	$(CC) -shared -o libblasfeo.so $(OBJS) -Wl,-Bsymbolic
	mv libblasfeo.so ./lib/
ifeq ($(TESTING_MODE), 1)
	$(CC) -shared -o libblasfeo_ref.so $(OBJS_REF)
	mv libblasfeo_ref.so ./lib/
endif
	@echo
	@echo " libblasfeo.so shared library build complete."
	@echo


# generate target header
target:
	touch ./include/blasfeo_target.h
ifeq ($(TARGET), X64_INTEL_HASWELL)
	echo "#ifndef TARGET_X64_INTEL_HASWELL" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	echo "#ifndef TARGET_X64_INTEL_SANDY_BRIDGE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_CORE)
	echo "#ifndef TARGET_X64_INTEL_CORE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_CORE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X64_INTEL_CORE" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
	echo "#ifndef TARGET_X64_AMD_BULLDOZER" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
	echo "#ifndef TARGET_X86_AMD_JAGUAR" > ./include/blasfeo_target.h
	echo "#define TARGET_X86_AMD_JAGUAR" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X86_AMD_JAGUAR" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
	echo "#ifndef TARGET_X86_AMD_BARCELONA" > ./include/blasfeo_target.h
	echo "#define TARGET_X86_AMD_BARCELONA" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET X86_AMD_BARCELONA" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), GENERIC)
	echo "#ifndef TARGET_GENERIC" > ./include/blasfeo_target.h
	echo "#define TARGET_GENERIC" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET GENERIC" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
	echo "#ifndef TARGET_ARMV7A_ARM_CORTEX_A15" > ./include/blasfeo_target.h
	echo "#define TARGET_ARMV7A_ARM_CORTEX_A15" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define TARGET ARMV7A_ARM_CORTEX_A15" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), HIGH_PERFORMANCE)
	echo "#ifndef LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#define LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define LA HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), BLAS_WRAPPER)
	echo "#ifndef LA_BLAS_WRAPPER" >> ./include/blasfeo_target.h
	echo "#define LA_BLAS_WRAPPER" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define LA BLAS_WRAPPER" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), REFERENCE)
	echo "#ifndef LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#define LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
#	echo "#define LA REFERENCE" >> ./include/blasfeo_target.h
endif
ifeq ($(EXT_DEP), 1)
	echo "#ifndef EXT_DEP" >> ./include/blasfeo_target.h
	echo "#define EXT_DEP" >> ./include/blasfeo_target.h
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
	make -C blas clean
	make -C examples clean
	make -C tests clean
	make -C benchmarks clean


# deep clean
deep_clean: clean
	rm -f ./include/blasfeo_target.h
	rm -f ./lib/libblasfeo.a
	rm -f ./lib/libblasfeo.so
	rm -f ./lib/libblasfeo_ref.a
	make -C examples clean_all
	make -C tests clean_all
	make -C benchmarks clean_all

# directory for tests  binaries
BINARY_DIR = build/$(LA)/$(TARGET)

### benchmarks

deploy_to_benchmarks:
	mkdir -p ./benchmarks/$(BINARY_DIR)/
	cp ./lib/libblasfeo.a ./benchmarks/$(BINARY_DIR)/

build_benchmarks:
	make -C benchmarks build
	@echo
	@echo "Benchmarks build complete."
	@echo

benchmarks: deploy_to_benchmarks build_benchmarks

run_benchmarks:
	make -C benchmarks run



build_benchmarks_all:
	make -C benchmarks all
	@echo
	@echo "Benchmarks build complete."
	@echo

benchmarks_all: deploy_to_benchmarks build_benchmarks_all

run_benchmarks_all:
	make -C benchmarks run_all

print_figures_benchmark_all:
	make -C benchmarks print_figures

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
