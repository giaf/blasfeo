###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS For Embedded Optimization.                                                      #
# Copyright (C) 2016-2017 by Gianluca Frison.                                                     #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# HPMPC is free software; you can redistribute it and/or                                          #
# modify it under the terms of the GNU Lesser General Public                                      #
# License as published by the Free Software Foundation; either                                    #
# version 2.1 of the License, or (at your option) any later version.                              #
#                                                                                                 #
# HPMPC is distributed in the hope that it will be useful,                                        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            #
# See the GNU Lesser General Public License for more details.                                     #
#                                                                                                 #
# You should have received a copy of the GNU Lesser General Public                                #
# License along with HPMPC; if not, write to the Free Software                                    #
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  #
#                                                                                                 #
# Author: Gianluca Frison, giaf (at) dtu.dk                                                       #
#                          gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################

cmake_minimum_required(VERSION 2.8.11)
project(blasfeo)

set(CMAKE_VERBOSE_MAKEFILE ON)
enable_language(C ASM)

# Target architecture
set(BLASFEO_TARGET "AUTOMATIC" CACHE STRING "Target architecture")
set_property(CACHE BLASFEO_TARGET
	PROPERTY STRINGS
		AUTOMATIC
		X64_INTEL_HASWELL
		X64_INTEL_SANDY_BRIDGE
		X64_INTEL_CORE
		X64_AMD_BULLDOZER
		ARMV8A_ARM_CORTEX_A57
		ARMV7A_ARM_CORTEX_A15
		GENERIC)

# What library to build
set(BLASFEO_LA_MODE "HIGH_PERFORMANCE" CACHE STRING "Level of performance")
set_property(CACHE BLASFEO_LA_MODE
	PROPERTY STRINGS
		HIGH_PERFORMANCE
		REFERENCE
		BLAS)

# BLAS and LAPACK version (for LA=BLAS in BLASFEO)
set(BLASFEO_REFERENCE_BLAS "NONE" CACHE STRING "Reference BLAS library")
set_property(CACHE BLASFEO_REFERENCE_BLAS
	PROPERTY STRINGS
		NONE
		OPENBLAS
		NETLIB
		MKL
		BLIS
		ATLAS)

# Macro level (code size vs performance in assembly kernels):
set(BLASFEO_MACRO_LEVEL "0" CACHE STRING "Macro level")
set_property(CACHE BLASFEO_MACRO_LEVEL
	PROPERTY STRINGS
		0  # no macro
		1  # all macro except gemm kernel
		2) # all macro

# Minimum kernel size
set(BLASFEO_MIN_KERNEL_SIZE "2x2" CACHE STRING "Minimum kernel size")
set_property(CACHE BLASFEO_MIN_KERNEL_SIZE
	PROPERTY STRINGS
		2x2
		4x4)

# Compile auxiliary functions with external dependencies (for memory allocation and printing)
option(BLASFEO_EXT_DEP "Compile external dependencies in BLASFEO" ON)

# Enable runtime checks
option(BLASFEO_RUNTIME_CHECKS "Enable runtime checking" OFF)

# Enable tests
option(BLASFEO_TESTING "Enable tests" OFF)

include(cmake/DetermineInstructionSet.cmake)
determine_instruction_set_flags("${BLASFEO_TARGET}")

configure_file(${PROJECT_SOURCE_DIR}/blasfeo_target.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/blasfeo_target.h @ONLY)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA_${BLASFEO_LA_MODE}")

if(BLASFEO_EXT_DEP)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DEXT_DEP")
endif()

if(BLASFEO_RUNTIME_CHECKS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDIM_CHECK")
endif()

if(BLASFEO_TESTING)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTESTING_MODE")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMACRO_LEVEL=${BLASFEO_MACRO_LEVEL}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(OS_FLAG "OS_LINUX")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(OS_FLAG "OS_MAC")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(OS_FLAG "OS_WINDOWS")
endif()

set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -D${OS_FLAG}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D${OS_FLAG}")

if(${BLASFEO_REFERENCE_BLAS} MATCHES OPENBLAS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_OPENBLAS -I/opt/openblas/include")
elseif(${BLASFEO_REFERENCE_BLAS} MATCHES BLIS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_BLIS -std=c99")
elseif(${BLASFEO_REFERENCE_BLAS} MATCHES NETLIB)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_NETLIB")
elseif(${BLASFEO_REFERENCE_BLAS} MATCHES MKL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_MKL -m64 -I/opt/intel/mkl/include")
elseif(${BLASFEO_REFERENCE_BLAS} MATCHES ATLAS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_ATLAS")
endif()

# kernel size
if(BLASFEO_MIN_KERNEL_SIZE MATCHES 2x2)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIN_KERNEL_SIZE=2")
	set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -DMIN_KERNEL_SIZE=2")
else()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIN_KERNEL_SIZE=4")
	set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -DMIN_KERNEL_SIZE=4")
endif()

# source files
add_library(blasfeo "")
add_subdirectory(auxiliary)
add_subdirectory(kernel)
add_subdirectory(blas)

if(${LA} MATCHES HIGH_PERFORMANCE)

	if(${TARGET} MATCHES X64_INTEL_HASWELL)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib8.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib48.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_8x8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_10xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x2_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_6xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x2_lib4.S


			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemv_8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dsymv_6_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgebp_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgelqf_4_lib4.S

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgetr_lib4.c


			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_24x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_16x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_8x8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_8x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_diag_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib8.c

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgead_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgetr_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgecpsc_lib8.S

			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib8.c
			)

	endif(${TARGET} MATCHES X64_INTEL_HASWELL)

	if(${TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib8.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib48.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_10xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x2_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_6xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x2_lib4.S

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_12_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dsymv_6_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgebp_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgelqf_4_lib4.S

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_16x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_8x8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_8x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_diag_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib8.c

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgead_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgecpsc_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgetr_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib8.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib8.c
			)

	endif(${TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

	if(${TARGET} MATCHES X64_INTEL_CORE)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/sse3/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgeqrf_4_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetrf_pivot_4_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES X64_INTEL_CORE)

	if(${TARGET} MATCHES X64_AMD_BULLDOZER)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/fma/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES X64_AMD_BULLDOZER)

	if(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_16x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

	if(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

	if(${TARGET} MATCHES GENERIC)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgeqrf_4_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetrf_pivot_4_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_sgetr_lib4.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blas/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/d_lapack_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blas/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES GENERIC)

else(${LA} MATCHES HIGH_PERFORMANCE) # REFERENCE BLAS

	file(GLOB AUX_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib.c
		)

	file(GLOB BLAS_SRC
		${PROJECT_SOURCE_DIR}/blas/d_blas1_lib.c
		${PROJECT_SOURCE_DIR}/blas/d_blas2_lib.c
		${PROJECT_SOURCE_DIR}/blas/d_blas2_diag_lib.c
		${PROJECT_SOURCE_DIR}/blas/d_blas3_lib.c
		${PROJECT_SOURCE_DIR}/blas/d_blas3_diag_lib.c
		${PROJECT_SOURCE_DIR}/blas/d_lapack_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_blas1_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_blas2_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_blas2_diag_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_blas3_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_blas3_diag_lib.c
		${PROJECT_SOURCE_DIR}/blas/s_lapack_lib.c
		)

endif(${LA} MATCHES HIGH_PERFORMANCE)

if(${EXT_DEP})

	file(GLOB EXT_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_ext_dep_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_ext_dep_lib4.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_ext_dep_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_ext_dep_lib4.c
		${PROJECT_SOURCE_DIR}/auxiliary/v_aux_ext_dep_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/i_aux_ext_dep_lib.c
		)

endif()


if(BLASFEO_TESTING)

	file(GLOB AUX_REF_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_ext_dep_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_ext_dep_libref.c
		)

	set(BLASFEO_REF_SRC ${AUX_REF_SRC})
	add_library(blasfeo_ref ${BLASFEO_REF_SRC})
	set_target_properties(blasfeo_ref PROPERTIES LINKER_LANGUAGE C)
endif()

set(BLASFEO_SRC ${AUX_SRC} ${KERNEL_SRC} ${BLAS_SRC} ${EXT_SRC})

# add library
add_library(blasfeo ${BLASFEO_SRC})

target_include_directories(blasfeo
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS blasfeo EXPORT blasfeoConfig
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin)


install(EXPORT blasfeoConfig DESTINATION cmake)

file(GLOB_RECURSE BLASFEO_HEADERS "include/*.h")
install(FILES ${BLASFEO_HEADERS} DESTINATION include)
