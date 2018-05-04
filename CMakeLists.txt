###################################################################################################
#                                                                                                 #
# This file is part of HPIPM.                                                                     #
#                                                                                                 #
# HPIPM -- High Performance Interior Point Method.                                                #
# Copyright (C) 2017 by Gianluca Frison.                                                          #
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
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################

cmake_minimum_required(VERSION 2.8.11)

project(blasfeo C ASM)

# enable_language(C ASM)


# Target architecture
set(TARGET X64_INTEL_HASWELL CACHE STRING "Target architecture")
# set(TARGET X64_INTEL_SANDY_BRIDGE
# set(TARGET X64_INTEL_CORE)
# set(TARGET X64_AMD_BULLDOZER)
# set(TARGET ARMV8A_ARM_CORTEX_A57)
# set(TARGET ARMV7A_ARM_CORTEX_A15)
# set(TARGET GENERIC)

# Linear Algebra library
set(LA HIGH_PERFORMANCE CACHE STRING "Linear algebra optimization level")
# set(LA REFERENCE)
# set(LA BLAS)

# BLAS and LAPACK version (for LA=BLAS in BLASFEO or for BLASFEO_BENCHMARS=ON)
set(REF_BLAS 0 CACHE STRING "Reference blas to use")
# set(REF_BLAS OPENBLAS)
# set(REF_BLAS NETLIB)
# set(REF_BLAS MKL)
# set(REF_BLAS BLIS)
# set(REF_BLAS ATLAS)

# Options
# set(BLASFEO_TESTING ON CACHE BOOL "Tests enabled")
set(BLASFEO_TESTING OFF CACHE BOOL "Tests disabled")
# set(BLASFEO_BENCHMARKS ON CACHE BOOL "Benchmarks enabled")
set(BLASFEO_BENCHMARKS OFF CACHE BOOL "Benchmarks disabled")

# Use C99 extension to math library
set(USE_C99_MATH ON CACHE BOOL "Use C99 extension to math library")

# Compile auxiliary functions with external dependencies
# (for memory allocation and printing)
set(EXT_DEP ON CACHE BOOL "Compile external dependencies in BLASFEO")

set(MIN_KERNEL_SIZE 2x2 CACHE STRING "Minimum kernel size")
# set(MIN_KERNEL_SIZE 4x4 CACHE STRING "Minimum kernel size")

configure_file(${PROJECT_SOURCE_DIR}/blasfeo_target.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/blasfeo_target.h @ONLY)

# C Compiler
# set(CC_COMPILER gcc CACHE STRING "compiler")
# set(CC_COMPILER clang)
# set(CC_COMPILER x86_64-w64-mingw32-gcc)

# build shared library
#set(BUILD_SHARED_LIBS ON CACHE STRING "Build shared libraries")

# installation directory
if(CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
	set(CMAKE_INSTALL_PREFIX "/opt/blasfeo")
endif()

# headers installation directory
set(BLASFEO_HEADERS_INSTALLATION_DIRECTORY "include"
	CACHE STRING "Headers local installation directory")

# Macro level (code size vs performance in assembly kernels): 0 (no macro),
# 1 (all macro but gemm kernel), 2 (all macro)
set(MACRO_LEVEL 0)

# enable runtine checks
set(RUNTIME_CHECKS OFF)

# compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")

	# build informations
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA=${LA}")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET=${TARGET}")

	# debugging flags
	# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
	# set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -g")
endif()

# testing
if(BLASFEO_TESTING MATCHES ON)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTESTING_MODE=1")
endif()

if(BLASFEO_BENCHMARKS MATCHES ON)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBENCHMARKS_MODE=1")
	if(NOT REF_BLAS)
		message(FATAL_ERROR
				"\nBENCHMARKS_MODE: No reference blas selected"
				"\nUse i.e. -DREF_BLAS=OPENBLAS")
	endif()
endif()

if(MIN_KERNEL_SIZE MATCHES 2x2)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIN_KERNEL_SIZE=2")
elseif(MIN_KERNEL_SIZE MATCHES 4x4)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIN_KERNEL_SIZE=4")
endif()

# search directories
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${BLASFEO_PATH}/include")

#
if(${LA} MATCHES HIGH_PERFORMANCE)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA_HIGH_PERFORMANCE")
endif()
if(${LA} MATCHES REFERENCE)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA_REFERENCE")
endif()
if(${LA} MATCHES BLAS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA_BLAS")
endif()

#
if(${RUNTIME_CHECKS})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDIM_CHECK")
endif()

#
if(${USE_C99_MATH})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_C99_MATH")
endif()

#
if(${EXT_DEP})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DEXT_DEP")
endif()

#
if(${MACRO_LEVEL} MATCHES 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMACRO_LEVEL=1")
endif()
if(${MACRO_LEVEL} MATCHES 2)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMACRO_LEVEL=2")
endif()

#
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOS_LINUX")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DOS_LINUX")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOS_MAC")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DOS_MAC")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOS_WINDOWS")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DOS_WINDOWS")
endif()

#
if(${REF_BLAS} MATCHES 0)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
endif(${REF_BLAS} MATCHES 0)

if(${REF_BLAS} MATCHES OPENBLAS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_OPENBLAS -I/opt/openblas/include")
endif(${REF_BLAS} MATCHES OPENBLAS)

if(${REF_BLAS} MATCHES BLIS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_BLIS -std=c99 -I/opt/blis/include/blis")
endif(${REF_BLAS} MATCHES BLIS)

if(${REF_BLAS} MATCHES NETLIB)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_NETLIB -I/opt/netlib/include")
endif(${REF_BLAS} MATCHES NETLIB)

if(${REF_BLAS} MATCHES MKL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_MKL -m64 -I/opt/intel/mkl/include")
endif(${REF_BLAS} MATCHES MKL)

if(${REF_BLAS} MATCHES ATLAS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREF_BLAS_ATLAS")
endif(${REF_BLAS} MATCHES ATLAS)

# architecture-specific flags
if(${TARGET} MATCHES X64_INTEL_HASWELL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_X64_INTEL_HASWELL")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -mavx -mavx2 -mfma")
	endif()
endif()

if(${TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_X64_INTEL_SANDY_BRIDGE")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -mavx")
	endif()
endif()

if(${TARGET} MATCHES X64_INTEL_CORE)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_X64_INTEL_CORE")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -msse3")
	endif()
endif()

if(${TARGET} MATCHES X64_AMD_BULLDOZER)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_X64_AMD_BULLDOZER")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -mavx -mfma")
	endif()
endif()

if(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_ARMV8A_ARM_CORTEX_A57")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DTARGET_ARMV8A_ARM_CORTEX_A57")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+crc+crypto+fp+simd")
	endif()
endif()

if(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_ARMV7A_ARM_CORTEX_A15")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DTARGET_ARMV7A_ARM_CORTEX_A15")
	if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a15")
		set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon-vfpv4")
	endif()
endif()

if(${TARGET} MATCHES GENERIC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_GENERIC")
endif()



# source files

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
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgetrf_pivot_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgebp_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgelqf_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c

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
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dsymv_6_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/c99/kernel_dgemv_4_lib4.c

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


if(BLASFEO_TESTING MATCHES ON)

	file(GLOB AUX_REF_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_ext_dep_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_ext_dep_libref.c
		${PROJECT_SOURCE_DIR}/blas/d_blas3_libref.c
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
install(FILES ${BLASFEO_HEADERS} DESTINATION ${BLASFEO_HEADERS_INSTALLATION_DIRECTORY})

#if(NOT CMAKE_SYSTEM MATCHES "dSpace" AND NOT CMAKE_BUILD_TYPE MATCHES "Release")
if(NOT CMAKE_SYSTEM MATCHES "dSpace")
	# tests
	add_subdirectory(tests)

	# benchmarks
	add_subdirectory(benchmarks)

	# examples
	add_subdirectory(examples)
endif()
