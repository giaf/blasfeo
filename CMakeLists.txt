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

cmake_minimum_required(VERSION 2.8.11)

project(blasfeo C ASM)

set(CMAKE_VERBOSE_MAKEFILE ON)

# enable_language(C ASM)


# Target architecture
set(TARGET X64_INTEL_HASWELL CACHE STRING "Target architecture")
# set(TARGET X64_INTEL_SANDY_BRIDGE CACHE STRING "Target architecture")
# set(TARGET X64_INTEL_CORE CACHE STRING "Target architecture")
# set(TARGET X64_AMD_BULLDOZER CACHE STRING "Target architecture")
# set(TARGET ARMV8A_ARM_CORTEX_A57 CACHE STRING "Target architecture")
# set(TARGET ARMV8A_ARM_CORTEX_A53 CACHE STRING "Target architecture")
# set(TARGET ARMV7A_ARM_CORTEX_A15 CACHE STRING "Target architecture")
# set(TARGET ARMV7A_ARM_CORTEX_A7 CACHE STRING "Target architecture")
# set(TARGET GENERIC)

# Populate a list of allowable targets and link it to the option
set(ALLOWED_TARGETS
		X64_AUTOMATIC
		X64_INTEL_HASWELL
		X64_INTEL_SANDY_BRIDGE
		X64_INTEL_CORE
		X64_AMD_BULLDOZER
		ARMV8A_ARM_CORTEX_A57
		ARMV8A_ARM_CORTEX_A53
		ARMV7A_ARM_CORTEX_A15
		ARMV7A_ARM_CORTEX_A7
		GENERIC
		)
set_property(CACHE TARGET PROPERTY STRINGS ${ALLOWED_TARGETS})

# This list contains the targets to try when the user sets the target to
# X64_AUTOMATIC. The order they appear in this list is the order in which
# they are tested, and the first working (both compile and run) will be
# selected for use
set(X64_AUTOMATIC_TARGETS
		X64_INTEL_HASWELL
		X64_AMD_BULLDOZER
		X64_INTEL_SANDY_BRIDGE
		X64_INTEL_CORE
		GENERIC
		)


list(FIND ALLOWED_TARGETS ${TARGET} isvalid)
if(${isvalid} EQUAL -1)
	message(FATAL_ERROR "Target ${TARGET} is not supported")
endif()

# Linear Algebra backend
set(LA HIGH_PERFORMANCE CACHE STRING "Linear algebra backend")
# set(LA REFERENCE CACHE STRING "Linear algebra backend")
# set(LA BLAS CACHE STRING "Linear algebra backend")

# Populate the list of allowed LA backends and link it to the option
set(ALLOWED_BACKENDS
	  HIGH_PERFORMANCE
	  REFERENCE
	  BLAS
	  )
set_property(CACHE LA PROPERTY STRINGS ${ALLOWED_BACKENDS})

list(FIND ALLOWED_BACKENDS ${LA} isvalid)
if(${isvalid} EQUAL -1)
	message(FATAL_ERROR "LA backend ${LA} is not supported")
endif()

# BLAS and LAPACK version (for LA=BLAS in BLASFEO or for BLASFEO_BENCHMARS=ON)
set(EXTERNAL_BLAS 0 CACHE STRING "Reference blas to use")
# set(EXTERNAL_BLAS OPENBLAS CACHE STRING "Reference blas to use")
# set(EXTERNAL_BLAS NETLIB CACHE STRING "Reference blas to use")
# set(EXTERNAL_BLAS MKL CACHE STRING "Reference blas to use")
# set(EXTERNAL_BLAS BLIS CACHE STRING "Reference blas to use")
# set(EXTERNAL_BLAS ATLAS CACHE STRING "Reference blas to use")

# Populate the list of allowed exernal BLAS tools and link it to the option
set(ALLOWED_EXTERNAL_BLAS
		0
		OPENBLAS
		NETLIB
		MKL
		BLIS
		ATLAS
		)
set_property(CACHE EXTERNAL_BLAS PROPERTY STRINGS ${ALLOWED_EXTERNAL_BLAS})

list(FIND ALLOWED_EXTERNAL_BLAS ${EXTERNAL_BLAS} isvalid)
if(${isvalid} EQUAL -1)
	message(FATAL_ERROR "External BLAS ${EXTERNAL_BLAS} is not supported")
endif()

# Pass the allowed items up to the parent scope (if there is one)
# This is detected by comparing the project name to the one set earlier in this file
if(NOT ${CMAKE_PROJECT_NAME} STREQUAL blasfeo)
	set(BLASFEO_ALLOWED_TARGETS ${ALLOWED_TARGETS} PARENT_SCOPE)
	set(BLASFEO_ALLOWED_BACKENDS ${ALLOWED_BACKENDS} PARENT_SCOPE)
	set(BLASFEO_ALLOWED_EXTERNAL_BLAS ${ALLOWED_EXTERNAL_BLAS} PARENT_SCOPE)
endif()

# Stack buffer size (malloc for larger)
set(K_MAX_STACK 300 CACHE STRING "Maximum k value using stack memory")

# Options
# set(BLASFEO_TESTING ON CACHE BOOL "Tests enabled")
set(BLASFEO_TESTING OFF CACHE BOOL "Tests disabled")

# set(BLASFEO_BENCHMARKS ON CACHE BOOL "Benchmarks enabled")
set(BLASFEO_BENCHMARKS OFF CACHE BOOL "Benchmarks disabled")

set(BLASFEO_EXAMPLES ON CACHE BOOL "Examples enabled")
# set(BLASFEO_EXAMPLES OFF CACHE BOOL "Examples disabled")

# Use C99 extension to math library
set(USE_C99_MATH ON CACHE BOOL "Use C99 extension to math library")

# Compile auxiliary functions with external dependencies
# (for memory allocation and printing)
set(EXT_DEP ON CACHE BOOL "Compile external dependencies in BLASFEO")

# Skip BLASFEO utils for unsupported systems and for Realease build
if(CMAKE_SYSTEM MATCHES "dSpace" OR CMAKE_BUILD_TYPE MATCHES "Release")
	set(BLASFEO_TESTING OFF CACHE BOOL "Examples disabled" FORCE)
	set(BLASFEO_BENCHMARKS OFF CACHE BOOL "Benchmarks disabled" FORCE)
	set(BLASFEO_EXAMPLES OFF CACHE BOOL "Examples disabled" FORCE)
endif()

configure_file(${PROJECT_SOURCE_DIR}/blasfeo_target.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/blasfeo_target.h @ONLY)

# C Compiler
# set(CC_COMPILER gcc CACHE STRING "compiler")
# set(CC_COMPILER clang)
# set(CC_COMPILER x86_64-w64-mingw32-gcc)
if(CMAKE_C_COMPILER_ID MATCHES MSVC)
	if(NOT ${TARGET} MATCHES GENERIC)
		message( FATAL_ERROR "MSVC compiler only supported for TARGET=GENERIC")
	endif()
endif()

# build shared library
set(BUILD_SHARED_LIBS OFF CACHE STRING "Build shared libraries")

# installation directory
set(CMAKE_INSTALL_PREFIX "/opt/blasfeo" CACHE STRING "Installation path")

# headers installation directory
set(BLASFEO_HEADERS_INSTALLATION_DIRECTORY "include" CACHE STRING "Headers local installation directory")

# Macro level (code size vs performance in assembly kernels): 0 (no macro),
# 1 (all macro but gemm kernel), 2 (all macro)
set(MACRO_LEVEL 0)

# enable runtine checks
set(RUNTIME_CHECKS OFF)

# testing
if(BLASFEO_TESTING MATCHES ON)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTESTING_MODE=1")
endif()

if(BLASFEO_BENCHMARKS MATCHES ON)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBENCHMARKS_MODE=1")
endif()

# search directories
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${BLASFEO_PATH}/include")

# Specify the linear algebra package to use
message(STATUS "Using linear algebra: ${LA}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLA_${LA}")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DK_MAX_STACK=${K_MAX_STACK}")

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

# External BLAS flags
set(C_FLAGS_BLAS_0        "-DEXTERNAL_BLAS_NONE")
set(C_FLAGS_BLAS_OPENBLAS "-DEXTERNAL_BLAS_OPENBLAS -I/opt/openblas/include")
set(C_FLAGS_BLAS_BLIS     "-DEXTERNAL_BLAS_BLIS -std=c99 -I/opt/blis/include/blis")
set(C_FLAGS_BLAS_NETLIB   "-DEXTERNAL_BLAS_NETLIB -I/opt/netlib/include")
set(C_FLAGS_BLAS_MKL      "-DEXTERNAL_BLAS_MKL -m64 -I/opt/intel/mkl/include")
set(C_FLAGS_BLAS_ATLAS    "-DEXTERNAL_BLAS_ATLAS")

message(STATUS "Using external BLAS: ${EXTERNAL_BLAS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_FLAGS_BLAS_${EXTERNAL_BLAS}}")


# common C flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fPIC")
endif()


# architecture-specific C flags
set(C_FLAGS_TARGET_X64_INTEL_HASWELL      "-m64 -mavx -mavx2 -mfma")
set(C_FLAGS_TARGET_X64_INTEL_SANDY_BRIDGE "-m64 -mavx")
set(C_FLAGS_TARGET_X64_INTEL_CORE         "-m64 -msse3")
set(C_FLAGS_TARGET_X64_AMD_BULLDOZER      "-m64 -mavx -mfma")
set(C_FLAGS_TARGET_ARMV8A_ARM_CORTEX_A57  "-march=armv8-a+crc+crypto+simd")
set(C_FLAGS_TARGET_ARMV8A_ARM_CORTEX_A53  "-march=armv8-a+crc+crypto+simd")
set(C_FLAGS_TARGET_ARMV7A_ARM_CORTEX_A15  "-marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a15")
set(C_FLAGS_TARGET_ARMV7A_ARM_CORTEX_A7   "-marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a7")
set(C_FLAGS_TARGET_GENERIC                "")

# architecture-specific assembly flags
set(ASM_FLAGS_TARGET_X64_INTEL_HASWELL      "")
set(ASM_FLAGS_TARGET_X64_INTEL_SANDY_BRIDGE "")
set(ASM_FLAGS_TARGET_X64_INTEL_CORE         "")
set(ASM_FLAGS_TARGET_X64_AMD_BULLDOZER      "")
set(ASM_FLAGS_TARGET_ARMV8A_ARM_CORTEX_A57  "")
set(ASM_FLAGS_TARGET_ARMV8A_ARM_CORTEX_A53  "")
set(ASM_FLAGS_TARGET_ARMV7A_ARM_CORTEX_A15  "-mfpu=neon-vfpv4")
set(ASM_FLAGS_TARGET_ARMV7A_ARM_CORTEX_A7   "-mfpu=neon-vfpv4")
set(ASM_FLAGS_TARGET_GENERIC                "")


if(${TARGET} MATCHES X64_AUTOMATIC)
	include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/X64AutomaticTargetSelection.cmake)
	X64AutomaticTargetSelection()
	message(STATUS "Detected target ${TARGET}")
else()
	include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestSingleTarget.cmake)
	TestSingleTarget()
	message(STATUS "Compiling for target: ${TARGET}")
endif()




# Append the appropriate flags based on the architecture and compiler
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_${TARGET}")
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -DTARGET_${TARGET}")

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_FLAGS_TARGET_${TARGET}}")
	set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} ${ASM_FLAGS_TARGET_${TARGET}}")
endif()


# source files

if(${LA} MATCHES HIGH_PERFORMANCE)

	if(${TARGET} MATCHES X64_INTEL_HASWELL)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib8.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib48.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_8x8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_8x4_lib4.c
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_10xX_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x2_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_6xX_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x2_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgemv_8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dsymv_6_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgebp_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgelqf_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_dgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dpack_lib4.S

			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_24x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_16x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_8x8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx2/kernel_sgemm_8x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_8x4_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_diag_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgead_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgetr_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgecpsc_lib8.S

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_x64.S
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib8.c
			)

	endif(${TARGET} MATCHES X64_INTEL_HASWELL)

	if(${TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib8.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib48.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_12x4_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_10xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_8x4_lib4.c
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_8x2_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_6xX_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x4_lib4.S
#			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_4x2_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_12_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dsymv_6_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgemv_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgebp_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgelqf_4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dpack_lib4.S

			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_16x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_8x8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_8x4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_8x4_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemm_diag_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_8_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgemv_4_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib8.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgead_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgecpsc_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_sgetr_lib8.S
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/avx/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_x64.S
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib8.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib8.c
			)

	endif(${TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

	if(${TARGET} MATCHES X64_INTEL_CORE)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/sse3/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_x64.S
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES X64_INTEL_CORE)

	if(${TARGET} MATCHES X64_AMD_BULLDOZER)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/fma/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_x64.S
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES X64_AMD_BULLDOZER)

	if(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_16x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_generic.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

	if(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A53)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC

			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_dpack_lib4.S


			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_16x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x8_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv8a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_generic.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV8A_ARM_CORTEX_A53)

	if(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_generic.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

	if(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A7)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_dgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_12x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_8x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/armv7a/kernel_sgemm_4x4_lib4.S
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_generic.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES ARMV7A_ARM_CORTEX_A7)

	if(${TARGET} MATCHES GENERIC)

		file(GLOB AUX_SRC
			${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
			${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib4.c
			${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib44.c
			)

		file(GLOB KERNEL_SRC
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dsymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgeqrf_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dgetr_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_dpack_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_4x4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemm_diag_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgemv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_ssymv_4_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetrf_pivot_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgecp_lib4.c
			${PROJECT_SOURCE_DIR}/kernel/generic/kernel_sgetr_lib4.c

			${PROJECT_SOURCE_DIR}/kernel/kernel_align_generic.c
			)

		file(GLOB BLAS_SRC
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib4.c

			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib4.c
			${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib4.c
			)

	endif(${TARGET} MATCHES GENERIC)

else(${LA} MATCHES HIGH_PERFORMANCE) # REFERENCE BLAS

	file(GLOB AUX_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/blasfeo_stdlib.c
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_lib.c
		${PROJECT_SOURCE_DIR}/auxiliary/m_aux_lib.c
		)

	file(GLOB BLAS_SRC
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas1_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas2_diag_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_diag_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_lapack_lib.c

		${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas1_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas2_diag_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/s_blas3_diag_lib.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/s_lapack_lib.c
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
		${PROJECT_SOURCE_DIR}/auxiliary/timing.c
		)

endif()


if(BLASFEO_TESTING MATCHES ON)

	file(GLOB AUX_REF_SRC
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/d_aux_ext_dep_libref.c
		${PROJECT_SOURCE_DIR}/auxiliary/s_aux_ext_dep_libref.c
		${PROJECT_SOURCE_DIR}/blasfeo_api/d_blas3_libref.c
		)

	set(BLASFEO_REF_SRC ${AUX_REF_SRC})
	add_library(blasfeo_ref ${BLASFEO_REF_SRC})
	set_target_properties(blasfeo_ref PROPERTIES LINKER_LANGUAGE C)
endif()

set(BLASFEO_SRC ${AUX_SRC} ${KERNEL_SRC} ${BLAS_SRC} ${EXT_SRC})

# add library
add_library(blasfeo ${BLASFEO_SRC})

if(EMBEDDED_TARGET MATCHES XILINX_NONE_ELF )
	target_link_libraries(blasfeo PUBLIC -Wl,--start-group xil c gcc -Wl,--end-group)
endif()


target_include_directories(blasfeo
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include/blasfeo/include>)


install(TARGETS blasfeo EXPORT blasfeoConfig
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin)


install(EXPORT blasfeoConfig DESTINATION cmake)

file(GLOB_RECURSE BLASFEO_HEADERS "include/*.h")
install(FILES ${BLASFEO_HEADERS} DESTINATION ${BLASFEO_HEADERS_INSTALLATION_DIRECTORY})

# tests
if(BLASFEO_TESTING MATCHES ON)
	add_subdirectory(tests)
endif()

# benchmarks
if(BLASFEO_BENCHMARKS MATCHES ON)
	add_subdirectory(benchmarks)
endif()

# examples
if(BLASFEO_EXAMPLES MATCHES ON)
	add_subdirectory(examples)
endif()
