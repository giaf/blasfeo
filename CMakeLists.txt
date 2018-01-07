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

target_include_directories(blasfeo
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS blasfeo EXPORT blasfeoConfig
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin)

install(EXPORT blasfeoConfig DESTINATION cmake)

file(GLOB_RECURSE BLASFEO_HEADERS "include/*.h")
install(FILES ${BLASFEO_HEADERS} DESTINATION include)
