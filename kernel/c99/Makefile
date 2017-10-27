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

include ../../Makefile.rule

OBJS =

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += kernel_dgemv_4_lib4.o \
		\
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o
endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += kernel_dgemv_4_lib4.o \
		\
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \

endif

ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_diag_lib4.o \
		kernel_dgemv_4_lib4.o \
		kernel_dsymv_4_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgeqrf_4_lib4.o \
		\
		kernel_dgecp_lib4.o \
		kernel_dgetr_lib4.o \
		\
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \
		\
		kernel_sgecp_lib4.o \
		kernel_sgetr_lib4.o
endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_diag_lib4.o \
		kernel_dgemv_4_lib4.o \
		kernel_dsymv_4_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgeqrf_4_lib4.o \
		\
		kernel_dgecp_lib4.o \
		kernel_dgetr_lib4.o

OBJS += \
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \
		\
		kernel_sgecp_lib4.o \
		kernel_sgetr_lib4.o
endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_diag_lib4.o \
		kernel_dgemv_4_lib4.o \
		kernel_dsymv_4_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgeqrf_4_lib4.o
		\
		kernel_dgecp_lib4.o \
		kernel_dgetr_lib4.o

OBJS += \
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \
		\
		kernel_sgecp_lib4.o \
		kernel_sgetr_lib4.o
endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_diag_lib4.o \
		kernel_dgemv_4_lib4.o \
		kernel_dsymv_4_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgeqrf_4_lib4.o \
		\
		kernel_dgecp_lib4.o \
		kernel_dgetr_lib4.o

OBJS += \
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \
		\
		kernel_sgecp_lib4.o \
		kernel_sgetr_lib4.o
endif

ifeq ($(TARGET), GENERIC)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_diag_lib4.o \
		kernel_dgemv_4_lib4.o \
		kernel_dsymv_4_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgeqrf_4_lib4.o \
		\
		kernel_dgecp_lib4.o \
		kernel_dgetr_lib4.o

OBJS += \
		kernel_sgemm_4x4_lib4.o \
		kernel_sgemm_diag_lib4.o \
		kernel_sgemv_4_lib4.o \
		kernel_ssymv_4_lib4.o \
		kernel_sgetrf_pivot_4_lib4.o \
		\
		kernel_sgecp_lib4.o \
		kernel_sgetr_lib4.o
endif

else # LA_REFERENCE | LA_BLAS

endif # LA choice

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s

