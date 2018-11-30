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

include ../../Makefile.rule

OBJS = 

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemm_8x4_lib4.o \
		kernel_dgemm_8x8_lib4.o \
		kernel_dgemm_12x4_lib4.o \
		kernel_dgemv_8_lib4.o \
		kernel_dsymv_6_lib4.o \
		kernel_dgetrf_pivot_4_lib4.o \
		kernel_dgetrf_pivot_lib.o \
		kernel_dgebp_lib4.o \
		kernel_dgelqf_4_lib4.o \
		kernel_dgetr_lib4.o \
		\
		kernel_sgemm_24x4_lib8.o \
		kernel_sgemm_16x4_lib8.o \
		kernel_sgemm_8x8_lib8.o \
		kernel_sgemm_8x4_lib8.o \

endif

else # LA_REFERENCE | LA_BLAS

endif # LA choice

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s

kernel_dgemm_12x4_lib4.o: kernel_dgemm_12x4_lib4.S kernel_dgemm_12x4_lib.S
kernel_dgemm_8x8_lib4.o: kernel_dgemm_8x8_lib4.S kernel_dgemm_8x8_lib.S
kernel_dgemm_8x4_lib4.o: kernel_dgemm_8x4_lib4.S kernel_dgemm_8x4_lib.S
kernel_dgemm_4x4_lib4.o: kernel_dgemm_4x4_lib4.S kernel_dgemm_4x4_lib.S
