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

include ../Makefile.rule

OBJS =
SRCS =

ifeq ($(SANDBOX_MODE), 1)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += kernel_avx2.o
SRCS += kernel_avx2.c
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += kernel_avx.o
SRCS += kernel_avx.c
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += kernel_sse3.o
SRCS += kernel_sse3.c
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += kernel_avx.o
SRCS += kernel_avx.c
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
OBJS += kernel_avx_x86.o
SRCS += kernel_avx_x86.c
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
OBJS += kernel_sse3_x86.o
SRCS += kernel_sse3_x86.c
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += kernel_armv8a.o
SRCS += kernel_armv8a.c
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
OBJS += kernel_armv8a.o
SRCS += kernel_armv8a.c
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += kernel_armv7a.o
SRCS += kernel_armv7a.c
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A7)
OBJS += kernel_armv7a.o
SRCS += kernel_armv7a.c
endif
ifeq ($(TARGET), GENERIC)
OBJS += kernel_generic.o
SRCS += kernel_generic.c
endif

endif

obj: $(OBJS)

disassembly:
	$(CC) $(CFLAGS) -S $(SRCS)



LIBS = ../lib/libblasfeo.a

OBJS2 = test_sandbox.o

CFLAGS += -I../include/

#build: common $(OBJS2)
build: $(OBJS2)
	$(CC) $(CFLAGS) $(OBJS2) $(LIBS) -o test.out

#common:
#	cp ../lib/libblasfeo.a .

run:
	./test.out

clean:
	rm -f *.out
	rm -f *.o
	rm -f *.s

