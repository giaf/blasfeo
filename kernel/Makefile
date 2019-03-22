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

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += kernel_align_x64.o
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += kernel_align_x64.o
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += kernel_align_x64.o
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += kernel_align_x64.o
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
OBJS += kernel_align_x86.o
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
OBJS += kernel_align_x86.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += kernel_align_generic.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
OBJS += kernel_align_generic.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += kernel_align_generic.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A7)
OBJS += kernel_align_generic.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += kernel_align_generic.o
endif

obj: $(OBJS)
ifeq ($(TARGET), X64_INTEL_HASWELL)
	( cd avx2; $(MAKE) obj)
	( cd avx; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	( cd avx; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), X64_INTEL_CORE)
	( cd sse3; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)
	( cd fma; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), X86_AMD_JAGUAR)
	( cd avx_x86; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), X86_AMD_BARCELONA)
	( cd sse3_x86; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
	( cd armv8a; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
	( cd armv8a; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
	( cd armv7a; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A7)
	( cd armv7a; $(MAKE) obj)
	( cd generic; $(MAKE) obj)
endif

ifeq ($(TARGET), GENERIC)
	( cd generic; $(MAKE) obj)
endif

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.gcda
	rm -f *.gcno
	rm -f *.gcov
	make -C avx2 clean
	make -C avx clean
	make -C avx_x86 clean
	make -C sse3 clean
	make -C sse3_x86 clean
	make -C fma clean
	make -C armv8a clean
	make -C armv7a clean
	make -C generic clean

