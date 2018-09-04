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

ifeq ($(BLAS_API), 1)

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += dgemm.o dpotrf.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += dgemm.o dpotrf.o
endif

endif

endif

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s

