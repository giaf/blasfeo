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

include ../Makefile.rule

OBJS = 

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += d_aux_lib4.o
#OBJS += s_aux_lib4.o
OBJS += s_aux_lib8.o
endif

ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
endif

ifeq ($(TARGET), GENERIC)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
endif

else # LA_REFERENCE | LA_BLAS

OBJS += d_aux_lib.o
OBJS += s_aux_lib.o

endif # LA choice

OBJS += d_aux_ext_dep_lib.o
OBJS += s_aux_ext_dep_lib.o
OBJS += v_aux_ext_dep_lib.o
OBJS += i_aux_ext_dep_lib.o 

obj: $(OBJS)
ifeq ($(TARGET), X64_INTEL_HASWELL)
	( cd avx2; $(MAKE) obj)
	( cd avx; $(MAKE) obj)
	( cd c99; $(MAKE) obj)
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	( cd avx; $(MAKE) obj)
	( cd c99; $(MAKE) obj)
endif
ifeq ($(TARGET), X64_INTEL_CORE)
	( cd c99; $(MAKE) obj)
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
	( cd c99; $(MAKE) obj)
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
	( cd c99; $(MAKE) obj)
endif
ifeq ($(TARGET), GENERIC)
	( cd c99; $(MAKE) obj)
endif


clean:
	rm -f *.o
	make -C avx2 clean
	make -C avx clean
	make -C c99 clean
