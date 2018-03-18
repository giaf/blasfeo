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
OBJS += s_aux_lib8.o
OBJS += m_aux_lib48.o
endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib8.o
OBJS += m_aux_lib48.o
endif

ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
OBJS += m_aux_lib44.o
endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
OBJS += m_aux_lib44.o
endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
OBJS += m_aux_lib44.o
endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
OBJS += m_aux_lib44.o
endif

ifeq ($(TARGET), GENERIC)
OBJS += d_aux_lib4.o
OBJS += s_aux_lib4.o
OBJS += m_aux_lib44.o
endif

else # LA_REFERENCE | LA_BLAS_WRAPPER

OBJS += d_aux_lib.o
OBJS += s_aux_lib.o
OBJS += m_aux_lib.o

endif # LA choice

ifeq ($(TESTING_MODE), 1)
#testing against reference
OBJS += d_aux_libref.o
OBJS += s_aux_libref.o
endif


#ext dep
ifeq ($(EXT_DEP), 1)

ifeq ($(LA), HIGH_PERFORMANCE)
OBJS += d_aux_ext_dep_lib4.o
OBJS += s_aux_ext_dep_lib4.o
endif
OBJS += d_aux_ext_dep_lib.o
OBJS += s_aux_ext_dep_lib.o
OBJS += v_aux_ext_dep_lib.o
OBJS += i_aux_ext_dep_lib.o

ifeq ($(TESTING_MODE), 1)
OBJS += d_aux_ext_dep_libref.o
OBJS += s_aux_ext_dep_libref.o
endif

endif


obj: $(OBJS)

clean:
	rm -f *.o

d_aux_lib.o: d_aux_lib.c x_aux_lib.c
s_aux_lib.o: s_aux_lib.c x_aux_lib.c
d_aux_ext_dep_lib.o: d_aux_ext_dep_lib.c x_aux_ext_dep_lib0.c x_aux_ext_dep_lib.c
s_aux_ext_dep_lib.o: s_aux_ext_dep_lib.c x_aux_ext_dep_lib0.c x_aux_ext_dep_lib.c
d_aux_ext_dep_lib4.o: d_aux_ext_dep_lib4.c x_aux_ext_dep_lib4.c
s_aux_ext_dep_lib4.o: s_aux_ext_dep_lib4.c x_aux_ext_dep_lib4.c
