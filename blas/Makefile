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
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib8.o s_blas2_lib8.o s_blas2_diag_lib.o s_blas3_lib8.o s_blas3_diag_lib8.o s_lapack_lib8.o
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib8.o s_blas2_lib8.o s_blas2_diag_lib.o s_blas3_lib8.o s_blas3_diag_lib8.o s_lapack_lib8.o
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), X86_AMD_JAGUAR)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), X86_AMD_BARCELONA)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A53)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif

else # LA_REFERENCE | LA_BLAS

OBJS += d_blas1_lib.o d_blas2_lib.o d_blas2_diag_lib.o d_blas3_lib.o d_blas3_diag_lib.o d_lapack_lib.o
OBJS += s_blas1_lib.o s_blas2_lib.o s_blas2_diag_lib.o s_blas3_lib.o s_blas3_diag_lib.o s_lapack_lib.o

endif # LA choice

ifeq ($(TESTING_MODE), 1)
#testing against reference
OBJS += d_blas3_libref.o
OBJS += d_lapack_libref.o
OBJS += s_blas3_libref.o
endif


obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s

d_blas1_lib.o: d_blas1_lib.c x_blas1_lib.c
s_blas1_lib.o: s_blas1_lib.c x_blas1_lib.c
d_blas2_lib.o: d_blas2_lib.c x_blas2_lib.c
s_blas2_lib.o: s_blas2_lib.c x_blas2_lib.c
d_blas2_diag_lib.o: d_blas2_diag_lib.c x_blas2_diag_lib.c
s_blas2_diag_lib.o: s_blas2_diag_lib.c x_blas2_diag_lib.c
d_blas3_lib.o: d_blas3_lib.c x_blas3_lib.c
s_blas3_lib.o: s_blas3_lib.c x_blas3_lib.c
d_blas3_diag_lib.o: d_blas3_diag_lib.c x_blas3_diag_lib.c
s_blas3_diag_lib.o: s_blas3_diag_lib.c x_blas3_diag_lib.c
d_lapack_lib.o: d_lapack_lib.c x_lapack_lib.c
d_lapack_libref.o: d_lapack_libref.c x_lapack_lib.c
s_lapack_lib.o: s_lapack_lib.c x_lapack_lib.c
