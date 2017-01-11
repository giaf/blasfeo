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
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
endif
ifeq ($(TARGET), X64_INTEL_CORE)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
endif

else # LA_REFERENCE | LA_BLAS

OBJS += d_blas1_lib.o d_blas2_lib.o d_blas3_lib.o d_blas3_diag_lib.o d_lapack_lib.o
OBJS +=

endif # LA choice

OBJS +=
OBJS += s_blas2_lib4.o s_blas3_lib4.o s_lapack_lib4.o

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s

