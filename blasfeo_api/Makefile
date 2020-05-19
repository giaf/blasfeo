###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS for embedded optimization.                                                      #
# Copyright (C) 2019 by Gianluca Frison.                                                          #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# The 2-Clause BSD License                                                                        #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without                              #
# modification, are permitted provided that the following conditions are met:                     #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
#    list of conditions and the following disclaimer.                                             #
# 2. Redistributions in binary form must reproduce the above copyright notice,                    #
#    this list of conditions and the following disclaimer in the documentation                    #
#    and/or other materials provided with the distribution.                                       #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
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
ifeq ($(TARGET), $(filter $(TARGET), ARMV8A_ARM_CORTEX_A57 ARMV8A_ARM_CORTEX_A53))
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), $(filter $(TARGET), ARMV7A_ARM_CORTEX_A15 ARMV7A_ARM_CORTEX_A9 ARMV7A_ARM_CORTEX_A7))
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif
ifeq ($(TARGET), GENERIC)
OBJS += d_blas1_lib4.o d_blas2_lib4.o d_blas2_diag_lib.o d_blas3_lib4.o d_blas3_diag_lib4.o d_lapack_lib4.o
OBJS += s_blas1_lib4.o s_blas2_lib4.o s_blas2_diag_lib.o s_blas3_lib4.o s_blas3_diag_lib4.o s_lapack_lib4.o
endif

endif

ifeq ($(LA), EXTERNAL_BLAS_WRAPPER)

OBJS += d_blas1_lib.o d_blas2_lib.o d_blas2_diag_lib.o d_blas3_lib.o d_blas3_diag_lib.o d_lapack_lib.o
OBJS += s_blas1_lib.o s_blas2_lib.o s_blas2_diag_lib.o s_blas3_lib.o s_blas3_diag_lib.o s_lapack_lib.o

endif # LA choice

ifeq ($(LA), REFERENCE)

OBJS += d_blas1_lib.o d_blas2_lib.o d_lapack_lib.o
OBJS += s_blas1_lib.o s_blas2_lib.o s_lapack_lib.o

endif # LA choice

ifeq ($(TESTING_MODE), 1)
#testing against reference
OBJS += s_blas1_libref.o
OBJS += s_blas2_libref.o
OBJS += s_lapack_libref.o

OBJS += d_blas1_libref.o
OBJS += d_blas2_libref.o
OBJS += d_lapack_libref.o
endif


obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.gcda
	rm -f *.gcno
	rm -f *.gcov

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
