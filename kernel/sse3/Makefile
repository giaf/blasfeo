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

include ../../Makefile.rule


ifeq ($(TARGET), X64_INTEL_CORE)
KERNEL_OBJS = \
		kernel_dgemm_4x4_lib4.o \
		kernel_dgemv_4_lib4.o \
		\
		kernel_sgemm_4x4_lib4.o \

endif

ifeq ($(TARGET), $(filter $(TARGET), X64_INTEL_HASWELL X64_INTEL_SANDY_BRIDGE X64_INTEL_CORE X64_AMD_BULLDOZER X64_INTEL_SKYLAKE_X))
KERNEL_ALIGN_OBJS = \
		kernel_align_x64.o \

endif


OBJS = 

ifeq ($(LA), HIGH_PERFORMANCE)
OBJS += $(KERNEL_OBJS)
endif # LA choice

ifeq ($(BLASFEO_HP_API), 1)
OBJS += $(KERNEL_OBJS)
endif

OBJS += $(KERNEL_ALIGN_OBJS)

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.gcda
	rm -f *.gcno
	rm -f *.gcov

kernel_dgemm_4x4_lib4.o: kernel_dgemm_4x4_lib4.S kernel_dgemm_4x4_lib.S
kernel_sgemm_4x4_lib4.o: kernel_sgemm_4x4_lib4.S kernel_sgemm_4x4_lib.S
