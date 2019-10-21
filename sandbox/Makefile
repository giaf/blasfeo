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
ifeq ($(TARGET), $(filter $(TARGET), ARMV8A_ARM_CORTEX_A57 ARMV8A_ARM_CORTEX_A53))
OBJS += kernel_armv8a.o
SRCS += kernel_armv8a.c
endif
ifeq ($(TARGET), $(filter $(TARGET), ARMV7A_ARM_CORTEX_A15 ARMV7A_ARM_CORTEX_A9 ARMV7A_ARM_CORTEX_A7))
OBJS += kernel_armv7a.o
SRCS += kernel_armv7a.c
endif
ifeq ($(TARGET), GENERIC)
OBJS += kernel_generic.o
SRCS += kernel_generic.c
endif

#OBJS += kernel_c_dummy.o
#OBJS += kernel_asm_dummy.o

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

