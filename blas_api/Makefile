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

ifeq ($(BLAS_API), 1)

ifeq ($(MF), PANELMAJ)

ifeq ($(LA), HIGH_PERFORMANCE)

OBJS += dcopy.o
OBJS += daxpy.o
OBJS += ddot.o
OBJS += dgemm.o
OBJS += dsyrk.o
OBJS += dtrmm.o
OBJS += dtrsm.o
OBJS += dlaswp.o
OBJS += dgesv.o
OBJS += dgetrf.o
OBJS += dgetrf_np.o
OBJS += dgetrs.o
OBJS += dposv.o
OBJS += dpotrf.o
OBJS += dpotrs.o
OBJS += dtrtrs.o

OBJS += saxpy.o
OBJS += sdot.o
OBJS += sgemm.o
OBJS += strsm.o

endif # HIGH_PERFROMANCE

else # MF COLMAJ ## ifeq ($(LA), REFERENCE)

OBJS += dgemm_ref.o
OBJS += dsyrk_ref.o
OBJS += dtrsm_ref.o
#OBJS += dpotrf_ref.o

OBJS += sgemm_ref.o
#OBJS += strsm_ref.o
#OBJS += spotrf_ref.o

endif # REFERENCE

endif # BLAS_API

obj: $(OBJS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.gcda
	rm -f *.gcno
	rm -f *.gcov

dgemm_ref.o: dgemm_ref.c xgemm_ref.c
dsyrk_ref.o: dsyrk_ref.c xsyrk_ref.c
dtrsm_ref.o: dtrsm_ref.c xtrsm_ref.c
dpotrf_ref.o: dpotrf_ref.c xpotrf_ref.c

sgemm_ref.o: sgemm_ref.c xgemm_ref.c
strsm_ref.o: strsm_ref.c xtrsm_ref.c
spotrf_ref.o: spotrf_ref.c xpotrf_ref.c
