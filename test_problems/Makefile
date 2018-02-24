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

# ----------- Include
include ../Makefile.rule

BINARY_DIR = build/$(LA)/$(TARGET)

# ----------- Envs

LIBS = 
SHARED_LIBS = 
ifeq ($(TESTING_MODE), 1)
LIBS += $(BINARY_DIR)/libblasfeo_ref.a
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo_ref
endif
LIBS += $(BINARY_DIR)/libblasfeo.a -lm
# SHARED_LIBS = -L. $(BINARY_DIR)/libblasfeo.so
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo -lm


# select one single test
OBJS =
#OBJS = test_d_debug.o
#OBJS = test_s_debug.o
#OBJS = test_d_aux.o
#OBJS = test_s_aux.o
#OBJS = test_s_blas.o
#OBJS = kernel_assembly.o test_assembly.o


# auxiliary test
AUX_OBJS =
AUX_OBJS += test_d_aux.o
AUX_OBJS += test_s_aux.o

# blas test
BLAS_OBJS =
BLAS_OBJS += test_d_blas.o
BLAS_OBJS += test_s_blas.o

%.o: %.c
	# build obj
	#
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ -o $(BINARY_DIR)/$@.out $(LIBS)
	#

# ----------- Targets

default: obj

obj: aux blas

bin_dir:
	# create bin folder if not existent
	mkdir -p $(BINARY_DIR)/

common: bin_dir
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)

one_test: common $(OBJS)

aux: common $(AUX_OBJS)

blas: common $(BLAS_OBJS)

run:
	./$(BINARY_DIR)/$(OBJS).out

run_aux:
	./$(BINARY_DIR)/test_d_aux.o.out
	./$(BINARY_DIR)/test_s_aux.o.out

run_blas:
	./$(BINARY_DIR)/test_d_blas.o.out
	./$(BINARY_DIR)/test_s_blas.o.out

clean:
	rm -rf build/*
