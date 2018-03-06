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
LIBS += $(BINARY_DIR)/libblasfeo.a
# SHARED_LIBS = -L. $(BINARY_DIR)/libblasfeo.so
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo

include ../Makefile.blas

# select one single test
ONE_OBJS =
# ONE_OBJS = test_d_debug.o
# ONE_OBJS = test_s_debug.o
# ONE_OBJS = test_d_aux.o
# ONE_OBJS = test_s_aux.o
# ONE_OBJS = test_s_blas.o

# auxiliary test
AUX_OBJS =
AUX_OBJS += test_d_aux.o
AUX_OBJS += test_s_aux.o

# blas test
BLAS_OBJS =
# BLAS_OBJS += test_d_blas.o
# BLAS_OBJS += test_s_blas.o

test_%.o: test_%.c
	@echo
	@echo build executable obj $(BINARY_DIR)/$@
	@echo
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ -o $(BINARY_DIR)/$@.out $(LIBS)

# ----------- Targets

.DEFAULT_GOAL := obj
.PHONY: obj

obj: aux blas

bin_dir:
	mkdir -p $(BINARY_DIR)/

common: bin_dir
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)
ifeq ($(TESTING_MODE), 1)
	cp ../lib/libblasfeo_ref.a ./$(BINARY_DIR)
endif

one: common $(ONE_LINK_OBJS) $(ONE_OBJS)
	@echo
	@echo build $(ONE_LINK_OBJS) $(ONE_OBJS)
	@echo


aux: common $(AUX_OBJS)

blas: common $(BLAS_OBJS)

run_one:
	./$(BINARY_DIR)/$(ONE_OBJS).out

run_aux:
	./$(BINARY_DIR)/test_d_aux.o.out
	./$(BINARY_DIR)/test_s_aux.o.out

run_blas:
	# ./$(BINARY_DIR)/test_d_blas.o.out
	# ./$(BINARY_DIR)/test_s_blas.o.out

clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*
