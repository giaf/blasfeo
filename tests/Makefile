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
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo

timing: ../auxiliary/timing.c
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/timing.o

include ../Makefile.blas

# set test verbosity
ifdef TEST_VERBOSITY
ifeq ($(TEST_VERBOSITY), 0)
else
CFLAGS += -DVERBOSE=$(TEST_VERBOSITY)
endif
endif
ifndef TEST_VERBOSITY
CFLAGS += -DVERBOSE=0
endif

ifdef ROUTINE
CFLAGS += -DROUTINE=$(ROUTINE)
else
CFLAGS += -DROUTINE=blasfeo_dgemm_nn
endif


ifndef ROUTINE_CLASS
ROUTINE_CLASS = GEMM
endif
ifeq ($(ROUTINE_CLASS), SYRK)
CFLAGS += -DROUTINE_CLASS_SYRK
endif
ifeq ($(ROUTINE_CLASS), GEMM)
CFLAGS += -DROUTINE_CLASS_GEMM
endif
ifeq ($(ROUTINE_CLASS), TRM)
CFLAGS += -DROUTINE_CLASS_TRM
endif


# select one single test
# ONE_OBJS =
ONE_OBJS = test_d_custom.o
# ONE_OBJS = test_s_custom.o
#
# ONE_OBJS = test_d_aux.o
# ONE_OBJS = test_s_aux.o

# auxiliary test
AUX_OBJS =
AUX_OBJS += test_d_aux.o
AUX_OBJS += test_s_aux.o

# blas test
BLAS_OBJS =
BLAS_OBJS += test_d_blas3.o

test_%.o: test_%.c
	# build executable obj $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) -Wall -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/timing.o $(BINARY_DIR)/$@ -o $(BINARY_DIR)/$@.out $(LIBS)

# ----------- Targets

.DEFAULT_GOAL := obj
.PHONY: obj

obj: one aux blas

bin_dir:
	mkdir -p $(BINARY_DIR)/

common: bin_dir timing
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)
ifeq ($(TESTING_MODE), 1)
	cp ../lib/libblasfeo_ref.a ./$(BINARY_DIR)
endif

one: common $(ONE_OBJS)

aux: common $(AUX_OBJS)
blas: common $(BLAS_OBJS)

run_one:
	./$(BINARY_DIR)/$(ONE_OBJS).out

build_blas: $(BLAS_OBJS)

run_aux:
	./$(BINARY_DIR)/test_d_aux.o.out
	./$(BINARY_DIR)/test_s_aux.o.out

run_blas:
	./$(BINARY_DIR)/test_d_blas3.o.out

build_run_blas: blas run_blas
build_run_aux: aux run_aux

update_blas: $(BLAS_OBJS) run_blas
update_aux: $(AUX_OBJS) run_aux

clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*
