###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS For Embedded Optimization.                                                      #
# Copyright (C) 2016-2018 by Gianluca Frison.                                                     #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# This program is free software: you can redistribute it and/or modify                            #
# it under the terms of the GNU General Public License as published by                            #
# the Free Software Foundation, either version 3 of the License, or                               #
# (at your option) any later version                                                              #.
#                                                                                                 #
# This program is distributed in the hope that it will be useful,                                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   #
# GNU General Public License for more details.                                                    #
#                                                                                                 #
# You should have received a copy of the GNU General Public License                               #
# along with this program.  If not, see <https://www.gnu.org/licenses/>.                          #
#                                                                                                 #
# The authors designate this particular file as subject to the "Classpath" exception              #
# as provided by the authors in the LICENSE file that accompained this code.                      #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################

# ----------- Include
include ../Makefile.rule

# ----------- Envs

LIBS =
SHARED_LIBS =

ifeq ($(TESTING_MODE), 1)
LIBS += $(BINARY_DIR)/libblasfeo_ref.a
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo_ref
endif

LIBS += $(BINARY_DIR)/libblasfeo.a
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo

include ../Makefile.blas

# set test verbosity
ifdef VERBOSE
ifeq ($(VERBOSE), 0)
else
CFLAGS += -DVERBOSE=$(VERBOSE)
endif
endif

ifndef VERBOSE
CFLAGS += -DVERBOSE=0
endif
ifdef ROUTINE
CFLAGS += -DROUTINE=$(ROUTINE)
endif

ifeq ($(PRECISION), d)
CFLAGS += -DPRECISION_DOUBLE
endif
ifeq ($(PRECISION), s)
CFLAGS += -DPRECISION_SINGLE
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
ifeq ($(ROUTINE_CLASS), GETRF)
CFLAGS += -DROUTINE_CLASS_GETRF
endif


# select one single test
ONE_OBJS = test_d_custom.o
# ONE_OBJS = test_s_custom.o

OBJS = test.o

%.o: %.c
	#
	# build executable obj $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ -o $(BINARY_DIR)/$@.out $(LIBS)
	#

# ----------- Targets

.DEFAULT_GOAL := obj
.PHONY: obj

obj: one aux blas

bin_dir:
	mkdir -p $(BINARY_DIR)/

common: bin_dir
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)
ifeq ($(TESTING_MODE), 1)
	cp ../lib/libblasfeo_ref.a ./$(BINARY_DIR)
endif

one: common $(ONE_OBJS)

run_one:
	./$(BINARY_DIR)/$(ONE_OBJS).out

build: $(OBJS)

run:
	./$(BINARY_DIR)/test.o.out
	#

update: build run
full: common build run

clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*
