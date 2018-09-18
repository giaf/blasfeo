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

include ../Makefile.rule

LIBS = $(BINARY_DIR)/libblasfeo.a

# add different link library for different REF_BLAS implementation
include ../Makefile.blas

LINK_OBJ =
EX_OBJ =

## dgemm kernel
EX_OBJ = example_kernel_dgemm.o
LINK_OBJ = kernel_dgemm.o

## factorization
# EX_OBJ = example_d_lu_factorization.o
# EX_OBJ = example_s_lu_factorization.o
# LINK_OBJ = tools.o

## riccati
EX_OBJ = example_d_riccati_recursion.o
# EX_OBJ = example_s_riccati_recursion.o
LINK_OBJ = tools.o

## getting started
#EX_OBJ = getting_started.o


CFLAGS += -I../include/


%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@

kernel_%.o: kernel_%.S
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@

$(EX_OBJ):%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ $(BINARY_DIR)/$(LINK_OBJ) -o $(BINARY_DIR)/$@.out $(LIBS)

bin_dir:
	mkdir -p ./$(BINARY_DIR)/

common: bin_dir $(LINK_OBJ)
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)

.DEFAULT_GOAL := build
.PHONY: build

build: common $(EX_OBJ)

run:
	./$(BINARY_DIR)/$(EX_OBJ).out

clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*
