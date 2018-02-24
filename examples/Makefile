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

BINARY_DIR = build/$(LA)/$(TARGET)
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
# EX_OBJ = example_d_riccati_recursion.o
# EX_OBJ = example_s_riccati_recursion.o
# LINK_OBJ = tools.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@

kernel_%.o: kernel_%.S
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@

$(EX_OBJ):%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ $(BINARY_DIR)/$(LINK_OBJ) -o $(BINARY_DIR)/$@.out $(LIBS)

bin_dir:
	mkdir -p ./$(BINARY_DIR)/

common: bin_dir
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)

default: build

build: $(LINK_OBJ) $(EX_OBJ) common

run:
	./$(BINARY_DIR)/$(EX_OBJ).out

clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*
