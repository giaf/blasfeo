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
LIBS = $(BINARY_DIR)/libblasfeo.a

include ../Makefile.blas

# all benchmarks
OBJS =
OBJS += benchmark_d_blas.o
OBJS += benchmark_s_blas.o

%.o: %.c
	# build obj
	$(CC) $(CFLAGS) -c $< -o $(BINARY_DIR)/$@
	$(CC) $(CFLAGS) $(BINARY_DIR)/$@ -o $(BINARY_DIR)/$@.out $(LIBS)

# ----------- Targets

bin_dir:
	# create bin folder if not existent
	mkdir -p $(BINARY_DIR)/

common: bin_dir
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)
	echo $(OBJS)

.DEFAULT_GOAL := build
.PHONY: build

build: common $(OBJS)

run:
	./$(BINARY_DIR)/benchmark_d_blas.o.out
	./$(BINARY_DIR)/benchmark_s_blas.o.out


clean:
	rm -rf ./*.o
	rm -rf ./*.out
	rm -rf ./$(BINARY_DIR)/*

clean_all: clean
	rm -rf ./build/*


# single benchmarks


BLAS_OBJS =
RUN_BLAS_OBJS =


# dgemm_nt

BLAS_OBJS += dgemm_nt
RUN_BLAS_OBJS += run_dgemm_nt

dgemm_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemm_nt_benchmark.o -DDOUBLE_PRECISION -DGEMM_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemm_nt_benchmark.o -o $(BINARY_DIR)/dgemm_nt_benchmark.out $(LIBS)

run_dgemm_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemm_nt
	./$(BINARY_DIR)/dgemm_nt_benchmark.out > ./$(BINARY_DIR)/data/dgemm_nt.m


# dgemm_nn

BLAS_OBJS += dgemm_nn
RUN_BLAS_OBJS += run_dgemm_nn

dgemm_nn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemm_nn_benchmark.o -DDOUBLE_PRECISION -DGEMM_NN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemm_nn_benchmark.o -o $(BINARY_DIR)/dgemm_nn_benchmark.out $(LIBS)

run_dgemm_nn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemm_nn
	./$(BINARY_DIR)/dgemm_nn_benchmark.out > ./$(BINARY_DIR)/data/dgemm_nn.m

all: $(BLAS_OBJS)

run_all: $(RUN_BLAS_OBJS)


# sgemm_nt

BLAS_OBJS += sgemm_nt
RUN_BLAS_OBJS += run_sgemm_nt

sgemm_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemm_nt_benchmark.o -DSINGLE_PRECISION -DGEMM_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemm_nt_benchmark.o -o $(BINARY_DIR)/sgemm_nt_benchmark.out $(LIBS)

run_sgemm_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemm_nt
	./$(BINARY_DIR)/sgemm_nt_benchmark.out > ./$(BINARY_DIR)/data/sgemm_nt.m


# sgemm_nt

BLAS_OBJS += sgemm_nn
RUN_BLAS_OBJS += run_sgemm_nn

sgemm_nn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemm_nn_benchmark.o -DSINGLE_PRECISION -DGEMM_NN
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemm_nn_benchmark.o -o $(BINARY_DIR)/sgemm_nn_benchmark.out $(LIBS)

run_sgemm_nn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemm_nn
	./$(BINARY_DIR)/sgemm_nn_benchmark.out > ./$(BINARY_DIR)/data/sgemm_nn.m

all: $(BLAS_OBJS)

run_all: $(RUN_BLAS_OBJS)
