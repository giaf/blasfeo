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
#	./$(BINARY_DIR)/benchmark_s_blas.o.out


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


# dpotrf_l

BLAS_OBJS += dpotrf_l
RUN_BLAS_OBJS += run_dpotrf_l

dpotrf_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dpotrf_l_benchmark.o -DDOUBLE_PRECISION -DPOTRF_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/dpotrf_l_benchmark.o -o $(BINARY_DIR)/dpotrf_l_benchmark.out $(LIBS)

run_dpotrf_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dpotrf_l
	./$(BINARY_DIR)/dpotrf_l_benchmark.out > ./$(BINARY_DIR)/data/dpotrf_l.m


# dgemv_n

BLAS_OBJS += dgemv_n
RUN_BLAS_OBJS += run_dgemv_n

dgemv_n: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemv_n_benchmark.o -DDOUBLE_PRECISION -DGEMV_N
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemv_n_benchmark.o -o $(BINARY_DIR)/dgemv_n_benchmark.out $(LIBS)

run_dgemv_n:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemv_n
	./$(BINARY_DIR)/dgemv_n_benchmark.out > ./$(BINARY_DIR)/data/dgemv_n.m


# dgemv_t

BLAS_OBJS += dgemv_t
RUN_BLAS_OBJS += run_dgemv_t

dgemv_t: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemv_t_benchmark.o -DDOUBLE_PRECISION -DGEMV_T
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemv_t_benchmark.o -o $(BINARY_DIR)/dgemv_t_benchmark.out $(LIBS)

run_dgemv_t:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemv_t
	./$(BINARY_DIR)/dgemv_t_benchmark.out > ./$(BINARY_DIR)/data/dgemv_t.m


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


# sgemm_nn

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


# spotrf_l

BLAS_OBJS += spotrf_l
RUN_BLAS_OBJS += run_spotrf_l

spotrf_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/spotrf_l_benchmark.o -DSINGLE_PRECISION -DPOTRF_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/spotrf_l_benchmark.o -o $(BINARY_DIR)/spotrf_l_benchmark.out $(LIBS)

run_spotrf_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking spotrf_l
	./$(BINARY_DIR)/spotrf_l_benchmark.out > ./$(BINARY_DIR)/data/spotrf_l.m


# sgemv_n

BLAS_OBJS += sgemv_n
RUN_BLAS_OBJS += run_sgemv_n

sgemv_n: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemv_n_benchmark.o -DSINGLE_PRECISION -DGEMV_N
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemv_n_benchmark.o -o $(BINARY_DIR)/sgemv_n_benchmark.out $(LIBS)

run_sgemv_n:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemv_n
	./$(BINARY_DIR)/sgemv_n_benchmark.out > ./$(BINARY_DIR)/data/sgemv_n.m


# sgemv_t

BLAS_OBJS += sgemv_t
RUN_BLAS_OBJS += run_sgemv_t

sgemv_t: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemv_t_benchmark.o -DSINGLE_PRECISION -DGEMV_T
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemv_t_benchmark.o -o $(BINARY_DIR)/sgemv_t_benchmark.out $(LIBS)

run_sgemv_t:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemv_t
	./$(BINARY_DIR)/sgemv_t_benchmark.out > ./$(BINARY_DIR)/data/sgemv_t.m


# print

print_figures:
	cp print_blas_benchmark.m ./$(BINARY_DIR)/data/
	cd ./$(BINARY_DIR)/data/ ; octave-cli print_blas_benchmark.m
	cd ./$(BINARY_DIR)/data/ ; epstopdf dgemm.eps -out dgemm.pdf ; rm dgemm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dpotrf.eps -out dpotrf.pdf ; rm dpotrf.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dgemv.eps -out dgemv.pdf ; rm dgemv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf sgemm.eps -out sgemm.pdf ; rm sgemm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf spotrf.eps -out spotrf.pdf ; rm spotrf.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf sgemv.eps -out sgemv.pdf ; rm sgemv.eps

# all

all: $(BLAS_OBJS)

run_all: $(RUN_BLAS_OBJS)
