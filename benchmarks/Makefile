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
	rm -rf ./$(BINARY_DIR)/*.o
	rm -rf ./$(BINARY_DIR)/*.out
	rm -rf ./$(BINARY_DIR)/libblasfeo.a

clean_all: clean
	rm -rf ./build/*


# single benchmarks


DP_BLAS2_OBJS =
DP_BLAS3_OBJS =
DP_LAPACK_OBJS =

SP_BLAS2_OBJS =
SP_BLAS3_OBJS =
SP_LAPACK_OBJS =

RUN_DP_BLAS2_OBJS =
RUN_DP_BLAS3_OBJS =
RUN_DP_LAPACK_OBJS =

RUN_SP_BLAS2_OBJS =
RUN_SP_BLAS3_OBJS =
RUN_SP_LAPACK_OBJS =


# dgemm_nt

DP_BLAS3_OBJS += dgemm_nt
RUN_DP_BLAS3_OBJS += run_dgemm_nt

dgemm_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemm_nt_benchmark.o -DDOUBLE_PRECISION -DGEMM_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemm_nt_benchmark.o -o $(BINARY_DIR)/dgemm_nt_benchmark.out $(LIBS)

run_dgemm_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemm_nt
	./$(BINARY_DIR)/dgemm_nt_benchmark.out > ./$(BINARY_DIR)/data/dgemm_nt.m


# dgemm_nn

DP_BLAS3_OBJS += dgemm_nn
RUN_DP_BLAS3_OBJS += run_dgemm_nn

dgemm_nn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemm_nn_benchmark.o -DDOUBLE_PRECISION -DGEMM_NN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemm_nn_benchmark.o -o $(BINARY_DIR)/dgemm_nn_benchmark.out $(LIBS)

run_dgemm_nn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemm_nn
	./$(BINARY_DIR)/dgemm_nn_benchmark.out > ./$(BINARY_DIR)/data/dgemm_nn.m


# dsyrk_ln

DP_BLAS3_OBJS += dsyrk_ln
RUN_DP_BLAS3_OBJS += run_dsyrk_ln

dsyrk_ln: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dsyrk_ln_benchmark.o -DDOUBLE_PRECISION -DSYRK_LN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dsyrk_ln_benchmark.o -o $(BINARY_DIR)/dsyrk_ln_benchmark.out $(LIBS)

run_dsyrk_ln:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dsyrk_ln
	./$(BINARY_DIR)/dsyrk_ln_benchmark.out > ./$(BINARY_DIR)/data/dsyrk_ln.m


# dtrmm_rlnn

DP_BLAS3_OBJS += dtrmm_rlnn
RUN_DP_BLAS3_OBJS += run_dtrmm_rlnn

dtrmm_rlnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrmm_rlnn_benchmark.o -DDOUBLE_PRECISION -DTRMM_RLNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrmm_rlnn_benchmark.o -o $(BINARY_DIR)/dtrmm_rlnn_benchmark.out $(LIBS)

run_dtrmm_rlnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrmm_rlnn
	./$(BINARY_DIR)/dtrmm_rlnn_benchmark.out > ./$(BINARY_DIR)/data/dtrmm_rlnn.m


# dtrsm_rltn

DP_BLAS3_OBJS += dtrsm_rltn
RUN_DP_BLAS3_OBJS += run_dtrsm_rltn

dtrsm_rltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrsm_rltn_benchmark.o -DDOUBLE_PRECISION -DTRSM_RLTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrsm_rltn_benchmark.o -o $(BINARY_DIR)/dtrsm_rltn_benchmark.out $(LIBS)

run_dtrsm_rltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrsm_rltn
	./$(BINARY_DIR)/dtrsm_rltn_benchmark.out > ./$(BINARY_DIR)/data/dtrsm_rltn.m


# dgetrf_rowpivot

DP_LAPACK_OBJS += dgetrf_rowpivot
RUN_DP_LAPACK_OBJS += run_dgetrf_rowpivot

dgetrf_rowpivot: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgetrf_rowpivot_benchmark.o -DDOUBLE_PRECISION -DGETRF_ROWPIVOT
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgetrf_rowpivot_benchmark.o -o $(BINARY_DIR)/dgetrf_rowpivot_benchmark.out $(LIBS)

run_dgetrf_rowpivot:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgetrf_rowpivot
	./$(BINARY_DIR)/dgetrf_rowpivot_benchmark.out > ./$(BINARY_DIR)/data/dgetrf_rowpivot.m


# dpotrf_l

DP_LAPACK_OBJS += dpotrf_l
RUN_DP_LAPACK_OBJS += run_dpotrf_l

dpotrf_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dpotrf_l_benchmark.o -DDOUBLE_PRECISION -DPOTRF_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/dpotrf_l_benchmark.o -o $(BINARY_DIR)/dpotrf_l_benchmark.out $(LIBS)

run_dpotrf_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dpotrf_l
	./$(BINARY_DIR)/dpotrf_l_benchmark.out > ./$(BINARY_DIR)/data/dpotrf_l.m


# dgemv_n

DP_BLAS2_OBJS += dgemv_n
RUN_DP_BLAS2_OBJS += run_dgemv_n

dgemv_n: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemv_n_benchmark.o -DDOUBLE_PRECISION -DGEMV_N
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemv_n_benchmark.o -o $(BINARY_DIR)/dgemv_n_benchmark.out $(LIBS)

run_dgemv_n:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemv_n
	./$(BINARY_DIR)/dgemv_n_benchmark.out > ./$(BINARY_DIR)/data/dgemv_n.m


# dgemv_t

DP_BLAS2_OBJS += dgemv_t
RUN_DP_BLAS2_OBJS += run_dgemv_t

dgemv_t: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemv_t_benchmark.o -DDOUBLE_PRECISION -DGEMV_T
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemv_t_benchmark.o -o $(BINARY_DIR)/dgemv_t_benchmark.out $(LIBS)

run_dgemv_t:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemv_t
	./$(BINARY_DIR)/dgemv_t_benchmark.out > ./$(BINARY_DIR)/data/dgemv_t.m


# dtrmv_lnn

DP_BLAS2_OBJS += dtrmv_lnn
RUN_DP_BLAS2_OBJS += run_dtrmv_lnn

dtrmv_lnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrmv_lnn_benchmark.o -DDOUBLE_PRECISION -DTRMV_LNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrmv_lnn_benchmark.o -o $(BINARY_DIR)/dtrmv_lnn_benchmark.out $(LIBS)

run_dtrmv_lnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrmv_lnn
	./$(BINARY_DIR)/dtrmv_lnn_benchmark.out > ./$(BINARY_DIR)/data/dtrmv_lnn.m


# dtrmv_ltn

DP_BLAS2_OBJS += dtrmv_ltn
RUN_DP_BLAS2_OBJS += run_dtrmv_ltn

dtrmv_ltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrmv_ltn_benchmark.o -DDOUBLE_PRECISION -DTRMV_LTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrmv_ltn_benchmark.o -o $(BINARY_DIR)/dtrmv_ltn_benchmark.out $(LIBS)

run_dtrmv_ltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrmv_ltn
	./$(BINARY_DIR)/dtrmv_ltn_benchmark.out > ./$(BINARY_DIR)/data/dtrmv_ltn.m


# dtrsv_lnn

DP_BLAS2_OBJS += dtrsv_lnn
RUN_DP_BLAS2_OBJS += run_dtrsv_lnn

dtrsv_lnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrsv_lnn_benchmark.o -DDOUBLE_PRECISION -DTRSV_LNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrsv_lnn_benchmark.o -o $(BINARY_DIR)/dtrsv_lnn_benchmark.out $(LIBS)

run_dtrsv_lnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrsv_lnn
	./$(BINARY_DIR)/dtrsv_lnn_benchmark.out > ./$(BINARY_DIR)/data/dtrsv_lnn.m


# dtrsv_ltn

DP_BLAS2_OBJS += dtrsv_ltn
RUN_DP_BLAS2_OBJS += run_dtrsv_ltn

dtrsv_ltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dtrsv_ltn_benchmark.o -DDOUBLE_PRECISION -DTRSV_LTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/dtrsv_ltn_benchmark.o -o $(BINARY_DIR)/dtrsv_ltn_benchmark.out $(LIBS)

run_dtrsv_ltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dtrsv_ltn
	./$(BINARY_DIR)/dtrsv_ltn_benchmark.out > ./$(BINARY_DIR)/data/dtrsv_ltn.m


# dgemv_nt

DP_BLAS2_OBJS += dgemv_nt
RUN_DP_BLAS2_OBJS += run_dgemv_nt

dgemv_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dgemv_nt_benchmark.o -DDOUBLE_PRECISION -DGEMV_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/dgemv_nt_benchmark.o -o $(BINARY_DIR)/dgemv_nt_benchmark.out $(LIBS)

run_dgemv_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dgemv_nt
	./$(BINARY_DIR)/dgemv_nt_benchmark.out > ./$(BINARY_DIR)/data/dgemv_nt.m


# dsymv_l

DP_BLAS2_OBJS += dsymv_l
RUN_DP_BLAS2_OBJS += run_dsymv_l

dsymv_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/dsymv_l_benchmark.o -DDOUBLE_PRECISION -DSYMV_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/dsymv_l_benchmark.o -o $(BINARY_DIR)/dsymv_l_benchmark.out $(LIBS)

run_dsymv_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking dsymv_l
	./$(BINARY_DIR)/dsymv_l_benchmark.out > ./$(BINARY_DIR)/data/dsymv_l.m


# sgemm_nt

SP_BLAS3_OBJS += sgemm_nt
RUN_SP_BLAS3_OBJS += run_sgemm_nt

sgemm_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemm_nt_benchmark.o -DSINGLE_PRECISION -DGEMM_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemm_nt_benchmark.o -o $(BINARY_DIR)/sgemm_nt_benchmark.out $(LIBS)

run_sgemm_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemm_nt
	./$(BINARY_DIR)/sgemm_nt_benchmark.out > ./$(BINARY_DIR)/data/sgemm_nt.m


# sgemm_nn

SP_BLAS3_OBJS += sgemm_nn
RUN_SP_BLAS3_OBJS += run_sgemm_nn

sgemm_nn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemm_nn_benchmark.o -DSINGLE_PRECISION -DGEMM_NN
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemm_nn_benchmark.o -o $(BINARY_DIR)/sgemm_nn_benchmark.out $(LIBS)

run_sgemm_nn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemm_nn
	./$(BINARY_DIR)/sgemm_nn_benchmark.out > ./$(BINARY_DIR)/data/sgemm_nn.m


# ssyrk_ln

SP_BLAS3_OBJS += ssyrk_ln
RUN_SP_BLAS3_OBJS += run_ssyrk_ln

ssyrk_ln: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/ssyrk_ln_benchmark.o -DSINGLE_PRECISION -DSYRK_LN
	$(CC) $(CFLAGS) $(BINARY_DIR)/ssyrk_ln_benchmark.o -o $(BINARY_DIR)/ssyrk_ln_benchmark.out $(LIBS)

run_ssyrk_ln:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking ssyrk_ln
	./$(BINARY_DIR)/ssyrk_ln_benchmark.out > ./$(BINARY_DIR)/data/ssyrk_ln.m


# strmm_rlnn

SP_BLAS3_OBJS += strmm_rlnn
RUN_SP_BLAS3_OBJS += run_strmm_rlnn

strmm_rlnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strmm_rlnn_benchmark.o -DSINGLE_PRECISION -DTRMM_RLNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strmm_rlnn_benchmark.o -o $(BINARY_DIR)/strmm_rlnn_benchmark.out $(LIBS)

run_strmm_rlnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strmm_rlnn
	./$(BINARY_DIR)/strmm_rlnn_benchmark.out > ./$(BINARY_DIR)/data/strmm_rlnn.m


# strsm_rltn

SP_BLAS3_OBJS += strsm_rltn
RUN_SP_BLAS3_OBJS += run_strsm_rltn

strsm_rltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strsm_rltn_benchmark.o -DSINGLE_PRECISION -DTRSM_RLTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strsm_rltn_benchmark.o -o $(BINARY_DIR)/strsm_rltn_benchmark.out $(LIBS)

run_strsm_rltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strsm_rltn
	./$(BINARY_DIR)/strsm_rltn_benchmark.out > ./$(BINARY_DIR)/data/strsm_rltn.m


# sgetrf_rowpivot

#SP_LAPACK_OBJS += sgetrf_rowpivot
#RUN_SP_LAPACK_OBJS += run_sgetrf_rowpivot

#sgetrf_rowpivot: common
#	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgetrf_rowpivot_benchmark.o -DSINGLE_PRECISION -DGETRF_ROWPIVOT
#	$(CC) $(CFLAGS) $(BINARY_DIR)/sgetrf_rowpivot_benchmark.o -o $(BINARY_DIR)/sgetrf_rowpivot_benchmark.out $(LIBS)

#run_sgetrf_rowpivot:
#	mkdir -p ./$(BINARY_DIR)/data
#	@echo
#	@echo benchmarking sgetrf_rowpivot
#	./$(BINARY_DIR)/sgetrf_rowpivot_benchmark.out > ./$(BINARY_DIR)/data/sgetrf_rowpivot.m


# spotrf_l

SP_LAPACK_OBJS += spotrf_l
RUN_SP_LAPACK_OBJS += run_spotrf_l

spotrf_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/spotrf_l_benchmark.o -DSINGLE_PRECISION -DPOTRF_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/spotrf_l_benchmark.o -o $(BINARY_DIR)/spotrf_l_benchmark.out $(LIBS)

run_spotrf_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking spotrf_l
	./$(BINARY_DIR)/spotrf_l_benchmark.out > ./$(BINARY_DIR)/data/spotrf_l.m


# sgemv_n

SP_BLAS2_OBJS += sgemv_n
RUN_SP_BLAS2_OBJS += run_sgemv_n

sgemv_n: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemv_n_benchmark.o -DSINGLE_PRECISION -DGEMV_N
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemv_n_benchmark.o -o $(BINARY_DIR)/sgemv_n_benchmark.out $(LIBS)

run_sgemv_n:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemv_n
	./$(BINARY_DIR)/sgemv_n_benchmark.out > ./$(BINARY_DIR)/data/sgemv_n.m


# sgemv_t

SP_BLAS2_OBJS += sgemv_t
RUN_SP_BLAS2_OBJS += run_sgemv_t

sgemv_t: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemv_t_benchmark.o -DSINGLE_PRECISION -DGEMV_T
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemv_t_benchmark.o -o $(BINARY_DIR)/sgemv_t_benchmark.out $(LIBS)

run_sgemv_t:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemv_t
	./$(BINARY_DIR)/sgemv_t_benchmark.out > ./$(BINARY_DIR)/data/sgemv_t.m


# strmv_lnn

SP_BLAS2_OBJS += strmv_lnn
RUN_SP_BLAS2_OBJS += run_strmv_lnn

strmv_lnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strmv_lnn_benchmark.o -DSINGLE_PRECISION -DTRMV_LNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strmv_lnn_benchmark.o -o $(BINARY_DIR)/strmv_lnn_benchmark.out $(LIBS)

run_strmv_lnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strmv_lnn
	./$(BINARY_DIR)/strmv_lnn_benchmark.out > ./$(BINARY_DIR)/data/strmv_lnn.m


# strmv_ltn

SP_BLAS2_OBJS += strmv_ltn
RUN_SP_BLAS2_OBJS += run_strmv_ltn

strmv_ltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strmv_ltn_benchmark.o -DSINGLE_PRECISION -DTRMV_LTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strmv_ltn_benchmark.o -o $(BINARY_DIR)/strmv_ltn_benchmark.out $(LIBS)

run_strmv_ltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strmv_ltn
	./$(BINARY_DIR)/strmv_ltn_benchmark.out > ./$(BINARY_DIR)/data/strmv_ltn.m


# strsv_lnn

SP_BLAS2_OBJS += strsv_lnn
RUN_SP_BLAS2_OBJS += run_strsv_lnn

strsv_lnn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strsv_lnn_benchmark.o -DSINGLE_PRECISION -DTRSV_LNN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strsv_lnn_benchmark.o -o $(BINARY_DIR)/strsv_lnn_benchmark.out $(LIBS)

run_strsv_lnn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strsv_lnn
	./$(BINARY_DIR)/strsv_lnn_benchmark.out > ./$(BINARY_DIR)/data/strsv_lnn.m


# strsv_ltn

SP_BLAS2_OBJS += strsv_ltn
RUN_SP_BLAS2_OBJS += run_strsv_ltn

strsv_ltn: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/strsv_ltn_benchmark.o -DSINGLE_PRECISION -DTRSV_LTN
	$(CC) $(CFLAGS) $(BINARY_DIR)/strsv_ltn_benchmark.o -o $(BINARY_DIR)/strsv_ltn_benchmark.out $(LIBS)

run_strsv_ltn:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking strsv_ltn
	./$(BINARY_DIR)/strsv_ltn_benchmark.out > ./$(BINARY_DIR)/data/strsv_ltn.m


# sgemv_nt

SP_BLAS2_OBJS += sgemv_nt
RUN_SP_BLAS2_OBJS += run_sgemv_nt

sgemv_nt: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/sgemv_nt_benchmark.o -DSINGLE_PRECISION -DGEMV_NT
	$(CC) $(CFLAGS) $(BINARY_DIR)/sgemv_nt_benchmark.o -o $(BINARY_DIR)/sgemv_nt_benchmark.out $(LIBS)

run_sgemv_nt:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking sgemv_nt
	./$(BINARY_DIR)/sgemv_nt_benchmark.out > ./$(BINARY_DIR)/data/sgemv_nt.m


# ssymv_l

SP_BLAS2_OBJS += ssymv_l
RUN_SP_BLAS2_OBJS += run_ssymv_l

ssymv_l: common
	$(CC) $(CFLAGS) -c x_blas_benchmark.c -o $(BINARY_DIR)/ssymv_l_benchmark.o -DSINGLE_PRECISION -DSYMV_L
	$(CC) $(CFLAGS) $(BINARY_DIR)/ssymv_l_benchmark.o -o $(BINARY_DIR)/ssymv_l_benchmark.out $(LIBS)

run_ssymv_l:
	mkdir -p ./$(BINARY_DIR)/data
	@echo
	@echo benchmarking ssymv_l
	./$(BINARY_DIR)/ssymv_l_benchmark.out > ./$(BINARY_DIR)/data/ssymv_l.m


# print

print_figures:
	cp print_blas_benchmark.m ./$(BINARY_DIR)/data/
	cd ./$(BINARY_DIR)/data/ ; octave-cli print_blas_benchmark.m
	cd ./$(BINARY_DIR)/data/ ; epstopdf dgemm.eps -out dgemm.pdf ; rm dgemm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dsyrk.eps -out dsyrk.pdf ; rm dsyrk.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dtrmm.eps -out dtrmm.pdf ; rm dtrmm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dtrsm.eps -out dtrsm.pdf ; rm dtrsm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dpotrf.eps -out dpotrf.pdf ; rm dpotrf.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dgemv.eps -out dgemv.pdf ; rm dgemv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dtrmv.eps -out dtrmv.pdf ; rm dtrmv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dtrsv.eps -out dtrsv.pdf ; rm dtrsv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf dsymv.eps -out dsymv.pdf ; rm dsymv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf sgemm.eps -out sgemm.pdf ; rm sgemm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf ssyrk.eps -out sgemm.pdf ; rm ssyrk.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf strmm.eps -out sgemm.pdf ; rm strmm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf strsm.eps -out sgemm.pdf ; rm strsm.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf spotrf.eps -out spotrf.pdf ; rm spotrf.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf sgemv.eps -out sgemv.pdf ; rm sgemv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf strmv.eps -out strmv.pdf ; rm strmv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf strsv.eps -out strsv.pdf ; rm strsv.eps
	cd ./$(BINARY_DIR)/data/ ; epstopdf ssymv.eps -out ssymv.pdf ; rm ssymv.eps

# build

dp_blas2: $(DP_BLAS2_OBJS)

dp_blas3: $(DP_BLAS3_OBJS)

dp_lapack: $(DP_LAPACK_OBJS)

dp_all: dp_blas3 dp_lapack dp_blas2

sp_blas2: $(SP_BLAS2_OBJS)

sp_blas3: $(SP_BLAS3_OBJS)

sp_lapack: $(SP_LAPACK_OBJS)

sp_all: sp_blas3 sp_lapack sp_blas2

all: dp_all sp_all

# run

run_dp_blas2: $(RUN_DP_BLAS2_OBJS)

run_dp_blas3: $(RUN_DP_BLAS3_OBJS)

run_dp_lapack: $(RUN_DP_LAPACK_OBJS)

run_dp_all: run_dp_blas3 run_dp_lapack run_dp_blas2

run_sp_blas2: $(RUN_SP_BLAS2_OBJS)

run_sp_blas3: $(RUN_SP_BLAS3_OBJS)

run_sp_lapack: $(RUN_SP_LAPACK_OBJS)

run_sp_all: run_sp_blas3 run_sp_lapack run_sp_blas2

run_all: run_dp_all run_sp_all


