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

LIBS =

LIBS += libblasfeo.a

# add different link library for different EXTERNAL_BLAS implementation
#include ../Makefile.external_blas
LIBS += $(LIBS_EXTERNAL_BLAS)

ifeq ($(COMPLEMENT_WITH_NETLIB_BLAS), 1)
LIBS += -lgfortran
endif

LIBS += -lm


## dgemm kernel
# OBJS = example_kernel_dgemm.o
# OBJS = kernel_dgemm.o

## factorization
# OBJS = example_d_lu_factorization.o
# OBJS = example_s_lu_factorization.o

## riccati
OBJS = example_d_riccati_recursion.o tools.o
# OBJS = example_s_riccati_recursion.o tools.o

## getting started
# OBJS = getting_started.o


CFLAGS += -I../include/


build: common $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o example.out

common:
	cp ../lib/libblasfeo.a .

run:
	./example.out

clean:
	rm -rf ./*.o
	rm -rf ./*.out

deep_clean: clean
