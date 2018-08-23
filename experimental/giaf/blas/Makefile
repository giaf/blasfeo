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

CC = gcc

CFLAGS = -O2 -fPIC -mavx2 -DOS_LINUX
ASFLAGS = -mavx2 -DOS_LINUX

#OBJS = kernel/avx/kernel_dgemm_4x4_lib.o tests/test.o
OBJS = kernel/avx2/kernel_dgemm_12x4_lib.o kernel/avx2/kernel_dgemm_8x4_lib.o kernel/avx2/kernel_dgemm_4x4_lib.o kernel/avx2/kernel_dpatr_lib4.o kernel/avx2/kernel_dpack_lib4.o blas/dgemm.o blas/dpotrf.o

#OBJS += tests/test.o
OBJS += tests/benchmark.o

build: $(OBJS)
	$(CC) $(OBJS) /home/gianluca/blasfeo/lib/libblasfeo.a /opt/openblas/lib/libopenblas.a -lm -o tests/test.out
	@echo ""
	@echo "Done!"
	@echo ""

run:
	mkdir -p tests/data
#	./tests/test.out
	./tests/test.out > tests/data/blas.m

print:
	cp tests/print_blas.m tests/data
	cd tests/data; octave-cli print_blas.m
	cd tests/data/ ; epstopdf blas.eps -out blas.pdf ; rm blas.eps

clean:
	rm -f *.o
