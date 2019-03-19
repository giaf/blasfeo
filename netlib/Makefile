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

BLAS_NETLIB = blas-3.8.0
BLAS_DIR = blas

obj: | $(BLAS_DIR)
	@echo "compiling blas ..."
	(cd $(BLAS_DIR); $(MAKE) all)

$(BLAS_DIR):
	@echo "downloading blas ..."
	wget www.netlib.org/blas/$(BLAS_NETLIB).tgz -O $(BLAS_NETLIB).tgz
	mkdir -p $(BLAS_DIR)
	tar -xzf $(BLAS_NETLIB).tgz -C $(BLAS_DIR) --strip-components 1

clean:
ifneq ($(wildcard $(BLAS_DIR)), )
	make -C $(BLAS_DIR) clean;
endif
#	if [ -d "./$(BLAS_DIR)" ]; then \
#		make -C $(BLAS_DIR) clean; \
#	fi

deep_clean:
	rm -rf $(BLAS_DIR)
	rm -rf $(BLAS_NETLIB).tgz

.PHONY: obj clean deep_clean
