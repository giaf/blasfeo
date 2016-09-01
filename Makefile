###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS For Embedded Optimization.                                                      #
# Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     #
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

include ./Makefile.rule

OBJS = 

ifeq ($(TARGET), X64_INTEL_HASWELL)
# kernel
OBJS += ./kernel/avx2/kernel_dgemm_4x4_lib4.o ./kernel/avx2/kernel_dgemm_8x4_lib4.o ./kernel/avx2/kernel_dgemm_12x4_lib4.o ./kernel/avx2/kernel_dgemv_8_lib4.o ./kernel/avx/kernel_dgemv_4_lib4.o
OBJS += ./kernel/c99/kernel_sgemm_4x4_lib4.o ./kernel/c99/kernel_sgemv_4_lib4.o
# blas
OBJS += ./blas/d_blas3_lib4.o ./blas/d_lapack_lib4.o ./blas/d_blas2_lib4.o
OBJS += ./blas/s_blas3_lib4.o ./blas/s_lapack_lib4.o ./blas/s_blas2_lib4.o
#aux
OBJS += ./aux/d_aux_lib4.o ./aux/d_aux_extern_depend_lib4.o 
OBJS += ./aux/s_aux_lib4.o ./aux/s_aux_extern_depend_lib4.o 
endif

# kernel
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
OBJS += ./kernel/avx/kernel_dgemm_4x4_lib4.o ./kernel/avx/kernel_dgemm_8x4_lib4.o ./kernel/avx/kernel_dgemv_12_lib4.o ./kernel/avx/kernel_dgemv_8_lib4.o  ./kernel/avx/kernel_dgemv_4_lib4.o 
OBJS += ./kernel/c99/kernel_sgemm_4x4_lib4.o ./kernel/c99/kernel_sgemv_4_lib4.o
# blas
OBJS += ./blas/d_blas3_lib4.o ./blas/d_lapack_lib4.o ./blas/d_blas2_lib4.o
OBJS += ./blas/s_blas3_lib4.o ./blas/s_lapack_lib4.o ./blas/s_blas2_lib4.o
#aux
OBJS += ./aux/d_aux_lib4.o ./aux/d_aux_extern_depend_lib4.o 
OBJS += ./aux/s_aux_lib4.o ./aux/s_aux_extern_depend_lib4.o 
endif

ifeq ($(TARGET), X64_INTEL_CORE)
# kernel
OBJS += ./kernel/sse3/kernel_dgemm_4x4_lib4.o ./kernel/c99/kernel_dgemv_4_lib4.o
OBJS += ./kernel/c99/kernel_sgemm_4x4_lib4.o ./kernel/c99/kernel_sgemv_4_lib4.o
# blas
OBJS += ./blas/d_blas3_lib4.o ./blas/d_lapack_lib4.o ./blas/d_blas2_lib4.o
OBJS += ./blas/s_blas3_lib4.o ./blas/s_lapack_lib4.o ./blas/s_blas2_lib4.o
#aux
OBJS += ./aux/d_aux_lib4.o ./aux/d_aux_extern_depend_lib4.o 
OBJS += ./aux/s_aux_lib4.o ./aux/s_aux_extern_depend_lib4.o 
endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)
# kernel
OBJS += ./kernel/fma/kernel_dgemm_4x4_lib4.o ./kernel/c99/kernel_dgemv_4_lib4.o
OBJS += ./kernel/c99/kernel_sgemm_4x4_lib4.o ./kernel/c99/kernel_sgemv_4_lib4.o
# blas
OBJS += ./blas/d_blas3_lib4.o ./blas/d_lapack_lib4.o ./blas/d_blas2_lib4.o
OBJS += ./blas/s_blas3_lib4.o ./blas/s_lapack_lib4.o ./blas/s_blas2_lib4.o
#aux
OBJS += ./aux/d_aux_lib4.o ./aux/d_aux_extern_depend_lib4.o 
OBJS += ./aux/s_aux_lib4.o ./aux/s_aux_extern_depend_lib4.o 
endif

ifeq ($(TARGET), GENERIC)
# kernel
OBJS += ./kernel/c99/kernel_dgemm_4x4_lib4.o ./kernel/c99/kernel_dgemv_4_lib4.o
OBJS += ./kernel/c99/kernel_sgemm_4x4_lib4.o ./kernel/c99/kernel_sgemv_4_lib4.o
# blas
OBJS += ./blas/d_blas3_lib4.o ./blas/d_lapack_lib4.o ./blas/d_blas2_lib4.o
OBJS += ./blas/s_blas3_lib4.o ./blas/s_lapack_lib4.o ./blas/s_blas2_lib4.o
#aux
OBJS += ./aux/d_aux_lib4.o ./aux/d_aux_extern_depend_lib4.o 
OBJS += ./aux/s_aux_lib4.o ./aux/s_aux_extern_depend_lib4.o 
endif

OBJS += ./aux/i_aux_extern_depend_lib4.o

all: clean static_library

static_library: target
	( cd aux; $(MAKE) obj)
	( cd kernel; $(MAKE) obj)
	( cd blas; $(MAKE) obj)
	ar rcs libblasfeo.a $(OBJS) 
	@echo
	@echo " libblasfeo.a static library build complete."
	@echo

target:
	touch ./include/blasfeo_target.h
ifeq ($(TARGET), X64_INTEL_HASWELL)
	echo "#ifndef TARGET_X64_INTEL_HASWELL" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	echo "#ifndef TARGET_X64_INTEL_SANDY_BRIDGE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_CORE)
	echo "#ifndef TARGET_X64_INTEL_CORE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_CORE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
	echo "#ifndef TARGET_X64_AMD_BULLDOZER" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), GENERIC)
	echo "#ifndef TARGET_GENERIC" > ./include/blasfeo_target.h
	echo "#define TARGET_GENERIC" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif

install_static:
	mkdir -p $(PREFIX)/blasfeo
	mkdir -p $(PREFIX)/blasfeo/lib
	cp -f libblasfeo.a $(PREFIX)/blasfeo/lib/
	mkdir -p $(PREFIX)/blasfeo/include
	cp -f ./include/*.h $(PREFIX)/blasfeo/include/

test_problem:
	cp libblasfeo.a ./test_problems/libblasfeo.a
	make -C test_problems obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test_problems/test.out

clean:
	rm -f libblasfeo.a
	make -C aux clean
	make -C kernel clean
	make -C blas clean
	make -C test_problems clean

