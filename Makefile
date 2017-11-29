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

include ./Makefile.rule

OBJS =

ifeq ($(LA), HIGH_PERFORMANCE)

ifeq ($(TARGET), X64_INTEL_HASWELL)

# aux
OBJS += auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib8.o \
		auxiliary/m_aux_lib48.o \

# kernels
OBJS += \
		kernel/avx2/kernel_dgemm_12x4_lib4.o \
		kernel/avx2/kernel_dgemm_8x8_lib4.o \
		kernel/avx2/kernel_dgemm_8x4_lib4.o \
		kernel/avx2/kernel_dgemm_4x4_lib4.o \
		kernel/avx2/kernel_dgemm_10xX_lib4.o \
		kernel/avx2/kernel_dgemm_8x2_lib4.o \
		kernel/avx2/kernel_dgemm_6xX_lib4.o \
		kernel/avx2/kernel_dgemm_4x2_lib4.o \
		kernel/avx/kernel_dgemm_diag_lib4.o \
		kernel/avx2/kernel_dgemv_8_lib4.o \
		kernel/avx/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/avx2/kernel_dsymv_6_lib4.o \
		kernel/avx2/kernel_dgetrf_pivot_4_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx2/kernel_dgebp_lib4.o \
		kernel/avx2/kernel_dgelqf_4_lib4.o \
		kernel/avx2/kernel_dgetr_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
		\
		kernel/avx2/kernel_sgemm_24x4_lib8.o \
		kernel/avx2/kernel_sgemm_16x4_lib8.o \
		kernel/avx2/kernel_sgemm_8x8_lib8.o \
		kernel/avx2/kernel_sgemm_8x4_lib8.o \
		kernel/avx/kernel_sgemm_diag_lib8.o \
		kernel/avx/kernel_sgemv_8_lib8.o \
		kernel/avx/kernel_sgemv_4_lib8.o \
		kernel/avx/kernel_sgecpsc_lib8.o \
		kernel/avx/kernel_sgetr_lib8.o \
		kernel/avx/kernel_sgead_lib8.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib8.o \
		blas/s_blas2_lib8.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib8.o \
		blas/s_blas3_diag_lib8.o \
		blas/s_lapack_lib8.o \

endif

ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib8.o \
		auxiliary/m_aux_lib48.o \

# kernels
OBJS += \
		kernel/avx/kernel_dgemm_8x4_lib4.o \
		kernel/avx/kernel_dgemm_8x2_lib4.o \
		kernel/avx/kernel_dgemm_4x4_lib4.o \
		kernel/avx/kernel_dgemm_4x2_lib4.o \
		kernel/avx/kernel_dgemm_12x4_lib4.o \
		kernel/avx/kernel_dgemm_10xX_lib4.o \
		kernel/avx/kernel_dgemm_6xX_lib4.o \
		kernel/avx/kernel_dgemm_diag_lib4.o \
		kernel/avx/kernel_dgemv_12_lib4.o \
		kernel/avx/kernel_dgemv_8_lib4.o \
		kernel/avx/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/avx/kernel_dsymv_6_lib4.o \
		kernel/avx/kernel_dgetrf_pivot_4_lib4.o \
		kernel/avx/kernel_dgeqrf_4_lib4.o \
		kernel/avx/kernel_dgebp_lib4.o \
		kernel/avx/kernel_dgecp_lib4.o \
		kernel/avx/kernel_dgetr_lib4.o \
		\
		kernel/avx/kernel_sgemm_16x4_lib8.o \
		kernel/avx/kernel_sgemm_8x8_lib8.o \
		kernel/avx/kernel_sgemm_8x4_lib8.o \
		kernel/avx/kernel_sgemm_diag_lib8.o \
		kernel/avx/kernel_sgemv_8_lib8.o \
		kernel/avx/kernel_sgemv_4_lib8.o \
		kernel/avx/kernel_sgecpsc_lib8.o \
		kernel/avx/kernel_sgetr_lib8.o \
		kernel/avx/kernel_sgead_lib8.o \

# blas
OBJS  += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib8.o \
		blas/s_blas2_lib8.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib8.o \
		blas/s_blas3_diag_lib8.o \
		blas/s_lapack_lib8.o \

endif

ifeq ($(TARGET), X64_INTEL_CORE)
# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/sse3/kernel_dgemm_4x4_lib4.o \
		kernel/sse3/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

ifeq ($(TARGET), X64_AMD_BULLDOZER)

# aux
OBJS += \
		auxiliarn/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/fma/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

ifeq ($(TARGET), ARMV8A_ARM_CORTEX_A57)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv8a/kernel_dgemm_8x4_lib4.o \
		kernel/armv8a/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/armv8a/kernel_sgemm_16x4_lib4.o \
		kernel/armv8a/kernel_sgemm_12x4_lib4.o \
		kernel/armv8a/kernel_sgemm_8x8_lib4.o \
		kernel/armv8a/kernel_sgemm_8x4_lib4.o \
		kernel/armv8a/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		\
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/armv7a/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		\
		kernel/armv7a/kernel_sgemm_12x4_lib4.o \
		kernel/armv7a/kernel_sgemm_8x4_lib4.o \
		kernel/armv7a/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

ifeq ($(TARGET), GENERIC)

# aux
OBJS += \
		auxiliary/d_aux_lib4.o \
		auxiliary/s_aux_lib4.o \
		auxiliary/m_aux_lib44.o \

# kernels
OBJS += \
		kernel/c99/kernel_dgemm_4x4_lib4.o \
		kernel/c99/kernel_dgemm_diag_lib4.o \
		kernel/c99/kernel_dgemv_4_lib4.o \
		kernel/c99/kernel_dsymv_4_lib4.o \
		kernel/c99/kernel_dgecp_lib4.o \
		kernel/c99/kernel_dgetr_lib4.o \
		kernel/c99/kernel_dgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_dgeqrf_4_lib4.o \
		\
		kernel/c99/kernel_sgemm_4x4_lib4.o \
		kernel/c99/kernel_sgemm_diag_lib4.o \
		kernel/c99/kernel_sgemv_4_lib4.o \
		kernel/c99/kernel_ssymv_4_lib4.o \
		kernel/c99/kernel_sgetrf_pivot_4_lib4.o \
		kernel/c99/kernel_sgecp_lib4.o \
		kernel/c99/kernel_sgetr_lib4.o \

# blas
OBJS += \
		blas/d_blas1_lib4.o \
		blas/d_blas2_lib4.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib4.o \
		blas/d_blas3_diag_lib4.o \
		blas/d_lapack_lib4.o \
		blas/s_blas1_lib4.o \
		blas/s_blas2_lib4.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib4.o \
		blas/s_blas3_diag_lib4.o \
		blas/s_lapack_lib4.o \

endif

else # LA_REFERENCE | LA_BLAS

# aux
OBJS += \
		auxiliary/d_aux_lib.o \
		auxiliary/s_aux_lib.o \
		auxiliary/m_aux_lib.o \

# blas
OBJS += \
		blas/d_blas1_lib.o \
		blas/d_blas2_lib.o \
		blas/d_blas2_diag_lib.o \
		blas/d_blas3_lib.o \
		blas/d_blas3_diag_lib.o \
		blas/d_lapack_lib.o \
		\
		blas/s_blas1_lib.o \
		blas/s_blas2_lib.o \
		blas/s_blas2_diag_lib.o \
		blas/s_blas3_lib.o \
		blas/s_blas3_diag_lib.o \
		blas/s_lapack_lib.o \

endif # LA choice

ifeq ($(EXT_DEP), 1)
# ext dep
OBJS += \
		auxiliary/d_aux_ext_dep_lib.o \
		auxiliary/d_aux_ext_dep_lib4.o \
		auxiliary/s_aux_ext_dep_lib.o \
		auxiliary/s_aux_ext_dep_lib4.o \
		auxiliary/v_aux_ext_dep_lib.o \
		auxiliary/i_aux_ext_dep_lib.o \

endif



ifeq ($(TESTING), 1)
# reference routine for testing
OBJS_REF =
# aux
OBJS_REF += \
		auxiliary/d_aux_libref.o \
		auxiliary/s_aux_libref.o \
		auxiliary/d_aux_ext_dep_libref.o \
		auxiliary/s_aux_ext_dep_libref.o \
#
endif


# Define targets


all: clean static_library


# compile static library
static_library: target
	( cd kernel; $(MAKE) obj)
	( cd auxiliary; $(MAKE) obj)
	( cd blas; $(MAKE) obj)
	ar rcs libblasfeo.a $(OBJS)
	mv libblasfeo.a ./lib/
ifeq ($(TESTING), 1)
	ar rcs libblasfeo_ref.a $(OBJS_REF)
	mv libblasfeo_ref.a ./lib/
endif
	@echo
	@echo " libblasfeo.a static library build complete."
	@echo


# compile shared library
shared_library: target
	( cd auxiliary; $(MAKE) obj)
	( cd kernel; $(MAKE) obj)
	( cd blas; $(MAKE) obj)
	gcc -shared -o libblasfeo.so $(OBJS)
	mv libblasfeo.so ./lib/
ifeq ($(TESTING), 1)
	gcc -shared -o libblasfeo_ref.so $(OBJS_REF)
	mv libblasfeo_ref.so ./lib/
endif
	@echo
	@echo " libblasfeo.so shared library build complete."
	@echo


# generate target header
target:
	touch ./include/blasfeo_target.h
ifeq ($(TARGET), X64_INTEL_HASWELL)
	echo "#ifndef TARGET_X64_INTEL_HASWELL" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET X64_INTEL_HASWELL" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_SANDY_BRIDGE)
	echo "#ifndef TARGET_X64_INTEL_SANDY_BRIDGE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET X64_INTEL_SANDY_BRIDGE" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_INTEL_CORE)
	echo "#ifndef TARGET_X64_INTEL_CORE" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_INTEL_CORE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET X64_INTEL_CORE" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), X64_AMD_BULLDOZER)
	echo "#ifndef TARGET_X64_AMD_BULLDOZER" > ./include/blasfeo_target.h
	echo "#define TARGET_X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET X64_AMD_BULLDOZER" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), GENERIC)
	echo "#ifndef TARGET_GENERIC" > ./include/blasfeo_target.h
	echo "#define TARGET_GENERIC" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET GENERIC" >> ./include/blasfeo_target.h
endif
ifeq ($(TARGET), ARMV7A_ARM_CORTEX_A15)
	echo "#ifndef TARGET_ARMV7A_ARM_CORTEX_A15" > ./include/blasfeo_target.h
	echo "#define TARGET_ARMV7A_ARM_CORTEX_A15" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define TARGET ARMV7A_ARM_CORTEX_A15" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), HIGH_PERFORMANCE)
	echo "#ifndef LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#define LA_HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define LA HIGH_PERFORMANCE" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), BLAS)
	echo "#ifndef LA_BLAS" >> ./include/blasfeo_target.h
	echo "#define LA_BLAS" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define LA BLAS" >> ./include/blasfeo_target.h
endif
ifeq ($(LA), REFERENCE)
	echo "#ifndef LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#define LA_REFERENCE" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
	echo "#define LA REFERENCE" >> ./include/blasfeo_target.h
endif
ifeq ($(EXT_DEP), 1)
	echo "#ifndef EXT_DEP" >> ./include/blasfeo_target.h
	echo "#define EXT_DEP" >> ./include/blasfeo_target.h
	echo "#endif" >> ./include/blasfeo_target.h
endif


# install static library & headers
install_static:
	mkdir -p $(PREFIX)/blasfeo
	mkdir -p $(PREFIX)/blasfeo/lib
	cp -f ./lib/libblasfeo.a $(PREFIX)/blasfeo/lib/
	mkdir -p $(PREFIX)/blasfeo/include
	cp -f ./include/*.h $(PREFIX)/blasfeo/include/


# install share library & headers
install_shared:
	mkdir -p $(PREFIX)/blasfeo
	mkdir -p $(PREFIX)/blasfeo/lib
	cp -f ./lib/libblasfeo.so $(PREFIX)/blasfeo/lib/
	mkdir -p $(PREFIX)/blasfeo/include
	cp -f ./include/*.h $(PREFIX)/blasfeo/include/


# clean .o files
clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C examples clean


# clean test problems
clean_test_problems:
	make -C test_problems clean


# deep clean
deep_clean: clean clean_test_problems
	rm -f ./include/blasfeo_target.h
	rm -f ./lib/libblasfeo.a
	rm -f ./lib/libblasfeo.so


# test problems

# directory for test problems binaries
BINARY_DIR = build/$(LA)/$(TARGET)


# copy static library into test path
deploy_to_test:
	mkdir -p ./test_problems/$(BINARY_DIR)
	cp ./lib/libblasfeo.a ./test_problems/$(BINARY_DIR)/
ifeq ($(TESTING), 1)
	cp ./lib/libblasfeo_ref.a ./test_problems/$(BINARY_DIR)/
endif


# one single test
build_test:
	make -C test_problems one_test
	@echo
	@echo " Test problem build complete."
	@echo

run_test:
	make -C test_problems run

test: deploy_to_test build_test


# aux test
build_test_aux:
	make -C test_problems aux
	@echo
	@echo " Test problem build complete."
	@echo

run_test_aux:
	make -C test_problems run_aux

test_aux: deploy_to_test build_test_aux


# blas test
build_test_blas:
	make -C test_problems blas
	@echo
	@echo " Test problem build complete."
	@echo

run_test_blas:
	make -C test_problems run_blas

test_blas: deploy_to_test build_test_blas


# deep build library (take into account flags changes)
# copy library
# build tests
test_aux_clean: clean static_library test_aux

# build tests (use existing library)
# run tests
update_test_aux: build_test_aux run_test_aux

# build library
# copy library
# build test
# run test
update_lib_test_aux: static_library test_aux run_test_aux

# deep build library (take into account flags changes)
# copy library
# build test
# run test
update_target_test_aux: test_aux_clean run_test_aux


