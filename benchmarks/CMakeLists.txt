###################################################################################################
#                                                                                                 #
# This file is part of HPIPM.                                                                     #
#                                                                                                 #
# HPIPM -- High Performance Interior Point Method.                                                #
# Copyright (C) 2017 by Gianluca Frison.                                                          #
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
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################


add_executable(benchmark_d_blas benchmark_d_blas.c)
add_executable(benchmark_s_blas benchmark_s_blas.c)

target_link_libraries(benchmark_d_blas blasfeo)
target_link_libraries(benchmark_s_blas blasfeo)

# link blas external blas implementations
if(BLASFEO_BENCHMARKS MATCHES ON)
	if(${REF_BLAS} MATCHES OPENBLAS)
		target_link_libraries(benchmark_d_blas /opt/openblas/lib/libopenblas.a -pthread -lgfortran -lm)
		target_link_libraries(benchmark_s_blas /opt/openblas/lib/libopenblas.a -pthread -lgfortran -lm)
	endif(${REF_BLAS} MATCHES OPENBLAS)

	if(${REF_BLAS} MATCHES BLIS)
		target_link_libraries(benchmark_d_blas /opt/netlib/liblapack.a /opt/blis/lib/libblis.a -fopenmp -lgfortran -lm)
		target_link_libraries(benchmark_s_blas /opt/netlib/liblapack.a /opt/blis/lib/libblis.a -fopenmp -lgfortran -lm)
	endif(${REF_BLAS} MATCHES BLIS)

	if(${REF_BLAS} MATCHES NETLIB)
		target_link_libraries(benchmark_d_blas /opt/netlib/liblapack.a /opt/netlib/libblas.a -lgfortran -lm)
		target_link_libraries(benchmark_s_blas /opt/netlib/liblapack.a /opt/netlib/libblas.a -lgfortran -lm)
	endif(${REF_BLAS} MATCHES NETLIB)

	if(${REF_BLAS} MATCHES MKL)
		target_link_libraries(benchmark_d_blas -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_gf_lp64.a /opt/intel/mkl/lib/intel64/libmkl_core.a /opt/intel/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group -ldl -lpthread -lm)
		target_link_libraries(benchmark_s_blas -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_gf_lp64.a /opt/intel/mkl/lib/intel64/libmkl_core.a /opt/intel/mkl/lib/intel64/libmkl_sequential.a -Wl,--end-group -ldl -lpthread -lm)
	endif(${REF_BLAS} MATCHES MKL)

	if(${REF_BLAS} MATCHES ATLAS)
		target_link_libraries(benchmark_d_blas /opt/atlas/lib/liblapack.a /opt/atlas/lib/libcblas.a /opt/atlas/lib/libf77blas.a /opt/atlas/lib/libatlas.a -lgfortran -lm)
		target_link_libraries(benchmark_s_blas /opt/atlas/lib/liblapack.a /opt/atlas/lib/libcblas.a /opt/atlas/lib/libf77blas.a /opt/atlas/lib/libatlas.a -lgfortran -lm)
	endif(${REF_BLAS} MATCHES ATLAS)

endif()
