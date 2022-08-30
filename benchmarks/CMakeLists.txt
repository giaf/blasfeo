###################################################################################################
#                                                                                                 #
# This file is part of BLASFEO.                                                                   #
#                                                                                                 #
# BLASFEO -- BLAS For Embedded Optimization.                                                      #
# Copyright (C) 2019 by Gianluca Frison.                                                          #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# The 2-Clause BSD License                                                                        #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without                              #
# modification, are permitted provided that the following conditions are met:                     #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
#    list of conditions and the following disclaimer.                                             #
# 2. Redistributions in binary form must reproduce the above copyright notice,                    #
#    this list of conditions and the following disclaimer in the documentation                    #
#    and/or other materials provided with the distribution.                                       #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################
add_executable(benchmark_d_blasfeo benchmark_d_blasfeo_api.c)
add_executable(benchmark_s_blasfeo benchmark_s_blasfeo_api.c)
add_executable(benchmark_d_blas benchmark_d_blas_api.c)
add_executable(benchmark_s_blas benchmark_s_blas_api.c)

if(CMAKE_C_COMPILER_ID MATCHES MSVC) # no explicit math library and no running BLAS (for now)

	target_link_libraries(benchmark_d_blasfeo blasfeo)
	target_link_libraries(benchmark_s_blasfeo blasfeo)
	target_link_libraries(benchmark_d_blas blasfeo)
	target_link_libraries(benchmark_s_blas blasfeo)

else() # add explicit math library

	target_link_libraries(benchmark_d_blasfeo blasfeo ${EXTERNAL_BLAS_LIBRARIES} m)
	target_link_libraries(benchmark_s_blasfeo blasfeo ${EXTERNAL_BLAS_LIBRARIES} m)
	target_link_libraries(benchmark_d_blas blasfeo ${EXTERNAL_BLAS_LIBRARIES} m)
	target_link_libraries(benchmark_s_blas blasfeo ${EXTERNAL_BLAS_LIBRARIES} m)

endif()
