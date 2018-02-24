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


add_executable(test_s_aux test_s_aux.c)
add_executable(test_d_aux test_d_aux.c)
add_executable(test_d_blas3 test_d_blas3.c)

target_link_libraries(test_d_aux blasfeo m)
target_link_libraries(test_s_aux blasfeo m)
target_link_libraries(test_d_blas3 blasfeo m)


if(BLASFEO_TESTING MATCHES ON)
	target_link_libraries(test_d_aux blasfeo_ref)
	target_link_libraries(test_s_aux blasfeo_ref)
	target_link_libraries(test_d_blas3 blasfeo_ref)
endif()
