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


add_executable(example_d_riccati_recursion example_d_riccati_recursion.c tools.c)
add_executable(example_s_riccati_recursion example_s_riccati_recursion.c tools.c)

target_link_libraries(example_d_riccati_recursion blasfeo m)
target_link_libraries(example_s_riccati_recursion blasfeo m)

add_executable(example_d_lu_factorization example_d_lu_factorization.c)
add_executable(example_s_lu_factorization example_s_lu_factorization.c)

target_link_libraries(example_d_lu_factorization blasfeo m)
target_link_libraries(example_s_lu_factorization blasfeo m)

add_executable(example_kernel_dgemm example_kernel_dgemm.c kernel_dgemm.S)
target_link_libraries(example_kernel_dgemm blasfeo m)




