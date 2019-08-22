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


add_executable(example_d_riccati_recursion example_d_riccati_recursion.c tools.c)
add_executable(example_s_riccati_recursion example_s_riccati_recursion.c tools.c)
add_executable(example_d_lu_factorization example_d_lu_factorization.c)
add_executable(example_s_lu_factorization example_s_lu_factorization.c)

target_link_libraries(example_d_riccati_recursion blasfeo m )
target_link_libraries(example_s_riccati_recursion blasfeo m )
target_link_libraries(example_d_lu_factorization blasfeo m)
target_link_libraries(example_s_lu_factorization blasfeo m)

if(EMBEDDED_TARGET MATCHES XILINX_NONE_ELF ) #Xilinx bare-metal
	set(CMAKE_EXE_LINKER_FLAGS 
	"${CMAKE_EXE_LINKER_FLAGS} -T ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Platform/lscript_ultrazed.ld")
endif()
