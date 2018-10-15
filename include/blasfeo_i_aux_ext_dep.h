/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#ifndef BLASFEO_I_AUX_EXT_DEP_H_
#define BLASFEO_I_AUX_EXT_DEP_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#ifdef EXT_DEP

// i_aux_extern_depend_lib
void int_zeros(int **pA, int row, int col);
void int_zeros_align(int **pA, int row, int col);
void int_free(int *pA);
void int_free_align(int *pA);
void int_print_mat(int row, int col, int *A, int lda);
int int_print_to_string_mat(char **buf_out, int row, int col, int *A, int lda);

#endif // EXT_DEP



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_I_AUX_EXT_DEP_H_
