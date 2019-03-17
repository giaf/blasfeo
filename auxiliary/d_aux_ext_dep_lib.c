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

#include <stdlib.h>
#include <stdio.h>

#if defined(LA_HIGH_PERFORMANCE)
#include "../include/blasfeo_block_size.h"
#endif

#include "../include/blasfeo_common.h"

#define REAL double
#define STRMAT blasfeo_dmat
#define STRVEC blasfeo_dvec
#define PS D_PS


#define ZEROS d_zeros
#define ZEROS_ALIGN d_zeros_align

#define FREE d_free
#define FREE_ALIGN d_free_align

#define PRINT_MAT d_print_mat
#define PRINT_TO_FILE_MAT d_print_to_file_mat
#define PRINT_TO_FILE_EXP_MAT d_print_to_file_exp_mat
#define PRINT_TO_STRING_MAT d_print_to_string_mat

#define PRINT_TRAN_MAT d_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT d_print_to_file_tran_mat

#define PRINT_EXP_MAT d_print_exp_mat
#define PRINT_EXP_TRAN_MAT d_print_exp_tran_mat

#include "x_aux_ext_dep_lib.c"


#if defined(LA_EXTERNAL_BLAS_WRAPPER) | defined(LA_REFERENCE)


#define ALLOCATE_STRMAT blasfeo_allocate_dmat
#define ALLOCATE_STRVEC blasfeo_allocate_dvec

#define FREE_STRMAT blasfeo_free_dmat
#define FREE_STRVEC blasfeo_free_dvec

#define PRINT_STRMAT blasfeo_print_dmat
#define PRINT_STRVEC blasfeo_print_dvec
#define PRINT_TRAN_STRVEC blasfeo_print_tran_dvec

#define PRINT_TO_FILE_STRMAT blasfeo_print_to_file_dmat
#define PRINT_TO_FILE_EXP_STRMAT blasfeo_print_to_file_exp_dmat
#define PRINT_TO_FILE_STRVEC blasfeo_print_to_file_dvec
#define PRINT_TO_FILE_TRAN_STRVEC d_print_to_file_tran_strvec
#define PRINT_TO_STRING_STRMAT blasfeo_print_to_string_dmat
#define PRINT_TO_STRING_STRVEC blasfeo_print_to_string_dvec
#define PRINT_TO_STRING_TRAN_STRVEC blasfeo_print_to_string_tran_dvec

#define PRINT_EXP_STRMAT blasfeo_print_exp_dmat
#define PRINT_EXP_STRVEC blasfeo_print_exp_dvec
#define PRINT_EXP_TRAN_STRVEC blasfeo_print_exp_tran_dvec

#include "x_aux_ext_dep_lib0.c"

#endif

