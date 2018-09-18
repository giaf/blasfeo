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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"


#define ZEROS s_zeros
#define ZEROS_ALIGN s_zeros_align

#define FREE s_free
#define FREE_ALIGN s_free_align

#define PRINT_MAT s_print_mat
#define PRINT_TO_FILE_MAT s_print_to_file_mat
#define PRINT_TO_FILE_EXP_MAT s_print_to_file_exp_mat
#define PRINT_TO_STRING_MAT s_print_to_string_mat

#define PRINT_TRAN_MAT s_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT s_print_to_file_tran_mat

#define PRINT_EXP_MAT s_print_exp_mat
#define PRINT_EXP_TRAN_MAT s_print_exp_tran_mat



#define REAL float
#define STRMAT blasfeo_smat_ref
#define STRVEC blasfeo_svec_ref


#define ALLOCATE_STRMAT blasfeo_allocate_smat_ref
#define ALLOCATE_STRVEC blasfeo_allocate_svec_ref

#define FREE_STRMAT blasfeo_free_smat_ref
#define FREE_STRVEC blasfeo_free_svec_ref

#define PRINT_STRMAT blasfeo_print_smat_ref
#define PRINT_STRVEC blasfeo_print_svec_ref
#define PRINT_TRAN_STRVEC blasfeo_print_tran_svec_ref

#define PRINT_TO_FILE_STRMAT blasfeo_print_to_file_smat_ref
#define PRINT_TO_FILE_EXP_STRMAT blasfeo_print_to_file_exp_smat_ref
#define PRINT_TO_FILE_STRVEC blasfeo_print_to_file_svec_ref
#define PRINT_TO_FILE_TRAN_STRVEC blasfeo_s_print_to_file_tran_strvec_ref
#define PRINT_TO_STRING_STRMAT blasfeo_print_to_string_smat_ref
#define PRINT_TO_STRING_STRVEC blasfeo_print_to_string_svec_ref
#define PRINT_TO_STRING_TRAN_STRVEC blasfeo_print_to_string_tran_svec_ref

#define PRINT_EXP_STRMAT blasfeo_print_exp_smat_ref
#define PRINT_EXP_STRVEC blasfeo_print_exp_svec_ref
#define PRINT_EXP_TRAN_STRVEC blasfeo_print_exp_tran_svec_ref

#include "x_aux_ext_dep_lib0.c"

