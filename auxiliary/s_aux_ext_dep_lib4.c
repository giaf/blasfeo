/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"


#define REAL float
#define STRMAT s_strmat
#define STRVEC s_strvec
#define PS S_PS

#if defined(LA_HIGH_PERFORMANCE)

#include "../include/blasfeo_block_size.h"

#define ZEROS s_zeros
#define ZEROS_ALIGN s_zeros_align
#define FREE s_free
#define FREE_ALIGN s_free_align

#define PRINT_MAT s_print_mat
#define PRINT_TO_FILE_MAT s_print_to_file_mat

#define PRINT_TRAN_MAT s_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT s_print_to_file_tran_mat

#define PRINT_E_MAT s_print_e_mat
#define PRINT_E_TRAN_MAT s_print_e_tran_mat

#define ALLOCATE_STRMAT s_allocate_strmat
#define ALLOCATE_STRVEC s_allocate_strvec

#define FREE_STRMAT s_free_strmat
#define FREE_STRVEC s_free_strvec

#define PRINT_STRMAT s_print_strmat
#define PRINT_STRVEC s_print_strvec
#define PRINT_TRAN_STRVEC s_print_tran_strvec

#define PRINT_TO_FILE_STRMAT s_print_to_file_strmat
#define PRINT_TO_FILE_STRVEC s_print_to_file_strvec
#define PRINT_TO_FILE_TRAN_STRVEC s_print_to_file_tran_strvec

#define PRINT_E_STRMAT s_print_e_strmat
#define PRINT_E_STRVEC s_print_e_strvec
#define PRINT_E_TRAN_STRVEC s_print_e_tran_strvec

#include "x_aux_ext_dep_lib4.c"

#endif

