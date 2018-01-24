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

#include "../include/blasfeo_block_size.h"
#include "blasfeo_common.h"

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
#define PRINT_TO_STRING_MAT d_print_to_string_mat
// #define PRINT_TO_STRING_STRMAT blasfeo_print_to_string_dmat
#define PRINT_TO_STRING_STRVEC blasfeo_print_to_string_dvec

#define PRINT_TRAN_MAT d_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT d_print_to_file_tran_mat

#define PRINT_E_MAT d_print_e_mat
#define PRINT_E_TRAN_MAT d_print_e_tran_mat

#include "x_aux_ext_dep_lib.c"


#if defined(LA_BLAS) | defined(LA_REFERENCE)


#define ALLOCATE_STRMAT blasfeo_allocate_dmat
#define ALLOCATE_STRVEC blasfeo_allocate_dvec

#define FREE_STRMAT blasfeo_free_dmat
#define FREE_STRVEC blasfeo_free_dvec

#define PRINT_STRMAT blasfeo_print_dmat
#define PRINT_STRVEC blasfeo_print_dvec
#define PRINT_TRAN_STRVEC blasfeo_print_tran_dvec

#define PRINT_TO_FILE_STRMAT blasfeo_print_to_file_dmat
#define PRINT_TO_FILE_STRVEC blasfeo_print_to_file_dvec
#define PRINT_TO_FILE_TRAN_STRVEC d_print_to_file_tran_strvec
#define PRINT_TO_STRING_STRMAT blasfeo_print_to_string_dmat
#define PRINT_TO_STRING_STRVEC blasfeo_print_to_string_dvec
#define PRINT_TO_STRING_TRAN_STRVEC blasfeo_print_to_string_tran_dvec

#define PRINT_E_STRMAT blasfeo_print_exp_dmat
#define PRINT_E_STRVEC blasfeo_print_exp_dvec
#define PRINT_E_TRAN_STRVEC blasfeo_print_exp_tran_dvec

#include "x_aux_ext_dep_lib0.c"

#endif

